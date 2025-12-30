import os
import io
import requests
import warnings
import unicodedata
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

# ---------------------------------------------------------
# [1] 설정 및 라이브러리 로드
# ---------------------------------------------------------
try:
    from adjustText import adjust_text
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "adjustText", "-q"])
    from adjustText import adjust_text

warnings.filterwarnings('ignore')

class PlotConfig:
    """차트 스타일 및 폰트 설정"""
    @staticmethod
    def set_style():
        sns.set(style='whitegrid')
        plt.rcParams['axes.unicode_minus'] = False
        PlotConfig._set_korean_font()

    @staticmethod
    def _set_korean_font():
        # 코랩/로컬 환경에 맞춰 한글 폰트 자동 설정
        font_candidates = ['NanumBarunGothic', 'Malgun Gothic', 'AppleGothic']
        colab_font = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
        if os.path.exists(colab_font):
            fm.fontManager.addfont(colab_font)
            plt.rc('font', family='NanumBarunGothic')
            return
        system_font = next((f for f in font_candidates if f in [f.name for f in fm.fontManager.ttflist]), 'sans-serif')
        plt.rc('font', family=system_font)


# ---------------------------------------------------------
# [2] GitHub 데이터 로더 (핵심 수정 부분)
# ---------------------------------------------------------
class GitHubDataLoader:
    """
    GitHub 레포지토리의 Raw 파일을 직접 읽어오는 로더입니다.
    """
    def __init__(self, repo_owner: str, repo_name: str, branch: str = 'main'):
        self.base_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}"

    def load_csv(self, filename: str) -> pd.DataFrame:
        """GitHub에서 CSV 파일을 다운로드하여 데이터프레임으로 변환합니다."""
        url = f"{self.base_url}/{filename}"
        print(f"📥 데이터 다운로드 중... ({url})")
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # 404 등 에러 체크
            
            # 한글 인코딩 처리 (cp949 또는 utf-8 시도)
            try:
                df = pd.read_csv(io.StringIO(response.text))
            except:
                df = pd.read_csv(io.BytesIO(response.content), encoding='cp949')
                
            print(f"✅ 로드 성공: {len(df)}개 종목")
            return self._standardize_data(df)
            
        except Exception as e:
            print(f"[에러] 데이터 로드 실패: {e}")
            return pd.DataFrame()

    def _standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        로드한 데이터의 컬럼 이름을 분석 코드에 맞게 변환하고,
        없는 컬럼(배당, 이익 등)은 0으로 채워 에러를 방지합니다.
        """
        # 1. 컬럼명 매핑 (한글 -> 영어)
        col_map = {
            '종목코드': 'Ticker',
            '종목명': 'Name',
            '종가': 'Close',
            '등락률': 'Chg_Pct',
            '상장시가총액': 'Marcap',
            '거래량': 'Volume'
        }
        df.rename(columns=col_map, inplace=True)
        
        # 2. 티커(종목코드) 표준화 (6자리 문자열)
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].apply(lambda x: f"{int(x):06d}" if isinstance(x, (int, float)) else str(x))

        # 3. 기준 날짜 컬럼 추가 (스냅샷 데이터이므로 오늘 날짜로 가정)
        if 'Date' not in df.columns:
            df['Date'] = pd.Timestamp.now().normalize()

        # 4. 분석에 필수적인 컬럼이 없으면 0으로 채움 (안전장치)
        required_cols = ['Dividend_Yield', 'DPS', 'Disparity_60d', 'vol_60', '영업이익']
        for col in required_cols:
            if col not in df.columns:
                # print(f"[알림] '{col}' 컬럼이 없어 0으로 초기화합니다.")
                df[col] = 0.0

        return df.set_index('Date')


# ---------------------------------------------------------
# [3] 피처 엔지니어링 (기존 로직 유지)
# ---------------------------------------------------------
class FeatureEngineer:
    """데이터 가공 및 지표 생성"""
    @staticmethod
    def create_features(snapshot: pd.DataFrame, mode: str = 'wide') -> pd.DataFrame:
        if snapshot.empty: return pd.DataFrame()
        
        df = snapshot.copy()
        if 'Ticker' in df.columns: df.set_index('Ticker', inplace=True)
        
        # 숫자형 변환 헬퍼
        def to_num(s): 
            return pd.to_numeric(s.astype(str).str.replace(r'[,%]', '', regex=True), errors='coerce').fillna(0)

        # 데이터 추출
        dy = to_num(df['Dividend_Yield'])
        dps = to_num(df['DPS'])
        op_profit = to_num(df['영업이익'])
        marcap = to_num(df['Marcap'])
        
        # 배당률 단위 보정 (3.5 -> 0.035)
        if dy.median() > 1.0: dy /= 100.0
        
        # 배당 유무 플래그
        payer = (dy > 0) | (dps > 0)
        
        # 모드별 필터링
        if mode == 'div_only':
            target_idx = payer[payer].index
            df = df.loc[target_idx]
            dy, op_profit, marcap, payer = dy[target_idx], op_profit[target_idx], marcap[target_idx], payer[target_idx]

        if df.empty: return pd.DataFrame()

        # X축: 배당 모멘텀 (현재 데이터가 부족하므로 배당수익률 위주로 계산)
        qt = QuantileTransformer(n_quantiles=min(100, len(df)), output_distribution='normal', random_state=42)
        
        # X축 입력값 구성
        if mode == 'wide':
            x_input = dy.copy()
            x_input[payer] += 2.0 # 배당주 우대
        else:
            x_input = dy.copy()
            
        # 정규화
        x_norm = qt.fit_transform(x_input.values.reshape(-1,1)).ravel()
        x_final = MinMaxScaler().fit_transform(x_norm.reshape(-1,1)).ravel()
        
        # Y축: 실적 (영업이익 랭크)
        y_final = op_profit.rank(pct=True).values

        return pd.DataFrame({
            'X_Momentum': x_final,
            'Y_Fundamental': y_final,
            'MarketCap': marcap.values,
            'Dividend_Yield': dy.values,
            'Cluster_Name': 'TBD' # 추후 할당
        }, index=df.index)


# ---------------------------------------------------------
# [4] 시각화 및 메인 실행
# ---------------------------------------------------------
class RallyMapVisualizer:
    def run(self, data: pd.DataFrame):
        print("\n🚀 [분석 시작] GitHub 데이터 기반")
        
        # 가장 최근 날짜 데이터 추출
        last_date = sorted(data.index.unique())[-1]
        snapshot = data.loc[last_date]
        if isinstance(snapshot, pd.Series): snapshot = snapshot.to_frame().T
        
        # 피처 생성
        fe = FeatureEngineer()
        feats = fe.create_features(snapshot, mode='wide')
        
        if feats.empty:
            print("❌ 분석할 데이터가 충분하지 않습니다.")
            return

        # 클러스터링 (간단화)
        # X, Y 좌표가 0인 경우가 많을 수 있어(데이터 부재) 노이즈 처리 주의
        X = feats[['X_Momentum', 'Y_Fundamental']].values
        db = DBSCAN(eps=0.1, min_samples=3).fit(X)
        feats['Cluster'] = db.labels_
        
        # 시각화
        plt.figure(figsize=(12, 8))
        plt.scatter(feats['X_Momentum'], feats['Y_Fundamental'], 
                    s=np.log1p(feats['MarketCap'])*5 + 10, 
                    c=feats['Cluster'], cmap='tab10', alpha=0.7, edgecolors='white')
        
        plt.title(f"GitHub Repo Data Map ({last_date.date()})", fontsize=15)
        plt.xlabel("Dividend Score (Data Missing=0)", fontsize=12)
        plt.ylabel("Profit Rank (Data Missing=0)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.show()
        print("✅ 시각화 완료")


# =========================================================
# 메인 실행부
# =========================================================
if __name__ == "__main__":
    PlotConfig.set_style()
    
    # 1. GitHub에서 데이터 로드
    # (ParkYoungsig/StockClustering 레포의 main 브랜치 사용)
    loader = GitHubDataLoader(repo_owner='ParkYoungsig', repo_name='StockClustering')
    
    # stock_list.csv 파일을 로드합니다.
    df = loader.load_csv('stock_list.csv')
    
    if not df.empty:
        # 2. 분석 및 시각화 실행
        viz = RallyMapVisualizer()
        viz.run(df)
    else:
        print("데이터를 불러오지 못했습니다.")
