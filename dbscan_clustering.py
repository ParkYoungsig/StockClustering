import os
import io
import requests
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

# ---------------------------------------------------------
# Dependency Check: adjustText
# 라벨 겹침 방지를 위해 필수. 환경에 없으면 런타임 설치 시도.
# ---------------------------------------------------------
try:
    from adjustText import adjust_text
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "adjustText", "-q"])
    from adjustText import adjust_text

warnings.filterwarnings('ignore')

class PlotConfig:
    """시각화 스타일 및 폰트 설정 관리"""
    
    @staticmethod
    def set_style():
        sns.set(style='whitegrid')
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        PlotConfig._load_web_font()

    @staticmethod
    def _load_web_font():
        # Colab/Docker 등 로컬 폰트가 없는 환경 대응을 위해 NanumGothic 다운로드
        font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        font_name = "NanumGothic.ttf"
        
        if not os.path.exists(font_name):
            try:
                response = requests.get(font_url)
                response.raise_for_status()
                with open(font_name, 'wb') as f:
                    f.write(response.content)
            except Exception:
                pass  # 네트워크 에러 시 시스템 기본 폰트로 Fallback

        if os.path.exists(font_name):
            fm.fontManager.addfont(font_name)
            plt.rc('font', family='NanumGothic')
        else:
            plt.rc('font', family='sans-serif')

class GitHubDataLoader:
    """GitHub Raw 데이터를 이용한 주가 데이터 로더"""
    
    def __init__(self, repo_owner: str, repo_name: str, branch: str = 'main'):
        self.base_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}"

    def load_date_data(self, date_str: str) -> pd.DataFrame:
        # 1. 해당 날짜 파일 시도 -> 2. 없으면 마스터 리스트(stock_list.csv) 시도
        filename_daily = f"{date_str}.csv"
        df = self._fetch_csv(filename_daily)
        
        if df.empty:
            df = self._fetch_csv('stock_list.csv')

        if df.empty:
            return pd.DataFrame()

        return self._standardize_data(df, date_str)

    def _fetch_csv(self, filename: str) -> pd.DataFrame:
        url = f"{self.base_url}/{filename}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            try:
                return pd.read_csv(io.StringIO(response.text))
            except:
                # EUC-KR/CP949 레거시 데이터 대응
                return pd.read_csv(io.BytesIO(response.content), encoding='cp949')
        except Exception:
            return pd.DataFrame()

    def _standardize_data(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        # 분석 편의를 위해 한글 컬럼 -> 영문 매핑
        col_map = {
            '종목코드': 'Ticker', '종목명': 'Name', 
            '종가': 'Close', '등락률': 'Chg_Pct', 
            '상장시가총액': 'Marcap', '거래량': 'Volume',
            '배당수익률': 'Dividend_Yield', '주당배당금': 'DPS'
        }
        df.rename(columns=col_map, inplace=True)
        
        # Ticker 정규화: 숫자형일 경우 6자리(005930) 문자열로 변환
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].apply(lambda x: f"{int(x):06d}" if isinstance(x, (int, float)) else str(x))

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            df['Date'] = pd.to_datetime(date_str)

        # 결측 시 0.0으로 채울 필수 컬럼들
        required_cols = ['Dividend_Yield', 'DPS', '영업이익', 'Marcap']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df.set_index('Date')

class FeatureEngineer:
    """Raw 데이터를 시각화용 X, Y 좌표 및 메타데이터로 변환"""
    
    @staticmethod
    def create_features(snapshot: pd.DataFrame, mode: str = 'wide') -> pd.DataFrame:
        if snapshot.empty: return pd.DataFrame()
        
        df = snapshot.copy()
        if 'Ticker' in df.columns: df.set_index('Ticker', inplace=True)
        
        # Helper: '1,000', '5%' 같은 문자열 포맷 제거 및 수치 변환
        def to_num(s): 
            return pd.to_numeric(s.astype(str).str.replace(r'[,%]', '', regex=True), errors='coerce').fillna(0)

        dy = to_num(df['Dividend_Yield'])
        dps = to_num(df['DPS'])
        op_profit = to_num(df['영업이익'])
        marcap = to_num(df['Marcap'])
        
        # Scale Check: 배당수익률이 이미 % 단위(3.5 등)로 되어있으면 소수점(0.035)으로 보정
        if dy.median() > 1.0: dy /= 100.0
        
        payer = (dy > 0) | (dps > 0)
        
        # mode='div_only': 배당주만 필터링해서 볼 때 사용
        if mode == 'div_only':
            target_idx = payer[payer].index
            df = df.loc[target_idx]
            dy, op_profit, marcap, payer = dy[target_idx], op_profit[target_idx], marcap[target_idx], payer[target_idx]

        if df.empty: return pd.DataFrame()

        # Transformation Logic:
        # 1. QuantileTransformer: 데이터 분포를 정규분포 형태로 강제 (이상치 완화)
        # 2. MinMaxScaler: 최종 Plotting을 위해 [0, 1] 구간으로 정규화
        qt = QuantileTransformer(n_quantiles=min(100, len(df)), output_distribution='normal', random_state=42)
        
        if mode == 'wide':
            # 시각화 시 배당주 그룹을 우측으로 밀어버리기 위한 Trick (+2.0 offset)
            x_input = dy.copy()
            x_input[payer] += 2.0 
        else:
            x_input = dy.copy()
            
        x_norm = qt.fit_transform(x_input.values.reshape(-1,1)).ravel()
        x_final = MinMaxScaler().fit_transform(x_norm.reshape(-1,1)).ravel()
        
        # Y축(펀더멘털)은 절대값이 아닌 상대적 순위(Rank) 사용
        y_final = op_profit.rank(pct=True).values

        names = df['Name'] if 'Name' in df.columns else pd.Series(index=df.index, data=df.index)

        return pd.DataFrame({
            'Name': names,
            'X_Momentum': x_final,     # 배당 모멘텀 (변환됨)
            'Y_Fundamental': y_final,  # 영업이익 순위
            'MarketCap': marcap.values,
            'Dividend_Yield': dy.values,
        }, index=df.index)

class RallyMapVisualizer:
    def run(self, data: pd.DataFrame, target_date_str: str):
        target_date = pd.to_datetime(target_date_str)
        
        # Index가 DatetimeIndex인 경우 해당 날짜 슬라이싱
        if isinstance(data.index, pd.DatetimeIndex):
            snapshot = data[data.index == target_date]
        else:
            snapshot = data.copy()

        if snapshot.empty:
            print(f"[WARN] {target_date_str} No data found.")
            return

        fe = FeatureEngineer()
        feats = fe.create_features(snapshot, mode='wide')
        
        if feats.empty:
            print("[WARN] Not enough features for plotting.")
            return

        self._plot(feats, target_date_str)

    def _plot(self, feats, date_str):
        X = feats[['X_Momentum', 'Y_Fundamental']].values
        
        # DBSCAN Clustering
        # 데이터 포인트가 적으면 min_samples를 1로 줄여서라도 노이즈 처리를 막음
        min_samples = 3 if len(feats) > 10 else 1
        db = DBSCAN(eps=0.1, min_samples=min_samples).fit(X)
        feats['Cluster'] = db.labels_
        
        plt.figure(figsize=(12, 8))
        
        # Bubble Size: 시총 격차가 너무 크므로 log1p 적용하여 완만하게 표현
        plt.scatter(feats['X_Momentum'], feats['Y_Fundamental'], 
                    s=np.log1p(feats['MarketCap'])*5 + 20, 
                    c=feats['Cluster'], cmap='tab10', alpha=0.8, edgecolors='white')
        
        # Annotation: 시총 상위 10개만 표기 (가독성 확보)
        texts = []
        top_stocks = feats.sort_values('MarketCap', ascending=False).head(10)
        for idx, row in top_stocks.iterrows():
            name = row['Name'] if isinstance(row['Name'], str) else str(idx)
            texts.append(plt.text(row['X_Momentum'], row['Y_Fundamental'], name, fontsize=9))

        # 텍스트 위치 자동 최적화
        try:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except:
            pass

        plt.title(f"Market Rally Map ({date_str})", fontsize=16, fontweight='bold')
        plt.xlabel("Dividend Momentum (Normalized)", fontsize=12)
        plt.ylabel("Fundamental Rank (Percentile)", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.show()

if __name__ == "__main__":
    PlotConfig.set_style()
    
    TARGET_DATE = "2024-05-20"
    
    # Repo 구조: ParkYoungsig/StockClustering
    loader = GitHubDataLoader(repo_owner='ParkYoungsig', repo_name='StockClustering')
    df = loader.load_date_data(TARGET_DATE)
    
    if not df.empty:
        viz = RallyMapVisualizer()
        viz.run(df, TARGET_DATE)
    else:
        print("[ERR] Failed to load data.")
