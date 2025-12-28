import os
import re
import unicodedata
import warnings
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

# 외부 라이브러리 의존성 체크 (adjustText)
try:
    from adjustText import adjust_text
except ImportError:
    import subprocess
    # 코랩 등 환경에서 라이브러리가 없을 경우 자동 설치
    subprocess.check_call(["pip", "install", "adjustText", "-q"])
    from adjustText import adjust_text

# 불필요한 경고 메시지 숨김 (깔끔한 출력을 위함)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class PlotConfig:
    """
    차트 스타일 및 폰트 설정을 관리하는 클래스입니다.
    """
    @staticmethod
    def set_style():
        # 기본 스타일 설정 (배경 격자 등)
        sns.set(style='whitegrid')
        # 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
        # 한글 폰트 설정 실행
        PlotConfig._set_korean_font()

    @staticmethod
    def _set_korean_font():
        """
        실행 환경(Colab, Windows, Mac)을 감지하여 
        적절한 한글 폰트를 설정합니다.
        """
        font_candidates = ['NanumBarunGothic', 'Malgun Gothic', 'AppleGothic']
        
        # 구글 코랩(Colab) 환경 전용 경로 확인
        colab_font = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
        if os.path.exists(colab_font):
            fm.fontManager.addfont(colab_font)
            plt.rc('font', family='NanumBarunGothic')
            return

        # 윈도우/맥 등 로컬 환경인 경우 설치된 폰트 중 하나 선택
        system_font = next((f for f in font_candidates if f in [f.name for f in fm.fontManager.ttflist]), 'sans-serif')
        plt.rc('font', family=system_font)


class DataLoader:
    """
    데이터 파일 로드 및 초기 전처리를 담당하는 클래스입니다.
    """
    def __init__(self, base_path: str):
        self.base_path = base_path

    def load_parquets(self) -> pd.DataFrame:
        """
        지정된 폴더 내의 모든 파케(parquet) 파일을 읽어 하나로 합칩니다.
        """
        if not os.path.exists(self.base_path):
            print(f"[경고] 폴더를 찾을 수 없습니다: {self.base_path}")
            return pd.DataFrame()

        dfs = []
        for file_name in os.listdir(self.base_path):
            if not file_name.endswith('.parquet'):
                continue
            
            file_path = os.path.join(self.base_path, file_name)
            try:
                df = pd.read_parquet(file_path)
                # 컬럼명 표준화 (Date, Ticker 등)
                df = self._standardize_columns(df, file_name)
                dfs.append(df)
            except Exception as e:
                print(f"[에러] 파일 로드 실패 ({file_name}): {e}")

        if not dfs:
            return pd.DataFrame()

        # 데이터 병합
        full_df = pd.concat(dfs, ignore_index=True)
        
        # 날짜 컬럼 찾기 (Date, day 등 대소문자 구분 없이)
        date_col = next((c for c in full_df.columns if c.lower() in ['date', 'day']), 'Date')
        
        full_df[date_col] = pd.to_datetime(full_df[date_col])
        return full_df.set_index(date_col).sort_index()

    def _standardize_columns(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        데이터프레임의 컬럼 이름과 타입을 표준화합니다.
        """
        # 인덱스에 날짜가 있는 경우 리셋
        if 'Date' not in df.columns and 'Ticker' not in df.columns:
            df = df.reset_index()
        
        if 'index' in df.columns:
            df.rename(columns={'index': 'Date'}, inplace=True)
        
        # 파일명에서 티커(종목코드) 추출
        if 'Ticker' not in df.columns:
            df['Ticker'] = filename.replace('.parquet', '')

        # 유니코드 문자 정규화 (한글 깨짐 방지)
        if df['Ticker'].dtype == object:
            df['Ticker'] = df['Ticker'].apply(lambda x: unicodedata.normalize('NFC', str(x)))

        # 카테고리 타입은 문자열로 변환 (오류 방지)
        for col in df.select_dtypes(['category']).columns:
            df[col] = df[col].astype(str)
            
        return df


class FeatureEngineer:
    """
    데이터에서 분석에 필요한 피처(지표)를 생성하고 가공하는 클래스입니다.
    """
    
    @staticmethod
    def parse_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
        """
        문자열로 된 숫자(예: '3.5%', '1,000')를 실제 숫자형으로 안전하게 변환합니다.
        """
        if pd.api.types.is_numeric_dtype(series):
            return series.fillna(default)
        
        # 콤마(,), 퍼센트(%) 제거 및 마이너스 기호 통일
        clean_series = series.astype(str).str.replace(r'[,%]', '', regex=True)\
                                         .str.replace('−', '-', regex=False)\
                                         .str.replace('nan', '', regex=False)\
                                         .str.strip()
        return pd.to_numeric(clean_series, errors='coerce').fillna(default)

    def extract_dividend_metrics(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        배당수익률(dy), 주당배당금(dps), 배당지급여부(flag)를 추출합니다.
        """
        idx = df.index
        
        # 컬럼 가져오기 및 숫자 변환
        dy = self.parse_numeric(df.get('Dividend_Yield', pd.Series(0, index=idx)))
        dps = self.parse_numeric(df.get('DPS', pd.Series(0, index=idx)))
        
        # 단위 보정: 배당수익률 중앙값이 1보다 크다면 퍼센트(%) 단위로 간주하여 100으로 나눔
        if (dy > 0).any() and dy[dy > 0].median() > 1.0:
            dy = dy / 100.0

        # 배당수익률이 비어있다면, DPS / 종가(Close)로 역산 시도
        if (dy == 0).all() and 'Close' in df.columns and (dps > 0).any():
            close = self.parse_numeric(df['Close']).replace(0, np.nan)
            dy = (dps / close).fillna(0.0)

        # 배당 지급 여부 (수익률이나 배당금이 0보다 크면 True)
        payer_flag = (dy > 0) | (dps > 0)
        return dy, dps, payer_flag

    def create_features(self, snapshot: pd.DataFrame, mode: str = 'wide') -> pd.DataFrame:
        """
        클러스터링을 위한 최종 피처 데이터프레임을 생성합니다.
        mode: 'wide' (전체, 배당X 포함) | 'div_only' (배당 지급 종목만)
        """
        if snapshot.empty:
            return pd.DataFrame()

        base = snapshot.set_index('Ticker') if 'Ticker' in snapshot.columns else snapshot.copy()
        dy, dps, payer = self.extract_dividend_metrics(base)

        # 모드에 따른 데이터 필터링
        if mode == 'div_only':
            base = base[payer]
            dy, dps, payer = dy.loc[base.index], dps.loc[base.index], payer.loc[base.index]

        if base.empty:
            return pd.DataFrame()

        # 보조 지표 파싱 (이격도, 영업이익)
        disp = self.parse_numeric(base.get('Disparity_60d', pd.Series(0, index=base.index)))
        op_val = self.parse_numeric(base.get('영업이익', pd.Series(0, index=base.index)))
        
        # 시가총액 계산 (Marcap 컬럼이 없으면 종가*거래량으로 대체)
        if 'Marcap' in base.columns:
            marcap = self.parse_numeric(base['Marcap'])
        elif 'Close' in base.columns and 'Volume' in base.columns:
            marcap = self.parse_numeric(base['Close']) * self.parse_numeric(base['Volume'])
        else:
            marcap = pd.Series(0, index=base.index)

        # 거래량 비율 (현재 거래량 / 60일 평균)
        vol = self.parse_numeric(base.get('Volume', pd.Series(0, index=base.index)))
        vol_avg = self.parse_numeric(base.get('vol_60', pd.Series(1, index=base.index))).replace(0, 1)
        vol_ratio = vol / vol_avg

        # 데이터 분포 정규화 (Quantile Transformation)
        qt = QuantileTransformer(n_quantiles=min(1000, len(base)), output_distribution='normal', random_state=42)
        
        # X축 설계 (배당 모멘텀)
        if mode == 'wide':
            # Wide 모드: 배당주와 비배당주를 시각적으로 확 벌려놓기 위해 가산점 부여
            dy_spread = dy.copy()
            dy_spread[payer] += 2.0 
            x_input = dy_spread.values.reshape(-1, 1)
        else:
            x_input = dy.values.reshape(-1, 1)

        dy_norm = qt.fit_transform(x_input).ravel()
        disp_norm = qt.fit_transform(disp.values.reshape(-1, 1)).ravel()
        vol_norm = qt.fit_transform(vol_ratio.values.reshape(-1, 1)).ravel()

        # 최종 점수 산출 (가중치: 배당 60% + 이격도 25% + 거래량 15%)
        x_raw = (dy_norm * 0.6) + (disp_norm * 0.25) + (vol_norm * 0.15)
        x_final = MinMaxScaler().fit_transform(x_raw.reshape(-1, 1)).ravel()
        
        # Y축: 실적(영업이익) 순위 (퍼센트 랭크)
        y_final = op_val.rank(pct=True).values

        return pd.DataFrame({
            'X_Momentum': x_final,
            'Y_Fundamental': y_final,
            'MarketCap': marcap.values,
            'Dividend_Yield': dy.values,
            'DPS': dps.values,
            'Payer_Flag': payer.values,
        }, index=base.index)


class RallyMapVisualizer:
    """
    데이터를 그룹화(클러스터링)하고 시각화(차트)하는 클래스입니다.
    """
    
    def __init__(self, drive_path: str):
        self.drive_path = drive_path

    def _assign_labels(self, row: pd.Series, mode: str) -> str:
        """
        좌표(X, Y)에 따라 그룹 이름을 붙여줍니다.
        """
        cx, cy = row['X_Momentum'], row['Y_Fundamental']
        if mode == 'wide':
            # Wide 모드: X축 절반 기준으로 비배당/배당 나눔
            if cx > 0.5:
                return "1. 산타 랠리 주도주" if cy > 0.6 else "3. 배당 테마주"
            else:
                return "2. 성장 우량주" if cy > 0.6 else "4. 낙폭 과대주"
        else:
            # Div Only 모드: 상대적 4분면
            if cx > 0.6 and cy > 0.6: return "1. 산타 랠리 주도주"
            if cx <= 0.6 and cy > 0.6: return "2. 저평가 실적주"
            if cx > 0.6 and cy <= 0.6: return "3. 고배당 테마주"
            return "4. 배당주 소외"

    def run(self, target_date: pd.Timestamp, data: pd.DataFrame, mode: str, eps: float, min_samples: int):
        """
        실제 분석을 수행하는 메인 함수입니다.
        """
        title_mode = "Wide Spread (전체)" if mode == 'wide' else "Dividend Only (배당주만)"
        print(f"\n🚀 [분석 시작: {title_mode}] 기준일: {target_date.date()}")

        try:
            snapshot = data.loc[target_date].copy()
            if isinstance(snapshot, pd.Series):
                snapshot = snapshot.to_frame().T
        except KeyError:
            print("해당 날짜의 데이터가 없습니다.")
            return

        fe = FeatureEngineer()
        feats = fe.create_features(snapshot, mode=mode)
        
        if feats.empty:
            print("피처 생성 실패: 데이터가 비어있습니다. 컬럼을 확인하세요.")
            return

        # 클러스터링 (DBSCAN 알고리즘)
        X = feats[['X_Momentum', 'Y_Fundamental']].values
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        feats['Cluster_ID'] = db.labels_
        feats['Cluster_Name'] = feats.apply(lambda r: self._assign_labels(r, mode), axis=1)

        self._plot_map(feats, target_date, title_mode)
        self._save_results(feats, target_date, mode)

    def _plot_map(self, df: pd.DataFrame, date: pd.Timestamp, title_suffix: str):
        """
        산점도(Scatter Plot)를 그립니다.
        """
        plt.figure(figsize=(15, 9))
        
        # 그룹별 색상 지정
        unique_groups = sorted(df['Cluster_Name'].unique())
        palette = sns.color_palette("bright", n_colors=len(unique_groups))
        color_map = dict(zip(unique_groups, palette))

        # 노이즈(어느 그룹에도 속하지 못한 종목)는 회색으로 연하게 표시
        noise = df[df['Cluster_ID'] == -1]
        if not noise.empty:
            plt.scatter(noise['X_Momentum'], noise['Y_Fundamental'], c='#EEEEEE', 
                        s=15, alpha=0.35, label='Noise (개별종목)', zorder=1)

        # 메인 그룹 그리기
        text_labels = []
        for name in unique_groups:
            subset = df[(df['Cluster_Name'] == name) & (df['Cluster_ID'] != -1)]
            if subset.empty: continue
            
            is_leader = "산타" in name  # 주도주 그룹 강조
            plt.scatter(subset['X_Momentum'], subset['Y_Fundamental'], 
                        c=[color_map[name]], 
                        s=np.log1p(subset['MarketCap']) * 4 + 20,  # 시총 클수록 점 크기 확대
                        alpha=1.0 if is_leader else 0.85,
                        edgecolors='black' if is_leader else 'white',
                        linewidths=0.7, label=name, zorder=5)

            # 라벨링: 각 그룹에서 시가총액 1등 종목 이름표 붙이기
            top_ticker = subset.nlargest(1, 'MarketCap').index[0]
            text_labels.append(top_ticker)

        # 추가 라벨링: 배당수익률 전체 TOP 3 종목
        text_labels += df.nlargest(3, 'Dividend_Yield').index.tolist()
        
        # 텍스트 겹침 방지 (adjustText 사용)
        texts = [plt.text(df.loc[t, 'X_Momentum'], df.loc[t, 'Y_Fundamental'], str(t), 
                 fontsize=11, fontweight='bold') for t in set(text_labels)]
        
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.4))

        plt.title(f"Year-End Rally Map ({title_suffix})\n{date.date()}", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Dividend Momentum Score (배당 모멘텀)", fontsize=12)
        plt.ylabel("Fundamental Rank (실적 체력)", fontsize=12)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.show()

    def _save_results(self, df: pd.DataFrame, date: pd.Timestamp, mode: str):
        """
        분석 결과를 CSV 파일로 저장합니다.
        """
        filename = f"Rally_Map_{mode.upper()}_{date.strftime('%Y%m%d')}.csv"
        path = os.path.join(self.drive_path, filename)
        df.reset_index().rename(columns={'index': 'Ticker'}).to_csv(path, index=False)
        print(f"💾 결과 저장 완료: {path}")


# =========================================
# 메인 실행 영역
# =========================================
if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')
    
    DRIVE_PATH = '/content/drive/MyDrive/data'
    
    # 1. 설정 초기화
    PlotConfig.set_style()
    loader = DataLoader(DRIVE_PATH)
    visualizer = RallyMapVisualizer(DRIVE_PATH)

    # 2. 데이터 로드
    full_data = loader.load_parquets()

    if not full_data.empty:
        # 가장 최근 날짜 가져오기
        last_date = sorted(full_data.index.unique())[-1]
        
        # 3. 시나리오별 실행
        # A) Wide Spread: 전체 종목 대상, 배당 유무로 좌우 강제 분리
        visualizer.run(target_date=last_date, data=full_data, mode='wide', eps=0.06, min_samples=6)

        # B) Dividend Only: 배당 주는 종목만 남겨서 비교
        visualizer.run(target_date=last_date, data=full_data, mode='div_only', eps=0.07, min_samples=4)
    else:
        print("[에러] 데이터를 불러오지 못했습니다. 드라이브 경로와 파일 상태를 확인해주세요.")
