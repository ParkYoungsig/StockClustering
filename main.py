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

try:
    from adjustText import adjust_text
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "adjustText", "-q"])
    from adjustText import adjust_text

warnings.filterwarnings('ignore')

class PlotConfig:
    @staticmethod
    def set_style():
        sns.set(style='whitegrid')
        plt.rcParams['axes.unicode_minus'] = False
        PlotConfig._load_web_font()

    @staticmethod
    def _load_web_font():
        font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        font_name = "NanumGothic.ttf"
        
        if not os.path.exists(font_name):
            try:
                response = requests.get(font_url)
                response.raise_for_status()
                with open(font_name, 'wb') as f:
                    f.write(response.content)
            except Exception:
                pass

        if os.path.exists(font_name):
            fm.fontManager.addfont(font_name)
            plt.rc('font', family='NanumGothic')
        else:
            plt.rc('font', family='sans-serif')

class GitHubDataLoader:
    def __init__(self, repo_owner: str, repo_name: str, branch: str = 'main'):
        self.base_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}"

    def load_date_data(self, date_str: str) -> pd.DataFrame:
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
                return pd.read_csv(io.BytesIO(response.content), encoding='cp949')
        except Exception:
            return pd.DataFrame()

    def _standardize_data(self, df: pd.DataFrame, date_str: str) -> pd.DataFrame:
        col_map = {
            '종목코드': 'Ticker', '종목명': 'Name', 
            '종가': 'Close', '등락률': 'Chg_Pct', 
            '상장시가총액': 'Marcap', '거래량': 'Volume',
            '배당수익률': 'Dividend_Yield', '주당배당금': 'DPS'
        }
        df.rename(columns=col_map, inplace=True)
        
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker'].apply(lambda x: f"{int(x):06d}" if isinstance(x, (int, float)) else str(x))

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            df['Date'] = pd.to_datetime(date_str)

        required_cols = ['Dividend_Yield', 'DPS', '영업이익', 'Marcap']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df.set_index('Date')

class FeatureEngineer:
    @staticmethod
    def create_features(snapshot: pd.DataFrame, mode: str = 'wide') -> pd.DataFrame:
        if snapshot.empty: return pd.DataFrame()
        
        df = snapshot.copy()
        if 'Ticker' in df.columns: df.set_index('Ticker', inplace=True)
        
        def to_num(s): 
            return pd.to_numeric(s.astype(str).str.replace(r'[,%]', '', regex=True), errors='coerce').fillna(0)

        dy = to_num(df['Dividend_Yield'])
        dps = to_num(df['DPS'])
        op_profit = to_num(df['영업이익'])
        marcap = to_num(df['Marcap'])
        
        if dy.median() > 1.0: dy /= 100.0
        
        payer = (dy > 0) | (dps > 0)
        
        if mode == 'div_only':
            target_idx = payer[payer].index
            df = df.loc[target_idx]
            dy, op_profit, marcap, payer = dy[target_idx], op_profit[target_idx], marcap[target_idx], payer[target_idx]

        if df.empty: return pd.DataFrame()

        qt = QuantileTransformer(n_quantiles=min(100, len(df)), output_distribution='normal', random_state=42)
        
        if mode == 'wide':
            x_input = dy.copy()
            x_input[payer] += 2.0 
        else:
            x_input = dy.copy()
            
        x_norm = qt.fit_transform(x_input.values.reshape(-1,1)).ravel()
        x_final = MinMaxScaler().fit_transform(x_norm.reshape(-1,1)).ravel()
        y_final = op_profit.rank(pct=True).values

        names = df['Name'] if 'Name' in df.columns else pd.Series(index=df.index, data=df.index)

        return pd.DataFrame({
            'Name': names,
            'X_Momentum': x_final,
            'Y_Fundamental': y_final,
            'MarketCap': marcap.values,
            'Dividend_Yield': dy.values,
        }, index=df.index)

class RallyMapVisualizer:
    def run(self, data: pd.DataFrame, target_date_str: str):
        target_date = pd.to_datetime(target_date_str)
        
        if isinstance(data.index, pd.DatetimeIndex):
            snapshot = data[data.index == target_date]
        else:
            snapshot = data.copy()

        if snapshot.empty:
            print(f"Error: {target_date_str} 데이터 없음")
            return

        fe = FeatureEngineer()
        feats = fe.create_features(snapshot, mode='wide')
        
        if feats.empty:
            print("Error: 분석 데이터 부족")
            return

        self._plot(feats, target_date_str)

    def _plot(self, feats, date_str):
        X = feats[['X_Momentum', 'Y_Fundamental']].values
        min_samples = 3 if len(feats) > 10 else 1
        db = DBSCAN(eps=0.1, min_samples=min_samples).fit(X)
        feats['Cluster'] = db.labels_
        
        plt.figure(figsize=(12, 8))
        plt.scatter(feats['X_Momentum'], feats['Y_Fundamental'], 
                    s=np.log1p(feats['MarketCap'])*5 + 20, 
                    c=feats['Cluster'], cmap='tab10', alpha=0.8, edgecolors='white')
        
        texts = []
        top_stocks = feats.sort_values('MarketCap', ascending=False).head(10)
        for idx, row in top_stocks.iterrows():
            name = row['Name'] if isinstance(row['Name'], str) else str(idx)
            texts.append(plt.text(row['X_Momentum'], row['Y_Fundamental'], name, fontsize=9))

        try:
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except:
            pass

        plt.title(f"Market Rally Map ({date_str})", fontsize=16, fontweight='bold')
        plt.xlabel("Dividend Momentum", fontsize=12)
        plt.ylabel("Fundamental Rank", fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.show()

if __name__ == "__main__":
    PlotConfig.set_style()
    
    TARGET_DATE = "2024-05-20"
    
    loader = GitHubDataLoader(repo_owner='ParkYoungsig', repo_name='StockClustering')
    df = loader.load_date_data(TARGET_DATE)
    
    if not df.empty:
        viz = RallyMapVisualizer()
        viz.run(df, TARGET_DATE)
    else:
        print("Error: 데이터 로드 실패")
