# NOTE: 이 파일은 프로젝트 전역 설정을 담습니다.
# 여러 모듈에서 import 되므로, 타입/Path 등 기본config. import는 반드시 안전하게 포함합니다.

from __future__ import annotations

from pathlib import Path
from typing import List

# # Default analysis parameters
# DEFAULT_ROLLING_PERIOD = 252 * 5  # 1 year for daily data
# DEFAULT_INTERVAL = '1d'
# DEFAULT_LEVERAGE_LIMIT = 1.0        # No leverage
# DEFAULT_SHORT_LIMIT = 0.0           # No shorting

# # Moving average periods for price charts
# MA_PERIODS = [5, 20, 50]

# # Risk-free rate (annualized)
# RISK_FREE_RATE = 0.0  # 0% by default, adjust as needed

# # Output settings
# OUTPUT_DPI = 150  # Chart resolution
# OUTPUT_FORMAT = 'png'

# # Database settings
# DB_PATH = 'data/qstats_plotter.db'
# USE_CACHE = True

# Logging settings
LOG_DIRECTORY     = "log"
LOG_LEVEL         = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_MAX_BYTES     = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT  = 5

# # Data fetching settings
# FETCH_TIMEOUT = 30  # seconds
# RETRY_ATTEMPTS = 3

# # Benchmark mappings
# BENCHMARK_ALIASES = {
#     'sp500': '^GSPC',
#     's&p500': '^GSPC',
#     'spy': 'SPY',
#     'kospi': 'KS11',
#     'kosdaq': 'KQ11',
#     'nasdaq': '^IXIC',
#     'dow': '^DJI',
# }

# Korean market tickers (for auto-detection)
KOREAN_MARKET_EXCHANGES = ["KRX", "KOSPI", "KOSDAQ"]

# Chart styling
CHART_STYLE = {
    "up_color": "#26a69a",
    "down_color": "#ef5350",
    "volume_color": "#6495ED",
    "ma_colors": ["#FF6B6B", "#4ECDC4", "#FFE66D"],
    "grid_alpha": 0.3,
    "line_width": 1.5,
}

# # Metric display settings
# METRICS_DECIMAL_PLACES = 4
# SUMMARY_METRICS = [
#     'sharpe',
#     'sortino',
#     'calmar',
#     'max_drawdown',
#     'win_rate',
#     'information_ratio'
# ]

# # File naming templates
# FILE_TEMPLATES = {
#     'price_chart': '{ticker}_price_chart.{ext}',
#     'risk_metrics': '{ticker}_risk_metrics.{ext}',
#     'drawdown': '{ticker}_drawdown.{ext}',
#     'distribution': '{ticker}_returns_dist.{ext}',
#     'price_data': '{ticker}_price_data.csv',
#     'metrics_data': '{ticker}_risk_metrics.csv',
#     'summary': '{ticker}_summary.txt',
# }


# ------------------------------------------#
# collect_create_data.py 를 위한 config 정보  #
# ------------------------------------------#

# Module-level constants
# LIST_FILE_LOCATION      = "./list"
DATA_FILE_LOCATION      = r"./data"
INPUT_FILE_LOCATION     = r"./input"
OUTPUT_FILE_LOCATION    = r"./output"

# ------------------------------------------ #
# kmeans_clustering.py 등을 위한 config 정보   #
# ------------------------------------------ #
# 박영식 수정 : 설정 혹은 파라미터만 남기고 나머지는 kmeans_clustering.py 파일로 이동
# ===== 필수 입력 (kmm_0.py에서 옮겨온 값들) ===== :contentReference[oaicite:2]{index=2}

# "INPUT_FILE_LOCATION" 사용하도록 소스 수정
# data_folder = ( 
#     # r"C:\Users\gkrry\AI-Quant\StockClustering\데이터\데이터 수집 및 전처리\data"
#     r"./data"
# )

dates: List[str] = [
    "2024-11-04",
    "2024-11-05",
    "2024-11-06",
    "2024-11-07",
    "2024-11-08",
]
feature_cols: List[str] = [
    "Return_20d",
    "Return_60d",
    "vol_20",
    "vol_60",
    "PER",
    "ROE",
    "ROE_YoY",
    "EPS_YoY",
]



# ------------------------------------------ #
# gmm_clustering.py 등을 위한 config 정보   #
# ------------------------------------------ #
"""GMM 파이프라인 설정(상수) 모음.

경로/기간/하이퍼파라미터(K 탐색 범위, 노이즈 필터 임계값 등)와
시각화에 사용할 클러스터 이름/색상 매핑을 정의합니다.
"""

# 프로젝트 루트 경로(현재 파일: <root>/src/config.py)
# 박영식 : 수정 run.bat 실행위치를 PROJECT_ROOT로 설정
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(".").resolve()

# ==========================================
# [프로젝트 경로 설정]
# 프로젝트 루트 기준 상대 경로
# ==========================================
DEFAULT_DATA_DIR_NAME    = str(DATA_FILE_LOCATION  )  # 데이터 소스 디렉토리
DEFAULT_GMM_RESULTS_NAME = str(Path(OUTPUT_FILE_LOCATION) / "gmm_outputs")  # 결과 저장 디렉토리

# ==========================================
# [파이프라인 하이퍼파라미터 설정]
# GMM 분석 및 전처리 관련 주요 상수
# ==========================================
SNAPSHOT_FREQ = "Y"  # 스냅샷 주기 ("M": 월말, "Y": 연말)
START_YEAR = 2015  # 분석 시작 연도
END_YEAR = None  # 분석 종료 연도 (None일 경우 데이터의 마지막 연도까지)
FALLBACK_DAYS = 7  # 스냅샷 추출 시 데이터 부재 시 허용할 조회 범위(일)
K_RANGE = range(2, 10)  # GMM 클러스터 개수(K) 탐색 범위 (2 ~ 9)

# ==========================================
# [Robustness 설정]
# - 학습 기간(윈도우)을 바꿔도 평가셋(기본: 최신 연도)에서 군집 할당이 얼마나 유지되는지
#   ARI/NMI로 정량화합니다.
# - exclude_eval_year=True이면 평가 연도 데이터는 학습에서 제외(일종의 OOS)합니다.
# ==========================================
ROBUSTNESS_WINDOW_YEARS = [3, 5, 7, 10]
ROBUSTNESS_EXCLUDE_EVAL_YEAR = True

# ==========================================
# [Robustness: 기간 슬라이싱 비교]
# - 예: 전체(2015-최신), 최근(2017-최신), 과거(2015-2020)
# - 각 케이스별로: K 선택, 클러스터 평균, 해석/중심 유사도 비교 결과를 저장
# - 기본은 OFF (한 번 실행에 여러 번 학습되므로 오래 걸릴 수 있음)
# ==========================================
ROBUSTNESS_PERIOD_SLICING_ENABLED = False

# end_year가 None이면 데이터의 최신 연도로 대체됩니다.
ROBUSTNESS_PERIOD_CASES = [
    {"name": "A_full", "start_year": START_YEAR, "end_year": END_YEAR},
    {"name": "B_recent", "start_year": 2017, "end_year": END_YEAR},
    {"name": "C_early", "start_year": START_YEAR, "end_year": 2020},
]

# ==========================================
# [Robustness: 36개월 롤링(1년 스텝), K 고정]
# - 사용자가 지정한 8개 윈도우를 그대로 실행합니다.
# - 각 윈도우에서 동일 피처/동일 전처리/동일 K로 학습 후,
#   최신 연도 클러스터 중심(평균 벡터)을 기준으로 구조 유사도를 비교합니다.
# ==========================================
ROBUSTNESS_ROLLING_WINDOWS_ENABLED = False
ROBUSTNESS_ROLLING_K_FIXED = 4
ROBUSTNESS_ROLLING_WINDOWS = [
    (2015, 2017),
    (2016, 2018),
    (2017, 2019),
    (2018, 2020),
    (2019, 2021),
    (2020, 2022),
    (2021, 2023),
    (2022, 2024),
]
GMM_COVARIANCE_TYPE = "diag"  # 롤링/소표본 안정성을 위한 공분산 구조
GMM_N_INIT = 1  # warm start 활용 위해 1회 초기화
GMM_MAX_ITER = 300
GMM_REG_COVAR = 1e-6
GMM_ALIGN_METRIC = "bhattacharyya"  # 헝가리안 정렬 시 비용 metric
MIN_CLUSTER_FRAC = 0.03  # 노이즈 필터링 임계값 (전체 데이터의 3% 미만 군집은 제거)

# ==========================================
# [피처 중복 제거(상관 기반)]
# - 상관(스피어만) 절대값이 임계값 이상인 피처는 중복으로 보고 일부 제거
# - 너무 낮추면 정보 손실 가능 → 0.97 정도가 무난한 출발점
# ==========================================
CORR_THRESHOLD = 0.97

# (자동 후보 확장/결측률 필터/PCA 자동 적용 로직은 현재 오케스트레이터에서 사용하지 않습니다)
MAX_MISSING_RATIO = 0.30

# 차원 축소(UMAP) 시각화 설정
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# ==========================================
# [시각화 설정]
# 클러스터별 표시 이름 및 색상 매핑
# ==========================================
CLUSTER_NAMES = {
    0: "0. Bearish Capitulation (하락/투매)",
    1: "1. Range-bound Neutral (보합/중립)",
    2: "2. Organic Growth (안정적 상승)",
    3: "3. Speculative Overheat (과열/정점)",
}

# 클러스터 해석(리포트 출력용)도 규칙이 아니라 매핑으로 관리합니다.
# 필요하면 프로젝트 상황에 맞게 문구만 조정하면 됩니다.
CLUSTER_INTERPRETATIONS = {
    0: "Bearish Capitulation: 하락 추세 속 투매/공포 구간",
    1: "Range-bound Neutral: 방향성 약한 횡보/중립 구간",
    2: "Organic Growth: 변동성 과하지 않은 안정적 상승 구간",
    3: "Speculative Overheat: 기대 과열/정점(조정 위험) 구간",
}

# 시각적 일관성을 위한 고정 색상 팔레트
CLUSTER_COLORS = {
    0: "#4575b4",  # Crash (파랑)
    1: "#91bfdb",  # Vol. Trap (하늘)
    2: "#fee090",  # Defensive (노랑)
    3: "#fc8d59",  # Smart Trend (주황)
}




# ------------------------------------------ #
# hdb_clustering.py 등을 위한 config 정보      #
# ------------------------------------------ #

# import os

# ---------------------------------------------------------
# [1] 경로 설정
# ---------------------------------------------------------
# 데이터가 들어있는 폴더 (Parquet 파일들)
DATA_FOLDER = r"./data"

# 결과물이 저장될 폴더 (이미지, CSV 등)
OUTPUT_FOLDER = r"./output/hdb_outputs"

# ---------------------------------------------------------
# [2] 분석 피처 설정 (우선순위 순서대로)
# ---------------------------------------------------------
# X축: 배당 관련 지표
# (파일에 'Dividend_Yield'가 없으면 '배당수익률'을 찾고, 그것도 없으면 'DPS'를 찾음)
X_FEATS = ["Dividend_Yield", "배당수익률", "DPS"]

# Y축: 모멘텀(주가 상승) 관련 지표
# (파일에 'Return_60d'가 없으면 'Return_20d' -> 'Chg_Pct' 순으로 찾음)
Y_FEATS = ["Return_60d", "Return_20d", "Chg_Pct", "Change", "등락률"]

# ---------------------------------------------------------
# [3] DBSCAN 클러스터링 설정
# ---------------------------------------------------------
# 목표로 하는 군집 개수 범위 (이 범위에 들어오도록 eps 자동 조절)
TARGET_CLUSTERS_MIN = 4
TARGET_CLUSTERS_MAX = 6

# eps 탐색 범위 (시작값, 끝값, 간격)
EPS_RANGE_START = 0.1
EPS_RANGE_END = 1.0
EPS_STEP = 0.02

# 최소 샘플 수 (이 숫자 이상의 종목이 모여야 군집으로 인정)
MIN_SAMPLES = 3

# ---------------------------------------------------------
# [4] 시각화 설정
# ---------------------------------------------------------
# 한글 폰트 (Windows: 'Malgun Gothic', Mac: 'AppleGothic', Linux: 'NanumGothic')
import platform
_os = platform.system()
if _os == "Windows":
    FONT_FAMILY = 'Malgun Gothic'
elif _os == "Darwin":
    FONT_FAMILY = 'AppleGothic'
else:
    FONT_FAMILY = 'NanumGothic'

# 그래프 크기 (가로, 세로)
FIG_SIZE = (14, 10)

# ---------------------------------------------------------
# [5] 기타
# ---------------------------------------------------------
# 저장할 때 인코딩 방식 (엑셀에서 안 깨지게 하려면 'utf-8-sig')
CSV_ENCODING = 'utf-8-sig'
