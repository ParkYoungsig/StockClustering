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
LOG_DIRECTORY = 'log'
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

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
KOREAN_MARKET_EXCHANGES = ['KRX', 'KOSPI', 'KOSDAQ']

# Chart styling
CHART_STYLE = {
    'up_color': '#26a69a',
    'down_color': '#ef5350',
    'volume_color': '#6495ED',
    'ma_colors': ['#FF6B6B', '#4ECDC4', '#FFE66D'],
    'grid_alpha': 0.3,
    'line_width': 1.5,
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
LIST_FILE_LOCATION = "../data/list"
DATA_FILE_LOCATION = "../data"


# ------------------------------------------ #
# kmeans_clustering.py 등을 위한 config 정보   #
# ------------------------------------------ #
# 박영식 수정 : 설정 혹은 파라미터만 남기고 나머지는 kmeans_clustering.py 파일로 이동
# ===== 필수 입력 (kmm_0.py에서 옮겨온 값들) ===== :contentReference[oaicite:2]{index=2}
data_folder = (
    # r"C:\Users\gkrry\AI-Quant\StockClustering\데이터\데이터 수집 및 전처리\data"
    r"../data"
)
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
# kmeans_clustering.py 등을 위한 config 정보   #
# ------------------------------------------ #
"""GMM 파이프라인 설정(상수) 모음.

경로/기간/하이퍼파라미터(K 탐색 범위, 노이즈 필터 임계값 등)와
시각화에 사용할 클러스터 이름/색상 매핑을 정의합니다.
"""

# from pathlib import Path

# ==========================================
# [프로젝트 경로 설정]
# 프로젝트 루트 기준 상대 경로
# ==========================================
DEFAULT_RESULTS_DIR_NAME = "gmm_results"  # 결과 저장 디렉토리
DEFAULT_DATA_DIR_NAME = "..\data"  # 데이터 소스 디렉토리

# ==========================================
# [파이프라인 하이퍼파라미터 설정]
# GMM 분석 및 전처리 관련 주요 상수
# ==========================================
SNAPSHOT_FREQ = "Y"  # 스냅샷 주기 ("M": 월말, "Y": 연말)
START_YEAR = 2015  # 분석 시작 연도
END_YEAR = None  # 분석 종료 연도 (None일 경우 데이터의 마지막 연도까지)
FALLBACK_DAYS = 7  # 스냅샷 추출 시 데이터 부재 시 허용할 조회 범위(일)
K_RANGE = range(3, 6)  # GMM 클러스터 개수(K) 탐색 범위 (3 ~ 5)
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
