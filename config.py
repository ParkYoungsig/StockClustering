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