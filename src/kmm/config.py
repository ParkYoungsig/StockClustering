# 박영식 수정 : 설정 혹은 파라미터만 남기고 나머지는 kmeans_clustering.py 파일로 이동

# ===== 필수 입력 (kmm_0.py에서 옮겨온 값들) ===== :contentReference[oaicite:2]{index=2}
data_folder = (
    r"C:\Users\gkrry\AI-Quant\StockClustering\데이터\데이터 수집 및 전처리\data"
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