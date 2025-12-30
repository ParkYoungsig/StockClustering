import os

os.environ["OMP_NUM_THREADS"] = "1"

from kmm_align_clusters_mixed import align_cluster_labels_wide_csv_mixed
from kmm_pipeline import load_parquets_to_df_all, rolling_cluster_table

# 데이터 폴더 경로 설정 및 데이터 로드
data_folder = (
    r"C:\Users\gkrry\AI-Quant\StockClustering\데이터\데이터 수집 및 전처리\data"
)
df_all = load_parquets_to_df_all(data_folder)

# 원하는 날짜 입력
dates = ["2024-12-02", "2024-12-03"]

# '종목명',
# 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Return_1d',
# 'Return_5d', 'Return_20d', 'Return_30d', 'Return_50d', 'Return_60d', 'Return_100d', 'Return_120d', 'Return_200d',
# 'vol_20', 'vol_60', 'vol_60_sqrt252', 'log_vol', 'vol_ratio_60', 'avg_log_vol_ratio_60', 'std_log_vol_ratio_60',
# 'RSI_14', 'RSI_14_60avg', 'MFI_14', 'ATR_14', 'NATR_14', 'ADX_14',
# 'Disparity_5d', 'Disparity_20d', 'Disparity_60d', 'Disparity_120d',
# 'Mean_60d', 'Median_60d', 'Std_60d', 'Sharpe_60d', 'Sharpe_252d', 'Sortino_60d', 'Sortino_252d','Skewness_60d', 'Zscore_60d',
# 'BPS', 'DPS', 'EPS', '배당수익률', '매출액', '영업이익', 'PER', 'PBR', 'ROE', '배당성향',
# 'ROE_YoY', 'EPS_YoY', '영업이익_YoY', '매출액_YoY', 'Ticker'

# 사용할 피처 컬럼 리스트 (None이면 numeric 전체 사용)
feature_cols = [
    "Return_5d",
    "Return_20d",
    "Return_30d",
    "Return_50d",
    "Return_60d",
    "Return_100d",
    "Return_120d",
    "Return_200d",
    "vol_20",
    "vol_60",
    "vol_60_sqrt252",
    "log_vol",
    "vol_ratio_60",
    "avg_log_vol_ratio_60",
    "std_log_vol_ratio_60",
    "RSI_14",
    "RSI_14_60avg",
    "MFI_14",
    "ATR_14",
    "NATR_14",
    "ADX_14",
    "Disparity_5d",
    "Disparity_20d",
    "Disparity_60d",
    "Disparity_120d",
    "Mean_60d",
    "Median_60d",
    "Std_60d",
    "Sharpe_60d",
    "Sharpe_252d",
    "Sortino_60d",
    "Sortino_252d",
    "Skewness_60d",
    "Zscore_60d",
    "BPS",
    "DPS",
    "EPS",
    "배당수익률",
    "매출액",
    "영업이익",
    "PER",
    "PBR",
    "ROE",
    "배당성향",
    "ROE_YoY",
    "EPS_YoY",
    "영업이익_YoY",
    "매출액_YoY",
]

# "Return_20d",
# "Return_60d",
# "vol_20",
# "vol_60",
# "PER",  # 아마 수익률이 높을수록 PER 높을 것
# "ROE",
# "ROE_YoY",
# "EPS_YoY",

out = rolling_cluster_table(
    df_all=df_all,
    dates=dates,
    feature_cols=feature_cols,
    lower_q=0.01,  # winsorization 하한 분위수
    upper_q=0.99,  # winsorization 상한 분위수
    variant="wins_std",  # "wins_std" or "wins_rob", Winsorization 후 표준화 방식 결정
    pca_n=8,  # 확인해보고 싶은 PCA 차원 수
    k_min=2,  # 최소 클러스터 수    --> 5로 고정했음. 자꾸 바뀌는게 스트레스 받음.
    k_max=5,  # 최대 클러스터 수 --> 엘보우/실루엣 기준 k 선택 범위
    n_init=200,
    corr_drop_mode="baseline",  # 권장: 첫날 기준으로 corr-drop 고정
    output_root="kmeans_outputs",
    save_per_date=True,  # 날짜별 k_eval/EVR/labels 저장
    make_plots=True,  # 날짜별 elbow/silhouette png 저장 원하면 True
    stability_seeds=list(
        range(5)
    ),  # 안정성 검사, 원치 않으면 =None, 원하면 =list(range(20))
    stability_n_init=20,
)

print("Saved to:", out["roll_dir"])

# ============================================================
# (추가) overlap + centroid 혼합 매칭으로 라벨 정렬(alignment) --> 자꾸 바뀌는 문제 해결 노력
# ============================================================
wide_csv_path = os.path.join(out["roll_dir"], "cluster_labels_wide.csv")

aligned_outputs = align_cluster_labels_wide_csv_mixed(
    input_csv_path=wide_csv_path,
    output_dir=out["roll_dir"],  # 정렬 결과도 동일 폴더에 저장
    per_date_root=os.path.join(out["roll_dir"], "per_date"),  # centroids 파일 위치
    w_overlap=0.25,  # 0.6 -> 0.25
    w_centroid=0.75,  # 0.4 -> 0.75
    min_jaccard=0.0,  # 0.05 -> 0.0 = 스킵 방지(오버랩 거의 없어도 매칭 시도)
)


# # ============================================================
# # aligned --> PCA 2D, 3D 임베딩 맵 저장
# # ============================================================
from kmm_pca_maps_aligned import plot_aligned_pca_maps

saved = plot_aligned_pca_maps(
    roll_dir=out["roll_dir"],
    dates=dates,
    aligned_wide_csv_path=os.path.join(
        out["roll_dir"], "cluster_labels_wide_aligned.csv"
    ),
    label_all=True,  # 너무 지저분하면 False로 바꿔도 됨
    label_fontsize=6,
)

print("Saved PCA maps:")
for p in saved:
    print(" -", p)


print("Aligned outputs:")
for k, v in aligned_outputs.items():
    print(" -", k, "=>", v)
