# kmm_1.py
from kmm_ready import KMeansClusteringConfig, run_kmeans_clustering


def main():
    data_folder = (
        r"C:\Users\gkrry\AI-Quant\StockClustering\데이터\데이터 수집 및 전처리\data"
    )
    dates = ["2024-12-02", "2024-12-03"]

    # 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'Return_1d',
    # 'Return_5d', 'Return_20d', 'Return_30d', 'Return_50d', 'Return_60d', 'Return_100d', 'Return_120d', 'Return_200d',
    # 'vol_20', 'vol_60', 'vol_60_sqrt252', 'log_vol', 'vol_ratio_60', 'avg_log_vol_ratio_60', 'std_log_vol_ratio_60',
    # 'RSI_14', 'RSI_14_60avg', 'MFI_14', 'ATR_14', 'NATR_14', 'ADX_14',
    # 'Disparity_5d', 'Disparity_20d', 'Disparity_60d', 'Disparity_120d',
    # 'Mean_60d', 'Median_60d', 'Std_60d', 'Sharpe_60d', 'Sharpe_252d', 'Sortino_60d', 'Sortino_252d','Skewness_60d', 'Zscore_60d',
    # 'BPS', 'DPS', 'EPS', '배당수익률', '매출액', '영업이익', 'PER', 'PBR', 'ROE', '배당성향',
    # 'ROE_YoY', 'EPS_YoY', '영업이익_YoY', '매출액_YoY', 'Ticker'

    feature_cols = [
        "Return_20d",
        "Return_60d",
        "vol_20",
        "vol_60",
        "PER",
        "ROE",
        "ROE_YoY",
        "EPS_YoY",
    ]

    cfg = KMeansClusteringConfig(
        data_folder=data_folder,
        dates=dates,
        feature_cols=feature_cols,
        lower_q=0.01,  # winsorization 하한 분위수
        upper_q=0.99,  # winsorization 상한 분위수
        variant="wins_std",  # "wins_std" or "wins_rob", Winsorization 후 표준화 방식 결정
        pca_n=8,  # 확인해보고 싶은 PCA 차원 수
        k_min=2,  # 최소 클러스터 수
        k_max=5,  # 최대 클러스터 수
        n_init=200,  # KMeans 반복 횟수
        corr_drop_mode="baseline",
        output_root="kmeans_outputs",  # 출력 폴더 루트
        save_per_date=True,
        make_plots=True,
        stability_seeds=list(range(5)),
        stability_n_init=20,
        do_align=True,  # 라벨 정렬 수행 여부
        w_overlap=0.25,
        w_centroid=0.75,
        min_jaccard=0.0,
        do_pca_maps=True,
        label_all=True,
        label_fontsize=6,
        omp_num_threads="1",
    )

    result = run_kmeans_clustering(cfg)
    # 필요하면 result["out"]["roll_dir"] 같은 것 계속 활용하면 됨.


if __name__ == "__main__":
    main()
# ============================================================# ============================================================# ============================================================# ============================================================# ============================================================
