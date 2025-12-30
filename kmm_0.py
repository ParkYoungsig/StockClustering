# kmm_1.py
from kmeans_clustering import KMeansClusteringConfig, run_kmeans_clustering


def main():
    data_folder = (
        r"C:\Users\gkrry\AI-Quant\StockClustering\데이터\데이터 수집 및 전처리\data"
    )
    dates = ["2024-12-02", "2024-12-03"]

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
        lower_q=0.01,
        upper_q=0.99,
        variant="wins_std",
        pca_n=8,
        k_min=2,
        k_max=5,
        n_init=200,
        corr_drop_mode="baseline",
        output_root="kmeans_outputs",
        save_per_date=True,
        make_plots=True,
        stability_seeds=list(range(5)),
        stability_n_init=20,
        do_align=True,
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
