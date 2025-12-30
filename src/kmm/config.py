from __future__ import annotations

from typing import List

# NOTE:
# - kmm_ready.py에 이미 KMeansClusteringConfig / run_kmeans_clustering가 구현돼 있습니다. :contentReference[oaicite:1]{index=1}
# - 우리는 "파라미터는 config.py에서만" 바꾸기 위해, 여기서 cfg를 만들어서 넘겨줍니다.
from kmm_ready import KMeansClusteringConfig  # src/kmm 가 sys.path에 잡히면 import 됨


def get_config() -> KMeansClusteringConfig:
    """
    Kmms 실행에 사용할 설정을 반환합니다.
    main.py에서는 이 설정을 건드리지 않고, 여기만 수정합니다.
    """

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

    # ===== 나머지 파라미터 (원하면 여기서만 수정) ===== :contentReference[oaicite:3]{index=3}
    cfg = KMeansClusteringConfig(
        data_folder=data_folder,
        dates=dates,
        feature_cols=feature_cols,
        lower_q=0.01,
        upper_q=0.99,
        variant="wins_std",  # "wins_std" or "wins_rob"
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

    return cfg
