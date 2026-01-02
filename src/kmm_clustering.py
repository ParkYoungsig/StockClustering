# kmeans_clustering.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from kmm.kmm_align_clusters_mixed import align_cluster_labels_wide_csv_mixed
from kmm.kmm_pca_maps_aligned import plot_aligned_pca_maps
from kmm.kmm_pipeline import load_parquets_to_df_all, rolling_cluster_table

# 박영식 수정된 config.py 파일에서 설정값 가져오기
from config import DATA_FILE_LOCATION, OUTPUT_FILE_LOCATION
# from config import data_folder, dates, feature_cols
from config import dates, feature_cols


@dataclass
class KMeansClusteringConfig:
    # 필수 입력
    data_folder: str
    dates: List[str]
    feature_cols: List[str]

    # rolling_cluster_table 관련
    lower_q: float = 0.01
    upper_q: float = 0.99
    variant: str = "wins_std"  # "wins_std" or "wins_rob"
    pca_n: int = 8
    k_min: int = 2
    k_max: int = 5
    n_init: int = 200
    corr_drop_mode: str = "baseline"
    OUTPUT_DIR   = Path(OUTPUT_FILE_LOCATION).resolve()
    output_root: str = OUTPUT_DIR / "kmeans_outputs"
    save_per_date: bool = True
    make_plots: bool = True
    stability_seeds: Optional[List[int]] = field(default_factory=lambda: list(range(5)))
    stability_n_init: int = 20

    # alignment 관련
    do_align: bool = True
    w_overlap: float = 0.25
    w_centroid: float = 0.75
    min_jaccard: float = 0.0

    # PCA map 저장 관련
    do_pca_maps: bool = True
    label_all: bool = True
    label_fontsize: int = 6

    # 환경
    omp_num_threads: str = "1"


def run_kmeans_clustering(cfg: KMeansClusteringConfig) -> Dict[str, Any]:
    """
    kmm_1.py의 전체 파이프라인을 함수 하나로 묶은 실행기.
    - parquet 로드
    - rolling_cluster_table 실행
    - (옵션) 라벨 정렬 alignment
    - (옵션) aligned PCA 맵 저장

    Returns
    -------
    dict with keys:
      - df_all
      - out (rolling_cluster_table 결과 dict)
      - aligned_outputs (alignment 결과 dict or None)
      - pca_map_paths (list[str] or None)
    """

    # OMP 스레드 고정 (윈도우에서 KMeans MKL 경고/과다 스레드 방지용)
    os.environ["OMP_NUM_THREADS"] = str(cfg.omp_num_threads)

    # 1) 데이터 로드
    df_all = load_parquets_to_df_all(cfg.data_folder)

    # 2) rolling cluster table
    out = rolling_cluster_table(
        df_all=df_all,
        dates=cfg.dates,
        feature_cols=cfg.feature_cols,
        lower_q=cfg.lower_q,
        upper_q=cfg.upper_q,
        variant=cfg.variant,
        pca_n=cfg.pca_n,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        n_init=cfg.n_init,
        corr_drop_mode=cfg.corr_drop_mode,
        output_root=cfg.output_root,
        save_per_date=cfg.save_per_date,
        make_plots=cfg.make_plots,
        stability_seeds=cfg.stability_seeds,
        stability_n_init=cfg.stability_n_init,
    )

    roll_dir = out["roll_dir"]
    print("Saved to:", roll_dir)

    aligned_outputs = None
    pca_map_paths = None

    # 3) alignment
    if cfg.do_align:
        wide_csv_path = os.path.join(roll_dir, "cluster_labels_wide.csv")

        aligned_outputs = align_cluster_labels_wide_csv_mixed(
            input_csv_path=wide_csv_path,
            output_dir=roll_dir,
            per_date_root=os.path.join(roll_dir, "per_date"),
            w_overlap=cfg.w_overlap,
            w_centroid=cfg.w_centroid,
            min_jaccard=cfg.min_jaccard,
        )

        print("Aligned outputs:")
        for k, v in aligned_outputs.items():
            print(" -", k, "=>", v)

    # 4) PCA maps (aligned wide csv 기준)
    if cfg.do_pca_maps:
        aligned_path = os.path.join(roll_dir, "cluster_labels_wide_aligned.csv")
        pca_map_paths = plot_aligned_pca_maps(
            roll_dir=roll_dir,
            dates=cfg.dates,
            aligned_wide_csv_path=aligned_path,
            label_all=cfg.label_all,
            label_fontsize=cfg.label_fontsize,
        )

        print("Saved PCA maps:")
        for p in pca_map_paths:
            print(" -", p)

    return {
        "df_all": df_all,
        "out": out,
        "aligned_outputs": aligned_outputs,
        "pca_map_paths": pca_map_paths,
    }

def init_kmeans_clustering() :
    # ===== 나머지 파라미터 (원하면 여기서만 수정) ===== :contentReference[oaicite:3]{index=3}
    OUTPUT_DIR   = Path(OUTPUT_FILE_LOCATION).resolve()

    cfg = KMeansClusteringConfig(
        data_folder=DATA_FILE_LOCATION,
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
        output_root=OUTPUT_DIR / "kmeans_outputs",
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

if __name__ == "__main__":
    cfg = init_kmeans_clustering()
    kmm_report = run_kmeans_clustering(cfg)
