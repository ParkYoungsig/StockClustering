"""GMM 클러스터링 파이프라인 실행 진입점."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib

import numpy as np
import pandas as pd

from config import DEFAULT_RESULTS_DIR_NAME,DEFAULT_DATA_DIR_NAME
from config import SNAPSHOT_FREQ,START_YEAR,END_YEAR,FALLBACK_DAYS,K_RANGE
from config import GMM_COVARIANCE_TYPE,GMM_N_INIT,GMM_MAX_ITER,GMM_REG_COVAR,GMM_ALIGN_METRIC
from config import MIN_CLUSTER_FRAC,CORR_THRESHOLD, MAX_MISSING_RATIO
from config import UMAP_N_NEIGHBORS, UMAP_MIN_DIST,CLUSTER_NAMES, CLUSTER_INTERPRETATIONS, CLUSTER_COLORS

from gmm.data_loader import FEATURE_COLUMNS, convert_df_to_snapshots
from gmm.pipeline_logic import select_best_k, train_gmm_per_year
from gmm.processer import (
    compute_cluster_stats,
    filter_noise,
    get_latest_year_frame,
    preprocess_features,
)
from gmm.report_metrics import compute_report_metrics
from gmm.reporter import (
    build_cluster_members_all_years,
    build_cluster_members_by_year,
    build_cluster_top_tickers,
    write_text_report,
)
from gmm.visualizer import (
    plot_cluster_boxplots,
    plot_cluster_heatmap,
    plot_parallel_coords,
    plot_radar_chart,
    plot_risk_return_scatter,
    plot_sankey,
    plot_umap_scatter,
)

logger = logging.getLogger(__name__)


def compute_umap_embedding(
    X: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray | None:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        return reducer.fit_transform(X)
    except Exception as e:
        logger.warning(f"UMAP 계산 건너뜀: {e}")
        return None


def save_artifacts(
    results_dir: Path,
    scaler,
    labels_map,
    final_k: int,
    features,
    means,
    model,
) -> None:
    """전처리 스케일러, 메타데이터, 모델 등을 아티팩트로 저장."""

    artifacts_dir = results_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, artifacts_dir / "scaler.pkl")
    joblib.dump(
        {
            "labels_per_year": labels_map,
            "final_k": final_k,
            "feature_columns": features,
            "cluster_means_latest": means,
        },
        artifacts_dir / "metadata.pkl",
    )

    if model is not None:
        joblib.dump(model, artifacts_dir / "gmm_latest_year.pkl")


def run_gmm_pipeline(
    *,
    df: pd.DataFrame,
    results_dir: Path,
    load_stats: Dict | None = None,
    manual_k: int | None = 4,
) -> str:
    logger.info("GMM 파이프라인 시작")

    if df is None or df.empty:
        return "Error: 입력 데이터가 유효하지 않습니다."

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not feature_cols:
        return "Error: 사용 가능한 Feature 컬럼이 없습니다."

    (
        df_clean,
        X_feats,
        _pcs_df,
        scaler,
        pca_model,
        prep_stats,
        feature_cols_used,
    ) = preprocess_features(
        df,
        feature_cols,
        use_pca=False,
        apply_iforest=True,
        low_variance_threshold=-1.0,
        corr_threshold=1.01,
        group_labels=df["Year"] if "Year" in df.columns else None,
    )
    if X_feats is None or len(X_feats) == 0:
        return "Error: 유효한 Feature 데이터가 없습니다."

    years = df_clean["Year"].to_numpy()

    if manual_k is None:
        final_k, bic_scores, stability_by_k, elbow_k, best_k_mean, best_k_median = (
            select_best_k(
                X_feats,
                years,
                K_RANGE,
                manual_k=None,
                results_dir=results_dir,
            )
        )
        k_values = list(K_RANGE)
    else:
        final_k = int(manual_k)
        bic_scores, k_values, stability_by_k, elbow_k, best_k_mean, best_k_median = (
            [],
            [],
            {},
            None,
            None,
            None,
        )
        logger.info(f"수동 K 사용: K={final_k}")

    labels_map, probs_map, _quality_map, last_model, sort_feature = train_gmm_per_year(
        X_feats, years, df_clean.index, feature_cols_used, final_k
    )

    target_year, df_latest = get_latest_year_frame(df_clean, labels_map)
    df_valid, _cluster_sizes, noise_summary = filter_noise(
        df_latest, MIN_CLUSTER_FRAC
    )

    means, stds, cluster_counts = compute_cluster_stats(df_valid, feature_cols_used)

    members_map = build_cluster_members_all_years(df_clean, labels_map)
    members_by_year = build_cluster_members_by_year(df_clean, labels_map)
    top_tickers_map = build_cluster_top_tickers(df_valid, top_n=10)

    df_valid.to_csv(results_dir / "final_clustered_data.csv", index=False)
    if probs_map and target_year in probs_map:
        probs_map[target_year].to_csv(results_dir / "final_probabilities_latest.csv")

    pd.DataFrame(
        [(cid, name) for cid, names in members_map.items() for name in names],
        columns=["cluster", "member"],
    ).to_csv(results_dir / "cluster_members_all_years.csv", index=False)

    # 연도별 멤버 요약 추가 저장
    rows_by_year = []
    for year, m in members_by_year.items():
        for cid, names in m.items():
            for name in names:
                rows_by_year.append((year, cid, name))
    if rows_by_year:
        pd.DataFrame(rows_by_year, columns=["year", "cluster", "member"]).to_csv(
            results_dir / "cluster_members_by_year.csv", index=False
        )

    save_artifacts(
        results_dir,
        scaler,
        labels_map,
        final_k,
        feature_cols_used,
        means,
        last_model,
    )

    pca_explained = (
        getattr(pca_model, "explained_variance_ratio_", [])
        if pca_model is not None
        else []
    )

    report_metrics = compute_report_metrics(
        df_clean=df_clean,
        df_valid=df_valid,
        X_feats=X_feats,
        probs_map=probs_map,
        target_year=target_year,
        low_conf_threshold=0.6,
        horizon_return=20,
    )

    write_text_report(
        results_dir / "report.txt",
        load_stats or {},
        prep_stats or {},
        bic_scores or [],
        k_values or [],
        final_k,
        np.array(pca_explained) if pca_explained is not None else np.array([]),
        means,
        stds,
        cluster_counts,
        top_tickers_map,
        noise_summary=noise_summary,
        stability_summary=(
            {"mean_by_k": stability_by_k, "elbow_k": elbow_k}
            if stability_by_k
            else None
        ),
        quality_summary=report_metrics.get("quality_summary"),
        silhouette_summary=report_metrics.get("silhouette_summary"),
        transition_summary=report_metrics.get("transition_summary"),
        ex_post_summary=report_metrics.get("ex_post_summary"),
        best_k_mean=best_k_mean,
        best_k_median=best_k_median,
        label_alignment_feature=sort_feature,
        cluster_names=CLUSTER_NAMES,
        cluster_interpretations=getattr(config, "CLUSTER_INTERPRETATIONS", None),
    )

    plot_cluster_heatmap(
        means, results_dir / "heatmap.png", cluster_names=CLUSTER_NAMES
    )
    plot_radar_chart(
        means,
        results_dir / "radar.png",
        cluster_names=CLUSTER_NAMES,
        cluster_colors=CLUSTER_COLORS,
    )
    plot_parallel_coords(
        df_valid,
        feature_cols_used,
        results_dir / "parallel.png",
        cluster_names=CLUSTER_NAMES,
        cluster_colors=CLUSTER_COLORS,
    )
    plot_risk_return_scatter(
        means,
        results_dir / "risk_return.png",
        cluster_names=CLUSTER_NAMES,
        cluster_colors=CLUSTER_COLORS,
    )
    plot_cluster_boxplots(
        df_valid,
        feature_cols_used,
        results_dir / "cluster_boxplots.png",
        cluster_colors=CLUSTER_COLORS,
    )

    if "Ticker" in df_clean.columns:
        plot_sankey(
            df_clean,
            labels_map,
            results_dir / "sankey.html",
            cluster_names=CLUSTER_NAMES,
            cluster_colors=CLUSTER_COLORS,
        )

    embedding = compute_umap_embedding(X_feats)
    if embedding is not None and len(embedding) == len(df_clean):
        labels_series = pd.Series(-1, index=df_clean.index)
        for year, lbl in labels_map.items():
            common_idx = labels_series.index.intersection(lbl.index)
            labels_series.loc[common_idx] = lbl.loc[common_idx].astype(int)
        labels_array = labels_series.to_numpy()
        plot_umap_scatter(
            embedding,
            labels_array,
            results_dir / "umap.png",
            cluster_names=CLUSTER_NAMES,
            cluster_colors=CLUSTER_COLORS,
        )

    msg = (
        f"GMM 분석 완료 (최종 K: {final_k}, 최신 연도: {target_year}, "
        f"유효 샘플: {len(df_valid)})"
    )
    logger.info(msg)
    return msg


class GMM:
    """엔트리에서 스냅샷 변환 후 파이프라인을 실행하는 얇은 래퍼."""

    def __init__(self, df: pd.DataFrame | None = None, results_dir: Path | None = None):
        self.results_dir = results_dir or Path(DEFAULT_RESULTS_DIR_NAME)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.load_stats: Dict | None = None

        if df is not None and not df.empty:
            try:
                self.df = convert_df_to_snapshots(
                    df,
                    freq=SNAPSHOT_FREQ,
                    start_year=START_YEAR,
                    end_year=END_YEAR,
                )
            except Exception as e:
                logger.warning(f"스냅샷 변환 실패, 원본 사용: {e}")
                self.df = df
        else:
            self.df = df

    def run(self, manual_k: int | None = 4) -> str:
        return run_gmm_pipeline(
            df=self.df,
            results_dir=self.results_dir,
            load_stats=self.load_stats,
            manual_k=manual_k,
        )
