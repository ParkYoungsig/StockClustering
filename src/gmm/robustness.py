"""GMM robustness(기간 변화/롤링 윈도우) 평가 유틸.

- gmm_clustering.py를 너무 비대하게 만들지 않기 위해, 기간/윈도우 비교 로직을 이 모듈로 분리합니다.
- 이 모듈은 결과 파일(heatmap/MD/txt)을 저장하고, 메인 리포트에 넣을 요약 딕셔너리를 반환합니다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config

from gmm.model import (
    corr_similarity_matrix,
    cosine_similarity_matrix,
    match_centroids_by_similarity,
)
from gmm.pipeline_logic import select_best_k, train_gmm_per_year
from gmm.processer import (
    compute_cluster_stats,
    filter_noise,
    get_latest_year_frame,
    preprocess_features,
)
from gmm.reporter import write_period_slicing_robustness_report
from gmm.visualizer import plot_robustness_heatmap

logger = logging.getLogger(__name__)


def _safe_int(v: Any, default: int | None = None) -> int | None:
    try:
        return int(v)
    except Exception:
        return default


def run_period_slicing_robustness(
    *,
    df_base: pd.DataFrame,
    feature_cols: list[str],
    results_dir: Path,
    file_prefix: str,
) -> None:
    """기간 슬라이싱(케이스별) robustness를 수행하고 산출물을 저장합니다.

    - 케이스별 출력: K 선택 근거 차트 + 최신연도 클러스터 평균/개수 CSV
    - 케이스 간 비교: centroid cosine/corr (Hungarian matching 후 평균) 히트맵
    - 텍스트 리포트: gmm_period_robustness_report.txt

    NOTE: 메인 리포트(gmm_report.txt)에 넣을 요약은 rolling 쪽에서 반환하고,
          period slicing은 별도 txt로 저장합니다(너무 길어지는 걸 방지).
    """

    cases = config.ROBUSTNESS_PERIOD_CASES or []
    if not cases:
        return
    if "Year" not in df_base.columns:
        return

    years_all = df_base["Year"].dropna().astype(int)
    if years_all.empty:
        return
    data_min_year = int(years_all.min())
    data_max_year = int(years_all.max())

    case_outputs: dict[str, dict] = {}
    case_means: dict[str, pd.DataFrame] = {}
    ordered_names: list[str] = []

    for c in cases:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "case")
        start_y = _safe_int(c.get("start_year"), default=data_min_year)
        end_y = _safe_int(c.get("end_year"), default=data_max_year)
        if start_y is None:
            start_y = data_min_year
        if end_y is None:
            end_y = data_max_year
        start_y = max(data_min_year, int(start_y))
        end_y = min(data_max_year, int(end_y))
        if start_y > end_y:
            continue

        df_case = df_base[
            (df_base["Year"] >= start_y) & (df_base["Year"] <= end_y)
        ].copy()
        if df_case.empty:
            continue

        (
            df_clean,
            X_feats,
            _pcs_df,
            _scaler,
            _pca_model,
            prep_stats,
            feature_cols_used,
        ) = preprocess_features(
            df_case,
            feature_cols,
            use_pca=False,
            apply_iforest=True,
            low_variance_threshold=-1.0,
            corr_threshold=1.01,
            group_labels=df_case["Year"] if "Year" in df_case.columns else None,
        )
        if X_feats is None or len(X_feats) == 0:
            continue

        years = df_clean["Year"].to_numpy()
        case_prefix = f"{file_prefix}period_{name}_"

        (
            suggested_k,
            bic_scores,
            silhouette_by_k,
            best_k_mean,
        ) = select_best_k(
            X_feats,
            years,
            config.K_RANGE,
            manual_k=None,
            results_dir=results_dir,
            file_prefix=case_prefix,
        )

        final_k = int(suggested_k)
        labels_map, _probs_map, _quality_map, _last_model, _sort_feature = (
            train_gmm_per_year(
                X_feats, years, df_clean.index, feature_cols_used, final_k
            )
        )
        latest_year, df_latest = get_latest_year_frame(df_clean, labels_map)
        df_valid, _cluster_sizes, _noise_summary = filter_noise(
            df_latest, config.MIN_CLUSTER_FRAC
        )
        means, _stds, counts = compute_cluster_stats(df_valid, feature_cols_used)

        means.to_csv(results_dir / f"{case_prefix}cluster_means_latest.csv")
        counts.to_csv(results_dir / f"{case_prefix}cluster_counts_latest.csv")

        ordered_names.append(name)
        case_means[name] = means
        case_outputs[name] = {
            "start_year": int(start_y),
            "end_year": int(end_y),
            "best_k": int(final_k),
            "latest_year": int(latest_year),
            "n_latest": int(len(df_valid)),
            "prep_stats": prep_stats or {},
            "best_k_bic_mean": best_k_mean,
            "k_values": list(config.K_RANGE),
            "bic_mean": list(bic_scores or []),
            "silhouette_mean": dict(silhouette_by_k or {}),
            "case_prefix": case_prefix,
        }

    if len(ordered_names) < 2:
        return

    n = len(ordered_names)
    cosine_mean = np.full((n, n), np.nan, dtype=float)
    corr_mean = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                cosine_mean[i, j] = 1.0
                corr_mean[i, j] = 1.0
                continue
            a_df = case_means.get(ordered_names[i])
            b_df = case_means.get(ordered_names[j])
            if a_df is None or b_df is None:
                continue
            common = [c for c in a_df.columns if c in b_df.columns]
            if len(common) < 2:
                continue
            a = a_df[common].to_numpy(dtype=float)
            b = b_df[common].to_numpy(dtype=float)

            sim_cos = cosine_similarity_matrix(a, b)
            sim_cor = corr_similarity_matrix(a, b)
            mapping = match_centroids_by_similarity(sim_cos)

            matched_cos = [float(sim_cos[r, c]) for r, c in mapping.items()]
            matched_cor = [float(sim_cor[r, c]) for r, c in mapping.items()]
            cosine_mean[i, j] = (
                float(np.mean(matched_cos)) if matched_cos else float("nan")
            )
            corr_mean[i, j] = (
                float(np.mean(matched_cor)) if matched_cor else float("nan")
            )

    plot_robustness_heatmap(
        cosine_mean,
        ordered_names,
        results_dir / f"{file_prefix}period_centroid_cosine_mean.png",
        title="Centroid Cosine Similarity (Mean after Matching)",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_robustness_heatmap(
        corr_mean,
        ordered_names,
        results_dir / f"{file_prefix}period_centroid_corr_mean.png",
        title="Centroid Correlation Similarity (Mean after Matching)",
        vmin=-1.0,
        vmax=1.0,
    )

    report_payload = {
        "labels": ordered_names,
        "cases": case_outputs,
        "pairwise": {
            "centroid_cosine_mean": cosine_mean,
            "centroid_corr_mean": corr_mean,
        },
    }
    write_period_slicing_robustness_report(
        results_dir / f"{file_prefix}period_robustness_report.txt",
        report_payload,
    )


def _write_rolling_windows_md(
    output_path: Path,
    *,
    windows: list[tuple[int, int]],
    k_fixed: int,
    per_window: dict[str, dict],
    heatmap_files: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Rolling Window Robustness (K fixed)\n\n")
        f.write(f"- K fixed: **{k_fixed}**\n")
        f.write("- Window: 36 months(=3 years), step: 1 year\n")
        f.write(
            "- Metric: centroid similarity (cosine/correlation) after Hungarian matching\n\n"
        )

        f.write("## Windows\n\n")
        f.write(
            "| Window | Years | LatestYear | n_latest | GrowthCluster(Return) | OverheatCluster(Zscore/Disparity) |\n"
        )
        f.write("|---:|---|---:|---:|---|---|\n")
        for idx, (sy, ey) in enumerate(windows, start=1):
            key = f"W{idx}_{sy}_{ey}"
            row = per_window.get(key) or {}
            growth = row.get("growth_probe") or {}
            over = row.get("overheat_probe") or {}
            f.write(
                "| {i} | {sy}–{ey} | {ly} | {n} | {g} | {o} |\n".format(
                    i=idx,
                    sy=sy,
                    ey=ey,
                    ly=row.get("latest_year"),
                    n=row.get("n_latest"),
                    g=growth.get("text", "-"),
                    o=over.get("text", "-"),
                )
            )
        f.write("\n")

        f.write("## Outputs\n\n")
        for name, rel in heatmap_files.items():
            f.write(f"- {name}: `{rel}`\n")


def run_rolling_windows_robustness(
    *,
    df_base: pd.DataFrame,
    feature_cols: list[str],
    results_dir: Path,
    file_prefix: str,
) -> dict:
    """36개월(3년) 롤링 + 1년 스텝(8개 윈도우), K=4 고정 robustness.

    Returns:
        main 리포트(gmm_report.txt)에 넣을 수 있는 요약 dict
    """

    windows = list(config.ROBUSTNESS_ROLLING_WINDOWS or [])
    if not windows:
        return {"status": "skipped", "reason": "no_windows"}
    if "Year" not in df_base.columns:
        return {"status": "skipped", "reason": "no_year"}

    k_fixed = int(config.ROBUSTNESS_ROLLING_K_FIXED)
    per_window: dict[str, dict] = {}
    centroids_by_window: dict[str, pd.DataFrame] = {}

    for idx, (start_y, end_y) in enumerate(windows, start=1):
        df_w = df_base[
            (df_base["Year"] >= int(start_y)) & (df_base["Year"] <= int(end_y))
        ].copy()
        if df_w.empty:
            continue

        (
            df_clean,
            X_feats,
            _pcs_df,
            _scaler,
            _pca_model,
            _prep_stats,
            feature_cols_used,
        ) = preprocess_features(
            df_w,
            feature_cols,
            use_pca=False,
            apply_iforest=True,
            low_variance_threshold=-1.0,
            corr_threshold=1.01,
            group_labels=df_w["Year"] if "Year" in df_w.columns else None,
        )
        if X_feats is None or len(X_feats) == 0:
            continue

        years = df_clean["Year"].to_numpy()
        labels_map, _probs_map, _quality_map, _last_model, _sort_feature = (
            train_gmm_per_year(
                X_feats, years, df_clean.index, feature_cols_used, k_fixed
            )
        )
        latest_year, df_latest = get_latest_year_frame(df_clean, labels_map)
        df_valid, _cluster_sizes, _noise_summary = filter_noise(
            df_latest, config.MIN_CLUSTER_FRAC
        )
        means, _stds, counts = compute_cluster_stats(df_valid, feature_cols_used)

        key = f"W{idx}_{int(start_y)}_{int(end_y)}"
        prefix = f"{file_prefix}rolling_{int(start_y)}_{int(end_y)}_"
        means.to_csv(results_dir / f"{prefix}cluster_means_latest.csv")
        counts.to_csv(results_dir / f"{prefix}cluster_counts_latest.csv")

        growth_feat = (
            "Return_120d"
            if "Return_120d" in means.columns
            else ("Return_20d" if "Return_20d" in means.columns else None)
        )
        growth_probe = {}
        if growth_feat:
            try:
                cid = int(means[growth_feat].idxmax())
            except Exception:
                cid = means[growth_feat].idxmax()
            growth_probe = {
                "cluster": cid,
                "mean": float(means.loc[cid, growth_feat]),
                "text": f"C{cid} ({growth_feat}={float(means.loc[cid, growth_feat]):.3f})",
            }

        over_probe = {}
        if "Zscore_60d" in means.columns and "Disparity_60d" in means.columns:
            score = means["Zscore_60d"].astype(float) + means["Disparity_60d"].astype(
                float
            )
            try:
                cid2 = int(score.idxmax())
            except Exception:
                cid2 = score.idxmax()
            over_probe = {
                "cluster": cid2,
                "text": f"C{cid2} (Z+D={float(score.loc[cid2]):.3f})",
            }

        per_window[key] = {
            "start_year": int(start_y),
            "end_year": int(end_y),
            "latest_year": int(latest_year),
            "n_latest": int(len(df_valid)),
            "growth_probe": growth_probe,
            "overheat_probe": over_probe,
        }
        centroids_by_window[key] = means

    keys = list(centroids_by_window.keys())
    if len(keys) < 2:
        return {"status": "skipped", "reason": "insufficient_windows"}

    n = len(keys)
    cosine_mean = np.full((n, n), np.nan, dtype=float)
    corr_mean = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                cosine_mean[i, j] = 1.0
                corr_mean[i, j] = 1.0
                continue
            a_df = centroids_by_window[keys[i]]
            b_df = centroids_by_window[keys[j]]
            common = [c for c in a_df.columns if c in b_df.columns]
            if len(common) < 2:
                continue
            a = a_df[common].to_numpy(dtype=float)
            b = b_df[common].to_numpy(dtype=float)
            sim_cos = cosine_similarity_matrix(a, b)
            sim_cor = corr_similarity_matrix(a, b)
            mapping = match_centroids_by_similarity(sim_cos)
            matched_cos = [float(sim_cos[r, c]) for r, c in mapping.items()]
            matched_cor = [float(sim_cor[r, c]) for r, c in mapping.items()]
            cosine_mean[i, j] = (
                float(np.mean(matched_cos)) if matched_cos else float("nan")
            )
            corr_mean[i, j] = (
                float(np.mean(matched_cor)) if matched_cor else float("nan")
            )

    cos_path = results_dir / f"{file_prefix}rolling_centroid_cosine_mean.png"
    cor_path = results_dir / f"{file_prefix}rolling_centroid_corr_mean.png"
    plot_robustness_heatmap(
        cosine_mean,
        keys,
        cos_path,
        title="Rolling Windows: Centroid Cosine Similarity (Mean after Matching)",
        vmin=-1.0,
        vmax=1.0,
    )
    plot_robustness_heatmap(
        corr_mean,
        keys,
        cor_path,
        title="Rolling Windows: Centroid Correlation Similarity (Mean after Matching)",
        vmin=-1.0,
        vmax=1.0,
    )

    md_path = results_dir / f"{file_prefix}rolling_robustness.md"
    heat = {
        "centroid cosine heatmap": str(cos_path),
        "centroid corr heatmap": str(cor_path),
    }
    _write_rolling_windows_md(
        md_path,
        windows=windows,
        k_fixed=k_fixed,
        per_window=per_window,
        heatmap_files=heat,
    )

    # summary numbers for main report
    off_diag = ~np.eye(n, dtype=bool)
    cos_vals = cosine_mean[off_diag]
    cor_vals = corr_mean[off_diag]

    def _nan_stats(arr: np.ndarray) -> dict[str, float | None]:
        arr = np.asarray(arr, dtype=float)
        if np.all(np.isnan(arr)):
            return {"mean": None, "median": None}
        return {"mean": float(np.nanmean(arr)), "median": float(np.nanmedian(arr))}

    return {
        "status": "ok",
        "k_fixed": int(k_fixed),
        "windows": windows,
        "labels": keys,
        "md_path": str(md_path),
        "heatmaps": {"cosine": str(cos_path), "corr": str(cor_path)},
        "cosine_mean_matched": cosine_mean,
        "corr_mean_matched": corr_mean,
        "summary": {
            "cosine": _nan_stats(cos_vals),
            "corr": _nan_stats(cor_vals),
        },
    }
