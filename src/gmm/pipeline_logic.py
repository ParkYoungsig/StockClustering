"""GMM 파이프라인 핵심 로직 모듈.

- K 탐색(BIC(mean)/Silhouette(mean))과 학습/라벨 정렬을 담당
- 오케스트레이터와 리포터 사이의 순수 로직을 모아둠
"""

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import config

from gmm.model import (
    align_clusters,
    align_labels_by_feature,
    evaluate_bic_across_years,
    evaluate_silhouette_across_years,
    remap_labels,
)
from gmm.visualizer import (
    plot_bic_curve,
    plot_silhouette_curve,
)

logger = logging.getLogger(__name__)


def select_best_k(
    X: np.ndarray,
    years: np.ndarray,
    k_range: Iterable[int],
    manual_k: int | None,
    results_dir: Path,
    file_prefix: str = "",
) -> Tuple[
    int,
    List[float],
    Dict[int, float],
    int | None,
]:
    """
    데이터에 가장 적합한 클러스터 개수(K)를 탐색합니다.

    [절차]
    1. BIC(베이지안 정보 기준) 점수 계산 (연도별 → mean 집계)
    2. Silhouette 점수 계산 (연도별 → mean 집계)
    3. mean curve 2개 저장 (BIC / Silhouette)
    4. 최적 K 결정 (자동 또는 수동)

    Returns:
        Tuple: (최종 K, mean BIC 리스트, mean silhouette_by_k, mean BIC 기준 best_k)
    """

    logger.info("최적 K 탐색 시작 (BIC(mean) / Silhouette(mean) 평가)...")

    k_list = list(k_range)

    # BIC 평가 (연도별 → 평균/중앙값)
    _, mean_bic, _median_bic, best_k_mean, _best_k_median = evaluate_bic_across_years(
        X, years=years, k_values=k_list
    )

    # Silhouette (연도별 silhouette → K별 평균)
    _sil_per_year, sil_mean_by_k, _sil_median_by_k = evaluate_silhouette_across_years(
        X, years=years, k_values=k_list
    )

    # 결과 시각화 (요구사항: mean curve 2개만 유지)
    plot_bic_curve(k_list, mean_bic, results_dir / f"{file_prefix}bic_curve_mean.png")
    if sil_mean_by_k:
        ks = sorted(sil_mean_by_k.keys())
        plot_silhouette_curve(
            ks,
            [sil_mean_by_k[k] for k in ks],
            results_dir / f"{file_prefix}silhouette_curve_mean.png",
        )

    # 최종 K 결정 (수동 설정 우선)
    if manual_k is not None:
        suggested = best_k_mean if best_k_mean is not None else "N/A"
        logger.info(
            f"수동 설정된 K={manual_k}를 사용합니다. (알고리즘 추천: {suggested})"
        )
        final_k = manual_k
    else:
        final_k = best_k_mean if best_k_mean is not None else k_list[0]
        logger.info(f"알고리즘 자동 선택 K: {final_k} (Mean BIC 기준)")

    return (
        final_k,
        mean_bic,
        sil_mean_by_k,
        best_k_mean,
    )


def train_gmm_per_year(
    X: np.ndarray,
    years: np.ndarray,
    df_idx: pd.Index,
    feature_cols: List[str],
    k: int,
) -> Tuple[
    Dict[int, pd.Series],
    Dict[int, pd.DataFrame],
    Dict[int, Dict],
    GaussianMixture | None,
    str,
]:
    """
    연도별 GMM 학습 + Hungarian 라벨 정렬 + warm start(means_init).

    Returns:
        labels_per_year: 정렬된 하드 라벨
        probs_per_year: 정렬된 소속확률(soft label)
        latest_model: 최신 연도 모델
        sort_feature: 첫 연도 정렬에 사용한 기준 Feature
    """

    labels_per_year: Dict[int, pd.Series] = {}
    probs_per_year: Dict[int, pd.DataFrame] = {}
    quality_per_year: Dict[int, Dict] = {}
    latest_model: GaussianMixture | None = None

    preferred = [c for c in ["Return_120d", "Return_20d"] if c in feature_cols]
    sort_feature = preferred[0] if preferred else feature_cols[0]
    sort_idx = feature_cols.index(sort_feature)

    unique_years = sorted(np.unique(years))
    prev_centers: np.ndarray | None = None
    prev_covs: np.ndarray | None = None

    for year in unique_years:
        mask = years == year
        X_year = X[mask]

        # GMM은 n_samples >= n_components가 필요
        min_required = max(2, int(k))
        if X_year.shape[0] < min_required:
            quality_per_year[year] = {
                "status": "skipped",
                "reason": f"insufficient_samples({X_year.shape[0]} < {min_required})",
            }
            logger.warning(
                "연도 %s 스킵: 샘플 %d개 < 최소 %d개 (freq=%s)",
                year,
                X_year.shape[0],
                min_required,
                config.SNAPSHOT_FREQ,
            )
            continue

        model = GaussianMixture(
            n_components=k,
            covariance_type=config.GMM_COVARIANCE_TYPE,
            random_state=42,
            n_init=config.GMM_N_INIT,
            max_iter=config.GMM_MAX_ITER,
            reg_covar=config.GMM_REG_COVAR,
            means_init=prev_centers,
            init_params="kmeans",
        )
        model.fit(X_year)

        proba_raw = model.predict_proba(X_year)
        raw_labels = np.argmax(proba_raw, axis=1)

        if prev_centers is None:
            aligned_labels, mapping = align_labels_by_feature(
                raw_labels, X_year, sort_idx
            )
            aligned_proba = np.zeros_like(proba_raw)
            aligned_centers = np.zeros_like(model.means_)
            aligned_covs = np.zeros_like(model.covariances_)
            for old, new in mapping.items():
                aligned_proba[:, new] = proba_raw[:, old]
                aligned_centers[new] = model.means_[old]
                aligned_covs[new] = model.covariances_[old]
        else:
            mapping = align_clusters(
                prev_centers,
                prev_covs,
                model.means_,
                model.covariances_,
                metric=config.GMM_ALIGN_METRIC,
            )
            aligned_labels = remap_labels(raw_labels, mapping)
            aligned_proba = np.zeros_like(proba_raw)
            aligned_centers = np.zeros_like(model.means_)
            aligned_covs = np.zeros_like(model.covariances_)
            for curr, aligned in mapping.items():
                aligned_proba[:, aligned] = proba_raw[:, curr]
                aligned_centers[aligned] = model.means_[curr]
                aligned_covs[aligned] = model.covariances_[curr]

        prev_centers = aligned_centers
        prev_covs = aligned_covs
        model.means_ = aligned_centers  # keep aligned ordering
        model.covariances_ = aligned_covs

        labels_per_year[year] = pd.Series(aligned_labels, index=df_idx[mask])
        probs_per_year[year] = pd.DataFrame(
            aligned_proba,
            index=df_idx[mask],
            columns=[f"cluster_{i}_prob" for i in range(k)],
        )

        # 품질 지표 계산 (간단 버전)
        cluster_sizes = pd.Series(aligned_labels).value_counts(normalize=True)
        min_frac = float(cluster_sizes.min()) if not cluster_sizes.empty else 0.0
        quality_per_year[year] = {
            "status": "ok",
            "min_cluster_frac": min_frac,
            "samples": int(X_year.shape[0]),
        }

        if year == unique_years[-1]:
            latest_model = model

    return labels_per_year, probs_per_year, quality_per_year, latest_model, sort_feature
