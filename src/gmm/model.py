"""
GMM(Gaussian Mixture Model) 모델링 및 평가 유틸리티
- BIC(Bayesian Information Criterion) 기반 최적 K 탐색
- 군집 안정성(Stability) 평가
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture

from src.gmm import config


def _fit_gmm_labels(
    X: np.ndarray, k: int, random_state: int = 42, means_init: np.ndarray | None = None
) -> Tuple[GaussianMixture, np.ndarray]:
    """단일 GMM 모델을 학습하고 예측 라벨을 반환합니다."""
    model = GaussianMixture(
        n_components=k,
        covariance_type=config.GMM_COVARIANCE_TYPE,
        random_state=random_state,
        n_init=config.GMM_N_INIT,
        max_iter=config.GMM_MAX_ITER,
        reg_covar=config.GMM_REG_COVAR,
        means_init=means_init,
        init_params="kmeans",
    )
    model.fit(X)
    return model, model.predict(X)


def sweep_bic_gmm(
    X: np.ndarray,
    k_values: Iterable[int] = range(2, 8),
    random_state: int = 42,
) -> List[float]:
    """주어진 K 후보군에 대해 GMM을 학습하고 BIC 점수 리스트를 반환합니다."""
    bic_scores: List[float] = []
    for k in k_values:
        gm = GaussianMixture(
            n_components=k,
            covariance_type=config.GMM_COVARIANCE_TYPE,
            random_state=random_state,
            n_init=config.GMM_N_INIT,
            max_iter=config.GMM_MAX_ITER,
            reg_covar=config.GMM_REG_COVAR,
            init_params="kmeans",
        )
        gm.fit(X)
        bic_scores.append(gm.bic(X))
    return bic_scores


def evaluate_bic_across_years(
    X: np.ndarray,
    years: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
) -> Tuple[Dict[int, List[float]], List[float], List[float], int, int]:
    """
    연도별로 BIC를 계산하고 집계(Mean/Median)하여 최적의 K를 제안합니다.

    Returns:
        Tuple: (연도별 BIC, 평균 BIC, 중앙값 BIC, Mean 기준 Best K, Median 기준 Best K)
    """
    k_values = list(k_values)
    unique_years = sorted(np.unique(years))
    per_year: Dict[int, List[float]] = {}

    # 연도별 BIC 계산
    for year in unique_years:
        mask = years == year
        per_year[year] = sweep_bic_gmm(
            X[mask], k_values=k_values, random_state=random_state
        )

    # 전체 통계 집계
    bic_matrix = np.array([per_year[y] for y in unique_years])
    mean_bic = bic_matrix.mean(axis=0).tolist()
    median_bic = np.median(bic_matrix, axis=0).tolist()

    best_k_mean = k_values[int(np.argmin(mean_bic))]
    best_k_median = k_values[int(np.argmin(median_bic))]

    return per_year, mean_bic, median_bic, best_k_mean, best_k_median


def compute_stability_scores(
    X: np.ndarray,
    years: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
) -> Tuple[Dict[int, float], int]:
    """
    연도 간 군집 중심(Center)의 변화를 추적하여 안정성(Stability)을 평가합니다.

    [알고리즘]
    - 연속된 연도 간 클러스터 중심점의 거리를 측정하여 매칭합니다.
    - 변화가 적을수록 높은 점수를 부여합니다.
    - Elbow Method를 통해 안정성이 급격히 떨어지는 구간을 탐색합니다.

    Returns:
        Tuple: (K별 안정성 점수, Elbow 포인트 K)
    """
    k_values = list(k_values)
    unique_years = sorted(np.unique(years))
    stability: Dict[int, float] = {}

    for k in k_values:
        centers_per_year: Dict[int, np.ndarray] = {}
        # 연도별 클러스터 중심점 계산
        for year in unique_years:
            mask = years == year
            model, _ = _fit_gmm_labels(X[mask], k=k, random_state=random_state)
            centers_per_year[year] = model.means_

        # 연도 간 거리 측정 (Greedy Matching)
        pairwise_distances: List[float] = []
        for i in range(len(unique_years) - 1):
            a, b = (
                centers_per_year[unique_years[i]],
                centers_per_year[unique_years[i + 1]],
            )
            dist_matrix = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            used_a = set()
            used_b = set()

            for _ in range(min(len(a), len(b))):
                remaining = [
                    (dist_matrix[i, j], i, j)
                    for i in range(len(a))
                    for j in range(len(b))
                    if i not in used_a and j not in used_b
                ]
                if not remaining:
                    break
                best_dist, ia, jb = min(remaining, key=lambda t: t[0])
                used_a.add(ia)
                used_b.add(jb)
                pairwise_distances.append(float(best_dist))

        # 안정성 점수 계산 (거리의 역수)
        stability[k] = (
            float(1.0 / (1.0 + np.mean(pairwise_distances)))
            if pairwise_distances
            else 0.0
        )

    # Elbow 포인트 탐색
    stability_values = [stability[k] for k in k_values]
    diffs = np.diff(stability_values)
    elbow_idx = int(np.argmin(diffs) + 1) if len(diffs) else 0
    elbow_k = k_values[elbow_idx]

    return stability, elbow_k


def align_labels_by_feature(
    labels: np.ndarray,
    df_features: np.ndarray,
    feature_index: int,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    특정 Feature(예: 수익률)의 평균을 기준으로 클러스터 번호를 정렬합니다.
    (예: 수익률이 낮은 순서대로 0, 1, 2... 할당)

    Returns:
        Tuple: (재할당된 라벨, 매핑 딕셔너리)
    """
    unique_labels = np.unique(labels)
    means = {}
    for lbl in unique_labels:
        mask = labels == lbl
        if mask.sum() == 0:
            means[lbl] = np.inf
        else:
            means[lbl] = float(np.mean(df_features[mask, feature_index]))

    sorted_labels = sorted(unique_labels, key=lambda l: means.get(l, np.inf))
    mapping = {old: new for new, old in enumerate(sorted_labels)}
    remapped = np.vectorize(mapping.get)(labels)

    return remapped, mapping


def align_clusters(
    prev_centers: np.ndarray,
    prev_covs: np.ndarray,
    curr_centers: np.ndarray,
    curr_covs: np.ndarray,
    metric: str = "bhattacharyya",
) -> Dict[int, int]:
    """이전/현재 중심을 거리 최소화 방식(Hungarian)으로 정렬."""

    def _cov_mat(cov_entry: np.ndarray) -> np.ndarray:
        return np.diag(cov_entry) if cov_entry.ndim == 1 else cov_entry

    if metric == "bhattacharyya":
        k = curr_centers.shape[0]
        cost_matrix = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                cost_matrix[i, j] = bhattacharyya_distance(
                    curr_centers[i],
                    _cov_mat(curr_covs[i]),
                    prev_centers[j],
                    _cov_mat(prev_covs[j]),
                )
    else:
        cost_matrix = cdist(curr_centers, prev_centers, metric=metric)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def remap_labels(labels: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """라벨을 주어진 매핑으로 재할당."""
    return np.vectorize(mapping.get)(labels)


def bhattacharyya_distance(
    m1: np.ndarray, S1: np.ndarray, m2: np.ndarray, S2: np.ndarray
) -> float:
    """Bhattacharyya distance between two Gaussians (supports diag/full)."""

    # Ensure 2D cov
    S1 = np.diag(S1) if S1.ndim == 1 else S1
    S2 = np.diag(S2) if S2.ndim == 1 else S2
    S = 0.5 * (S1 + S2)

    try:
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        inv_S = np.linalg.pinv(S)

    dm = (m2 - m1).reshape(-1, 1)
    term1 = 0.125 * float(dm.T @ inv_S @ dm)

    det_S = np.linalg.det(S)
    det_S1 = np.linalg.det(S1)
    det_S2 = np.linalg.det(S2)
    eps = 1e-12
    term2 = 0.5 * np.log((det_S + eps) / np.sqrt((det_S1 + eps) * (det_S2 + eps)) + eps)
    return float(term1 + term2)
