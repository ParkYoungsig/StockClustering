"""GMM(Gaussian Mixture Model) 모델링 및 평가 유틸리티.

- BIC(Bayesian Information Criterion) 기반 K 후보 평가
- Silhouette 기반 분리도 평가(연도별 평균/중앙값 집계)
- 라벨 정렬(헝가리안 매칭/정렬) 및 robustness(ARI/NMI) 유틸
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

import config


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
        if X.shape[0] <= int(k):
            bic_scores.append(float("nan"))
            continue
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
    bic_matrix = np.array([per_year[y] for y in unique_years], dtype=float)
    mean_bic_arr = np.nanmean(bic_matrix, axis=0)
    median_bic_arr = np.nanmedian(bic_matrix, axis=0)
    mean_bic = mean_bic_arr.tolist()
    median_bic = median_bic_arr.tolist()

    best_k_mean = (
        k_values[int(np.nanargmin(mean_bic_arr))]
        if not np.all(np.isnan(mean_bic_arr))
        else k_values[0]
    )
    best_k_median = (
        k_values[int(np.nanargmin(median_bic_arr))]
        if not np.all(np.isnan(median_bic_arr))
        else k_values[0]
    )

    return per_year, mean_bic, median_bic, best_k_mean, best_k_median


def evaluate_silhouette_across_years(
    X: np.ndarray,
    years: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, float], Dict[int, float]]:
    """연도별 silhouette를 계산하고 K별로 평균/중앙값을 집계합니다.

    - 각 연도별로 GMM을 fit하고 hard label로 silhouette_score(Euclidean)을 계산
    - silhouette은 값이 클수록(최대 1에 가까울수록) 분리도가 좋다고 해석
    """

    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return {}, {}, {}

    k_values = list(k_values)
    unique_years = sorted(np.unique(years))
    per_year: Dict[int, Dict[int, float]] = {}

    for year in unique_years:
        mask = years == year
        Xy = X[mask]
        if Xy.shape[0] <= 2:
            continue
        per_k: Dict[int, float] = {}
        for k in k_values:
            if Xy.shape[0] <= k:
                continue
            gm = GaussianMixture(
                n_components=k,
                covariance_type=config.GMM_COVARIANCE_TYPE,
                random_state=random_state,
                n_init=config.GMM_N_INIT,
                max_iter=config.GMM_MAX_ITER,
                reg_covar=config.GMM_REG_COVAR,
                init_params="kmeans",
            )
            try:
                labels = gm.fit_predict(Xy)
                if len(np.unique(labels)) < 2:
                    continue
                per_k[k] = float(silhouette_score(Xy, labels, metric="euclidean"))
            except Exception:
                continue
        if per_k:
            per_year[int(year)] = per_k

    if not per_year:
        return {}, {}, {}

    mean_by_k: Dict[int, float] = {}
    median_by_k: Dict[int, float] = {}
    for k in k_values:
        vals = [v.get(k) for v in per_year.values() if k in v]
        vals_f = [float(x) for x in vals if x is not None]
        if not vals_f:
            continue
        mean_by_k[k] = float(np.mean(vals_f))
        median_by_k[k] = float(np.median(vals_f))

    return per_year, mean_by_k, median_by_k


def evaluate_window_robustness(
    X: np.ndarray,
    years: np.ndarray,
    *,
    k: int,
    window_years: List[int],
    eval_year: int | None = None,
    exclude_eval_year: bool = True,
    random_state: int = 42,
) -> Dict:
    """기간(윈도우)을 바꿔도 군집이 유지되는지(robustness) 평가합니다.

    - 여러 학습 윈도우(최근 N년)로 각각 GMM을 학습
    - 동일 평가셋(기본: 최신 연도)에서 예측 라벨을 얻고
    - baseline(ALL years) 대비 ARI/NMI를 계산

    ARI/NMI는 라벨 순열(permutation)에 불변이므로 별도 라벨 매칭이 필요 없습니다.
    """

    if X is None or len(X) == 0:
        return {"status": "skipped", "reason": "empty_X"}

    years_arr = np.asarray(years)
    uniq_raw = np.unique(years_arr)
    unique_years: List[int] = []
    for y in uniq_raw:
        try:
            if isinstance(y, float) and np.isnan(y):
                continue
            unique_years.append(int(y))
        except Exception:
            continue
    unique_years = sorted(set(unique_years))
    if not unique_years:
        return {"status": "skipped", "reason": "no_years"}

    eval_y = int(eval_year) if eval_year is not None else int(unique_years[-1])
    eval_mask = years_arr == eval_y
    n_eval = int(eval_mask.sum())
    if n_eval < 2:
        return {
            "status": "skipped",
            "reason": f"insufficient_eval_samples({n_eval})",
            "eval_year": eval_y,
            "k": int(k),
        }

    X_eval = X[eval_mask]

    def _fit_and_predict(train_mask: np.ndarray) -> np.ndarray | None:
        X_train = X[train_mask]
        if X_train.shape[0] <= int(k):
            return None
        model = GaussianMixture(
            n_components=int(k),
            covariance_type=config.GMM_COVARIANCE_TYPE,
            random_state=random_state,
            n_init=config.GMM_N_INIT,
            max_iter=config.GMM_MAX_ITER,
            reg_covar=config.GMM_REG_COVAR,
            init_params="kmeans",
        )
        model.fit(X_train)
        return model.predict(X_eval)

    # baseline (ALL train years)
    train_mask_all = np.ones_like(years_arr, dtype=bool)
    if exclude_eval_year:
        train_mask_all &= years_arr != eval_y
    baseline_labels = _fit_and_predict(train_mask_all)
    if baseline_labels is None:
        return {
            "status": "skipped",
            "reason": "baseline_insufficient_train_samples",
            "eval_year": eval_y,
            "k": int(k),
            "exclude_eval_year": bool(exclude_eval_year),
        }

    results: Dict[str, Dict] = {}
    label_bank: Dict[str, np.ndarray] = {"ALL": baseline_labels}

    max_train_year = eval_y - 1 if exclude_eval_year else eval_y
    min_train_year = int(unique_years[0])

    for w in window_years:
        try:
            w_int = int(w)
        except Exception:
            continue
        if w_int <= 0:
            continue

        train_end = int(max_train_year)
        train_start = max(min_train_year, train_end - w_int + 1)
        train_mask = (years_arr >= train_start) & (years_arr <= train_end)
        if exclude_eval_year:
            train_mask &= years_arr != eval_y

        pred = _fit_and_predict(train_mask)
        key = f"W{w_int}"
        if pred is None:
            results[key] = {
                "status": "skipped",
                "train_year_start": int(train_start),
                "train_year_end": int(train_end),
                "n_train": int(train_mask.sum()),
            }
            continue

        results[key] = {
            "status": "ok",
            "train_year_start": int(train_start),
            "train_year_end": int(train_end),
            "n_train": int(train_mask.sum()),
            "ari_vs_all": float(adjusted_rand_score(baseline_labels, pred)),
            "nmi_vs_all": float(
                normalized_mutual_info_score(
                    baseline_labels, pred, average_method="arithmetic"
                )
            ),
        }
        label_bank[key] = pred

    # pairwise matrix (heatmap용)
    ordered = ["ALL"] + [
        f"W{int(w)}" for w in window_years if f"W{int(w)}" in label_bank
    ]
    n = len(ordered)
    ari_mat = np.full((n, n), np.nan, dtype=float)
    nmi_mat = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            li = label_bank.get(ordered[i])
            lj = label_bank.get(ordered[j])
            if li is None or lj is None:
                continue
            ari_mat[i, j] = adjusted_rand_score(li, lj)
            nmi_mat[i, j] = normalized_mutual_info_score(
                li, lj, average_method="arithmetic"
            )

    return {
        "status": "ok",
        "eval_year": int(eval_y),
        "exclude_eval_year": bool(exclude_eval_year),
        "k": int(k),
        "n_eval": int(n_eval),
        "scores": results,
        "pairwise": {"labels": ordered, "ari": ari_mat, "nmi": nmi_mat},
    }


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 행렬(a: nxd, b: mxd) 간 코사인 유사도(nxm)."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D arrays")
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have same feature dimension")

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_safe = np.where(a_norm == 0, 1.0, a_norm)
    b_safe = np.where(b_norm == 0, 1.0, b_norm)
    a_unit = a / a_safe
    b_unit = b / b_safe
    return a_unit @ b_unit.T


def corr_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """두 행렬 간 피어슨 상관 유사도(nxm).

    - 각 행(클러스터 중심 벡터)을 평균 0으로 센터링 후 상관 계산
    - 0분산 벡터는 0으로 처리
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D arrays")
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must have same feature dimension")

    a0 = a - np.mean(a, axis=1, keepdims=True)
    b0 = b - np.mean(b, axis=1, keepdims=True)
    a_std = np.linalg.norm(a0, axis=1, keepdims=True)
    b_std = np.linalg.norm(b0, axis=1, keepdims=True)
    a_safe = np.where(a_std == 0, 1.0, a_std)
    b_safe = np.where(b_std == 0, 1.0, b_std)
    return (a0 / a_safe) @ (b0 / b_safe).T


def match_centroids_by_similarity(similarity: np.ndarray) -> Dict[int, int]:
    """유사도 행렬을 최대화하도록 헝가리안 매칭을 수행.

    Returns:
        mapping: row_index -> col_index
    """

    if similarity.ndim != 2:
        raise ValueError("similarity must be 2D")
    # cost = -similarity (maximize similarity)
    row_ind, col_ind = linear_sum_assignment(-similarity)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


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
    """이전/현재 클러스터 중심을 헝가리안 정렬로 매칭한다."""

    def _cov_mat(cov_entry: np.ndarray) -> np.ndarray:
        c = np.asarray(cov_entry)
        return np.diag(c) if c.ndim == 1 else c

    # Normalize center arrays to 2D (n_clusters x n_features)
    prev_centers = np.asarray(prev_centers)
    curr_centers = np.asarray(curr_centers)
    if prev_centers.ndim == 1:
        prev_centers = prev_centers.reshape(1, -1)
    if curr_centers.ndim == 1:
        curr_centers = curr_centers.reshape(1, -1)

    if metric == "bhattacharyya":
        k = int(curr_centers.shape[0])
        cost_matrix = np.zeros((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                cost_matrix[i, j] = float(
                    bhattacharyya_distance(
                        curr_centers[i],
                        _cov_mat(curr_covs[i]),
                        prev_centers[j],
                        _cov_mat(prev_covs[j]),
                    )
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
    """두 가우시안 분포 간 Bhattacharyya 거리(대각/풀 공분산 지원)."""
    # Coerce inputs to arrays and normalize shapes
    m1 = np.asarray(m1).ravel()
    m2 = np.asarray(m2).ravel()
    S1 = np.asarray(S1)
    S2 = np.asarray(S2)

    # Ensure covariance matrices are 2D
    S1 = np.diag(S1) if S1.ndim == 1 else S1
    S2 = np.diag(S2) if S2.ndim == 1 else S2

    # Ensure consistent dimensionality
    if m1.ndim != 1 or m2.ndim != 1:
        m1 = np.ravel(m1)
        m2 = np.ravel(m2)

    d = m1.shape[0]
    if m2.shape[0] != d:
        raise ValueError("m1 and m2 must have the same length")
    if S1.shape != (d, d):
        raise ValueError("S1 must be shape (d,d)")
    if S2.shape != (d, d):
        raise ValueError("S2 must be shape (d,d)")

    S = 0.5 * (S1 + S2)

    try:
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        inv_S = np.linalg.pinv(S)

    dm = (m2 - m1).reshape(-1, 1)
    quad = dm.T @ inv_S @ dm
    # quad may be a 1x1 array; squeeze to scalar safely
    quad_val = float(np.squeeze(quad))
    term1 = 0.125 * quad_val

    det_S = np.linalg.det(S)
    det_S1 = np.linalg.det(S1)
    det_S2 = np.linalg.det(S2)
    eps = 1e-12
    # guard against negative/zero det issues
    denom = np.sqrt((det_S1 + eps) * (det_S2 + eps))
    term2 = 0.5 * float(np.log((det_S + eps) / denom + eps))
    return float(term1 + term2)
