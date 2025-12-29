"""
데이터 전처리(Preprocessing) 모듈
- Feature Scaling (QuantileTransformer)
- PCA (Principal Component Analysis) 차원 축소
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer


def _choose_pca_components(
    explained_variance_ratio: np.ndarray,
    min_components: int = 2,
    max_components: int = 3,
    variance_threshold: float = 0.90,
) -> int:
    """설명된 분산 비율(Explained Variance)을 기준으로 적절한 PCA 차원 수를 결정합니다."""
    cum_var = np.cumsum(explained_variance_ratio)
    needed = int(np.searchsorted(cum_var, variance_threshold) + 1)
    return int(max(min_components, min(max_components, needed)))


def preprocess_features(
    raw: pd.DataFrame,
    feature_columns: List[str],
    pca_min_components: int = 2,
    pca_max_components: int = 3,
    variance_threshold: float = 0.90,
    use_pca: bool = True,
    apply_iforest: bool = True,
    iforest_contamination: float = 0.05,
    low_variance_threshold: float = 1e-4,
    corr_threshold: float = 0.99,
    group_labels: np.ndarray | None = None,
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    QuantileTransformer,
    PCA | None,
    Dict,
    List[str],
]:
    """
    Feature 데이터를 정제하고 스케일링 및 차원 축소를 수행합니다.

    [처리 단계]
    1. 필수 파생변수(NATR) 확인/계산
    2. 이상치 필터링(NATR > 20, Disparity_60d > 30)
    3. 결측치(NaN/Inf) 제거 및 데이터 타입 변환
    4. Quantile Scaling (정규분포화)
    5. (옵션) PCA 차원 축소 수행

    Returns:
        Tuple: (정제된 DF, 변환된 Numpy 배열, PC 데이터프레임, 스케일러(또는 그룹별 스케일러 dict), PCA 모델, 통계 정보, 사용된 Feature 목록)
    """
    df = raw.copy()
    removed_low_var: List[str] = []
    removed_high_corr: List[str] = []
    group_series = None
    if group_labels is not None:
        if len(group_labels) != len(df):
            raise ValueError("group_labels 길이가 데이터와 다릅니다.")
        group_series = pd.Series(group_labels, index=df.index)

    # 파생변수 NATR 재확인 및 계산
    if "ATR_14" in df.columns and "Close" in df.columns:
        atr = pd.to_numeric(df["ATR_14"], errors="coerce")
        close = pd.to_numeric(df["Close"], errors="coerce")
        close = close.mask(close <= 0)
        for natr_col in ("NATR_14", "NATR"):
            if natr_col in feature_columns and natr_col not in df.columns:
                df[natr_col] = (atr / close) * 100

    # 숫자형 변환 선행 (이상치 필터 전에 수행)
    for col in feature_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 이상치 필터링: 비정상 변동성/이격도 데이터 제거
    outlier_mask = pd.Series(False, index=df.index)
    if "NATR" in df.columns:
        outlier_mask = outlier_mask | (df["NATR"] > 20)
    if "Disparity_60d" in df.columns:
        outlier_mask = outlier_mask | (df["Disparity_60d"] > 30)
    if outlier_mask.any():
        df = df.loc[~outlier_mask].copy()
        if group_series is not None:
            group_series = group_series.loc[df.index]

    rows_after_outlier = len(df)

    # 데이터 정제 (결측치 제거)
    df = df.dropna(subset=feature_columns).copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns)
    if group_series is not None:
        group_series = group_series.loc[df.index]

    # IsolationForest로 이상치 제거 (윈도우별 적용 권장)
    if apply_iforest and len(df) >= max(20, len(feature_columns) * 10):
        X_raw = df[feature_columns].to_numpy(dtype=float)
        iforest = IsolationForest(
            n_estimators=200,
            contamination=iforest_contamination,
            random_state=42,
        )
        preds = iforest.fit_predict(X_raw)
        df = df.loc[preds == 1].copy()
        if group_series is not None:
            group_series = group_series.loc[df.index]

    stats = {
        "rows_before": len(raw),
        "rows_after_outlier_filter": rows_after_outlier,
        "rows_after_dropna": len(df),
        "rows_after_iforest": len(df),
        "low_variance_threshold": float(low_variance_threshold),
        "corr_threshold": float(corr_threshold),
        "removed_low_variance": removed_low_var,
        "removed_high_corr": removed_high_corr,
    }
    if df.empty:
        return (
            df,
            np.empty((0, 0)),
            pd.DataFrame(),
            QuantileTransformer(),
            PCA(),
            stats,
            [],
        )

    # Low-variance feature 제거
    variances = df[feature_columns].var(axis=0)
    low_var_cols = variances[variances < low_variance_threshold].index.tolist()
    if low_var_cols:
        df = df.drop(columns=low_var_cols)
        removed_low_var.extend(low_var_cols)

    # 상관 필터링 (스피어만, 절대값 기준)
    feature_cols_available = [c for c in feature_columns if c in df.columns]
    if len(feature_cols_available) > 1:
        corr_matrix = df[feature_cols_available].corr(method="spearman").abs()
        to_drop = set()
        for i, col in enumerate(feature_cols_available):
            if col in to_drop:
                continue
            high_corr = corr_matrix.columns[
                (corr_matrix.iloc[i, :] >= corr_threshold)
                & (corr_matrix.columns != col)
            ]
            to_drop.update(high_corr)
        if to_drop:
            df = df.drop(columns=list(to_drop))
            removed_high_corr.extend(sorted(to_drop))

    feature_cols_used = [c for c in feature_columns if c in df.columns]
    if not feature_cols_used:
        stats.update({"feature_cols_used": feature_cols_used})
        return (
            df,
            np.empty((0, 0)),
            pd.DataFrame(),
            QuantileTransformer(),
            PCA(),
            stats,
            feature_cols_used,
        )

    # Numpy 배열 변환 및 스케일링
    X = df[feature_cols_used].to_numpy(dtype=float)
    if group_series is not None:
        scaler_map: Dict = {}
        X_scaled = np.zeros_like(X)
        for g in group_series.unique():
            idx = group_series == g
            scaler_g = QuantileTransformer(
                output_distribution="normal",
                random_state=42,
                n_quantiles=min(1000, idx.sum()),
            )
            X_scaled[idx] = scaler_g.fit_transform(X[idx])
            scaler_map[int(g)] = scaler_g
        scaler = scaler_map
        stats["scaling_mode"] = "per_group"
    else:
        scaler = QuantileTransformer(
            output_distribution="normal",
            random_state=42,
            n_quantiles=min(1000, len(df)),
        )
        X_scaled = scaler.fit_transform(X)
        stats["scaling_mode"] = "global"

    # PCA 미사용 시
    if not use_pca:
        X_out = X_scaled
        pcs_df = pd.DataFrame(X_out, columns=feature_cols_used, index=df.index)
        stats.update(
            {
                "pca_components": 0,
                "explained_variance_ratio": np.array([]),
                "used_pca": False,
                "feature_cols_used": feature_cols_used,
            }
        )
        return df, X_out, pcs_df, scaler, None, stats, feature_cols_used

    # PCA 수행
    pca_full = PCA(
        n_components=min(len(feature_cols_used), X_scaled.shape[0]), random_state=42
    )
    pca_full.fit(X_scaled)
    n_comp = _choose_pca_components(
        pca_full.explained_variance_ratio_,
        min_components=pca_min_components,
        max_components=pca_max_components,
        variance_threshold=variance_threshold,
    )
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pcs_cols = [f"PC{i+1}" for i in range(n_comp)]
    pcs_df = pd.DataFrame(X_pca, columns=pcs_cols, index=df.index)

    stats.update(
        {
            "pca_components": n_comp,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "used_pca": True,
            "feature_cols_used": feature_cols_used,
        }
    )
    return df, X_pca, pcs_df, scaler, pca, stats, feature_cols_used
