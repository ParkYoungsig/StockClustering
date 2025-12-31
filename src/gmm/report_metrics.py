"""리포트용 지표 계산 모듈.

- GMM 품질(소속확률 기반)
- Silhouette (거리 기반 분리도)
- 사후 성과(Forward Return/Drawdown)

주의: 이 모듈은 '계산'만 담당하고, 출력 포맷/파일 저장은 reporter가 담당합니다.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def compute_quality_from_probabilities(
    df_valid: pd.DataFrame,
    probs_latest: pd.DataFrame | None,
    low_conf_threshold: float = 0.6,
) -> Optional[Dict]:
    if probs_latest is None or probs_latest.empty or df_valid is None or df_valid.empty:
        return None

    probs_latest = probs_latest.reindex(df_valid.index)
    probs_latest = probs_latest.dropna(how="all")
    if probs_latest.empty:
        return None

    aligned_idx = probs_latest.index.intersection(df_valid.index)
    if aligned_idx.empty:
        return None

    probs_latest = probs_latest.loc[aligned_idx]
    hard = df_valid.loc[aligned_idx, "cluster"].astype(int).to_numpy()

    max_resp = probs_latest.max(axis=1)
    soft_argmax = probs_latest.to_numpy().argmax(axis=1)

    by_cluster = max_resp.groupby(df_valid.loc[aligned_idx, "cluster"]).mean().to_dict()

    return {
        "overall": {
            "mean_max_resp": float(max_resp.mean()),
            "low_conf_threshold": float(low_conf_threshold),
            "low_conf_ratio": float((max_resp < low_conf_threshold).mean()),
            "hard_soft_mismatch_ratio": float((soft_argmax != hard).mean()),
        },
        "by_cluster": by_cluster,
    }


def compute_silhouette_summary(
    df_clean: pd.DataFrame,
    df_valid: pd.DataFrame,
    X_feats: np.ndarray,
) -> Optional[Dict]:
    if df_clean is None or df_clean.empty or df_valid is None or df_valid.empty:
        return None

    try:
        from sklearn.metrics import silhouette_samples, silhouette_score
    except Exception:
        return None

    pos = df_clean.index.get_indexer(df_valid.index)
    pos = pos[pos >= 0]
    if len(pos) <= 1:
        return None

    y_valid = df_valid["cluster"].astype(int).to_numpy()
    if len(np.unique(y_valid)) < 2:
        return None

    X_valid = X_feats[pos]

    try:
        overall = float(silhouette_score(X_valid, y_valid, metric="euclidean"))
        s_samples = silhouette_samples(X_valid, y_valid, metric="euclidean")
        by_cluster = pd.Series(s_samples).groupby(pd.Series(y_valid)).mean().to_dict()
        return {"overall": overall, "by_cluster": by_cluster}
    except Exception:
        return None


# note: transition/transition_summary 계산은 더 이상 사용하지 않음.
# 대신 필요했던 ex-post(Forward return/drawdown) 계산은 compute_report_metrics
# 내부에서 필요에 따라 수행합니다.


def compute_report_metrics(
    *,
    df_clean: pd.DataFrame,
    df_valid: pd.DataFrame,
    X_feats: np.ndarray,
    probs_map: Dict[int, pd.DataFrame] | None,
    target_year: int,
    low_conf_threshold: float = 0.6,
    horizon_return: int = 20,
) -> Dict:
    """리포트에 필요한 추가 지표들을 한 번에 계산해서 반환합니다."""

    probs_latest = probs_map.get(target_year) if probs_map else None

    quality_summary = compute_quality_from_probabilities(
        df_valid=df_valid,
        probs_latest=probs_latest,
        low_conf_threshold=low_conf_threshold,
    )

    silhouette_summary = compute_silhouette_summary(
        df_clean=df_clean,
        df_valid=df_valid,
        X_feats=X_feats,
    )

    # ex-post 계산: Forward return / Drawdown (horizon_return days)
    ex_post_summary = None
    try:
        needed = {"Date", "Ticker", "Close", "cluster"}
        if (
            df_valid is not None
            and not df_valid.empty
            and needed.issubset(set(df_valid.columns))
        ):
            df_t = df_valid[["Date", "Ticker", "Close", "cluster"]].copy()
            df_t["Date"] = pd.to_datetime(df_t["Date"], errors="coerce")
            close_raw = df_t.loc[:, "Close"]
            if isinstance(close_raw, pd.DataFrame):
                close_raw = close_raw.iloc[:, 0]
            df_t["__close"] = pd.to_numeric(close_raw, errors="coerce")
            df_t = df_t.dropna(subset=["Date", "Ticker", "__close"]).sort_values(
                ["Ticker", "Date"]
            )

            h = int(horizon_return)
            close = df_t.groupby("Ticker")["__close"]
            fwd_ret = close.shift(-h) / df_t["__close"] - 1.0

            def _future_min_excl_today(s: pd.Series) -> pd.Series:
                s2 = s.shift(-1)
                return s2[::-1].rolling(window=h, min_periods=1).min()[::-1]

            fut_min = close.transform(_future_min_excl_today)
            fwd_dd = fut_min / df_t["__close"] - 1.0
            df_t["fwd_return_20d"] = fwd_ret
            df_t["fwd_drawdown_20d"] = fwd_dd

            ex_post_summary = (
                df_t.dropna(subset=["fwd_return_20d"])
                .groupby("cluster")
                .agg(
                    fwd_return_20d_mean=("fwd_return_20d", "mean"),
                    fwd_return_20d_hit_ratio=(
                        "fwd_return_20d",
                        lambda x: float((x > 0).mean()),
                    ),
                    fwd_drawdown_20d_mean=("fwd_drawdown_20d", "mean"),
                )
            )
    except Exception:
        ex_post_summary = None

    return {
        "quality_summary": quality_summary,
        "silhouette_summary": silhouette_summary,
        "ex_post_summary": ex_post_summary,
    }
