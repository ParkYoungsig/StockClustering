"""GMM 결과 후처리 유틸.

- 최신 연도 데이터에 클러스터 라벨 결합
- 소규모 클러스터를 노이즈로 처리하여 제거(옵션)
- 리포트 작성을 위한 클러스터별 종목 목록 생성
"""

from typing import Dict, List, Tuple

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_latest_year_frame(
    df_clean: pd.DataFrame, labels_per_year: Dict[int, pd.Series]
) -> Tuple[int, pd.DataFrame]:
    """전체 데이터에서 최신 연도 데이터를 추출하고 클러스터 라벨을 병합합니다."""
    target_year = int(df_clean["Year"].max())

    if not labels_per_year:
        raise ValueError("라벨 정보가 없습니다.")

    available_years = sorted(labels_per_year.keys())
    selected_year = target_year if target_year in labels_per_year else available_years[-1]

    if selected_year != target_year:
        logger.warning(
            "최신 연도(%s)에 라벨 없음 → 가장 최근 사용 가능 연도(%s)로 대체", target_year, selected_year
        )

    mask_year = df_clean["Year"] == selected_year
    df_latest = df_clean.loc[mask_year].copy()
    df_latest["cluster"] = labels_per_year[selected_year]
    return selected_year, df_latest


def filter_noise(
    df_latest: pd.DataFrame, min_cluster_frac: float
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    너무 작은 규모의 군집을 노이즈(-1)로 처리하여 제거합니다.

    Args:
        df_latest: 최신 연도 데이터프레임
        min_cluster_frac: 최소 군집 비율 (예: 0.01 = 1%)

    Returns:
        Tuple: (필터링된 DF, 군집 크기 정보, 노이즈 통계 요약)
    """
    size_threshold = max(1, int(len(df_latest) * min_cluster_frac))
    cluster_sizes_raw = df_latest["cluster"].value_counts()

    # 임계값 미만 군집 식별
    small_clusters = cluster_sizes_raw[
        cluster_sizes_raw < size_threshold
    ].index.tolist()

    df_flagged = df_latest.copy()
    if small_clusters:
        df_flagged.loc[df_flagged["cluster"].isin(small_clusters), "cluster"] = -1

    cluster_sizes = df_flagged["cluster"].value_counts().sort_index()
    noise_count = int(cluster_sizes.get(-1, 0))
    df_valid = df_flagged[df_flagged["cluster"] != -1]

    summary = {
        "size_threshold": size_threshold,
        "noise_rows": noise_count,
        "removed_clusters": small_clusters,
    }
    return df_valid, cluster_sizes, summary


def compute_cluster_stats(
    df_valid: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """클러스터별 요약 통계(평균/표준편차/개수)를 계산합니다."""

    means = df_valid.groupby("cluster")[feature_cols].mean()
    stds = df_valid.groupby("cluster")[feature_cols].std(ddof=0)
    counts = df_valid["cluster"].value_counts().sort_index()
    return means, stds, counts
