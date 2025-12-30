"""GMM 멀티 스냅샷(연도/시점별) 파이프라인 텍스트 리포트 유틸.

클러스터링 결과를 사람이 읽기 쉬운 형태로 요약하여 텍스트 파일로 저장합니다.
주요 내용: 데이터 로딩/전처리 요약, PCA 사용 여부, BIC 기반 K 선택, 안정성 지표,
클러스터별 평균/개수/구성 종목 및 (설정 기반) 해석.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def _cluster_label(cid: int, cluster_names: Optional[Dict[int, str]]) -> str:
    try:
        key = int(cid)
    except Exception:
        key = cid
    return (
        cluster_names.get(key, f"Cluster {cid}") if cluster_names else f"Cluster {cid}"
    )


def _format_time_tag(row: pd.Series) -> str | None:
    """연말이면 연도만, 월말 스냅샷이면 연-월(YearMonth 기준)로 포맷."""

    ym = row.get("YearMonth")
    if pd.notna(ym):
        return str(ym)

    year = row.get("Year")
    if pd.notna(year):
        try:
            return str(int(year))
        except Exception:
            return str(year)
    return None


def write_text_report(
    output_path: Path,
    load_stats: Dict,
    prep_stats: Dict,
    bic_scores: List[float],
    k_values: Iterable[int],
    best_k: int,
    pca_explained: np.ndarray,
    cluster_means: pd.DataFrame,
    cluster_stds: pd.DataFrame | None,
    cluster_counts: pd.Series,
    cluster_members: Dict[int, List[str]],
    noise_summary: Dict | None = None,
    stability_summary: Dict | None = None,
    quality_summary: Dict | None = None,
    silhouette_summary: Dict | None = None,
    transition_summary: Dict | None = None,
    ex_post_summary: pd.DataFrame | None = None,
    best_k_mean: int | None = None,
    best_k_median: int | None = None,
    label_alignment_feature: str | None = None,
    cluster_names: Optional[Dict[int, str]] = None,
    cluster_interpretations: Optional[Dict[int, str]] = None,
) -> None:
    """GMM 파이프라인 결과를 텍스트 리포트로 저장합니다.

    - 데이터/전처리 요약
    - (선택) PCA 설명력
    - BIC 기반 K 선택 결과
    - (선택) 안정성(연도 간 유지) 요약
    - 클러스터 평균/개수/구성 종목 및 규칙 기반 해석
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("GMM 스냅샷 리포트\n")
        f.write("================\n\n")
        f.write("데이터 요약:\n")
        f.write(f"  읽은 파일 수: {load_stats.get('total_files')}\n")
        f.write(f"  사용 스냅샷 수: {load_stats.get('snapshots')}\n")
        if "start_year" in load_stats:
            f.write(
                f"  분석 연도 범위: {load_stats.get('start_year')} - {load_stats.get('end_year')}\n"
            )
            f.write(f"  폴백 윈도우(일): {load_stats.get('fallback_days')}\n")
        else:
            f.write(
                f"  기준일: {load_stats.get('reference_date')} (폴백: {load_stats.get('fallback_date')})\n"
            )
        f.write(f"  결측 제거 후 최종 행 수: {prep_stats.get('rows_after_dropna')}\n\n")

        # 피처 중복 제거/정리 요약 (상관/저분산)
        feature_candidates_total = prep_stats.get("feature_candidates_total")
        feature_candidates_base = prep_stats.get("feature_candidates_base") or []
        feature_candidates_extra = (
            prep_stats.get("feature_candidates_extra_numeric") or []
        )
        feature_cols_input_count = prep_stats.get("feature_cols_input_count")
        feature_cols_used = prep_stats.get("feature_cols_used") or []
        removed_low_var = prep_stats.get("removed_low_variance") or []
        removed_high_corr = prep_stats.get("removed_high_corr") or []
        dropped_high_missing = prep_stats.get("dropped_high_missing") or []
        max_missing_ratio = prep_stats.get("max_missing_ratio")
        low_var_thr = prep_stats.get("low_variance_threshold")
        corr_thr = prep_stats.get("corr_threshold")
        if (
            feature_candidates_total is not None
            or feature_cols_input_count is not None
            or feature_cols_used
            or removed_low_var
            or removed_high_corr
            or dropped_high_missing
        ):
            f.write("피처 정리(자동 중복 제거):\n")
            if feature_candidates_total is not None:
                f.write(f"  후보 전체: {int(feature_candidates_total)}\n")
                f.write(f"    - 기본 FEATURE_COLUMNS: {len(feature_candidates_base)}\n")
                f.write(f"    - 추가 numeric 후보: {len(feature_candidates_extra)}\n")
            if feature_cols_input_count is not None:
                f.write(
                    f"  결측률 필터 통과(전처리 입력): {int(feature_cols_input_count)}\n"
                )
            if feature_cols_used:
                f.write(f"  최종 사용 피처 수: {len(feature_cols_used)}\n")
            if max_missing_ratio is not None:
                f.write(f"  결측률 필터 임계값(> 제외): {max_missing_ratio}\n")
            if low_var_thr is not None:
                f.write(f"  저분산 제거 임계값: {low_var_thr}\n")
            if corr_thr is not None:
                f.write(f"  상관(스피어만) 제거 임계값(|ρ|≥): {corr_thr}\n")
            if dropped_high_missing:
                f.write(
                    f"  제거(결측률, {len(dropped_high_missing)}): {', '.join(map(str, dropped_high_missing))}\n"
                )
            if removed_low_var:
                f.write(
                    f"  제거(저분산, {len(removed_low_var)}): {', '.join(map(str, removed_low_var))}\n"
                )
            if removed_high_corr:
                f.write(
                    f"  제거(고상관 중복, {len(removed_high_corr)}): {', '.join(map(str, removed_high_corr))}\n"
                )
            f.write("\n")

        if noise_summary:
            f.write("노이즈 처리(최소 클러스터 크기 기준):\n")
            f.write(f"  크기 임계값: {noise_summary.get('size_threshold')}\n")
            f.write(f"  노이즈로 재할당된 행 수: {noise_summary.get('noise_rows')}\n")
            f.write(f"  제거된 클러스터: {noise_summary.get('removed_clusters')}\n\n")

        if pca_explained is not None and len(pca_explained) > 0:
            f.write("PCA 요약:\n")
            f.write(
                "  설명분산비(components): "
                + ", ".join(f"{v:.4f}" for v in pca_explained)
                + "\n"
            )
            f.write(f"  사용 컴포넌트 수: {len(pca_explained)}\n\n")
        else:
            f.write("PCA: 생략(표준화된 원본 팩터 사용)\n\n")

        f.write("GMM (BIC 기반 K 선택, K=2..7):\n")
        f.write(
            f"  최적 K(평균 BIC): {best_k_mean if best_k_mean is not None else best_k}\n"
        )
        if best_k_median is not None:
            f.write(f"  최적 K(중앙값 BIC): {best_k_median}\n")
        if label_alignment_feature:
            f.write(f"  클러스터 라벨 정렬 기준 피처: {label_alignment_feature}\n")
        for k, bic in zip(k_values, bic_scores):
            f.write(f"  K={k}: BIC={bic:.2f}\n")
        f.write("\n")

        if stability_summary:
            f.write("안정성(연도 간 클러스터 유지):\n")
            f.write(f"  K별 평균 안정성: {stability_summary.get('mean_by_k')}\n")
            f.write(
                f"  엘보 포인트(안정성 급락 지점): {stability_summary.get('elbow_k')}\n\n"
            )

        f.write("클러스터별 피처 평균:\n")
        means_disp = cluster_means.copy()
        means_disp.index = [
            _cluster_label(cid, cluster_names) for cid in means_disp.index
        ]
        f.write(means_disp.to_string())
        f.write("\n")

        if cluster_stds is not None and not cluster_stds.empty:
            f.write("\n클러스터별 피처 표준편차:\n")
            stds_disp = cluster_stds.copy()
            stds_disp.index = [
                _cluster_label(cid, cluster_names) for cid in stds_disp.index
            ]
            f.write(stds_disp.to_string())
            f.write("\n")

        f.write("\n클러스터별 개수(최신 연도):\n")
        for cid, count in cluster_counts.items():
            f.write(f"  {_cluster_label(cid, cluster_names)}: {count}개\n")

        if quality_summary:
            f.write("\n군집 품질(Separation & Confidence):\n")
            overall = quality_summary.get("overall") or {}
            if overall:
                if overall.get("mean_max_resp") is not None:
                    f.write(
                        f"  평균 소속확률(max responsibility): {overall.get('mean_max_resp'):.4f}\n"
                    )
                if overall.get("low_conf_ratio") is not None:
                    thr = overall.get("low_conf_threshold", 0.6)
                    f.write(
                        f"  저신뢰 비율(max prob < {thr}): {overall.get('low_conf_ratio'):.4%}\n"
                    )
                if overall.get("hard_soft_mismatch_ratio") is not None:
                    f.write(
                        f"  Hard vs Soft 불일치 비율: {overall.get('hard_soft_mismatch_ratio'):.4%}\n"
                    )

            by_cluster = quality_summary.get("by_cluster")
            if isinstance(by_cluster, dict) and by_cluster:
                f.write("  클러스터별 평균 소속확률:\n")
                for cid, v in by_cluster.items():
                    try:
                        v_f = float(v)
                    except Exception:
                        continue
                    f.write(f"    - {_cluster_label(cid, cluster_names)}: {v_f:.4f}\n")

        if silhouette_summary:
            f.write("\nSilhouette Score (Euclidean, standardized):\n")
            if silhouette_summary.get("overall") is not None:
                f.write(f"  전체 평균: {silhouette_summary.get('overall'):.4f}\n")
            per_c = silhouette_summary.get("by_cluster")
            if isinstance(per_c, dict) and per_c:
                f.write("  클러스터별 평균:\n")
                for cid, v in per_c.items():
                    try:
                        v_f = float(v)
                    except Exception:
                        continue
                    f.write(f"    - {_cluster_label(cid, cluster_names)}: {v_f:.4f}\n")

        if transition_summary:
            f.write("\n시간적 전이/유지(Transition & Persistence):\n")
            for horizon_key in ["horizon_1", "horizon_20"]:
                block = transition_summary.get(horizon_key)
                if not isinstance(block, dict):
                    continue
                h = block.get("h")
                if h is None:
                    continue
                f.write(f"  Horizon={h}:\n")
                persist = block.get("persistence_by_cluster")
                if isinstance(persist, dict) and persist:
                    for cid, p in persist.items():
                        try:
                            p_f = float(p)
                        except Exception:
                            continue
                        f.write(
                            f"    - {_cluster_label(cid, cluster_names)} → 동일 상태: {p_f:.4%}\n"
                        )

        if ex_post_summary is not None and not ex_post_summary.empty:
            f.write("\n사후 성과 검증(ex-post, forward):\n")
            disp = ex_post_summary.copy()
            disp.index = [_cluster_label(int(cid), cluster_names) for cid in disp.index]
            f.write(disp.to_string())
            f.write("\n")

        if load_stats:
            f.write("\n데이터 커버리지(티커):\n")
            loaded = load_stats.get("tickers_loaded") or []
            f.write(
                f"  로드된 티커({len(loaded)}): {', '.join(loaded) if loaded else '없음'}\n"
            )
            dropped_date = load_stats.get("dropped_no_date") or []
            dropped_empty = load_stats.get("dropped_no_valid") or []
            dropped_feat = load_stats.get("dropped_missing_features") or []
            f.write(
                f"  제외(날짜 없음): {', '.join(dropped_date) if dropped_date else '없음'}\n"
            )
            f.write(
                f"  제외(기간 내 데이터 없음): {', '.join(dropped_empty) if dropped_empty else '없음'}\n"
            )
            f.write(
                f"  제외(필수 피처 누락): {', '.join(dropped_feat) if dropped_feat else '없음'}\n"
            )

        if cluster_interpretations:
            f.write("\n클러스터 해석(매핑 기반):\n")
            for cid in cluster_means.index:
                try:
                    key = int(cid)
                except Exception:
                    key = cid
                interp = cluster_interpretations.get(key)
                if interp:
                    f.write(f"  {_cluster_label(cid, cluster_names)}: {interp}\n")

        f.write("\n클러스터(상태)에 가장 자주 진입한 종목 TOP (최신 연도):\n")
        for cid, members in cluster_members.items():
            f.write(f"  {_cluster_label(cid, cluster_names)} ({len(members)}): ")
            f.write(", ".join(members))
            f.write("\n")


def build_cluster_members_all_years(
    df_clean: pd.DataFrame, labels_per_year: Dict[int, pd.Series]
) -> Dict[int, List[str]]:
    """전 기간(연말/월말) 클러스터 종목 명단을 연/월 태그와 함께 반환."""

    if df_clean is None or df_clean.empty or not labels_per_year:
        return {}

    def _format_name(row: pd.Series) -> str:
        ticker = row.get("Ticker")
        code = row.get("Code")
        name = str(row.get("Name", "")).strip()
        base = ticker if pd.notna(ticker) else code if pd.notna(code) else "N/A"
        time_tag = _format_time_tag(row)

        parts = []
        if name and name.lower() != "nan":
            parts.append(name)
        if time_tag:
            parts.append(str(time_tag))

        if parts:
            return f"{base} ({', '.join(parts)})"
        return str(base)

    members: Dict[int, List[str]] = {}

    for year, labels in labels_per_year.items():
        mask = df_clean["Year"] == int(year)
        if mask.sum() == 0:
            continue
        df_year = df_clean.loc[mask].copy()
        df_year["cluster"] = labels

        # Ticker 보강
        if "Ticker" not in df_year.columns and "Code" in df_year.columns:
            df_year["Ticker"] = df_year["Code"]

        sort_col = (
            "Ticker"
            if "Ticker" in df_year.columns
            else ("Code" if "Code" in df_year.columns else None)
        )

        for cid, grp in df_year.groupby("cluster"):
            grp_sorted = grp.sort_values(by=sort_col) if sort_col else grp
            names = [_format_name(r) for _, r in grp_sorted.iterrows()]
            unique_names = list(dict.fromkeys(names))
            members.setdefault(cid, []).extend(unique_names)

    # 클러스터별로 중복 제거(전 기간 기준, 순서 보존)
    for cid, lst in members.items():
        members[cid] = list(dict.fromkeys(lst))

    return members


def build_cluster_members_by_year(
    df_clean: pd.DataFrame, labels_per_year: Dict[int, pd.Series]
) -> Dict[int, Dict[int, List[str]]]:
    """연도별 클러스터 멤버를 연/월 태그와 함께 반환합니다."""

    if df_clean is None or df_clean.empty or not labels_per_year:
        return {}

    out: Dict[int, Dict[int, List[str]]] = {}

    def _format_name(row: pd.Series) -> str:
        ticker = row.get("Ticker")
        code = row.get("Code")
        name = str(row.get("Name", "")).strip()
        base = ticker if pd.notna(ticker) else code if pd.notna(code) else "N/A"
        time_tag = _format_time_tag(row)

        parts = []
        if name and name.lower() != "nan":
            parts.append(name)
        if time_tag:
            parts.append(str(time_tag))

        if parts:
            return f"{base} ({', '.join(parts)})"
        return str(base)

    for year, labels in labels_per_year.items():
        mask = df_clean["Year"] == int(year)
        if mask.sum() == 0:
            continue
        df_year = df_clean.loc[mask].copy()
        df_year["cluster"] = labels

        if "Ticker" not in df_year.columns and "Code" in df_year.columns:
            df_year["Ticker"] = df_year["Code"]

        sort_col = (
            "Ticker"
            if "Ticker" in df_year.columns
            else ("Code" if "Code" in df_year.columns else None)
        )

        year_map: Dict[int, List[str]] = {}
        for cid, grp in df_year.groupby("cluster"):
            grp_sorted = grp.sort_values(by=sort_col) if sort_col else grp
            names = [_format_name(r) for _, r in grp_sorted.iterrows()]
            unique_names = list(dict.fromkeys(names))
            year_map[int(cid)] = unique_names

        if year_map:
            out[int(year)] = year_map

    return out


def build_cluster_top_tickers(
    df_latest: pd.DataFrame, top_n: int = 10
) -> Dict[int, List[str]]:
    """클러스터(상태)별로 가장 자주 등장한 종목 TOP N을 반환합니다."""

    if df_latest is None or df_latest.empty:
        return {}

    id_col = (
        "Ticker"
        if "Ticker" in df_latest.columns
        else ("Code" if "Code" in df_latest.columns else None)
    )
    if id_col is None or "cluster" not in df_latest.columns:
        return {}

    out: Dict[int, List[str]] = {}
    for cid, grp in df_latest.groupby("cluster"):
        vc = grp[id_col].astype(str).value_counts().head(top_n)

        rows_by_id = {k: grp[grp[id_col].astype(str) == k] for k in vc.index}

        def _tag_for_id(k: str) -> str | None:
            rows = rows_by_id.get(k)
            if rows is None or rows.empty:
                return None
            if "Date" in rows.columns:
                rows = rows.sort_values("Date")
                ref_row = rows.iloc[-1]
            else:
                ref_row = rows.iloc[0]
            return _format_time_tag(ref_row)

        formatted = []
        for k, v in vc.items():
            tag = _tag_for_id(k)
            label = f"{k} [{tag}]" if tag else k
            formatted.append(f"{label} ({int(v)})")

        out[int(cid)] = formatted
    return out
