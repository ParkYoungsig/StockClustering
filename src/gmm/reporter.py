"""GMM 멀티 스냅샷(연도/시점별) 파이프라인 텍스트 리포트 유틸.

클러스터링 결과를 사람이 읽기 쉬운 형태로 요약하여 텍스트 파일로 저장합니다.
주요 내용: 데이터 로딩/전처리 요약, PCA 사용 여부, BIC/Silhouette 기반 K 근거,
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
    silhouette_by_k: Dict[int, float] | None,
    k_values: Iterable[int],
    best_k: int,
    pca_explained: np.ndarray,
    cluster_means: pd.DataFrame,
    cluster_stds: pd.DataFrame | None,
    cluster_counts: pd.Series,
    *,
    target_year: int | None = None,
    cluster_means_10y: pd.DataFrame | None = None,
    cluster_stds_10y: pd.DataFrame | None = None,
    cluster_counts_10y_avg: pd.Series | None = None,
    cluster_members: Dict[int, List[str]],
    noise_summary: Dict | None = None,
    quality_summary: Dict | None = None,
    silhouette_summary: Dict | None = None,
    ex_post_summary: pd.DataFrame | None = None,
    robustness_summary: Dict | None = None,
    rolling_robustness_summary: Dict | None = None,
    best_k_mean: int | None = None,
    label_alignment_feature: str | None = None,
    cluster_names: Optional[Dict[int, str]] = None,
    cluster_interpretations: Optional[Dict[int, str]] = None,
    include_output_file_list: bool = True,
) -> None:
    """GMM 파이프라인 결과를 텍스트 리포트로 저장합니다.

    - 데이터/전처리 요약
    - (선택) PCA 설명력
    - BIC(mean) / Silhouette(mean) 기반 K 근거 요약
    - (선택) robustness(윈도우/기간 변화) 요약
    - 클러스터 평균/개수/구성 종목 및 규칙 기반 해석
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = output_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        _write_report(
            f,
            md=True,
            output_path=output_path,
            load_stats=load_stats,
            prep_stats=prep_stats,
            bic_scores=bic_scores,
            silhouette_by_k=silhouette_by_k,
            k_values=k_values,
            best_k=best_k,
            pca_explained=pca_explained,
            cluster_means=cluster_means,
            cluster_stds=cluster_stds,
            cluster_counts=cluster_counts,
            target_year=target_year,
            cluster_means_10y=cluster_means_10y,
            cluster_stds_10y=cluster_stds_10y,
            cluster_counts_10y_avg=cluster_counts_10y_avg,
            cluster_members=cluster_members,
            noise_summary=noise_summary,
            quality_summary=quality_summary,
            silhouette_summary=silhouette_summary,
            ex_post_summary=ex_post_summary,
            robustness_summary=robustness_summary,
            rolling_robustness_summary=rolling_robustness_summary,
            best_k_mean=best_k_mean,
            label_alignment_feature=label_alignment_feature,
            cluster_names=cluster_names,
            cluster_interpretations=cluster_interpretations,
            include_output_file_list=include_output_file_list,
        )


def _write_report(
    f,
    md=False,
    load_stats=None,
    prep_stats=None,
    output_path: Path | None = None,
    bic_scores=None,
    silhouette_by_k=None,
    k_values=None,
    best_k=None,
    pca_explained=None,
    cluster_means=None,
    cluster_stds=None,
    cluster_counts=None,
    *,
    target_year: int | None = None,
    cluster_means_10y=None,
    cluster_stds_10y=None,
    cluster_counts_10y_avg=None,
    cluster_members=None,
    noise_summary=None,
    quality_summary=None,
    silhouette_summary=None,
    ex_post_summary=None,
    robustness_summary=None,
    rolling_robustness_summary=None,
    best_k_mean=None,
    label_alignment_feature=None,
    cluster_names=None,
    cluster_interpretations=None,
    include_output_file_list=True,
):
    # 안전한 로컬 참조
    ls = load_stats or {}
    ps = prep_stats or {}
    op = output_path
    # 헤더
    if md:
        f.write("# GMM 스냅샷 리포트\n\n")
        f.write("## 데이터 요약\n")
    else:
        f.write("GMM 스냅샷 리포트\n")
        f.write("================\n\n")
        f.write("데이터 요약:\n")

    # load_stats 키가 변경되어도 대응하도록 주요 카운트를 보정합니다.
    # data_loader는 과거에 `rows_raw`/`rows_snapshots`를 반환할 수 있습니다.
    total_files_val = ls.get("total_files") or ls.get("rows_raw")
    snapshots_val = ls.get("snapshots") or ls.get("rows_snapshots")

    f.write(f"- 읽은 파일 수: {total_files_val}\n")
    f.write(f"- 사용 스냅샷 수: {snapshots_val}\n")
    if "start_year" in ls:
        f.write(f"- 분석 연도 범위: {ls.get('start_year')} - {ls.get('end_year')}\n")
        f.write(f"- 폴백 윈도우(일): {ls.get('fallback_days')}\n")
    else:
        f.write(
            f"- 기준일: {ls.get('reference_date')} (폴백: {ls.get('fallback_date')})\n"
        )
    f.write(f"- 결측 제거 후 최종 행 수: {ps.get('rows_after_dropna')}\n\n")

    # target_year 라벨 준비
    year_label = str(target_year) if target_year is not None else "latest"

    # 피처 정리
    if md:
        f.write("## 피처 정리(자동 중복 제거)\n")
    else:
        f.write("피처 정리(자동 중복 제거):\n")
    if md:
        if prep_stats and prep_stats.get("feature_cols_used"):
            f.write(
                f"- 최종 사용 피처 수: {len(prep_stats.get('feature_cols_used'))}\n"
            )
        if prep_stats and prep_stats.get("low_variance_threshold") is not None:
            f.write(
                f"- 저분산 제거 임계값: {prep_stats.get('low_variance_threshold')}\n"
            )
        if prep_stats and prep_stats.get("corr_threshold") is not None:
            f.write(
                f"- 상관(스피어만) 제거 임계값(|ρ|≥): {prep_stats.get('corr_threshold')}\n"
            )
    else:
        if prep_stats and prep_stats.get("feature_cols_used"):
            f.write(
                f"  최종 사용 피처 수: {len(prep_stats.get('feature_cols_used'))}\n"
            )
        if prep_stats and prep_stats.get("low_variance_threshold") is not None:
            f.write(
                f"  저분산 제거 임계값: {prep_stats.get('low_variance_threshold')}\n"
            )
        if prep_stats and prep_stats.get("corr_threshold") is not None:
            f.write(
                f"  상관(스피어만) 제거 임계값(|ρ|≥): {prep_stats.get('corr_threshold')}\n"
            )
    f.write("\n")

    # 노이즈 처리
    if md:
        f.write("## 노이즈 처리(최소 클러스터 크기 기준)\n")
    else:
        f.write("노이즈 처리(최소 클러스터 크기 기준):\n")
    if noise_summary:
        if md:
            f.write(f"- 크기 임계값: {noise_summary.get('size_threshold')}\n")
            f.write(f"- 노이즈로 재할당된 행 수: {noise_summary.get('noise_rows')}\n")
            f.write(f"- 제거된 클러스터: {noise_summary.get('removed_clusters')}\n")
        else:
            f.write(f"  크기 임계값: {noise_summary.get('size_threshold')}\n")
            f.write(f"  노이즈로 재할당된 행 수: {noise_summary.get('noise_rows')}\n")
            f.write(f"  제거된 클러스터: {noise_summary.get('removed_clusters')}\n")
    f.write("\n")

    # PCA
    if md:
        f.write("## PCA\n")
    else:
        f.write("PCA: ")
    if pca_explained is not None and len(pca_explained) > 0:
        if md:
            f.write(
                "설명분산비(components): "
                + ", ".join(f"{v:.4f}" for v in pca_explained)
                + "\n"
            )
            f.write(f"- 사용 컴포넌트 수: {len(pca_explained)}\n\n")
        else:
            f.write("\n")
            f.write(
                "  설명분산비(components): "
                + ", ".join(f"{v:.4f}" for v in pca_explained)
                + "\n"
            )
            f.write(f"  사용 컴포넌트 수: {len(pca_explained)}\n\n")
    else:
        if md:
            f.write("생략(표준화된 원본 팩터 사용)\n\n")
        else:
            f.write("생략(표준화된 원본 팩터 사용)\n\n")

    # 군집 수(K) 선택 근거
    if md:
        f.write("## 군집 수(K) 선택 근거\n")
        f.write("(낮을수록 좋음: BIC, 높을수록 좋음: Silhouette)\n\n")
        if label_alignment_feature:
            f.write(f"- 라벨 정렬 기준 피처: {label_alignment_feature}\n")
        if best_k_mean is not None:
            f.write(f"- BIC(mean) 기준 추천 K: {best_k_mean}\n")
        f.write(f"- 최종 사용 K: {best_k}\n\n")
        # K 후보별 지표 표
        f.write("| K | BIC(mean) | Silhouette(mean) |\n")
        f.write("|---|-----------:|------------------:|\n")
        k_list = list(k_values or [])
        for i, k in enumerate(k_list):
            bic_v = bic_scores[i] if bic_scores and i < len(bic_scores) else None
            sil_v = (
                float(silhouette_by_k.get(int(k)))
                if silhouette_by_k and int(k) in silhouette_by_k
                else None
            )
            bic_s = f"{bic_v:10.2f}" if bic_v is not None else "(n/a)"
            sil_s = f"{sil_v:16.4f}" if sil_v is not None else "(n/a)"
            f.write(f"| {int(k):>2} | {bic_s} | {sil_s} |\n")
        f.write("\n")
    else:
        f.write("군집 수(K) 선택 근거:\n")
        f.write("  (낮을수록 좋음: BIC, 높을수록 좋음: Silhouette)\n")
        if label_alignment_feature:
            f.write(f"  라벨 정렬 기준 피처: {label_alignment_feature}\n")
        if best_k_mean is not None:
            f.write(f"  BIC(mean) 기준 추천 K: {best_k_mean}\n")
        f.write(f"  최종 사용 K: {best_k}\n\n")
        k_list = list(k_values or [])
        f.write("K 후보별 지표:\n")
        f.write("  K |   BIC(mean) | Silhouette(mean)\n")
        f.write("  --|------------|------------------\n")
        for i, k in enumerate(k_list):
            bic_v = bic_scores[i] if bic_scores and i < len(bic_scores) else None
            sil_v = (
                float(silhouette_by_k.get(int(k)))
                if silhouette_by_k and int(k) in silhouette_by_k
                else None
            )
            bic_s = f"{bic_v:10.2f}" if bic_v is not None else "     (n/a)"
            sil_s = f"{sil_v:16.4f}" if sil_v is not None else "        (n/a)"
            f.write(f"  {int(k):>2} | {bic_s} | {sil_s}\n")
        f.write("\n")

    # Robustness
    if md:
        f.write("## 기간(윈도우) 변경 Robustness(군집 유지)\n")
    else:
        f.write("기간(윈도우) 변경 Robustness(군집 유지):\n")
    if robustness_summary and robustness_summary.get("status") == "ok":
        if md:
            f.write(
                f"- 평가 연도: {robustness_summary.get('eval_year')} (n={robustness_summary.get('n_eval')})\n"
            )
            f.write(
                f"- 평가 연도 학습 제외(exclude_eval_year): {robustness_summary.get('exclude_eval_year')}\n"
            )
            f.write("- 기준(baseline): ALL(학습 가능한 전체 연도)\n")
            f.write("\n| Window | Train Years | n_train | ARI vs ALL | NMI vs ALL |\n")
            f.write("|--------|-------------|--------:|------------:|-----------:|\n")
        else:
            f.write(
                f"  평가 연도: {robustness_summary.get('eval_year')} (n={robustness_summary.get('n_eval')})\n"
            )
            f.write(
                f"  평가 연도 학습 제외(exclude_eval_year): {robustness_summary.get('exclude_eval_year')}\n"
            )
            f.write("  기준(baseline): ALL(학습 가능한 전체 연도)\n")
            f.write("  Window | Train Years | n_train | ARI vs ALL | NMI vs ALL\n")
            f.write("  ------|------------|--------|-----------:|----------:\n")
        scores = robustness_summary.get("scores") or {}
        for key, row in scores.items():
            if not isinstance(row, dict):
                continue
            train_span = f"{row.get('train_year_start')}~{row.get('train_year_end')}"
            n_train = row.get("n_train")
            ari = row.get("ari_vs_all")
            nmi = row.get("nmi_vs_all")
            ari_s = f"{float(ari):.4f}" if ari is not None else "(n/a)"
            nmi_s = f"{float(nmi):.4f}" if nmi is not None else "(n/a)"
            if md:
                f.write(f"| {key} | {train_span} | {n_train} | {ari_s} | {nmi_s} |\n")
            else:
                f.write(
                    f"  {key:>5} | {train_span:>10} | {str(n_train):>6} | {ari_s:>9} | {nmi_s:>8}\n"
                )
        f.write("\n")

    # 이하 모든 주요 섹션(클러스터별 통계, 품질, 해석, 산출물 등)도 md일 때 헤더/표/리스트로 변환 추가 적용 가능

    # 피처 중복 제거/정리 요약 (상관/저분산)
    feature_candidates_total = ps.get("feature_candidates_total")
    feature_candidates_base = ps.get("feature_candidates_base") or []
    feature_candidates_extra = ps.get("feature_candidates_extra_numeric") or []
    feature_cols_input_count = ps.get("feature_cols_input_count")
    feature_cols_used = ps.get("feature_cols_used") or []
    removed_low_var = ps.get("removed_low_variance") or []
    removed_high_corr = ps.get("removed_high_corr") or []
    dropped_high_missing = ps.get("dropped_high_missing") or []
    max_missing_ratio = ps.get("max_missing_ratio")
    low_var_thr = ps.get("low_variance_threshold")
    corr_thr = ps.get("corr_threshold")
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

    # 클러스터별 피처 평균: 최신 연도 표와 10년치 평균 표를 분리 출력
    if cluster_means is not None:
        means_disp = cluster_means.copy()
        means_disp.index = [_cluster_label(cid, cluster_names) for cid in means_disp.index]
        f.write(f"클러스터별 피처 평균 ({year_label}):\n")
        f.write(means_disp.to_string())
        f.write("\n\n")
    if cluster_means_10y is not None:
        cm10 = cluster_means_10y.copy()
        cm10.index = [_cluster_label(cid, cluster_names) for cid in cm10.index]
        f.write("클러스터별 피처 평균 (10년치 평균):\n")
        f.write(cm10.to_string())
        f.write("\n")

    if cluster_stds is not None and not getattr(cluster_stds, "empty", True):
        stds_disp = cluster_stds.copy()
        stds_disp.index = [_cluster_label(cid, cluster_names) for cid in stds_disp.index]
        f.write("\n클러스터별 피처 표준편차 ({year_label}):\n")
        f.write(stds_disp.to_string())
        f.write("\n\n")
    if cluster_stds_10y is not None and not getattr(cluster_stds_10y, "empty", True):
        s10 = cluster_stds_10y.copy()
        s10.index = [_cluster_label(cid, cluster_names) for cid in s10.index]
        f.write("클러스터별 피처 표준편차 (10년치 평균):\n")
        f.write(s10.to_string())
        f.write("\n")

    f.write("\n클러스터별 개수(최신 연도 vs 10년 평균):\n")
    if cluster_counts is not None:
        # Normalize cluster_counts to a dict {cluster_id: count}
        counts_map: Dict[int, int] = {}
        try:
            if hasattr(cluster_counts, "items"):
                counts_map = {int(k): int(v) for k, v in cluster_counts.items()}
            elif hasattr(cluster_counts, "shape") or hasattr(cluster_counts, "__iter__"):
                # numpy array or list-like: assume index 0..N-1
                counts_map = {int(i): int(v) for i, v in enumerate(list(cluster_counts))}
        except Exception:
            counts_map = {}

        latest_total = sum(counts_map.values()) if counts_map else None

        for cid, count in sorted(counts_map.items()):
            pct = f"{int(round((count / latest_total) * 100))}%" if latest_total and latest_total > 0 else "(n/a)"
            out_line = f"  {_cluster_label(cid, cluster_names)}: {int(count)}개 ({pct})"
            # append 10-year average if available
            if cluster_counts_10y_avg is not None:
                try:
                    avg10 = (
                        cluster_counts_10y_avg.get(int(cid))
                        if hasattr(cluster_counts_10y_avg, "get")
                        else cluster_counts_10y_avg[int(cid)]
                    )
                    if avg10 is not None:
                        avg_pct = f"{int(round((avg10 / latest_total) * 100))}%" if latest_total and latest_total > 0 else "(n/a)"
                        out_line += f"  | 10yr avg: {int(round(avg10))}개 ({avg_pct})"
                except Exception:
                    pass
            f.write(out_line + "\n")

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

    # transition_summary 출력은 요구사항에 따라 리포트에서 제거되었습니다.

    if ex_post_summary is not None and not getattr(ex_post_summary, "empty", True):
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
        for cid in cluster_means.index if cluster_means is not None else []:
            try:
                key = int(cid)
            except Exception:
                key = cid
            interp = (cluster_interpretations or {}).get(key)
            if interp:
                f.write(f"  {_cluster_label(cid, cluster_names)}: {interp}\n")

    f.write("\n클러스터(상태)에 가장 자주 진입한 종목 TOP (최신 연도):\n")
    if cluster_members:
        for cid, members in (
            cluster_members.items() if hasattr(cluster_members, "items") else []
        ):
            f.write(f"  {_cluster_label(cid, cluster_names)} ({len(members)}): ")
            f.write(", ".join(members))
            f.write("\n")

    if include_output_file_list:
        # 기본적으로 리포트가 쓰여지는 output 폴더(출력 루트)를 사용
        results_dir = None
        if op and isinstance(op, Path):
            results_dir = op.parent
        elif ls.get("output_root"):
            results_dir = Path(ls.get("output_root"))
        results_dir = results_dir or Path(".")
        core_outputs: List[str] = []
        appendix_outputs: List[str] = []
        try:
            for p in sorted(results_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(results_dir).as_posix()
                if not p.name.startswith("gmm_"):
                    continue
                if "/" not in rel and not rel.startswith("gmm_appendix/"):
                    core_outputs.append(rel)
                    continue
                appendix_outputs.append(rel)
        except Exception:
            core_outputs = []
            appendix_outputs = []

        f.write("\n산출물 파일 목록(자동):\n")
        core_outputs = sorted(dict.fromkeys(core_outputs))
        appendix_outputs = sorted(dict.fromkeys(appendix_outputs))

        if not core_outputs and not appendix_outputs:
            f.write("  - (없음)\n")
        else:
            f.write("  [Core]\n")
            if core_outputs:
                for name in core_outputs:
                    f.write(f"    - {name}\n")
            else:
                f.write("    - (없음)\n")

            f.write("  [Appendix]\n")
            if appendix_outputs:
                for name in appendix_outputs:
                    f.write(f"    - {name}\n")
            else:
                f.write("    - (없음)\n")


def write_period_slicing_robustness_report(
    output_path: Path,
    summary: Dict,
) -> None:
    """기간 슬라이싱 기반 robustness 비교 결과를 텍스트로 저장합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    md_path = output_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# 기간 슬라이싱 Robustness 리포트\n\n")

        cases = summary.get("cases") or {}
        labels = summary.get("labels") or []

        f.write("## 케이스 요약(각 기간별 재학습)\n")
        f.write("| Case | Years | BestK | LatestYear | n_latest |\n")
        f.write("|------|-------|------:|-----------:|--------:|\n")
        for name in labels or list(cases.keys()):
            row = cases.get(name) or {}
            years_span = f"{row.get('start_year')}~{row.get('end_year')}"
            f.write(
                f"| {name} | {years_span:>9} | {str(row.get('best_k')):>5} | {str(row.get('latest_year')):>10} | {str(row.get('n_latest')):>7} |\n"
            )
        f.write("\n")

        pair = summary.get("pairwise") or {}
        if pair:
            f.write("## 케이스 간 중심(centroid) 유사도 (클러스터 매칭 후 평균)\n")
            if pair.get("centroid_cosine_mean") is not None:
                f.write("- cosine(mean matched): (matrix attached as artifact)\n")
            if pair.get("centroid_corr_mean") is not None:
                f.write("- corr(mean matched): (matrix attached as artifact)\n")
            f.write("\n")


def build_cluster_members_all_years(
    df_clean: pd.DataFrame, labels_per_year: Dict[int, pd.Series]
) -> Dict[int, List[str]]:
    """(Deprecated) 전 기간 멤버 리스트는 제출용 산출물에서 제외합니다."""

    # 호환을 위해 함수는 남겨두되, 더 이상 사용하지 않습니다.
    return {}


def build_cluster_members_by_year(
    df_clean: pd.DataFrame, labels_per_year: Dict[int, pd.Series]
) -> Dict[int, Dict[int, List[str]]]:
    """연도별 클러스터 멤버를 연/월 태그와 함께 반환합니다."""

    if df_clean is None or df_clean.empty or not labels_per_year:
        return {}

    out: Dict[int, Dict[int, List[str]]] = {}

    def _format_name(row: pd.Series) -> str:
        """멤버 표기: '종목명 티커 (YYYY)' 또는 '티커 (YYYY)'."""

        ticker = row.get("Ticker")
        code = row.get("Code")
        name = str(row.get("Name", "")).strip()
        base = (
            str(ticker) if pd.notna(ticker) else str(code) if pd.notna(code) else "N/A"
        )
        time_tag = _format_time_tag(row)
        tag = str(time_tag) if time_tag else ""

        has_name = bool(name) and name.lower() != "nan"
        if has_name and tag:
            return f"{name} {base} ({tag})"
        if has_name:
            return f"{name} {base}"
        if tag:
            return f"{base} ({tag})"
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

        # 동일 연도에서 동일 티커가 여러 행으로 존재할 수 있는 케이스 방어
        # (한 티커가 여러 클러스터로 중복되는 현상 방지)
        if sort_col:
            if "Date" in df_year.columns:
                df_year = df_year.sort_values("Date")
            df_year = df_year.drop_duplicates(subset=[sort_col], keep="last")

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
