"""GMM 클러스터링 파이프라인 실행 진입점."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict

import joblib

import numpy as np
import pandas as pd

from config import DEFAULT_RESULTS_DIR_NAME, DEFAULT_DATA_DIR_NAME
from config import SNAPSHOT_FREQ, START_YEAR, END_YEAR, K_RANGE
from config import MIN_CLUSTER_FRAC
from config import (
    CLUSTER_NAMES,
    CLUSTER_INTERPRETATIONS,
    CLUSTER_COLORS,
)
from config import UMAP_N_NEIGHBORS, UMAP_MIN_DIST
from config import ROBUSTNESS_WINDOW_YEARS, ROBUSTNESS_EXCLUDE_EVAL_YEAR
from config import ROBUSTNESS_PERIOD_SLICING_ENABLED
from config import ROBUSTNESS_ROLLING_WINDOWS_ENABLED

from gmm.data_loader import (
    FEATURE_COLUMNS,
    convert_df_to_snapshots,
    ensure_date_ticker,
    load_snapshots,
)
from gmm.pipeline_logic import select_best_k, train_gmm_per_year
from gmm.processer import (
    compute_cluster_stats,
    filter_noise,
    get_latest_year_frame,
    preprocess_features,
)
from gmm.report_metrics import compute_report_metrics
from gmm.reporter import (
    build_cluster_members_by_year,
    build_cluster_top_tickers,
    write_text_report,
)
from gmm.model import evaluate_window_robustness
from gmm.robustness import run_period_slicing_robustness, run_rolling_windows_robustness
from gmm.visualizer import (
    plot_cluster_boxplots,
    plot_cluster_heatmap,
    plot_parallel_coords,
    plot_radar_chart,
    plot_risk_return_scatter,
    plot_robustness_heatmap,
    plot_robustness_vs_window,
    plot_sankey,
    plot_umap_scatter,
)

logger = logging.getLogger(__name__)


# 모든 출력 파일 이름 앞에 붙일 접두사
FILE_PREFIX = "gmm_"


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

    artifacts_dir = results_dir / f"{FILE_PREFIX}artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, artifacts_dir / f"{FILE_PREFIX}scaler.pkl")
    joblib.dump(
        {
            "labels_per_year": labels_map,
            "final_k": final_k,
            "feature_columns": features,
            "cluster_means_latest": means,
        },
        artifacts_dir / f"{FILE_PREFIX}metadata.pkl",
    )

    if model is not None:
        joblib.dump(model, artifacts_dir / f"{FILE_PREFIX}latest_year.pkl")


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

    # K 선택 근거(BIC(mean)/Silhouette(mean))는 manual_k 여부와 관계없이 항상 산출
    (
        suggested_k,
        bic_scores,
        silhouette_by_k,
        best_k_mean,
    ) = select_best_k(
        X_feats,
        years,
        K_RANGE,
        manual_k=None,
        results_dir=results_dir,
        file_prefix=FILE_PREFIX,
    )
    k_values = list(K_RANGE)

    if manual_k is None:
        final_k = int(suggested_k)
        logger.info(f"자동 K 사용: K={final_k}")
    else:
        final_k = int(manual_k)
        logger.info(f"수동 K 사용: K={final_k} (추천: BIC(mean)={best_k_mean})")

    labels_map, probs_map, _quality_map, last_model, sort_feature = train_gmm_per_year(
        X_feats, years, df_clean.index, feature_cols_used, final_k
    )

    target_year, df_latest = get_latest_year_frame(df_clean, labels_map)
    df_valid, _cluster_sizes, noise_summary = filter_noise(df_latest, MIN_CLUSTER_FRAC)

    # 핵심 산출물 외 나머지는 appendix로 분리
    appendix_dir = results_dir / "gmm_appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)

    # --- Robustness: 기간(윈도우) 바꿔도 군집이 유지되는지 ---
    robustness_summary = evaluate_window_robustness(
        X_feats,
        years,
        k=final_k,
        window_years=list(ROBUSTNESS_WINDOW_YEARS),
        eval_year=int(target_year),
        exclude_eval_year=bool(ROBUSTNESS_EXCLUDE_EVAL_YEAR),
    )

    if robustness_summary and robustness_summary.get("status") == "ok":
        win_years = [int(w) for w in ROBUSTNESS_WINDOW_YEARS]
        scores = robustness_summary.get("scores") or {}
        ari_series = []
        nmi_series = []
        for w in win_years:
            row = scores.get(f"W{int(w)}") or {}
            ari_series.append(
                float(row.get("ari_vs_all"))
                if row.get("status") == "ok"
                else float("nan")
            )
            nmi_series.append(
                float(row.get("nmi_vs_all"))
                if row.get("status") == "ok"
                else float("nan")
            )

        plot_robustness_vs_window(
            win_years,
            ari_series,
            nmi_series,
            results_dir / f"{FILE_PREFIX}robustness_vs_window.png",
            title="Robustness vs Training Window (ARI/NMI vs ALL)",
        )

        pair = robustness_summary.get("pairwise") or {}
        labels = pair.get("labels") or []
        ari_mat = pair.get("ari")
        nmi_mat = pair.get("nmi")
        if ari_mat is not None and len(labels) > 0:
            plot_robustness_heatmap(
                ari_mat,
                list(labels),
                appendix_dir / f"{FILE_PREFIX}robustness_pairwise_ari.png",
                title="Pairwise ARI (Higher is Better)",
            )
        if nmi_mat is not None and len(labels) > 0:
            plot_robustness_heatmap(
                nmi_mat,
                list(labels),
                appendix_dir / f"{FILE_PREFIX}robustness_pairwise_nmi.png",
                title="Pairwise NMI (Higher is Better)",
            )

    means, stds, cluster_counts = compute_cluster_stats(df_valid, feature_cols_used)

    members_by_year = build_cluster_members_by_year(df_clean, labels_map)
    top_tickers_map = build_cluster_top_tickers(df_valid, top_n=10)

    # 10년치(또는 사용 가능한 최근 10년) 평균/표준편차/클러스터 개수 산출
    means_10y = None
    stds_10y = None
    counts_10y_avg = None
    try:
        min_year = min(int(y) for y in labels_map.keys())
        max_year = int(target_year)
        start_10y = max(min_year, max_year - 9)
        frames = []
        per_year_counts = {}
        for y in range(start_10y, max_year + 1):
            labels = labels_map.get(int(y))
            if labels is None:
                continue
            common_idx = df_clean.index.intersection(labels.index)
            if len(common_idx) == 0:
                continue
            df_y = df_clean.loc[common_idx].copy()
            df_y["cluster"] = labels.loc[common_idx].astype(int)
            frames.append(df_y)
            per_year_counts[int(y)] = df_y.groupby("cluster").size()

        if frames:
            df_10y_all = pd.concat(frames, axis=0)
            if not df_10y_all.empty:
                means_10y = df_10y_all.groupby("cluster")[feature_cols_used].mean()
                stds_10y = df_10y_all.groupby("cluster")[feature_cols_used].std()
            if per_year_counts:
                counts_df = pd.DataFrame(per_year_counts).fillna(0).T
                counts_10y_avg = counts_df.mean(axis=0)
    except Exception:
        means_10y = None
        stds_10y = None
        counts_10y_avg = None

    # CSV는 제출/공유용으로 members_by_year만 저장합니다.
    # 포맷: year + cluster_0~cluster_3 컬럼(각 셀은 줄바꿈으로 멤버 나열)
    if members_by_year:
        years_sorted = sorted(members_by_year.keys())
        cluster_ids = list(range(int(final_k))) if int(final_k) > 0 else [0, 1, 2, 3]
        rows = []
        for y in years_sorted:
            per_c = members_by_year.get(int(y)) or {}
            row = {"year": int(y)}
            for cid in cluster_ids:
                members = per_c.get(int(cid)) or []
                row[f"cluster_{int(cid)}"] = "\n".join(map(str, members))
            rows.append(row)

        pd.DataFrame(rows).to_csv(
            appendix_dir / f"{FILE_PREFIX}cluster_members_by_year.csv",
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )

    save_artifacts(
        appendix_dir,
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

    # --- Robustness: 기간 슬라이싱(정석) ---
    if ROBUSTNESS_PERIOD_SLICING_ENABLED:
        try:
            run_period_slicing_robustness(
                df_base=df_clean,
                feature_cols=feature_cols,
                results_dir=appendix_dir,
                file_prefix=FILE_PREFIX,
            )
        except Exception as e:
            logger.warning(f"기간 슬라이싱 robustness 건너뜀: {e}")

    # --- Robustness: 36개월 롤링(1년 스텝), K=4 고정 ---
    rolling_robustness_summary = None
    if ROBUSTNESS_ROLLING_WINDOWS_ENABLED:
        try:
            rolling_robustness_summary = run_rolling_windows_robustness(
                df_base=df_clean,
                feature_cols=feature_cols,
                results_dir=appendix_dir,
                file_prefix=FILE_PREFIX,
            )
        except Exception as e:
            logger.warning(f"롤링 윈도우 robustness 건너뜀: {e}")

    plot_cluster_heatmap(
        means, results_dir / f"{FILE_PREFIX}heatmap.png", cluster_names=CLUSTER_NAMES
    )
    plot_radar_chart(
        means,
        appendix_dir / f"{FILE_PREFIX}radar.png",
        cluster_names=CLUSTER_NAMES,
        cluster_colors=CLUSTER_COLORS,
    )
    plot_parallel_coords(
        df_valid,
        feature_cols_used,
        appendix_dir / f"{FILE_PREFIX}parallel.png",
        cluster_names=CLUSTER_NAMES,
        cluster_colors=CLUSTER_COLORS,
    )
    plot_risk_return_scatter(
        means,
        appendix_dir / f"{FILE_PREFIX}risk_return.png",
        cluster_names=CLUSTER_NAMES,
        cluster_colors=CLUSTER_COLORS,
    )
    plot_cluster_boxplots(
        df_valid,
        feature_cols_used,
        results_dir / f"{FILE_PREFIX}cluster_boxplots.png",
        cluster_colors=CLUSTER_COLORS,
    )

    if "Ticker" in df_clean.columns:
        plot_sankey(
            df_clean,
            labels_map,
            results_dir / f"{FILE_PREFIX}sankey.html",
            cluster_names=CLUSTER_NAMES,
            cluster_colors=CLUSTER_COLORS,
        )

    embedding = compute_umap_embedding(
        X_feats,
        n_neighbors=int(UMAP_N_NEIGHBORS),
        min_dist=float(UMAP_MIN_DIST),
    )
    if embedding is not None and len(embedding) == len(df_clean):
        labels_series = pd.Series(-1, index=df_clean.index)
        for year, lbl in labels_map.items():
            common_idx = labels_series.index.intersection(lbl.index)
            labels_series.loc[common_idx] = lbl.loc[common_idx].astype(int)
        labels_array = labels_series.to_numpy()
        plot_umap_scatter(
            embedding,
            labels_array,
            results_dir / f"{FILE_PREFIX}umap.png",
            cluster_names=CLUSTER_NAMES,
            cluster_colors=CLUSTER_COLORS,
        )

    # --- 정리: 루트에는 핵심 8개만 남기고 나머지는 appendix로 이동 ---
    core_files = {
        f"{FILE_PREFIX}bic_curve_mean.png",
        f"{FILE_PREFIX}silhouette_curve_mean.png",
        f"{FILE_PREFIX}cluster_boxplots.png",
        f"{FILE_PREFIX}heatmap.png",
        f"{FILE_PREFIX}robustness_vs_window.png",
        f"{FILE_PREFIX}sankey.html",
        f"{FILE_PREFIX}umap.png",
        f"{FILE_PREFIX}report.md",
    }

    def _move_to_appendix(p: Path) -> None:
        dest = appendix_dir / p.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(p), str(dest))

    try:
        for p in results_dir.iterdir():
            if p.name == "gmm_appendix":
                continue
            # gmm_artifacts(폴더) 포함: prefix로 시작하는 파일/폴더는 이동 대상
            if p.name.startswith(FILE_PREFIX) and p.name not in core_files:
                _move_to_appendix(p)
    except Exception as e:
        logger.warning(f"appendix 정리 단계 건너뜀: {e}")

    # 리포트는 모든 산출물 저장 이후 작성(산출물 리스트 누락 방지)
    write_text_report(
        results_dir / f"{FILE_PREFIX}report.md",
        load_stats or {},
        prep_stats or {},
        bic_scores or [],
        silhouette_by_k or {},
        k_values or [],
        final_k,
        np.array(pca_explained) if pca_explained is not None else np.array([]),
        means,
        stds,
        cluster_counts,
        cluster_members=top_tickers_map,
        target_year=target_year,
        cluster_means_10y=means_10y,
        cluster_stds_10y=stds_10y,
        cluster_counts_10y_avg=counts_10y_avg,
        noise_summary=noise_summary,
        quality_summary=report_metrics.get("quality_summary"),
        silhouette_summary=report_metrics.get("silhouette_summary"),
        ex_post_summary=report_metrics.get("ex_post_summary"),
        robustness_summary=robustness_summary,
        rolling_robustness_summary=rolling_robustness_summary,
        best_k_mean=best_k_mean,
        label_alignment_feature=sort_feature,
        cluster_names=CLUSTER_NAMES,
        cluster_interpretations=CLUSTER_INTERPRETATIONS,
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

        if df is None or df.empty:
            # 단독 실행/외부 주입이 없을 때: 로더에서 직접 스냅샷 로드
            self.df, self.load_stats = load_snapshots(
                data_dir=(
                    Path(DEFAULT_DATA_DIR_NAME)
                    if DEFAULT_DATA_DIR_NAME
                    else Path("data")
                ),
                start_year=START_YEAR,
                end_year=END_YEAR,
                freq=SNAPSHOT_FREQ,
            )
            return

        # 외부(df) 주입 경로: 멀티인덱스를 컬럼으로 풀고 스냅샷 변환
        df_norm = ensure_date_ticker(df)
        if not {"Date", "Ticker"}.issubset(df_norm.columns):
            raise KeyError(
                "입력 df에 Date/Ticker가 없습니다. data_load() 결과(MultiIndex) 또는 Date/Ticker 컬럼을 포함한 DF를 넣어주세요."
            )

        if df is not None and not df.empty:
            try:
                self.df = convert_df_to_snapshots(
                    df_norm,
                    freq=SNAPSHOT_FREQ,
                    start_year=START_YEAR,
                    end_year=END_YEAR,
                )
            except Exception as e:
                logger.warning(f"스냅샷 변환 실패, 원본 사용: {e}")
                self.df = df_norm

    def run(self, manual_k: int | None = 4) -> str:
        return run_gmm_pipeline(
            df=self.df,
            results_dir=self.results_dir,
            load_stats=self.load_stats,
            manual_k=manual_k,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    result = GMM().run(manual_k=4)
    print(result)
