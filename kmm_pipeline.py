"""
kmeans_pca_pipeline.py

Notebook k_means_01.ipynb (cells 1~7) 구조를 .py로 옮긴 버전.
- 데이터 폴더(parquet) 로드 -> 특정 날짜 스냅샷 -> 피처 선택/정리 -> 결측/inf 처리
- winsorize + scaling(Standard/Robust) 버전 생성
- PCA 결과(설명분산/누적/로딩) variant별 비교 출력 + 파일 저장

필수 패키지: pandas, numpy, scikit-learn, matplotlib, pyarrow(or fastparquet)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler


# ======================================================================================
# 1) 데이터 로드
# ======================================================================================
def load_parquets_to_df_all(
    data_folder: str,
    ticker_from_filename: bool = True,
    ticker_col: str = "Ticker",
) -> pd.DataFrame:
    """
    data_folder 안의 모든 .parquet 파일을 읽어 하나의 df_all로 합칩니다.
    - 파일명에서 티커를 추출하는 관례: "<TICKER>_....parquet" (언더스코어 기준 앞부분)
    """
    parquet_files = [
        f for f in os.listdir(data_folder) if f.lower().endswith(".parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {data_folder}")

    all_stocks: List[pd.DataFrame] = []
    for file in sorted(parquet_files):
        fp = os.path.join(data_folder, file)
        # print("[READ]", fp) --> 출력
        try:
            df = pd.read_parquet(fp)  # default: pyarrow(usually)
        except Exception as e:
            # pyarrow가 특정 parquet에서 터질 때 fastparquet으로 재시도(설치돼 있으면)
            try:
                df = pd.read_parquet(fp, engine="fastparquet")
                print(f"[FALLBACK] fastparquet succeeded: {os.path.basename(fp)}")
            except Exception:
                raise RuntimeError(
                    f"Failed to read parquet: {fp}\\n  original: {type(e).__name__}: {e}"
                ) from e

        if ticker_from_filename:
            ticker_code = file.split("_")[0]
            df[ticker_col] = ticker_code

        all_stocks.append(df)

    df_all = pd.concat(all_stocks, axis=0, ignore_index=False)
    return df_all


# ======================================================================================
# 2) 날짜 스냅샷
# ======================================================================================
def snapshot_by_date(
    df_all: pd.DataFrame,
    date: str,
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    keep_ticker: bool = True,
) -> pd.DataFrame:
    """
    df_all에서 특정 날짜(YYYY-MM-DD)의 행만 뽑아 스냅샷 데이터프레임을 만듭니다.

    지원:
    - df_all.index 가 datetime-like 인 경우: index로 필터
    - 아니면 date_col 컬럼이 있는 경우: date_col로 필터
    """
    target = pd.to_datetime(date).date()

    if isinstance(df_all.index, pd.DatetimeIndex):
        mask = df_all.index.date == target
        snap = df_all.loc[mask].copy()
    elif date_col in df_all.columns:
        d = pd.to_datetime(df_all[date_col], errors="coerce")
        mask = d.dt.date == target
        snap = df_all.loc[mask].copy()
    else:
        raise ValueError(
            f"Cannot find datetime index or '{date_col}' column to snapshot by date."
        )

    # 흔히 snapshot은 tickers가 섞여있으니 정렬
    if keep_ticker and ticker_col in snap.columns:
        snap = snap.sort_values(ticker_col)

    return snap


# ======================================================================================
# 3) 피처 컬럼 선택
# ======================================================================================
def select_feature_columns(
    snap: pd.DataFrame,
    feature_cols: Optional[Iterable[str]] = None,
    ticker_col: str = "Ticker",
    keep_ticker: bool = True,
) -> pd.DataFrame:
    """
    - feature_cols 가 주어지면: 그 컬럼만 선택(없는 컬럼은 자동 제외 + 경고 출력)
    - feature_cols 가 None이면: snap의 numeric 컬럼 전부 사용
    """
    if feature_cols is None:
        cols = snap.select_dtypes(include="number").columns.tolist()
    else:
        feature_cols = list(feature_cols)
        missing = [c for c in feature_cols if c not in snap.columns]
        if missing:
            print(f"[WARN] missing columns (ignored): {missing}")
        cols = [c for c in feature_cols if c in snap.columns]

    if keep_ticker and ticker_col in snap.columns:
        cols = [ticker_col] + [c for c in cols if c != ticker_col]

    return snap.loc[:, cols].copy()


# ======================================================================================
# 4) 상관 높은 피처 제거
# ======================================================================================
def drop_high_corr_features(
    df_feat: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    numeric 컬럼들에 대해 |corr| >= threshold 인 피처를 제거합니다.
    간단하게 upper triangle 방식으로 중복 제거.
    반환:
      - reduced_df
      - dropped_cols
      - corr_matrix (numeric)
    """
    num = df_feat.select_dtypes(include="number").copy()
    if num.shape[1] == 0:
        raise ValueError("No numeric columns found to compute correlation.")

    corr = num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] >= threshold)]

    reduced = df_feat.drop(columns=to_drop, errors="ignore")
    return reduced, to_drop, corr


# ======================================================================================
# 5) inf -> NaN, 결측치 중위값 대치
# ======================================================================================
def inf_to_nan_and_median_impute(
    df_feat: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, int]:
    """
    df_feat 전체에서 inf/-inf를 NaN으로 바꾸고,
    numeric 컬럼은 각 컬럼 중위값으로 NaN을 대치합니다.
    반환:
      - X_imputed (전체 컬럼 유지; 단, non-numeric NaN은 남을 수 있음)
      - na_before (컬럼별 NaN 개수)
      - na_after_total (대치 후 남은 NaN 총합)
    """
    X_raw = df_feat.copy().replace([np.inf, -np.inf], np.nan)
    na_before = X_raw.isna().sum().sort_values(ascending=False)

    med = X_raw.median(numeric_only=True)
    X_imputed = X_raw.copy()
    X_imputed[med.index] = X_imputed[med.index].fillna(med)

    na_after_total = int(X_imputed.isna().sum().sum())
    return X_imputed, na_before, na_after_total


# ======================================================================================
# 6) winsorize + scaling
# ======================================================================================
def winsorize_df(
    X: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    numeric 컬럼별 quantile 기반 winsorize(clip) 수행.
    """
    Xw = X.copy()
    num_cols = Xw.select_dtypes(include="number").columns
    clip_info = []

    for c in num_cols:
        lo = Xw[c].quantile(lower_q)
        hi = Xw[c].quantile(upper_q)
        before = Xw[c].copy()
        Xw[c] = Xw[c].clip(lo, hi)
        n_clipped = int((before != Xw[c]).sum())
        clip_info.append((c, float(lo), float(hi), n_clipped))

    info = pd.DataFrame(
        clip_info, columns=["feature", "clip_lo", "clip_hi", "n_clipped"]
    )
    return Xw, info


def build_winsorize_and_scalers(
    X_imputed: pd.DataFrame,
    lower_q: float,
    upper_q: float,
) -> Dict[str, pd.DataFrame]:
    """
    X_imputed (numeric only 권장)로부터 아래 3종 변환 결과 생성:
      - only_wins: winsorize만
      - wins_std: winsorize + StandardScaler
      - wins_rob: winsorize + RobustScaler

    반환 dict에는 clip_table도 같이 담음.
    """
    # scaler는 numeric만 대상으로 (비수치 컬럼 있으면 오류나므로)
    X_num = X_imputed.select_dtypes(include="number").copy()
    X_wins, clip_table = winsorize_df(X_num, lower_q, upper_q)

    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    X_std = pd.DataFrame(
        std_scaler.fit_transform(X_wins), index=X_wins.index, columns=X_wins.columns
    )
    X_rob = pd.DataFrame(
        rob_scaler.fit_transform(X_wins), index=X_wins.index, columns=X_wins.columns
    )

    return {
        "only_wins": X_wins,
        "wins_std": X_std,
        "wins_rob": X_rob,
        "clip_table": clip_table,
    }


# ======================================================================================
# 7) PCA 비교 출력/저장
# ======================================================================================
def pca_compare_variants(
    X_map: Dict[str, pd.DataFrame],
    n_components: int,
    topk_loadings: int = 10,
    random_state: int = 42,
    save_dir: Optional[str] = None,
    show_plots: bool = True,
) -> Dict[str, dict]:
    """
    X_map에 있는 모든 데이터프레임에 대해 PCA를 수행하고:
    - explained variance ratio / cumulative
    - PC별 EVR 테이블
    - 로딩(가중치 벡터) 상위 feature 테이블(최대 절대 로딩 기준 topk)
    - 누적 설명분산 비교 그래프
    을 출력하고(옵션) save_dir에 저장합니다.
    """
    results: Dict[str, dict] = {}
    summary_rows = []
    pc_names = [f"PC{i}" for i in range(1, n_components + 1)]
    loading_cols = [f"PC{i}_loading" for i in range(1, n_components + 1)]

    for variant, X in X_map.items():
        pca = PCA(n_components=n_components, random_state=random_state)
        Z = pca.fit_transform(X)

        ev_table = pd.DataFrame(
            {"explained_variance_ratio": explained, "cum_explained": cum_explained},
            index=pc_names,
        )

        loadings = pd.DataFrame(
            pca.components_.T,
            index=X.columns,
            columns=loading_cols,
        )

        max_abs = loadings.abs().max(axis=1)
        top_feats = max_abs.sort_values(ascending=False).head(topk_loadings).index
        top_loadings = loadings.loc[top_feats].copy()
        top_loadings = top_loadings.loc[
            max_abs.loc[top_feats].sort_values(ascending=False).index
        ]

        results[variant] = {
            "pca": pca,
            "Z": Z,
            "explained": explained,
            "cum_explained": cum_explained,
            "ev_table": ev_table,
            "loadings": loadings,
            "top_loadings": top_loadings,
        }

        summary_rows.append(
            {
                "variant": variant,
                "n_components": n_components,
                "final_cum_explained": float(cum_explained[-1]),
            }
        )

        # 출력(노트북 스타일 print)
        print(f"\n[PCA_VARIANT={variant}] n_components={n_components}")
        print("explained variance ratio:", explained)
        print("cum explained:", cum_explained, " / final:", cum_explained[-1])

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            ev_table.to_csv(
                os.path.join(save_dir, f"pca_ev_table_{variant}.csv"),
                encoding="utf-8-sig",
            )
            top_loadings.to_csv(
                os.path.join(save_dir, f"pca_top_loadings_{variant}.csv"),
                encoding="utf-8-sig",
            )
            # 전체 로딩은 컬럼이 많을 수 있어 parquet로 저장
            loadings.to_parquet(
                os.path.join(save_dir, f"pca_loadings_{variant}.parquet")
            )

    # 비교 표 저장
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "final_cum_explained", ascending=False
    )
    if save_dir:
        summary_df.to_csv(
            os.path.join(save_dir, "pca_summary_final_cum_explained.csv"),
            encoding="utf-8-sig",
        )

    # 누적 설명분산 비교 그래프
    # (matplotlib은 import 비용이 커서 필요한 시점에만 로드)
    import matplotlib

    if (save_dir is not None) and (not show_plots):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = range(1, n_components + 1)
    plt.figure(figsize=(8, 4.5))
    for variant in X_map.keys():
        cum_explained = results[variant]["cum_explained"]
        plt.plot(
            xs,
            cum_explained,
            marker="o",
            label=f"{variant} (final={cum_explained[-1]:.3f})",
        )
    plt.title(f"PCA cumulative explained variance (compare) | n={n_components}")
    plt.xlabel("number of components")
    plt.ylabel("cumulative explained variance")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()

    if save_dir:
        plt.savefig(
            os.path.join(save_dir, f"pca_cum_explained_compare_n{n_components}.png"),
            dpi=150,
        )
    if show_plots:
        plt.show()
    else:
        plt.close()

    # EVR 비교 테이블도 저장/반환
    evr_compare = pd.DataFrame(
        {v: results[v]["explained"] for v in X_map.keys()}, index=pc_names
    )
    cum_compare = pd.DataFrame(
        {v: results[v]["cum_explained"] for v in X_map.keys()}, index=pc_names
    )

    if save_dir:
        evr_compare.to_csv(
            os.path.join(save_dir, "pca_explained_variance_ratio_compare.csv"),
            encoding="utf-8-sig",
        )
        cum_compare.to_csv(
            os.path.join(save_dir, "pca_cum_explained_compare.csv"),
            encoding="utf-8-sig",
        )

    results["_summary"] = {
        "summary_df": summary_df,
        "evr_compare": evr_compare,
        "cum_compare": cum_compare,
    }
    return results


# ======================================================================================
# Orchestrator: 1~7 한 번에 실행 + 저장
# ======================================================================================
@dataclass
class PipelineConfig:
    data_folder: str
    date: str  # "YYYY-MM-DD"
    feature_cols: Optional[List[str]]  # None이면 numeric 전체 사용
    lower_q: float = 0.01
    upper_q: float = 0.99
    corr_threshold: float = 0.95
    pca_n_components: int = 15
    topk_loadings: int = 10
    output_root: str = "outputs"
    date_col: str = "Date"
    ticker_col: str = "Ticker"


def run_pipeline_cells_1_to_7(cfg: PipelineConfig) -> Dict[str, object]:
    """
    노트북 1~7 셀에 해당하는 흐름을 한 번에 실행하고,
    결과물을 cfg.output_root 아래에 저장합니다.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.output_root, f"run_{cfg.date}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # 1) load
    df_all = load_parquets_to_df_all(
        cfg.data_folder, ticker_from_filename=True, ticker_col=cfg.ticker_col
    )
    print(f"[1] df_all loaded: shape={df_all.shape}")
    df_all.to_parquet(os.path.join(out_dir, "df_all.parquet"))

    # 2) snapshot
    snap = snapshot_by_date(
        df_all, cfg.date, date_col=cfg.date_col, ticker_col=cfg.ticker_col
    )
    print(f"[2] snapshot: date={cfg.date} shape={snap.shape}")
    snap.to_parquet(os.path.join(out_dir, f"snapshot_{cfg.date}.parquet"))

    # 3) select features
    feat_df = select_feature_columns(
        snap, cfg.feature_cols, ticker_col=cfg.ticker_col, keep_ticker=True
    )
    print(f"[3] selected features: shape={feat_df.shape}")
    feat_df.to_parquet(os.path.join(out_dir, "features_selected.parquet"))

    # 4) drop high corr (numeric 기반)
    reduced_df, dropped_cols, corr = drop_high_corr_features(
        feat_df, threshold=cfg.corr_threshold
    )
    print(
        f"[4] drop_high_corr: dropped={len(dropped_cols)} | remaining_cols={reduced_df.shape[1]}"
    )
    pd.Series(dropped_cols).to_csv(
        os.path.join(out_dir, "dropped_high_corr_cols.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    corr.to_parquet(os.path.join(out_dir, "corr_matrix_abs.parquet"))
    reduced_df.to_parquet(os.path.join(out_dir, "features_after_corr_drop.parquet"))

    # 5) inf->nan + median impute
    X_imputed_all, na_before, na_after_total = inf_to_nan_and_median_impute(reduced_df)
    print("[5] NA before (top10):")
    print(na_before.head(10))
    print("[5] NA after total:", na_after_total)

    na_before.to_csv(
        os.path.join(out_dir, "na_before_by_col.csv"), encoding="utf-8-sig"
    )
    pd.Series([na_after_total], name="na_after_total").to_csv(
        os.path.join(out_dir, "na_after_total.csv"), index=False, encoding="utf-8-sig"
    )

    # winsorize/scalers는 numeric만으로 진행 (Ticker 같은 non-numeric 제외)
    X_imputed = X_imputed_all.select_dtypes(include="number").copy()
    X_imputed.to_parquet(os.path.join(out_dir, "X_imputed_numeric.parquet"))

    # 6) winsorize + scalers
    transformed = build_winsorize_and_scalers(X_imputed, cfg.lower_q, cfg.upper_q)
    clip_table = transformed["clip_table"]
    clip_table.to_csv(
        os.path.join(out_dir, "winsorize_clip_table.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    transformed["only_wins"].to_parquet(os.path.join(out_dir, "X_only_wins.parquet"))
    transformed["wins_std"].to_parquet(os.path.join(out_dir, "X_wins_std.parquet"))
    transformed["wins_rob"].to_parquet(os.path.join(out_dir, "X_wins_rob.parquet"))

    print(f"[6] winsorize+scalers done | clip_table rows={clip_table.shape[0]}")

    # 7) PCA compare
    pca_dir = os.path.join(out_dir, "pca")
    pca_inputs = {
        "wins_std": transformed["wins_std"],
        "wins_rob": transformed["wins_rob"],
        "only_wins": transformed["only_wins"],
    }
    pca_res = pca_compare_variants(
        pca_inputs,
        n_components=cfg.pca_n_components,
        topk_loadings=cfg.topk_loadings,
        random_state=42,
        save_dir=pca_dir,
        show_plots=True,
    )
    print(f"[7] PCA compare saved to: {pca_dir}")

    # config도 저장
    pd.Series(cfg.__dict__).to_json(
        os.path.join(out_dir, "run_config.json"), force_ascii=False, indent=2
    )

    return {
        "out_dir": out_dir,
        "df_all": df_all,
        "snapshot": snap,
        "features_selected": feat_df,
        "features_after_corr_drop": reduced_df,
        "X_imputed_numeric": X_imputed,
        "transformed": transformed,
        "pca_result": pca_res,
        "dropped_high_corr_cols": dropped_cols,
        "na_before": na_before,
        "na_after_total": na_after_total,
    }


# ======================================================================================
# 8) KMeans 평가/최종 라벨 + Rolling(날짜 루프) 군집 테이블 생성
# ======================================================================================
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def evaluate_kmeans_k_range(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 15,
    n_init: int = 200,
    random_state: int = 42,
    max_iter: int = 300,
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """
    k range로 KMeans를 돌려 inertia / silhouette 등을 평가합니다.
    반환:
      - k_eval: DataFrame(k, inertia, silhouette, max_cluster_share, min_cluster_size)
      - labels_by_k: {k: labels}
    """
    rows = []
    labels_by_k: Dict[int, np.ndarray] = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        lab = km.fit_predict(X)
        inertia = km.inertia_
        sil = silhouette_score(X, lab)

        counts = pd.Series(lab).value_counts().sort_index()
        max_share = float(counts.max() / counts.sum())

        rows.append(
            {
                "k": k,
                "inertia": float(inertia),
                "silhouette": float(sil),
                "max_cluster_share": max_share,
                "min_cluster_size": int(counts.min()),
            }
        )
        labels_by_k[k] = lab

    k_eval = pd.DataFrame(rows).sort_values("k")
    return k_eval, labels_by_k


# seed 안정성 평가 (random_state 변화에 따른 라벨 변화)
def kmeans_seed_stability_report(
    X: pd.DataFrame,
    k: int,
    seeds: List[int],
    n_init: int = 20,
    max_iter: int = 300,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    같은 데이터(X), 같은 k에서 random_state(seed)만 바꿔 KMeans를 반복 실행하고,
    첫 번째 seed의 라벨을 기준(reference)으로 ARI/NMI를 계산해 안정성을 요약합니다.

    반환:
      - per_seed_df: seed별 labels + (ARI/NMI vs reference)
      - summary: ari_mean/ari_min/nmi_mean/nmi_min
    """
    if seeds is None or len(seeds) < 2:
        raise ValueError("seeds must have length >= 2")

    labels_map: Dict[int, np.ndarray] = {}
    for s in seeds:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=int(s),
        )
        labels_map[s] = km.fit_predict(X)

    ref_seed = seeds[0]
    ref_labels = labels_map[ref_seed]

    rows = []
    for s in seeds:
        lab = labels_map[s]
        if s == ref_seed:
            ari = 1.0
            nmi = 1.0
        else:
            ari = float(adjusted_rand_score(ref_labels, lab))
            nmi = float(normalized_mutual_info_score(ref_labels, lab))

        rows.append(
            {
                "seed": int(s),
                "ref_seed": int(ref_seed),
                "ari_vs_ref": ari,
                "nmi_vs_ref": nmi,
            }
        )

    per_seed_df = pd.DataFrame(rows)

    # ref(=1.0) 포함하면 평균이 살짝 올라가서, 보통은 ref 제외한 값으로 요약하는 편이 깔끔함
    non_ref = per_seed_df[per_seed_df["seed"] != ref_seed]

    summary = {
        "ari_mean": float(non_ref["ari_vs_ref"].mean()) if len(non_ref) else 1.0,
        "ari_min": float(non_ref["ari_vs_ref"].min()) if len(non_ref) else 1.0,
        "nmi_mean": float(non_ref["nmi_vs_ref"].mean()) if len(non_ref) else 1.0,
        "nmi_min": float(non_ref["nmi_vs_ref"].min()) if len(non_ref) else 1.0,
    }
    return per_seed_df, summary


def pca_then_kmeans_for_one_date(
    snap: pd.DataFrame,
    date: str,
    feature_cols: Optional[Iterable[str]],
    lower_q: float,
    upper_q: float,
    corr_drop_cols: Optional[List[str]] = None,
    corr_threshold: float = 0.95,
    variant: str = "wins_rob",  # "wins_std" / "wins_rob" / "only_wins"
    pca_n: int = 8,
    pca_target_cum: float = 0.80,
    k_min: int = 2,
    k_max: int = 15,
    n_init: int = 200,
    random_state: int = 42,
    ticker_col: str = "Ticker",
    make_plots: bool = False,
    save_dir: Optional[str] = None,
    stability_seeds: Optional[List[int]] = None,
    stability_n_init: int = 20,
) -> Dict[str, object]:
    """
    한 날짜 스냅샷에 대해:
      (선택) 상관 제거 -> inf/NaN 처리 -> winsorize+scaling -> PCA -> KMeans(k range 평가 + best_k)
    를 수행하고, 종목별 라벨을 반환합니다.

    주의:
    - 날짜별로 PCA/KMeans를 각각 fit하면, cluster id는 날짜 간 '라벨 번호'가 직접 비교되지 않을 수 있습니다.
      (라벨 permutation 문제) 그래도 "그 날짜에서의 군집 구성 변화"를 보는 데는 유용합니다.
    """
    # 3) feature 선택(티커 포함 가능)
    feat_df = select_feature_columns(
        snap, feature_cols, ticker_col=ticker_col, keep_ticker=True
    )

    # 4) (옵션 A) baseline에서 받은 corr_drop_cols가 있으면 그대로 드랍
    if corr_drop_cols is not None:
        reduced_df = feat_df.drop(columns=corr_drop_cols, errors="ignore")
        corr = None
        dropped_cols = corr_drop_cols
    else:
        # (옵션 B) 날짜별로 corr 기반 제거를 새로 계산
        reduced_df, dropped_cols, corr = drop_high_corr_features(
            feat_df, threshold=corr_threshold
        )
    # ============================================================
    # (핵심) index를 "Ticker(6자리)"로 강제해서 이후 PCA/Z_df 저장시 Ticker가 진짜 티커가 되게 함
    # - df_all이 DatetimeIndex(날짜 반복)인 구조라서, 그냥 두면 Z_df.index가 날짜가 됨
    # ============================================================
    if ticker_col in reduced_df.columns:
        t = reduced_df[ticker_col].astype(str)
    else:
        t = reduced_df.index.astype(str)

    tickers_norm = (
        t.str.strip()
        .str.replace(r"\.0$", "", regex=True)  # 70.0 같은 케이스 방지
        .str.replace(r"\.KS$", "", regex=True)  # 혹시 005930.KS가 섞이면 제거
        .str.replace(r"\.KQ$", "", regex=True)
        .str.zfill(6)  # 무조건 6자리
    )

    # 중복 ticker 방어 (중복이면 join/align에서 깨질 수 있음)
    mask = ~tickers_norm.duplicated(keep="first")
    reduced_df = reduced_df.loc[mask].copy()
    tickers_norm = tickers_norm.loc[mask]

    # 이후 파이프라인이 전부 ticker index를 쓰도록 강제
    reduced_df.index = tickers_norm.values

    # 5) inf->nan + median impute
    X_imputed_all, na_before, na_after_total = inf_to_nan_and_median_impute(reduced_df)

    # numeric만 KMeans/PCA 대상으로
    X_num = X_imputed_all.select_dtypes(include="number").copy()

    # 6) winsorize + scalers
    transformed = build_winsorize_and_scalers(X_num, lower_q, upper_q)
    X_for = transformed[variant]
    # 7) PCA
    # - play_k_means.py에서 pca_n(예: 8)을 주면:
    #   (1) 일단 PC 1~pca_n까지 PCA를 fit해서 EVR/cum/로딩을 저장하고,
    #   (2) cum_explained가 pca_target_cum(기본 0.80) 이상이 되는 '최소 차원(n_used)'만 골라
    #       그 n_used 차원으로 KMeans를 수행합니다.
    #
    #   => pca_ev_<date>.csv / pca_loadings_<date>.csv 는 항상 PC1~PC{pca_n_cap} 정보를 담고,
    #      KMeans는 그 중 n_used 차원만 사용합니다.
    pca_n_cap = int(min(pca_n, X_for.shape[1], X_for.shape[0]))
    if pca_n_cap < 1:
        raise ValueError(
            f"pca_n_cap must be >= 1 (got {pca_n_cap}). Check data/feature columns."
        )

    pca = PCA(n_components=pca_n_cap, random_state=random_state)
    Z_full = pca.fit_transform(X_for)

    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    # cum_explained >= target 인 첫 위치(1-based)
    hit = np.where(cum_explained >= pca_target_cum)[0]
    n_used = int(hit[0] + 1) if len(hit) else pca_n_cap

    # KMeans는 n_used 차원만 사용
    Z = Z_full[:, :n_used]
    Z_df = pd.DataFrame(
        Z, index=X_for.index, columns=[f"PC{i}" for i in range(1, n_used + 1)]
    )

    # ============================================================
    # (추가) K-means에 실제로 사용한 PCA 좌표(Z_df) 저장
    # - aligned 라벨 플롯을 "이 좌표"로만 그리기 위해 필요
    # ============================================================
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        z_out = Z_df.copy()
        z_out.insert(0, "Ticker", pd.Index(z_out.index).astype(str).str.zfill(6))
        z_out.to_csv(
            os.path.join(save_dir, f"pca_coords_used_{date}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # --- PCA loadings + explained variance (for pca_loadings_<date>.csv)
    loading_cols = [f"PC{i}_loading" for i in range(1, pca_n_cap + 1)]
    loadings = pd.DataFrame(
        pca.components_.T, index=X_for.columns, columns=loading_cols
    )
    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    # 한 파일에서 같이 보기 쉽게: (상단 2행) EVR / CUM + (하단) feature loadings
    evr_row = pd.Series(explained, index=loading_cols, name="explained_variance_ratio")
    cum_row = pd.Series(cum_explained, index=loading_cols, name="cum_explained")
    pca_loadings_table = pd.concat(
        [evr_row.to_frame().T, cum_row.to_frame().T, loadings], axis=0
    )

    # 8) KMeans 평가
    k_eval, labels_by_k = evaluate_kmeans_k_range(
        Z_df, k_min=k_min, k_max=k_max, n_init=n_init, random_state=random_state
    )
    best_k = int(k_eval.sort_values("silhouette", ascending=False).iloc[0]["k"])

    km_final = KMeans(
        n_clusters=best_k,
        init="k-means++",
        n_init=n_init,
        max_iter=300,
        random_state=random_state,
    )
    final_labels = km_final.fit_predict(Z_df)

    # ============================================================
    # (추가) seed 반복 안정성(ARI/NMI) - 같은 Z_df, 같은 best_k에서 seed만 변경
    # ============================================================
    stability_df = None
    stability_summary = None
    if stability_seeds is not None and len(stability_seeds) >= 2:
        stability_df, stability_summary = kmeans_seed_stability_report(
            X=Z_df,
            k=best_k,
            seeds=stability_seeds,
            n_init=stability_n_init,
            max_iter=300,
        )
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            stability_df.to_csv(
                os.path.join(save_dir, f"stability_seed_{date}.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    # ✅ (추가) 날짜별 centroid 저장 (PCA 공간)
    centroids = pd.DataFrame(
        km_final.cluster_centers_,
        columns=[f"PC{i}" for i in range(1, Z_df.shape[1] + 1)],
    )
    centroids.insert(0, "cluster", range(centroids.shape[0]))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        centroids.to_csv(
            os.path.join(save_dir, f"centroids_{date}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
    centroids.to_csv(
        os.path.join(save_dir, f"centroids_{date}.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    tickers = (
        pd.Index(Z_df.index)
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
        .values
    )

    label_df = pd.DataFrame(
        {
            "date": pd.to_datetime(date).date(),
            ticker_col: tickers,
            "cluster": final_labels,
        }
    )

    # 저장 옵션
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # 요약 저장
        k_eval.to_csv(
            os.path.join(save_dir, f"k_eval_{date}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame({"dropped_corr_cols": dropped_cols}).to_csv(
            os.path.join(save_dir, f"corr_dropped_cols_{date}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        transformed["clip_table"].to_csv(
            os.path.join(save_dir, f"winsor_clip_table_{date}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        # PCA 요약
        ev = pd.DataFrame(
            {
                "explained_variance_ratio": pca.explained_variance_ratio_,
                "cum_explained": np.cumsum(pca.explained_variance_ratio_),
            }
        )
        ev.to_csv(
            os.path.join(save_dir, f"pca_ev_{date}.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        pca_loadings_table.to_csv(
            os.path.join(save_dir, f"pca_loadings_{date}.csv"), encoding="utf-8-sig"
        )
        label_df.to_csv(
            os.path.join(save_dir, f"labels_{date}.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        # 그래프 저장(옵션)
        if make_plots:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 4))
            plt.plot(k_eval["k"], k_eval["inertia"], marker="o")
            plt.title(
                f"Elbow | {date} | variant={variant} | PCA cap={pca_n} | n_used={n_used}"
            )
            plt.xlabel("k")
            plt.ylabel("inertia")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"elbow_{date}.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(7, 4))
            plt.plot(k_eval["k"], k_eval["silhouette"], marker="o")
            plt.title(
                f"Silhouette | {date} | variant={variant} | PCA cap={pca_n} | n_used={n_used}"
            )
            plt.xlabel("k")
            plt.ylabel("silhouette")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"silhouette_{date}.png"), dpi=150)
            plt.close()

    return {
        "date": date,
        "variant": variant,
        "pca_n": pca_n,
        "pca_n_cap": pca_n_cap,
        "pca_n_used": n_used,
        "best_k": best_k,
        "k_eval": k_eval,
        "labels": label_df,
        "Z_df": Z_df,
        "pca_model": pca,
        "kmeans_model": km_final,
        "na_before": na_before,
        "na_after_total": na_after_total,
        "corr_dropped_cols": dropped_cols,
        "stability_seed_df": stability_df,
        "stability_summary": stability_summary,
    }


def rolling_cluster_table(
    df_all: pd.DataFrame,
    dates: List[str],
    feature_cols: Optional[List[str]],
    lower_q: float,
    upper_q: float,
    variant: str = "wins_rob",
    pca_n: int = 8,
    k_min: int = 2,
    k_max: int = 15,
    n_init: int = 200,
    random_state: int = 42,
    corr_threshold: float = 0.95,
    corr_drop_mode: str = "baseline",  # "baseline" or "per_date" or "none"
    date_col: str = "Date",
    ticker_col: str = "Ticker",
    output_root: str = "outputs",
    save_per_date: bool = True,
    make_plots: bool = False,
    stability_seeds: Optional[List[int]] = None,
    stability_n_init: int = 20,
) -> Dict[str, object]:
    """
    여러 날짜에 대해 반복(rolling)으로:
      snapshot -> (전처리+PCA+KMeans) -> label_df
    를 수행하고,
      - long 형태(label_long): [date, Ticker, cluster]
      - wide 형태(label_wide): index=Ticker, columns=date, values=cluster
    를 저장/반환합니다.

    corr_drop_mode:
      - "baseline": 첫 날짜에서 corr 기반 제거한 컬럼을 모든 날짜에 동일 적용(권장: 피처 일관성)
      - "per_date": 날짜별로 corr 기반 제거를 매번 새로 계산(피처 수가 날짜마다 달라질 수 있음)
      - "none": 상관 제거 생략
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    roll_dir = os.path.join(output_root, f"rolling_{dates[0]}_to_{dates[-1]}_{ts}")
    os.makedirs(roll_dir, exist_ok=True)

    # baseline corr drop columns 계산(옵션)
    baseline_drop_cols: Optional[List[str]] = None
    if corr_drop_mode == "baseline":
        snap0 = snapshot_by_date(
            df_all, dates[0], date_col=date_col, ticker_col=ticker_col
        )
        feat0 = select_feature_columns(
            snap0, feature_cols, ticker_col=ticker_col, keep_ticker=True
        )
        _, dropped0, _corr0 = drop_high_corr_features(feat0, threshold=corr_threshold)
        baseline_drop_cols = dropped0
        pd.DataFrame({"baseline_corr_dropped_cols": baseline_drop_cols}).to_csv(
            os.path.join(roll_dir, "baseline_corr_dropped_cols.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    label_long_list: List[pd.DataFrame] = []
    meta_rows: List[dict] = []
    stability_rows: List[dict] = []

    for d in dates:
        snap = snapshot_by_date(df_all, d, date_col=date_col, ticker_col=ticker_col)

        if corr_drop_mode == "none":
            corr_drop_cols = []
            # 구현 단순화를 위해 corr_drop_cols를 빈 리스트로 주고,
            # 실제 drop은 "corr_drop_cols is not None" 로직에 의해 no-op
            corr_drop_cols_for_func = []
        elif corr_drop_mode == "baseline":
            corr_drop_cols_for_func = baseline_drop_cols
        else:  # "per_date"
            corr_drop_cols_for_func = None

        per_date_dir = os.path.join(roll_dir, "per_date") if save_per_date else None
        if per_date_dir:
            os.makedirs(per_date_dir, exist_ok=True)

        res = pca_then_kmeans_for_one_date(
            snap=snap,
            date=d,
            feature_cols=feature_cols,
            lower_q=lower_q,
            upper_q=upper_q,
            corr_drop_cols=corr_drop_cols_for_func,
            corr_threshold=corr_threshold,
            variant=variant,
            pca_n=pca_n,
            k_min=k_min,
            k_max=k_max,
            n_init=n_init,
            random_state=random_state,
            ticker_col=ticker_col,
            make_plots=make_plots,
            save_dir=(os.path.join(per_date_dir, d) if per_date_dir else None),
            stability_seeds=stability_seeds,
            stability_n_init=stability_n_init,
        )

        label_long_list.append(res["labels"])
        meta_rows.append(
            {
                "date": d,
                "variant": variant,
                "pca_n": int(res.get("pca_n_used", pca_n)),
                "best_k": res["best_k"],
                "best_silhouette": float(
                    res["k_eval"]
                    .sort_values("silhouette", ascending=False)
                    .iloc[0]["silhouette"]
                ),
            }
        )

        if res.get("stability_summary") is not None:
            ss = res["stability_summary"]
            stability_rows.append(
                {
                    "date": d,
                    "variant": variant,
                    "n_used": int(res.get("pca_n_used", pca_n)),
                    "best_k": int(res["best_k"]),
                    "ari_mean": float(ss["ari_mean"]),
                    "ari_min": float(ss["ari_min"]),
                    "nmi_mean": float(ss["nmi_mean"]),
                    "nmi_min": float(ss["nmi_min"]),
                    "n_seeds": int(len(stability_seeds)),
                }
            )

        print(
            f"[ROLL] {d} done | n_used={int(res.get('pca_n_used', pca_n))} | "
            f"best_k={int(res['best_k'])} | best_sil={meta_rows[-1]['best_silhouette']:.4f}"
        )

    label_long = pd.concat(label_long_list, axis=0, ignore_index=True)

    # wide table: ticker x date -> cluster
    label_wide = label_long.pivot_table(
        index=ticker_col, columns="date", values="cluster", aggfunc="first"
    ).sort_index()

    meta_df = pd.DataFrame(meta_rows)

    # 저장
    label_long.to_csv(
        os.path.join(roll_dir, "cluster_labels_long.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    label_wide.to_csv(
        os.path.join(roll_dir, "cluster_labels_wide.csv"), encoding="utf-8-sig"
    )
    meta_df.to_csv(
        os.path.join(roll_dir, "rolling_meta_by_date.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    label_long.to_parquet(os.path.join(roll_dir, "cluster_labels_long.parquet"))
    label_wide.to_parquet(os.path.join(roll_dir, "cluster_labels_wide.parquet"))
    meta_df.to_parquet(os.path.join(roll_dir, "rolling_meta_by_date.parquet"))

    # 실행 설정 저장
    run_cfg = {
        "dates": dates,
        "feature_cols": feature_cols,
        "lower_q": lower_q,
        "upper_q": upper_q,
        "variant": variant,
        "pca_n": pca_n,
        "k_min": k_min,
        "k_max": k_max,
        "n_init": n_init,
        "random_state": random_state,
        "corr_threshold": corr_threshold,
        "corr_drop_mode": corr_drop_mode,
    }
    pd.Series(run_cfg).to_json(
        os.path.join(roll_dir, "rolling_run_config.json"), force_ascii=False, indent=2
    )

    # 안정성 요약 저장(있는 경우)
    if len(stability_rows) > 0:
        stab_df = pd.DataFrame(stability_rows)
        stab_df.to_csv(
            os.path.join(roll_dir, "rolling_stability_by_date.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    return {
        "roll_dir": roll_dir,
        "label_long": label_long,
        "label_wide": label_wide,
        "meta_by_date": meta_df,
    }


# if __name__ == "__main__":
#     # 예시 실행 (직접 실행할 때만)
#     cfg = PipelineConfig(
#         data_folder=r"C:\path\to\your\data_folder",
#         date="2024-12-30",
#         feature_cols=None,  # None이면 numeric 전부
#         lower_q=0.01,
#         upper_q=0.99,
#         corr_threshold=0.95,
#         pca_n_components=15,
#         topk_loadings=10,
#         output_root="outputs",
#     )
#     run_pipeline_cells_1_to_7(cfg)
#         output_root="outputs",
#     )
#     run_pipeline_cells_1_to_7(cfg)
#         output_root="outputs",
#     )
#     run_pipeline_cells_1_to_7(cfg)
#         output_root="outputs",
#     )
#     run_pipeline_cells_1_to_7(cfg)
#         output_root="outputs",
#     )
#     run_pipeline_cells_1_to_7(cfg)
#         output_root="outputs",
#     )
#     run_pipeline_cells_1_to_7(cfg)
#         output_root="outputs",
#     )
#     run_pipeline_cells_1_to_7(cfg)
