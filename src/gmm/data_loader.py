"""GMM 파이프라인용 데이터 로더."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from config import DEFAULT_RESULTS_DIR_NAME, DEFAULT_DATA_DIR_NAME
from config import SNAPSHOT_FREQ, START_YEAR, END_YEAR, FALLBACK_DAYS, K_RANGE
from config import (
    GMM_COVARIANCE_TYPE,
    GMM_N_INIT,
    GMM_MAX_ITER,
    GMM_REG_COVAR,
    GMM_ALIGN_METRIC,
)
from config import MIN_CLUSTER_FRAC, CORR_THRESHOLD, MAX_MISSING_RATIO
from config import (
    UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST,
    CLUSTER_NAMES,
    CLUSTER_INTERPRETATIONS,
    CLUSTER_COLORS,
)

logger = logging.getLogger(__name__)

# NOTE: config 모듈을 직접 참조하지 않고, src/config.py에서 import한 값을 사용합니다.
DEFAULT_DATA_DIR = Path(DEFAULT_DATA_DIR_NAME)
HF_REPO_ID = "yumin99/stock-clustering-data"
HF_MERGED_FILE = "merged_stock_data.parquet"

FEATURE_COLUMNS = [
    "Return_120d",
    "ADX_14",
    "Disparity_60d",
    "vol_60_sqrt252",
    "NATR",
    "Sharpe_60d",
    "Sortino_60d",
    "Zscore_60d",
    "MFI_14",
    "Return_20d",
]


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """참고 스크립트의 컬럼 우선순위를 그대로 적용."""

    priority_cols = [
        "Date",
        "Ticker",
        "Code",
        "Name",
        "종목명",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Adj Close",
        "시가",
        "고가",
        "저가",
        "종가",
        "거래량",
        "Change",
        "Chg_Pct",
        "등락률",
        "Marcap",
        "상장시가총액",
    ]
    existing = [c for c in priority_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]


def ensure_date_ticker(
    df: pd.DataFrame, filename_stem: str | None = None
) -> pd.DataFrame:
    """Date/Ticker를 표준 컬럼으로 보장합니다.

    - MultiIndex면 reset_index()로 컬럼화
    - Ticker가 없고 filename_stem이 있으면 파일명에서 티커를 보강
    """

    out = df.copy()

    if isinstance(out.index, pd.MultiIndex):
        idx_names = list(out.index.names)
        out = out.reset_index()
        rename_map: Dict[str, str] = {}
        if len(idx_names) >= 1:
            rename_map[idx_names[0] or out.columns[0]] = "Date"
        if len(idx_names) >= 2:
            rename_map[idx_names[1] or out.columns[1]] = "Ticker"
        out = out.rename(columns=rename_map)
    else:
        if "Date" not in out.columns:
            out = out.reset_index()
        if "index" in out.columns:
            out = out.rename(columns={"index": "Date"})

    if "Ticker" not in out.columns and filename_stem is not None:
        code_str = (
            filename_stem.split("_", 1)[0] if "_" in filename_stem else filename_stem
        )
        out["Ticker"] = code_str

    return out


def _merge_local_raw_files(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """로컬 원본 parquet들을 참고 스크립트 방식으로 병합하여 메모리로 반환.

    - 멀티인덱스는 풀어서 Date/Ticker를 강제 생성
    - 종목명/종목코드 컬럼을 Name/Ticker로 표준화
    - 컬럼 순서를 참고 스크립트와 동일하게 정렬
    - merged_stock_data.parquet과 merged_original_index.parquet은 스킵
    """

    # 병합본 파일은 건너뛰고 원본만 병합
    skip_names = {HF_MERGED_FILE}
    files = [f for f in data_dir.glob("*.parquet") if f.name not in skip_names]

    if not files:
        raise FileNotFoundError("로컬 원본 parquet 파일이 없습니다.")

    frames_flat: list[pd.DataFrame] = []

    for idx, file in enumerate(files):
        try:
            df = pd.read_parquet(file)
            df_flat = df.copy()

            # 1) 컬럼명 변경 (종목명/종목코드 → Name/Ticker)
            df_flat = df_flat.rename(columns={"종목명": "Name", "종목코드": "Ticker"})

            # 2) Date/Ticker 강제 생성 (멀티인덱스 포함)
            df_flat = ensure_date_ticker(df_flat, file.stem)

            # 3) 컬럼 순서 정렬
            df_flat = _reorder_columns(df_flat)

            frames_flat.append(df_flat)

            if (idx + 1) % 100 == 0:
                logger.info("원본 병합 진행 중: %s/%s", idx + 1, len(files))
        except Exception as e:  # noqa: BLE001
            logger.warning("원본 병합 중 오류 (%s): %s", file.name, e)

    if not frames_flat:
        raise ValueError("원본 병합 결과가 비었습니다.")

    merged_flat = pd.concat(frames_flat, ignore_index=True)
    logger.info(
        "로컬 원본 병합 완료: %s행, Date=%s, Ticker=%s",
        len(merged_flat),
        "Date" in merged_flat.columns,
        "Ticker" in merged_flat.columns,
    )
    return merged_flat


def _load_local(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    fp = data_dir / HF_MERGED_FILE
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_parquet(fp)
    logger.info(
        "로컬 로드: %s행, %s컬럼 | Date: %s | Ticker: %s",
        len(df),
        len(df.columns),
        "Date" in df.columns,
        "Ticker" in df.columns,
    )
    logger.debug("로컬 컬럼: %s", list(df.columns))
    return df


def _load_from_huggingface() -> pd.DataFrame:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise ModuleNotFoundError(
            "Hugging Face에서 데이터를 로드하려면 'datasets' 패키지가 필요합니다. "
            "로컬 data 폴더에 parquet이 있으면 datasets 없이도 실행됩니다. "
            "설치: pip install datasets"
        ) from e

    uri = f"hf://datasets/{HF_REPO_ID}/{HF_MERGED_FILE}"
    ds = load_dataset("parquet", data_files=uri, split="train")
    df = ds.to_pandas()
    logger.info(
        "HF 로드: %s행, %s컬럼 | Date: %s | Ticker: %s",
        len(df),
        len(df.columns),
        "Date" in df.columns,
        "Ticker" in df.columns,
    )
    logger.debug("HF 컬럼: %s", list(df.columns))
    return df


def _load_local_raw_merged(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """완성본이 없을 때 로컬 원본 parquet을 즉석 병합하여 로드."""

    df = _merge_local_raw_files(data_dir)
    logger.info(
        "로컬 원본 병합 로드: %s행, %s컬럼 | Date: %s | Ticker: %s",
        len(df),
        len(df.columns),
        "Date" in df.columns,
        "Ticker" in df.columns,
    )
    logger.debug("로컬 원본 병합 컬럼: %s", list(df.columns))
    return df


def convert_df_to_snapshots(
    df: pd.DataFrame,
    *,
    freq: str = SNAPSHOT_FREQ,
    start_year: int = START_YEAR,
    end_year: int | None = END_YEAR,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    # 최소 정규화: 소문자 컬럼명을 표준 컬럼명으로 맞춤
    col_map = {c.lower(): c for c in out.columns}
    if "date" in col_map and "Date" not in out.columns:
        out = out.rename(columns={col_map["date"]: "Date"})
    if "ticker" in col_map and "Ticker" not in out.columns:
        out = out.rename(columns={col_map["ticker"]: "Ticker"})
    if "name" in col_map and "Name" not in out.columns:
        out = out.rename(columns={col_map["name"]: "Name"})

    # NATR_14만 있을 때 NATR 컬럼 보강
    if "NATR" not in out.columns and "NATR_14" in out.columns:
        out["NATR"] = out["NATR_14"]

    if "Date" not in out.columns:
        raise KeyError(f"Date 컬럼이 없습니다. 사용 가능 컬럼: {list(out.columns)}")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month

    year_end = end_year if end_year is not None else 9999
    out = out.loc[(out["Year"] >= start_year) & (out["Year"] <= year_end)]

    frames = []
    freq_upper = freq.upper()
    for _, g in out.groupby("Ticker"):
        g = g.sort_values("Date")
        if freq_upper == "M":
            g["YearMonth"] = g["Date"].dt.to_period("M")
            g = g.drop_duplicates(subset=["YearMonth"], keep="last")
        else:
            g = g.drop_duplicates(subset=["Year"], keep="last")
        # 필수 피처 결측 제거 (존재하는 컬럼만 대상)
        present_feats = [c for c in FEATURE_COLUMNS if c in g.columns]
        if present_feats:
            g = g.dropna(subset=present_feats)
        frames.append(g)

    if not frames:
        return pd.DataFrame()

    snapshots = pd.concat(frames, ignore_index=True)

    # 스냅샷 기준 컬럼 일관화: 월 스냅샷이면 YearMonth/Month를 명시적으로 포함
    snapshots["Year"] = snapshots["Date"].dt.year
    snapshots["Month"] = snapshots["Date"].dt.month
    if freq_upper == "M":
        snapshots["YearMonth"] = snapshots["Date"].dt.to_period("M").astype(str)
    elif "YearMonth" in snapshots.columns:
        # 연말 스냅샷만 있는 경우 혼선을 막기 위해 YearMonth 제거
        snapshots = snapshots.drop(columns=["YearMonth"])

    return _clean_features(snapshots)


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """피처 결측 제거 후 분위수 클리핑 + z-score 스케일링을 수행합니다.

    클리핑/스케일링은 기본적으로 Year 그룹별로 적용합니다.
    (Ticker-Year 단위는 그룹 수가 너무 많아 대용량에서 매우 느려질 수 있음)
    """

    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not features:
        return df

    cleaned = df.dropna(subset=features).copy()

    frames = []
    group_cols = ["Year"] if "Year" in cleaned.columns else []

    if group_cols:
        for _, g in cleaned.groupby(group_cols):
            g = _clip_and_scale(g, features)
            frames.append(g)
        return pd.concat(frames, ignore_index=True) if frames else cleaned

    # fallback: 전체 기준
    return _clip_and_scale(cleaned, features)


def _clip_and_scale(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in features:
        q = out[col].quantile([0.01, 0.99])
        q_low = q.iloc[0] if not pd.isna(q.iloc[0]) else None
        q_hi = q.iloc[1] if not pd.isna(q.iloc[1]) else None
        if q_low is not None and q_hi is not None:
            out[col] = out[col].clip(q_low, q_hi)

        mean = out[col].mean()
        std = out[col].std(ddof=0)
        if std and std > 0:
            out[col] = (out[col] - mean) / std
    return out


def load_snapshots(
    data_dir: Path = DEFAULT_DATA_DIR,
    start_year: int = START_YEAR,
    end_year: int | None = END_YEAR,
    fallback_days: int = FALLBACK_DAYS,  # kept for signature compatibility
    freq: str = SNAPSHOT_FREQ,
) -> Tuple[pd.DataFrame, Dict]:
    """3단 방어 로직으로 스냅샷 데이터프레임을 반환합니다.

    우선순위:
    1) 로컬 완성본 merged_stock_data.parquet
    2) 로컬 원본 parquet 즉석 병합 (_merge_local_raw_files)
    3) Hugging Face 병합본 다운로드
    """

    t0 = time.perf_counter()
    source = ""

    try:
        df_raw = _load_local(data_dir)
        source = "local-merged"
    except FileNotFoundError:
        logger.info("로컬 완성본 없음 → 로컬 원본 병합 시도")
        try:
            df_raw = _load_local_raw_merged(data_dir)
            source = "local-raw-merged"
        except FileNotFoundError:
            logger.info("로컬 원본 파일 없음 → Hugging Face에서 로드")
            df_raw = _load_from_huggingface()
            source = "huggingface"

    snapshots_df = convert_df_to_snapshots(
        df_raw,
        freq=freq,
        start_year=start_year,
        end_year=end_year,
    )

    if snapshots_df.empty:
        raise ValueError("스냅샷 결과가 비어 있습니다. 입력 데이터를 확인하세요.")

    stats = {
        "source": source,
        "rows_raw": len(df_raw),
        "rows_snapshots": len(snapshots_df),
        "start_year": start_year,
        "end_year": snapshots_df["Year"].max(),
        "frequency": freq.upper(),
        "tickers_loaded": sorted(snapshots_df["Ticker"].unique()),
    }

    t_total = time.perf_counter() - t0
    logger.info(
        "데이터 로드 완료: %s행 → %s 스냅샷 (%s), %.3fs",
        len(df_raw),
        len(snapshots_df),
        source,
        t_total,
    )
    return snapshots_df, stats
