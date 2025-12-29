"""GMM 파이프라인용 단순 데이터 로더(정돈 버전).

전제:
- merged_stock_data.parquet에 Date/Ticker/Name 컬럼이 이미 존재
- 데이터 정합성 확보를 가정하고 최소한의 방어 로직만 유지
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from datasets import load_dataset  # type: ignore

from src.gmm import config

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")
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

DESIRED_COLUMNS: list[str] = sorted({"Date", "Ticker", "Name", *FEATURE_COLUMNS})


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


def convert_df_to_snapshots(
    df: pd.DataFrame,
    *,
    freq: str = config.SNAPSHOT_FREQ,
    start_year: int = config.START_YEAR,
    end_year: int | None = config.END_YEAR,
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

    return _clean_features(snapshots)


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """피처 결측 제거 후 분위수 클리핑 + z-score 스케일링을 수행합니다.

    클리핑/스케일링은 Ticker-Year 그룹별로 적용해 교차섹션 왜곡을 완화하고,
    그룹 크기가 작아 분산이 0이면 해당 컬럼은 그대로 둡니다.
    """

    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if not features:
        return df

    cleaned = df.dropna(subset=features).copy()

    frames = []
    group_cols = (
        ["Ticker", "Year"] if {"Ticker", "Year"}.issubset(cleaned.columns) else []
    )

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
    start_year: int = config.START_YEAR,
    end_year: int | None = config.END_YEAR,
    fallback_days: int = config.FALLBACK_DAYS,  # kept for signature compatibility
    freq: str = config.SNAPSHOT_FREQ,
) -> Tuple[pd.DataFrame, Dict]:
    """로컬 병합본(우선) 또는 HF 병합본을 읽어 스냅샷 데이터프레임을 반환합니다."""

    t0 = time.perf_counter()

    try:
        df_raw = _load_local(data_dir)
        source = "local"
    except FileNotFoundError:
        logger.info("로컬 병합 파일 없음 → Hugging Face에서 로드")
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
