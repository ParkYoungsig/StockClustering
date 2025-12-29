"""GMM 클러스터링 파이프라인 실행 진입점."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from src.gmm import config
from src.gmm.data_loader import convert_df_to_snapshots

from src.gmm.orchestrator import run_gmm_pipeline

logger = logging.getLogger(__name__)


def _normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """입력 데이터를 안전하게 정규화한다 (인덱스 평탄화, Date 변환, Year/Ticker 보강)."""

    out = df

    # MultiIndex가 있으면 평탄화
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()

    # Date 컬럼 정규화
    if "Date" not in out.columns:
        for cand in (
            "date",
            "DATE",
            "timestamp",
            "ts",
            "time",
            "Time",
            "일자",
            "날짜",
            "거래일",
        ):
            if cand in out.columns:
                out = out.rename(columns={cand: "Date"})
                break

    if "Date" in out.columns:
        if not pd.api.types.is_datetime64_any_dtype(out["Date"]):
            s = pd.to_numeric(out["Date"], errors="coerce")
            med = s.dropna().median()
            if pd.notna(med) and med > 1e12:
                out["Date"] = pd.to_datetime(out["Date"], unit="ms", errors="coerce")
            elif pd.notna(med) and med > 1e9:
                out["Date"] = pd.to_datetime(out["Date"], unit="s", errors="coerce")
            else:
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    # Year 파생
    if (
        "Year" not in out.columns
        and "Date" in out.columns
        and pd.api.types.is_datetime64_any_dtype(out["Date"])
    ):
        out["Year"] = out["Date"].dt.year

    # Ticker 보강
    if "Ticker" not in out.columns:
        if "Code" in out.columns:
            out["Ticker"] = out["Code"]
        elif "종목코드" in out.columns:
            out["Ticker"] = out["종목코드"]
        else:
            out["Ticker"] = "N/A"

    return out


class GMM:
    """오케스트레이션된 GMM 파이프라인을 감싸는 얇은 래퍼."""

    def __init__(self, df: pd.DataFrame | None = None, results_dir: Path | None = None):
        self.results_dir = results_dir or Path(config.DEFAULT_RESULTS_DIR_NAME)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.load_stats: Dict | None = None

        if df is not None and not df.empty:
            try:
                self.df = convert_df_to_snapshots(
                    df,
                    freq=config.SNAPSHOT_FREQ,
                    start_year=config.START_YEAR,
                    end_year=config.END_YEAR,
                )
            except Exception as e:
                logger.warning(f"스냅샷 변환 실패, 원본 사용: {e}")
                self.df = df
        else:
            self.df = df

        # 입력 df 표준화: MultiIndex/유닉스타임/Year/Ticker 폴백
        if self.df is not None and not self.df.empty:
            self.df = _normalize_input_df(self.df)

    def run(self, manual_k: int | None = 4) -> str:
        return run_gmm_pipeline(
            df=self.df,
            results_dir=self.results_dir,
            load_stats=self.load_stats,
            manual_k=manual_k,
        )
