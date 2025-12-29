"""GMM 클러스터링 실행 진입점."""

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
    """팀 입력 포맷이 미확정이어도 최대한 안전하게 표준화.

    - MultiIndex(종목코드, date)면 reset_index로 컬럼화
    - date/Date(유닉스타임 가능) → datetime
    - Year 없으면 Date로 생성
    - Ticker 없으면 Code/종목코드로 폴백
    """

    out = df

    # 0) MultiIndex → columns
    if isinstance(out.index, pd.MultiIndex):
        out = out.reset_index()

    # 1) Date 컬럼 탐색/정규화
    if "Date" not in out.columns:
        for cand in ("date", "DATE", "timestamp", "ts", "time", "Time", "일자", "날짜", "거래일"):
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

    # 2) Year 생성
    if "Year" not in out.columns and "Date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["Date"]):
        out["Year"] = out["Date"].dt.year

    # 3) Ticker 폴백
    if "Ticker" not in out.columns:
        if "Code" in out.columns:
            out["Ticker"] = out["Code"]
        elif "종목코드" in out.columns:
            out["Ticker"] = out["종목코드"]
        else:
            out["Ticker"] = "N/A"

    return out


class GMM:
    """얇은 오케스트레이션 래퍼 (기존 모듈 호출 중심)."""

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
