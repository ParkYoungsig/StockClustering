"""GMM 파이프라인 입력 데이터 로더.

`data_dir` 내 종목별 Parquet 파일을 읽어 연말/월말 스냅샷으로 리샘플링하고,
필수 피처 유효성 검사 및 간단한 파생변수(NATR) 보강을 수행합니다.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List, Tuple
from pathlib import Path

import pandas as pd

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pq = None

from src.gmm import config

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")

# 분석에 사용할 기술적 지표 컬럼 정의
FEATURE_COLUMNS = [
    "Return_120d",  # 120일 수익률
    "ADX_14",  # 추세 강도
    "Disparity_60d",  # 60일 이격도
    "vol_60_sqrt252",  # 연환산 변동성
    "NATR",  # 정규화된 ATR (변동성 지표)
    "Sharpe_60d",  # 샤프 지수
    "Sortino_60d",  # 소르티노 지수
    "Zscore_60d",  # 표준점수
    "RSI_14",  # 상대강도지수
    "Return_20d",  # 20일 수익률
]

# 최소로 읽을 컬럼 집합 (없는 컬럼은 자동 제외)
_META_COLUMNS: list[str] = [
    "Date",
    "Close",
    "ATR_14",
    "Code",
    "종목코드",
    "Ticker",
    "Name",
]
DESIRED_COLUMNS: list[str] = sorted(set(FEATURE_COLUMNS + _META_COLUMNS))


def convert_df_to_snapshots(
    df: pd.DataFrame,
    *,
    freq: str = config.SNAPSHOT_FREQ,
    start_year: int = config.START_YEAR,
    end_year: int | None = config.END_YEAR,
) -> pd.DataFrame:
    """DataFrame 입력을 월말/연말 스냅샷으로 변환합니다.

    - Date/Year/Month 보강 후 연도 범위 필터링
    - freq에 따라 YearMonth 또는 Year 기준으로 중복 제거
    - Ticker/Name 기본 보강, NATR 보강, 필수 피처 유효성 검사
    """

    if df.empty:
        return df

    out = df.copy()

    # Date 보강
    if "Date" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "Date"})
        else:
            raise ValueError("DataFrame에 Date 컬럼이 없습니다.")

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    # Ticker/Name 보강 (전역)
    if "Ticker" not in out.columns:
        if "Code" in out.columns:
            out["Ticker"] = out["Code"]
        elif "종목코드" in out.columns:
            out["Ticker"] = out["종목코드"]
        else:
            out["Ticker"] = "N/A"
    if "Name" not in out.columns:
        out["Name"] = out.get("Ticker", "N/A")

    # NATR 보강 (데이터에 없을 때만)
    if "ATR_14" in out.columns and "Close" in out.columns:
        atr = pd.to_numeric(out["ATR_14"], errors="coerce")
        close = pd.to_numeric(out["Close"], errors="coerce").mask(lambda x: x <= 0)
        for natr_col in ("NATR_14", "NATR"):
            if natr_col in FEATURE_COLUMNS and natr_col not in out.columns:
                out[natr_col] = (atr / close) * 100

    frames: list[pd.DataFrame] = []
    freq_upper = freq.upper()
    year_end = end_year if end_year else 9999

    for _, g in out.groupby("Ticker"):
        g = g.copy()
        g["Year"] = g["Date"].dt.year
        g["Month"] = g["Date"].dt.month

        mask = (g["Year"] >= start_year) & (g["Year"] <= year_end)
        g = g.loc[mask].sort_values("Date")
        if g.empty:
            continue

        if freq_upper == "M":
            g["YearMonth"] = g["Date"].dt.to_period("M")
            g = g.drop_duplicates(subset=["YearMonth"], keep="last")
        else:
            g = g.drop_duplicates(subset=["Year"], keep="last")

        # 필수 피처 유효성 검사
        needed_cols = [c for c in FEATURE_COLUMNS if c in g.columns]
        if needed_cols:
            g = g.dropna(subset=needed_cols)
        if not g.empty:
            frames.append(g)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_snapshots(
    data_dir: Path = DEFAULT_DATA_DIR,
    start_year: int = 2015,
    end_year: int | None = None,
    fallback_days: int = 7,
    freq: str = config.SNAPSHOT_FREQ,
) -> Tuple[pd.DataFrame, Dict]:
    """
    개별 종목 파일(Parquet)을 로드하여 시계열 스냅샷 데이터를 구성합니다.

    [기능]
    - data_dir 내의 모든 .parquet 파일을 탐색합니다.
    - 각 종목별로 연말(Year-end) 또는 월말(Month-end) 기준의 데이터를 추출합니다.
    - 필수 Feature 컬럼 존재 여부를 검증하고 결측치를 처리합니다.

    Args:
        data_dir (Path): 데이터 파일 경로
        start_year (int): 추출 시작 연도
        end_year (int | None): 추출 종료 연도 (None인 경우 데이터 끝까지)
        freq (str): 스냅샷 주기 ("Y": 연말, "M": 월말)

    Returns:
        Tuple[pd.DataFrame, Dict]:
            - 병합된 전체 스냅샷 데이터프레임
            - 로드 통계 정보 (파일 수, 로드된 티커 수, 제외된 티커 등)
    """

    t0 = time.perf_counter()

    # 1. 파일 목록 탐색 (병합 파일 우선)
    merged_path = data_dir / "merged_stock_data.parquet"
    files = list(data_dir.glob("*.parquet"))
    if merged_path.exists():
        files = [merged_path]

    if not files:
        raise FileNotFoundError(f"경로에서 parquet 파일을 찾을 수 없습니다: {data_dir}")

    logger.info(f"총 {len(files)}개 파일 발견. 데이터 로드 시작 (주기: {freq})...")

    # 1-1. 병합 파일 처리 경로 (ticker 컬럼 기반 스냅샷 변환)
    if len(files) == 1 and files[0] == merged_path:
        fp = merged_path
        logger.info("병합 파케이 파일 감지: merged_stock_data.parquet (티커별 스냅샷 변환)")

        columns_to_load: Iterable[str] | None = None
        if pq is not None:
            try:
                schema = pq.read_schema(fp)
                available = set(schema.names)
                columns_to_load = [c for c in DESIRED_COLUMNS if c in available]
            except Exception:
                columns_to_load = None

        df_all = pd.read_parquet(fp, columns=columns_to_load)
        snapshots_df = convert_df_to_snapshots(
            df_all,
            freq=freq,
            start_year=start_year,
            end_year=end_year,
        )

        if snapshots_df.empty:
            raise ValueError("병합 파일에서 유효한 스냅샷을 생성하지 못했습니다.")

        real_end_year = snapshots_df["Year"].max()
        stats = {
            "total_files": 1,
            "snapshots": len(snapshots_df),
            "start_year": start_year,
            "end_year": real_end_year,
            "files_loaded": 1,
            "frequency": freq.upper(),
            "tickers_loaded": sorted(snapshots_df["Ticker"].unique()),
            "dropped_no_date": [],
            "dropped_no_valid": [],
            "dropped_missing_features": [],
        }

        t_total = time.perf_counter() - t0
        logger.info(
            f"병합 파일 로드 완료: 총 {len(snapshots_df)} 건 스냅샷 생성. 소요시간 {t_total:.3f}s"
        )
        return snapshots_df, stats

    snapshots = []
    loaded_tickers: set[str] = set()
    dropped_no_date: list[str] = []
    dropped_no_valid: list[str] = []
    dropped_missing_features: list[str] = []

    for fp in files:
        t_file_start = time.perf_counter()
        try:
            # 2. 파일명 파싱 (Format: Ticker_Name.parquet)
            stem = fp.stem
            if "_" in stem:
                ticker, name = stem.split("_", 1)
            else:
                ticker, name = stem, stem

            # 3. 데이터 로드 및 컬럼 정규화 (최소 컬럼만 읽기)
            columns_to_load: Iterable[str] | None = None
            if pq is not None:
                try:
                    schema = pq.read_schema(fp)
                    available = set(schema.names)
                    columns_to_load = [c for c in DESIRED_COLUMNS if c in available]
                except Exception:
                    columns_to_load = None

            # 날짜만 먼저 읽어 범위 밖이면 스킵 (가벼운 선필터)
            if columns_to_load is not None and "Date" in columns_to_load:
                try:
                    date_only = pd.read_parquet(fp, columns=["Date"])
                    date_only["Date"] = pd.to_datetime(
                        date_only["Date"], errors="coerce"
                    )
                    date_only["Year"] = date_only["Date"].dt.year
                    mask = (date_only["Year"] >= start_year) & (
                        date_only["Year"] <= (end_year if end_year else 9999)
                    )
                    if not mask.any():
                        dropped_no_valid.append(ticker)
                        continue
                except Exception:
                    # 선필터 실패 시 전체 로드 시도
                    pass

            df = pd.read_parquet(fp, columns=columns_to_load)
            df.columns = [
                c.capitalize() if c.lower() in ["date", "close"] else c
                for c in df.columns
            ]

            # 4. 날짜 컬럼 검증
            if "Date" not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index().rename(columns={"index": "Date"})
                else:
                    dropped_no_date.append(ticker)
                    continue

            df["Date"] = pd.to_datetime(df["Date"])
            df["Year"] = df["Date"].dt.year
            df["Month"] = df["Date"].dt.month

            # 5. 연도 범위 필터링
            mask = (df["Year"] >= start_year) & (
                df["Year"] <= (end_year if end_year else 9999)
            )
            df_filtered = df.loc[mask].copy()

            if df_filtered.empty:
                dropped_no_valid.append(ticker)
                continue

            # 6. 주기별 스냅샷 추출 (Resampling)
            df_filtered = df_filtered.sort_values("Date")

            if freq.upper() == "M":
                # 월말 데이터 추출
                df_filtered["YearMonth"] = df_filtered["Date"].dt.to_period("M")
                snapshots_last = df_filtered.drop_duplicates(
                    subset=["YearMonth"], keep="last"
                ).copy()
            else:
                # 연말 데이터 추출 (기본값)
                snapshots_last = df_filtered.drop_duplicates(
                    subset=["Year"], keep="last"
                ).copy()

            # 메타데이터 추가
            snapshots_last["Ticker"] = ticker
            snapshots_last["Name"] = name

            # 파생 변수(NATR) 계산
            # - 데이터에 없으면 ATR_14/Close로 보강
            if "ATR_14" in snapshots_last.columns and "Close" in snapshots_last.columns:
                atr = pd.to_numeric(snapshots_last["ATR_14"], errors="coerce")
                close = pd.to_numeric(snapshots_last["Close"], errors="coerce")
                close = close.mask(close <= 0)
                for natr_col in ("NATR_14", "NATR"):
                    if (
                        natr_col in FEATURE_COLUMNS
                        and natr_col not in snapshots_last.columns
                    ):
                        snapshots_last[natr_col] = (atr / close) * 100

            # 유효성 검사 (필수 Feature 확인)
            needed_cols = [c for c in FEATURE_COLUMNS if c in snapshots_last.columns]
            valid_rows = snapshots_last.dropna(subset=needed_cols)

            if valid_rows.empty:
                dropped_missing_features.append(ticker)
                continue

            snapshots.append(valid_rows)
            loaded_tickers.add(ticker)

            t_file_elapsed = time.perf_counter() - t_file_start
            logger.debug(f"파일 로드 완료 ({fp.name}): {t_file_elapsed:.3f}s")

        except Exception as e:
            logger.warning(f"파일 처리 중 오류 발생 ({fp.name}): {e}")
            continue

    if not snapshots:
        raise ValueError("유효한 데이터가 하나도 로드되지 않았습니다.")

    # 최종 병합
    final_df = pd.concat(snapshots, ignore_index=True)
    real_end_year = final_df["Year"].max()

    # 로드 통계 집계
    stats = {
        "total_files": len(files),
        "snapshots": len(final_df),
        "start_year": start_year,
        "end_year": real_end_year,
        "files_loaded": len(files),
        "frequency": freq.upper(),
        "tickers_loaded": sorted(loaded_tickers),
        "dropped_no_date": sorted(set(dropped_no_date)),
        "dropped_no_valid": sorted(set(dropped_no_valid)),
        "dropped_missing_features": sorted(set(dropped_missing_features)),
    }

    t_total = time.perf_counter() - t0
    logger.info(
        f"데이터 로드 완료: 총 {len(final_df)} 건의 스냅샷 생성됨. 소요시간 {t_total:.3f}s"
    )
    return final_df, stats
