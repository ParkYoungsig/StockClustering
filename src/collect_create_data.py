import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

tqdm_auto.pandas()
warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# 설정파일(/src/config.py)에서 List 파일과 데이터 파일 위치 가져오기
# -------------------------------------------------------------------
from config import DATA_FILE_LOCATION, LIST_FILE_LOCATION

DATA_DIR = Path(DATA_FILE_LOCATION).resolve()
LIST_DIR = Path(LIST_FILE_LOCATION).resolve()
STOCK_LIST_FILE = LIST_DIR / "stock_list.csv"
FINANCIALS_FILE = LIST_DIR / "financials.xlsx"
DELIST_REPORT_FILE = LIST_DIR / "stock_delist.md"

# Technical analysis constants
SQRT_252 = np.sqrt(252)
ALPHA_14 = 1 / 14
EPSILON = 1e-10

RETURN_WINDOWS = [1, 5, 20, 30, 50, 60, 100, 120, 200]
DISPARITY_WINDOWS = [5, 20, 60, 120]
DELISTING_THRESHOLD_DAYS = 10

# Adaptive worker counts based on CPU cores
CPU_COUNT = os.cpu_count() or 2
MAX_DOWNLOAD_WORKERS = min(8, max(1, CPU_COUNT // 2))
MAX_EXCEL_WORKERS = min(4, max(1, CPU_COUNT // 2))


# Utility function to clean ticker codes
def clean_ticker(ticker_series, strip_prefix=True):
    """
    Clean ticker codes to 6-digit format.

    Args:
        ticker_series: pandas Series of ticker codes
        strip_prefix: If True, removes 'a' or 'A' prefix before padding

    Returns:
        pandas Series of cleaned 6-digit ticker codes
    """
    ticker_str = ticker_series.astype(str)
    if strip_prefix:
        ticker_str = ticker_str.str.lstrip("aA")
    return ticker_str.str.zfill(6)


def data_download(start_date="2015-01-01", end_date="2024-12-31"):
    print(f"작업 디렉토리: {DATA_DIR}")
    print("파일 상태:")
    print(f"  ✓ stock_list.csv: {STOCK_LIST_FILE.exists()}")
    print(f"  ✓ financials.xlsx: {FINANCIALS_FILE.exists()}")

    if not STOCK_LIST_FILE.exists():
        raise FileNotFoundError(f"stock_list.csv를 찾을 수 없습니다: {STOCK_LIST_FILE}")
    if not FINANCIALS_FILE.exists():
        raise FileNotFoundError(
            f"financials.xlsx를 찾을 수 없습니다: {FINANCIALS_FILE}"
        )

    ticker_df = pd.read_csv(STOCK_LIST_FILE, encoding="cp949")
    ticker_series = clean_ticker(ticker_df.iloc[:, 0], strip_prefix=False)
    tickers = ticker_series.tolist()

    ticker_to_name = dict(zip(ticker_series, ticker_df.iloc[:, 1]))

    print(f"\n✓ {len(ticker_df)}개 종목 로드 완료")
    print(f"✓ {len(tickers)}개 티커 추출 완료")
    print(f"데이터 수집 기간: {start_date} ~ {end_date}")
    print(f"수집할 종목 수: {len(tickers)}\n")
    print(
        f"병렬 처리 설정: CPU {CPU_COUNT}코어 → 다운로드 {MAX_DOWNLOAD_WORKERS}개, Excel {MAX_EXCEL_WORKERS}개 워커\n"
    )

    # ========== PARALLEL EXECUTION: Stock Downloads + Excel Loading ==========
    print("=" * 80)
    print("데이터 수집 시작")
    print("=" * 80 + "\n")

    def download_stocks_task():
        """Task 1: Download stock data from API"""

        def download_single_stock(ticker):
            try:
                stock_df = fdr.DataReader(ticker, start_date, end_date)

                if not stock_df.empty:
                    stock_df["Ticker"] = ticker
                    stock_df = stock_df.reset_index()
                    return ("success", ticker, stock_df)
                else:
                    stock_name = ticker_to_name.get(ticker, "알수없음")
                    return ("empty", ticker, stock_name)

            except Exception as e:
                stock_name = ticker_to_name.get(ticker, "알수없음")
                return ("error", ticker, stock_name, str(e)[:100])

        all_stocks = []
        failed_tickers = []
        delisted_info = []
        list_lock = Lock()

        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
            future_to_ticker = {
                executor.submit(download_single_stock, ticker): ticker
                for ticker in tickers
            }

            for future in tqdm(
                as_completed(future_to_ticker),
                total=len(tickers),
                desc="Stock Download",
                ncols=80,
                position=0,
                leave=True,
            ):
                result = future.result()

                with list_lock:
                    if result[0] == "success":
                        _, ticker, stock_df = result
                        all_stocks.append(stock_df)

                    elif result[0] == "empty":
                        _, ticker, stock_name = result
                        failed_tickers.append(ticker)
                        delisted_info.append(
                            {
                                "종목코드": ticker,
                                "종목명": stock_name,
                                "상태": "데이터없음",
                                "마지막거래일": "N/A",
                                "사유": "데이터를 가져올 수 없음",
                            }
                        )

                    else:
                        _, ticker, stock_name, error_msg = result
                        failed_tickers.append(ticker)
                        delisted_info.append(
                            {
                                "종목코드": ticker,
                                "종목명": stock_name,
                                "상태": "오류발생",
                                "마지막거래일": "N/A",
                                "사유": error_msg,
                            }
                        )

        print(
            f"\n주식 다운로드 완료: {len(all_stocks)}개 성공, {len(failed_tickers)}개 실패\n"
        )

        return all_stocks, failed_tickers, delisted_info

    def load_excel_task():
        """Task 2: Load financial data from Excel"""

        # Pre-compute ticker set once (shared by all functions)
        tickers_set = set(tickers)

        def load_financial_sheet_filtered(sheet_name):
            # Create a new ExcelFile instance for each thread (calamine is not thread-safe)
            try:
                excel_file_obj = pd.ExcelFile(FINANCIALS_FILE, engine="calamine")
            except ImportError:
                excel_file_obj = pd.ExcelFile(FINANCIALS_FILE, engine="openpyxl")

            # Single-pass: Load sheet and filter columns after
            df_raw = excel_file_obj.parse(sheet_name, header=None)

            all_tickers_clean = clean_ticker(df_raw.iloc[0, 1:], strip_prefix=True)
            dates = pd.to_datetime(df_raw.iloc[3:, 0])

            mask = all_tickers_clean.isin(tickers_set)
            cols_to_keep = [i + 1 for i, keep in enumerate(mask) if keep]
            tickers_kept = all_tickers_clean[mask].tolist()

            values = df_raw.iloc[3:, cols_to_keep]

            df = pd.DataFrame(values.values, index=dates, columns=tickers_kept)
            df = df.apply(pd.to_numeric, errors="coerce")
            df = df.sort_index()

            return sheet_name, df

        def process_metric_rows(filtered_rows, dates):
            """Helper to process metric rows into DataFrame with dates as index"""
            if filtered_rows.empty:
                return pd.DataFrame(index=dates)

            metric_data = filtered_rows.set_index("ticker_clean").iloc[:, 3:-2].T

            # Handle length mismatch between dates and transposed data
            # This can happen if the Excel file has trailing columns
            min_len = min(len(metric_data), len(dates))
            metric_data = metric_data.iloc[:min_len]
            metric_data.index = dates[:min_len]

            return metric_data.apply(pd.to_numeric, errors="coerce").sort_index()

        def load_raw_sheet_metrics():
            # Create a new ExcelFile instance for each thread (calamine is not thread-safe)
            try:
                excel_file_obj = pd.ExcelFile(FINANCIALS_FILE, engine="calamine")
            except ImportError:
                excel_file_obj = pd.ExcelFile(FINANCIALS_FILE, engine="openpyxl")

            df_raw = excel_file_obj.parse("RAW", header=None)

            # Clean ticker column using modular function
            ticker_col = clean_ticker(df_raw.iloc[:, 0], strip_prefix=True)
            metric_col = df_raw.iloc[:, 2].astype(str)

            # Calculate dates once
            dates = pd.to_datetime(df_raw.iloc[0, 3:])

            # Filter rows early to reduce memory
            ticker_mask = ticker_col.isin(tickers_set)
            df_filtered = df_raw[ticker_mask].copy()

            if df_filtered.empty:
                return pd.DataFrame(index=dates), pd.DataFrame(index=dates)

            df_filtered["ticker_clean"] = ticker_col[ticker_mask]
            df_filtered["metric"] = metric_col[ticker_mask]

            # Use modular function to process both metrics
            mask_revenue = df_filtered["metric"].str.contains("매출액", na=False)
            mask_op_profit = df_filtered["metric"].str.contains("영업이익", na=False)

            df_revenue = process_metric_rows(df_filtered[mask_revenue], dates)
            df_op_profit = process_metric_rows(df_filtered[mask_op_profit], dates)

            return df_revenue, df_op_profit

        financial_sheets = ["BPS", "DPS", "EPS", "배당수익률"]
        sheet_results = {}
        total_tasks = len(financial_sheets) + 1  # +1 for RAW_METRICS

        with ThreadPoolExecutor(max_workers=MAX_EXCEL_WORKERS) as executor:
            futures = {
                executor.submit(load_financial_sheet_filtered, sheet): sheet
                for sheet in financial_sheets
            }
            futures[executor.submit(load_raw_sheet_metrics)] = "RAW_METRICS"

            with tqdm(
                total=total_tasks,
                desc="Financial Data",
                ncols=80,
                position=1,
                leave=True,
            ) as pbar:
                for future in as_completed(futures):
                    task_name = futures[future]
                    if task_name == "RAW_METRICS":
                        df_revenue, df_op_profit = future.result()
                    else:
                        sheet_name, df = future.result()
                        sheet_results[sheet_name] = df
                    pbar.update(1)

        df_bps = sheet_results["BPS"]
        df_dps = sheet_results["DPS"]
        df_eps = sheet_results["EPS"]
        df_div_yield = sheet_results["배당수익률"]

        print("\n재무 데이터 로드 완료")
        print(
            f"  BPS/DPS/EPS/배당수익률: {df_bps.shape[0]} dates x {df_bps.shape[1]} tickers"
        )
        print(
            f"  매출액/영업이익: {df_revenue.shape[0]} dates x {df_revenue.shape[1]} tickers\n"
        )

        return df_bps, df_dps, df_eps, df_div_yield, df_revenue, df_op_profit

    # Run both tasks in parallel
    with ThreadPoolExecutor(max_workers=2) as main_executor:
        stock_future = main_executor.submit(download_stocks_task)
        excel_future = main_executor.submit(load_excel_task)

        all_stocks, failed_tickers, delisted_info = stock_future.result()
        df_bps, df_dps, df_eps, df_div_yield, df_revenue, df_op_profit = (
            excel_future.result()
        )

    print("=" * 80)
    print("데이터 수집 완료!")
    print("=" * 80 + "\n")

    # ========== Process stock data ==========
    if not all_stocks:
        raise ValueError("수집된 데이터가 없습니다!")

    df_all = pd.concat(all_stocks, ignore_index=True, copy=False)
    df_all = df_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print(f"결합된 DataFrame 크기: {df_all.shape}")
    print(f"날짜 범위: {df_all['Date'].min()} ~ {df_all['Date'].max()}")
    print(f"고유 종목 수: {df_all['Ticker'].nunique()}\n")

    df_all["종목명"] = df_all["Ticker"].map(ticker_to_name).astype("category")

    cols = df_all.columns.tolist()
    cols.remove("종목명")
    cols.insert(1, "종목명")
    df_all = df_all[cols]

    print(f"DataFrame 크기: {df_all.shape}\n")

    def calculate_all_features(group):
        for window in RETURN_WINDOWS:
            group[f"Return_{window}d"] = group["Close"].pct_change(periods=window) * 100

        group["vol_20"] = (
            group["Close"].pct_change().rolling(window=20, min_periods=20).std()
        )
        group["vol_60"] = (
            group["Close"].pct_change().rolling(window=60, min_periods=60).std()
        )
        group["vol_60_sqrt252"] = group["vol_60"] * SQRT_252
        group["log_vol"] = np.log(group["Volume"] + 1)
        log_vol_mean_60 = group["log_vol"].rolling(window=60, min_periods=60).mean()
        group["vol_ratio_60"] = group["log_vol"] - log_vol_mean_60
        group["avg_log_vol_ratio_60"] = (
            group["vol_ratio_60"].rolling(window=60, min_periods=60).mean()
        )
        group["std_log_vol_ratio_60"] = (
            group["vol_ratio_60"].rolling(window=60, min_periods=60).std()
        )

        delta = group["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=ALPHA_14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=ALPHA_14, adjust=False, min_periods=14).mean()
        rs = avg_gain / (avg_loss + EPSILON)
        group["RSI_14"] = 100 - (100 / (1 + rs))
        group["RSI_14_60avg"] = (
            group["RSI_14"].rolling(window=60, min_periods=60).mean()
        )

        typical_price = (group["High"] + group["Low"] + group["Close"]) / 3
        money_flow = typical_price * group["Volume"]
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.ewm(
            alpha=ALPHA_14, adjust=False, min_periods=14
        ).mean()
        negative_mf = negative_flow.ewm(
            alpha=ALPHA_14, adjust=False, min_periods=14
        ).mean()
        mfi_ratio = positive_mf / (negative_mf + EPSILON)
        group["MFI_14"] = 100 - (100 / (1 + mfi_ratio))

        high_low = group["High"] - group["Low"]
        high_close = np.abs(group["High"] - group["Close"].shift(1))
        low_close = np.abs(group["Low"] - group["Close"].shift(1))
        true_range = pd.Series(
            np.maximum.reduce([high_low, high_close, low_close]), index=group.index
        )
        group["ATR_14"] = true_range.ewm(
            alpha=ALPHA_14, adjust=False, min_periods=14
        ).mean()
        group["NATR_14"] = (group["ATR_14"] / group["Close"]) * 100

        high_diff = group["High"].diff()
        low_diff = -group["Low"].diff()
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        smoothed_tr = true_range.ewm(
            alpha=ALPHA_14, adjust=False, min_periods=14
        ).mean()
        smoothed_plus_dm = plus_dm.ewm(
            alpha=ALPHA_14, adjust=False, min_periods=14
        ).mean()
        smoothed_minus_dm = minus_dm.ewm(
            alpha=ALPHA_14, adjust=False, min_periods=14
        ).mean()
        plus_di = 100 * (smoothed_plus_dm / (smoothed_tr + EPSILON))
        minus_di = 100 * (smoothed_minus_dm / (smoothed_tr + EPSILON))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + EPSILON)
        group["ADX_14"] = dx.ewm(alpha=ALPHA_14, adjust=False, min_periods=14).mean()

        for window in DISPARITY_WINDOWS:
            ma = group["Close"].rolling(window=window, min_periods=window).mean()
            group[f"Disparity_{window}d"] = ((group["Close"] - ma) / ma) * 100

        daily_returns = group["Close"].pct_change()
        group["Mean_60d"] = daily_returns.rolling(window=60, min_periods=60).mean()
        group["Median_60d"] = daily_returns.rolling(window=60, min_periods=60).median()
        group["Std_60d"] = daily_returns.rolling(window=60, min_periods=60).std()
        group["Sharpe_60d"] = group["Mean_60d"] / (group["Std_60d"] + EPSILON)

        mean_252 = daily_returns.rolling(window=252, min_periods=252).mean()
        std_252 = daily_returns.rolling(window=252, min_periods=252).std()
        group["Sharpe_252d"] = mean_252 / (std_252 + EPSILON)

        downside_returns = daily_returns.where(daily_returns < 0, 0)
        downside_std_60 = downside_returns.rolling(window=60, min_periods=60).std()
        group["Sortino_60d"] = group["Mean_60d"] / (downside_std_60 + EPSILON)

        downside_std_252 = downside_returns.rolling(window=252, min_periods=252).std()
        group["Sortino_252d"] = mean_252 / (downside_std_252 + EPSILON)

        group["Skewness_60d"] = daily_returns.rolling(window=60, min_periods=60).skew()
        group["Zscore_60d"] = (
            group["Close"] - group["Close"].rolling(window=60, min_periods=60).mean()
        ) / (group["Close"].rolling(window=60, min_periods=60).std() + EPSILON)

        risk_cols = [
            "Mean_60d",
            "Median_60d",
            "Std_60d",
            "Sharpe_60d",
            "Sharpe_252d",
            "Sortino_60d",
            "Sortino_252d",
            "Skewness_60d",
            "Zscore_60d",
        ]
        group[risk_cols] = group[risk_cols].replace([np.inf, -np.inf], np.nan)

        return group

    try:
        df_all = df_all.groupby("Ticker", group_keys=False).progress_apply(
            calculate_all_features
        )
    except AttributeError:
        df_all = df_all.groupby("Ticker", group_keys=False).apply(
            calculate_all_features
        )

    print(f"\nDataFrame 크기: {df_all.shape}")
    print(f"총 컬럼 수: {len(df_all.columns)}\n")

    # ========== Financial data already loaded in parallel above ==========

    print("Applying lags...\n")

    def apply_financial_lag(df_financial):
        df_lagged = df_financial.copy()
        df_lagged.index = df_lagged.index + pd.DateOffset(months=3)
        return df_lagged

    def apply_quarterly_lag(df_quarterly):
        df_lagged = df_quarterly.copy()

        new_index = []
        for date in df_lagged.index:
            month = date.month
            year = date.year

            if month == 3:
                new_date = pd.Timestamp(year=year, month=4, day=1)
            elif month == 6:
                new_date = pd.Timestamp(year=year, month=7, day=1)
            elif month == 9:
                new_date = pd.Timestamp(year=year, month=10, day=1)
            elif month == 12:
                new_date = pd.Timestamp(year=year + 1, month=1, day=1)
            else:
                new_date = date

            new_index.append(new_date)

        df_lagged.index = pd.DatetimeIndex(new_index)
        return df_lagged

    df_bps_lagged = apply_financial_lag(df_bps)
    df_dps_lagged = apply_financial_lag(df_dps)
    df_eps_lagged = apply_financial_lag(df_eps)
    df_div_yield_lagged = apply_financial_lag(df_div_yield)

    df_revenue_lagged = apply_quarterly_lag(df_revenue)
    df_op_profit_lagged = apply_quarterly_lag(df_op_profit)

    print("✓ Lags applied:")
    print("  - Daily data (BPS, DPS, EPS, dividend yield): 3 months lag")
    print("  - Quarterly data (revenue, operating profit): next quarter start")
    print("    · Q1 (Mar 31) → Apr 1")
    print("    · Q2 (Jun 30) → Jul 1")
    print("    · Q3 (Sep 30) → Oct 1")
    print("    · Q4 (Dec 31) → Jan 1 (next year)\n")

    print("Merging financial data and calculating derived metrics...\n")

    def merge_financial_data(group):
        ticker = group.name

        group = group.reset_index()

        if "Date" not in group.columns and "index" in group.columns:
            group = group.rename(columns={"index": "Date"})

        group["Date"] = pd.to_datetime(group["Date"])
        group = group.sort_values("Date")

        mapping = {
            "BPS": df_bps_lagged,
            "DPS": df_dps_lagged,
            "EPS": df_eps_lagged,
            "배당수익률": df_div_yield_lagged,
            "매출액": df_revenue_lagged,
            "영업이익": df_op_profit_lagged,
        }

        for col_name, df_source in mapping.items():
            if ticker in df_source.columns:
                f_data = df_source[ticker].dropna().to_frame(name=col_name)
                f_data = f_data.reset_index()
                f_data.columns = ["Date", col_name]
                f_data["Date"] = pd.to_datetime(f_data["Date"])
                f_data = f_data.sort_values("Date")

                group = pd.merge_asof(group, f_data, on="Date", direction="backward")
            else:
                group[col_name] = np.nan

        group["PER"] = group["Close"] / group["EPS"]
        group["PBR"] = group["Close"] / group["BPS"]
        group["ROE"] = group["PBR"] / group["PER"]
        group["배당성향"] = (group["DPS"] / group["EPS"]) * 100

        group["ROE_YoY"] = (group["ROE"] / group["ROE"].shift(252) - 1) * 100
        group["EPS_YoY"] = (group["EPS"] / group["EPS"].shift(252) - 1) * 100
        group["영업이익_YoY"] = (
            group["영업이익"] / group["영업이익"].shift(252) - 1
        ) * 100
        group["매출액_YoY"] = (group["매출액"] / group["매출액"].shift(252) - 1) * 100

        metrics = [
            "PER",
            "PBR",
            "ROE",
            "배당성향",
            "ROE_YoY",
            "EPS_YoY",
            "영업이익_YoY",
            "매출액_YoY",
        ]
        group[metrics] = group[metrics].replace([np.inf, -np.inf], np.nan)

        group["Ticker"] = ticker

        return group.set_index(["Date", "Ticker"])

    try:
        df_all = df_all.groupby("Ticker", group_keys=False).progress_apply(
            merge_financial_data
        )
    except AttributeError:
        df_all = df_all.groupby("Ticker", group_keys=False).apply(merge_financial_data)

    print(f"\n최종 DataFrame 크기: {df_all.shape}")
    print(f"총 컬럼 수: {len(df_all.columns)}\n")

    # Rotate data directories: data -> data_old (keep only 2 versions)
    output_dir = DATA_DIR
    old_dir = DATA_DIR.parent / "data_old"

    if output_dir.exists():
        # If data_old exists, delete it first
        if old_dir.exists():
            import shutil

            shutil.rmtree(old_dir)
            print(f"Removed old backup: {old_dir}")

        # Rename current data to data_old
        output_dir.rename(old_dir)
        print(f"Backed up current data: {output_dir} -> {old_dir}")

    # Create fresh data directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created new data directory: {output_dir}\n")

    all_tickers = df_all.index.get_level_values("Ticker").unique()

    print(f"개별 종목 파일 저장 중: {output_dir}\n")

    stock_metadata = {}
    col_count = 0

    for ticker in tqdm(all_tickers, desc="Saving Files", ncols=80):
        stock_df = df_all.xs(ticker, level="Ticker").copy()

        cols_to_drop = ["Ticker", "index", "level_0"]
        stock_df = stock_df.drop(
            columns=[c for c in cols_to_drop if c in stock_df.columns]
        )

        stock_df = stock_df.sort_index()

        if col_count == 0:
            col_count = stock_df.shape[1]

        stock_metadata[ticker] = {
            "last_date": stock_df.index.max(),
            "last_close": stock_df.loc[stock_df.index.max(), "Close"],
        }

        stock_name = ticker_to_name.get(ticker, "알수없음")

        filename = f"{ticker}_{stock_name}.parquet"
        filepath = output_dir / filename

        stock_df.to_parquet(
            filepath, engine="pyarrow", compression="snappy", index=True
        )

    unique_ticker_count = len(all_tickers)

    print("\n상장폐지 종목 검사 중...\n")

    end_date_dt = pd.to_datetime(end_date)
    delisting_threshold = end_date_dt - pd.Timedelta(days=DELISTING_THRESHOLD_DAYS)

    for ticker, metadata in tqdm(
        stock_metadata.items(), desc="Delisting Check", ncols=80
    ):
        last_date = metadata["last_date"]
        last_close = metadata["last_close"]

        stock_name = ticker_to_name.get(ticker, "알수없음")

        is_delisted = False
        delisting_reason = ""

        if pd.isna(last_close) or last_close == 0:
            is_delisted = True
            delisting_reason = "종가가 0 또는 NaN"
        elif last_date < delisting_threshold:
            is_delisted = True
            delisting_reason = (
                f"거래 중단 ({last_date.strftime('%Y-%m-%d')}에 마지막 거래)"
            )

        if is_delisted:
            delisted_info.append(
                {
                    "종목코드": ticker,
                    "종목명": stock_name,
                    "상태": "상장폐지",
                    "마지막거래일": last_date.strftime("%Y-%m-%d"),
                    "사유": delisting_reason,
                }
            )

    print(f"\n{'=' * 80}")
    print("데이터 다운로드 및 처리 완료!")
    print(f"{'=' * 80}")
    print(f"✓ {unique_ticker_count}개 파일 저장 완료")
    print(f"✓ 출력 디렉토리: {output_dir}")
    print(f"✓ 파일당 컬럼 수: {col_count}개")
    print(f"{'=' * 80}\n")

    if delisted_info:
        delisted_md_path = DELIST_REPORT_FILE

        delisted_count = 0
        error_count = 0
        for d in delisted_info:
            if d["상태"] == "상장폐지":
                delisted_count += 1
            else:
                error_count += 1
        active_count = unique_ticker_count - delisted_count

        with open(delisted_md_path, "w", encoding="utf-8") as f:
            f.write("# 상장폐지 및 데이터 오류 종목 목록\n\n")
            f.write(f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"데이터 수집 기간: {start_date} ~ {end_date}\n\n")
            f.write("## 📊 요약\n\n")
            f.write(f"- 총 분석 종목 수: {unique_ticker_count}개\n")
            f.write(f"- 정상 거래 종목: {active_count}개\n")
            f.write(f"- 상장폐지 종목: {delisted_count}개\n")
            f.write(f"- 데이터 오류 종목: {error_count}개\n\n")
            f.write("---\n\n")
            f.write("## 🔍 상세 목록\n\n")
            f.write("| 종목코드 | 종목명 | 상태 | 마지막거래일 | 사유 |\n")
            f.write("|---------|--------|------|-------------|------|\n")
            for info in delisted_info:
                f.write(
                    f"| {info['종목코드']} | {info['종목명']} | {info['상태']} | {info['마지막거래일']} | {info['사유']} |\n"
                )
            f.write("\n---\n\n")
            f.write("## 📝 참고사항\n\n")
            f.write(
                f"- 이 리스트는 {start_date}부터 {end_date}까지의 데이터를 기준으로 작성되었습니다.\n"
            )
            f.write(
                f"- 마지막 거래일이 {delisting_threshold.strftime('%Y-%m-%d')} 이전인 종목은 상장폐지된 것으로 간주됩니다.\n"
            )
            f.write("- 종가가 0이거나 NaN인 종목은 상장폐지된 것으로 간주됩니다.\n")
            f.write("- **상장폐지 종목의 데이터는 마지막 거래일까지 보존됩니다.**\n")

        print(f"\n✓ 상장폐지 및 오류 종목 정보 저장: {delisted_md_path}")
        print(f"  - 정상 거래: {active_count}개")
        print(f"  - 상장폐지: {delisted_count}개")
        print(f"  - 데이터 오류: {error_count}개\n")

    return df_all


def data_load():
    print(f"데이터 디렉토리: {DATA_DIR}")
    print("데이터 폴더 확인 중...\n")

    # Simply use DATA_DIR without any search logic
    if not DATA_DIR.exists():
        print(f"\n오류: 데이터 폴더를 찾을 수 없습니다: {DATA_DIR.absolute()}")
        print("data_download()를 실행하여 데이터를 먼저 수집하세요.")
        return None

    parquet_files = list(DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        print(f"\n오류: '{DATA_DIR.name}' 폴더는 존재하지만 parquet 파일이 없습니다!")
        return None

    print(f"\n데이터 폴더: {DATA_DIR.absolute()}")
    print(f"{len(parquet_files)}개 parquet 파일 발견")

    num_files = len(parquet_files)
    max_parquet_workers = min(CPU_COUNT, 16, num_files)
    print(f"병렬 로딩: {max_parquet_workers}개 워커\n")

    def load_single_parquet(file_path):
        """Load a single parquet file and add ticker"""
        try:
            stock_df = pd.read_parquet(file_path, engine="pyarrow")

            # Extract ticker from filename (format: TICKER_NAME.parquet)
            ticker_code = file_path.stem.split("_")[0]
            stock_df["Ticker"] = ticker_code

            return ("success", stock_df, None)
        except Exception as e:
            return ("error", None, (file_path.name, str(e)))

    all_stocks = []
    failed_files = []

    # Parallel loading with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_parquet_workers) as executor:
        futures = {executor.submit(load_single_parquet, fp): fp for fp in parquet_files}

        for future in tqdm(
            as_completed(futures),
            total=len(parquet_files),
            desc="Loading Parquet Files",
            ncols=80,
        ):
            status, stock_df, error_info = future.result()

            if status == "success":
                all_stocks.append(stock_df)
            else:
                failed_files.append(error_info)

    if not all_stocks:
        print("\n오류: 읽을 수 있는 parquet 파일이 없습니다!")
        return None

    print(f"\n병합 중 ({len(all_stocks)}개 파일)...")
    df_all = pd.concat(all_stocks, ignore_index=False)
    df_all.index = pd.to_datetime(df_all.index)
    df_all.index.name = "Date"
    df_all = df_all.set_index("Ticker", append=True)
    df_all = df_all.sort_index()

    print(f"DataFrame 로드 완료: {len(df_all):,}행")

    if failed_files:
        print(f"\n경고: 읽지 못한 파일 {len(failed_files)}개")
        for filename, error in failed_files[:3]:
            print(f"  {filename}: {error[:50]}")
        if len(failed_files) > 3:
            print(f"  ... 외 {len(failed_files) - 3}개")

    print(f"\n{'=' * 80}")
    print(f"Index: {df_all.index.names} | Columns: {len(df_all.columns)}")
    print(f"{'=' * 80}\n")

    return df_all


def data_query(df):
    if df is None:
        print("❌ 오류: DataFrame이 None입니다. 먼저 data_load()를 실행하세요.")
        return None

    idx = pd.IndexSlice

    print("\n" + "=" * 80)
    print("데이터 쿼리")
    print("=" * 80)

    start_date_input = input("\n시작 날짜 (YYYY-MM-DD) [엔터=전체 시작일]: ").strip()
    if start_date_input == "":
        start_date = df.index.get_level_values("Date").min()
        print(f"→ 시작 날짜: {start_date.strftime('%Y-%m-%d')} (전체 시작일)")
    else:
        start_date = start_date_input
        print(f"→ 시작 날짜: {start_date}")

    end_date_input = input("종료 날짜 (YYYY-MM-DD) [엔터=전체 종료일]: ").strip()
    if end_date_input == "":
        end_date = df.index.get_level_values("Date").max()
        print(f"→ 종료 날짜: {end_date.strftime('%Y-%m-%d')} (전체 종료일)")
    else:
        end_date = end_date_input
        print(f"→ 종료 날짜: {end_date}")

    tickers_input = input(
        "\n찾을 티커 (쉼표로 구분, 예: 005930,000660) [엔터=전체]: "
    ).strip()
    if tickers_input == "":
        tickers_to_find = []
        print("→ 티커: 전체")
    else:
        tickers_to_find = [t.strip() for t in tickers_input.split(",")]
        print(f"→ 티커: {tickers_to_find}")

    columns_input = input(
        "찾을 컬럼 (쉼표로 구분, 예: Close,Volume,PER) [엔터=전체]: "
    ).strip()
    if columns_input == "":
        columns_to_find = []
        print("→ 컬럼: 전체")
    else:
        columns_to_find = [c.strip() for c in columns_input.split(",")]
        print(f"→ 컬럼: {columns_to_find}")

    print("\n" + "=" * 80)
    print("쿼리 실행 중...")
    print("=" * 80 + "\n")

    if tickers_to_find:
        df_final = df.loc[idx[start_date:end_date, tickers_to_find], :]
    else:
        df_final = df.loc[idx[start_date:end_date, :], :]

    if columns_to_find:
        available_cols = [c for c in columns_to_find if c in df_final.columns]
        missing_cols = [c for c in columns_to_find if c not in df_final.columns]

        if missing_cols:
            print(f"⚠️  존재하지 않는 컬럼: {missing_cols}")

        if available_cols:
            df_final = df_final[available_cols]
        else:
            print("❌ 오류: 유효한 컬럼이 없습니다!")
            return None

    print(f"\n{'=' * 80}")
    print(f"검색 결과 ({start_date} ~ {end_date})")
    print(f"{'=' * 80}")
    print(f"발견된 티커: {df_final.index.get_level_values('Ticker').unique().tolist()}")
    print(f"유지된 컬럼: {df_final.columns.tolist()}")
    print(f"총 행 수: {len(df_final):,}개")
    print(f"{'=' * 80}\n")

    print("처음 10행:")
    print(df_final.head(10))
    print("\n마지막 10행:")
    print(df_final.tail(10))

    return df_final
