import warnings
from datetime import datetime
from pathlib import Path

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

tqdm_auto.pandas()
warnings.filterwarnings("ignore")

# Module-level constants
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
STOCK_LIST_FILE = SCRIPT_DIR / "stock_list.csv"
FINANCIALS_FILE = SCRIPT_DIR / "financials.xlsx"
DELIST_REPORT_FILE = SCRIPT_DIR / "stock_delist.md"

# Technical analysis constants
SQRT_252 = np.sqrt(252)
ALPHA_14 = 1 / 14
EPSILON = 1e-10

# Analysis windows
RETURN_WINDOWS = [1, 5, 20, 30, 50, 60, 100, 120, 200]
DISPARITY_WINDOWS = [5, 20, 60, 120]

# Delisting detection threshold (days)
DELISTING_THRESHOLD_DAYS = 10


def data_download(start_date="2015-01-01", end_date="2024-12-31"):
    print(f"작업 디렉토리: {SCRIPT_DIR}")
    print("파일 상태:")
    print(f"  ✓ stock_list.csv: {STOCK_LIST_FILE.exists()}")
    print(f"  ✓ financials.xlsx: {FINANCIALS_FILE.exists()}")

    if not STOCK_LIST_FILE.exists():
        raise FileNotFoundError(f"stock_list.csv를 찾을 수 없습니다: {STOCK_LIST_FILE}")
    if not FINANCIALS_FILE.exists():
        raise FileNotFoundError(f"financials.xlsx를 찾을 수 없습니다: {FINANCIALS_FILE}")

    ticker_df = pd.read_csv(STOCK_LIST_FILE, encoding="cp949")
    tickers = ticker_df.iloc[:, 0].astype(str).str.zfill(6).tolist()

    ticker_to_name = dict(
        zip(ticker_df.iloc[:, 0].astype(str).str.zfill(6), ticker_df.iloc[:, 1])
    )

    print(f"\n✓ {len(ticker_df)}개 종목 로드 완료")
    print(f"✓ {len(tickers)}개 티커 추출 완료")
    print(f"데이터 수집 기간: {start_date} ~ {end_date}")
    print(f"수집할 종목 수: {len(tickers)}\n")

    all_stocks = []
    failed_tickers = []
    delisted_info = []

    print("데이터 수집 시작...\n")

    for ticker in tqdm(tickers, desc="주식 데이터 다운로드 중"):
        try:
            stock_df = fdr.DataReader(ticker, start_date, end_date)

            if not stock_df.empty:
                stock_df["Ticker"] = ticker
                stock_df = stock_df.reset_index()
                all_stocks.append(stock_df)
            else:
                failed_tickers.append(ticker)
                stock_name = ticker_to_name.get(ticker, "알수없음")
                delisted_info.append(
                    {
                        "종목코드": ticker,
                        "종목명": stock_name,
                        "상태": "데이터없음",
                        "마지막거래일": "N/A",
                        "사유": "데이터를 가져올 수 없음",
                    }
                )

        except Exception as e:
            failed_tickers.append(ticker)
            stock_name = ticker_to_name.get(ticker, "알수없음")
            delisted_info.append(
                {
                    "종목코드": ticker,
                    "종목명": stock_name,
                    "상태": "오류발생",
                    "마지막거래일": "N/A",
                    "사유": str(e)[:100],
                }
            )
            print(f"\n{ticker} 다운로드 오류: {str(e)[:100]}")

    print(f"\n✓ 성공적으로 다운로드: {len(all_stocks)}개 종목")
    if failed_tickers:
        print(f"✗ 실패: {len(failed_tickers)}개 종목")

    if not all_stocks:
        raise ValueError("수집된 데이터가 없습니다!")

    df_all = pd.concat(all_stocks, ignore_index=True)
    df_all = df_all.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    print(f"결합된 DataFrame 크기: {df_all.shape}")
    print(f"날짜 범위: {df_all['Date'].min()} ~ {df_all['Date'].max()}")
    print(f"고유 종목 수: {df_all['Ticker'].nunique()}\n")

    print("종목명 추가 및 카테고리 타입 변환 중...\n")

    df_all["종목명"] = df_all["Ticker"].map(ticker_to_name).astype("category")

    cols = df_all.columns.tolist()
    cols.remove("종목명")
    cols.insert(1, "종목명")
    df_all = df_all[cols]

    print("✓ 종목명을 카테고리 타입으로 추가")
    print(f"DataFrame 크기: {df_all.shape}\n")

    print("모든 기술적 특성 계산 중 (수익률, 거래량, 지표, 이격도, 리스크)...\n")

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

    print("\n✓ 하나의 groupby 패스로 모든 기술적 특성 추가 완료!")
    print("  - 수익률: 9개 컬럼")
    print("  - 거래량: 7개 컬럼")
    print("  - 지표: 6개 컬럼 (RSI, MFI, ATR, NATR, ADX)")
    print("  - 이격도: 4개 컬럼")
    print("  - 리스크: 9개 컬럼")
    print(f"\nDataFrame 크기: {df_all.shape}")
    print(f"총 컬럼 수: {len(df_all.columns)}\n")

    print("재무제표 데이터를 Excel에서 로드 중...\n")

    excel_file_obj = pd.ExcelFile(FINANCIALS_FILE)

    def load_financial_sheet_filtered(excel_obj, sheet_name, tickers_to_keep):
        df_raw = excel_obj.parse(sheet_name, header=None)

        all_tickers = df_raw.iloc[0, 1:].astype(str).tolist()
        company_names = df_raw.iloc[1, 1:].astype(str).tolist()
        dates = pd.to_datetime(df_raw.iloc[3:, 0])

        all_tickers_clean = [str(col).lstrip("aA").zfill(6) for col in all_tickers]

        tickers_set = set(tickers_to_keep)
        cols_to_keep = [
            i + 1 for i, t in enumerate(all_tickers_clean) if t in tickers_set
        ]
        tickers_kept = [all_tickers_clean[i - 1] for i in cols_to_keep]

        values = df_raw.iloc[3:, cols_to_keep]

        df = pd.DataFrame(values.values, index=dates, columns=tickers_kept)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.sort_index()

        sheet_ticker_to_name = {
            tickers_kept[i]: company_names[cols_to_keep[i] - 1]
            for i in range(len(tickers_kept))
        }

        return df, sheet_ticker_to_name

    def load_raw_sheet_metric_filtered(excel_obj, metric_name, tickers_to_keep):
        df_raw = excel_obj.parse("RAW", header=None)
        dates = pd.to_datetime(df_raw.iloc[0, 3:])
        metric_rows = df_raw[
            df_raw.iloc[:, 2].astype(str).str.contains(metric_name, na=False)
        ]

        tickers_set = set(tickers_to_keep)
        ticker_data = {}

        for idx, row in metric_rows.iterrows():
            ticker = str(row.iloc[0]).lstrip("aA").zfill(6)
            if ticker in tickers_set:
                values = row.iloc[3:].values
                ticker_series = pd.Series(values, index=dates)
                ticker_data[ticker] = ticker_series

        df_metric = pd.DataFrame(ticker_data)
        df_metric = df_metric.apply(pd.to_numeric, errors="coerce")
        df_metric = df_metric.sort_index()

        return df_metric

    print(f"Excel 파일: {FINANCIALS_FILE}")

    print("BPS 시트 로드 중...")
    df_bps, bps_names = load_financial_sheet_filtered(excel_file_obj, "BPS", tickers)
    print(f"  ✓ BPS: {df_bps.shape[0]}개 날짜 × {df_bps.shape[1]}개 티커 (필터됨)")

    print("DPS 시트 로드 중...")
    df_dps, dps_names = load_financial_sheet_filtered(excel_file_obj, "DPS", tickers)
    print(f"  ✓ DPS: {df_dps.shape[0]}개 날짜 × {df_dps.shape[1]}개 티커 (필터됨)")

    print("EPS 시트 로드 중...")
    df_eps, eps_names = load_financial_sheet_filtered(excel_file_obj, "EPS", tickers)
    print(f"  ✓ EPS: {df_eps.shape[0]}개 날짜 × {df_eps.shape[1]}개 티커 (필터됨)")

    print("배당수익률 시트 로드 중...")
    df_div_yield, div_yield_names = load_financial_sheet_filtered(
        excel_file_obj, "배당수익률", tickers
    )
    print(
        f"  ✓ 배당수익률: {df_div_yield.shape[0]}개 날짜 × {df_div_yield.shape[1]}개 티커 (필터됨)"
    )

    print("RAW 시트에서 매출액 로드 중...")
    df_revenue = load_raw_sheet_metric_filtered(excel_file_obj, "매출액", tickers)
    print(
        f"  ✓ 매출액: {df_revenue.shape[0]}개 날짜 × {df_revenue.shape[1]}개 티커 (필터됨)"
    )

    print("RAW 시트에서 영업이익 로드 중...")
    df_op_profit = load_raw_sheet_metric_filtered(excel_file_obj, "영업이익", tickers)
    print(
        f"  ✓ 영업이익: {df_op_profit.shape[0]}개 날짜 × {df_op_profit.shape[1]}개 티커 (필터됨)"
    )

    print("\n✓ 재무 데이터 로드 완료\n")

    print("지연 적용 중...\n")

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

    print("✓ 지연 적용 완료:")
    print("  - 일간 데이터 (BPS, DPS, EPS, 배당수익률): 3개월 지연")
    print("  - 분기 데이터 (매출액, 영업이익): 다음 분기 시작부터 사용 가능")
    print("    · Q1 (3월 31일) → 4월 1일")
    print("    · Q2 (6월 30일) → 7월 1일")
    print("    · Q3 (9월 30일) → 10월 1일")
    print("    · Q4 (12월 31일) → 1월 1일 (다음 해)\n")

    print("재무 데이터 병합 및 파생 지표 계산 중...\n")

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
        df_all = df_all.groupby("Ticker", group_keys=False).apply(
            merge_financial_data
        )

    print("✓ 재무제표 컬럼 추가:")
    print("  - BPS, DPS, EPS, 배당수익률, 매출액, 영업이익")
    print("\n✓ 파생 지표 계산:")
    print("  - PER, PBR, ROE, 배당성향")
    print("  - ROE_YoY, EPS_YoY, 영업이익_YoY, 매출액_YoY")
    print(f"\n최종 DataFrame 크기: {df_all.shape}")
    print(f"총 컬럼 수: {len(df_all.columns)}\n")

    output_dir = DATA_DIR
    base_output_dir = output_dir
    counter = 1
    while output_dir.exists():
        output_dir = Path(f"{base_output_dir} ({counter})")
        counter += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    all_tickers = df_all.index.get_level_values("Ticker").unique()

    print(f"개별 종목 파일 저장 중: {output_dir}\n")

    stock_metadata = {}

    for ticker in tqdm(all_tickers, desc="parquet 파일 저장 중"):
        stock_df = df_all.xs(ticker, level="Ticker").copy()

        cols_to_drop = ["Ticker", "index", "level_0"]
        stock_df = stock_df.drop(
            columns=[c for c in cols_to_drop if c in stock_df.columns]
        )

        stock_df = stock_df.sort_index()

        stock_metadata[ticker] = {
            "last_date": stock_df.index.max(),
            "last_close": stock_df.loc[stock_df.index.max(), "Close"],
        }

        stock_name = ticker_to_name.get(ticker, "알수없음")

        filename = f"{ticker}_{stock_name}.parquet"
        filepath = output_dir / filename

        stock_df.to_parquet(filepath, compression="snappy", index=True)

    unique_ticker_count = len(all_tickers)
    col_count = stock_df.shape[1] if not all_tickers.empty else 0

    print("\n상장폐지 종목 검사 중...\n")

    end_date_dt = pd.to_datetime(end_date)
    delisting_threshold = end_date_dt - pd.Timedelta(days=DELISTING_THRESHOLD_DAYS)

    for ticker, metadata in tqdm(stock_metadata.items(), desc="상장폐지 검사 중"):
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
    print("\n특성 분류:")
    print("  - OHLCV: 7개 (종목명, Open, High, Low, Close, Volume, Change)")
    print("  - 수익률: 9개 (1d, 5d, 20d, 30d, 50d, 60d, 100d, 120d, 200d)")
    print(
        "  - 거래량: 7개 (vol_20, vol_60, vol_60_sqrt252, log_vol, vol_ratio_60, avg_log_vol_ratio_60, std_log_vol_ratio_60)"
    )
    print("  - 기술지표: 6개 (RSI_14, RSI_14_60avg, MFI_14, ATR_14, NATR_14, ADX_14)")
    print("  - 이격도: 4개 (5d, 20d, 60d, 120d)")
    print(
        "  - 리스크: 9개 (Mean_60d, Median_60d, Std_60d, Sharpe_60d, Sharpe_252d, Sortino_60d, Sortino_252d, Skewness_60d, Zscore_60d)"
    )
    print("  - 재무제표: 6개 (BPS, DPS, EPS, 배당수익률, 매출액, 영업이익)")
    print(
        "  - 파생지표: 8개 (PER, PBR, ROE, 배당성향, ROE_YoY, EPS_YoY, 영업이익_YoY, 매출액_YoY)"
    )
    print(f"  - 총합: {col_count}개 컬럼")
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
    print(f"📍 스크립트 위치: {SCRIPT_DIR}")
    print(f"📂 데이터 폴더 탐색 중: {DATA_DIR.absolute()}")

    if not DATA_DIR.exists():
        print(
            f"\n❌ 오류: '{DATA_DIR.absolute()}' 위치에서 'data' 폴더를 찾을 수 없습니다."
        )
        print(
            "💡 collect_create_data.py 파일과 같은 위치에 'data' 폴더를 만들고 .parquet 파일들을 넣어주세요."
        )
        return None

    parquet_files = list(DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        print("\n❌ 오류: 'data' 폴더는 존재하지만, 내부에 .parquet 파일이 없습니다!")
        return None

    print(f"\n✅ 데이터 폴더 발견: {DATA_DIR.absolute()}")
    print(f"✅ {len(parquet_files)}개의 parquet 파일을 찾았습니다.")

    all_stocks = []
    failed_files = []

    for file_path in tqdm(parquet_files, desc="parquet 파일 로드 중"):
        try:
            stock_df = pd.read_parquet(file_path)

            ticker_code = file_path.stem.split("_")[0]
            stock_df["Ticker"] = ticker_code

            all_stocks.append(stock_df)
        except Exception as e:
            failed_files.append((file_path.name, str(e)))

    if not all_stocks:
        print("\n❌ 읽을 수 있는 parquet 파일이 없습니다!")
        return None

    df_all = pd.concat(all_stocks, ignore_index=False)
    df_all.index = pd.to_datetime(df_all.index)
    df_all.index.name = "Date"
    df_all = df_all.set_index("Ticker", append=True)
    df_all = df_all.sort_index()

    print(f"\n✓ 멀티인덱스 DataFrame 생성 완료 (총 {len(df_all):,}행 로드)")

    if failed_files:
        print(f"\n⚠️  읽지 못한 파일: {len(failed_files)}개")
        for filename, error in failed_files[:5]:
            print(f"  - {filename}: {error[:50]}")
        if len(failed_files) > 5:
            print(f"  ... 외 {len(failed_files) - 5}개 더")

    print(f"\n{'=' * 80}")
    print(f"인덱스: {df_all.index.names}")
    print(f"컬럼 수: {len(df_all.columns)}개")
    print(f"{'=' * 80}")
    print("\n컬럼 목록:")
    for i, col in enumerate(df_all.columns, 1):
        print(f"{i}. {col}")
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


if __name__ == "__main__":
    print("=" * 80)
    print("주식 데이터 수집 및 처리 시스템")
    print("=" * 80)
    print("\n사용 가능한 함수:")
    print("1. data_download(start_date='2015-01-01', end_date='2024-12-31')")
    print("   - stock_list.csv에서 종목을 읽고 데이터를 수집하여 data 폴더에 저장")
    print("\n2. df = data_load()")
    print("   - data 폴더의 parquet 파일들을 로드하여 멀티인덱스 DataFrame 생성")
    print("\n3. result = data_query(df)")
    print("   - 날짜, 티커, 컬럼으로 데이터 필터링")
    print("=" * 80 + "\n")
