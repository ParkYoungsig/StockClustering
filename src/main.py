# Basic Module
import argparse
import sys
# from datetime import datetime, timedelta

# 데이터 수집 및 가공
import collect_create_data

# 군집화 알고리즘별 모듈
from gmm_clustering import *
from hdb_clustering import *
from kmm_clustering import *

# dependencies
# 처리 및 오류 로깅 모듈
from lib.logging_config import logger

# 그리기 모듈

# 데이터 수집 및 가공

# 군집화 알고리즘별 모듈
# import agglomerative_clustering


def parse_arguments():
    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description="Argparse Test")

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument(
        "--collect", "-c", type=bool, default=False, help="True or False"
    )

    # parser.add_argument(
    #     "--target",
    #     "-t",
    #     type=str,
    #     default="KOSPI200",
    #     help="Stock or Index Name, default: KOSPI200",
    # )

    # parser.add_argument(
    #     "--start",
    #     "-s",
    #     type=str,
    #     default=(datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d"),
    #     help="Start date (YYYY-MM-DD), default: 1 year ago",
    # )

    # parser.add_argument(
    #     "--end",
    #     "-e",
    #     type=str,
    #     default=datetime.now().strftime("%Y-%m-%d"),
    #     help="End date (YYYY-MM-DD), default: today",
    # )

    # parser.add_argument(
    #     "--verbose", "-v", action="store_true", help="Enable verbose logging"
    # )

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    # 입력받은 인자값 출력
    logger.info(f"Args Data Collect : {args.collect}")
    # logger.info(f"Args Target Index : {args.target}")
    # logger.info(f"Args Start Day    : {args.start}")
    # logger.info(f"Args End Day      : {args.end}")

    # # Set logging level
    # if args.verbose:
    #     logger.setLevel("DEBUG")

    if args.collect:
        logger.info("Start collecting")
        try:
            # collect_create_data.data_download(start_date=args.start, end_date=args.end)
            collect_create_data.data_download()
            logger.info("End   collecting")
        except Exception as e:
            logger.error(f"Data collection failed: {str(e)}", exc_info=True)
            sys.exit(1)
    else:
        try:
            logger.info("Start loading data from collected already")
            df = collect_create_data.data_load()

            if df is None:
                logger.error("Failed to load data. Please run with --collect first.")
                sys.exit(1)

            logger.info("End   loading data")

            # logger.info("Start KMeans clustering")
            # cfg = init_kmeans_clustering()
            # kmm_report = run_kmeans_clustering(cfg)
            # logger.info("End   KMeans clustering")

            logger.info("Start GMM clustering")
            gmm = GMM(df)
            gmm_report = gmm.run()
            logger.info("End   GMM clustering")

            # logger.info("Start HDBScan clustering")
            # df = init_hdb_clustering()
            # run_hdb_clustering(df)
            # logger.info("End   HDBScan clustering")

            # logger.info("Gather and Summurize report")
            # # reportAll(kms_report, gmm_report, hdb_report)

        except KeyboardInterrupt:
            logger.info("Analysis interrupted")
            sys.exit(1)

        except Exception as e:
            logger.error(f"Fatal error: {str(e)}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()