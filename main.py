# Basic Module
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# dependencies

# 처리 및 오류 로깅 모듈
from logging_config import logger

# 그리기 모듈
from plotter import Plotter

# 데이터 수집 및 가공
import collect_create_data 

# 군집화 알고리즘별 모듈
import kmeans_clustering
import gmm_clustering
import hdbscan_clustering
# import agglomerative_clustering 

def parse_arguments():

    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Argparse Test')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument(
        '--collect', '-c',
        type=bool,  
        default=False,
        help='True or False'
    )

    parser.add_argument(
        '--target', '-t',
        type=str,   
        default="KOSPI200",
        help='Stock or Index Name, default: KOSPI200'
    )

    parser.add_argument(
        '--start', '-s',
        type=str,
        default=(datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD), default: 1 year ago'
    )

    parser.add_argument(
        '--end', '-e',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD), default: today'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args    = parser.parse_args()

    return args
    

def main():
    
    args = parse_arguments()
    
    # 입력받은 인자값 출력
    logger.info(f"Args Data Collect : {args.collect}")
    logger.info(f"Args Target Index : {args.target}")
    logger.info(f"Args Start Day    : {args.start}")
    logger.info(f"Args End Day      : {args.end}")

    # # Set logging level
    if args.verbose: logger.setLevel('DEBUG')
    
    if args.collect : 
        logger.info(f"Start collecting")
        # 여기에 데이터를 가져오는 함수를 넣어주세요
        # 기초데이터를 가져온 후 추가적인 지표를 생성하는 경우 아래에 해당하는 함수를 별도 호출해주세요.
        logger.info(f"End   collecting")
    else :
        try:
            logger.info(f"Start loading data from collected already")
            # 사전 수집되고 생성된 데이터를 군집화 모듈에서 사용할 수 있도록 로딩
            # df = LoadData(StockList)
            logger.info(f"End   loading data")

            logger.info(f"Start KMeans clustering")
            # kms = KMeans(df)
            # ksm_report = kms.run()
            logger.info(f"End   KMeans clustering")

            logger.info(f"Start GMM clustering")
            # gmm = GMM(df)
            # gmm_report = gmm.run()
            logger.info(f"End   GMM clustering")

            logger.info(f"Start HDBScan clustering")
            # hdb = HDBScan(df)
            # hdb_report = hdb.run()
            logger.info(f"End   HDBScan clustering")

            logger.info(f"Gather and Summurize report")
            # reportAll(kms_report, gmm_report, hdb_report)

        except KeyboardInterrupt:
            logger.info("Analysis interrupted")
            sys.exit(1)

        except Exception as e:
            logger.error(f"Fatal error: {str(e)}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()
