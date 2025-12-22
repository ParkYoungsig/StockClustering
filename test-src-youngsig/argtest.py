import argparse
from datetime import datetime, timedelta


def parse_arguments():

    # 인자값을 받을 수 있는 인스턴스 생성
    parser = argparse.ArgumentParser(description='Argparse Test')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument(
        '--gattering', '-g',
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

    args    = parser.parse_args()

    return args
    

def main() :
    # args 에 위의 내용 저장
    
    args = parse_arguments()

    # 입력받은 인자값 출력
    print(args.gattering)
    print(args.target)
    print(args.start)
    print(args.end)

if __name__ == "__main__":
    main()
