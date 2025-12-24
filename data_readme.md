# collect_create_data.py 함수 설명

## 1. def data_download(start_date="2015-01-01", end_date="2024-12-31")

**주식 데이터를 다운로드하고 모든 지표를 계산해서 parquet 파일로 저장**

- list.csv에서 종목 코드 읽기
- FinanceDataReader로 주가 데이터(OHLCV) 다운로드
- 기술적 지표 35개 계산 (수익률, 거래량, RSI/MFI/ATR/ADX, 이격도, 리스크)
- BPS_DPS_PER.xlsx에서 재무제표 데이터 로드 및 병합
- PER, PBR, ROE, YoY 성장률 등 파생 지표 8개 계산
- data 폴더에 종목별 parquet 파일 저장 (`{티커}_{종목명}.parquet`)
- data 폴더가 이미 존재하면 data (1), data (2), ... 순서대로 새 폴더 생성

---

## 2. def data_load()

**data 폴더의 parquet 파일들을 하나의 DataFrame으로 로드**

- data 폴더의 모든 .parquet 파일 검색
- 개별 파일을 읽어서 하나의 DataFrame으로 결합
- Date, Ticker 멀티인덱스 구조로 반환
- 컬럼 목록과 로드 실패 파일 정보 출력

---

## 3. def data_query(df)

**DataFrame에서 날짜/티커/컬럼 기준으로 대화형 조회**

- 시작 날짜, 종료 날짜 입력받기 (엔터 = 전체)
- 티커 목록 입력받기 (쉼표 구분, 엔터 = 전체)
- 컬럼 목록 입력받기 (쉼표 구분, 엔터 = 전체)
- 조건에 맞는 데이터 필터링해서 반환
- 결과의 처음 10행, 마지막 10행 출력
