# 실행방법 및 파일내역 소개

## 실행 방법

- 데이터 수집: 최초 1회만 수행, 데이터 셋을 바꾸려는 경우에 재수행 필요

    D:\StockClustering> down.bat

- 군집화 수행: 알고리즘 수행

    D:\StockClustering> run.bat

## 파일 내역

- ProjectOverview.md: 프로젝트 개요
- down.bat: 종목 리스트에 지정된 종목들에 대한 정보 수집 실행, (데이터 구간 : 2015-01-01 ~ 2024-12-31)
- down2now.bat: 종목 리스트에 지정된 종목들에 대한 정보 수집 실행, (데이터 구간 : 2015-01-01 ~ 2025-12-31)
- run.bat: 수집된 정보를 기반으로 알고리즘별로 군집화하고 결과 수집 실행
- mkreq.bat: Requirements.txt 파일 생성
- ptree.bat: ProjectRee.txt 파일 생성
- readme.md: 본 파일(root 레벨 리드미 파일)
- requirements.txt: 파이썬 모듈 설치 정보
- ProjectTree.txt: 전체 폴더 및 파일 리스팅
- .gitattribute: github 동기화를 위한 속성 설정 파일
- .gitignore: github동기화를 위한 예외 설정 파일

    /
    ├── PrrojectOverview.md
    ├── down.bat
    ├── down2now.bat
    ├── mkreq.bat
    ├── run.bat
    ├── ProjectTree.txt
    ├── ptree.bat
    ├── readme.md
    ├── requirements.txt
    ├── .gitattribute
    ├── .gitignore

## 폴더(디렉터리) 내역

- data: 수집된 주식별 데이터 파일(.parquet)들 저장 폴더
- git-command: 기본적인 git 명령어 모음과, 상황별 사용방법 방법 소개
- input: 수집 대상 종목리스트 및 결과 리스트 저장 폴더
- log: 수행 진행상황을 로깅하는 파일 저장 폴더, 일자별로 생성
- output: 각 모듈별 결과물을 보관하는 폴더
- src: 프로젝트 수행을 위한 소스 폴더

    /
    ├── data/
    ├── git-command/
    ├── input/
    ├── log/
    ├── output/
    └── src/
