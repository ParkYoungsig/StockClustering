# '/src' 내 파일 및 폴더 내역 및 설명 

## 파일 내역

- main.py: 프로그램의 데이터 수집/적재와 군집화 알고리즘 수행을 통제하는 핵심 로직
- config.py: 데이터 수집/적재와 각 군집화 알고리즘에서 사용하는 설정정보 파일
- collect_create_data.py: 데이터 수집/가공/저장/적재를 위한 모듈, readme 파일 'load_data' 폴더에 존재
- gmm_clustering.py: 군집화 알고리즘 gmm 주처리 모듈, readme 파일 및 하위 모듈은 'gmm' 폴더에 존재
- hdb_clustering.py: 군집화 알고리즘 dbscan 주처리 모듈, readme 파일 및 하위 모듈은 'hdb' 폴더에 존재
- kmm_clustering.py: : 군집화 알고리즘 kmeans 주처리 모듈, readme 파일 및 하위 모듈은 'kmm' 폴더에 존재

## 폴더(디렉터리) 내역

- gmm: 군집화 gmm 처리를 위한 하위 모듈 및 reafme 파일 저장 위치
- hdb: 군집화 dbscan 처리를 위한 하위 모듈 및 reafme 파일 저장 위치
- kmm: 군집화 kmeans 처리를 위한 하위 모듈 및 reafme 파일 저장 위치
- lib: main 및 기타 모듈에소 공통으로 사용하는 라이브러리 파일 저장 위치
- load_data:  데이터 수집/가공/저장/적재 모듈 readme 파일 위치
- utils: 프로젝트 수행과 직접 연관성은 없는 유틸리티성 파이썬 프로그램 저장위치
