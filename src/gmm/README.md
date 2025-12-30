# GMM 모듈 (src/gmm)

이 폴더는 **주식(또는 지수) 팩터 데이터를 스냅샷(연말/월말) 단위로 만들고**, 연도별로 **Gaussian Mixture Model(GMM)** 을 학습한 뒤 **연도 간 클러스터 라벨을 정렬**하여 “시장 레짐(상태)”을 일관되게 추적하는 파이프라인의 핵심 모듈을 담고 있습니다.

- 실행 진입점(엔트리)은 리포지토리 루트의 `gmm_clustering.py`이며, 본 폴더는 그 내부에서 import 되어 사용됩니다.
- 단독 검증용 스크립트는 리포지토리 루트의 `test_my_gmm.py`입니다.

---

## 폴더 구성

- `config.py`
  - 파이프라인 설정 상수 모음
  - 스냅샷 주기(`SNAPSHOT_FREQ`), 연도 범위(`START_YEAR`, `END_YEAR`), K 탐색 범위(`K_RANGE`), 노이즈 필터 임계값(`MIN_CLUSTER_FRAC`) 등
  - 시각화/리포트용 클러스터 이름(`CLUSTER_NAMES`)과 해석 문구(`CLUSTER_INTERPRETATIONS`)도 여기서 관리

- `data_loader.py`
  - 데이터 로딩(3단 방어) + 스냅샷 변환 + 기본 피처 클리닝/스케일링
  - **우선순위 로딩**
    1) `data/merged_stock_data.parquet` (로컬 병합본)
    2) 로컬 `data/*.parquet` 원본들을 즉석 병합
    3) Hugging Face 데이터셋 fallback (`yumin99/stock-clustering-data`)

- `preprocessor.py`
  - 모델 입력 피처 전처리(결측 제거, 이상치 제거, 스케일링, (옵션) PCA)
  - 현재 `gmm_clustering.py`에서는 `use_pca=False`로 사용하도록 구성되어 있습니다.

- `model.py`
  - BIC 기반 K 후보 평가, 안정성(stability) 평가, 라벨 정렬(헝가리안 매칭/정렬) 등 수학/알고리즘 유틸

- `pipeline_logic.py`
  - 파이프라인의 “핵심 학습 로직”
  - K 선택(`select_best_k`) 및 **연도별 학습 + 연도 간 라벨 정렬 + warm start**(`train_gmm_per_year`) 담당

- `postprocessor.py`
  - 최신 연도 프레임 구성(`get_latest_year_frame`)
  - 작은 군집을 노이즈(-1)로 처리하는 필터(`filter_noise`)
  - 클러스터별 평균/표준편차/개수 집계(`compute_cluster_stats`)

- `reporter.py`
  - 텍스트 리포트(`report.txt`) 생성
  - 멤버 리스트 생성(전기간 / 연도별) 및 최신 연도 기준 Top 티커 요약
  - 시간 태그 정책: **월 스냅샷이면 `YYYY-MM(YearMonth)`**, 연 스냅샷이면 **`YYYY(Year)`**

- `report_metrics.py`
  - 리포트에 넣기 위한 품질 지표(예: 소속확률 기반 신뢰도, 실루엣, 전이/유지율, 사후 성과 요약 등) 계산

- `visualizer.py`
  - BIC/안정성 곡선, heatmap/radar/parallel plot, sankey, UMAP 등 시각화 저장 유틸
  - Plotly(선택), UMAP(선택) 등 설치 여부에 따라 일부 그래프가 자동 skip 될 수 있습니다.

---

## 데이터/스냅샷 규칙

### 필수 컬럼

- 최소 요구: `Date`
- 식별자: `Ticker` (없으면 일부 로직이 `Code`로 fallback)
- 스냅샷 변환 후: `Year`, `Month`가 생성됩니다.

### 스냅샷 주기(`SNAPSHOT_FREQ`)

`data_loader.convert_df_to_snapshots()` 기준:

- `"M"` (월말 스냅샷)
  - 각 Ticker별로 월(YearMonth)마다 마지막 관측치 1개를 선택
  - 결과 DF에 `YearMonth`를 **항상 `YYYY-MM` 문자열로 생성**

- `"Y"` (연말 스냅샷)
  - 각 Ticker별로 연(Year)마다 마지막 관측치 1개를 선택
  - 혼선 방지를 위해 결과 DF에서 `YearMonth` 컬럼이 있으면 제거

---

## 사용 피처(기본)

`data_loader.py`의 `FEATURE_COLUMNS` 기준(존재하는 컬럼만 사용):

- `Return_120d`, `Return_20d`
- `ADX_14`, `MFI_14`
- `Disparity_60d`, `vol_60_sqrt252`, `NATR`
- `Sharpe_60d`, `Sortino_60d`, `Zscore_60d`

참고:
- 입력에 `NATR`가 없고 `NATR_14`만 있으면 `NATR`을 자동 보강합니다.

---

## 파이프라인 흐름(요약)

1) **로드/스냅샷**
- `data_loader.load_snapshots()`
- 로드 후 `convert_df_to_snapshots()`로 연말/월말 스냅샷 데이터 생성

2) **전처리**
- `preprocessor.preprocess_features()`
- 결측 제거 + (옵션) IsolationForest로 이상치 제거 + 스케일링
- 현재 엔트리(`gmm_clustering.py`)에서는 `use_pca=False`로 사용

3) **학습/정렬**
- `pipeline_logic.train_gmm_per_year()`
  - 연도별로 GMM 학습
  - 첫 연도는 특정 피처 평균 기준으로 라벨 정렬
  - 이후 연도는 **헝가리안 매칭으로 중심을 정렬**해 라벨 일관성 유지
  - 이전 연도의 중심을 이용한 warm start(`means_init`)를 적용

4) **후처리/리포트/시각화**
- `postprocessor.get_latest_year_frame()`로 최신 연도 프레임 구성
- `postprocessor.filter_noise()`로 너무 작은 군집을 노이즈 처리
- `reporter.write_text_report()`로 `report.txt` 생성
- `visualizer.py`에서 주요 그래프 생성

---

## 산출물(결과 파일)

기본적으로 결과는 `gmm_clustering.GMM`이 사용하는 `results_dir`(기본: `gmm_results/`) 아래에 저장됩니다.

주요 산출물:

- `final_clustered_data.csv`
  - 최신 연도 기준 유효 샘플(`cluster != -1`)에 클러스터 라벨 포함

- `final_probabilities_latest.csv`
  - 최신 연도 기준 각 샘플의 클러스터 소속확률(soft assignment)

- `cluster_members_all_years.csv`
  - **전 기간**(연/월 스냅샷 전체) 기준 클러스터 멤버 리스트
  - 멤버 표기는 `Ticker (Name, YYYY)` 또는 `Ticker (Name, YYYY-MM)` 형태

- `cluster_members_by_year.csv`
  - **연도별** 멤버 요약(팀 프로젝트 공유용)
  - 컬럼: `year, cluster, member`

- `report.txt`
  - 데이터/전처리/선택된 K/클러스터 평균/품질지표/전이 요약 등 텍스트 리포트
  - “Top 티커” 섹션은 **최신 연도** 기준입니다.

- `artifacts/`
  - `scaler.pkl`: 전처리 스케일러(또는 그룹별 스케일러)
  - `metadata.pkl`: `labels_per_year`, `final_k`, `feature_columns`, `cluster_means_latest` 등
  - `gmm_latest_year.pkl`: 최신 연도 모델(있을 때만)

---

## 실행 방법(권장)

현재 `gmm_clustering.py`는 함수/클래스를 제공하는 엔트리 파일이며, `__main__` 실행 블록이 없어서 파일을 직접 실행해도 아무 것도 수행하지 않을 수 있습니다.

가장 간단한 실행은 단독 테스트 스크립트를 쓰는 것입니다.

```bash
# 프로젝트 루트에서
python test_my_gmm.py
```

직접 코드로 호출하려면:

```python
from pathlib import Path
from src.gmm.data_loader import load_snapshots
from gmm_clustering import GMM

df, stats = load_snapshots(data_dir=Path("data"))
gmm = GMM(df, results_dir=Path("gmm_results"))
print(gmm.run(manual_k=4))
```

---

## 의존성

필수(요약):
- numpy, pandas, scikit-learn, scipy
- matplotlib, seaborn
- pyarrow(Parquet), joblib
- datasets / huggingface_hub(HF fallback 로딩)

선택:
- umap-learn(UMAP 시각화)
- plotly(Sankey)

`requirements.txt`를 기준으로 설치하세요.
