# GMM 모듈 (src/gmm)

이 폴더는 **주식(또는 지수) 팩터 데이터를 스냅샷(연말/월말) 단위로 만들고**, 연도별로 **Gaussian Mixture Model(GMM)** 을 학습한 뒤 **연도 간 클러스터 라벨을 정렬**하여 “시장 레짐(상태)”을 일관되게 추적하는 파이프라인의 핵심 모듈을 담고 있습니다.

- 실행 진입점은 `src/gmm_clustering.py`의 `GMM` / `run_gmm_pipeline()`이며, 본 폴더는 그 내부에서 import 되어 사용됩니다.
- 팀 프로젝트의 통합 엔트리는 `src/main.py`입니다.

---

## 최근 업데이트(핵심 요약)

최근 반영된 주요 변경/추가 사항은 다음과 같습니다.

- **K=4 고정(실제 학습)**: `GMM.run(manual_k=4)`가 기본이며 최종 산출물은 K=4로 생성됩니다.
- **K 선택 근거는 별도 계산 유지**: 실제 학습이 K=4로 고정이어도, `K_RANGE(2~9)`에 대한 **BIC(mean) / Silhouette(mean)** 는 항상 계산/차트 저장됩니다.
- **Robustness 검증 기능 추가/정리**
  - (1) 학습 윈도우(최근 N년) 변경 시 최신 연도 평가셋에서의 ARI/NMI 비교
  - (2) 기간 슬라이싱 케이스별 재학습 + centroid 유사도(헝가리안 매칭 후 평균) 비교
  - (3) **36개월(3년) 롤링 + 1년 스텝, 8개 윈도우** centroid 유사도 heatmap + MD 요약
- **코드 정리**: 기간/롤링 robustness 로직을 `robustness.py`로 분리하여 `gmm_clustering.py` 비대화를 완화했습니다.
- **리포트 확장**: `gmm_report.txt`에 rolling robustness(3년 롤링 8개 윈도우) 요약 섹션이 포함됩니다.

---

## 폴더 구성

- `config.py` (프로젝트 전역: `src/config.py`)
  - 파이프라인 설정 상수 모음
  - 스냅샷 주기(`SNAPSHOT_FREQ`), 연도 범위(`START_YEAR`, `END_YEAR`), K 탐색 범위(`K_RANGE`), 노이즈 필터 임계값(`MIN_CLUSTER_FRAC`) 등
  - 시각화/리포트용 클러스터 이름(`CLUSTER_NAMES`)과 해석 문구(`CLUSTER_INTERPRETATIONS`)도 여기서 관리
  - 결과/데이터 기본 경로는 `PROJECT_ROOT` 기준으로 `output/`, `data/`를 사용합니다.

- `data_loader.py`
  - 데이터 로딩(3단 방어) + 스냅샷 변환 + 기본 피처 클리닝/스케일링
  - **우선순위 로딩**
    1) `data/merged_stock_data.parquet` (로컬 병합본)
    2) 로컬 `data/*.parquet` 원본들을 즉석 병합
    3) Hugging Face 데이터셋 fallback (`yumin99/stock-clustering-data`)
  - 로컬 원본 `data/*.parquet`은 보통 **인덱스가 `Date`만 있고 `Ticker` 컬럼이 없을 수 있는데**, 이 경우 파일명(`{티커}_{종목명}.parquet`)에서 티커를 추출해 `Ticker` 컬럼을 보강합니다.

- `processer.py`
  - 모델 입력 피처 전처리(결측 제거, 이상치 제거, 스케일링, (옵션) PCA)
  - 최신 연도 프레임 구성, 노이즈 필터, 클러스터 통계 집계 포함

- `model.py`
  - BIC/Silhouette 평가 유틸, 라벨 정렬(헝가리안 매칭/정렬), robustness(ARI/NMI) 계산 등 수학/알고리즘 유틸

- `robustness.py`
  - robustness(기간 변화/롤링 윈도우) 검증 로직 모음
  - `gmm_clustering.py`에서 토글 설정에 따라 호출되어 산출물(heatmap/MD/txt/CSV)을 저장

- `pipeline_logic.py`
  - 파이프라인의 “핵심 학습 로직”
  - K 선택(`select_best_k`) 및 **연도별 학습 + 연도 간 라벨 정렬 + warm start**(`train_gmm_per_year`) 담당


- `reporter.py`
  - 텍스트 리포트(`gmm_report.txt`) 생성
  - 멤버 리스트 생성(전기간 / 연도별) 및 최신 연도 기준 Top 티커 요약
  - 시간 태그 정책: **월 스냅샷이면 `YYYY-MM(YearMonth)`**, 연 스냅샷이면 **`YYYY(Year)`**
  - rolling robustness 요약(3년 롤링 8개 윈도우)을 리포트에 포함

- `report_metrics.py`
  - 리포트에 넣기 위한 품질 지표(예: 소속확률 기반 신뢰도, 실루엣, 전이/유지율, 사후 성과 요약 등) 계산

- `visualizer.py`
  - BIC(mean)/Silhouette(mean) 곡선, heatmap/radar/parallel plot, sankey, UMAP 등 시각화 저장 유틸
  - Plotly(선택), UMAP(선택) 등 설치 여부에 따라 일부 그래프가 자동 skip 될 수 있습니다.

---

## 데이터/스냅샷 규칙

### 필수 컬럼

- 최소 요구: `Date`
- 식별자: `Ticker` (파이프라인에서 강하게 권장)
- 스냅샷 변환 후: `Year`, `Month`가 생성됩니다.

참고:
- `load_snapshots()`의 “로컬 원본 즉석 병합” 경로는 원본이 `Date` 인덱스만 가진 형태여도 파일명 기반으로 `Ticker`를 만들어 표준 스키마(`Date`, `Ticker` 컬럼)로 정규화합니다.
- 반대로, 외부에서 `GMM(df=...)`로 **DataFrame을 직접 주입**하는 경우에는 파일명 힌트가 없으므로 `Ticker`가 컬럼 또는 멀티인덱스 레벨로 **이미 포함되어 있어야** 합니다.

### 스냅샷 주기(`SNAPSHOT_FREQ`)

`data_loader.convert_df_to_snapshots()` 기준:

 # GMM 모듈 (src/gmm) — 간단 안내

 이 폴더는 연/월 스냅샷을 만들어 연도별로 GMM을 학습하고, 연도 간 라벨을 정렬해 시장 상태(regime)를 추적합니다.

 핵심 요약
 - 리포트: Markdown 단일 파일(`gmm_report.md`)만 생성합니다.
 - K 선택 근거: BIC(mean) + Silhouette(mean) (AIC/Stability 제거).
 - 전이(persistence) 출력은 리포트에서 제외했습니다. 다만 ex-post(Forward return / Drawdown) 요약은 포함됩니다.
 - 출력 구조: results root에는 핵심 8개만 남기고 나머지는 `gmm_appendix/`로 이동합니다.

 주요 파일
 - 진입점: `src/gmm_clustering.py` (`GMM` / `run_gmm_pipeline()`)
 - 주요 모듈: `data_loader.py`, `processer.py`, `pipeline_logic.py`, `model.py`, `visualizer.py`, `report_metrics.py`, `reporter.py`, `robustness.py`

 출력(요약)
 - Core (results root)
   - `gmm_bic_curve_mean.png`, `gmm_silhouette_curve_mean.png`, `gmm_cluster_boxplots.png`,
     `gmm_heatmap.png`, `gmm_robustness_vs_window.png`, `gmm_sankey.html`, `gmm_umap.png`,
     `gmm_report.md`
 - Appendix (`gmm_appendix/`)
   - `gmm_cluster_members_by_year.csv` (공유용 단일 CSV)
   - 기타 시각화/아티팩트(UMAP/parallel/radar, rolling/period robustness outputs, `gmm_artifacts/`)

 CSV 포맷 (`gmm_cluster_members_by_year.csv`)
 - 행: `year`(한 행 = 한 연도)
 - 컬럼: `year`, `cluster_0`, `cluster_1`, ...
 - 셀 값: 줄바꿈으로 구분된 `종목명 티커 (YYYY)` 리스트

 실행
 - 권장 (프로젝트 루트):
 ```bash
 python -m src.main
 ```
 - 직접 호출 예시:
 ```python
 from pathlib import Path
 from gmm_clustering import GMM

 gmm = GMM(df, results_dir=Path("./output"))
 print(gmm.run(manual_k=4))  # 기본은 K=4
 ```

 Robustness
 - 기간/롤링 robustness는 `src/config.py` 토글로 제어됩니다. period-slicing은 옵션이며 실행 시간이 길 수 있습니다.

 기타
 - `report_metrics.py`는 Silhouette 및 probability-based quality 지표와 ex-post(Forward return/drawdown)를 계산해 반환합니다.
 - 일부 시각화는 선택 패키지(UMAP, Plotly) 설치 여부에 따라 자동으로 생략됩니다.

 의존성
 - 필수: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `joblib`, `pyarrow`
 - 선택: `umap-learn`, `plotly`

 원하시면 이 README를 더 짧게 요약한 버전으로도 만들어드릴게요.

즉, 실행 위치가 `StockClustering` 폴더여도 **기본 결과는 Team_Project 레벨의 `output/`**로 떨어지는 것이 정상입니다.

모든 산출물 파일명에는 접두사 `gmm_`가 붙습니다.

주요 산출물:

- `gmm_cluster_members_by_year.csv`
  - **제출/공유용 멤버 요약(유일 CSV)**
  - 포맷: `year` + `cluster_0~cluster_3` 컬럼
  - 각 셀: `종목명 티커 (YYYY)` 형태로 줄바꿈 리스트
  - 저장 위치: `gmm_appendix/` (루트는 핵심 8개 산출물만 유지)

- `gmm_report.txt`
  - 데이터/전처리/선택된 K/클러스터 평균/품질지표/전이 요약 등 텍스트 리포트
  - “Top 티커” 섹션은 **최신 연도** 기준입니다.

- `gmm_artifacts/`
  - `gmm_scaler.pkl`: 전처리 스케일러
  - `gmm_metadata.pkl`: `labels_per_year`, `final_k`, `feature_columns`, `cluster_means_latest` 등
  - `gmm_latest_year.pkl`: 최신 연도 모델(있을 때만)

- `gmm_appendix/`
  - 제출/공유에 필수적이지 않은 부가 산출물(UMAP/parallel/radar, pairwise robustness 등)이 저장됩니다.

추가 산출물(근거/robustness):
- `gmm_bic_curve_mean.png`, `gmm_silhouette_curve_mean.png`
  - K 선택 근거를 위한 차트입니다(요구사항: mean curve 2개만 유지).
  - **중요:** 실제 클러스터링은 기본 `K=4`로 고정이지만, 위 근거 차트는 별도로 계속 계산/저장됩니다.
- `gmm_robustness_vs_window.png`, `gmm_robustness_pairwise_ari.png`, `gmm_robustness_pairwise_nmi.png`
  - 학습 윈도우(최근 N년)를 바꿔도 최신 연도 평가셋에서 라벨이 얼마나 유지되는지(ARI/NMI) 평가합니다.

- 기간 슬라이싱(케이스별) robustness:
  - 케이스별 prefix로 저장됩니다.
    - 예: `gmm_period_A_full_bic_curve_mean.png`, `gmm_period_B_recent_cluster_means_latest.csv` 등
  - 케이스 간 centroid 비교 heatmap:
    - `gmm_period_centroid_cosine_mean.png`
    - `gmm_period_centroid_corr_mean.png`
  - 요약 텍스트:
    - `gmm_period_robustness_report.txt`

- 36개월(3년) 롤링 + 1년 스텝(8개 윈도우), K 고정 robustness:
  - 윈도우별 최신 연도 centroid/개수:
    - 예: `gmm_rolling_2015_2017_cluster_means_latest.csv`, `gmm_rolling_2015_2017_cluster_counts_latest.csv`
  - 윈도우 간 centroid 유사도 heatmap(헝가리안 매칭 후 평균):
    - `gmm_rolling_centroid_cosine_mean.png`
    - `gmm_rolling_centroid_corr_mean.png`
  - 실행 요약(MD):
    - `gmm_rolling_robustness.md`

---

## 실행 방법(권장)

가장 일반적인 실행:

```bash
# 프로젝트 루트에서
python -m src.main
```

직접 GMM만 호출하려면:

```python
from pathlib import Path
from gmm_clustering import GMM

gmm = GMM(df, results_dir=Path(".\\output"))
print(gmm.run(manual_k=4))
```

### K=4 고정 정책

- 실제 모델 학습/라벨링(최종 산출물)은 기본적으로 `manual_k=4`를 사용합니다.
  - `gmm_clustering.GMM.run()`의 기본값이 `manual_k=4`입니다.
- 다만 `K_RANGE(range(2, 10))` 기반의 **BIC(mean) / Silhouette(mean)** 는 **별도로 계산**하여 리포트/차트에 함께 저장합니다.

### Robustness(기간/윈도우 변경 검증)

`src/config.py`에서 토글로 켤 수 있습니다.

- 학습 윈도우(최근 N년) 변화 평가(ARI/NMI):
  - `ROBUSTNESS_WINDOW_YEARS`, `ROBUSTNESS_EXCLUDE_EVAL_YEAR`
- 기간 슬라이싱 비교(케이스별 재학습):
  - `ROBUSTNESS_PERIOD_SLICING_ENABLED`, `ROBUSTNESS_PERIOD_CASES`
- 36개월 롤링(1년 스텝) 8개 윈도우, K=4 고정:
  - `ROBUSTNESS_ROLLING_WINDOWS_ENABLED`, `ROBUSTNESS_ROLLING_WINDOWS`, `ROBUSTNESS_ROLLING_K_FIXED`

참고:
- rolling(3년 롤링) 모드는 메인 리포트(`gmm_report.txt`)에도 요약 통계(윈도우 간 off-diagonal mean/median)가 함께 출력됩니다.
- 기간 슬라이싱(period slicing) 모드는 케이스 수만큼 재학습이 수행되므로 시간이 오래 걸 수 있어 기본 OFF입니다.

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
