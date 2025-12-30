# KMM (KMeans + PCA) Rolling Clustering Pipeline

`kmm_0.py` 한 파일만 실행하면, **여러 날짜(dates)**에 대해  
**(전처리 → PCA → KMeans → 결과 저장 → (옵션) 라벨 정렬(alignment) → (옵션) aligned PCA 맵 저장**  
까지 한 번에 돌아가도록 구성된 미니 파이프라인입니다.

---

## 1) 파일 구성 (5개)

- **kmm_0.py**: 실행 엔트리포인트(사용자가 수정하는 곳: `data_folder`, `dates`, `feature_cols`, 각종 파라미터)
- **kmeans_clustering.py**: `KMeansClusteringConfig` + `run_kmeans_clustering()` (전체 파이프라인 오케스트레이터)
- **kmm_pipeline.py**: 핵심 파이프라인(Parquet 로드, 스냅샷, 전처리, PCA, KMeans, rolling 저장)
- **kmm_align_clusters_mixed.py**: 날짜 간 클러스터 **라벨 번호**를 “겹침 + centroid 거리”로 정렬(permutaion 문제 완화)
- **kmm_pca_maps_aligned.py**: 정렬된 라벨 기준으로 날짜별 PCA 2D/3D 산점도 이미지 저장

---

## 2) 빠른 시작 (Quickstart)

### (1) 의존성 설치
아래는 최소 권장입니다.

```bash
pip install pandas numpy scikit-learn matplotlib pyarrow
```

- parquet 읽기에서 `pyarrow`가 안 되면:
  ```bash
  pip install fastparquet
  ```
- 라벨 정렬에서 **Hungarian(최적 할당)**을 쓰고 싶으면(권장, 선택):
  ```bash
  pip install scipy
  ```

### (2) 실행
```bash
python kmm_0.py
```

Windows에서 경로는 **raw string**(r"…")으로 두는 게 안전합니다.

---

## 3) 입력 데이터 규칙 (중요)

### Parquet 폴더 구조
`data_folder` 아래에 티커별 parquet 파일이 있어야 합니다.

- 파일명 관례: **`<TICKER>_....parquet`**  
  예) `005930_something.parquet`  
  → `<TICKER>`를 자동 추출하여 `Ticker` 컬럼에 넣습니다.

### Parquet 내부 데이터
각 parquet은 “일자별 row”가 있고 아래 중 하나여야 합니다.

- (A) **DatetimeIndex**가 날짜 인덱스이거나
- (B) `Date` 컬럼이 존재

> `kmm_pipeline.snapshot_by_date()`는 위 규칙으로 날짜를 필터합니다.

---

## 4) 파이프라인 흐름 (무슨 일이 일어나나)

`run_kmeans_clustering()` 내부 동작 요약:

1. **Parquet 전체 로드** → `df_all`
2. **rolling_cluster_table(dates)**  
   날짜별 반복:
   - 해당 날짜 스냅샷 생성
   - 피처 선택(`feature_cols`)
   - (옵션) 상관 높은 피처 제거(`corr_drop_mode`)
   - `inf/-inf → NaN` 치환 후 **중위값(median) 대치**
   - **Winsorize(quantile clip)** (`lower_q`, `upper_q`)
   - 스케일링: `variant="wins_std"` 또는 `"wins_rob"`
   - PCA:
     - cap = `pca_n` 까지 fit
     - 누적 설명력이 `pca_target_cum(기본 0.80)` 이상 되는 최소 차원 `n_used`를 골라 KMeans에 사용
   - KMeans:
     - `k_min..k_max` 평가 후 (실루엣 최대) `best_k` 선택
     - 최종 라벨 저장
   - (옵션) 시각화(elbow/silhouette) 저장
   - (옵션) seed 안정성(ARI/NMI) 저장
3. (옵션) **alignment**: 날짜 간 라벨 번호 정렬
4. (옵션) **aligned PCA map**: 정렬된 라벨로 2D/3D PCA 산점도 저장

---

## 5) kmm_0.py에서 주로 바꾸는 설정

### 필수
- `data_folder`: parquet 폴더 경로
- `dates`: ["YYYY-MM-DD", ...]
- `feature_cols`: 사용할 피처 컬럼 리스트

### 자주 튜닝하는 파라미터
- `lower_q`, `upper_q`: winsorize 분위수 (예: 0.01~0.99)
- `variant`:
  - `"wins_std"`: winsorize 후 StandardScaler
  - `"wins_rob"`: winsorize 후 RobustScaler
- `pca_n`: PCA 최대 차원(cap)
- `k_min`, `k_max`: 탐색할 k 범위
- `n_init`: KMeans 반복 초기화 횟수(안정성↑, 시간↑)
- `corr_drop_mode`:
  - `"baseline"`(권장): 첫 날짜에서 drop한 피처를 전체 날짜에 동일 적용(피처 일관성)
  - `"per_date"`: 날짜별로 상관 제거를 매번 새로(피처가 날짜마다 바뀔 수 있음)
  - `"none"`: 상관 제거 안 함
- `output_root`: 결과 저장 루트 폴더

### 라벨 정렬(alignment)
- `do_align=True/False`
- `w_overlap`, `w_centroid`: 비용함수 가중치(겹침 vs centroid)
- `min_jaccard`: 너무 겹침이 없을 때 매칭을 “약한 매칭/신규 클러스터”로 처리하는 기준

### PCA 맵 저장
- `do_pca_maps=True/False`
- `label_all`: 모든 종목 티커 라벨 표시 여부(많으면 지저분해질 수 있음)

---

## 6) 출력물 설명 (어디에 뭐가 저장되나)

실행하면 `output_root` 아래에 이런 폴더가 생깁니다:

```
kmeans_outputs/
  rolling_<start>_to_<end>_<timestamp>/
    rolling_run_config.json
    baseline_corr_dropped_cols.csv               (corr_drop_mode="baseline"일 때)
    rolling_meta_by_date.csv                     (날짜별 best_k, silhouette 등)
    rolling_stability_by_date.csv                (옵션: seed 안정성 요약)
    cluster_labels_long.csv                      (date, Ticker, cluster)
    cluster_labels_wide.csv                      (Ticker x date)
    cluster_labels_long.parquet
    cluster_labels_wide.parquet
    rolling_meta_by_date.parquet

    per_date/
      <DATE>/
        labels_<DATE>.csv
        k_eval_<DATE>.csv
        pca_ev_<DATE>.csv
        pca_loadings_<DATE>.csv
        pca_coords_used_<DATE>.csv               (KMeans에 실제 사용한 PCA 좌표)
        centroids_<DATE>.csv                     (PCA 공간 centroid)
        winsor_clip_table_<DATE>.csv
        corr_dropped_cols_<DATE>.csv
        stability_seed_<DATE>.csv                (옵션)
        elbow_<DATE>.png                         (옵션)
        silhouette_<DATE>.png                    (옵션)

    # (do_align=True일 때)
    cluster_labels_wide_aligned.csv
    cluster_alignment_log.csv
    cluster_transitions_long.csv
    cluster_churn_by_date.csv

    # (do_pca_maps=True일 때)
    pca_maps_aligned/
      pca2d_aligned_<DATE>.png
      pca3d_aligned_<DATE>.png                   (PC3가 있을 때만)
```

---

## 7) 흔한 에러/주의사항

- **parquet 로드 실패**
  - `pyarrow` 설치 확인
  - 특정 파일만 실패하면 `fastparquet`로 fallback 가능(설치 필요)

- **alignment에서 SciPy 없음**
  - SciPy가 없으면 내부에서 **greedy fallback**을 사용합니다(최적은 아닐 수 있음).

- **PCA map 생성 실패**
  - `per_date/<DATE>/pca_coords_used_<DATE>.csv` (또는 `pca_coords_plot_<DATE>.csv`)가 있어야 합니다.
  - dates 리스트와 per_date 폴더/파일명이 정확히 일치해야 합니다.

- **속도**
  - `n_init`이 크고 `k_max`가 크면 시간이 급격히 늘어납니다.
  - Windows에서 MKL 스레드 폭주를 막기 위해 `OMP_NUM_THREADS="1"`을 기본으로 둡니다.

---

## 8) 해석 팁 (추천 워크플로우)

1) `rolling_meta_by_date.csv`로 날짜별 `best_k`와 실루엣이 안정적인지 확인  
2) `cluster_labels_wide.csv` vs `cluster_labels_wide_aligned.csv` 비교  
3) `cluster_churn_by_date.csv`로 “군집 전이(churn)”이 큰 날짜를 포착  
4) 그 날짜의 `pca2d_aligned_<DATE>.png`를 열고, 어떤 종목들이 어디로 이동했는지 확인  
5) `per_date/<DATE>/pca_loadings_<DATE>.csv`에서 PC별 로딩(기여 피처)로 군집 의미를 해석

---

## 9) 실행 예시(샘플)

`kmm_0.py`에서 대략 이런 형태로 세팅합니다(실제 값은 환경에 맞게 변경):

```python
data_folder = r"C:\Users\...\data"
dates = ["2024-12-02", "2024-12-03", "2024-12-04"]
feature_cols = ["Return_20d", "Return_60d", "vol_20", "vol_60", "PER", "ROE"]
```

---

## 10) 라이선스/면책
본 코드는 연구/학습 목적의 분석 파이프라인 예시입니다.  
투자 판단은 본인 책임입니다.
