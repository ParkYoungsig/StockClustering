# 📊 시장 지도 (Market Map) - DBSCAN 클러스터링

## 🎯 한 줄 설명
한국 주식들을 **배당**과 **수익률**로 분류하는 자동 시각화 도구.

---

## 🚀 빠른 시작 (3단계)

### 1️⃣ 설치
```bash
pip install pandas numpy matplotlib seaborn scikit-learn adjustText
```

### 2️⃣ 준비물
- `data/` 폴더에 **parquet 파일들**
- **NanumGothic.ttf** (프로젝트 루트)

### 3️⃣ 실행
```bash
python dbscan_clean_final.py
>> 분석할 날짜 (YYYY-MM-DD): 2024-05-30
```

---

## 📁 폴더 구조
```
StockClustering/
├── dbscan_clean_final.py
├── NanumGothic.ttf
├── data/           # Parquet 파일들
└── output/         # 결과 (PNG, CSV)
```

---

## 🎨 4가지 군집
| 이름 | 설명 |
|------|------|
| 💎 배당+성장 | 배당도 높고 수익률도 높음 |
| 🛡️ 고배당 | 배당 높지만 수익률 낮음 |
| 🚀 고성장 | 배당 낮지만 수익률 높음 |
| 📉 소외주 | 배당도 낮고 수익률도 낮음 |

---

## 📊 차트 읽기
- **X축** = 배당 지표 (오른쪽이 높음)
- **Y축** = 수익률 (위쪽이 높음)
- **버블 크기** = 시가총액
- **색상** = 군집 분류

---

## 📤 출력 파일
1. **market_map_YYYY-MM-DD.png** - 차트 이미지
2. **clusters_YYYY-MM-DD.csv** - 군집별 통계
3. **details_YYYY-MM-DD.csv** - 전체 종목 데이터

---

## ⚙️ 군집 수 조절
```python
# dbscan_clean_final.py 수정

TARGET_MIN = 4      # 최소 군집
TARGET_MAX = 6      # 최대 군집

# 더 많은 군집 원할 때
# TARGET_MAX = 10, EPS_MIN = 0.05
```

---

## 🆘 자주 나는 에러

| 에러 | 해결 |
|------|------|
| parquet 없음 | `data/` 폴더 확인 |
| 한글 깨짐 | NanumGothic.ttf 확인 |
| adjustText 설치 안 됨 | `pip install adjustText --no-cache-dir` |

---

## 💡 필수 Parquet 컬럼
```
- Date: 분석 날짜
- Dividend_Yield / 배당수익률 / DPS: 배당 지표
- Return_60d / Return_20d / Chg_Pct: 수익률
- Marcap / 시가총액: 시가총액
- 종목명 / Name: 종목 이름
```

---

**다 준비됐습니다! 실행하세요! 🚀**
