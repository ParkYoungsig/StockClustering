import os
import glob
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from adjustText import adjust_text

# 설정 파일 불러오기 (DBSCAN 전용 설정)
# import config
from config import (DATA_FOLDER,OUTPUT_FOLDER, X_FEATS,Y_FEATS)
from config import (TARGET_CLUSTERS_MIN,TARGET_CLUSTERS_MAX)
from config import (EPS_RANGE_START, EPS_RANGE_END, EPS_STEP, MIN_SAMPLES)
from config import (FONT_FAMILY, FIG_SIZE,CSV_ENCODING)


plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False

class DataProcessor:
    @staticmethod
    def _safe_float(val):
        try:
            if isinstance(val, (pd.Series, np.ndarray, list)):
                if len(val) == 0:
                    return 0.0
                return DataProcessor._safe_float(
                    val.iloc[0] if hasattr(val, "iloc") else val[0]
                )

            if isinstance(val, str):
                val = re.sub(r"[^0-9.\-]", "", val)
                if not val:
                    return 0.0

            v = float(val)
            if np.isnan(v) or np.isinf(v):
                return 0.0
            return v
        except:
            return 0.0

    @staticmethod
    def load_snapshot(target_date_str):
        target_date = pd.to_datetime(target_date_str)
        data_dir = os.path.abspath(DATA_FOLDER)
        files = glob.glob(os.path.join(data_dir, "*.parquet"))

        print(f"[INFO] 데이터 폴더: {data_dir}")
        print(f"[INFO] {len(files)}개 파일 로딩 중... ({target_date_str})")

        clean_rows = []

        for file_path in files:
            try:
                df = pd.read_parquet(file_path)
                df = df.loc[:, ~df.columns.duplicated()]

                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date")

                if target_date not in df.index:
                    continue

                row = df.loc[target_date]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

                filename = os.path.basename(file_path)
                name_match = re.match(r"(\d{6})_(.+)\.parquet", filename)
                ticker = name_match.group(1) if name_match else "Unknown"
                name = (
                    name_match.group(2)
                    if name_match
                    else filename.replace(".parquet", "")
                )

                marcap = 0.0
                for mc in ["Marcap", "상장시가총액", "시가총액"]:
                    if mc in row.index:
                        marcap = DataProcessor._safe_float(row[mc])
                        break

                x_val = 0.0
                for f in X_FEATS:
                    if f in row.index:
                        x_val = DataProcessor._safe_float(row[f])
                        break

                y_val = 0.0
                for f in Y_FEATS:
                    if f in row.index:
                        y_val = DataProcessor._safe_float(row[f])
                        break

                clean_rows.append(
                    {
                        "Ticker": ticker,
                        "Name": name,
                        "Marcap": marcap,
                        "X_Raw": x_val,
                        "Y_Raw": y_val,
                    }
                )
            except:
                continue

        return pd.DataFrame(clean_rows)


class AutoDBSCAN:
    def run(self, df):
        # 1. NaN 제거
        df["X_Raw"] = df["X_Raw"].fillna(0.0)
        df["Y_Raw"] = df["Y_Raw"].fillna(0.0)

        # 2. 이상치 처리 (Winsorizing: 상위 1% 값을 99% 값으로 대체)
        # 이렇게 하면 극단값이 사라져서 그래프가 예쁘게 펴짐
        p99_x = df["X_Raw"].quantile(0.99)
        p99_y = df["Y_Raw"].quantile(0.99)
        p01_y = df["Y_Raw"].quantile(0.01)

        df["X_Clipped"] = df["X_Raw"].clip(upper=p99_x)
        df["Y_Clipped"] = df["Y_Raw"].clip(lower=p01_y, upper=p99_y)

        # 3. 스케일링 (RobustScaler로 중앙값 중심 정렬)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[["X_Clipped", "Y_Clipped"]])

        # 4. DBSCAN 자동 튜닝
        best_labels = None
        best_eps = 0.5
        found = False

        print(
            f"[TUNING] 목표 군집: {TARGET_CLUSTERS_MIN}~{TARGET_CLUSTERS_MAX}개 찾는 중..."
        )

        for eps in np.arange(
            EPS_RANGE_START, EPS_RANGE_END, EPS_STEP
        ):
            db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X_scaled)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if TARGET_CLUSTERS_MIN <= n_clusters <= TARGET_CLUSTERS_MAX:
                best_labels = labels
                best_eps = eps
                found = True
                print(f" -> 성공: eps={eps:.2f}, 군집={n_clusters}개")
                break

            if 2 <= n_clusters <= 10:
                best_labels = labels
                best_eps = eps

        if not found and best_labels is None:
            print(" -> 기본값 실행")
            db = DBSCAN(eps=0.5, min_samples=MIN_SAMPLES).fit(X_scaled)
            best_labels = db.labels_
            best_eps = 0.5
        elif not found:
            print(f" -> 차선책 실행 (eps={best_eps:.2f})")

        df["Cluster"] = best_labels

        # 5. 라벨링
        cluster_info = []
        unique_labels = sorted(list(set(best_labels) - {-1}))

        med_x = df["X_Clipped"].median()
        med_y = df["Y_Clipped"].median()

        for c in unique_labels:
            group = df[df["Cluster"] == c]
            g_med_x = group["X_Clipped"].median()
            g_med_y = group["Y_Clipped"].median()

            if g_med_x > med_x and g_med_y > med_y:
                label = "💎 배당성장주 (Growth+Yield)"
            elif g_med_x > med_x and g_med_y <= med_y:
                label = "🛡️ 고배당 방어주 (High Yield)"
            elif g_med_x <= med_x and g_med_y > med_y:
                label = "🚀 고성장 기대주 (High Growth)"
            else:
                label = "📉 소외주/가치주 (Value/Lagging)"

            cluster_info.append({"Cluster": c, "Label": label, "Count": len(group)})

        return df, pd.DataFrame(cluster_info), best_eps


class Visualizer:
    def plot(self, df, cluster_info, date_str, eps):
        plt.figure(figsize=FIG_SIZE)

        # 1. 노이즈 (흐릿하게) - 원본 값 사용하되 너무 큰 건 잘림
        noise = df[df["Cluster"] == -1]
        plt.scatter(
            noise["X_Raw"],
            noise["Y_Raw"],
            c="lightgray",
            marker=".",
            s=10,
            alpha=0.1,
            label="_nolegend_",
        )

        # 2. 군집 그리기
        label_map = dict(zip(cluster_info["Cluster"], cluster_info["Label"]))
        unique_labels = sorted(list(set(df["Cluster"]) - {-1}))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            group = df[df["Cluster"] == k]
            label_text = label_map.get(k, f"G{k}")

            plt.scatter(
                group["X_Raw"],
                group["Y_Raw"],
                s=np.log1p(group["Marcap"]) * 3 + 50,
                color=col,
                label=f"{label_text}",
                alpha=0.8,
                edgecolors="w",
            )

            cx, cy = group["X_Raw"].median(), group["Y_Raw"].median()
            plt.text(
                cx,
                cy,
                f"G{k}",
                fontsize=12,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2),
            )

        # 3. 텍스트
        texts = []
        clustered_df = df[df["Cluster"] != -1]
        top_stocks = clustered_df.sort_values("Marcap", ascending=False).head(30)

        for _, row in top_stocks.iterrows():
            texts.append(
                plt.text(
                    row["X_Raw"],
                    row["Y_Raw"],
                    row["Name"],
                    fontsize=9,
                    fontweight="bold",
                )
            )

        try:
            adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
        except:
            pass

        # 4. [핵심] 그래프 범위 제한 (Zoom In 효과)
        # 상위 1% 값까지만 보여주도록 축 범위 설정
        x_limit = df["X_Raw"].quantile(0.99) * 1.1  # 여유 10%
        y_min = df["Y_Raw"].quantile(0.01) * 1.1
        y_max = df["Y_Raw"].quantile(0.99) * 1.1

        # 만약 배당이 너무 작으면 최소 10%는 보여줌
        if x_limit < 10:
            x_limit = 10

        plt.xlim(-0.5, x_limit)
        plt.ylim(y_min, y_max)

        n_clusters = len(unique_labels)
        plt.title(
            f"Market Map: Dividend vs Momentum ({n_clusters} Groups) - {date_str}",
            fontsize=16,
        )
        plt.xlabel("Dividend Yield (%)", fontsize=12)
        plt.ylabel("Momentum (Return %)", fontsize=12)
        plt.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.3)

        out_dir = os.path.abspath(OUTPUT_FOLDER)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        save_path = os.path.join(out_dir, f"dbscan_market_map_{date_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        cluster_info.to_csv(
            os.path.join(out_dir, f"cluster_summary_{date_str}.csv"),
            index=False,
            encoding=CSV_ENCODING,
        )
        df.to_csv(
            os.path.join(out_dir, f"cluster_details_{date_str}.csv"),
            index=False,
            encoding=CSV_ENCODING,
        )

        print(f"[저장 완료] {save_path}")
        print("\n[군집 요약]")
        print(cluster_info[["Label", "Count"]].to_string(index=False))
        plt.show()

def report_AutoDBScan() :
    viz = Visualizer()

    print("=" * 60)
    print(" [StockClustering] 주식 시장 지도 (배당 vs 모멘텀)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n>> 날짜 (YYYY-MM-DD) [q:종료]: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit"]:
                break
            if "python" in user_input:
                continue

            df = proc.load_snapshot(user_input)
            if df.empty:
                print("데이터 없음")
                continue

            df_clustered, cluster_info, final_eps = auto_dbscan.run(df)
            viz.plot(df_clustered, cluster_info, user_input, final_eps)

        except Exception as e:
            print(f"[오류] {e}")
            traceback.print_exc()

if __name__ == "__main__":
    proc = DataProcessor()
    auto_dbscan = AutoDBSCAN()
    report_AutoDBScan()