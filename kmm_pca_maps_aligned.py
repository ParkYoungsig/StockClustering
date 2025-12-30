# plot_pca_maps_aligned.py
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


def _load_coords(per_date_dir: str, date: str) -> pd.DataFrame:
    """
    우선순위:
      1) pca_coords_plot_{date}.csv (PC1~PC3)
      2) pca_coords_used_{date}.csv (n_used 차원)
    return: index=Ticker, columns include PC1,PC2,(PC3 optional)
    """
    f1 = os.path.join(per_date_dir, f"pca_coords_plot_{date}.csv")
    f2 = os.path.join(per_date_dir, f"pca_coords_used_{date}.csv")

    if os.path.exists(f1):
        df = pd.read_csv(f1)
    elif os.path.exists(f2):
        df = pd.read_csv(f2)
    else:
        raise FileNotFoundError(f"좌표 파일이 없습니다: {f1} / {f2}")

    df["Ticker"] = df["Ticker"].astype(str)
    df = df.set_index("Ticker")
    return df


def plot_aligned_pca_maps(
    *,
    roll_dir: str,
    dates: list[str],
    aligned_wide_csv_path: str,
    out_subdir: str = "pca_maps_aligned",
    dpi: int = 170,
    label_all: bool = True,
    label_fontsize: int = 6,
    point_size: int = 18,
):
    """
    날짜별로:
    - 저장된 PCA 좌표(PC1~PC3)를 로드
    - aligned cluster 라벨을 붙임
    - 2D(PC1,PC2), 3D(PC1,PC2,PC3) 플롯 저장
    """
    out_dir = os.path.join(roll_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    aligned = pd.read_csv(aligned_wide_csv_path, index_col=0)
    aligned.index = aligned.index.astype(str)

    saved = []

    for d in dates:
        per_date_dir = os.path.join(roll_dir, "per_date", d)
        coords = _load_coords(per_date_dir, d)

        if d not in aligned.columns:
            print(f"[WARN] aligned 라벨에 날짜 컬럼이 없음: {d}")
            continue

        labels = aligned[d].dropna()
        labels.index = labels.index.astype(str)

        df = coords.join(labels.rename("cluster"), how="inner").dropna(
            subset=["cluster"]
        )
        if df.empty:
            print(f"[WARN] 매칭되는 ticker가 거의/전혀 없음: {d} (coords vs aligned)")
            continue

        df["cluster"] = df["cluster"].astype(int)

        # 색상 매핑 (aligned cluster id 기반으로 고정)
        uniq = np.sort(df["cluster"].unique())
        cmap = get_cmap("tab20")
        color_map = {c: cmap(i % 20) for i, c in enumerate(uniq)}
        colors = df["cluster"].map(color_map)

        # ---------- 2D ----------
        if ("PC1" in df.columns) and ("PC2" in df.columns):
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111)

            ax.scatter(
                df["PC1"],
                df["PC2"],
                s=point_size,
                c=list(colors),
                alpha=0.85,
                linewidths=0,
            )

            ax.set_title(f"PCA 2D (PC1, PC2) | aligned clusters | {d}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            # 라벨링
            if label_all:
                for tkr, r in df.iterrows():
                    ax.text(r["PC1"], r["PC2"], tkr, fontsize=label_fontsize, alpha=0.9)

            # 범례
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=str(c),
                    markerfacecolor=color_map[c],
                    markersize=7,
                )
                for c in uniq
            ]
            ax.legend(handles=handles, title="cluster", loc="best", frameon=True)

            f2d = os.path.join(out_dir, f"pca2d_aligned_{d}.png")
            plt.tight_layout()
            plt.savefig(f2d, dpi=dpi)
            plt.close(fig)
            saved.append(f2d)
        else:
            print(f"[WARN] 2D에 필요한 PC1/PC2가 없음: {d}")

        # ---------- 3D ----------
        if ("PC1" in df.columns) and ("PC2" in df.columns) and ("PC3" in df.columns):
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")

            ax.scatter(
                df["PC1"],
                df["PC2"],
                df["PC3"],
                s=point_size,
                c=list(colors),
                alpha=0.85,
            )

            ax.set_title(f"PCA 3D (PC1, PC2, PC3) | aligned clusters | {d}")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")

            if label_all:
                for tkr, r in df.iterrows():
                    ax.text(
                        r["PC1"],
                        r["PC2"],
                        r["PC3"],
                        tkr,
                        fontsize=label_fontsize,
                        alpha=0.9,
                    )

            f3d = os.path.join(out_dir, f"pca3d_aligned_{d}.png")
            plt.tight_layout()
            plt.savefig(f3d, dpi=dpi)
            plt.close(fig)
            saved.append(f3d)
        else:
            print(f"[INFO] PC3가 없어서 3D는 스킵: {d}")

    return saved
