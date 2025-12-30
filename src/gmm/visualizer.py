"""GMM 기반 클러스터링(시장 레짐) 분석 결과 시각화 유틸리티.

본 모듈은 GMM 파이프라인에서 생성되는 평가 지표(BIC, 안정성 등)와
클러스터별 특성(평균/분포), 연도 간 레짐 전이 등을 이미지/HTML로 저장합니다.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import platform

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# OS별 기본 폰트 설정(한글 ToFu 방지) + 마이너스 기호 깨짐 방지
_os_name = platform.system()
if _os_name == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif _os_name == "Darwin":
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="NanumGothic")
plt.rc("axes", unicode_minus=False)


# --- 내부 도우미 함수(중복 제거용) ---
def _save_and_close(fig: plt.Figure, output_path: Path) -> None:
    """그림 저장 후 리소스(메모리) 해제까지 공통 처리."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _cluster_label(cid: int, cluster_names: Optional[Dict[int, str]]) -> str:
    try:
        key = int(cid)
    except Exception:
        key = cid
    return (
        cluster_names.get(key, f"Cluster {cid}") if cluster_names else f"Cluster {cid}"
    )


def _cluster_color(cid: int, cluster_colors: Optional[Dict[int, str]]) -> str:
    try:
        key = int(cid)
    except Exception:
        key = cid
    if cluster_colors and key in cluster_colors:
        return cluster_colors[key]
    # 지정 색상이 없으면 tab10 팔레트로 fallback
    cmap = plt.cm.get_cmap("tab10")
    try:
        return to_hex(cmap(key % 10))
    except Exception:
        return "#888888"


# --- 시각화 함수 ---
def plot_bic_curve(
    k_values: List[int], bic_scores: List[float], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, bic_scores, marker="o")
    ax.set(
        xlabel="n_components (K)", ylabel="BIC", title="BIC Scores (Lower is Better)"
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    _save_and_close(fig, output_path)


def plot_cluster_heatmap(
    cluster_means: pd.DataFrame,
    output_path: Path,
    cluster_names: Optional[Dict[int, str]] = None,
) -> None:
    plot_df = cluster_means.copy()
    if not plot_df.empty:
        plot_df.index = [_cluster_label(cid, cluster_names) for cid in plot_df.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Cluster-wise Feature Means")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.subplots_adjust(left=0.35)
    _save_and_close(fig, output_path)


def plot_stability_curve(
    k_values: List[int], stability_scores: List[float], output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, stability_scores, marker="o", color="tab:green")
    ax.set(
        xlabel="K", ylabel="Stability Score", title="Stability vs K (Higher is Better)"
    )
    ax.grid(True, linestyle="--", alpha=0.6)
    _save_and_close(fig, output_path)


def plot_umap_scatter(
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    cluster_names: Optional[Dict[int, str]] = None,
    cluster_colors: Optional[Dict[int, str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    point_colors = [_cluster_color(cid, cluster_colors) for cid in labels]
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=point_colors, s=60, alpha=0.85
    )
    ax.set(xlabel="Dim 1", ylabel="Dim 2", title="Market Regime Manifold (UMAP)")
    ax.grid(True, linestyle="--", alpha=0.6)

    unique_c = np.unique(labels)
    handles = []
    for cid in unique_c:
        color = _cluster_color(cid, cluster_colors)
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                color=color,
                linestyle="",
                label=_cluster_label(cid, cluster_names),
            )
        )
    ax.legend(handles=handles, title="Regime")
    _save_and_close(fig, output_path)


def plot_radar_chart(
    cluster_means: pd.DataFrame,
    output_path: Path,
    cluster_names: Optional[Dict[int, str]] = None,
    cluster_colors: Optional[Dict[int, str]] = None,
) -> None:
    if cluster_means.empty:
        return
    features = list(cluster_means.columns)

    # 0~1 정규화(Min-Max)
    norm = cluster_means.copy()
    for col in features:
        mn, mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col] - mn) / (mx - mn) if mx > mn else 0.5

    # 레이더 차트 각도 계산
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # 닫힌 도형 만들기

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for cluster, row in norm.iterrows():
        values = row.values.tolist() + [row.values[0]]
        color = _cluster_color(cluster, cluster_colors)
        ax.plot(
            angles, values, label=_cluster_label(cluster, cluster_names), color=color
        )
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, rotation=20)
    ax.set_title("Cluster Feature Profile (Normalized)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.25, 1.0))
    _save_and_close(fig, output_path)


def plot_parallel_coords(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_path: Path,
    cluster_names: Optional[Dict[int, str]] = None,
    cluster_colors: Optional[Dict[int, str]] = None,
) -> None:
    if df.empty:
        return
    # Z-score 정규화
    data_norm = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()
    data_norm["cluster"] = df["cluster"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for cluster, grp in data_norm.groupby("cluster"):
        color = _cluster_color(cluster, cluster_colors)
        # 전체 흐름 (연하게)
        ax.plot(feature_cols, grp[feature_cols].T, color=color, alpha=0.15)
        # 평균 흐름 (진하게)
        ax.plot(
            feature_cols,
            grp[feature_cols].mean(),
            color=color,
            lw=3,
            label=_cluster_label(cluster, cluster_names),
        )

    ax.set(title="Parallel Coordinates", ylabel="Z-score")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(True, ls="--", alpha=0.6)
    plt.xticks(rotation=20)
    _save_and_close(fig, output_path)


def plot_sankey(
    df: pd.DataFrame,
    labels_per_year: Dict[int, pd.Series],
    output_path: Path,
    cluster_names: Optional[Dict[int, str]] = None,
    cluster_colors: Optional[Dict[int, str]] = None,
) -> None:
    if go is None:
        return  # Plotly 미설치 시 skip

    years = sorted(labels_per_year.keys())
    if len(years) < 2:
        return

    nodes, node_map, node_colors = [], {}, []
    for y in years:
        for c in sorted(labels_per_year[y].unique()):
            key = (y, int(c))
            lbl = f"{y}-C{int(c)}"
            node_map[key] = len(nodes)
            nodes.append(lbl)
            node_colors.append(_cluster_color(c, cluster_colors))

    src, tgt, val = [], [], []
    for i in range(len(years) - 1):
        y0, y1 = years[i], years[i + 1]
        # 두 연도에 모두 존재하는 Ticker(교집합)만 대상으로 전이 계산
        merged = pd.merge(
            df[df["Year"] == y0][["Ticker"]].assign(c0=labels_per_year[y0]),
            df[df["Year"] == y1][["Ticker"]].assign(c1=labels_per_year[y1]),
            on="Ticker",
        )
        if merged.empty:
            continue

        flows = merged.groupby(["c0", "c1"]).size().reset_index(name="count")
        for _, r in flows.iterrows():
            c0, c1 = int(r["c0"]), int(r["c1"])
            src.append(node_map[(y0, c0)])
            tgt.append(node_map[(y1, c1)])
            val.append(r["count"])

    if not val:
        return
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    label=nodes,
                    pad=15,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    color=node_colors,
                ),
                link=dict(
                    source=src,
                    target=tgt,
                    value=val,
                    color=[node_colors[s] for s in src],
                ),
            )
        ]
    )
    fig.update_layout(title="Cluster Transitions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg"}:
        fig.write_image(str(output_path), scale=2)
    else:
        fig.write_html(str(output_path))
def plot_risk_return_scatter(
    cluster_means: pd.DataFrame,
    output_path: Path,
    cluster_names: Optional[Dict[int, str]] = None,
    cluster_colors: Optional[Dict[int, str]] = None,
) -> None:
    """클러스터 평균 수익률-변동성 산점도(리스크-리턴 맵)"""
    req = {"vol_60_sqrt252", "Return_120d"}
    if cluster_means.empty or not req.issubset(set(cluster_means.columns)):
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    x = cluster_means["vol_60_sqrt252"]
    y = cluster_means["Return_120d"]
    clusters = cluster_means.index

    point_colors = [_cluster_color(cid, cluster_colors) for cid in clusters]
    scatter = ax.scatter(x, y, s=220, c=point_colors, edgecolor="black", alpha=0.85)
    for i, cid in enumerate(clusters):
        short_label = (
            f"C{int(cid)}"
            if isinstance(cid, (int, np.integer)) or str(cid).isdigit()
            else f"C{cid}"
        )
        ax.annotate(
            short_label,
            (x.iloc[i], y.iloc[i]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=11,
            weight="bold",
        )

    legend_lines = []
    for cid in clusters:
        short_label = (
            f"C{int(cid)}"
            if isinstance(cid, (int, np.integer)) or str(cid).isdigit()
            else f"C{cid}"
        )
        legend_lines.append(f"{short_label}: {_cluster_label(cid, cluster_names)}")
    legend_text = "\n".join(legend_lines)
    ax.text(
        1.05,
        0.95,
        legend_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, linewidth=0.5),
    )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(x.mean(), color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("Risk (vol_60_sqrt252)")
    ax.set_ylabel("Return (120d)")
    ax.set_title("Risk-Return Profile by Regime")
    ax.grid(True, alpha=0.3)
    _save_and_close(fig, output_path)


def plot_cluster_boxplots(
    df: pd.DataFrame,
    feature_cols: List[str],
    output_path: Path,
    cluster_colors: Optional[Dict[int, str]] = None,
) -> None:
    """클러스터 지문(Cluster Fingerprint)을 Box Plot으로 시각화합니다.

    - 각 클러스터의 특징(feature) 분포를 비교하기 위한 시각화입니다.
    - seaborn FutureWarning을 피하기 위해 hue/legend 옵션을 명시합니다.
    """
    if df.empty:
        return
    # 1) 데이터 녹이기(Wide -> Long Format)
    melted = df.melt(
        id_vars=["cluster"],
        value_vars=feature_cols,
        var_name="feature",
        value_name="value",
    )
    palette_map = {
        cid: _cluster_color(cid, cluster_colors)
        for cid in melted["cluster"].dropna().unique()
    }
    # 2) 서브플롯 준비(여러 개의 작은 그래프)
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    # 3) 그래프 그리기
    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        # seaborn 경고 회피: hue를 명시하고 legend를 끕니다.
        sns.boxplot(
            data=melted[melted["feature"] == feat],
            x="cluster",
            y="value",
            hue="cluster",  # <--- 추가
            legend=False,  # <--- 추가
            ax=ax,
            palette=palette_map or "Set2",
        )
        ax.set_title(feat)
        ax.grid(True, linestyle="--", alpha=0.3)
    # 남는 서브플롯은 비활성화
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    _save_and_close(fig, output_path)
