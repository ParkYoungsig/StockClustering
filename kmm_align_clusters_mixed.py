# align_clusters_mixed.py
# ------------------------------------------------------------
# Align (relabel) daily KMeans cluster IDs across dates.
# Uses a mixed cost:
#   cost = w_overlap*(1 - JaccardOverlap) + w_centroid*(NormalizedCentroidDistance)
#
# Requires:
# - cluster_labels_wide.csv (Ticker x Date -> raw cluster id)
# - per-date centroids CSVs (optional but recommended), e.g.
#     per_date/<DATE>/centroids_<DATE>.csv
#   If centroid files are missing, falls back to overlap-only.
# ------------------------------------------------------------

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _hungarian_assign(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Min-cost assignment. Uses SciPy if available, else greedy fallback."""
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        # Greedy fallback (not optimal, but avoids extra deps)
        pairs = []
        used_cols = set()
        for i in range(cost.shape[0]):
            j = int(
                np.argmin(
                    [
                        cost[i, jj] if jj not in used_cols else 1e9
                        for jj in range(cost.shape[1])
                    ]
                )
            )
            if j not in used_cols:
                used_cols.add(j)
                pairs.append((i, j))
        return pairs


def _read_centroids_for_date(
    per_date_root: str,
    date: str,
    centroids_filename: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Tries to load centroids for a date.
    Default expected path:
      <per_date_root>/<date>/centroids_<date>.csv
    """
    if centroids_filename is None:
        centroids_filename = f"centroids_{date}.csv"

    cand = os.path.join(per_date_root, date, centroids_filename)
    if os.path.exists(cand):
        df = pd.read_csv(cand, encoding="utf-8-sig")
        return df
    return None


def align_cluster_labels_wide_csv_mixed(
    input_csv_path: str,
    output_dir: str,
    per_date_root: Optional[str] = None,
    ticker_col: str = "Ticker",
    date_cols: Optional[List[str]] = None,
    w_overlap: float = 0.6,
    w_centroid: float = 0.4,
    min_jaccard: float = 0.05,
    prefix: str = "",
) -> Dict[str, str]:
    """
    Aligns cluster labels across dates using overlap + centroid distance.

    Parameters
    ----------
    input_csv_path : str
        cluster_labels_wide.csv path.
    output_dir : str
        output directory for aligned CSVs.
    per_date_root : Optional[str]
        Root directory holding per-date folders.
        If None, tries: <output_dir>/per_date
    ticker_col : str
        Ticker column name.
    date_cols : Optional[List[str]]
        Process these date columns in order. If None, all columns except ticker_col.
    w_overlap, w_centroid : float
        Weights in cost mixing (must sum to 1.0 ideally).
    min_jaccard : float
        Minimum overlap required to treat a match as valid.
        If below, the cluster becomes a new global id (unless centroid info suggests otherwise is not used in this simple guard).
    prefix : str
        Output filename prefix.

    Returns
    -------
    dict of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    if per_date_root is None:
        per_date_root = os.path.join(output_dir, "per_date")

    wide = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    wide[ticker_col] = (
        wide[ticker_col]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)  # 혹시 5930.0 같은 케이스 방지
        .str.zfill(6)
    )

    if ticker_col not in wide.columns:
        ticker_col = wide.columns[0]

    if date_cols is None:
        date_cols = [c for c in wide.columns if c != ticker_col]
    date_cols = list(date_cols)

    # numeric labels
    for c in date_cols:
        wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("Int64")

    aligned = pd.DataFrame({ticker_col: wide[ticker_col].astype(str).values})

    # ---- Day 1: raw -> global ids
    first = date_cols[0]
    raw_first = wide[first]
    uniq = [int(x) for x in raw_first.dropna().unique()]
    uniq.sort()
    raw_to_global = {rl: i for i, rl in enumerate(uniq)}
    next_global = len(raw_to_global)

    aligned[first] = raw_first.map(
        lambda x: raw_to_global.get(int(x), pd.NA) if pd.notna(x) else pd.NA
    ).astype("Int64")

    # prev membership & prev centroid vectors (global_id -> vector)
    prev_members: Dict[int, set] = {}
    for g in aligned[first].dropna().unique():
        g = int(g)
        prev_members[g] = set(aligned.loc[aligned[first] == g, ticker_col].tolist())

    # load centroids for day1 (optional)
    prev_centroids_df = _read_centroids_for_date(per_date_root, first)
    prev_centroids: Dict[int, np.ndarray] = {}
    if prev_centroids_df is not None and len(prev_centroids_df):
        # Expect a column named 'cluster' or 'raw_cluster'. If missing, assume first column is cluster id.
        if "cluster" in prev_centroids_df.columns:
            cid_col = "cluster"
        elif "raw_cluster" in prev_centroids_df.columns:
            cid_col = "raw_cluster"
        else:
            cid_col = prev_centroids_df.columns[0]

        feat_cols = [c for c in prev_centroids_df.columns if c != cid_col]
        for _, row in prev_centroids_df.iterrows():
            rl = int(row[cid_col])
            g = raw_to_global.get(rl)
            if g is None:
                continue
            prev_centroids[int(g)] = row[feat_cols].to_numpy(dtype=float)

    prev_date = first
    align_logs = []
    transitions = []
    churn_rows = []

    # ---- Next days
    for d in date_cols[1:]:
        raw_today = wide[d]
        raw_labels = [int(x) for x in raw_today.dropna().unique()]
        raw_labels.sort()

        # membership sets for overlap
        today_members_raw = {
            rl: set(wide.loc[wide[d] == rl, ticker_col].tolist()) for rl in raw_labels
        }
        prev_globals = sorted(prev_members.keys())

        # centroid vectors for today raw clusters (optional)
        today_centroids_df = _read_centroids_for_date(per_date_root, d)
        today_centroids_raw: Dict[int, np.ndarray] = {}
        if today_centroids_df is not None and len(today_centroids_df):
            if "cluster" in today_centroids_df.columns:
                cid_col = "cluster"
            elif "raw_cluster" in today_centroids_df.columns:
                cid_col = "raw_cluster"
            else:
                cid_col = today_centroids_df.columns[0]
            feat_cols = [c for c in today_centroids_df.columns if c != cid_col]
            for _, row in today_centroids_df.iterrows():
                rl = int(row[cid_col])
                today_centroids_raw[rl] = row[feat_cols].to_numpy(dtype=float)

        # build cost matrices
        n_prev, n_today = len(prev_globals), len(raw_labels)
        if n_prev == 0 or n_today == 0:
            # degenerate, just create new ids
            raw_to_global_today = {
                rl: (next_global + i) for i, rl in enumerate(raw_labels)
            }
            next_global += len(raw_labels)
        else:
            # overlap cost
            overlap_cost = np.ones((n_prev, n_today), dtype=float)
            overlap_sim = np.zeros((n_prev, n_today), dtype=float)
            for i, g in enumerate(prev_globals):
                for j, rl in enumerate(raw_labels):
                    sim = _jaccard(prev_members[g], today_members_raw[rl])
                    overlap_sim[i, j] = sim
                    overlap_cost[i, j] = 1.0 - sim

            # centroid distance cost (normalize to 0..1)
            have_centroids = (len(prev_centroids) == n_prev) and (
                len(today_centroids_raw) == n_today
            )
            if have_centroids:
                # build dist matrix on common dims
                dist = np.zeros((n_prev, n_today), dtype=float)
                for i, g in enumerate(prev_globals):
                    vg = prev_centroids.get(g)
                    for j, rl in enumerate(raw_labels):
                        vt = today_centroids_raw.get(rl)
                        if vg is None or vt is None:
                            dist[i, j] = np.nan
                            continue
                        m = min(len(vg), len(vt))
                        if m == 0:
                            dist[i, j] = np.nan
                            continue
                        dist[i, j] = float(np.linalg.norm(vg[:m] - vt[:m]))
                # normalize ignoring nan
                valid = dist[np.isfinite(dist)]
                if valid.size:
                    dmin, dmax = float(valid.min()), float(valid.max())
                    denom = (dmax - dmin) if (dmax > dmin) else 1.0
                    dist_norm = (dist - dmin) / denom
                    dist_norm = np.where(np.isfinite(dist_norm), dist_norm, 1.0)
                else:
                    dist_norm = np.ones_like(dist)
            else:
                dist_norm = np.ones(
                    (n_prev, n_today), dtype=float
                )  # neutral / fallback

            # mixed cost
            w1 = float(w_overlap)
            w2 = float(w_centroid)
            if w1 + w2 == 0:
                w1, w2 = 1.0, 0.0
            else:
                s = w1 + w2
                w1, w2 = w1 / s, w2 / s

            cost = w1 * overlap_cost + w2 * dist_norm

            pairs = _hungarian_assign(cost)

            raw_to_global_today: Dict[int, int] = {}
            used_raw = set()

            for i, j in pairs:
                if i >= n_prev or j >= n_today:
                    continue
                g = prev_globals[i]
                rl = raw_labels[j]

                sim = float(overlap_sim[i, j])
                raw_to_global_today[rl] = g
                used_raw.add(rl)

                align_logs.append(
                    {
                        "date": d,
                        "prev_date": prev_date,
                        "prev_global": int(g),
                        "today_raw": int(rl),
                        "jaccard": sim,
                        "centroid_cost": (
                            float(dist_norm[i, j])
                            if np.isfinite(dist_norm[i, j])
                            else None
                        ),
                        "mixed_cost": float(cost[i, j]),
                        "status": "matched" if sim >= min_jaccard else "weak_match",
                    }
                )

            # Any raw not assigned -> new global
            for rl in raw_labels:
                if rl not in used_raw:
                    raw_to_global_today[rl] = next_global
                    align_logs.append(
                        {
                            "date": d,
                            "prev_date": prev_date,
                            "prev_global": None,
                            "today_raw": int(rl),
                            "jaccard": None,
                            "centroid_cost": None,
                            "mixed_cost": None,
                            "status": "new_cluster",
                        }
                    )
                    next_global += 1

        # apply mapping
        aligned[d] = raw_today.map(
            lambda x: raw_to_global_today.get(int(x), pd.NA) if pd.notna(x) else pd.NA
        ).astype("Int64")

        # transitions
        prev_lab = aligned[prev_date]
        curr_lab = aligned[d]
        mask = prev_lab.notna() & curr_lab.notna()

        if mask.any():
            tmp = pd.DataFrame(
                {
                    "date": d,
                    "prev_date": prev_date,
                    "Ticker": aligned.loc[mask, ticker_col].values,
                    "from": prev_lab[mask].astype(int).values,
                    "to": curr_lab[mask].astype(int).values,
                }
            )
            transitions.append(tmp)
            churn_rows.append(
                {
                    "date": d,
                    "prev_date": prev_date,
                    "churn": float((tmp["from"] != tmp["to"]).mean()),
                    "n": int(len(tmp)),
                }
            )
        else:
            churn_rows.append(
                {"date": d, "prev_date": prev_date, "churn": np.nan, "n": 0}
            )

        # update prev_members
        prev_members = {}
        for g in aligned[d].dropna().unique():
            g = int(g)
            prev_members[g] = set(aligned.loc[aligned[d] == g, ticker_col].tolist())

        # update prev_centroids: map today's raw centroid -> global id (if we have centroid file)
        prev_centroids = {}
        if today_centroids_df is not None and len(today_centroids_raw):
            for rl, g in raw_to_global_today.items():
                v = today_centroids_raw.get(rl)
                if v is None:
                    continue
                prev_centroids[int(g)] = v

        prev_date = d

    # save outputs
    pre = prefix or ""
    aligned_path = os.path.join(output_dir, f"{pre}cluster_labels_wide_aligned.csv")
    log_path = os.path.join(output_dir, f"{pre}cluster_alignment_log.csv")
    trans_path = os.path.join(output_dir, f"{pre}cluster_transitions_long.csv")
    churn_path = os.path.join(output_dir, f"{pre}cluster_churn_by_date.csv")

    pd.DataFrame(align_logs).to_csv(log_path, index=False, encoding="utf-8-sig")
    (
        pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()
    ).to_csv(trans_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(churn_rows).to_csv(churn_path, index=False, encoding="utf-8-sig")
    aligned.to_csv(aligned_path, index=False, encoding="utf-8-sig")

    return {
        "aligned_wide_csv": aligned_path,
        "alignment_log_csv": log_path,
        "transitions_long_csv": trans_path,
        "churn_by_date_csv": churn_path,
    }
