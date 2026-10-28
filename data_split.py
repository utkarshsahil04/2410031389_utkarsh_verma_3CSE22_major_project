"""
Train/test split + hospital sharding: IID (equal) or non-IID (label / feature skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_split_df(df: pd.DataFrame, test_frac: float = 0.10, seed: int = 42):
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_test = max(int(test_frac * len(shuffled)), 1)
    test_df = shuffled.iloc[:n_test].reset_index(drop=True)
    train_df = shuffled.iloc[n_test:].reset_index(drop=True)
    return train_df, test_df


def split_iid(train_df: pd.DataFrame, n_hospitals: int) -> list[pd.DataFrame]:
    """Equal-sized contiguous shards after shuffle (approx IID)."""
    size = len(train_df) // n_hospitals
    shards = [
        train_df.iloc[i * size : (i + 1) * size].reset_index(drop=True)
        for i in range(n_hospitals - 1)
    ]
    shards.append(train_df.iloc[(n_hospitals - 1) * size :].reset_index(drop=True))
    return shards


def split_non_iid(
    train_df: pd.DataFrame,
    n_hospitals: int,
    *,
    label_col: str | None = None,
    feature_col: str | None = None,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """
    Non-IID partition with label skew + optional feature skew.

    Label skew: hospital i gets a Dirichlet-biased share of each class
    (stronger positive-rate at hospital 0, weaker at the last).

    Feature skew: if feature_col exists (Age / Glucose / …), sort by that
    feature and assign quantile bands to hospitals so feature distributions differ.
    """
    rng = np.random.default_rng(seed)
    label_col = label_col or train_df.columns[-1]
    y = train_df[label_col].values.astype(int)

    # Prefer Age/Glucose-like columns for feature skew when present
    if feature_col is None:
        for cand in ("Age", "age", "Glucose", "glucose", "Chol", "chol"):
            if cand in train_df.columns:
                feature_col = cand
                break

    indices = np.arange(len(train_df))
    hospital_idxs: list[list[int]] = [[] for _ in range(n_hospitals)]

    if feature_col and feature_col in train_df.columns:
        # Feature-quantile bands (feature skew)
        order = np.argsort(train_df[feature_col].values)
        bands = np.array_split(order, n_hospitals)
        for h, band in enumerate(bands):
            hospital_idxs[h].extend(band.tolist())
    else:
        # Label skew via biased sampling per class
        for cls in np.unique(y):
            cls_idx = indices[y == cls]
            rng.shuffle(cls_idx)
            # Dirichlet-like proportions: hospital 0 gets more of class 1, etc.
            if cls == 1:
                weights = np.linspace(0.55, 0.15, n_hospitals)
            else:
                weights = np.linspace(0.15, 0.55, n_hospitals)
            weights = weights / weights.sum()
            cuts = (np.cumsum(weights) * len(cls_idx)).astype(int)
            start = 0
            for h, end in enumerate(cuts):
                if h == n_hospitals - 1:
                    end = len(cls_idx)
                hospital_idxs[h].extend(cls_idx[start:end].tolist())
                start = end

    shards = []
    for h in range(n_hospitals):
        idxs = np.array(hospital_idxs[h], dtype=int)
        rng.shuffle(idxs)
        if len(idxs) == 0:
            # Degenerate edge case — give at least one row
            idxs = np.array([h % len(train_df)])
        shards.append(train_df.iloc[idxs].reset_index(drop=True))
    return shards


def split_for_hospitals(
    df: pd.DataFrame,
    hospital_names: list[str],
    *,
    mode: str = "iid",
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[pd.DataFrame], pd.DataFrame, dict]:
    """
    mode: "iid" | "noniid"
    Returns shards, test_df, split_info
    """
    train_df, test_df = train_test_split_df(df, test_frac=test_frac, seed=seed)
    n = len(hospital_names)
    mode_l = mode.strip().lower()
    if mode_l in ("iid", "equal"):
        shards = split_iid(train_df, n)
        kind = "iid"
    elif mode_l in ("noniid", "non-iid", "skew"):
        shards = split_non_iid(train_df, n, seed=seed)
        kind = "noniid"
    else:
        raise ValueError(f"Unknown split mode '{mode}'. Use iid|noniid.")

    # Label-rate report for simulation output
    label = df.columns[-1]
    rates = []
    for name, shard in zip(hospital_names, shards):
        pos = float(shard[label].mean()) if len(shard) else 0.0
        rates.append(
            {
                "hospital": name,
                "n": len(shard),
                "positive_rate": round(pos, 4),
            }
        )

    info = {
        "mode": kind,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "hospital_stats": rates,
    }
    return shards, test_df, info
