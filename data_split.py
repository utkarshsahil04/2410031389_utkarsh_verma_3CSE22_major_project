"""
Train/test split + hospital sharding: IID (equal) or non-IID (age-band feature skew).
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
    feature_col: str | None = None,
    seed: int = 42,
) -> list[pd.DataFrame]:
    """
    Non-IID partition by FEATURE skew: sort by Age (or Glucose / Chol / first column) and give
    each hospital a contiguous quantile band. Hospitals therefore see different feature
    distributions, and as a side effect different positive rates (older patients -> more disease).
    It is not an explicit label-skew split.
    """
    rng = np.random.default_rng(seed)
    if len(train_df) < n_hospitals:
        raise ValueError("need at least one training row per hospital")

    if feature_col is None:
        feature_col = next(
            (c for c in ("Age", "age", "Glucose", "glucose", "Chol", "chol") if c in train_df.columns),
            train_df.columns[0],
        )

    order = np.argsort(train_df[feature_col].values, kind="stable")
    hospital_idxs = np.array_split(order, n_hospitals)

    shards = []
    for h in range(n_hospitals):
        idxs = hospital_idxs[h].copy()
        rng.shuffle(idxs)
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
