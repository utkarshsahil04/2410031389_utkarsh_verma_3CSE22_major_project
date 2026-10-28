"""
Fed-Heart-Disease style loader: UCI Heart Disease multi-center CSVs.

Downloads four public processed center files (Cleveland, Hungarian,
Switzerland, VA Long Beach) from the UCI mirror — no credentials required.

Binary target: Outcome = 1 if angiographic disease status (num) > 0.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data", "heart")
COMBINED_PATH = os.path.join(DATA_DIR, "heart_combined.csv")

# UCI ML Repository — processed heart-disease files (public)
_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
CENTERS = {
    "cleveland": f"{_UCI}/processed.cleveland.data",
    "hungarian": f"{_UCI}/processed.hungarian.data",
    "switzerland": f"{_UCI}/processed.switzerland.data",
    "va": f"{_UCI}/processed.va.data",
}

COLUMNS = [
    "Age",
    "Sex",
    "Cp",
    "Trestbps",
    "Chol",
    "Fbs",
    "Restecg",
    "Thalach",
    "Exang",
    "Oldpeak",
    "Slope",
    "Ca",
    "Thal",
    "Num",
]

# Fallback single-file mirror if UCI is unreachable
_FALLBACK_URL = (
    "https://raw.githubusercontent.com/selva86/datasets/master/Heart.csv"
)


def _parse_center(path: str, center: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df["center"] = center
    return df


def ensure_heart() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(COMBINED_PATH):
        return COMBINED_PATH

    frames = []
    print("[+] Downloading UCI Heart Disease (multi-center)...")
    try:
        for name, url in CENTERS.items():
            dest = os.path.join(DATA_DIR, f"{name}.data")
            if not os.path.exists(dest):
                urllib.request.urlretrieve(url, dest)
            frames.append(_parse_center(dest, name))
        df = pd.concat(frames, ignore_index=True)
    except Exception as exc:
        print(f"    UCI download failed ({exc}); trying fallback mirror...")
        dest = os.path.join(DATA_DIR, "heart_fallback.csv")
        urllib.request.urlretrieve(_FALLBACK_URL, dest)
        raw = pd.read_csv(dest)
        # Normalise column names toward our schema
        rename = {
            "age": "Age",
            "sex": "Sex",
            "cp": "Cp",
            "trestbps": "Trestbps",
            "chol": "Chol",
            "fbs": "Fbs",
            "restecg": "Restecg",
            "thalach": "Thalach",
            "exang": "Exang",
            "oldpeak": "Oldpeak",
            "slope": "Slope",
            "ca": "Ca",
            "thal": "Thal",
            "AHD": "Num",
            "target": "Num",
        }
        raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
        if "Num" not in raw.columns and "Outcome" in raw.columns:
            raw["Num"] = raw["Outcome"]
        raw["center"] = "fallback"
        df = raw

    # Binary label
    if df["Num"].dtype == object:
        # AHD Yes/No style
        df["Outcome"] = (
            df["Num"].astype(str).str.lower().isin(["yes", "1", "true"]).astype(int)
        )
    else:
        df["Outcome"] = (pd.to_numeric(df["Num"], errors="coerce").fillna(0) > 0).astype(
            int
        )

    feature_cols = [
        c
        for c in [
            "Age",
            "Sex",
            "Cp",
            "Trestbps",
            "Chol",
            "Fbs",
            "Restecg",
            "Thalach",
            "Exang",
            "Oldpeak",
            "Slope",
            "Ca",
            "Thal",
        ]
        if c in df.columns
    ]
    out = df[feature_cols + ["Outcome", "center"]].copy()
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    # Median-impute any remaining NaNs in features
    for c in feature_cols:
        out[c] = out[c].fillna(out[c].median())

    out.to_csv(COMBINED_PATH, index=False)
    print(f"    Saved to {COMBINED_PATH}  ({len(out)} rows)\n")
    return COMBINED_PATH


def load_heart():
    ensure_heart()
    df = pd.read_csv(COMBINED_PATH)
    centers = sorted(df["center"].dropna().unique().tolist()) if "center" in df.columns else []
    # Model input: drop center from features but keep for non-IID / reporting
    feature_df = df.drop(columns=["center"], errors="ignore")
    if feature_df.columns[-1] != "Outcome":
        cols = [c for c in feature_df.columns if c != "Outcome"] + ["Outcome"]
        feature_df = feature_df[cols]
    meta = {
        "name": "heart",
        "path": COMBINED_PATH,
        "n_rows": len(feature_df),
        "n_features": feature_df.shape[1] - 1,
        "description": "UCI Heart Disease multi-center (Fed-Heart-Disease style)",
        "centers": centers,
        "center_column": df["center"] if "center" in df.columns else None,
    }
    # Attach center series for splitters that want hospital=center
    feature_df = feature_df.copy()
    if "center" in df.columns:
        feature_df.attrs["center"] = df["center"].values
    return feature_df, meta
