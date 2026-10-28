"""PIMA Indians Diabetes — default smoke-test dataset (public, auto-download)."""
from __future__ import annotations

import os
import urllib.request

import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data")
DATA_PATH = os.path.join(DATA_DIR, "diabetes.csv")

DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def ensure_pima() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    print("[+] Downloading PIMA Diabetes dataset...")
    urllib.request.urlretrieve(DATASET_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    df.to_csv(DATA_PATH, index=False)
    print(f"    Saved to {DATA_PATH}  ({len(df)} rows)\n")
    return DATA_PATH


def load_pima():
    ensure_pima()
    df = pd.read_csv(DATA_PATH)
    # Guarantee Outcome is last column
    if "Outcome" in df.columns and df.columns[-1] != "Outcome":
        cols = [c for c in df.columns if c != "Outcome"] + ["Outcome"]
        df = df[cols]
    meta = {
        "name": "pima",
        "path": DATA_PATH,
        "n_rows": len(df),
        "n_features": df.shape[1] - 1,
        "description": "PIMA Indians Diabetes (public benchmark)",
        "centers": None,
    }
    return df, meta
