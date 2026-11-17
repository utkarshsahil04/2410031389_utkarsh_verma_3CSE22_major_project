"""
MIMIC adapter stub — does NOT download real MIMIC (PhysioNet credentials required).

Provides a tiny synthetic MIMIC-shaped table so the pipeline can be smoke-tested
with `--dataset mimic`. See README for how to obtain real MIMIC-III/IV access.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, "data", "mimic")
SYNTH_PATH = os.path.join(DATA_DIR, "mimic_synthetic_tiny.csv")

# Column names loosely inspired by common MIMIC mortality / ICU feature demos
COLUMNS = [
    "age",
    "gender",
    "heart_rate",
    "sys_bp",
    "dias_bp",
    "resp_rate",
    "temp_c",
    "spo2",
    "glucose",
    "wbc",
    "Outcome",  # binary in-hospital mortality proxy (synthetic)
]

README_NOTE = """
MIMIC-III / MIMIC-IV access requirements
---------------------------------------
Real MIMIC data is credentialed on PhysioNet and MUST NOT be auto-downloaded
by this prototype.

1. Complete the CITI "Data or Specimens Only Research" training.
2. Sign the PhysioNet credentialed health data use agreement.
3. Request access to MIMIC-III or MIMIC-IV on https://physionet.org/
4. Place approved extracts under code/data/mimic/ (never commit PHI).
5. Extend datasets/mimic.py::load_mimic() to read your local extract.

Until then, `--dataset mimic` uses a tiny synthetic MIMIC-shaped sample only.
"""


def ensure_mimic_stub() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    readme_path = os.path.join(DATA_DIR, "README_MIMIC_ACCESS.txt")
    if not os.path.exists(readme_path):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(README_NOTE.strip() + "\n")

    if os.path.exists(SYNTH_PATH):
        return SYNTH_PATH

    rng = np.random.default_rng(42)
    n = 600  # large enough that the 10% test split has positive cases
    age = rng.integers(18, 90, size=n)
    gender = rng.integers(0, 2, size=n)
    hr = rng.normal(85, 15, size=n).clip(40, 180)
    sys_bp = rng.normal(120, 20, size=n).clip(70, 220)
    dias_bp = rng.normal(75, 12, size=n).clip(40, 130)
    resp = rng.normal(18, 4, size=n).clip(8, 40)
    temp = rng.normal(36.8, 0.6, size=n).clip(35.0, 40.5)
    spo2 = rng.normal(96, 3, size=n).clip(70, 100)
    glucose = rng.normal(140, 50, size=n).clip(50, 400)
    wbc = rng.normal(9, 4, size=n).clip(1, 30)

    # Synthetic risk score → binary label (not clinical truth). Tuned for ~19% positives
    # (ICU-like); the old -4.0 intercept gave ~2%, so tests often had no positive patient
    # and "predict nobody dies" scored a meaningless 100%.
    logit = -3.8 + 2.0 * (
        0.04 * (age - 50)
        + 0.03 * (hr - 80)
        - 0.02 * (sys_bp - 120)
        - 0.15 * (spo2 - 96)
        + 0.08 * (resp - 18)
        + 0.004 * (glucose - 120)
        + 0.06 * (wbc - 8)
    )
    prob = 1 / (1 + np.exp(-logit))
    outcome = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "heart_rate": np.round(hr, 1),
            "sys_bp": np.round(sys_bp, 1),
            "dias_bp": np.round(dias_bp, 1),
            "resp_rate": np.round(resp, 1),
            "temp_c": np.round(temp, 2),
            "spo2": np.round(spo2, 1),
            "glucose": np.round(glucose, 1),
            "wbc": np.round(wbc, 2),
            "Outcome": outcome,
        }
    )
    df.to_csv(SYNTH_PATH, index=False)
    print(f"[+] Wrote synthetic MIMIC-shaped sample -> {SYNTH_PATH} ({len(df)} rows)")
    print("    Real MIMIC is NOT downloaded. See data/mimic/README_MIMIC_ACCESS.txt\n")
    return SYNTH_PATH


def load_mimic():
    ensure_mimic_stub()
    df = pd.read_csv(SYNTH_PATH)
    meta = {
        "name": "mimic",
        "path": SYNTH_PATH,
        "n_rows": len(df),
        "n_features": df.shape[1] - 1,
        "description": (
            "Synthetic MIMIC-shaped sample only — real MIMIC needs PhysioNet access"
        ),
        "centers": None,
        "is_synthetic": True,
        "access_readme": os.path.join(DATA_DIR, "README_MIMIC_ACCESS.txt"),
    }
    return df, meta
