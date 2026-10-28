"""
Dataset loaders for the Blockchain + FL prototype.

  pima   — PIMA Indians Diabetes (default smoke test, auto-download)
  heart  — UCI multi-center heart-disease tables (Fed-Heart-Disease style)
  mimic  — stub / tiny synthetic MIMIC-shaped sample (no PhysioNet download)

Usage:
    from datasets import load_dataset
    df, meta = load_dataset("pima")
"""
from __future__ import annotations

from datasets.heart import ensure_heart, load_heart
from datasets.mimic import ensure_mimic_stub, load_mimic
from datasets.pima import ensure_pima, load_pima

AVAILABLE = ("pima", "heart", "mimic")


def load_dataset(name: str = "pima"):
    """
    Return (dataframe, metadata_dict).
    DataFrame: feature columns + binary target as last column named Outcome.
    """
    key = name.strip().lower()
    if key in ("pima", "diabetes"):
        ensure_pima()
        return load_pima()
    if key in ("heart", "fed-heart", "fed_heart", "heartdisease"):
        ensure_heart()
        return load_heart()
    if key in ("mimic", "mimic-iii", "mimic_iii"):
        ensure_mimic_stub()
        return load_mimic()
    raise ValueError(
        f"Unknown dataset '{name}'. Choose from: {', '.join(AVAILABLE)}"
    )


__all__ = [
    "AVAILABLE",
    "load_dataset",
    "ensure_pima",
    "load_pima",
    "ensure_heart",
    "load_heart",
    "ensure_mimic_stub",
    "load_mimic",
]
