"""Load and normalize fl_state.json / chain_state.json for the Streamlit UI."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL_STATE_PATH = os.path.join(CODE_DIR, "fl_state.json")
CHAIN_STATE_PATH = os.path.join(CODE_DIR, "chain_state.json")
EXPERIMENT_RESULTS_CANDIDATES = (
    os.path.join(CODE_DIR, "experiment_results.json"),
    os.path.join(CODE_DIR, "experiments_results.json"),
    os.path.join(CODE_DIR, "results", "experiments.json"),
)


def load_json(path: str) -> Any | None:
    """Return parsed JSON or None if missing / invalid."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        return None


def load_fl_state() -> dict | None:
    data = load_json(FL_STATE_PATH)
    return data if isinstance(data, dict) else None


def load_chain_state() -> list | None:
    data = load_json(CHAIN_STATE_PATH)
    return data if isinstance(data, list) else None


def load_experiment_results() -> tuple[str | None, Any]:
    for path in EXPERIMENT_RESULTS_CANDIDATES:
        data = load_json(path)
        if data is not None:
            return path, data
    return None, None


def status_label(status: str | None) -> str:
    mapping = {
        "initializing": "Initializing",
        "training": "Training",
        "complete": "Complete",
        "error": "Error",
    }
    if not status:
        return "No run yet"
    return mapping.get(status.lower(), status.replace("_", " ").title())


def format_timestamp(ts: float | int | str | None) -> str:
    if ts is None or ts == "":
        return "—"
    try:
        value = float(ts)
        # Heuristic: simulation sometimes writes synthetic epoch values.
        if value > 1e12:
            value /= 1000.0
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except (TypeError, ValueError, OSError, OverflowError):
        return str(ts)


def short_hash(value: str | None, n: int = 16) -> str:
    if not value:
        return "—"
    text = str(value)
    if len(text) <= n + 3:
        return text
    return text[:n] + "..."


def rounds_dataframe(fl_state: dict):
    """Build a pandas DataFrame of round metrics (lazy import)."""
    import pandas as pd

    rounds = fl_state.get("rounds") or []
    if not rounds:
        return pd.DataFrame(columns=["round", "accuracy", "avg_loss"])
    rows = []
    for r in rounds:
        rows.append(
            {
                "round": r.get("round"),
                "accuracy": r.get("accuracy"),
                "avg_loss": r.get("avg_loss"),
            }
        )
    return pd.DataFrame(rows)


def hospital_rows(fl_state: dict) -> list[dict]:
    hospitals = fl_state.get("hospitals") or {}
    latest_losses: dict[str, float] = {}
    rounds = fl_state.get("rounds") or []
    if rounds:
        client_losses = rounds[-1].get("client_losses") or {}
        # Map hospital_1 → Hospital_A style when names differ
        name_list = list(hospitals.keys())
        for key, loss in client_losses.items():
            latest_losses[key] = loss
            if key.startswith("hospital_"):
                try:
                    idx = int(key.split("_")[-1]) - 1
                    if 0 <= idx < len(name_list):
                        latest_losses[name_list[idx]] = loss
                except ValueError:
                    pass

    rows = []
    for name, info in hospitals.items():
        info = info or {}
        rows.append(
            {
                "Hospital": name,
                "Records": info.get("records", "—"),
                "Registered": "Yes" if info.get("registered") else "No",
                "Record hash": info.get("record_hash", "—"),
                "Block #": info.get("block_index", "—"),
                "Latest local loss": latest_losses.get(name, "—"),
            }
        )
    return rows


def chain_table_rows(chain: list) -> list[dict]:
    rows = []
    for block in chain:
        data = block.get("data") or {}
        event = data.get("event", "—")
        actor = (
            data.get("actor")
            or data.get("hospital")
            or data.get("from")
            or data.get("to")
            or "—"
        )
        if data.get("from") and data.get("to") and event == "grant_access":
            actor = f"{data.get('from')} → {data.get('to')}"
        details = {k: v for k, v in data.items() if k not in ("event",)}
        rows.append(
            {
                "Index": block.get("index"),
                "Event": event,
                "Actor": actor,
                "Timestamp": format_timestamp(block.get("timestamp")),
                "Hash": short_hash(block.get("hash"), 20),
                "Prev hash": short_hash(block.get("previous_hash"), 12),
                "Details": json.dumps(details, default=str)[:180],
            }
        )
    return rows
