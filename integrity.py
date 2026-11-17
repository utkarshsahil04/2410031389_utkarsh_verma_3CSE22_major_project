"""
Live dataset integrity: compare each hospital's local CSV with the hash
anchored on the blockchain, and anchor edits as signed data_update blocks.

ponytail: plain files + the existing chain_state.json ledger; no watcher thread,
          callers just poll scan(). Upgrade path: inotify/watchdog + real Fabric.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime

import pandas as pd

from chain.crypto import NodeKeyPair
from chain.inprocess import InProcessChain

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "hospital_data")
CHAIN_PATH = os.path.join(BASE, "chain_state.json")
FL_STATE_PATH = os.path.join(BASE, "fl_state.json")
HOSPITALS = ["Hospital_A", "Hospital_B", "Hospital_C"]
SEED = "demo-seed"


def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_text(s: str) -> str:
    return hash_bytes(s.encode())


def _path(h: str) -> str:
    return os.path.join(DATA_DIR, f"{h}.csv")


def _snap(h: str) -> str:
    return os.path.join(DATA_DIR, ".registered", f"{h}.csv")


def _write_json(path: str, obj) -> None:
    # write-then-replace so a polling reader never sees half a file
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_csv(path: str, df: pd.DataFrame) -> str:
    data = df.to_csv(index=False).encode()  # exact bytes HospitalClient.get_data_hash() hashes
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:  # binary: no \r\n translation on Windows
        f.write(data)
    return hash_bytes(data)


def write_hospital_files(shards: dict[str, pd.DataFrame], dataset: str, split: str) -> dict[str, str]:
    hashes = {}
    for h, df in shards.items():
        hashes[h] = _write_csv(_path(h), df)
        _write_csv(_snap(h), df)
    _write_json(os.path.join(DATA_DIR, "meta.json"), {"dataset": dataset, "split": split})
    return hashes


def read_meta() -> dict | None:
    return _read_json(os.path.join(DATA_DIR, "meta.json"))


def load_shard(h: str) -> pd.DataFrame:
    return pd.read_csv(_path(h))


def save_shard(h: str, df: pd.DataFrame) -> str:
    return _write_csv(_path(h), df)


def _load_chain() -> tuple[list | None, bool]:
    """(blocks, verified). Never trust chain_state.json without re-verifying it."""
    blocks = _read_json(CHAIN_PATH)
    if not isinstance(blocks, list) or not blocks:
        return None, False
    try:
        return blocks, InProcessChain.from_blocks(blocks, seed=SEED).verify_chain()
    except Exception:  # malformed / old-format ledger
        return blocks, False


def chain_record(h: str, blocks: list | None = None) -> dict | None:
    if blocks is None:
        blocks = _read_json(CHAIN_PATH)
    if not isinstance(blocks, list):
        return None
    for b in reversed(blocks):
        d = b.get("data", {})
        # only the hospital itself may anchor its data (signature checked by verify_chain)
        if (d.get("event") in ("register_record", "data_update") and d.get("hospital") == h
                and b.get("signer_id") == h and d.get("record_hash")):
            return {
                "hash": d["record_hash"],
                "block_index": b["index"],
                "event": d["event"],
                "timestamp": b["timestamp"],
            }
    return None


def _read_text_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def scan() -> list[dict]:
    blocks, chain_ok = _load_chain()
    out = []
    for h in HOSPITALS:
        p = _path(h)
        rec = chain_record(h, blocks)
        row = {
            "hospital": h, "path": p, "status": "missing", "live_hash": None, "chain_valid": chain_ok,
            "chain_hash": rec["hash"] if rec else None,
            "chain_block_index": rec["block_index"] if rec else None,
            "rows": None, "columns": None, "modified_at": None, "values_identical": None,
        }
        out.append(row)
        try:
            with open(p, "rb") as f:
                raw = f.read()
            row["modified_at"] = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            continue
        row["live_hash"] = hash_bytes(raw)
        try:
            df = _read_text_frame(p)
        except Exception:  # any parse failure (bad encoding, empty, ragged)
            row["status"] = "unreadable"
            continue
        row["rows"], row["columns"] = len(df), len(df.columns)
        if rec is None:
            row["status"] = "not_registered"
        elif not chain_ok:
            row["status"] = "chain_invalid"  # the on-chain hash itself can't be trusted
        elif row["live_hash"] == rec["hash"]:
            row["status"] = "match"
        else:
            row["status"] = "changed"
            row["values_identical"] = changed_cells(h)["values_identical"]
    return out


def changed_cells(h: str, limit: int = 50) -> dict:
    res = {"modified": [], "rows_added": 0, "rows_removed": 0, "values_identical": False,
           "columns_added": [], "columns_removed": [], "error": None}
    try:
        old, new = _read_text_frame(_snap(h)), _read_text_frame(_path(h))
    except Exception as exc:
        res["error"] = f"could not compare with the registered version ({type(exc).__name__})"
        return res
    res["values_identical"] = old.equals(new)
    res["columns_added"] = [c for c in new.columns if c not in old.columns]
    res["columns_removed"] = [c for c in old.columns if c not in new.columns]
    res["rows_added"] = max(0, len(new) - len(old))
    res["rows_removed"] = max(0, len(old) - len(new))
    cols = [c for c in old.columns if c in new.columns]
    n = min(len(old), len(new))
    for c in cols:
        diff = (old[c].iloc[:n].values != new[c].iloc[:n].values).nonzero()[0]
        for i in diff:
            res["modified"].append({"row": int(i), "column": c, "old": old[c].iloc[i], "new": new[c].iloc[i]})
    res["modified"].sort(key=lambda m: (m["row"], m["column"]))
    res["modified"] = res["modified"][:limit]
    return res


def _simulation_running(fl: dict) -> bool:
    # ponytail: the simulation rewrites fl_state.json every round; a "training" status that
    # has not been touched for 60 s is a crashed/stopped run, not a live one.
    if fl.get("status") not in ("training", "initializing"):
        return False
    try:
        return datetime.now().timestamp() - os.path.getmtime(FL_STATE_PATH) < 60
    except OSError:
        return False


def record_new_version(h: str) -> dict:
    if h not in HOSPITALS:
        raise RuntimeError(f"Unknown hospital {h!r}.")
    fl = _read_json(FL_STATE_PATH) or {}
    if _simulation_running(fl):
        raise RuntimeError("A simulation is running; wait for it to finish before recording a new version.")
    blocks = _read_json(CHAIN_PATH)
    if not isinstance(blocks, list) or not blocks:
        raise RuntimeError("No blockchain found. Run the simulation first.")
    if any(b.get("index", 0) > 0 and not (b.get("public_key_full") and b.get("signature_full")) for b in blocks):
        raise RuntimeError("Run the simulation again so the ledger can be reloaded and signed (old chain_state.json has truncated keys).")
    rec = chain_record(h, blocks)
    if rec is None:
        raise RuntimeError(f"{h} was never registered on the blockchain. Run the simulation first.")
    try:
        with open(_path(h), "rb") as f:
            raw = f.read()
        _read_text_frame(_path(h))
    except Exception as exc:
        raise RuntimeError(f"{h} data file is missing or not readable as CSV: {exc}")
    try:
        chain = InProcessChain.from_blocks(blocks, seed=SEED)
        ok = chain.verify_chain()
    except Exception:
        ok = False
    if not ok:
        raise RuntimeError("The existing blockchain does not verify; refusing to extend it.")
    new_hash = hash_bytes(raw)
    if new_hash == rec["hash"]:
        raise RuntimeError(f"{h}'s data already matches the blockchain; nothing new to record.")
    cells = changed_cells(h, limit=10**9)
    block = chain._add(
        {
            "event": "data_update",
            "hospital": h,
            "record_hash": new_hash,
            "previous_record_hash": rec["hash"],
            "metadata": {"n_records": len(_read_text_frame(_path(h))), "changed_cells": len(cells["modified"])},
        },
        signer=NodeKeyPair.generate(h, seed=SEED),
    )
    if not chain.verify_chain():
        raise RuntimeError("Internal error: the new block did not verify; nothing was written.")
    _write_json(CHAIN_PATH, chain.to_dict())
    if fl:
        fl["chain_length"], fl["chain_valid"] = len(chain), True
        _write_json(FL_STATE_PATH, fl)
    os.makedirs(os.path.dirname(_snap(h)), exist_ok=True)
    shutil.copyfile(_path(h), _snap(h))
    return block.to_dict()


def restore_original(h: str) -> str:
    if not os.path.isfile(_snap(h)):
        raise RuntimeError(f"No registered copy of {h}'s data to restore. Run the simulation first.")
    shutil.copyfile(_snap(h), _path(h))
    with open(_path(h), "rb") as f:
        return hash_bytes(f.read())
