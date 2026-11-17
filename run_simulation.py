"""
Main simulation: runs the full Blockchain + FL prototype end-to-end.
Writes fl_state.json and chain_state.json after every round so the dashboard
can poll and display live progress.

Usage:
    python run_simulation.py
    python run_simulation.py --rounds 5
    python run_simulation.py --dataset heart --split noniid
    python run_simulation.py --dataset pima --split iid --rounds 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

from chain import ChainBackend, get_chain
from data_split import split_for_hospitals
from datasets import load_dataset
import integrity
from hospital_node import ConsentError, HospitalNode
from fl_server import FedAvgServer, combine_stats

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
FL_STATE_PATH = os.path.join(BASE, "fl_state.json")
CHAIN_STATE_PATH = os.path.join(BASE, "chain_state.json")

FL_ROUNDS = 10
HOSPITAL_NAMES = ["Hospital_A", "Hospital_B", "Hospital_C"]
ROUND_DELAY = 0.6  # seconds — gives dashboard time to poll between rounds


def write_state(fl_state: dict, chain: ChainBackend):
    # Dashboard expects a JSON list of blocks; sync metadata lives in fl_state.
    with open(FL_STATE_PATH, "w") as f:
        json.dump(fl_state, f, indent=2)
    with open(CHAIN_STATE_PATH, "w") as f:
        json.dump(chain.to_dict(), f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Blockchain + FL medical simulation")
    p.add_argument("--rounds", type=int, default=FL_ROUNDS)
    p.add_argument(
        "--dataset",
        choices=["pima", "heart", "mimic"],
        default="pima",
        help="pima (default smoke) | heart (UCI multi-center) | mimic (synthetic stub)",
    )
    p.add_argument(
        "--split",
        choices=["iid", "noniid"],
        default="iid",
        help="IID equal shards or non-IID label/feature skew",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=ROUND_DELAY,
        help="Seconds between rounds (dashboard polling)",
    )
    p.add_argument(
        "--keep-data",
        action="store_true",
        help="Reuse hospital_data/*.csv (e.g. edited by hand) instead of re-splitting the dataset",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rounds = args.rounds

    print("=" * 54)
    print("  Blockchain-FL Medical Record Sharing  —  Simulation")
    print("=" * 54)
    print(f"  dataset={args.dataset}  split={args.split}  rounds={rounds}")
    print(f"  chain_backend={os.environ.get('CHAIN_BACKEND', 'inprocess')}")

    df, data_meta = load_dataset(args.dataset)
    # Drop attrs-only helpers; keep numeric feature frame
    model_df = df.copy()
    if "center" in model_df.columns:
        model_df = model_df.drop(columns=["center"])

    shards, test_df, split_info = split_for_hospitals(
        model_df, HOSPITAL_NAMES, mode=args.split
    )

    if args.keep_data:
        # Validate before any state file is touched.
        meta = integrity.read_meta()
        try:
            if not meta or meta.get("dataset") != args.dataset:
                raise ValueError(
                    f"existing hospital data is for dataset {meta.get('dataset') if meta else None!r}, "
                    f"not {args.dataset!r}"
                )
            shards = [integrity.load_shard(n) for n in HOSPITAL_NAMES]
            for n, df in zip(HOSPITAL_NAMES, shards):
                if list(df.columns) != list(model_df.columns):
                    raise ValueError(f"{n}.csv columns {list(df.columns)} differ from {list(model_df.columns)}")
                bad = df.apply(pd.to_numeric, errors="coerce").isna().values  # empty or non-numeric
                if bad.any():
                    r, c = (int(i[0]) for i in bad.nonzero())
                    raise ValueError(f"{n}.csv row {r} column {df.columns[c]!r} is empty or not a number")
            args.split = meta.get("split", args.split)  # shards keep the split they were made with
        except Exception as exc:
            print(
                f"ERROR: --keep-data cannot reuse hospital_data/: {exc}. "
                f"Run once without --keep-data (same --dataset) to create it.",
                file=sys.stderr,
            )
            sys.exit(1)
        # ponytail: test set and split_info stats still come from the original split.
        print("  --keep-data: training on the current files in hospital_data/")

    # Write shard files (canonical bytes) so file hash == registered hash.
    file_hashes = integrity.write_hospital_files(
        dict(zip(HOSPITAL_NAMES, shards)), args.dataset, args.split
    )
    chain = get_chain()

    fl_state: dict = {
        "status": "initializing",
        "dataset": data_meta.get("name", args.dataset),
        "dataset_meta": {
            k: v
            for k, v in data_meta.items()
            if k != "center_column" and not hasattr(v, "__array__")
        },
        "split": split_info,
        "hospitals": {name: {"records": 0, "registered": False} for name in HOSPITAL_NAMES},
        "rounds": [],
        "total_rounds": rounds,
        "current_round": 0,
        "final_accuracy": None,
        "chain_valid": True,
        "chain_length": 1,
        "replica_sync": {},
        "update_hashes": [],
        "consent_checks": [],
    }
    write_state(fl_state, chain)

    # ── Registration phase ────────────────────────────────────────────────────
    print("\nRegistering hospitals on blockchain...\n")
    nodes: list[HospitalNode] = []
    record_hashes: dict[str, str] = {}
    for name, shard in zip(HOSPITAL_NAMES, shards):
        node = HospitalNode(name, shard, chain)
        block, rec_hash = node.register_data()
        record_hashes[name] = rec_hash
        if rec_hash != file_hashes[name]:
            sys.exit(f"ERROR: {name} registered hash differs from its data file hash")
        idx = getattr(block, "index", len(chain) - 1)
        fl_state["hospitals"][name] = {
            "records": len(shard),
            "registered": True,
            "record_hash": rec_hash[:16] + "...",
            "block_index": idx,
            "positive_rate": round(float(shard.iloc[:, -1].mean()), 4),
        }
        print(
            f"  [OK] {name:12s}  {len(shard):3d} records  ->  Block #{idx}  "
            f"hash: {rec_hash[:16]}...  pos_rate={fl_state['hospitals'][name]['positive_rate']}"
        )
        nodes.append(node)

    # Federated feature scaling: clients share only (n, mean, var); server pools them
    scaler_mean, scaler_std = combine_stats([n.client.local_feature_stats() for n in nodes])
    for n in nodes:
        n.client.set_scaler(scaler_mean, scaler_std)

    # Consent demo: grant A→B, then enforce gate
    grant_ref = record_hashes["Hospital_A"]
    chain.grant_access(
        "Hospital_A", "Hospital_B", grant_ref,
        signer=next(n.keys for n in nodes if n.hospital_id == "Hospital_A"),
    )
    print(f"\n  [consent] Hospital_A granted Hospital_B access to ref {grant_ref[:16]}...")

    # Positive check — B may use A's ref
    try:
        nodes[1].use_record_ref(grant_ref)
        fl_state["consent_checks"].append(
            {"actor": "Hospital_B", "ref": grant_ref[:16] + "...", "allowed": True}
        )
        print("  [consent] Hospital_B use_record_ref: ALLOWED")
    except ConsentError as exc:
        fl_state["consent_checks"].append(
            {"actor": "Hospital_B", "allowed": False, "error": str(exc)}
        )
        print(f"  [consent] unexpected deny: {exc}")

    # Negative check — C must be denied without a grant
    try:
        nodes[2].use_record_ref(grant_ref)
        fl_state["consent_checks"].append(
            {"actor": "Hospital_C", "ref": grant_ref[:16] + "...", "allowed": True}
        )
        print("  [consent] Hospital_C use_record_ref: unexpectedly ALLOWED")
    except ConsentError:
        fl_state["consent_checks"].append(
            {"actor": "Hospital_C", "ref": grant_ref[:16] + "...", "allowed": False}
        )
        print("  [consent] Hospital_C use_record_ref: DENIED (no grant) ✓")

    fl_state["chain_length"] = len(chain)
    fl_state["replica_sync"] = {n.hospital_id: n.sync_replica() for n in nodes}

    chain.log_event(
        "fl_session_start",
        "coordinator",
        details={
            "rounds": rounds,
            "hospitals": HOSPITAL_NAMES,
            "dataset": data_meta.get("name", args.dataset),
            "split": split_info["mode"],
        },
    )

    fl_state["status"] = "training"
    write_state(fl_state, chain)

    # ── FL Training rounds ────────────────────────────────────────────────────
    server = FedAvgServer(test_df, scaler_mean, scaler_std)
    global_weights = None

    print(f"\nStarting Federated Learning -- {rounds} rounds, {len(nodes)} hospitals\n")
    print(f"  {'Round':>5}  {'Accuracy':>9}  {'Avg Loss':>9}  {'Blocks':>6}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*6}")

    for rnd in range(1, rounds + 1):
        round_updates = []
        round_hashes = []
        for node in nodes:
            weights, n_samples, loss, upd_hash = node.participate_in_round(
                rnd, global_weights
            )
            round_updates.append((weights, n_samples, loss))
            round_hashes.append(
                {
                    "hospital": node.hospital_id,
                    "round": rnd,
                    "update_hash": upd_hash[:16] + "...",
                }
            )

        global_weights = server.aggregate(round_updates)
        accuracy, avg_loss = server.evaluate(rnd, round_updates)

        chain.log_event(
            "fl_round_complete",
            "coordinator",
            details={
                "round": rnd,
                "accuracy": round(accuracy, 4),
                "avg_loss": round(avg_loss, 4),
                "model_hash": server.model_hash()[:16] + "...",
                "participating_hospitals": HOSPITAL_NAMES,
                "update_hashes": [h["update_hash"] for h in round_hashes],
            },
        )

        fl_state["current_round"] = rnd
        fl_state["rounds"] = server.round_metrics
        fl_state["chain_valid"] = chain.verify_chain()
        fl_state["chain_length"] = len(chain)
        fl_state["update_hashes"].extend(round_hashes)
        fl_state["replica_sync"] = {n.hospital_id: n.sync_replica() for n in nodes}
        fl_state["chain_sync"] = chain.sync_status()
        write_state(fl_state, chain)

        print(f"  {rnd:>5}  {accuracy:>9.4f}  {avg_loss:>9.4f}  {len(chain):>6}")
        time.sleep(args.delay)

    # ── Final state ───────────────────────────────────────────────────────────
    fl_state["status"] = "complete"
    fl_state["final_accuracy"] = round(accuracy, 4)
    fl_state["chain_valid"] = chain.verify_chain()
    fl_state["chain_length"] = len(chain)
    fl_state["replica_sync"] = {n.hospital_id: n.sync_replica() for n in nodes}
    write_state(fl_state, chain)

    print(f"\n{'='*54}")
    print(f"  Dataset / split : {args.dataset} / {split_info['mode']}")
    print(f"  Final accuracy  : {accuracy:.4f}  ({accuracy * 100:.1f}%)")
    print(
        f"  Blockchain      : {len(chain)} blocks  --  "
        f"integrity {'VALID' if chain.verify_chain() else 'INVALID'}"
    )
    sync_ok = all(s.get("last_sync_ok") for s in fl_state["replica_sync"].values())
    print(f"  Replica sync    : {'OK' if sync_ok else 'CHECK fl_state.json'}")
    print(f"  State files     : fl_state.json  |  chain_state.json")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()
