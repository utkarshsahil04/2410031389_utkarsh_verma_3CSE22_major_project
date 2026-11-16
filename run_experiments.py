"""
Baselines and attack experiments (Roadmap Step 5).

Compares:
  - Centralized SGD (pooled upper bound)
  - Local-only (no FL — average of per-hospital models)
  - FedAvg (current simulator path)
  - Label-flip / weight-poison attack (accuracy drop)
  - Chain tamper attempt (verify_chain must fail)

Usage:
    python run_experiments.py
    python run_experiments.py --dataset pima --rounds 3 --out experiment_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from chain import InProcessChain, get_chain
from data_split import split_for_hospitals
from datasets import load_dataset
from fl_client import HospitalClient
from fl_server import FedAvgServer
from hospital_node import HospitalNode

BASE = os.path.dirname(__file__)
DEFAULT_OUT = os.path.join(BASE, "experiment_results.json")
HOSPITAL_NAMES = ["Hospital_A", "Hospital_B", "Hospital_C"]


def _xy(df: pd.DataFrame):
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    return X, y


def _auc(y_true, proba) -> float | None:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return None


def eval_sgd(coef, intercept, X, y) -> dict:
    model = SGDClassifier(loss="log_loss", random_state=42)
    model.coef_ = np.asarray(coef)
    model.intercept_ = np.asarray(intercept)
    model.classes_ = np.array([0, 1])
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "auc": _auc(y, proba),
    }


def run_centralized(train_df: pd.DataFrame, test_df: pd.DataFrame, epochs: int = 5) -> dict:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df.iloc[:, :-1].values)
    ytr = train_df.iloc[:, -1].values.astype(int)
    Xte = scaler.transform(test_df.iloc[:, :-1].values)
    yte = test_df.iloc[:, -1].values.astype(int)

    model = SGDClassifier(
        loss="log_loss", alpha=0.001, max_iter=epochs, tol=None, random_state=42
    )
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    return {
        "name": "centralized_sgd",
        "accuracy": float(accuracy_score(yte, preds)),
        "auc": _auc(yte, proba),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def run_local_only(shards: list[pd.DataFrame], test_df: pd.DataFrame) -> dict:
    """Train one model per hospital; report mean accuracy on shared test set."""
    accs, aucs = [], []
    Xte_raw, yte = _xy(test_df)
    for shard in shards:
        client = HospitalClient("local", shard)
        weights, _, _ = client.train(None)
        # Evaluate with hospital's own scaler on raw test — approximate local quality
        Xte = client.scaler.transform(Xte_raw)
        m = eval_sgd(weights["coef"], weights["intercept"], Xte, yte)
        accs.append(m["accuracy"])
        if m["auc"] is not None:
            aucs.append(m["auc"])
    return {
        "name": "local_only",
        "accuracy": float(np.mean(accs)),
        "auc": float(np.mean(aucs)) if aucs else None,
        "per_hospital_accuracy": [round(a, 4) for a in accs],
    }


def run_fedavg(
    shards: list[pd.DataFrame],
    test_df: pd.DataFrame,
    rounds: int,
    *,
    poison: str | None = None,
) -> dict:
    """
    poison:
      None — clean FedAvg
      label_flip — Hospital_C flips all local labels
      weight_poison — Hospital_C scales coef by large factor after train
    """
    chain = InProcessChain(seed="exp")
    nodes = []
    for name, shard in zip(HOSPITAL_NAMES, shards):
        data = shard.copy()
        if poison == "label_flip" and name == "Hospital_C":
            data.iloc[:, -1] = 1 - data.iloc[:, -1].astype(int)
        nodes.append(HospitalNode(name, data, chain))
        nodes[-1].register_data()

    server = FedAvgServer(test_df)
    global_weights = None
    history = []

    for rnd in range(1, rounds + 1):
        updates = []
        for node in nodes:
            w, n, loss, _ = node.participate_in_round(rnd, global_weights)
            if poison == "weight_poison" and node.hospital_id == "Hospital_C":
                # Large antagonistic scaling so FedAvg accuracy visibly drops
                w = {
                    "coef": -w["coef"] * 25.0,
                    "intercept": -w["intercept"] * 25.0,
                }
            updates.append((w, n, loss))
        global_weights = server.aggregate(updates)
        acc, avg_loss = server.evaluate(rnd, updates)
        history.append({"round": rnd, "accuracy": round(acc, 4), "avg_loss": round(avg_loss, 4)})

    Xte_raw, yte = _xy(test_df)
    # Server eval already uses its own scaler fit on test — use last accuracy
    name = "fedavg" if poison is None else f"fedavg_{poison}"
    return {
        "name": name,
        "accuracy": history[-1]["accuracy"] if history else None,
        "auc": None,  # server path reports accuracy; AUC optional
        "history": history,
        "final_model_hash": server.model_hash()[:16] + "...",
        "poison": poison,
    }


def run_chain_tamper() -> dict:
    chain = get_chain("inprocess")
    chain.add_record("Hospital_A", "abc123", {"n": 1})
    chain.log_event("fl_round", "coordinator", {"round": 1})
    assert chain.verify_chain(), "pre-tamper chain must be valid"
    # Tamper payload in-place
    chain.chain[1].data["record_hash"] = "TAMPERED"
    valid_after = chain.verify_chain()
    return {
        "name": "chain_tamper",
        "verify_before_expected": True,
        "verify_after_tamper": valid_after,
        "assert_fails": valid_after is False,
    }


def print_table(results: list[dict]):
    print("\n" + "=" * 64)
    print(f"  {'Experiment':<28}  {'Accuracy':>10}  {'AUC':>10}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}")
    for r in results:
        if r["name"] == "chain_tamper":
            status = "FAIL(ok)" if r.get("assert_fails") else "UNEXPECTED"
            print(f"  {'chain_tamper':<28}  {status:>10}  {'—':>10}")
            continue
        acc = r.get("accuracy")
        auc = r.get("auc")
        acc_s = f"{acc:.4f}" if acc is not None else "—"
        auc_s = f"{auc:.4f}" if auc is not None else "—"
        print(f"  {r['name']:<28}  {acc_s:>10}  {auc_s:>10}")
    print("=" * 64 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pima", "heart", "mimic"], default="pima")
    ap.add_argument("--split", choices=["iid", "noniid"], default="iid")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    t0 = time.time()
    df, meta = load_dataset(args.dataset)
    if "center" in df.columns:
        df = df.drop(columns=["center"])

    shards, test_df, split_info = split_for_hospitals(
        df, HOSPITAL_NAMES, mode=args.split
    )
    train_df = pd.concat(shards, ignore_index=True)

    results = []
    print(f"[exp] dataset={meta['name']} split={split_info['mode']} rounds={args.rounds}")

    print("[exp] centralized SGD...")
    results.append(run_centralized(train_df, test_df, epochs=max(args.rounds, 5)))

    print("[exp] local-only...")
    results.append(run_local_only(shards, test_df))

    print("[exp] FedAvg (clean)...")
    fed = run_fedavg(shards, test_df, args.rounds, poison=None)
    results.append(fed)

    print("[exp] FedAvg + label-flip (Hospital_C)...")
    atk_label = run_fedavg(shards, test_df, args.rounds, poison="label_flip")
    atk_label["accuracy_drop_vs_fedavg"] = round(
        (fed["accuracy"] or 0) - (atk_label["accuracy"] or 0), 4
    )
    results.append(atk_label)

    print("[exp] FedAvg + weight-poison (Hospital_C)...")
    atk_w = run_fedavg(shards, test_df, args.rounds, poison="weight_poison")
    atk_w["accuracy_drop_vs_fedavg"] = round(
        (fed["accuracy"] or 0) - (atk_w["accuracy"] or 0), 4
    )
    results.append(atk_w)

    print("[exp] chain tamper...")
    tamper = run_chain_tamper()
    assert tamper["assert_fails"], "Tampered chain must fail verify_chain"
    results.append(tamper)

    payload = {
        "dataset": meta["name"],
        "split": split_info,
        "rounds": args.rounds,
        "elapsed_sec": round(time.time() - t0, 2),
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print_table(results)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
