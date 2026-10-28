# Implementation Roadmap

Ordered upgrade path for the Blockchain + Federated Learning prototype.
Do these **in sequence** — each step builds on the previous interface.

---

## Suggested order

| Step | Goal | Status |
|---|---|---|
| **1** | Abstract chain interface with two backends: in-process (now) + Fabric gateway (later) | **Done** |
| **2** | Signed blocks, per-node chain copies, and update-hash commits | **Done** |
| **3** | Consent gating and non-IID data splits | **Done** |
| **4** | Switch dataset to Fed-Heart-Disease, then MIMIC | **Partial** (heart Done; MIMIC stub) |
| **5** | Run baselines and attack experiments | **Done** |
| **6** | Replace backend with real Fabric; sklearn → Flower + PyTorch if time allows | **Partial** (Fabric mock Done; Flower stub) |

---

## Step 1 — Abstract chain backends

**Why first:** Every later blockchain feature (signatures, Fabric, consent) must call one stable API so the FL simulator does not care which ledger is underneath.

**Done when:**

- [x] `ChainBackend` ABC: `add_record`, `grant_access`, `log_event`, `verify_chain`, `to_dict`, `__len__`
- [x] `InProcessChain` — current SHA-256 append-only chain
- [x] `FabricGateway` — stub documenting the future Hyperledger client
- [x] Factory: `get_chain(backend="inprocess"|"fabric")` (env: `CHAIN_BACKEND`)
- [x] Simulation / hospitals only depend on the abstract interface

**How to select backend:**

```powershell
# default — local prototype
python run_simulation.py

# Fabric mock (no Docker network required)
$env:CHAIN_BACKEND = "fabric"
python run_simulation.py
```

---

## Step 2 — Signed blocks, per-node copies, update-hash commits

**Status: Done**

- [x] Each block (non-genesis) signed with Ed25519 node keys (`chain/crypto.py`)
- [x] `ChainReplica` per hospital syncs from authority; rejects unsigned / invalid peers
- [x] After each FL round, commit `SHA-256(model_update)` per hospital (`fl_update_hash` events)
- [x] Sync / signature metadata in `fl_state.json` (`replica_sync`, `chain_sync`, `update_hashes`)

---

## Step 3 — Consent gating + non-IID splits

**Status: Done**

- [x] `ChainBackend.has_access` + `HospitalNode.use_record_ref` (raises `ConsentError` without grant)
- [x] Simulation demos allow B after grant, deny C without grant
- [x] `--split iid|noniid` — non-IID uses feature quantile skew (Age/Glucose/…) + label bias
- [x] Split stats written to `fl_state.json` / printed in simulation output

---

## Step 4 — Datasets

**Status: Partial**

| Dataset | Status | Notes |
|---|---|---|
| PIMA | **Done** | Default smoke test (`--dataset pima`) |
| Fed-Heart-Disease style | **Done** | UCI multi-center heart tables (`--dataset heart`) |
| MIMIC | **Stub** | No PhysioNet download; synthetic sample + access README |

```powershell
python run_simulation.py --dataset pima
python run_simulation.py --dataset heart --split noniid
python run_simulation.py --dataset mimic   # synthetic only
```

---

## Step 5 — Baselines and attacks

**Status: Done** — `python run_experiments.py`

| Experiment | Metric |
|---|---|
| Centralized SGD (pooled data upper bound) | Accuracy / AUC |
| Local-only (no FL) | Accuracy / AUC |
| FedAvg (current) | Accuracy vs rounds |
| Label-flip / weight-poison attack | Drop vs clean FedAvg |
| Chain tamper attempt | `verify_chain` must fail |

Results → `experiment_results.json` (+ printed table).

---

## Step 6 — Real Fabric + Flower/PyTorch

**Status: Partial**

| Piece | Status |
|---|---|
| `FabricGateway` mock mode (`FABRIC_MODE=mock`) | **Done** — submit/evaluate semantics, demo-safe |
| Real Hyperledger Fabric Gateway SDK | **Stub** — `connect()` TODOs; `FABRIC_MODE=sdk` raises |
| Flower + PyTorch swap | **Stub** — `fl_adapters/flower_pytorch_stub.py` |

Default demo does **not** require Docker Fabric.

---

## Non-goals (stay out of scope)

- Real hospital EHR connectors  
- Real patient data  
- Regulatory certification / production SLA  

---

## Current code map

```
code/chain/
  __init__.py       # get_chain() factory
  interface.py      # ChainBackend ABC (+ has_access)
  crypto.py         # Ed25519 NodeKeyPair
  inprocess.py      # InProcessChain (signed blocks)
  replica.py        # per-hospital ChainReplica
  fabric.py         # FabricGateway mock + SDK hooks
code/datasets/      # pima / heart / mimic loaders
code/data_split.py  # iid | noniid
code/run_experiments.py
code/fl_adapters/flower_pytorch_stub.py
code/blockchain.py  # backward-compatible re-export
```
