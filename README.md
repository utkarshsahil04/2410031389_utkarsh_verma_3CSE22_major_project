# Blockchain Medical Record Sharing with Federated Learning

B.Tech major-project prototype that combines a **permissioned blockchain audit trail** with **Federated Learning (FedAvg)** so hospitals can collaborate on a clinical risk model **without sharing raw patient records**.

Two UIs are available: a lightweight HTML dashboard (`api.py`) and a full **Streamlit** control panel.

> Design diagrams: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
> Roadmap status: [`ROADMAP.md`](ROADMAP.md)  
> Phase-I notes: [`project_understanding.md`](project_understanding.md)

---

## Features

| Area | What you get |
|---|---|
| Federated Learning | 3 hospital nodes, FedAvg, IID / non-IID splits |
| Blockchain | Ed25519-signed blocks, per-node replicas, update-hash commits |
| Consent | `grant_access` required before cross-hospital record use |
| Datasets | PIMA (default), UCI Heart, MIMIC-shaped synthetic stub |
| Experiments | Centralized / local / FedAvg / poison / tamper baselines |
| Backends | `inprocess` (default) · Fabric **mock** (`CHAIN_BACKEND=fabric`) |
| UIs | HTML dashboard + Streamlit app |

Public / synthetic clinical benchmarks only — **no live hospital PHI**.

---

## Requirements

- Python **3.10+** (tested with 3.12)
- Windows / macOS / Linux

Install (from `code/`):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Packages: `numpy`, `pandas`, `scikit-learn`, `cryptography`, `streamlit`, `plotly`

---

## Quick start

```powershell
cd "d:\COllege\3rd Year\major project\code"
.\venv\Scripts\Activate.ps1
```

### Option A — Streamlit UI (recommended)

```powershell
streamlit run streamlit_app.py
```

Tabs: Overview · Run Simulation · Hospitals · FL Metrics · Blockchain Audit · Experiments

### Option B — HTML dashboard + CLI

**Terminal 1 — dashboard**

```powershell
python api.py
```

Open **http://localhost:5000**

**Terminal 2 — simulation**

```powershell
python run_simulation.py
```

### Useful simulation flags

```powershell
python run_simulation.py --rounds 2
python run_simulation.py --dataset heart --split noniid
python run_simulation.py --dataset pima --split iid --rounds 10
python run_simulation.py --dataset mimic --rounds 2
```

| Flag | Values | Default |
|---|---|---|
| `--rounds` | int | `10` |
| `--dataset` | `pima` \| `heart` \| `mimic` | `pima` |
| `--split` | `iid` \| `noniid` | `iid` |
| `--delay` | seconds between rounds | `0.6` |

### Chain backend

```powershell
$env:CHAIN_BACKEND = "inprocess"   # default local signed chain
$env:CHAIN_BACKEND = "fabric"      # FabricGateway mock (no Docker required)
$env:FABRIC_MODE = "mock"          # default; "sdk" is stubbed for real Fabric later
```

### Experiments & self-check

```powershell
python run_experiments.py --dataset pima --rounds 3
# → experiment_results.json

python self_check.py
```

---

## How it works

1. **Load + split** — choose dataset; shard across Hospital_A / B / C (IID or non-IID); hold out a test set.
2. **Register** — each hospital hashes its shard, anchors a **signed** on-chain registration, syncs a local replica.
3. **Consent** — `grant_access(A→B)` allows B to use A’s record ref; C is denied without a grant.
4. **Train** — each round: local SGD → commit `SHA-256(update)` → FedAvg → log global model hash → replica sync.
5. **Visualize** — HTML dashboard and/or Streamlit poll `fl_state.json` / `chain_state.json`.

```
Hospitals (private shards) ──weights only──► FedAvg Server ──metrics──► UIs
         │                                         │
         └──── signed hashes / grants / audit ─────┴──► ChainBackend
                                                      (inprocess | fabric mock)
```

---

## Datasets

| Name | Flag | Source | Notes |
|---|---|---|---|
| PIMA Indians Diabetes | `--dataset pima` | Public auto-download → `data/diabetes.csv` | Default smoke test (768 rows) |
| UCI Heart Disease | `--dataset heart` | Public multi-center UCI tables | Better multi-hospital demo |
| MIMIC-shaped | `--dataset mimic` | **Synthetic stub only** | Real MIMIC needs PhysioNet credentials — see `code/data/mimic/README_MIMIC_ACCESS.txt` |

---

## Project structure

```
major project/
├── README.md
├── ARCHITECTURE.md
├── ROADMAP.md
├── project_understanding.md
└── code/
    ├── streamlit_app.py       # Streamlit control panel
    ├── ui/                    # Streamlit helpers (state, runner)
    ├── .streamlit/config.toml
    ├── api.py                 # HTML dashboard HTTP server
    ├── dashboard/             # Static HTML/CSS/JS UI
    ├── run_simulation.py      # End-to-end Blockchain + FL run
    ├── run_experiments.py     # Baselines + attack experiments
    ├── hospital_node.py       # FL client + consent + replica + update hashes
    ├── fl_client.py           # Local SGD training
    ├── fl_server.py           # FedAvg + evaluation
    ├── data_split.py          # IID / non-IID partitioning
    ├── datasets/              # pima / heart / mimic loaders
    ├── chain/                 # ChainBackend, crypto, replicas, Fabric mock
    ├── fl_adapters/           # Flower/PyTorch migration stub
    ├── blockchain.py          # Backward-compatible re-export
    ├── self_check.py
    ├── requirements.txt
    ├── data/                  # Downloaded / generated CSVs
    ├── fl_state.json          # Runtime FL metrics (generated)
    ├── chain_state.json       # Runtime chain dump (generated)
    └── experiment_results.json
```

---

## API endpoints (HTML dashboard)

| Endpoint | Description |
|---|---|
| `GET /` | Dashboard UI |
| `GET /state` | FL simulation state (`fl_state.json`) |
| `GET /chain` | Blockchain dump (`chain_state.json`) |

Port: **5000**

---

## Experiments (what gets measured)

| Experiment | Purpose |
|---|---|
| Centralized SGD | Pooled-data upper bound |
| Local-only | No collaboration baseline |
| FedAvg | Current federated path |
| Label-flip / weight-poison | Accuracy drop under attack |
| Chain tamper | `verify_chain()` must fail |

---

## Scope & limitations

**In scope**

- Signed permissioned chain + FedAvg FL prototype
- Consent gating, replicas, non-IID splits
- Fabric **mock** submit/evaluate path
- Streamlit + HTML dashboards
- Public / synthetic clinical data only

**Stubbed / out of scope**

- Live Hyperledger Fabric network (SDK hooks only)
- Flower + PyTorch training stack (adapter stub in `fl_adapters/`)
- Real MIMIC download / real hospital EHR integration
- Regulatory certification (e.g. DPDP production compliance)

---

## References

- McMahan et al. (2017) — FedAvg
- Bonawitz et al. (2017) — Secure Aggregation (future upgrade)
- Nakamoto (2008) — blockchain audit foundations
- PIMA Indians Diabetes Database — public ML benchmark
- UCI Heart Disease — public multi-center clinical tables
