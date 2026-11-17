# Full project walkthrough

## What this project is

Hospitals want to **train a shared medical ML model** and keep an **audit trail of access**, but they **must not pool raw patient rows**.

Your prototype solves that with two layers:

1. **Federated Learning (FedAvg)** — each hospital trains locally; only model weights are shared  
2. **Permissioned blockchain** — only hashes, consent grants, and FL audit events go on-chain (signed)

Public clinical data only (PIMA / UCI Heart / synthetic MIMIC stub) — not live hospital PHI.

---


## End-to-end workflow

```mermaid
flowchart TD
  A[Start: run_simulation / Streamlit / api.py] --> B[datasets: load PIMA or Heart or MIMIC stub]
  B --> C[data_split: train/test + IID or non-IID shards]
  C --> D[get_chain: InProcess or Fabric mock]
  D --> E[Hospital_A / B / C register data hashes on-chain]
  E --> F[grant_access A to B - consent demo]
  F --> G[FL round loop]
  G --> H[Each hospital: fl_client.train locally]
  H --> I[Commit update hash + participation on chain]
  I --> J[FedAvgServer: aggregate + evaluate]
  J --> K[Log global model hash on chain]
  K --> L[Write fl_state.json + chain_state.json]
  L --> M{More rounds?}
  M -->|yes| G
  M -->|no| N[Dashboards read JSON and show metrics/audit]
```

**In one sentence:** load data → split across 3 hospitals → register hashes → train locally each round → FedAvg → log everything on a signed chain → UIs read the JSON.

---

## Runtime: how you actually run it

| What you run | Role |
|---|---|
| `python run_simulation.py` | The “brain” — runs the full FL + chain story |
| `python api.py` | Serves HTML dashboard at `http://localhost:5000` |
| `python -m streamlit run streamlit_app.py` | Teacher-friendly UI: how it works, run the demo, live data check, charts, audit, experiments |
| `python run_experiments.py` | Baselines + attack tests → `experiment_results.json` |
| `python self_check.py` | Quick sanity asserts |

Typical demo: start `api.py` **or** Streamlit, then run the simulation (Streamlit can start it for you).

---

## File-by-file: what each piece does

### Entry points / orchestrators

| File | Job |
|---|---|
| **`run_simulation.py`** | Main workflow: load dataset → split → create hospitals → register → consent → FL rounds → write `fl_state.json` / `chain_state.json` |
| **`run_experiments.py`** | Compare centralized vs local vs FedAvg; label-flip / weight-poison; tamper the chain and prove `verify_chain()` fails |
| **`api.py`** | Tiny HTTP server: `/` dashboard, `/state` → FL JSON, `/chain` → blockchain JSON |
| **`streamlit_app.py`** | Main UI (tabs: How it works, Run the demo, Live Data Check, Hospitals, Training results, Blockchain audit, Experiments) |
| **`integrity.py`** | Live dataset integrity: re-hash `hospital_data/*.csv`, compare with the on-chain hash, list changed cells, record a signed `data_update`, undo |
| **`self_check.py`** | Unit-style checks without a test framework |

### Federated Learning core

| File | Job |
|---|---|
| **`fl_client.py`** | One hospital’s local trainer (`SGDClassifier`). Input: global weights. Output: local weights + sample count + loss. Never sends raw rows. |
| **`fl_server.py`** | **FedAvg**: weighted average of client weights; evaluates accuracy on held-out test set; hashes global model |
| **`hospital_node.py`** | Glue for one hospital: FL client + chain logging + **consent gate** + **replica sync** + **update-hash** commits |
| **`data_split.py`** | Hold-out test set; shard train data **IID** (equal) or **non-IID** (age-band feature skew, so hospitals see different patients) |
| **`fl_adapters/`** | Stub docs for later Flower + PyTorch swap (not used in the default demo) |

### Blockchain (`chain/`)

| File | Job |
|---|---|
| **`chain/interface.py`** | Abstract API: `add_record`, `grant_access`, `has_access`, `log_event`, `verify_chain`, `to_dict` |
| **`chain/crypto.py`** | Ed25519 keypairs / signatures for hospitals |
| **`chain/inprocess.py`** | Default local signed append-only ledger (your “blockchain” for demos) |
| **`chain/replica.py`** | Each hospital’s copy of the chain; rejects bad/unsigned blocks |
| **`chain/fabric.py`** | Fabric **mock** (same API); real SDK path is stubbed |
| **`chain/__init__.py`** | `get_chain()` factory — picks backend from `CHAIN_BACKEND` |
| **`blockchain.py`** | Thin re-export so old `from blockchain import …` still works |

**On-chain (allowed):** dataset hash, access grants, FL participation, update hashes, model hash  
**Off-chain (private):** actual patient feature rows at each hospital

### Datasets

| File | Job |
|---|---|
| **`datasets/__init__.py`** | `load_dataset("pima"|"heart"|"mimic")` |
| **`datasets/pima.py`** | PIMA diabetes (default; auto-download if missing) |
| **`datasets/heart.py`** | UCI Heart: 920 rows from 4 centres; mostly-missing columns dropped, the rest median-imputed |
| **`datasets/mimic.py`** | Tiny **synthetic** MIMIC-shaped sample (no PhysioNet download) |
| **`data/`** | Stored CSVs / center files used by those loaders |

### UIs

| File | Job |
|---|---|
| **`dashboard/index.html` + `app.js` + `style.css`** | Browser dashboard; polls `/state` and `/chain` |
| **`ui/state.py`** | Streamlit helpers: load/normalize JSON state |
| **`ui/runner.py`** | Streamlit starts `run_simulation.py` as a subprocess |
| **`.streamlit/config.toml`** | Streamlit theme |

### Generated state (written while running)

| File | Job |
|---|---|
| **`fl_state.json`** | Live FL status: hospitals, rounds, accuracy, consent/replica meta |
| **`chain_state.json`** | Full block list for the audit UI |
| **`experiment_results.json`** | Output of `run_experiments.py` |
| **`hospital_data/`** | Each hospital's private shard (`Hospital_A.csv` …), the registered copies (`.registered/`) and `meta.json` |

### Docs / config

| File | Job |
|---|---|
| **`README.md`** | How to install and run |
| **`ARCHITECTURE.md`** | Design diagrams and component map |
| **`ROADMAP.md`** | What steps are Done / Partial / Stub |
| **`project_understanding.md`** | Phase-I problem/solution notes |
| **`PROJECT_GUIDE.md`** | This file — full walkthrough of files and workflow |
| **`requirements.txt`** | Python dependencies |

---

## One FL round (closer look)

Before round 1, each hospital shares only **feature statistics** (row count, mean, variance); the server pools them into one global scaler that every hospital and the test evaluation use.

For each round `r`:

1. **Hospital_A/B/C** call `participate_in_round(r, global_weights)`  
2. Local `fl_client.train` does one SGD pass on **only that hospital’s shard**  
3. Hospital logs `fl_round_participation` + `fl_update_hash` (SHA-256 of its weights), **signed with its own key**, and syncs its replica  
4. **FedAvgServer** re-hashes each received update, rejects any that differ from the committed hash, then averages weights (weighted by number of samples)  
5. Server evaluates on the **shared test set** (accuracy / loss)  
6. Coordinator logs `fl_round_complete` + global model hash  
7. State JSON files update → dashboards refresh  

Raw CSV rows never leave the hospital node.

---

## Consent workflow (why blockchain matters here)

1. A registers its dataset → on-chain `register_record` with hash, **signed by A**  
2. A calls `grant_access(A → B)`, signed by A. Only the owner of a record can grant it (B granting A's record to C is refused)  
3. B can `use_record_ref(A’s hash)` → allowed  
4. C tries the same without a grant → **`ConsentError`**  

`has_access` re-verifies the chain first, so editing a grant block revokes access instead of leaving a stale permission.
So the chain is not storing medical files; it stores **who was allowed to reference what**, plus FL audit.

---

## Live data integrity (the "change the data" demo)

1. Every run writes each hospital's shard to `hospital_data/<Hospital>.csv`; the SHA-256 of those exact bytes is what gets registered.
2. **Live Data Check** re-hashes the three files every 2 seconds and compares them with the latest registration **signed by that hospital** on the re-verified chain.
3. Edit any number (in the tab's editor, Excel or VS Code) → the card turns red within about 2 s and the changed cells are listed.
4. **Record new version on blockchain** → a `data_update` block signed by that hospital (old hash → new hash) → card turns green.
5. **Train on the current (edited) hospital data** (`--keep-data`) → the next run registers and trains on the edited files, producing a new model hash.
6. If someone forges `chain_state.json`, every card says the blockchain failed verification instead of trusting it.

## What the chain does not guarantee (say this before you are asked)

- Deleting the newest blocks is not detectable without an external anchor (e.g. a second organisation's copy).
- Demo keys come from a public seed; a real deployment would issue them from a certificate authority.
- A hospital can still send a poisoned update; the chain makes it **traceable**, robust aggregation would make it **harmless**.

---

## Mental model of folders

```
code/
├── run_*.py, api.py, streamlit_app.py   ← you start here
├── fl_*.py, hospital_node.py, data_split.py  ← FL logic
├── chain/                               ← trust / audit / signatures
├── datasets/ + data/                    ← what gets trained on
├── dashboard/ + ui/                     ← what you see
└── *_state.json                         ← live bridge between sim and UI
```

---

## What to say in a viva (short)

> “Three hospitals keep data locally. They train with FedAvg and share only weights (plus feature statistics for a common scaler). Dataset fingerprints, consent grants and model-update hashes are written to a permissioned chain where each member signs its own actions. If anyone edits a hospital's data, the live check shows the fingerprint no longer matches the chain. We evaluate with public PIMA and multi-centre UCI Heart data, non-IID splits, and attack/tamper experiments.”
