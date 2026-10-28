# 🏥 Major Project: Blockchain Medical Record Sharing with Federated Learning
### Phase-I Report — Deep Understanding Summary

---

## 🎯 The Core Problem (What You're Solving)

Healthcare generates **massive amounts of patient data**, but it's **siloed** — each hospital stores data in its own database. This causes:

| Problem | Impact |
|---|---|
| Fragmented EHRs across hospitals | Patients can't get continuity of care |
| No standard cross-institution sharing | Manual sharing via fax/print — no audit trail |
| Centralized databases = single point of failure | Ransomware, breaches, data leaks |
| No patient-controlled consent mechanism | Patients have no visibility or control |
| Privacy laws prevent raw data transfer | Hospitals can't collaborate on ML models |

**In India specifically**: ABDM (Ayushman Bharat Digital Mission) is trying to fix this, but the problem is unsolved at scale.

---

## 💡 Your Proposed Solution: Two Technologies Combined

You're combining **two paradigms** that each solve *half* of the problem:

### 1. 🔗 Permissioned Blockchain
- **What it does**: Maintains a tamper-evident audit trail of *who accessed what and when*
- **What goes ON-chain**: Cryptographic hashes of records, access-control metadata, consent records, audit entries
- **What stays OFF-chain**: Actual medical records, images, EHR documents (stored at each hospital)
- **Why permissioned (not public)**: Hospitals are known, vetted institutions — not anonymous participants. Hyperledger Fabric is the planned platform.
- **Why NOT centralized**: No single point of failure; no one entity controls all data

### 2. 🤖 Federated Learning (FL)
- **What it does**: Allows hospitals to collaboratively train ML models **without sharing raw patient data**
- **How**: Each hospital trains a model locally → shares only **model updates** (weights/gradients) → a central aggregator averages them (FedAvg algorithm by McMahan et al., 2017)
- **Why FL alone isn't enough**: No mechanism for participant trust, authentication, audit, or detecting poisoned model updates

### 🔑 Why BOTH Together?

> **Blockchain alone** doesn't help train better ML models — the data stays siloed analytically.  
> **Federated Learning alone** doesn't solve trust, auditability, consent, or malicious participants.  
> **Together**: FL handles privacy-preserving collaborative training; Blockchain handles trust, audit, and access control.

---

## 🏗️ Proposed Architecture (Conceptual)

```
Two parallel flows that intersect at the institutional level:

DATA & MODEL FLOW:
Patient/Data Owner
    ↓
Hospital's Local EHR Repository
    ↓
Local FL Client (trains on local data)
    ↓
Federated Aggregation Layer (averages model updates)
    ↓
Updated Global Model

TRUST & AUDIT FLOW:
Healthcare Institution
    ↓
Permissioned Blockchain Network
    ↓
Smart Contracts (access permissions, consent, audit logs)
    ↓
Independently Verifiable Audit Trail
```

**On-chain** (Hyperledger Fabric / smart contracts):
- Record hashes (SHA-256 fingerprints)
- Access permission entries
- Consent/status metadata
- Audit/transaction records

**Off-chain** (each hospital's own infrastructure):
- Actual EHR documents
- Medical images
- Lab files
- Any large/sensitive clinical data

---

## 🎯 Project Objectives

1. Design a **permissioned blockchain architecture** for secure medical record sharing
2. Implement **Federated Learning** for privacy-preserving collaborative ML across hospitals
3. Integrate both into a **unified prototype system**
4. Evaluate **security and privacy mechanisms** (authentication, access control, audit trails)
5. Assess feasibility across technical, economic, legal, and scalability dimensions

---

## 📏 Scope (What You WILL and WON'T Do)

### ✅ IN Scope
- Permissioned blockchain with 3-5 simulated hospital nodes
- FL training workflow (local training → update sharing → aggregation)
- On-chain metadata (hashes, access grants, consent)
- Off-chain raw data (stored locally at each simulated institution)
- Synthetic/public datasets only (e.g., MIMIC, PhysioNet)
- Simple REST API frontend for demonstration
- Docker containers to simulate multiple institutional nodes

### ❌ OUT of Scope
- Real hospital EHR integration
- Real patient data (zero real patient data will be used)
- Production-grade deployment or clinical use
- Regulatory certification (DPDP Act 2023 compliance)
- Formal penetration testing or security certification
- National-scale performance evaluation

---

## ⚠️ Key Security Threats Acknowledged

| Threat | Description |
|---|---|
| **Data Poisoning** | Malicious hospital corrupts local training data |
| **Model Poisoning** | Malicious participant manipulates weights/gradients sent to aggregator |
| **Sybil Attacks** | One attacker poses as multiple hospitals to dominate model aggregation |
| **Inference Attacks** | Adversary reconstructs private training data from model updates |

**Mitigation (planned)**: Secure Aggregation (Bonawitz et al., 2017) — server sees only the *sum* of updates, not individual contributions.

---

## ✅ Feasibility Assessment Summary

| Dimension | Verdict | Notes |
|---|---|---|
| Technical | ✅ Feasible | PyTorch/TF + Flower (FL), Hyperledger Fabric (blockchain), Docker |
| Economic | ✅ Feasible | Open-source stack; existing student hardware |
| Operational | ✅ Feasible (research prototype) | Simulated multi-institution workflow |
| Schedule | ✅ Fits 2-semester B.Tech timeline | Phase-I = survey/design; Phase-II = prototype/eval |
| Legal & Regulatory | ✅ For research prototype | No real patient data; DPDP Act out of scope |
| Security & Privacy | ✅ With acknowledged residual risk | FL + blockchain reduce but don't eliminate risk |
| Data | ✅ Synthetic/public datasets | No real patient data will ever be used |
| Scalability | ⚠️ Open question at production scale | Prototype scale is feasible; production needs empirical study |

---

## 🛠️ Technology Stack (Planned for Phase-II)

- **Blockchain**: Hyperledger Fabric (permissioned, chaincode/smart contracts)
- **FL Framework**: Flower (simulates multiple FL clients on one machine)
- **ML Backend**: PyTorch or TensorFlow
- **APIs**: REST APIs connecting FL + blockchain to a frontend
- **Infrastructure**: Docker containers (simulating hospital nodes)
- **Data**: Synthetic/public healthcare datasets

---

## 📚 Key Papers Supporting This Work

| Paper | Relevance |
|---|---|
| Nakamoto (2008) — Bitcoin | Foundation of the blockchain audit layer |
| McMahan et al. (2017) — FedAvg | Core FL algorithm being used |
| Bonawitz et al. (2017) — Secure Aggregation | Privacy mechanism for FL model updates |
| Warnat-Herresthal et al. (2021) — Swarm Learning | Empirical proof blockchain+FL can work in clinical settings |
| Myrzashova et al. (2023) — Systematic Review | Confirms blockchain-FL integration in healthcare is underexplored → your research gap |

---

## 🔮 What Phase-II Will Build

- Working prototype of the integrated system
- Performance evaluation (accuracy, latency, throughput)
- Security testing of implemented mechanisms
- Empirical scalability assessment
- Finalized architecture diagram (Figure 1 placeholder to be filled)

---

> **Bottom line**: This is a well-scoped B.Tech Major Project that sits at the intersection of **distributed systems**, **privacy-preserving ML**, and **healthcare informatics**. The research gap is real — most existing work treats blockchain and FL separately in healthcare; their integrated design is comparatively underexplored. Your project proposes to close that gap with a working prototype using synthetic data.
