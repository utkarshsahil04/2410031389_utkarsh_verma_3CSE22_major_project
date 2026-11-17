"""
Hospital node: one participating institution in the simulation.
Wires together a FL client (local training), chain audit logging,
local chain replica sync, and consent-gated record access.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from chain import ChainBackend, ChainReplica, NodeKeyPair
from chain.inprocess import InProcessChain
from fl_client import HospitalClient


def hash_weights(weights: dict) -> str:
    """SHA-256 of a hospital model update (coef + intercept)."""
    payload = {
        "coef": np.asarray(weights["coef"]).tolist(),
        "intercept": np.asarray(weights["intercept"]).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ConsentError(PermissionError):
    """Raised when a hospital tries to use a record without an on-chain grant."""


class HospitalNode:
    def __init__(
        self,
        hospital_id: str,
        data: pd.DataFrame,
        chain: ChainBackend,
        *,
        keys: NodeKeyPair | None = None,
    ):
        self.hospital_id = hospital_id
        self.client = HospitalClient(hospital_id, data)
        self.chain = chain
        self.keys = keys or NodeKeyPair.generate(hospital_id, seed="demo-seed")
        self.replica = ChainReplica(hospital_id, keys=self.keys)
        self.record_hash: str | None = None

    def _authority_ledger(self) -> InProcessChain | None:
        if isinstance(self.chain, InProcessChain):
            return self.chain
        mock = getattr(self.chain, "mock_ledger", None)
        return mock if isinstance(mock, InProcessChain) else None

    def register_data(self):
        """Hash local dataset and anchor it on-chain (raw data stays off-chain)."""
        record_hash = self.client.get_data_hash()
        self.record_hash = record_hash
        block = self.chain.add_record(
            self.hospital_id,
            record_hash,
            metadata={
                "n_records": len(self.client.data),
                "n_features": self.client.n_features,
                "description": "EHR dataset shard — off-chain; only hash stored here",
                "signer": self.hospital_id,
            },
            signer=self.keys,
        )
        self.sync_replica()
        return block, record_hash

    def sync_replica(self) -> dict:
        """Pull signed blocks from the shared authority into the local replica."""
        ledger = self._authority_ledger()
        if ledger is None:
            return {
                "hospital_id": self.hospital_id,
                "last_sync_ok": False,
                "note": "replica sync requires InProcessChain or Fabric mock ledger",
            }
        return self.replica.sync_from(ledger)

    def use_record_ref(self, record_hash: str) -> bool:
        """
        Consent gate: Hospital B may only use Hospital A's record ref if
        grant_access was logged on-chain for this requester + hash.
        """
        if not self.chain.has_access(self.hospital_id, record_hash):
            raise ConsentError(
                f"{self.hospital_id} has no grant_access for record {record_hash[:16]}..."
            )
        return True

    def participate_in_round(self, round_num: int, global_weights: dict | None):
        """Train locally, audit participation + update hash on-chain."""
        local_weights, n_samples, loss = self.client.train(global_weights)
        update_hash = hash_weights(local_weights)

        self.chain.log_event(
            "fl_round_participation",
            self.hospital_id,
            details={
                "round": round_num,
                "n_samples": n_samples,
                "local_loss": round(loss, 4),
                "update_hash": update_hash,
            },
            signer=self.keys,
        )
        # Explicit update-hash commit (Step 2) — not only global model hash
        self.chain.log_event(
            "fl_update_hash",
            self.hospital_id,
            details={
                "round": round_num,
                "update_hash": update_hash,
                "n_samples": n_samples,
            },
            signer=self.keys,
        )
        self.sync_replica()
        return local_weights, n_samples, loss, update_hash
