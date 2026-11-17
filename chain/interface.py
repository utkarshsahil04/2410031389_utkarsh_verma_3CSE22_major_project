"""
Abstract blockchain backend used by the FL simulation.

Both InProcessChain (default) and FabricGateway (mock / future SDK) implement
this API so hospital_node / run_simulation never import a concrete ledger.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from chain.crypto import NodeKeyPair


class ChainBackend(ABC):
    """Permissioned audit-ledger interface for medical-record + FL events."""

    @abstractmethod
    def add_record(
        self,
        hospital_id: str,
        record_hash: str,
        metadata: dict | None = None,
        *,
        signer: NodeKeyPair | None = None,
    ) -> Any:
        """Anchor a dataset/record hash. Raw rows stay off-chain."""

    @abstractmethod
    def grant_access(
        self,
        granting_hospital: str,
        requesting_hospital: str,
        record_hash: str,
        *,
        signer: NodeKeyPair | None = None,
    ) -> Any:
        """Log a consented cross-hospital access grant."""

    @abstractmethod
    def has_access(self, requesting_hospital: str, record_hash: str) -> bool:
        """True iff a prior grant_access allows requesting_hospital to use record_hash."""

    @abstractmethod
    def log_event(
        self,
        event_type: str,
        actor: str,
        details: dict | None = None,
        *,
        signer: NodeKeyPair | None = None,
    ) -> Any:
        """General audit entry (FL rounds, session start, update hashes, …)."""

    @abstractmethod
    def verify_chain(self) -> bool:
        """Return True iff the ledger passes integrity (+ signature) checks."""

    @abstractmethod
    def to_dict(self) -> list[dict]:
        """Serialize blocks/events for the dashboard (`chain_state.json`)."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of blocks / committed transactions."""

    def sync_status(self) -> dict:
        """Optional dashboard metadata (replicas / signatures). Override when available."""
        return {"backend": type(self).__name__, "length": len(self)}
