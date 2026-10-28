"""
Hyperledger Fabric gateway — mock mode for demos + hooks for a real SDK.

Default behaviour (no live Fabric network required):
  FABRIC_MODE=mock (default) delegates to an internal InProcessChain while
  exposing Fabric-like submit/evaluate method names and connection stubs.

Real network (Roadmap Step 6 follow-up):
  Set FABRIC_MODE=sdk and fill in connect() with the Hyperledger Fabric
  Gateway SDK (gateway.connect → network → contract).

Env:
  CHAIN_BACKEND=fabric
  FABRIC_MODE=mock|sdk   (default mock)
  FABRIC_CHANNEL, FABRIC_CHAINCODE, FABRIC_PEER_ENDPOINT
"""
from __future__ import annotations

import os
from typing import Any

from chain.interface import ChainBackend
from chain.inprocess import InProcessChain

_SDK_MSG = (
    "FabricGateway real-SDK mode is not wired yet. "
    "Use FABRIC_MODE=mock (default) for demos, or implement connect() with "
    "the Hyperledger Fabric Gateway Python SDK."
)


class FabricGateway(ChainBackend):
    """
    Fabric-shaped ChainBackend.

    mock  — in-memory ledger with submit/evaluate semantics (demo-safe)
    sdk   — raises until real peer identity / TLS / gateway are configured
    """

    def __init__(
        self,
        channel: str | None = None,
        chaincode: str | None = None,
        peer_endpoint: str | None = None,
        mode: str | None = None,
    ):
        self.channel = channel or os.environ.get("FABRIC_CHANNEL", "mychannel")
        self.chaincode = chaincode or os.environ.get(
            "FABRIC_CHAINCODE", "medical_audit"
        )
        self.peer_endpoint = peer_endpoint or os.environ.get("FABRIC_PEER_ENDPOINT")
        self.mode = (
            mode or os.environ.get("FABRIC_MODE") or "mock"
        ).strip().lower()

        # Placeholder for real gateway / network / contract handles
        self._gateway = None
        self._network = None
        self._contract = None

        # Mock ledger mirrors Fabric "world state + audit blocks"
        self._mock: InProcessChain | None = None
        if self.mode in ("mock", "sim", "simulation", "local"):
            self._mock = InProcessChain(seed="fabric-mock")
        elif self.mode not in ("sdk", "real", "live"):
            raise ValueError(
                f"Unknown FABRIC_MODE '{self.mode}'. Use 'mock' or 'sdk'."
            )

    # ── connection hooks (real Fabric) ───────────────────────────────────────

    def connect(self) -> None:
        """
        Open Fabric Gateway connection (identity, TLS, peer).

        TODO (real network):
          from hyperledger.fabric_gateway import connect, Gateway
          self._gateway = connect(
              peer_endpoint=self.peer_endpoint,
              identity=...,
              signer=...,
              tls_root_cert=...,
          )
          self._network = self._gateway.get_network(self.channel)
          self._contract = self._network.get_contract(self.chaincode)
        """
        if self.mode in ("mock", "sim", "simulation", "local"):
            return  # nothing to connect
        raise NotImplementedError(_SDK_MSG)

    def _require_mock(self) -> InProcessChain:
        if self._mock is None:
            raise NotImplementedError(_SDK_MSG)
        return self._mock

    def _submit(self, fn_name: str, *args: Any) -> Any:
        """Simulate Fabric contract.SubmitTransaction(fn_name, ...)."""
        ledger = self._require_mock()
        if fn_name == "RegisterRecord":
            hospital_id, record_hash, metadata = args
            return ledger.add_record(hospital_id, record_hash, metadata)
        if fn_name == "GrantAccess":
            granting, requesting, record_hash = args
            return ledger.grant_access(granting, requesting, record_hash)
        if fn_name == "LogEvent":
            event_type, actor, details = args
            return ledger.log_event(event_type, actor, details)
        raise ValueError(f"Unknown Fabric submit fn: {fn_name}")

    def _evaluate(self, fn_name: str, *args: Any) -> Any:
        """Simulate Fabric contract.EvaluateTransaction(fn_name, ...)."""
        ledger = self._require_mock()
        if fn_name == "HasAccess":
            return ledger.has_access(args[0], args[1])
        if fn_name == "GetChainInfo":
            return {"length": len(ledger), "valid": ledger.verify_chain()}
        if fn_name == "GetAuditTrail":
            return ledger.to_dict()
        if fn_name == "VerifyChain":
            return ledger.verify_chain()
        raise ValueError(f"Unknown Fabric evaluate fn: {fn_name}")

    # ── ChainBackend API ─────────────────────────────────────────────────────

    def add_record(
        self, hospital_id: str, record_hash: str, metadata: dict | None = None
    ) -> Any:
        return self._submit("RegisterRecord", hospital_id, record_hash, metadata or {})

    def grant_access(
        self, granting_hospital: str, requesting_hospital: str, record_hash: str
    ) -> Any:
        return self._submit(
            "GrantAccess", granting_hospital, requesting_hospital, record_hash
        )

    def has_access(self, requesting_hospital: str, record_hash: str) -> bool:
        return bool(self._evaluate("HasAccess", requesting_hospital, record_hash))

    def log_event(
        self, event_type: str, actor: str, details: dict | None = None
    ) -> Any:
        return self._submit("LogEvent", event_type, actor, details or {})

    def verify_chain(self) -> bool:
        return bool(self._evaluate("VerifyChain"))

    def to_dict(self) -> list[dict]:
        return list(self._evaluate("GetAuditTrail"))

    def sync_status(self) -> dict:
        ledger = self._require_mock()
        base = ledger.sync_status()
        base.update(
            {
                "backend": "FabricGateway",
                "mode": self.mode,
                "channel": self.channel,
                "chaincode": self.chaincode,
                "peer_endpoint": self.peer_endpoint,
            }
        )
        return base

    def __len__(self) -> int:
        return len(self._require_mock())

    # Expose mock ledger for replica sync in demos
    @property
    def mock_ledger(self) -> InProcessChain | None:
        return self._mock
