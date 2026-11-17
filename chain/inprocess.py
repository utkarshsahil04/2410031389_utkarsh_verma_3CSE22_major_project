"""
In-process permissioned chain — default backend for local demos.

Blocks are Ed25519-signed by the acting node. Peers keep local replicas
(see chain.replica) and reject unsigned / invalid peer blocks.

ponytail: Single-process append-only list, not a distributed network.
          Upgrade path: FabricGateway implementing the same ChainBackend API.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from chain.crypto import MEMBER_IDS, NodeKeyPair, member_keys, member_public_keys
from chain.interface import ChainBackend


def _canonical_payload(
    index: int, timestamp: float, data: dict, previous_hash: str, signer_id: str, public_key: str
) -> bytes:
    # signer_id and public_key are signed too, so identity cannot be relabelled
    return json.dumps(
        {
            "index": index,
            "timestamp": timestamp,
            "data": data,
            "previous_hash": previous_hash,
            "signer_id": signer_id,
            "public_key": public_key,
        },
        sort_keys=True,
    ).encode()


GENESIS_DATA = {"event": "genesis", "description": "Blockchain initialized"}
GENESIS_TS = 0.0  # fixed so the genesis hash is a constant every node can recompute


def _genesis() -> "Block":
    return Block(0, dict(GENESIS_DATA), "0" * 64, timestamp=GENESIS_TS)


class Block:
    def __init__(
        self,
        index: int,
        data: dict,
        previous_hash: str,
        *,
        signer_id: str = "genesis",
        public_key: str = "",
        signature: str = "",
        timestamp: float | None = None,
    ):
        self.index = index
        self.timestamp = time.time() if timestamp is None else timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.signer_id = signer_id
        self.public_key = public_key
        self.signature = signature
        self.hash = self._compute_hash()

    def payload_bytes(self) -> bytes:
        return _canonical_payload(
            self.index, self.timestamp, self.data, self.previous_hash,
            self.signer_id, self.public_key,
        )

    def _compute_hash(self) -> str:
        # Include signature material so tampering with signer fields fails verify_chain
        content = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "data": self.data,
                "previous_hash": self.previous_hash,
                "signer_id": self.signer_id,
                "public_key": self.public_key,
                "signature": self.signature,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def sign_with(self, keys: NodeKeyPair) -> None:
        self.signer_id = keys.node_id
        self.public_key = keys.public_bytes_hex()
        self.signature = keys.sign(self.payload_bytes())
        self.hash = self._compute_hash()

    def signature_valid(self) -> bool:
        if self.index == 0:
            return True  # genesis is unsigned by convention
        if not self.public_key or not self.signature:
            return False
        return NodeKeyPair.verify(self.public_key, self.payload_bytes(), self.signature)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "signer_id": self.signer_id,
            "public_key": self.public_key[:16] + "..." if self.public_key else "",
            "signature": self.signature[:16] + "..." if self.signature else "",
            "signature_valid": self.signature_valid(),
            # full values so chain_state.json can be reloaded and re-verified
            "public_key_full": self.public_key,
            "signature_full": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        blk = cls(
            d["index"],
            d["data"],
            d["previous_hash"],
            signer_id=d.get("signer_id", ""),
            public_key=d.get("public_key", ""),
            signature=d.get("signature", ""),
            timestamp=d.get("timestamp"),
        )
        # Restore full keys if truncated in dashboard dump — callers should use
        # internal sync from live Block objects, not truncated to_dict().
        if "public_key_full" in d:
            blk.public_key = d["public_key_full"]
        if "signature_full" in d:
            blk.signature = d["signature_full"]
        blk.hash = d.get("hash", blk._compute_hash())
        return blk

    def to_sync_dict(self) -> dict:
        """Full serialization for peer sync (not truncated)."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
            "signer_id": self.signer_id,
            "public_key": self.public_key,
            "signature": self.signature,
        }


class InProcessChain(ChainBackend):
    def __init__(self, signer: NodeKeyPair | None = None, seed: str | None = None):
        # ponytail: `seed` is ignored; membership is always the fixed demo allowlist.
        self.members: dict[str, str] = member_public_keys()
        self.signer = signer or member_keys()["coordinator"]
        self.chain: list[Block] = [_genesis()]

    @classmethod
    def from_blocks(cls, blocks: list[dict], seed: str | None = None) -> "InProcessChain":
        """Rebuild a chain from to_dict()/export_blocks() output (needs full keys to verify)."""
        c = cls()
        c.chain = [Block.from_dict(b) for b in blocks]
        return c

    def _last(self) -> Block:
        return self.chain[-1]

    def _default_signer(self, actor: str) -> NodeKeyPair:
        # ponytail: the in-process demo can derive every member's key, so callers that pass
        #           no signer are signed as the acting member. Real nodes hold only their own.
        return member_keys().get(actor) or self.signer

    def _add(self, data: dict, signer: NodeKeyPair | None = None) -> Block:
        keys = signer or self.signer
        block = Block(len(self.chain), data, self._last().hash)
        block.sign_with(keys)
        self.chain.append(block)
        return block

    def add_record(
        self,
        hospital_id: str,
        record_hash: str,
        metadata: dict | None = None,
        *,
        signer: NodeKeyPair | None = None,
    ) -> Block:
        return self._add(
            {
                "event": "register_record",
                "hospital": hospital_id,
                "record_hash": record_hash,
                "metadata": metadata or {},
            },
            signer or self._default_signer(hospital_id),
        )

    # ponytail: O(n) scans over the chain; fine at demo size, index it if chains grow.
    def _owns(self, hospital: str, record_hash: str, upto: int | None = None) -> bool:
        """True iff `hospital` itself signed a register_record/data_update for the hash."""
        for blk in self.chain[: upto if upto is not None else len(self.chain)]:
            d = blk.data
            if (
                d.get("event") in ("register_record", "data_update")
                and d.get("hospital") == hospital
                and d.get("record_hash") == record_hash
                and blk.signer_id == hospital
            ):
                return True
        return False

    def grant_access(
        self,
        granting_hospital: str,
        requesting_hospital: str,
        record_hash: str,
        *,
        signer: NodeKeyPair | None = None,
    ) -> Block:
        keys = signer or self._default_signer(granting_hospital)
        if keys.node_id != granting_hospital:
            raise PermissionError(
                f"grant_access must be signed by the granter {granting_hospital}, not {keys.node_id}"
            )
        if not self._owns(granting_hospital, record_hash):
            raise PermissionError(
                f"{granting_hospital} has not registered record {record_hash[:16]}... and cannot grant it"
            )
        return self._add(
            {
                "event": "grant_access",
                "from": granting_hospital,
                "to": requesting_hospital,
                "record_hash": record_hash,
            },
            keys,
        )

    def revoke_access(
        self,
        granting_hospital: str,
        requesting_hospital: str,
        record_hash: str,
        *,
        signer: NodeKeyPair | None = None,
    ) -> Block:
        keys = signer or self._default_signer(granting_hospital)
        if keys.node_id != granting_hospital:
            raise PermissionError(
                f"revoke_access must be signed by the owner {granting_hospital}, not {keys.node_id}"
            )
        if not self._owns(granting_hospital, record_hash):
            raise PermissionError(
                f"{granting_hospital} has not registered record {record_hash[:16]}... and cannot revoke it"
            )
        if not self.has_access(requesting_hospital, record_hash):
            raise PermissionError(f"{requesting_hospital} has no active grant to revoke")
        return self._add(
            {
                "event": "revoke_access",
                "from": granting_hospital,
                "to": requesting_hospital,
                "record_hash": record_hash,
            },
            keys,
        )

    def has_access(self, requesting_hospital: str, record_hash: str) -> bool:
        """Derived from the verified chain: the owner's latest signed grant or revoke wins."""
        if not self.verify_chain():
            return False
        allowed = False
        for i, blk in enumerate(self.chain):
            d = blk.data
            if (
                d.get("event") in ("grant_access", "revoke_access")
                and d.get("to") == requesting_hospital
                and d.get("record_hash") == record_hash
                and d.get("from") == blk.signer_id
                and self._owns(blk.signer_id, record_hash, upto=i)
            ):
                allowed = d["event"] == "grant_access"
        return allowed

    def log_event(
        self,
        event_type: str,
        actor: str,
        details: dict | None = None,
        *,
        signer: NodeKeyPair | None = None,
    ) -> Block:
        return self._add(
            {
                "event": event_type,
                "actor": actor,
                "details": details or {},
            },
            signer or self._default_signer(actor),
        )

    def _block_ok(self, block: Block, position: int, prev: Block) -> bool:
        """Shared rule set for verify_chain and accept_peer_block (never raises)."""
        try:
            return (
                block.index == position
                and block.previous_hash == prev.hash
                and block.hash == block._compute_hash()
                and bool(block.public_key)
                and self.members.get(block.signer_id) == block.public_key
                and block.signature_valid()
            )
        except Exception:
            return False

    def accept_peer_block(self, block: Block) -> bool:
        """Validate and append a block from a peer. Unknown signers are rejected."""
        try:
            if not self._block_ok(block, len(self.chain), self._last()):
                return False
            d = block.data
            if d.get("event") in ("grant_access", "revoke_access") and not all(
                k in d for k in ("from", "to", "record_hash")
            ):
                return False
        except Exception:
            return False
        self.chain.append(block)
        return True

    def verify_chain(self) -> bool:
        # ponytail: truncation (dropping the last blocks) is undetectable without an
        #           external anchor, e.g. a head hash published outside this ledger.
        if not self.chain:
            return False
        g = self.chain[0]
        if g.hash != _genesis().hash or g.hash != g._compute_hash():
            return False
        return all(
            self._block_ok(self.chain[i], i, self.chain[i - 1])
            for i in range(1, len(self.chain))
        )

    def to_dict(self) -> list[dict]:
        return [b.to_dict() for b in self.chain]

    def export_blocks(self) -> list[dict]:
        return [b.to_sync_dict() for b in self.chain]

    def sync_status(self) -> dict:
        signed = sum(1 for b in self.chain if b.index > 0 and b.signature_valid())
        return {
            "backend": "InProcessChain",
            "length": len(self.chain),
            "signed_blocks": signed,
            "signer": self.signer.node_id,
            "known_nodes": list(MEMBER_IDS),
            "chain_valid": self.verify_chain(),
        }

    def __len__(self) -> int:
        return len(self.chain)


# Backward-compatible name used by older imports / docs
Blockchain = InProcessChain
