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

from chain.crypto import NodeKeyPair
from chain.interface import ChainBackend


def _canonical_payload(index: int, timestamp: float, data: dict, previous_hash: str) -> bytes:
    return json.dumps(
        {
            "index": index,
            "timestamp": timestamp,
            "data": data,
            "previous_hash": previous_hash,
        },
        sort_keys=True,
    ).encode()


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
        return _canonical_payload(self.index, self.timestamp, self.data, self.previous_hash)

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
    def __init__(self, signer: NodeKeyPair | None = None, seed: str = "demo-seed"):
        self.signer = signer or NodeKeyPair.generate("coordinator", seed=seed)
        self._known_keys: dict[str, str] = {
            self.signer.node_id: self.signer.public_bytes_hex()
        }
        genesis = Block(
            0, {"event": "genesis", "description": "Blockchain initialized"}, "0" * 64
        )
        self.chain: list[Block] = [genesis]
        self._access_grants: set[tuple[str, str]] = set()  # (to_hospital, record_hash)

    def register_node_key(self, node_id: str, public_key_hex: str) -> None:
        self._known_keys[node_id] = public_key_hex

    def _last(self) -> Block:
        return self.chain[-1]

    def _add(self, data: dict, signer: NodeKeyPair | None = None) -> Block:
        keys = signer or self.signer
        block = Block(len(self.chain), data, self._last().hash)
        block.sign_with(keys)
        self.register_node_key(keys.node_id, keys.public_bytes_hex())
        self.chain.append(block)
        return block

    def add_record(
        self, hospital_id: str, record_hash: str, metadata: dict | None = None
    ) -> Block:
        return self._add(
            {
                "event": "register_record",
                "hospital": hospital_id,
                "record_hash": record_hash,
                "metadata": metadata or {},
            }
        )

    def grant_access(
        self, granting_hospital: str, requesting_hospital: str, record_hash: str
    ) -> Block:
        block = self._add(
            {
                "event": "grant_access",
                "from": granting_hospital,
                "to": requesting_hospital,
                "record_hash": record_hash,
            }
        )
        self._access_grants.add((requesting_hospital, record_hash))
        return block

    def has_access(self, requesting_hospital: str, record_hash: str) -> bool:
        if (requesting_hospital, record_hash) in self._access_grants:
            return True
        # Rebuild from chain for replicas / reloads
        for blk in self.chain:
            d = blk.data
            if (
                d.get("event") == "grant_access"
                and d.get("to") == requesting_hospital
                and d.get("record_hash") == record_hash
            ):
                return True
        return False

    def log_event(
        self, event_type: str, actor: str, details: dict | None = None
    ) -> Block:
        return self._add(
            {
                "event": event_type,
                "actor": actor,
                "details": details or {},
            }
        )

    def accept_peer_block(self, block: Block) -> bool:
        """
        Validate and append a block from a peer (used by replicas).
        Rejects unsigned, bad signatures, or broken hash links.
        """
        if len(self.chain) != block.index:
            return False
        if block.previous_hash != self._last().hash:
            return False
        if not block.signature_valid():
            return False
        # Optional: require known public key for signer
        known = self._known_keys.get(block.signer_id)
        if known and known != block.public_key:
            return False
        if block.hash != block._compute_hash():
            return False
        self.chain.append(block)
        if block.data.get("event") == "grant_access":
            self._access_grants.add((block.data["to"], block.data["record_hash"]))
        self.register_node_key(block.signer_id, block.public_key)
        return True

    def verify_chain(self) -> bool:
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.previous_hash != prev.hash:
                return False
            if curr.hash != curr._compute_hash():
                return False
            if not curr.signature_valid():
                return False
        return True

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
            "known_nodes": list(self._known_keys.keys()),
            "chain_valid": self.verify_chain(),
        }

    def __len__(self) -> int:
        return len(self.chain)


# Backward-compatible name used by older imports / docs
Blockchain = InProcessChain
