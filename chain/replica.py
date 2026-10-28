"""
Per-hospital local chain replica.

Each node syncs from the authoritative InProcessChain and rejects peer blocks
that are unsigned, have invalid Ed25519 signatures, or break hash links.
"""
from __future__ import annotations

from chain.crypto import NodeKeyPair
from chain.inprocess import Block, InProcessChain


class ChainReplica:
    """Local copy of the ledger kept by one hospital node."""

    def __init__(self, hospital_id: str, keys: NodeKeyPair | None = None):
        self.hospital_id = hospital_id
        self.keys = keys or NodeKeyPair.generate(hospital_id, seed="demo-seed")
        self.local = InProcessChain(signer=self.keys)
        # Drop genesis-only start; we will rebuild from authority on first sync
        self.last_sync_ok = False
        self.rejected_blocks = 0
        self.synced_height = 0

    def sync_from(self, authority: InProcessChain) -> dict:
        """
        Pull blocks from the shared authority chain into the local replica.
        Returns a sync-status dict for the dashboard / fl_state.json.
        """
        # Register known keys from authority
        for nid, pk in authority._known_keys.items():
            self.local.register_node_key(nid, pk)

        # Rebuild local chain from authority export (simplest correct sync)
        exported = authority.export_blocks()
        rebuilt = InProcessChain(signer=self.keys)
        rebuilt.chain = []
        rebuilt._access_grants = set()
        rebuilt._known_keys = dict(authority._known_keys)

        for i, raw in enumerate(exported):
            blk = Block(
                raw["index"],
                raw["data"],
                raw["previous_hash"],
                signer_id=raw.get("signer_id", ""),
                public_key=raw.get("public_key", ""),
                signature=raw.get("signature", ""),
                timestamp=raw.get("timestamp"),
            )
            # Hash must match authority (includes signature fields)
            if blk.hash != raw["hash"]:
                # Recompute after fields set — authority hash is source of truth check
                blk.hash = blk._compute_hash()
                if blk.hash != raw["hash"]:
                    self.rejected_blocks += 1
                    self.last_sync_ok = False
                    return self.status()

            if i == 0:
                rebuilt.chain = [blk]
                continue

            if not rebuilt.accept_peer_block(blk):
                self.rejected_blocks += 1
                self.last_sync_ok = False
                return self.status()

        self.local = rebuilt
        self.synced_height = len(self.local)
        self.last_sync_ok = self.local.verify_chain() and self.synced_height == len(
            authority
        )
        return self.status()

    def try_accept_block(self, block: Block) -> bool:
        ok = self.local.accept_peer_block(block)
        if not ok:
            self.rejected_blocks += 1
        else:
            self.synced_height = len(self.local)
        self.last_sync_ok = ok and self.local.verify_chain()
        return ok

    def status(self) -> dict:
        return {
            "hospital_id": self.hospital_id,
            "synced_height": self.synced_height,
            "local_length": len(self.local),
            "last_sync_ok": self.last_sync_ok,
            "rejected_blocks": self.rejected_blocks,
            "chain_valid": self.local.verify_chain() if self.local.chain else False,
            "public_key": self.keys.public_bytes_hex()[:16] + "...",
        }
