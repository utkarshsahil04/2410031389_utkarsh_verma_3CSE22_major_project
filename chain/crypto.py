"""
Node signing keys for the permissioned chain.

Uses Ed25519 via the `cryptography` package (stdlib has no signing API).
ponytail: Deterministic seed option keeps demos reproducible across runs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def _seed_to_private(seed_material: bytes) -> Ed25519PrivateKey:
    """Derive a stable 32-byte Ed25519 seed from arbitrary material."""
    seed = hashlib.sha256(seed_material).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)


@dataclass
class NodeKeyPair:
    """Ed25519 identity for a hospital / coordinator node."""

    node_id: str
    private_key: Ed25519PrivateKey
    public_key: Ed25519PublicKey

    @classmethod
    def generate(cls, node_id: str, seed: str | None = None) -> "NodeKeyPair":
        if seed is None:
            priv = Ed25519PrivateKey.generate()
        else:
            priv = _seed_to_private(f"{node_id}:{seed}".encode())
        return cls(node_id=node_id, private_key=priv, public_key=priv.public_key())

    def public_bytes_hex(self) -> str:
        return self.public_key.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()

    def private_bytes_hex(self) -> str:
        return self.private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        ).hex()

    def sign(self, message: bytes) -> str:
        return self.private_key.sign(message).hex()

    @staticmethod
    def verify(public_key_hex: str, message: bytes, signature_hex: str) -> bool:
        try:
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
            pub.verify(bytes.fromhex(signature_hex), message)
            return True
        except (InvalidSignature, ValueError):
            return False
