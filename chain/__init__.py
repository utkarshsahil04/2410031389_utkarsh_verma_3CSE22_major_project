"""
Blockchain backends for the medical-record + FL prototype.

Usage:
    from chain import get_chain
    chain = get_chain()                      # in-process (default)
    chain = get_chain("fabric")              # Fabric mock (or sdk stubs)
    # or: set CHAIN_BACKEND=inprocess|fabric
"""
from __future__ import annotations

import os

from chain.crypto import NodeKeyPair
from chain.fabric import FabricGateway
from chain.inprocess import Blockchain, InProcessChain
from chain.interface import ChainBackend
from chain.replica import ChainReplica

__all__ = [
    "ChainBackend",
    "InProcessChain",
    "Blockchain",
    "FabricGateway",
    "ChainReplica",
    "NodeKeyPair",
    "get_chain",
]


def get_chain(backend: str | None = None, **kwargs) -> ChainBackend:
    """
    Factory for ledger backends.

    backend: "inprocess" (default) | "fabric"
    Falls back to env CHAIN_BACKEND when backend is None.
    """
    name = (backend or os.environ.get("CHAIN_BACKEND") or "inprocess").strip().lower()
    if name in ("inprocess", "local", "sim", "simulation"):
        return InProcessChain(**kwargs) if kwargs else InProcessChain()
    if name in ("fabric", "hyperledger"):
        return FabricGateway(**kwargs)
    raise ValueError(
        f"Unknown chain backend '{name}'. Use 'inprocess' or 'fabric'."
    )
