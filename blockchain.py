"""
Backward-compatible entry point for the chain package.

Prefer:  from chain import get_chain, ChainBackend
Legacy:  from blockchain import Blockchain
"""
from chain import Blockchain, ChainBackend, FabricGateway, InProcessChain, get_chain

__all__ = [
    "Blockchain",
    "ChainBackend",
    "FabricGateway",
    "InProcessChain",
    "get_chain",
]
