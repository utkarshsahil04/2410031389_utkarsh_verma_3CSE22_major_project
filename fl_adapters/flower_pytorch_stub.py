"""
Flower + PyTorch adapter STUB (Roadmap Step 6 — optional).

The default demo uses sklearn SGDClassifier via fl_client.HospitalClient.
This module documents how to swap in Flower clients + a tiny PyTorch model
without changing the blockchain / hospital_node orchestration much.

Status: STUB — not imported by run_simulation.py. No Flower/PyTorch dependency
is required for the college prototype demo.

Swap sketch
-----------
1. pip install flwr torch   (optional; not in requirements.txt by default)
2. Implement TorchLogistic with state_dict <-> numpy weight bridge
3. Wrap each HospitalNode shard as a flwr.client.NumPyClient
4. Start flwr.server with FedAvg strategy; after each round call
   chain.log_event("fl_round_complete", ...) with model_hash
5. Keep consent gating + update-hash commits in hospital_node unchanged

Interface preserved today:
    HospitalClient.train(global_weights) -> (local_weights, n_samples, loss)
A Flower NumPyClient would map:
    get_parameters / set_parameters / fit / evaluate
onto the same weight dict {"coef", "intercept"} or a flat numpy vector.
"""
from __future__ import annotations

from typing import Any


class FlowerPyTorchAdapterStub:
    """Placeholder documenting the sklearn → Flower/PyTorch migration."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.backend = "stub"
        self.notes = (
            "Install flwr + torch to implement a real adapter. "
            "Default demo remains sklearn FedAvg."
        )

    def build_model(self) -> Any:
        raise NotImplementedError(
            "Flower/PyTorch adapter is a stub. "
            "See fl_adapters/flower_pytorch_stub.py module docstring."
        )

    def weights_to_numpy(self, weights: dict) -> list:
        """Map HospitalClient weight dict → Flower NumPyClient parameters."""
        import numpy as np

        return [
            np.asarray(weights["coef"], dtype=np.float32),
            np.asarray(weights["intercept"], dtype=np.float32),
        ]

    def numpy_to_weights(self, params: list) -> dict:
        return {"coef": params[0], "intercept": params[1]}


def migration_checklist() -> list[str]:
    return [
        "Keep ChainBackend + HospitalNode.use_record_ref unchanged",
        "Replace HospitalClient.train body with torch training loop",
        "Export coef/intercept (or state_dict hash) for on-chain update_hash",
        "Optional: run flwr.server.start_server alongside run_simulation coordinator",
        "Do not add flwr/torch to default requirements until needed",
    ]
