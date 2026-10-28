"""
Federated Learning aggregation server.
Implements FedAvg: weighted average of client updates by sample count.
Reference: McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks
           from Decentralized Data.
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


def fedavg(updates: list[tuple[dict, int]]) -> dict:
    """
    Weighted average of model weights by sample count.
    updates: [(weights_dict, n_samples), ...]
    ponytail: Synchronous single-round aggregation, no async/streaming.
              Upgrade path: async aggregation with secure aggregation protocol (Bonawitz 2017).
    """
    total = sum(n for _, n in updates)
    agg_coef = sum(w["coef"] * (n / total) for w, n in updates)
    agg_intercept = sum(w["intercept"] * (n / total) for w, n in updates)
    return {"coef": agg_coef, "intercept": agg_intercept}


class FedAvgServer:
    def __init__(self, test_data: pd.DataFrame):
        scaler = StandardScaler()
        self.X_test = scaler.fit_transform(test_data.iloc[:, :-1].values)
        self.y_test = test_data.iloc[:, -1].values.astype(int)
        self.global_weights: dict | None = None
        self.round_metrics: list[dict] = []

    def aggregate(self, updates: list[tuple[dict, int, float]]) -> dict:
        """Aggregate client updates. updates = [(weights, n_samples, loss), ...]"""
        self.global_weights = fedavg([(w, n) for w, n, _ in updates])
        return self.global_weights

    def evaluate(self, round_num: int, updates: list[tuple[dict, int, float]]) -> tuple[float, float]:
        """Evaluate global model on held-out test set. Returns (accuracy, avg_loss)."""
        model = SGDClassifier(loss="log_loss", random_state=42)
        model.coef_ = self.global_weights["coef"]
        model.intercept_ = self.global_weights["intercept"]
        model.classes_ = np.array([0, 1])

        preds = model.predict(self.X_test)
        accuracy = float(np.mean(preds == self.y_test))
        avg_loss = float(np.mean([loss for _, _, loss in updates]))

        self.round_metrics.append(
            {
                "round": round_num,
                "accuracy": round(accuracy, 4),
                "avg_loss": round(avg_loss, 4),
                "client_losses": {
                    f"hospital_{i + 1}": round(loss, 4)
                    for i, (_, _, loss) in enumerate(updates)
                },
            }
        )
        return accuracy, avg_loss

    def model_hash(self) -> str:
        """SHA-256 of current global weights — anchored on-chain after each round."""
        payload = {
            "coef": self.global_weights["coef"].tolist(),
            "intercept": self.global_weights["intercept"].tolist(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
