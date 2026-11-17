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


def combine_stats(stats: list[tuple[int, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool per-client (n, mean, variance) into the global mean/std of the union of all rows:
    var = sum_i n_i * (var_i + (mean_i - mean)^2) / N   (population variance, as StandardScaler).
    """
    total = sum(n for n, _, _ in stats)
    mean = sum(n * m for n, m, _ in stats) / total
    var = sum(n * (v + (m - mean) ** 2) for n, m, v in stats) / total
    std = np.sqrt(var)
    std[std == 0] = 1.0  # constant feature: avoid divide-by-zero (same as StandardScaler)
    return mean, std


class FedAvgServer:
    def __init__(self, test_data: pd.DataFrame, mean: np.ndarray, std: np.ndarray):
        """mean/std = the global scaler from combine_stats; the test set is only transformed, never fit."""
        self.X_test = (test_data.iloc[:, :-1].values.astype(float) - mean) / std
        self.y_test = test_data.iloc[:, -1].values.astype(int)
        self.global_weights: dict | None = None
        self.round_metrics: list[dict] = []

    def aggregate(self, updates: list[tuple[dict, int, float]]) -> dict:
        """Aggregate client updates. updates = [(weights, n_samples, loss), ...]"""
        self.global_weights = fedavg([(w, n) for w, n, _ in updates])
        return self.global_weights

    def _model(self) -> SGDClassifier:
        model = SGDClassifier(loss="log_loss", random_state=42)
        model.coef_ = self.global_weights["coef"]
        model.intercept_ = self.global_weights["intercept"]
        model.classes_ = np.array([0, 1])
        return model

    def predict_proba(self) -> np.ndarray:
        """P(class 1) of the global model on the held-out test set (for AUC)."""
        return self._model().predict_proba(self.X_test)[:, 1]

    def evaluate(self, round_num: int, updates: list[tuple[dict, int, float]]) -> tuple[float, float]:
        """Evaluate global model on held-out test set. Returns (accuracy, avg_loss)."""
        preds = self._model().predict(self.X_test)
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
