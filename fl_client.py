"""
Federated Learning hospital client.
Each hospital trains on its own local data shard and returns updated weights.
ponytail: SGDClassifier with partial_fit -- trains in <1s per round, no GPU needed.
          Upgrade path: replace with a PyTorch nn.Module;
          the train() interface (global_weights in -> local_weights out) stays the same.
"""
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier


class HospitalClient:
    def __init__(self, hospital_id: str, data: pd.DataFrame):
        self.hospital_id = hospital_id
        self.data = data

        # Raw features stay local. Scaling uses GLOBAL statistics pooled by the server
        # (local_feature_stats -> server combine_stats -> set_scaler); no local-only scaler.
        self.X_raw = data.iloc[:, :-1].values.astype(float)
        self.X = None
        self.y = data.iloc[:, -1].values.astype(int)
        self.n_features = self.X_raw.shape[1]

    def local_feature_stats(self) -> tuple[int, np.ndarray, np.ndarray]:
        """Sufficient statistics only (n, mean, variance) -- no rows leave the hospital."""
        return len(self.X_raw), self.X_raw.mean(axis=0), self.X_raw.var(axis=0)

    def set_scaler(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Standardise local features with the shared global mean/std."""
        self.X = (self.X_raw - mean) / std

    def get_data_hash(self) -> str:
        """SHA-256 of the local dataset -- anchored on-chain as proof of registration."""
        content = self.data.to_csv(index=False).encode()
        return hashlib.sha256(content).hexdigest()

    def train(self, global_weights: dict | None) -> tuple[dict, int, float]:
        """
        One local training epoch starting from global_weights.
        Uses SGD partial_fit so each FL round is a genuine gradient step on top of
        the current global model -- accuracy visibly improves across rounds.
        Returns (local_weights, n_samples, log_loss).
        ponytail: SGD one epoch per round instead of full convergence.
                  Upgrade path: swap for PyTorch model; same interface.
        """
        assert self.X is not None, "call set_scaler() with the global scaler before train()"
        model = SGDClassifier(
            loss="log_loss", alpha=0.001, max_iter=1, tol=None, random_state=42
        )
        model.classes_ = np.array([0, 1])

        # Inject global weights as the starting point for local training
        if global_weights is not None:
            model.coef_ = global_weights["coef"].copy()
            model.intercept_ = global_weights["intercept"].copy()

        # partial_fit = one pass over local data (one FL round epoch)
        model.partial_fit(self.X, self.y, classes=np.array([0, 1]))

        # Log-loss on local data
        proba = np.clip(model.predict_proba(self.X), 1e-7, 1 - 1e-7)
        loss = float(
            -np.mean(self.y * np.log(proba[:, 1]) + (1 - self.y) * np.log(proba[:, 0]))
        )

        local_weights = {"coef": model.coef_.copy(), "intercept": model.intercept_.copy()}
        return local_weights, len(self.X), loss
