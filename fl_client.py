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
from sklearn.preprocessing import StandardScaler


class HospitalClient:
    def __init__(self, hospital_id: str, data: pd.DataFrame):
        self.hospital_id = hospital_id
        self.data = data

        # Fit scaler on local shard only -- mimics real FL (no shared normalisation)
        self.scaler = StandardScaler()
        X_raw = data.iloc[:, :-1].values
        self.X = self.scaler.fit_transform(X_raw)
        self.y = data.iloc[:, -1].values.astype(int)
        self.n_features = self.X.shape[1]

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
