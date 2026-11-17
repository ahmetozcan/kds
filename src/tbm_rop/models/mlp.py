"""PyTorch MLP regressor with a scikit-learn-compatible interface.

Replaces the original Keras model. Early stopping monitors an internal
validation split carved from the training data — never the evaluation fold —
to avoid early-stopping leakage.
"""

import logging
from typing import Any

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import register_model

logger = logging.getLogger(__name__)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Feed-forward MLP with dropout, Adam, and early stopping.

    Args:
        hidden_units: Units per hidden layer, e.g. ``(256, 128, 64)``.
        dropout_rate: Dropout applied after each hidden layer.
        learning_rate: Adam learning rate.
        epochs: Maximum training epochs.
        batch_size: Minibatch size.
        patience: Early-stopping patience (epochs without val improvement).
        val_fraction: Fraction of training data held out for early stopping.
        random_state: Seed for weight init and the internal split.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-detect.
    """

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (256, 128, 64),
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        epochs: int = 150,
        batch_size: int = 512,
        patience: int = 15,
        val_fraction: float = 0.15,
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device

    def _build_network(self, n_features: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = n_features
        for units in self.hidden_units:
            layers += [nn.Linear(in_dim, units), nn.ReLU(), nn.Dropout(self.dropout_rate)]
            in_dim = units
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPRegressor":
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X).astype(np.float32)

        # Standardize the target for optimization stability; predictions are
        # mapped back to the original scale in predict().
        self._y_mean = float(y.mean())
        self._y_std = float(y.std() + 1e-12)
        y = (y - self._y_mean) / self._y_std

        n_val = max(1, int(len(X) * self.val_fraction))
        perm = rng.permutation(len(X))
        val_idx, train_idx = perm[:n_val], perm[n_val:]

        X_train = torch.from_numpy(X[train_idx]).to(device)
        y_train = torch.from_numpy(y[train_idx]).to(device)
        X_val = torch.from_numpy(X[val_idx]).to(device)
        y_val = torch.from_numpy(y[val_idx]).to(device)

        self._network = self._build_network(X.shape[1]).to(device)
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_val, best_state, epochs_no_improve = np.inf, None, 0
        for epoch in range(self.epochs):
            self._network.train()
            shuffle = torch.randperm(len(X_train), device=device)
            for start in range(0, len(X_train), self.batch_size):
                batch = shuffle[start : start + self.batch_size]
                optimizer.zero_grad()
                pred = self._network(X_train[batch]).squeeze(-1)
                loss = loss_fn(pred, y_train[batch])
                loss.backward()
                optimizer.step()

            self._network.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._network(X_val).squeeze(-1), y_val).item()

            if val_loss < best_val - 1e-7:
                best_val, epochs_no_improve = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in self._network.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.debug("Early stopping at epoch %d (val MSE %.4f)", epoch, best_val)
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)
        self._device = device
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        self._network.eval()
        with torch.no_grad():
            preds = self._network(torch.from_numpy(X).to(self._device)).squeeze(-1)
        return preds.cpu().numpy() * self._y_std + self._y_mean


@register_model("mlp")
def build_mlp(params: dict[str, Any]) -> TorchMLPRegressor:
    """MLP constructor; translates tuning-style params into estimator args.

    Accepts either ``hidden_units`` directly or the Optuna-style
    ``n_layers`` + ``n_units_l{i}`` convention used by the original pipeline.
    """
    params = dict(params)
    if "n_layers" in params:
        n_layers = params.pop("n_layers")
        units = tuple(params.pop(f"n_units_l{i}") for i in range(n_layers))
        params["hidden_units"] = units
    if "hidden_units" in params and not isinstance(params["hidden_units"], tuple):
        params["hidden_units"] = tuple(params["hidden_units"])
    return TorchMLPRegressor(**params)
