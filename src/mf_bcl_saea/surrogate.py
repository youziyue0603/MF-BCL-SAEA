"""Residual multi-fidelity Deep Ensemble used by the reconstruction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import phase_transition_descriptors
from .schema import Evaluation, Fidelity, LCEBounds, PaperProtocol


class _ResidualNet(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@dataclass
class Prediction:
    mean: np.ndarray
    std: np.ndarray
    source: str


class ResidualMultiFidelityEnsemble:
    """LF base prediction plus adjacent LF->MF and MF->HF residual corrections.

    The exact residual-network implementation is not published. This follows the
    paper's stated adjacent correction topology while exposing a clean component
    boundary rather than claiming source-code identity.
    """

    def __init__(self, bounds: LCEBounds, protocol: PaperProtocol = PaperProtocol(), seed: int = 0) -> None:
        self.bounds = bounds
        self.protocol = protocol
        self.seed = seed
        self.records: dict[bytes, dict[Fidelity, Evaluation]] = defaultdict(dict)
        self.models_lm: list[_ResidualNet] = []
        self.models_mh: list[_ResidualNet] = []
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.fitted = False

    @property
    def input_dimension(self) -> int:
        return 20  # 14 normalized variables plus 6 printed phase descriptors

    def _key(self, x: np.ndarray) -> bytes:
        return np.asarray(x, dtype=np.float64).tobytes()

    def observe(self, evaluation: Evaluation) -> None:
        self.records[self._key(evaluation.x)][evaluation.fidelity] = evaluation

    def _features(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        normalized = (x - self.bounds.lower) / (self.bounds.upper - self.bounds.lower)
        return np.hstack((normalized, phase_transition_descriptors(x)))

    def _paired_residual_data(self, source: Fidelity, target: Fidelity) -> tuple[np.ndarray, np.ndarray]:
        features, residuals = [], []
        for levels in self.records.values():
            if source in levels and target in levels and levels[source].converged and levels[target].converged:
                features.append(self._features(levels[source].x)[0])
                residuals.append(levels[target].objectives - levels[source].objectives)
        if not features:
            return np.empty((0, self.input_dimension)), np.empty((0, 2))
        return np.asarray(features), np.asarray(residuals)

    def fit(self) -> bool:
        x_lm, y_lm = self._paired_residual_data(Fidelity.LOW, Fidelity.MEDIUM)
        x_mh, y_mh = self._paired_residual_data(Fidelity.MEDIUM, Fidelity.HIGH)
        # Adjacent corrections must both exist; otherwise retain direct LF output.
        if min(len(x_lm), len(x_mh)) < 4:
            self.fitted = False
            return False
        all_features = np.vstack((x_lm, x_mh))
        self.feature_mean = all_features.mean(axis=0)
        self.feature_std = np.maximum(all_features.std(axis=0), 1e-8)
        self.models_lm = self._fit_ensemble(x_lm, y_lm, ensemble_offset=0)
        self.models_mh = self._fit_ensemble(x_mh, y_mh, ensemble_offset=10_000)
        self.fitted = True
        return True

    def _fit_ensemble(self, features: np.ndarray, targets: np.ndarray, ensemble_offset: int) -> list[_ResidualNet]:
        standardized = (features - self.feature_mean) / self.feature_std
        x_tensor = torch.tensor(standardized, dtype=torch.float32)
        y_tensor = torch.tensor(targets, dtype=torch.float32)
        models: list[_ResidualNet] = []
        for member in range(self.protocol.ensemble_members):
            torch.manual_seed(self.seed + ensemble_offset + member)
            model = _ResidualNet(self.input_dimension, self.protocol.ensemble_hidden_layers, self.protocol.ensemble_dropout)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.protocol.ensemble_learning_rate)
            member_rng = np.random.default_rng(self.seed + ensemble_offset + member)
            indices = member_rng.integers(0, len(x_tensor), size=len(x_tensor))
            dataset = TensorDataset(x_tensor[indices], y_tensor[indices])
            loader = DataLoader(dataset, batch_size=min(self.protocol.ensemble_batch_size, len(dataset)), shuffle=True)
            model.train()
            for _ in range(self.protocol.ensemble_epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(model(batch_x), batch_y)
                    loss.backward()
                    optimizer.step()
            model.eval()
            models.append(model)
        return models

    def predict(self, x: np.ndarray, lf_objectives: np.ndarray) -> Prediction:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        lf_objectives = np.atleast_2d(np.asarray(lf_objectives, dtype=float))
        if not self.fitted or self.feature_mean is None or self.feature_std is None:
            return Prediction(mean=lf_objectives.copy(), std=np.zeros_like(lf_objectives), source="LF only; residual pairs unavailable")
        features = self._features(x)
        tensor = torch.tensor((features - self.feature_mean) / self.feature_std, dtype=torch.float32)
        member_predictions = []
        with torch.no_grad():
            for lm, mh in zip(self.models_lm, self.models_mh, strict=True):
                member_predictions.append((torch.tensor(lf_objectives, dtype=torch.float32) + lm(tensor) + mh(tensor)).numpy())
        stacked = np.asarray(member_predictions)
        return Prediction(mean=stacked.mean(axis=0), std=stacked.std(axis=0, ddof=0), source="LF plus LM/MH residual Deep Ensemble")
