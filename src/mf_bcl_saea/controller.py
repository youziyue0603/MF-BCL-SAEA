"""Bayesian operator controller, prioritized replay and EWC regularization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .schema import AblationFlags, PaperProtocol


@dataclass
class OperatorAction:
    crossover_probability: float
    crossover_eta: float
    mutation_probability: float
    mutation_eta: float
    sampled: bool


class _ControllerNet(nn.Module):
    def __init__(self, state_dimension: int, hidden: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = state_dimension
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(previous, 4)
        self.log_std = nn.Linear(previous, 4)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.trunk(state)
        return self.mean(latent), torch.clamp(self.log_std(latent), -4.0, 1.0)


@dataclass
class ReplayItem:
    state: np.ndarray
    reward: float
    priority: float


class BayesianOperatorController:
    """Controller with printed architecture/ranges and explicit practical choices.

    The manuscript identifies dual mean/std branches and a reparameterized policy
    but omits its complete loss/advantage implementation. This implements a
    documented REINFORCE realization and must not be represented as recovered
    author source code.
    """

    state_dimension = 25

    def __init__(self, protocol: PaperProtocol = PaperProtocol(), flags: AblationFlags = AblationFlags(), seed: int = 0) -> None:
        self.protocol, self.flags, self.seed = protocol, flags, seed
        torch.manual_seed(seed)
        self.model = _ControllerNet(self.state_dimension, protocol.controller_hidden_layers, protocol.controller_dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=protocol.controller_learning_rate)
        self.replay: list[ReplayItem] = []
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}
        self.pending: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        self.rng = np.random.default_rng(seed)

    def default_action(self) -> OperatorAction:
        return OperatorAction(.7, 17.5, .125, 30.0, False)

    def sample(self, state: np.ndarray) -> OperatorAction:
        state_t = torch.tensor(np.asarray(state, dtype=np.float32)[None, :])
        mean, log_std = self.model(state_t)
        if not self.flags.stochastic_controller or self.rng.random() < self.protocol.random_default_operator_probability:
            return self.default_action()
        distribution = torch.distributions.Normal(mean, torch.exp(log_std))
        raw = distribution.rsample()
        log_probability = distribution.log_prob(raw).sum(dim=1)
        self.pending.append((state_t.detach(), raw.detach(), float(log_probability.detach().item())))
        value = torch.sigmoid(raw)[0].detach().numpy()
        action = self._map_action(value, sampled=True)
        return action

    def _map_action(self, unit: np.ndarray, sampled: bool) -> OperatorAction:
        p_c_min, p_c_max = self.protocol.crossover_probability_range
        eta_c_min, eta_c_max = self.protocol.crossover_distribution_index_range
        p_m_min, p_m_max = self.protocol.mutation_probability_range
        eta_m_min, eta_m_max = self.protocol.mutation_distribution_index_range
        return OperatorAction(
            p_c_min + unit[0] * (p_c_max - p_c_min), eta_c_min + unit[1] * (eta_c_max - eta_c_min),
            p_m_min + unit[2] * (p_m_max - p_m_min), eta_m_min + unit[3] * (eta_m_max - eta_m_min), sampled,
        )

    def observe_reward(self, state: np.ndarray, reward: float, phase_sensitivity: float, hv_contribution: float, uncertainty: float) -> None:
        priority = (
            self.protocol.replay_phase_weight * max(phase_sensitivity, 0.0)
            + self.protocol.replay_hv_weight * max(hv_contribution, 0.0)
            + self.protocol.replay_uncertainty_weight * max(uncertainty, 0.0)
        )
        self.replay.append(ReplayItem(np.asarray(state, dtype=np.float32), float(reward), float(priority)))
        if len(self.replay) > self.protocol.replay_capacity:
            self.replay.sort(key=lambda item: item.priority, reverse=True)
            self.replay = self.replay[: self.protocol.replay_capacity]

    def snapshot_fisher(self) -> None:
        if not self.flags.ewc:
            return
        self.model.eval()
        self.fisher = {name: torch.zeros_like(parameter) for name, parameter in self.model.named_parameters()}
        if not self.replay:
            self.anchor = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
            return
        samples = self.replay[-min(len(self.replay), self.protocol.recent_high_fidelity_samples):]
        for item in samples:
            self.model.zero_grad(set_to_none=True)
            mean, _ = self.model(torch.tensor(item.state[None, :]))
            loss = mean.square().mean()
            loss.backward()
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    self.fisher[name] += parameter.grad.detach().square() / len(samples)
        self.anchor = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}

    def update(self) -> float | None:
        if not self.pending and not self.replay:
            return None
        self.model.train()
        rewards = np.array([item.reward for item in self.replay[-self.protocol.recent_high_fidelity_samples:]], dtype=float)
        baseline = float(rewards.mean()) if len(rewards) else 0.0
        losses: list[torch.Tensor] = []
        for state, raw, _ in self.pending:
            mean, log_std = self.model(state)
            distribution = torch.distributions.Normal(mean, torch.exp(log_std))
            log_probability = distribution.log_prob(raw).sum()
            reward = baseline if len(rewards) else 0.0
            losses.append(-log_probability * reward)
        if not losses and self.replay:
            # Replay keeps the controller trainable at later task transitions.
            for item in self.replay[-min(16, len(self.replay)):]:
                mean, _ = self.model(torch.tensor(item.state[None, :]))
                losses.append(-mean.mean() * (item.reward - baseline))
        loss = torch.stack(losses).mean()
        if self.flags.ewc and self.fisher and self.anchor:
            ewc = torch.zeros(())
            for name, parameter in self.model.named_parameters():
                ewc += torch.sum(self.fisher[name] * (parameter - self.anchor[name]).square())
            loss = loss + self.protocol.ewc_lambda * torch.clamp(ewc, max=self.protocol.ewc_per_sample_cap)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.protocol.gradient_clip_norm)
        self.optimizer.step()
        self.pending.clear()
        return float(loss.detach().item())
