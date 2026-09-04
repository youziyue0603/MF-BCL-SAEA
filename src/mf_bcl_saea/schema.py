"""Paper-locked contracts and explicit reconstruction assumptions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

import numpy as np


class Fidelity(str, Enum):
    LOW = "L"
    MEDIUM = "M"
    HIGH = "H"


@dataclass(frozen=True)
class FidelitySpec:
    global_element_size_mm: float
    interface_element_size_mm: float
    nominal_runtime_min: float | None


@dataclass(frozen=True)
class PaperProtocol:
    """Values printed in the manuscript, except where clearly labelled otherwise."""

    design_dimension: int = 14
    objective_dimension: int = 2
    population_size: int = 50
    maximum_generations: int = 200
    high_fidelity_budget: int = 200
    high_fidelity_upgrades_per_generation: int = 5
    ensemble_members: int = 5
    ensemble_hidden_layers: tuple[int, int, int] = (256, 128, 64)
    ensemble_dropout: float = 0.10
    ensemble_learning_rate: float = 1e-3
    ensemble_batch_size: int = 32
    ensemble_epochs: int = 50
    crossover_probability_range: tuple[float, float] = (0.5, 0.9)
    crossover_distribution_index_range: tuple[float, float] = (5.0, 30.0)
    mutation_probability_range: tuple[float, float] = (0.05, 0.20)
    mutation_distribution_index_range: tuple[float, float] = (10.0, 50.0)
    controller_hidden_layers: tuple[int, int, int] = (256, 128, 64)
    controller_dropout: float = 0.10
    controller_learning_rate: float = 1e-3
    controller_update_interval_generations: int = 5
    random_default_operator_probability: float = 0.05
    pseudo_task_count: int = 4
    replay_capacity: int = 100
    replay_phase_weight: float = 0.4
    replay_hv_weight: float = 0.3
    replay_uncertainty_weight: float = 0.3
    ewc_lambda: float = 1000.0
    ewc_per_sample_cap: float = 1000.0
    recent_high_fidelity_samples: int = 30
    gradient_clip_norm: float = 1.0
    independent_run_count: int = 15
    update_score_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    fidelity: Mapping[Fidelity, FidelitySpec] = field(
        default_factory=lambda: {
            Fidelity.LOW: FidelitySpec(0.08, 0.04, 4.0),
            Fidelity.MEDIUM: FidelitySpec(0.04, 0.02, None),
            Fidelity.HIGH: FidelitySpec(0.02, 0.01, 37.0),
        }
    )

    def fast(self) -> "PaperProtocol":
        """Small test-only override. It is never a paper-equivalent run."""
        return replace(
            self,
            population_size=12,
            maximum_generations=4,
            high_fidelity_budget=12,
            high_fidelity_upgrades_per_generation=3,
            ensemble_hidden_layers=(32, 16, 8),
            ensemble_epochs=3,
            ensemble_batch_size=8,
            controller_hidden_layers=(32, 16, 8),
            recent_high_fidelity_samples=8,
            replay_capacity=24,
        )


@dataclass(frozen=True)
class LCEBounds:
    """Provisional fixture bounds, not claimed as the manuscript's registered bounds."""

    lower: np.ndarray
    upper: np.ndarray
    provenance: str = "Provisional smoke-test bounds; replace with archived 1.0 bounds."

    def __post_init__(self) -> None:
        if self.lower.shape != (14,) or self.upper.shape != (14,):
            raise ValueError("LCE bounds must contain exactly 14 variables.")
        if np.any(self.upper <= self.lower):
            raise ValueError("Every upper bound must exceed its lower bound.")

    @classmethod
    def provisional_fixture(cls) -> "LCEBounds":
        # Derived only from the explicitly labelled non-evidentiary preview package.
        return cls(
            lower=np.array([10, 10, 50, 1, -.001, .5, 950, -.001, .5, 950, .1, .1, 10, 5.0]),
            upper=np.array([4000, 4000, 500, 100, .001, 2, 1500, .001, 2, 1500, 2000, 2000, 100, 100.0]),
        )


@dataclass(frozen=True)
class Evaluation:
    x: np.ndarray
    objectives: np.ndarray
    constraints: np.ndarray
    fidelity: Fidelity
    converged: bool
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.converged and bool(np.all(self.constraints <= 0.0))


@dataclass(frozen=True)
class AblationFlags:
    phase_conditioned_state: bool = True
    ewc: bool = True
    phase_weighted_upgrade: bool = True
    stochastic_controller: bool = True
    uncertainty: bool = True
    phase_features: bool = True
    continual_controller: bool = True
