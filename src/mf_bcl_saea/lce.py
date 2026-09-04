"""LCE evaluator boundary and a deterministic test fixture.

The final manuscript specifies Abaqus 2024/CAX4RT mesh levels but does not make
its input decks, material cards, boundary conditions or constraint thresholds
available. This module consequently separates a registered external evaluator
from a non-evidentiary reduced-equivalent fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .features import normalized_design, phase_transition_descriptors, sigmoid
from .schema import Evaluation, Fidelity, FidelitySpec, LCEBounds, PaperProtocol


class LCEFidelityEvaluator(Protocol):
    bounds: LCEBounds

    def evaluate(self, x: np.ndarray, fidelity: Fidelity) -> Evaluation:
        """Run one design at the requested L/M/H solver level."""


@dataclass
class ExternalAbaqusEvaluator:
    """Production adapter around an archived, registered Abaqus run function.

    `run_case` must launch or query the real archived solver case and return a
    mapping containing `objectives`, `constraints`, and `converged`. This class
    intentionally does not synthesize Abaqus input decks from missing paper
    details.
    """

    bounds: LCEBounds
    run_case: Callable[[np.ndarray, Fidelity, FidelitySpec], dict[str, object]]
    protocol: PaperProtocol = PaperProtocol()

    def evaluate(self, x: np.ndarray, fidelity: Fidelity) -> Evaluation:
        x = np.asarray(x, dtype=float)
        if x.shape != (14,):
            raise ValueError("External evaluator expects a single 14-dimensional design.")
        result = self.run_case(x, fidelity, self.protocol.fidelity[fidelity])
        return Evaluation(
            x=x.copy(),
            objectives=np.asarray(result["objectives"], dtype=float),
            constraints=np.asarray(result["constraints"], dtype=float),
            fidelity=fidelity,
            converged=bool(result["converged"]),
            metadata={"evidence": "registered_external_abaqus", **dict(result.get("metadata", {}))},
        )


@dataclass
class ReducedEquivalentLCEFixture:
    """Deterministic software fixture, explicitly not a FEM or paper evidence."""

    bounds: LCEBounds
    protocol: PaperProtocol = PaperProtocol()
    seed: int = 0

    @classmethod
    def provisional(cls, seed: int = 0, protocol: PaperProtocol | None = None) -> "ReducedEquivalentLCEFixture":
        return cls(bounds=LCEBounds.provisional_fixture(), protocol=protocol or PaperProtocol(), seed=seed)

    def evaluate(self, x: np.ndarray, fidelity: Fidelity) -> Evaluation:
        x = np.asarray(x, dtype=float)
        if x.shape != (14,):
            raise ValueError("Fixture expects a single 14-dimensional design.")
        unit = normalized_design(x, self.bounds.lower, self.bounds.upper)
        descriptors = phase_transition_descriptors(x)[0]
        # Smooth deterministic stand-in with fidelity-specific bias. It provides
        # testable residual structure only; it is not a constitutive model.
        phase = np.exp(-descriptors[0]) * (0.3 + descriptors[1])
        radial_ratio, thickness_ratio, slenderness = descriptors[3:]
        outer_stress = (
            0.25 + 0.55 * (unit[0] - .44) ** 2 + 0.31 * (unit[2] - .36) ** 2
            + 0.16 * (unit[10] - .52) ** 2 + 0.11 * phase + 0.05 * np.sin(4 * unit[3])
        )
        inner_stress = (
            0.22 + 0.45 * (unit[1] - .30) ** 2 + 0.28 * (unit[11] - .58) ** 2
            + 0.07 * radial_ratio + 0.03 * thickness_ratio + 0.02 * slenderness + 0.12 * phase
        )
        hf = np.array([outer_stress, inner_stress], dtype=float)
        if fidelity is Fidelity.LOW:
            objectives = hf + np.array([.055, -.035]) + .04 * np.array([unit[4] - unit[7], unit[5] - unit[8]])
        elif fidelity is Fidelity.MEDIUM:
            objectives = hf + np.array([.020, -.012]) + .015 * np.array([unit[4] - unit[7], unit[5] - unit[8]])
        else:
            objectives = hf

        # Software-only constraint proxy. Actual helix conditions must come from
        # the registered Abaqus/post-processing workflow before research claims.
        annulus = x[0] - x[1] - 0.5 * x[2]
        aspect = x[3] / np.maximum(x[0] / 1000.0, 1e-8)
        transition_gate = sigmoid((x[12] - 15.0) / 5.0) * sigmoid((105.0 - x[12]) / 5.0)
        constraints = np.array([-annulus / 100.0, 0.25 - aspect / 20.0, 0.10 - transition_gate], dtype=float)
        converged = bool(np.all(np.isfinite(objectives)))
        return Evaluation(
            x=x.copy(), objectives=objectives, constraints=constraints,
            fidelity=fidelity, converged=converged,
            metadata={
                "evidence": "non_evidentiary_reduced_equivalent_fixture",
                "fixture_non_evidentiary": True,
                "fidelity_contract": self.protocol.fidelity[fidelity].__dict__,
            },
        )
