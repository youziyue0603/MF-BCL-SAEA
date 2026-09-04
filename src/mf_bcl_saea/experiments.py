"""Repeated-run utilities; publication statistics require external Abaqus evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .metrics import hypervolume_2d, igd_plus
from .optimizer import OptimizationResult


@dataclass
class RepeatedRunSummary:
    high_fidelity_calls: np.ndarray
    hypervolume: np.ndarray | None
    igd_plus: np.ndarray | None
    paper_equivalent: bool


def run_independent_trials(
    make_optimizer: Callable[[int], object],
    count: int = 15,
    reference_point: np.ndarray | None = None,
    reference_front: np.ndarray | None = None,
) -> tuple[list[OptimizationResult], RepeatedRunSummary]:
    results: list[OptimizationResult] = [make_optimizer(seed).run() for seed in range(count)]
    calls = np.array([result.high_fidelity_evaluations for result in results], dtype=int)
    hv = None if reference_point is None else np.array([hypervolume_2d(result.objective_front, reference_point) for result in results])
    igd = None if reference_front is None else np.array([igd_plus(result.objective_front, reference_front) for result in results])
    return results, RepeatedRunSummary(calls, hv, igd, all(result.paper_equivalent for result in results))
