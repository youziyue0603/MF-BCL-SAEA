"""Minimization metrics used by the reported experimental protocol."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for index, value in enumerate(values):
        if np.any(np.all(values <= value, axis=1) & np.any(values < value, axis=1)):
            keep[index] = False
    return keep


def hypervolume_2d(front: np.ndarray, reference_point: np.ndarray) -> float:
    """Exact two-objective minimization hypervolume for a supplied reference point."""
    front = np.asarray(front, dtype=float)
    reference_point = np.asarray(reference_point, dtype=float)
    if front.size == 0:
        return 0.0
    valid = front[np.all(front < reference_point, axis=1)]
    if len(valid) == 0:
        return 0.0
    valid = valid[nondominated_mask(valid)]
    valid = valid[np.argsort(valid[:, 0])]
    area = 0.0
    y_previous = reference_point[1]
    for x_value, y_value in valid:
        if y_value < y_previous:
            area += (reference_point[0] - x_value) * (y_previous - y_value)
            y_previous = y_value
    return float(area)


def igd_plus(approximation: np.ndarray, reference_front: np.ndarray) -> float:
    approximation = np.asarray(approximation, dtype=float)
    reference_front = np.asarray(reference_front, dtype=float)
    if len(approximation) == 0:
        return float("inf")
    positive_distances = np.maximum(approximation[None, :, :] - reference_front[:, None, :], 0.0)
    return float(np.mean(np.min(np.linalg.norm(positive_distances, axis=2), axis=1)))


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    result = wilcoxon(np.asarray(a, dtype=float), np.asarray(b, dtype=float), alternative="two-sided")
    return float(result.statistic), float(result.pvalue)
