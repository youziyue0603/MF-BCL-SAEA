"""Phase-conditioned descriptors stated in Section 3.2 of the manuscript."""

from __future__ import annotations

import numpy as np


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))


def phase_transition_descriptors(x: np.ndarray, heating_path_c: np.ndarray | None = None) -> np.ndarray:
    """Return the four zero-cost descriptor groups printed for the LCE task.

    The 14-variable ordering is the provisional fixture interface: outer radius,
    inner radius, thickness, length, outer alpha/k/rho, inner alpha/k/rho,
    E1, E2, transition temperature and transition width. Production adapters may
    translate their registered variable ordering before calling this function.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != 14:
        raise ValueError("Expected a 14-dimensional LCE design.")
    if heating_path_c is None:
        heating_path_c = np.linspace(20.0, 140.0, 121)
    heating_path_c = np.asarray(heating_path_c, dtype=float)
    transition_temperature = x[:, 12:13]
    transition_width = np.maximum(x[:, 13:14], 1e-8)
    # Minimum distance on the complete heating path, rather than final T only.
    min_normalized_temperature_distance = np.min(
        np.abs(heating_path_c[None, :, None] - transition_temperature[:, None, :])
        / transition_width[:, None, :],
        axis=1,
    )
    modulus_gradient = np.abs(x[:, 10:11] - x[:, 11:12]) / (np.abs(x[:, 10:11]) + np.abs(x[:, 11:12]) + 1e-8)
    expansion_mismatch = np.abs(x[:, 4:5] - x[:, 7:8])
    outer_radius = np.maximum(x[:, 0:1], 1e-8)
    inner_radius = np.maximum(x[:, 1:2], 1e-8)
    thickness = np.maximum(x[:, 2:3], 1e-8)
    length = np.maximum(x[:, 3:4], 1e-8)
    geometry_ratios = np.hstack((inner_radius / outer_radius, thickness / outer_radius, length / outer_radius))
    return np.hstack((min_normalized_temperature_distance, modulus_gradient, expansion_mismatch, geometry_ratios))


def phase_sensitivity(descriptors: np.ndarray) -> np.ndarray:
    """Operational form of the stated temperature-distance gate times modulus gradient.

    The paper gives the component semantics but no scale constant. The unit gate
    below is deliberately isolated so production runs can register a calibrated
    replacement without changing the optimization pipeline.
    """
    descriptors = np.atleast_2d(np.asarray(descriptors, dtype=float))
    if descriptors.shape[1] < 2:
        raise ValueError("Expected phase descriptors with distance and modulus gradient.")
    return np.exp(-descriptors[:, 0]) * descriptors[:, 1]


def normalized_design(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) - lower) / (upper - lower)
