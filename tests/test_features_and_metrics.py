import numpy as np

from mf_bcl_saea.features import phase_transition_descriptors
from mf_bcl_saea.metrics import hypervolume_2d, igd_plus, nondominated_mask
from mf_bcl_saea.schema import LCEBounds


def test_phase_descriptor_shape_and_heating_path_distance() -> None:
    bounds = LCEBounds.provisional_fixture()
    x = ((bounds.lower + bounds.upper) / 2.0)[None, :]
    x[0, 12] = 80.0
    x[0, 13] = 10.0
    descriptors = phase_transition_descriptors(x, np.array([20.0, 80.0, 140.0]))
    assert descriptors.shape == (1, 6)
    assert descriptors[0, 0] == 0.0


def test_two_objective_metrics() -> None:
    front = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [5.0, 5.0]])
    assert nondominated_mask(front).tolist() == [True, True, True, False]
    assert np.isclose(hypervolume_2d(front, np.array([5.0, 5.0])), 11.0)
    assert np.isclose(igd_plus(np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]])), 0.0)
