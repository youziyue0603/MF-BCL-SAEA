import numpy as np

from mf_bcl_saea.controller import BayesianOperatorController
from mf_bcl_saea.schema import PaperProtocol


def test_controller_action_stays_in_paper_ranges() -> None:
    controller = BayesianOperatorController(PaperProtocol().fast(), seed=5)
    action = controller.sample(np.zeros(controller.state_dimension, dtype=np.float32))
    assert .5 <= action.crossover_probability <= .9
    assert 5 <= action.crossover_eta <= 30
    assert .05 <= action.mutation_probability <= .2
    assert 10 <= action.mutation_eta <= 50
