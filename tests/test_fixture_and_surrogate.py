import numpy as np

from mf_bcl_saea.lce import ReducedEquivalentLCEFixture
from mf_bcl_saea.schema import Fidelity, PaperProtocol
from mf_bcl_saea.surrogate import ResidualMultiFidelityEnsemble


def test_fixture_is_explicitly_non_evidentiary() -> None:
    fixture = ReducedEquivalentLCEFixture.provisional()
    x = (fixture.bounds.lower + fixture.bounds.upper) / 2.0
    result = fixture.evaluate(x, Fidelity.HIGH)
    assert result.metadata["fixture_non_evidentiary"] is True
    assert result.objectives.shape == (2,)
    assert result.constraints.shape == (3,)


def test_residual_ensemble_trains_on_adjacent_pairs() -> None:
    protocol = PaperProtocol().fast()
    fixture = ReducedEquivalentLCEFixture.provisional(protocol=protocol)
    surrogate = ResidualMultiFidelityEnsemble(fixture.bounds, protocol=protocol, seed=3)
    rng = np.random.default_rng(3)
    sample = rng.uniform(fixture.bounds.lower, fixture.bounds.upper, size=(5, 14))
    for x in sample:
        for fidelity in (Fidelity.LOW, Fidelity.MEDIUM, Fidelity.HIGH):
            surrogate.observe(fixture.evaluate(x, fidelity))
    assert surrogate.fit()
    low = fixture.evaluate(sample[0], Fidelity.LOW)
    prediction = surrogate.predict(sample[0], low.objectives)
    assert prediction.mean.shape == (1, 2)
    assert prediction.std.shape == (1, 2)
    assert prediction.source.startswith("LF plus")
