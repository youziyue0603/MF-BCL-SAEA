from mf_bcl_saea.lce import ReducedEquivalentLCEFixture
from mf_bcl_saea.optimizer import MFBCLSAEA
from mf_bcl_saea.schema import Fidelity, PaperProtocol


def test_optimizer_respects_hf_budget_and_archive_contract() -> None:
    protocol = PaperProtocol().fast()
    fixture = ReducedEquivalentLCEFixture.provisional(protocol=protocol, seed=2)
    result = MFBCLSAEA(fixture, protocol=protocol, seed=2).run()
    assert result.high_fidelity_evaluations <= protocol.high_fidelity_budget
    assert all(item.fidelity is Fidelity.HIGH and item.feasible for item in result.archive)
    assert result.paper_equivalent is False
    assert result.logs
