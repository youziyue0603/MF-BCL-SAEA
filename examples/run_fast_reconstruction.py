"""Run a software smoke test, not a paper-equivalent experiment."""

from __future__ import annotations

from mf_bcl_saea.lce import ReducedEquivalentLCEFixture
from mf_bcl_saea.optimizer import MFBCLSAEA
from mf_bcl_saea.schema import PaperProtocol


def main() -> None:
    protocol = PaperProtocol().fast()
    evaluator = ReducedEquivalentLCEFixture.provisional(seed=7, protocol=protocol)
    result = MFBCLSAEA(evaluator, protocol=protocol, seed=7).run()
    print("NON-EVIDENTIARY SOFTWARE SMOKE TEST")
    print(f"HF evaluations: {result.high_fidelity_evaluations}")
    print(f"HF-feasible archive size: {len(result.archive)}")
    print(f"Paper-equivalent evidence: {result.paper_equivalent}")
    for limitation in result.limitations:
        print(f"- {limitation}")


if __name__ == "__main__":
    main()
