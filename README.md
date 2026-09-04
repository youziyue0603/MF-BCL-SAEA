# MF-BCL-SAEA

Code for multi-fidelity Bayesian continual-learning inverse design of LCE fiber actuators.

## Files

- `src/mf_bcl_saea/schema.py`: parameters, fidelity levels, and design bounds.
- `src/mf_bcl_saea/lce.py`: LCE evaluation interface and Abaqus connection.
- `src/mf_bcl_saea/features.py`: phase-transition descriptors.
- `src/mf_bcl_saea/surrogate.py`: multi-fidelity Deep Ensemble.
- `src/mf_bcl_saea/controller.py`: Bayesian controller, replay, and EWC.
- `src/mf_bcl_saea/nsga2.py`: NSGA-II operators.
- `src/mf_bcl_saea/optimizer.py`: main optimization loop and HF archive.
- `src/mf_bcl_saea/metrics.py`: HV, IGD+, RMSE, and Wilcoxon test.
- `configs/paper_contract.yaml`: manuscript parameters.
- `examples/run_fast_reconstruction.py`: quick run example.
- `scripts/audit_conformance.py`: parameter audit.
- `tests/`: unit tests.

## Run

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python .\examples\run_fast_reconstruction.py
```
