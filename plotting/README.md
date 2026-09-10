# Figure Reproduction

This directory contains the fixed source data and plotting code for the final
quantitative figures. The plotting script reads the CSV files without changing
their values.

## Files

- `data/main_comparison_*.csv`: run-level and summary method comparison.
- `data/pairwise_statistics.csv`: paired statistical comparisons.
- `data/budget_convergence_runs.csv`: budget-dependent convergence records.
- `data/pareto_*.csv`: Pareto points and selected design variables.
- `data/ablation_runs.csv` and `data/factorial_interaction_runs.csv`: ablation data.
- `data/surrogate_*.csv`: holdout predictions and calibration metrics.
- `data/analytic_benchmark_runs.csv`, `data/engineering_benchmark_runs.csv`, and
  `data/robustness_runs.csv`: generalization and robustness records.
- `data/mesh_convergence.csv`, `data/cross_fidelity_residuals.csv`, and
  `data/axisymmetric_vs_3D.csv`: numerical verification records.
- `data/material_thermal_curves.csv`, `data/device_*.csv`, and
  `data/inverse_design_parity.csv`: material and device validation records.
- `data/hyperparameter_sensitivity.csv`: sensitivity results.
- `scripts/plot_all.py`: reads the fixed data and exports all main and supplementary figures.

## Run

```bash
python plotting/scripts/plot_all.py
```

Figures are written to `plotting/figures` as PDF, SVG, PNG, and TIFF files.
