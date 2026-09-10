from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MAIN = ROOT / "figures" / "main"
SI = ROOT / "figures" / "supplementary"
for folder in (MAIN, SI):
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)

COLORS = {
    "MF-BCL-SAEA": "#B64342",
    "MF-EMTO": "#484878",
    "qNEHVI": "#6677A8",
    "LC-SAEA": "#8D9BC1",
    "C-SAEA": "#AEB8D3",
    "MOEA/D-GP": "#6F9FA7",
    "SMS-EGO": "#87B5AD",
    "ParEGO": "#B9B2CF",
    "HF-NSGA-II": "#A8A8A8",
}
OURS = "MF-BCL-SAEA"
ALGORITHMS = list(COLORS)


def save_figure(fig: plt.Figure, folder: Path, stem: str, dpi: int = 600) -> None:
    for ax in fig.axes:
        ax.grid(False, which="both", axis="both")
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        folder / f"{stem}.tiff",
        dpi=dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=8,
    )


def make_fig5(summary: pd.DataFrame) -> None:
    order = ALGORITHMS
    s = summary.set_index("algorithm").loc[order].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.20))
    specs = [
        ("HV_mean", "HV_ci95", "Hypervolume ↑"),
        ("IGD_plus_mean", None, "IGD+ ↓"),
        ("HF_calls_mean", "HF_calls_sd", "HF calls to 90% oracle HV ↓"),
        ("feasible_fraction_mean", None, "Feasible fraction ↑"),
    ]
    for ax, (value, err, ylabel), label in zip(axes.flat, specs, "abcd"):
        vals = s[value].to_numpy()
        errors = s[err].to_numpy() if err else None
        x = np.arange(len(order))
        ax.bar(
            x, vals, color=[COLORS[a] for a in order], edgecolor="white", linewidth=0.5
        )
        if errors is not None:
            ax.errorbar(
                x, vals, yerr=errors, fmt="none", ecolor="#333333", capsize=2, lw=0.8
            )
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=5.7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E8E8E8", lw=0.6)
        panel(ax, label)
    fig.suptitle(
        "Main LCE inverse-design benchmark (30 paired seeds; equal HF budget)",
        y=0.925,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.2)
    save_figure(fig, MAIN, "Figure_5_main_algorithm_comparison")


def make_fig6(pareto: pd.DataFrame, candidates: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.20, 4.80))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.55, 1, 1],
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.82,
        wspace=0.42,
        hspace=0.68,
    )
    ax = fig.add_subplot(gs[:, 0])
    for alg in ALGORITHMS:
        g = pareto[pareto.algorithm == alg]
        ax.plot(
            g.outer_stress_MPa,
            g.inner_stress_MPa,
            lw=1.2 if alg == OURS else 0.7,
            alpha=1 if alg == OURS else 0.65,
            color=COLORS[alg],
            label=alg,
        )
    ax.set_xlabel("Outer peak stress (MPa)")
    ax.set_ylabel("Inner peak stress (MPa)")
    ax.grid(color="#ECECEC", lw=0.55)
    ax.legend(fontsize=5.4, ncol=2, loc="upper right")
    panel(ax, "a")

    selected = candidates.iloc[[2, len(candidates) // 2, -3]].copy()
    selected["role"] = ["low outer stress", "balanced", "low inner stress"]
    for j, (_, row) in enumerate(selected.iterrows()):
        axh = fig.add_subplot(gs[j // 2, 1 + j % 2])
        t = np.linspace(0, 2 * np.pi * row.turns, 450)
        radius = row.curvature_per_mm / (
            row.curvature_per_mm**2 + (row.pitch_mm / (2 * np.pi)) ** 2
        )
        x = radius * np.cos(t)
        z = row.pitch_mm * t / (2 * np.pi)
        axh.plot(z, x, color="#B64342", lw=1.5)
        axh.set_aspect("auto")
        axh.set_xlabel("Axial coordinate (mm)", fontsize=6)
        axh.set_ylabel("Radial coordinate (mm)", fontsize=6)
        axh.set_title(
            f"{row.role}\nσout={row.outer_stress_MPa:.2f}, σin={row.inner_stress_MPa:.2f} MPa",
            fontsize=6.2,
        )
        axh.grid(color="#EFEFEF", lw=0.5)
        panel(axh, chr(ord("b") + j))
    ax_blank = fig.add_subplot(gs[1, 2])
    ax_blank.axis("off")
    ax_blank.text(
        0.02,
        0.88,
        "Feasibility gates",
        fontweight="bold",
        fontsize=7,
        transform=ax_blank.transAxes,
    )
    gates = [
        r"curvature: 0.75–0.90 mm$^{-1}$",
        "pitch: 2.4–3.0 mm",
        "turns: 5.0–7.2",
        "helix fit R² ≥ 0.85",
        "shape error ≤ 0.10",
    ]
    ax_blank.text(
        0.02,
        0.67,
        "\n".join(gates),
        fontsize=6.3,
        linespacing=1.55,
        va="top",
        transform=ax_blank.transAxes,
    )
    panel(ax_blank, "e")
    fig.suptitle(
        "Pareto quality and representative feasible helical morphologies",
        y=0.91,
        fontsize=8,
    )
    save_figure(fig, MAIN, "Figure_6_pareto_and_helical_designs")


def make_fig7(convergence: pd.DataFrame, main_runs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.10))
    for alg in ALGORITHMS:
        g = convergence[convergence.algorithm == alg].groupby("HF_calls")
        x = np.array(sorted(g.groups))
        hv = g.HV.mean().reindex(x).to_numpy()
        hv_sd = g.HV.std().reindex(x).to_numpy()
        axes[0, 0].plot(
            x, hv, color=COLORS[alg], lw=1.5 if alg == OURS else 0.8, label=alg
        )
        if alg in (OURS, "MF-EMTO", "qNEHVI"):
            axes[0, 0].fill_between(
                x, hv - hv_sd, hv + hv_sd, color=COLORS[alg], alpha=0.10
            )
        igd = g.IGD_plus.mean().reindex(x).to_numpy()
        axes[0, 1].plot(x, igd, color=COLORS[alg], lw=1.5 if alg == OURS else 0.8)
    axes[0, 0].set(xlabel="High-fidelity calls", ylabel="HV ↑")
    axes[0, 1].set(xlabel="High-fidelity calls", ylabel="IGD+ ↓")
    axes[0, 0].legend(fontsize=5.2, ncol=3)
    for ax in axes[0]:
        ax.grid(color="#ECECEC", lw=0.55)

    selected = [OURS, "MF-EMTO", "qNEHVI", "LC-SAEA"]
    for alg in selected:
        vals = np.sort(
            main_runs[main_runs.algorithm == alg].HF_calls_to_90pct_oracle_HV.to_numpy()
        )
        y = np.arange(1, len(vals) + 1) / len(vals)
        axes[1, 0].step(vals, y, where="post", color=COLORS[alg], lw=1.4, label=alg)
    axes[1, 0].set(xlabel="HF calls to 90% oracle HV", ylabel="Empirical CDF")
    axes[1, 0].legend(fontsize=5.6)
    axes[1, 0].grid(color="#ECECEC", lw=0.55)

    budget_summary = (
        convergence.groupby(["algorithm", "HF_calls"]).HV.mean().reset_index()
    )
    for alg in selected:
        g = budget_summary[budget_summary.algorithm == alg]
        efficiency = (g.HV - 0.60) / g.HF_calls
        axes[1, 1].plot(g.HF_calls, efficiency, color=COLORS[alg], lw=1.4, label=alg)
    axes[1, 1].set(xlabel="High-fidelity calls", ylabel="(HV − 0.60) / HF call")
    axes[1, 1].grid(color="#ECECEC", lw=0.55)
    for ax, label in zip(axes.flat, "abcd"):
        panel(ax, label)
    fig.suptitle(
        "Budget-dependent convergence and sample efficiency", y=0.925, fontsize=8
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.3)
    save_figure(fig, MAIN, "Figure_7_convergence_and_budget_efficiency")


def make_fig8(ablation: pd.DataFrame, factorial: pd.DataFrame) -> None:
    summary = (
        ablation.groupby("variant")
        .agg(
            HV=("HV", "mean"),
            HV_sd=("HV", "std"),
            calls=("HF_calls_to_90pct_oracle_HV", "mean"),
        )
        .reset_index()
    )
    order = summary.sort_values("HV", ascending=True)
    fig, axes = plt.subplots(
        1, 3, figsize=(7.20, 3.25), gridspec_kw={"width_ratios": [1.25, 1, 1]}
    )
    y = np.arange(len(order))
    axes[0].errorbar(
        order.HV,
        y,
        xerr=order.HV_sd,
        fmt="o",
        color="#B64342",
        ecolor="#A0A0A0",
        capsize=2,
    )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order.variant, fontsize=5.7)
    axes[0].set_xlabel("HV ↑")
    axes[0].grid(axis="x", color="#ECECEC", lw=0.55)

    axes[1].barh(
        y,
        order.calls,
        color=["#B64342" if v.startswith("Full") else "#B8C3D9" for v in order.variant],
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("HF calls to target ↓")
    axes[1].grid(axis="x", color="#ECECEC", lw=0.55)

    matrix = np.zeros((4, 4))
    module_names = [
        "physics",
        "selective_upgrade",
        "bayesian_controller",
        "continual_learning",
    ]
    full_mean = factorial[(factorial[module_names] == 1).all(axis=1)].HV.mean()
    for i, m1 in enumerate(module_names):
        for j, m2 in enumerate(module_names):
            if i == j:
                subset = factorial[factorial[m1] == 0]
            else:
                subset = factorial[(factorial[m1] == 0) & (factorial[m2] == 0)]
            matrix[i, j] = full_mean - subset.HV.mean()
    im = axes[2].imshow(matrix, cmap="Reds", vmin=0, vmax=matrix.max())
    axes[2].set_xticks(range(4))
    axes[2].set_yticks(range(4))
    short = ["Phys.", "Upgrade", "Ctrl.", "CL"]
    axes[2].set_xticklabels(short, rotation=35, ha="right", fontsize=5.7)
    axes[2].set_yticklabels(short, fontsize=5.7)
    for i in range(4):
        for j in range(4):
            axes[2].text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=5.3,
                color="white" if matrix[i, j] > matrix.max() * 0.55 else "black",
            )
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label("HV loss", fontsize=6)
    for ax, label in zip(axes, "abc"):
        panel(ax, label)
    fig.suptitle("Ablation evidence and component interactions", y=0.925, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.2)
    save_figure(fig, MAIN, "Figure_8_ablation_and_interactions")


def make_fig9(pred: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.10))
    ours = pred[pred.variant == "MF-BCL-SAEA ensemble"]
    for zone, color, label in [
        (False, "#6677A8", "non-transition"),
        (True, "#B64342", "phase-transition"),
    ]:
        g = ours[ours.phase_zone == zone]
        axes[0, 0].scatter(
            g.outer_true_MPa,
            g.outer_pred_MPa,
            s=7,
            alpha=0.45,
            color=color,
            label=label,
        )
        axes[0, 1].scatter(
            g.inner_true_MPa,
            g.inner_pred_MPa,
            s=7,
            alpha=0.45,
            color=color,
            label=label,
        )
    for ax, xlabel, ylabel in [
        (axes[0, 0], "HF outer stress (MPa)", "Predicted outer stress (MPa)"),
        (axes[0, 1], "HF inner stress (MPa)", "Predicted inner stress (MPa)"),
    ]:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], "--", color="#666666", lw=0.8)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=5.7)
        ax.grid(color="#ECECEC", lw=0.5)

    phase_metrics = metrics[metrics.zone == "phase-transition"].copy()
    x = np.arange(len(phase_metrics))
    axes[1, 0].bar(
        x - 0.17,
        phase_metrics.outer_RMSE_MPa,
        width=0.34,
        color="#B64342",
        label="outer",
    )
    axes[1, 0].bar(
        x + 0.17,
        phase_metrics.inner_RMSE_MPa,
        width=0.34,
        color="#6677A8",
        label="inner",
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(
        ["Full", "No physics", "Single fidelity", "Single net"],
        rotation=30,
        ha="right",
        fontsize=5.6,
    )
    axes[1, 0].set_ylabel("Phase-zone RMSE (MPa) ↓")
    axes[1, 0].legend(fontsize=5.8)
    axes[1, 0].grid(axis="y", color="#ECECEC", lw=0.5)

    nominal = np.linspace(0.5, 0.99, 10)
    for variant, color in [
        ("MF-BCL-SAEA ensemble", "#B64342"),
        ("ensemble without physics", "#6677A8"),
        ("single deterministic network", "#A8A8A8"),
    ]:
        g = pred[(pred.variant == variant) & pred.phase_zone]
        z = np.abs(g.outer_pred_MPa - g.outer_true_MPa) / g.outer_sigma_MPa
        empirical = [np.mean(z <= q) for q in np.linspace(0.67, 2.58, len(nominal))]
        axes[1, 1].plot(
            nominal, empirical, "o-", ms=3, lw=1.1, color=color, label=variant
        )
    axes[1, 1].plot([0.5, 1], [0.5, 1], "--", color="#666666", lw=0.8)
    axes[1, 1].set(xlabel="Nominal coverage", ylabel="Empirical coverage")
    axes[1, 1].legend(fontsize=5.1)
    axes[1, 1].grid(color="#ECECEC", lw=0.5)
    for ax, label in zip(axes.flat, "abcd"):
        panel(ax, label)
    fig.suptitle(
        "Surrogate accuracy and uncertainty calibration near the phase transition",
        y=0.925,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.3)
    save_figure(fig, MAIN, "Figure_9_surrogate_calibration")


def make_fig10(analytic: pd.DataFrame, engineering: pd.DataFrame) -> None:
    task_mean = analytic.groupby(["task", "algorithm"]).normalized_HV.mean().unstack()
    task_mean = task_mean[ALGORITHMS]
    ranks = task_mean.rank(axis=1, ascending=False, method="average")
    fig, axes = plt.subplots(
        1, 3, figsize=(7.20, 3.60), gridspec_kw={"width_ratios": [1.55, 1, 1]}
    )
    im = axes[0].imshow(
        ranks.to_numpy(), aspect="auto", cmap="viridis_r", vmin=1, vmax=len(ALGORITHMS)
    )
    axes[0].set_xticks(range(len(ALGORITHMS)))
    axes[0].set_xticklabels(ALGORITHMS, rotation=45, ha="right", fontsize=5.2)
    axes[0].set_yticks(range(len(ranks)))
    axes[0].set_yticklabels(ranks.index, fontsize=4.9)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02)
    cb.set_label("Rank ↓", fontsize=6)

    avg_rank = ranks.mean().sort_values()
    axes[1].barh(
        np.arange(len(avg_rank)), avg_rank, color=[COLORS[a] for a in avg_rank.index]
    )
    axes[1].set_yticks(np.arange(len(avg_rank)))
    axes[1].set_yticklabels(avg_rank.index, fontsize=5.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean rank across 24 tasks ↓")
    axes[1].grid(axis="x", color="#ECECEC", lw=0.5)

    eng = (
        engineering.groupby(["task", "algorithm"])
        .normalized_HV.mean()
        .unstack()[ALGORITHMS]
    )
    ours = eng[OURS]
    best_baseline = eng.drop(columns=[OURS]).max(axis=1)
    gain = (ours / best_baseline - 1) * 100
    colors = ["#B64342" if v >= 0 else "#6677A8" for v in gain]
    axes[2].barh(np.arange(len(gain)), gain, color=colors)
    axes[2].axvline(0, color="#555555", lw=0.8)
    axes[2].set_yticks(np.arange(len(gain)))
    axes[2].set_yticklabels(gain.index, fontsize=5.5)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("HV gain over strongest baseline (%)")
    axes[2].grid(axis="x", color="#ECECEC", lw=0.5)
    for ax, label in zip(axes, "abc"):
        panel(ax, label)
    fig.suptitle(
        "Generalization across analytical and engineering optimization tasks",
        y=0.925,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.1)
    save_figure(fig, MAIN, "Figure_10_generalization")


def make_fig11(robustness: pd.DataFrame) -> None:
    scenarios = [
        ("objective_noise_pct", "Objective noise (%)"),
        ("missing_fidelity_pairs_pct", "Missing fidelity pairs (%)"),
        ("phase_temperature_shift_C", "Transition shift (°C)"),
        ("failed_HF_evaluations_pct", "Failed HF evaluations (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 4.80))
    for ax, (scenario, xlabel), label in zip(axes.flat, scenarios, "abcd"):
        g = robustness[robustness.scenario == scenario]
        for alg in [OURS, "MF-EMTO", "qNEHVI", "LC-SAEA"]:
            h = (
                g[g.algorithm == alg]
                .groupby("level")
                .normalized_HV.agg(["mean", "std"])
                .reset_index()
            )
            ax.plot(
                h.level,
                h["mean"],
                "o-",
                ms=3,
                lw=1.4 if alg == OURS else 1.0,
                color=COLORS[alg],
                label=alg,
            )
            if alg in (OURS, "MF-EMTO"):
                ax.fill_between(
                    h.level,
                    h["mean"] - h["std"],
                    h["mean"] + h["std"],
                    color=COLORS[alg],
                    alpha=0.10,
                )
        ax.set(xlabel=xlabel, ylabel="Normalized HV ↑")
        ax.grid(color="#ECECEC", lw=0.55)
        panel(ax, label)
    axes[0, 0].legend(fontsize=5.7, ncol=2)
    fig.suptitle(
        "Robustness under corrupted multi-fidelity information and solver failures",
        y=0.925,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.3)
    save_figure(fig, MAIN, "Figure_11_robustness_stress_tests")


def make_fig12(
    mesh: pd.DataFrame, residual: pd.DataFrame, verify: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.00))
    axes[0, 0].plot(
        mesh.mesh_size_mm,
        mesh.outer_peak_stress_MPa,
        "o-",
        color="#B64342",
        label="outer stress",
    )
    axes[0, 0].plot(
        mesh.mesh_size_mm,
        mesh.inner_peak_stress_MPa,
        "s-",
        color="#6677A8",
        label="inner stress",
    )
    axes[0, 0].invert_xaxis()
    axes[0, 0].set(xlabel="Mesh size (mm)", ylabel="Peak stress (MPa)")
    axes[0, 0].legend(fontsize=5.8)
    axes[0, 0].grid(color="#ECECEC", lw=0.5)

    axes[0, 1].scatter(
        residual.LF_outer_MPa,
        residual.HF_outer_MPa,
        s=8,
        alpha=0.45,
        color="#A8A8A8",
        label="LF",
    )
    axes[0, 1].scatter(
        residual.MF_outer_MPa,
        residual.HF_outer_MPa,
        s=8,
        alpha=0.45,
        color="#6677A8",
        label="MF",
    )
    lim = [0.25, 1.05]
    axes[0, 1].plot(lim, lim, "--", color="#555555", lw=0.8)
    axes[0, 1].set(
        xlabel="Lower-fidelity outer stress (MPa)",
        ylabel="HF outer stress (MPa)",
        xlim=lim,
        ylim=lim,
    )
    axes[0, 1].legend(fontsize=5.8)
    axes[0, 1].grid(color="#ECECEC", lw=0.5)

    axes[1, 0].barh(
        np.arange(len(verify)), verify.relative_deviation_pct, color="#8D9BC1"
    )
    axes[1, 0].axvline(5, color="#B64342", linestyle="--", lw=0.9, label="5% tolerance")
    axes[1, 0].set_yticks(np.arange(len(verify)))
    axes[1, 0].set_yticklabels(verify.metric, fontsize=5.6)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("2D–3D relative deviation (%)")
    axes[1, 0].legend(fontsize=5.8)
    axes[1, 0].grid(axis="x", color="#ECECEC", lw=0.5)

    axes[1, 1].plot(mesh.element_count, mesh.runtime_min, "o-", color="#6F9FA7")
    axes[1, 1].set(xlabel="Element count", ylabel="Runtime (min)")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(color="#ECECEC", lw=0.5, which="both")
    for ax, label in zip(axes.flat, "abcd"):
        panel(ax, label)
    fig.suptitle(
        "Numerical fidelity, residual structure and model-reduction accuracy",
        y=0.925,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.3)
    save_figure(fig, MAIN, "Figure_12_numerical_validation")


def make_fig13(
    thermal: pd.DataFrame,
    device: pd.DataFrame,
    cycles_df: pd.DataFrame,
    parity: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.20, 5.00))
    th = (
        thermal.groupby("temperature_C")
        .agg(
            DSC=("DSC_heat_flow_arb", "mean"),
            E=("DMA_storage_modulus_MPa", "mean"),
            contraction=("free_axial_contraction_fraction", "mean"),
        )
        .reset_index()
    )
    axes[0, 0].plot(th.temperature_C, th.DSC, color="#B64342", lw=1.4)
    axes[0, 0].axvspan(82, 97, color="#F6CFCB", alpha=0.45)
    axes[0, 0].set(xlabel="Temperature (°C)", ylabel="DSC heat flow (a.u.)")
    axes[0, 0].grid(color="#ECECEC", lw=0.5)

    ax2 = axes[0, 1]
    ax2.plot(th.temperature_C, th.E, color="#6677A8", lw=1.4, label="storage modulus")
    ax2.set_yscale("log")
    ax2.set(xlabel="Temperature (°C)", ylabel="Storage modulus (MPa)")
    ax2.grid(color="#ECECEC", lw=0.5, which="both")
    twin = ax2.twinx()
    twin.plot(
        th.temperature_C,
        100 * th.contraction,
        color="#B64342",
        lw=1.1,
        label="contraction",
    )
    twin.set_ylabel("Free contraction (%)", color="#B64342")
    twin.spines["right"].set_visible(True)

    d = (
        device.groupby(["design", "temperature_C"])
        .agg(curv=("curvature_per_mm", "mean"), force=("blocking_force_mN", "mean"))
        .reset_index()
    )
    for design, color in [
        ("Baseline", "#A8A8A8"),
        ("Balanced inverse design", "#B64342"),
        ("High-force inverse design", "#6677A8"),
    ]:
        g = d[d.design == design]
        axes[0, 2].plot(g.temperature_C, g.curv, lw=1.3, color=color, label=design)
    axes[0, 2].set(xlabel="Temperature (°C)", ylabel=r"Helical curvature (mm$^{-1}$)")
    axes[0, 2].legend(fontsize=5.0)
    axes[0, 2].grid(color="#ECECEC", lw=0.5)

    for design, color in [
        ("Baseline", "#A8A8A8"),
        ("Balanced inverse design", "#B64342"),
        ("High-force inverse design", "#6677A8"),
    ]:
        g = d[d.design == design]
        axes[1, 0].plot(g.temperature_C, g.force, lw=1.3, color=color, label=design)
    axes[1, 0].set(xlabel="Temperature (°C)", ylabel="Blocking force (mN)")
    axes[1, 0].grid(color="#ECECEC", lw=0.5)

    cyc = (
        cycles_df.groupby("cycle")
        .agg(
            curv=("curvature_retention", "mean"),
            force=("force_retention", "mean"),
            curv_sd=("curvature_retention", "std"),
        )
        .reset_index()
    )
    axes[1, 1].plot(
        cyc.cycle, 100 * cyc.curv, "o-", ms=2.5, color="#B64342", label="curvature"
    )
    axes[1, 1].plot(
        cyc.cycle, 100 * cyc.force, "s-", ms=2.5, color="#6677A8", label="force"
    )
    axes[1, 1].set(xlabel="Thermal cycle", ylabel="Retention (%)")
    axes[1, 1].legend(fontsize=5.8)
    axes[1, 1].grid(color="#ECECEC", lw=0.5)

    grouped = (
        parity.groupby("design_id")
        .agg(
            pred=("predicted_curvature_per_mm", "mean"),
            obs=("observed_curvature_per_mm", "mean"),
            obs_sd=("observed_curvature_per_mm", "std"),
        )
        .reset_index()
    )
    axes[1, 2].errorbar(
        grouped.pred,
        grouped.obs,
        yerr=grouped.obs_sd,
        fmt="o",
        ms=3,
        color="#B64342",
        ecolor="#A8A8A8",
        capsize=2,
    )
    lim = [0.73, 0.92]
    axes[1, 2].plot(lim, lim, "--", color="#555555", lw=0.8)
    axes[1, 2].set(
        xlabel=r"Predicted curvature (mm$^{-1}$)",
        ylabel=r"Observed curvature (mm$^{-1}$)",
        xlim=lim,
        ylim=lim,
    )
    axes[1, 2].grid(color="#ECECEC", lw=0.5)
    for ax, label in zip(axes.flat, "abcdef"):
        panel(ax, label)
    fig.suptitle("Material and actuator validation", y=0.925, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.25)
    save_figure(fig, MAIN, "Figure_13_material_and_device_validation")


def make_si_figures(
    analytic: pd.DataFrame,
    hyper: pd.DataFrame,
    pairwise: pd.DataFrame,
    candidates: pd.DataFrame,
    parity: pd.DataFrame,
) -> None:
    task_mean = (
        analytic.groupby(["task", "algorithm"])
        .normalized_HV.mean()
        .unstack()[ALGORITHMS]
    )
    fig, ax = plt.subplots(figsize=(7.20, 4.00))
    improvement = (
        task_mean.sub(task_mean.drop(columns=[OURS]).max(axis=1), axis=0)[OURS] * 100
    )
    ax.bar(
        np.arange(len(improvement)),
        improvement,
        color=["#B64342" if x >= 0 else "#6677A8" for x in improvement],
    )
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(np.arange(len(improvement)))
    ax.set_xticklabels(improvement.index, rotation=50, ha="right", fontsize=5.5)
    ax.set_ylabel("Ours − strongest baseline (HV points)")
    ax.grid(axis="y", color="#ECECEC", lw=0.5)
    panel(ax, "a")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, SI, "Figure_S1_taskwise_HV_gain")

    fig, axes = plt.subplots(2, 2, figsize=(7.20, 4.80))
    for ax, (param, g), label in zip(
        axes.flat, hyper.groupby("parameter", sort=False), "abcd"
    ):
        s = g.groupby("level").HV.agg(["mean", "std"]).reset_index()
        ax.errorbar(
            s.level,
            s["mean"],
            yerr=s["std"],
            fmt="o-",
            color="#B64342",
            capsize=2,
            lw=1.2,
        )
        if param == "EWC_lambda":
            ax.set_xscale("symlog", linthresh=10)
        ax.set(xlabel=param.replace("_", " "), ylabel="HV")
        ax.grid(color="#ECECEC", lw=0.5)
        panel(ax, label)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, SI, "Figure_S2_hyperparameter_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.10))
    y = np.arange(len(pairwise))
    labels = pairwise.comparison.str.replace(f"{OURS} vs ", "", regex=False)
    axes[0].barh(y, pairwise.relative_gain_pct, color="#B64342")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=5.7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("HV gain (%)")
    axes[1].barh(y, pairwise.A12, color="#6677A8")
    axes[1].axvline(0.5, color="#555555", linestyle="--", lw=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Vargha–Delaney A12")
    for ax, label in zip(axes, "ab"):
        ax.grid(axis="x", color="#ECECEC", lw=0.5)
        panel(ax, label)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, SI, "Figure_S3_pairwise_effect_sizes")

    vars14 = [
        "RR_um",
        "rr_um",
        "HH_um",
        "LL_mm",
        "alpha_out_1_per_C",
        "k_out_W_mK",
        "rho_out_kg_m3",
        "alpha_in_1_per_C",
        "k_in_W_mK",
        "rho_in_kg_m3",
        "E1_MPa",
        "E2_MPa",
        "Tc_C",
        "Delta_C",
    ]
    norm = candidates[vars14].copy()
    norm = (norm - norm.min()) / (norm.max() - norm.min())
    fig, ax = plt.subplots(figsize=(7.20, 3.80))
    cmap = plt.get_cmap("coolwarm")
    stress_span = candidates.outer_stress_MPa.max() - candidates.outer_stress_MPa.min()
    score = (
        candidates.outer_stress_MPa - candidates.outer_stress_MPa.min()
    ) / stress_span
    for i in range(len(norm)):
        ax.plot(range(14), norm.iloc[i], color=cmap(score.iloc[i]), alpha=0.45, lw=0.8)
    ax.set_xticks(range(14))
    ax.set_xticklabels(vars14, rotation=55, ha="right", fontsize=5.2)
    ax.set_ylabel("Normalized design value")
    ax.grid(color="#ECECEC", lw=0.45)
    panel(ax, "a")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, SI, "Figure_S4_pareto_design_variables")

    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.80))
    pairs = [
        (
            "predicted_curvature_per_mm",
            "observed_curvature_per_mm",
            r"Curvature (mm$^{-1}$)",
        ),
        ("predicted_pitch_mm", "observed_pitch_mm", "Pitch (mm)"),
        (
            "predicted_blocking_force_mN",
            "observed_blocking_force_mN",
            "Blocking force (mN)",
        ),
    ]
    for ax, (pred_col, obs_col, title), label in zip(axes, pairs, "abc"):
        g = (
            parity.groupby("design_id")
            .agg(pred=(pred_col, "mean"), obs=(obs_col, "mean"), sd=(obs_col, "std"))
            .reset_index()
        )
        ax.errorbar(
            g.pred,
            g.obs,
            yerr=g.sd,
            fmt="o",
            ms=3,
            color="#B64342",
            ecolor="#A8A8A8",
            capsize=2,
        )
        lo = min(g.pred.min(), g.obs.min())
        hi = max(g.pred.max(), g.obs.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=0.8)
        ax.set(xlabel=f"Predicted {title}", ylabel=f"Observed {title}")
        ax.grid(color="#ECECEC", lw=0.5)
        panel(ax, label)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, SI, "Figure_S5_inverse_design_parity")


def main() -> None:
    main_runs = pd.read_csv(DATA / "main_comparison_runs.csv")
    main_summary = pd.read_csv(DATA / "main_comparison_summary.csv")
    pairwise = pd.read_csv(DATA / "pairwise_statistics.csv")
    pareto = pd.read_csv(DATA / "pareto_front_points.csv")
    candidates = pd.read_csv(DATA / "pareto_candidate_14D_designs.csv")
    convergence = pd.read_csv(DATA / "budget_convergence_runs.csv")
    ablation = pd.read_csv(DATA / "ablation_runs.csv")
    factorial = pd.read_csv(DATA / "factorial_interaction_runs.csv")
    predictions = pd.read_csv(DATA / "surrogate_holdout_predictions.csv")
    metrics = pd.read_csv(DATA / "surrogate_holdout_metrics.csv")
    analytic = pd.read_csv(DATA / "analytic_benchmark_runs.csv")
    engineering = pd.read_csv(DATA / "engineering_benchmark_runs.csv")
    robustness = pd.read_csv(DATA / "robustness_runs.csv")
    mesh = pd.read_csv(DATA / "mesh_convergence.csv")
    residuals = pd.read_csv(DATA / "cross_fidelity_residuals.csv")
    verification = pd.read_csv(DATA / "axisymmetric_vs_3D.csv")
    thermal = pd.read_csv(DATA / "material_thermal_curves.csv")
    device = pd.read_csv(DATA / "device_temperature_response.csv")
    cycles = pd.read_csv(DATA / "device_cycle_retention.csv")
    parity = pd.read_csv(DATA / "inverse_design_parity.csv")
    sensitivity = pd.read_csv(DATA / "hyperparameter_sensitivity.csv")

    make_fig5(main_summary)
    make_fig6(pareto, candidates)
    make_fig7(convergence, main_runs)
    make_fig8(ablation, factorial)
    make_fig9(predictions, metrics)
    make_fig10(analytic, engineering)
    make_fig11(robustness)
    make_fig12(mesh, residuals, verification)
    make_fig13(thermal, device, cycles, parity)
    make_si_figures(analytic, sensitivity, pairwise, candidates, parity)


if __name__ == "__main__":
    main()
