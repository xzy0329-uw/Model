#!/usr/bin/env python3
"""Run pCa-force curves and export simulation data, Hill fits, and SEM to Excel."""

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


# Simulation protocol
PCA_VALUES = np.array([9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 4.5], dtype=float)
REPLICATES = 100
DURATION_MS = 1000
DT_MS = 1
STEADY_WINDOW_MS = 600
NROWS = 4
NCOLS = 4


def lattice_from_z(z_line_nm, z0=1000.0, d0=14.0, nu=0.5):
    """Experimental d1,0 relation used for the two sarcomere lengths."""
    return d0 * (z0 / z_line_nm) ** nu


CONDITIONS = {
    "SL 2.0": {"sl_um": 2.0, "z_line_nm": 1000.0},
    "SL 2.3": {"sl_um": 2.3, "z_line_nm": 1150.0},
}


# Keep this block identical to the parameter block in the pCa experiment.
# Do not include z_line or lattice_spacing; those are condition-specific.
MODEL_PARAMETERS = {
    "tm_k_12": 15000.0,
    "tm_k_23": 0.40,
    "tm_k_34": 0.20,
    "tm_k_41": 0.5,
    "tm_K1": 15000.0,
    "tm_K2": 40.0,
    "tm_K3": 0.05,
    "tm_coop_magnitude": 2.0,
    "tm_span_base": 62.0,
    "tm_span_force50": -8.0,
    "tm_span_steep": 0.8,
    "xb_r12_coeff": 250.0,
    "xb_r23_coeff": 0.60,
    "xb_r34_coeff": 0.15,
    "xb_r45_coeff": 0.60,
    "xb_r51": 0.10,
    "xb_r15": 0.01,
    "xb_srx_k0": 0.003,
    "xb_r16": 0.010,
    "xb_lda_enabled": 1.0,
    "xb_lda_gain": 5.0,
    "xb_lda_lattice_gain": 0.5,
    "xb_lattice_reference": 14.0,
    "xb_lattice_binding_beta": 1.0,
}


def hill_pca(pca, fmin, fmax, pca50, hill_coefficient):
    return fmin + (fmax - fmin) / (
        1.0 + 10.0 ** (hill_coefficient * (pca - pca50))
    )


def sem(values, axis=0):
    values = np.asarray(values)
    return values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def fit_one_replicate(active_force):
    p0 = [0.0, max(float(np.max(active_force)), 1.0), 5.7, 1.3]
    bounds = ([0.0, 0.0, 4.5, 0.1], [np.inf, np.inf, 7.5, 8.0])
    return curve_fit(
        hill_pca,
        PCA_VALUES,
        active_force,
        p0=p0,
        bounds=bounds,
        maxfev=100000,
    )[0]


def run_condition(topology, label, condition):
    z_line = condition["z_line_nm"]
    d10 = lattice_from_z(z_line)
    print(f"Running {label}: z_line={z_line:.1f} nm, d1,0={d10:.3f} nm")

    result = run(
        topology,
        pCa=PCA_VALUES.tolist(),
        z_line=z_line,
        lattice_spacing=d10,
        duration_ms=DURATION_MS,
        dt=DT_MS,
        replicates=REPLICATES,
        dynamic_params=MODEL_PARAMETERS,
    )

    # Shape: (n_pCa, n_replicates, n_timepoints)
    steady_raw_force = np.asarray(result.axial_force)[..., -STEADY_WINDOW_MS:].mean(axis=-1)
    baseline_index = int(np.where(np.isclose(PCA_VALUES, 9.0))[0][0])
    baseline_per_replicate = steady_raw_force[baseline_index, :]
    active_force = steady_raw_force - baseline_per_replicate[None, :]

    records = []
    for pca_index, pca in enumerate(PCA_VALUES):
        for replicate in range(REPLICATES):
            records.append({
                "condition": label,
                "SL_um": condition["sl_um"],
                "z_line_nm": z_line,
                "lattice_spacing_nm": d10,
                "pCa": pca,
                "replicate": replicate + 1,
                "raw_force_pN": steady_raw_force[pca_index, replicate],
                "baseline_force_pN": baseline_per_replicate[replicate],
                "active_force_pN": active_force[pca_index, replicate],
            })

    fit_records = []
    for replicate in range(REPLICATES):
        try:
            fmin, fmax, pca50, hill = fit_one_replicate(active_force[:, replicate])
        except (RuntimeError, ValueError):
            continue
        fit_records.append({
            "condition": label,
            "SL_um": condition["sl_um"],
            "z_line_nm": z_line,
            "lattice_spacing_nm": d10,
            "replicate": replicate + 1,
            "Fmin_pN": fmin,
            "Fmax_pN": fmax,
            "pCa50": pca50,
            "hill_coefficient": hill,
        })

    active_mean = active_force.mean(axis=1)
    active_sem = sem(active_force, axis=1)
    raw_mean = steady_raw_force.mean(axis=1)
    raw_sem = sem(steady_raw_force, axis=1)
    curve_summary = pd.DataFrame({
        "condition": label,
        "SL_um": condition["sl_um"],
        "z_line_nm": z_line,
        "lattice_spacing_nm": d10,
        "pCa": PCA_VALUES,
        "raw_force_mean_pN": raw_mean,
        "raw_force_sem_pN": raw_sem,
        "active_force_mean_pN": active_mean,
        "active_force_sem_pN": active_sem,
        "n_replicates": REPLICATES,
    })

    return pd.DataFrame(records), pd.DataFrame(fit_records), curve_summary


def main():
    static, dynamic = get_cardiac_params()
    dynamic = dynamic.copy(**MODEL_PARAMETERS)
    topology = SarcTopology.create(
        nrows=NROWS,
        ncols=NCOLS,
        static_params=static,
        dynamic_params=dynamic,
    )
    topology = jax.device_put(topology)

    print(f"Topology: {topology.n_thick} thick, {topology.n_thin} thin")
    print(f"Crowns: {topology.n_crowns}/thick, Sites: {topology.n_sites}/thin")
    print(f"Total XBs: {topology.total_xbs}")

    replicate_frames = []
    fit_frames = []
    curve_frames = []
    for label, condition in CONDITIONS.items():
        replicate_frame, fit_frame, curve_frame = run_condition(topology, label, condition)
        replicate_frames.append(replicate_frame)
        fit_frames.append(fit_frame)
        curve_frames.append(curve_frame)

    replicate_data = pd.concat(replicate_frames, ignore_index=True)
    fit_replicates = pd.concat(fit_frames, ignore_index=True)
    curve_summary = pd.concat(curve_frames, ignore_index=True)

    fit_summary = (
        fit_replicates.groupby(["condition", "SL_um", "z_line_nm", "lattice_spacing_nm"], as_index=False)
        .agg(
            n_successful_fits=("replicate", "count"),
            Fmax_mean_pN=("Fmax_pN", "mean"),
            Fmax_sem_pN=("Fmax_pN", sem),
            pCa50_mean=("pCa50", "mean"),
            pCa50_sem=("pCa50", sem),
            hill_coefficient_mean=("hill_coefficient", "mean"),
            hill_coefficient_sem=("hill_coefficient", sem),
            Fmin_mean_pN=("Fmin_pN", "mean"),
            Fmin_sem_pN=("Fmin_pN", sem),
        )
    )

    print("\nFit summary:")
    print(fit_summary.to_string(index=False))

    output_xlsx = Path("pCa_force_results.xlsx")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        fit_summary.to_excel(writer, sheet_name="fit_summary", index=False)
        curve_summary.to_excel(writer, sheet_name="force_summary", index=False)
        fit_replicates.to_excel(writer, sheet_name="fit_replicates", index=False)
        replicate_data.to_excel(writer, sheet_name="force_replicates", index=False)
        pd.DataFrame(
            [{"parameter": key, "value": value} for key, value in MODEL_PARAMETERS.items()]
        ).to_excel(writer, sheet_name="model_parameters", index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"SL 2.0": "tab:blue", "SL 2.3": "tab:green"}
    pca_smooth = np.linspace(PCA_VALUES.min(), PCA_VALUES.max(), 400)

    for _, row in fit_summary.iterrows():
        label = row["condition"]
        curve = curve_summary[curve_summary["condition"] == label]
        ax.errorbar(
            curve["pCa"],
            curve["active_force_mean_pN"],
            yerr=curve["active_force_sem_pN"],
            fmt="o",
            capsize=3,
            color=colors[label],
            label=f"{label} simulation",
        )
        fit_curve = hill_pca(
            pca_smooth,
            row["Fmin_mean_pN"],
            row["Fmax_mean_pN"],
            row["pCa50_mean"],
            row["hill_coefficient_mean"],
        )
        ax.plot(
            pca_smooth,
            fit_curve,
            color=colors[label],
            linewidth=2,
            label=(
                f"{label} fit: Fmax={row['Fmax_mean_pN']:.1f} pN, "
                f"pCa50={row['pCa50_mean']:.3f}, "
                f"nH={row['hill_coefficient_mean']:.2f}"
            ),
        )

    ax.invert_xaxis()
    ax.set_xlabel("pCa")
    ax.set_ylabel("Active force (pN)")
    ax.set_title("pCa-Force Curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()

    output_png = Path("pCa_force_curve.png")
    fig.savefig(output_png, dpi=200)
    plt.show()

    print(f"\nWrote: {output_xlsx.resolve()}")
    print(f"Wrote: {output_png.resolve()}")


if __name__ == "__main__":
    main()
