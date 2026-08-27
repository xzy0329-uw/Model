#!/usr/bin/env python3
"""Quick provisional WT/KO pCa curves using guessed titin parameters.

This script deliberately skips PTP optimization.  It is for generating a
preliminary four-curve figure and replicate-level workbook before the slower
PTP fit is run.
"""

import argparse
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


PCA_VALUES = np.array([9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 4.5], dtype=float)
STEADY_WINDOW_MS = 600
DURATION_MS = 1000
DT_MS = 1.0

# Provisional assumptions only. Replace these after fitting actual PTP data.
# KO is initially assumed to have lower passive stiffness than WT.
TITIN_BY_GROUP = {
    "WT": {"titin_a": 55.0, "titin_b": 0.0080, "titin_rest": 140.0},
    "KO": {"titin_a": 43.3, "titin_b": 0.0060, "titin_rest": 140.0},
}

# All non-titin parameters remain fixed.
MODEL_PARAMETERS = {
    "tm_k_12": 15000.0, "tm_k_23": 0.40,
    "tm_k_34": 0.20, "tm_k_41": 0.5,
    "tm_K1": 15000.0, "tm_K2": 40.0, "tm_K3": 0.05,
    "tm_coop_magnitude": 2.0, "tm_span_base": 62.0,
    "tm_span_force50": -8.0, "tm_span_steep": 0.8,
    "xb_r12_coeff": 250.0, "xb_r23_coeff": 0.60,
    "xb_r34_coeff": 0.15, "xb_r45_coeff": 0.60,
    "xb_r51": 0.10, "xb_r15": 0.01,
    "xb_srx_k0": 0.003, "xb_r16": 0.010,
    "xb_lda_enabled": 1.0, "xb_lda_gain": 5.0,
    "xb_lda_lattice_gain": 1.0,
    "xb_lattice_reference": 14.0,
    "xb_lattice_binding_beta": 1.5,
}


def lattice_from_z(z_line_nm, z0=1000.0, d0=14.0, nu=0.5):
    return d0 * (z0 / z_line_nm) ** nu


def hill_pca(pca, fmin, fmax, pca50, hill):
    return fmin + (fmax - fmin) / (1.0 + 10.0 ** (hill * (pca - pca50)))


def sem(values, axis=0):
    values = np.asarray(values)
    return values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def fit_hill(active_force):
    return curve_fit(
        hill_pca,
        PCA_VALUES,
        active_force,
        p0=[0.0, max(float(active_force.max()), 1.0), 5.7, 1.3],
        bounds=([0.0, 0.0, 4.5, 0.1], [np.inf, np.inf, 7.5, 8.0]),
        maxfev=100000,
    )[0]


def simulate_condition(topology, group, sl_um, replicates):
    z_line = sl_um * 500.0
    spacing = lattice_from_z(z_line)
    parameters = {**MODEL_PARAMETERS, **TITIN_BY_GROUP[group]}
    label = f"{group} SL {sl_um:.1f}"
    print(f"Running {label}: d1,0={spacing:.3f} nm")

    result = run(
        topology,
        pCa=PCA_VALUES.tolist(),
        z_line=z_line,
        lattice_spacing=spacing,
        duration_ms=DURATION_MS,
        dt=DT_MS,
        replicates=replicates,
        dynamic_params=parameters,
    )
    raw_force = np.asarray(result.axial_force)[..., -STEADY_WINDOW_MS:].mean(axis=-1)
    baseline = raw_force[0, :]
    active_force = raw_force - baseline[None, :]

    replicate_rows = []
    for i, pca in enumerate(PCA_VALUES):
        for replicate in range(replicates):
            replicate_rows.append({
                "group": group, "SL_um": sl_um, "z_line_nm": z_line,
                "lattice_spacing_nm": spacing, "pCa": pca,
                "replicate": replicate + 1,
                "raw_force_pN": raw_force[i, replicate],
                "baseline_force_pN": baseline[replicate],
                "active_force_pN": active_force[i, replicate],
            })

    fit_rows = []
    for replicate in range(replicates):
        try:
            fmin, fmax, pca50, hill = fit_hill(active_force[:, replicate])
        except (RuntimeError, ValueError):
            continue
        fit_rows.append({
            "group": group, "SL_um": sl_um, "z_line_nm": z_line,
            "lattice_spacing_nm": spacing, "replicate": replicate + 1,
            "Fmin_pN": fmin, "Fmax_pN": fmax,
            "pCa50": pca50, "hill_coefficient": hill,
        })

    summary = pd.DataFrame({
        "group": group, "SL_um": sl_um, "z_line_nm": z_line,
        "lattice_spacing_nm": spacing, "pCa": PCA_VALUES,
        "active_force_mean_pN": active_force.mean(axis=1),
        "active_force_sem_pN": sem(active_force, axis=1),
        "raw_force_mean_pN": raw_force.mean(axis=1),
        "raw_force_sem_pN": sem(raw_force, axis=1),
        "n_replicates": replicates,
    })
    return pd.DataFrame(replicate_rows), pd.DataFrame(fit_rows), summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=32)
    args = parser.parse_args()

    static, dynamic = get_cardiac_params()
    dynamic = dynamic.copy(**MODEL_PARAMETERS)
    topology = jax.device_put(SarcTopology.create(
        nrows=4, ncols=4, static_params=static, dynamic_params=dynamic
    ))
    print(f"Topology: {topology.n_thick} thick, {topology.n_thin} thin")
    print("Using provisional titin parameters; this is not a PTP fit.")

    force_frames, fit_frames, summary_frames = [], [], []
    for group in ("WT", "KO"):
        for sl_um in (2.0, 2.3):
            force, fits, summary = simulate_condition(topology, group, sl_um, args.replicates)
            force_frames.append(force)
            fit_frames.append(fits)
            summary_frames.append(summary)

    force_replicates = pd.concat(force_frames, ignore_index=True)
    fit_replicates = pd.concat(fit_frames, ignore_index=True)
    force_summary = pd.concat(summary_frames, ignore_index=True)
    fit_summary = (
        fit_replicates.groupby(["group", "SL_um", "z_line_nm", "lattice_spacing_nm"], as_index=False)
        .agg(
            n_successful_fits=("replicate", "count"),
            Fmax_mean_pN=("Fmax_pN", "mean"), Fmax_sem_pN=("Fmax_pN", sem),
            pCa50_mean=("pCa50", "mean"), pCa50_sem=("pCa50", sem),
            hill_coefficient_mean=("hill_coefficient", "mean"),
            hill_coefficient_sem=("hill_coefficient", sem),
        )
    )
    print("\nProvisional fit summary:")
    print(fit_summary.to_string(index=False))

    output_xlsx = Path("provisional_WT_KO_titin_pCa_results.xlsx")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        fit_summary.to_excel(writer, sheet_name="fit_summary", index=False)
        force_summary.to_excel(writer, sheet_name="force_summary", index=False)
        fit_replicates.to_excel(writer, sheet_name="fit_replicates", index=False)
        force_replicates.to_excel(writer, sheet_name="force_replicates", index=False)
        pd.DataFrame([
            {"group": group, **parameters}
            for group, parameters in TITIN_BY_GROUP.items()
        ]).to_excel(writer, sheet_name="provisional_titin", index=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {"WT": "tab:blue", "KO": "tab:red"}
    styles = {2.0: "-", 2.3: "--"}
    smooth_pca = np.linspace(PCA_VALUES.min(), PCA_VALUES.max(), 400)
    for _, row in fit_summary.iterrows():
        group, sl_um = row["group"], row["SL_um"]
        curve = force_summary[
            (force_summary["group"] == group) & (force_summary["SL_um"] == sl_um)
        ]
        label = f"{group} SL {sl_um:.1f}"
        ax.errorbar(
            curve["pCa"], curve["active_force_mean_pN"],
            yerr=curve["active_force_sem_pN"], fmt="o", capsize=3,
            color=colors[group], label=f"{label} data",
        )
        ax.plot(
            smooth_pca,
            hill_pca(smooth_pca, 0.0, row["Fmax_mean_pN"], row["pCa50_mean"], row["hill_coefficient_mean"]),
            color=colors[group], linestyle=styles[sl_um], linewidth=2,
            label=f"{label}: Fmax={row['Fmax_mean_pN']:.0f}, pCa50={row['pCa50_mean']:.3f}",
        )

    ax.invert_xaxis()
    ax.set_xlabel("pCa")
    ax.set_ylabel("Active force (pN)")
    ax.set_title("Provisional WT/KO pCa-Force Curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_png = Path("provisional_WT_KO_SL20_SL23_pCa_force.png")
    fig.savefig(output_png, dpi=220)
    plt.show()
    print(f"\nWrote: {output_xlsx.resolve()}")
    print(f"Wrote: {output_png.resolve()}")


if __name__ == "__main__":
    main()
