#!/usr/bin/env python3
"""Scan KO titin_a while keeping all other model parameters fixed.

The scan compares SL 2.0 and SL 2.3 pCa-force curves.  It is intended as a
fast sensitivity analysis before fitting the PTP data.  The output workbook
contains every replicate as well as fitted Hill-curve summaries.
"""

import argparse

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


PCA_VALUES = np.array([9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 4.5])
TITIN_A_VALUES = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 43.3])
SL_VALUES = (2.0, 2.3)
STEADY_WINDOW_MS = 600
DURATION_MS = 1000
DT_MS = 1.0

# These must match the verified pCa model. Only titin_a changes in this scan.
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

# Hold the remaining titin parameters fixed so the scan isolates titin_a.
FIXED_TITIN_PARAMETERS = {"titin_b": 0.0080, "titin_rest": 140.0}


def lattice_from_z(z_line_nm, z0=1000.0, d0=14.0, nu=0.5):
    return d0 * (z0 / z_line_nm) ** nu


def hill_pca(pca, fmin, fmax, pca50, hill):
    return fmin + (fmax - fmin) / (1.0 + 10.0 ** (hill * (pca - pca50)))


def fit_hill(force):
    initial = [0.0, float(np.max(force)), 5.7, 1.3]
    bounds = ([0.0, 0.0, 4.0, 0.1], [np.inf, np.inf, 8.0, 10.0])
    return curve_fit(hill_pca, PCA_VALUES, force, p0=initial, bounds=bounds, maxfev=100000)[0]


def simulate(topology, titin_a, sl_um, replicates):
    z_line = sl_um * 500.0
    lattice_spacing = lattice_from_z(z_line)
    parameters = {
        **MODEL_PARAMETERS,
        **FIXED_TITIN_PARAMETERS,
        "titin_a": float(titin_a),
    }
    print(f"Running titin_a={titin_a:5.1f}, SL {sl_um:.1f}, d1,0={lattice_spacing:.3f} nm")
    result = run(
        topology,
        pCa=PCA_VALUES.tolist(),
        z_line=z_line,
        lattice_spacing=lattice_spacing,
        duration_ms=DURATION_MS,
        dt=DT_MS,
        replicates=replicates,
        dynamic_params=parameters,
    )
    raw = np.asarray(result.axial_force)[..., -STEADY_WINDOW_MS:].mean(axis=-1)
    active = raw - raw[0][None, :]
    return z_line, lattice_spacing, raw, active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=8)
    args = parser.parse_args()

    static, dynamic = get_cardiac_params()
    dynamic = dynamic.copy(**MODEL_PARAMETERS)
    topology = jax.device_put(SarcTopology.create(
        nrows=4, ncols=4, static_params=static, dynamic_params=dynamic
    ))

    force_rows, fit_rows = [], []
    for titin_a in TITIN_A_VALUES:
        for sl_um in SL_VALUES:
            z_line, spacing, raw, active = simulate(topology, titin_a, sl_um, args.replicates)
            for pca_index, pca in enumerate(PCA_VALUES):
                for replicate in range(args.replicates):
                    force_rows.append({
                        "titin_a": titin_a, "SL_um": sl_um, "z_line_nm": z_line,
                        "lattice_spacing_nm": spacing, "pCa": pca,
                        "replicate": replicate + 1, "raw_force_pN": raw[pca_index, replicate],
                        "active_force_pN": active[pca_index, replicate],
                    })
            for replicate in range(args.replicates):
                try:
                    fmin, fmax, pca50, hill = fit_hill(active[:, replicate])
                except (RuntimeError, ValueError):
                    continue
                fit_rows.append({
                    "titin_a": titin_a, "SL_um": sl_um, "replicate": replicate + 1,
                    "Fmin_pN": fmin, "Fmax_pN": fmax,
                    "pCa50": pca50, "hill_coefficient": hill,
                })

    force_frame = pd.DataFrame(force_rows)
    fit_frame = pd.DataFrame(fit_rows)
    summary = fit_frame.groupby(["titin_a", "SL_um"], as_index=False).agg(
        n_successful_fits=("replicate", "count"),
        Fmax_mean_pN=("Fmax_pN", "mean"),
        Fmax_sem_pN=("Fmax_pN", "sem"),
        pCa50_mean=("pCa50", "mean"),
        pCa50_sem=("pCa50", "sem"),
        hill_mean=("hill_coefficient", "mean"),
        hill_sem=("hill_coefficient", "sem"),
    )

    with pd.ExcelWriter("KO_titin_a_sweep.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        fit_frame.to_excel(writer, sheet_name="fit_replicates", index=False)
        force_frame.to_excel(writer, sheet_name="force_replicates", index=False)
        pd.DataFrame([{
            "titin_b_fixed": FIXED_TITIN_PARAMETERS["titin_b"],
            "titin_rest_fixed_nm": FIXED_TITIN_PARAMETERS["titin_rest"],
            "replicates": args.replicates,
        }]).to_excel(writer, sheet_name="settings", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    metrics = [
        ("Fmax_mean_pN", "Fmax_sem_pN", "Fmax (pN)"),
        ("pCa50_mean", "pCa50_sem", "pCa50"),
        ("hill_mean", "hill_sem", "Hill coefficient"),
    ]
    colors = {2.0: "tab:blue", 2.3: "tab:orange"}
    for axis, (mean_col, sem_col, ylabel) in zip(axes, metrics):
        for sl_um in SL_VALUES:
            subset = summary[summary["SL_um"] == sl_um]
            axis.errorbar(subset["titin_a"], subset[mean_col], yerr=subset[sem_col],
                          marker="o", capsize=3, color=colors[sl_um], label=f"SL {sl_um:.1f}")
        axis.set_xlabel("titin_a")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("KO titin_a sensitivity scan (titin_b and titin_rest fixed)")
    fig.tight_layout()
    fig.savefig("KO_titin_a_sweep.png", dpi=200)
    plt.close(fig)
    print("Wrote KO_titin_a_sweep.xlsx and KO_titin_a_sweep.png")


if __name__ == "__main__":
    main()
