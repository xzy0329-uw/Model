#!/usr/bin/env python3
"""Fit titin parameters to PTP passive-force data, then export four pCa curves.

Expected PTP CSV columns:
    group,sl_um,force,sem

Example rows:
    WT,2.00,125.4,4.2
    WT,2.30,382.7,8.6
    KO,2.00,102.8,3.9

`force` and `sem` must use pN, matching multifil_jax axial_force.  This script
fits only titin_a, titin_b, and titin_rest; all other MODEL_PARAMETERS remain
identical for WT and KO.
"""

import argparse
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


PCA_VALUES = np.array([9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 4.5], dtype=float)
STEADY_WINDOW_MS = 600
FIT_DURATION_MS = 700
PCA_DURATION_MS = 1000
DT_MS = 1
NROWS = 4
NCOLS = 4


# Keep these unchanged while fitting titin parameters.
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


def lattice_from_z(z_line_nm, z0=1000.0, d0=14.0, nu=0.5):
    return d0 * (z0 / np.asarray(z_line_nm)) ** nu


def sl_to_z_line(sl_um):
    """Current model convention: SL 2.0 um corresponds to z_line = 1000 nm."""
    return 500.0 * np.asarray(sl_um, dtype=float)


def hill_pca(pca, fmin, fmax, pca50, hill):
    return fmin + (fmax - fmin) / (1.0 + 10.0 ** (hill * (pca - pca50)))


def sem(values, axis=0):
    values = np.asarray(values)
    return values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ptp",
        default="data/ptp_passive_force.csv",
        help="PTP CSV with group, sl_um, force, sem columns.",
    )
    parser.add_argument("--fit-replicates", type=int, default=8)
    parser.add_argument("--pca-replicates", type=int, default=100)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--popsize", type=int, default=8)
    return parser.parse_args()


def load_ptp_data(path):
    data = pd.read_csv(path)
    required = {"group", "sl_um", "force", "sem"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if (data["sem"] <= 0).any():
        raise ValueError("PTP sem values must be positive.")
    return data.sort_values(["group", "sl_um"]).reset_index(drop=True)


def create_topology():
    static, dynamic = get_cardiac_params()
    dynamic = dynamic.copy(**MODEL_PARAMETERS)
    topology = SarcTopology.create(
        nrows=NROWS,
        ncols=NCOLS,
        static_params=static,
        dynamic_params=dynamic,
    )
    return jax.device_put(topology)


def steady_force(result):
    return np.asarray(result.axial_force)[..., -STEADY_WINDOW_MS:].mean(axis=-1)


def simulate_passive_force(topology, sl_um, titin_parameters, replicates):
    """Simulate passive force at pCa 9.0 for all requested SL values."""
    z_lines = sl_to_z_line(sl_um)
    spacings = lattice_from_z(z_lines)
    params = {**MODEL_PARAMETERS, **titin_parameters}
    result = run(
        topology,
        pCa=[9.0] * len(z_lines),
        z_line=z_lines.tolist(),
        lattice_spacing=spacings.tolist(),
        duration_ms=FIT_DURATION_MS,
        dt=DT_MS,
        replicates=replicates,
        dynamic_params=params,
    )
    # Passive tension is reported as a magnitude for comparison with PTP force.
    return np.abs(steady_force(result))


def fit_titin_group(topology, group_data, fit_replicates, maxiter, popsize):
    sl_um = group_data["sl_um"].to_numpy(dtype=float)
    observed = group_data["force"].to_numpy(dtype=float)
    observed_sem = group_data["sem"].to_numpy(dtype=float)

    def objective(theta):
        titin_parameters = {
            "titin_a": float(theta[0]),
            "titin_b": float(theta[1]),
            "titin_rest": float(theta[2]),
        }
        simulated = simulate_passive_force(
            topology, sl_um, titin_parameters, fit_replicates
        ).mean(axis=1)
        residual = (simulated - observed) / observed_sem
        return float(np.sum(residual ** 2))

    bounds = [
        (1.0, 300.0),       # titin_a, pN
        (0.0001, 0.0500),   # titin_b, 1/nm
        (50.0, 300.0),      # titin_rest, nm
    ]
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        polish=True,
        workers=1,
        updating="immediate",
        seed=12345,
        disp=True,
    )
    fitted = {
        "titin_a": float(result.x[0]),
        "titin_b": float(result.x[1]),
        "titin_rest": float(result.x[2]),
    }
    simulated_replicates = simulate_passive_force(
        topology, sl_um, fitted, fit_replicates
    )
    return fitted, float(result.fun), simulated_replicates


def fit_hill_curve(active_force):
    p0 = [0.0, max(float(active_force.max()), 1.0), 5.7, 1.3]
    bounds = ([0.0, 0.0, 4.5, 0.1], [np.inf, np.inf, 7.5, 8.0])
    return curve_fit(
        hill_pca, PCA_VALUES, active_force, p0=p0, bounds=bounds, maxfev=100000)[0]


def simulate_pca_curve(topology, group, sl_um, titin_parameters, replicates):
    z_line = float(sl_to_z_line(sl_um))
    spacing = float(lattice_from_z(z_line))
    params = {**MODEL_PARAMETERS, **titin_parameters}
    result = run(
        topology,
        pCa=PCA_VALUES.tolist(),
        z_line=z_line,
        lattice_spacing=spacing,
        duration_ms=PCA_DURATION_MS,
        dt=DT_MS,
        replicates=replicates,
        dynamic_params=params,
    )
    raw_force = steady_force(result)
    baseline_index = int(np.where(np.isclose(PCA_VALUES, 9.0))[0][0])
    baseline = raw_force[baseline_index, :]
    active_force = raw_force - baseline[None, :]

    force_rows = []
    for i, pca in enumerate(PCA_VALUES):
        for replicate in range(replicates):
            force_rows.append({
                "group": group,
                "SL_um": sl_um,
                "z_line_nm": z_line,
                "lattice_spacing_nm": spacing,
                "pCa": pca,
                "replicate": replicate + 1,
                "raw_force_pN": raw_force[i, replicate],
                "baseline_force_pN": baseline[replicate],
                "active_force_pN": active_force[i, replicate],
            })

    fit_rows = []
    for replicate in range(replicates):
        try:
            fmin, fmax, pca50, hill = fit_hill_curve(active_force[:, replicate])
        except (ValueError, RuntimeError):
            continue
        fit_rows.append({
            "group": group,
            "SL_um": sl_um,
            "z_line_nm": z_line,
            "lattice_spacing_nm": spacing,
            "replicate": replicate + 1,
            "Fmin_pN": fmin,
            "Fmax_pN": fmax,
            "pCa50": pca50,
            "hill_coefficient": hill,
        })

    summary = pd.DataFrame({
        "group": group,
        "SL_um": sl_um,
        "z_line_nm": z_line,
        "lattice_spacing_nm": spacing,
        "pCa": PCA_VALUES,
        "raw_force_mean_pN": raw_force.mean(axis=1),
        "raw_force_sem_pN": sem(raw_force, axis=1),
        "active_force_mean_pN": active_force.mean(axis=1),
        "active_force_sem_pN": sem(active_force, axis=1),
        "n_replicates": replicates,
    })
    return pd.DataFrame(force_rows), pd.DataFrame(fit_rows), summary


def main():
    args = parse_args()
    ptp_data = load_ptp_data(args.ptp)
    topology = create_topology()
    print(f"Topology: {topology.n_thick} thick, {topology.n_thin} thin")

    titin_by_group = {}
    ptp_fit_rows = []
    ptp_curve_rows = []

    for group, group_data in ptp_data.groupby("group", sort=True):
        print(f"\nFitting titin parameters for {group}")
        fitted, weighted_sse, simulated = fit_titin_group(
            topology,
            group_data,
            args.fit_replicates,
            args.maxiter,
            args.popsize,
        )
        titin_by_group[group] = fitted
        print(f"{group}: {fitted}; weighted SSE={weighted_sse:.3f}")

        for pca_index, (_, observed_row) in enumerate(group_data.iterrows()):
            ptp_curve_rows.append({
                "group": group,
                "SL_um": observed_row["sl_um"],
                "experimental_force_pN": observed_row["force"],
                "experimental_sem_pN": observed_row["sem"],
                "simulated_force_mean_pN": simulated[pca_index].mean(),
                "simulated_force_sem_pN": sem(simulated[pca_index]),
            })
        ptp_fit_rows.append({
            "group": group,
            "weighted_SSE": weighted_sse,
            **fitted,
        })

    force_frames = []
    hill_frames = []
    curve_frames = []
    for group, titin_parameters in titin_by_group.items():
        for sl_um in (2.0, 2.3):
            print(f"\nRunning pCa curve: {group}, SL {sl_um:.1f}")
            force_data, hill_data, curve_data = simulate_pca_curve(
                topology, group, sl_um, titin_parameters, args.pca_replicates
            )
            force_frames.append(force_data)
            hill_frames.append(hill_data)
            curve_frames.append(curve_data)

    force_replicates = pd.concat(force_frames, ignore_index=True)
    hill_replicates = pd.concat(hill_frames, ignore_index=True)
    force_summary = pd.concat(curve_frames, ignore_index=True)
    titin_fit_summary = pd.DataFrame(ptp_fit_rows)
    ptp_curve_summary = pd.DataFrame(ptp_curve_rows)

    hill_summary = (
        hill_replicates.groupby(["group", "SL_um", "z_line_nm", "lattice_spacing_nm"], as_index=False)
        .agg(
            n_successful_fits=("replicate", "count"),
            Fmax_mean_pN=("Fmax_pN", "mean"),
            Fmax_sem_pN=("Fmax_pN", sem),
            pCa50_mean=("pCa50", "mean"),
            pCa50_sem=("pCa50", sem),
            hill_coefficient_mean=("hill_coefficient", "mean"),
            hill_coefficient_sem=("hill_coefficient", sem),
        )
    )
    print("\npCa Hill summary:")
    print(hill_summary.to_string(index=False))

    output_xlsx = Path("WT_KO_titin_PTP_pCa_results.xlsx")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        titin_fit_summary.to_excel(writer, sheet_name="titin_fit_summary", index=False)
        ptp_curve_summary.to_excel(writer, sheet_name="PTP_fit_curve", index=False)
        hill_summary.to_excel(writer, sheet_name="pCa_fit_summary", index=False)
        force_summary.to_excel(writer, sheet_name="pCa_force_summary", index=False)
        hill_replicates.to_excel(writer, sheet_name="pCa_fit_replicates", index=False)
        force_replicates.to_excel(writer, sheet_name="pCa_force_replicates", index=False)
        pd.DataFrame(
            [{"parameter": key, "value": value} for key, value in MODEL_PARAMETERS.items()]
        ).to_excel(writer, sheet_name="fixed_parameters", index=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {"WT": "tab:blue", "KO": "tab:red"}
    styles = {2.0: "-", 2.3: "--"}
    pca_smooth = np.linspace(PCA_VALUES.min(), PCA_VALUES.max(), 400)

    for _, row in hill_summary.iterrows():
        group = row["group"]
        sl_um = row["SL_um"]
        curve = force_summary[
            (force_summary["group"] == group) & (force_summary["SL_um"] == sl_um)
        ]
        label = f"{group} SL {sl_um:.1f}"
        ax.errorbar(
            curve["pCa"], curve["active_force_mean_pN"],
            yerr=curve["active_force_sem_pN"], fmt="o", capsize=3,
            color=colors.get(group, "black"), label=f"{label} data",
        )
        ax.plot(
            pca_smooth,
            hill_pca(
                pca_smooth, 0.0, row["Fmax_mean_pN"],
                row["pCa50_mean"], row["hill_coefficient_mean"],
            ),
            linestyle=styles[sl_um], color=colors.get(group, "black"), linewidth=2,
            label=(f"{label} fit: Fmax={row['Fmax_mean_pN']:.0f}, "
                   f"pCa50={row['pCa50_mean']:.3f}, nH={row['hill_coefficient_mean']:.2f}"),
        )

    ax.invert_xaxis()
    ax.set_xlabel("pCa")
    ax.set_ylabel("Active force (pN)")
    ax.set_title("WT and KO pCa-Force Curves After PTP-Based Titin Fitting")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_png = Path("WT_KO_SL20_SL23_pCa_force.png")
    fig.savefig(output_png, dpi=220)
    plt.show()

    Path("fitted_titin_parameters.json").write_text(
        json.dumps(titin_by_group, indent=2), encoding="utf-8"
    )
    print(f"\nWrote: {output_xlsx.resolve()}")
    print(f"Wrote: {output_png.resolve()}")
    print("Wrote: fitted_titin_parameters.json")


if __name__ == "__main__":
    main()
