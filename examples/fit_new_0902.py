#!/usr/bin/env python3
"""Fit titin and tropomyosin parameters to WT data.

Required pCa CSV columns: sl_um, pca, force, sem
Required passive CSV columns: sl_um, force, sem
The model produces pN while the experiment may be tension (e.g. mN/mm^2).
For every candidate parameter vector this script estimates one shared linear
force scale for active pCa force and passive tension. The PTP point at 24%
stretch (normally SL 2.48 um when the reference SL is 2.0 um) constrains titin.

One passive point improves titin identifiability but does not replace fitting
the complete passive length-tension curve.
"""

import argparse
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, curve_fit

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


DEFAULT_DATA_FILE = Path("outputs/data/pca_force.csv")
DEFAULT_PASSIVE_FILE = Path("outputs/data/passive_length.csv")
TARGET_SL_UM = 2.0
PASSIVE_SL_UM = 2.48  # 24% stretch from a 2.0 um reference length
Z_LINE_NM = 1000.0
LATTICE_SPACING_NM = 14.0
STEADY_WINDOW_MS = 600
DURATION_MS = 1000
DT_MS = 1.0

# All parameters not being fitted remain fixed here. Keep this synchronized
# with the verified WT pCa protocol used elsewhere in the project.
FIXED_PARAMETERS = {
    "xb_lda_enabled": 1.0,
    "xb_lda_gain": 3.0,
    "xb_lda_strain_threshold": 1.0,
    "xb_lda_strain_scale": 0.5,
    "xb_lattice_reference": 14.0,
    "xb_lattice_binding_beta": 1.0,
    "xb_r12_coeff": 250.0,
    "xb_r23_coeff": 0.60,
    "xb_r34_coeff": 0.15,
    "xb_r45_coeff": 0.60,
    "xb_r51": 0.10,
    "xb_r15": 0.01,
    "xb_srx_k0": 0.003,
    "xb_r16": 0.010,
}

# Parameters fitted in log10 space. Bounds are deliberately broad but finite.
PARAMETER_SPECS = [
    ("titin_a", 1.0, 120.0),
    ("titin_b", 0.001, 0.020),
    ("titin_rest", 80.0, 220.0),
    ("tm_k_12", 1.0e3, 1.0e5),
    ("tm_k_23", 0.005, 5.0),
    ("tm_k_34", 0.005, 5.0),
    ("tm_k_41", 0.001, 5.0),
    ("tm_K1", 1.0e3, 5.0e4),
    ("tm_K2", 1.0, 1.0e5),
    ("tm_K3", 0.001, 5.0),
]


def hill_pca(pca, fmin, fmax, pca50, hill):
    return fmin + (fmax - fmin) / (1.0 + 10.0 ** (hill * (pca - pca50)))


def decode(theta):
    return {
        name: float(10.0 ** value)
        for value, (name, _, _) in zip(theta, PARAMETER_SPECS)
    }


def load_target(path):
    frame = pd.read_csv(path)
    required = {"sl_um", "pca", "force", "sem"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    target = frame[np.isclose(frame["sl_um"].astype(float), TARGET_SL_UM)].copy()
    if target.empty:
        raise ValueError(f"No rows with sl_um={TARGET_SL_UM} in {path}")
    target = target.sort_values("pca", ascending=False).reset_index(drop=True)
    if (target["sem"] <= 0).any():
        raise ValueError("Every experimental SEM must be positive.")
    return target


def load_passive_target(path, sl_um):
    frame = pd.read_csv(path)
    required = {"sl_um", "force", "sem"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    closest = frame.iloc[(frame["sl_um"].astype(float) - sl_um).abs().argmin()].copy()
    if not np.isclose(float(closest["sl_um"]), sl_um, atol=1e-6):
        raise ValueError(
            f"{path} has no row at SL {sl_um:.3f} um. "
            f"Nearest row is SL {float(closest['sl_um']):.3f} um."
        )
    if float(closest["sem"]) <= 0:
        raise ValueError("The passive-force SEM must be positive.")
    return closest


def lattice_from_z(z_line_nm, z0=1000.0, d0=14.0, nu=0.5):
    return d0 * (z0 / z_line_nm) ** nu


def simulate_active_force(topology, pca_values, parameters, replicates):
    result = run(
        topology,
        pCa=pca_values.tolist(),
        z_line=Z_LINE_NM,
        lattice_spacing=LATTICE_SPACING_NM,
        duration_ms=DURATION_MS,
        dt=DT_MS,
        replicates=replicates,
        dynamic_params=parameters,
    )
    steady_force = np.asarray(result.axial_force)[..., -STEADY_WINDOW_MS:].mean(axis=-1)
    baseline = steady_force[0, :]
    return steady_force[1:, :] - baseline[None, :]


def simulate_passive_force(topology, sl_um, parameters, replicates):
    z_line = sl_um * 500.0
    result = run(
        topology,
        pCa=[9.0],
        z_line=z_line,
        lattice_spacing=lattice_from_z(z_line),
        duration_ms=DURATION_MS,
        dt=DT_MS,
        replicates=replicates,
        dynamic_params=parameters,
    )
    steady_force = np.asarray(result.axial_force)[..., -STEADY_WINDOW_MS:].mean(axis=-1)
    # Passive PTP is reported as tension magnitude rather than axial sign.
    return np.abs(steady_force[0, :])


def optimal_scale(model_force, experimental_force, experimental_sem):
    weights = 1.0 / np.square(experimental_sem)
    numerator = np.sum(weights * model_force * experimental_force)
    denominator = np.sum(weights * np.square(model_force))
    return max(0.0, numerator / max(denominator, 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--passive-data", type=Path, default=DEFAULT_PASSIVE_FILE)
    parser.add_argument("--passive-sl", type=float, default=PASSIVE_SL_UM)
    parser.add_argument("--fit-replicates", type=int, default=8)
    parser.add_argument("--final-replicates", type=int, default=32)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    target = load_target(args.data)
    passive_target = load_passive_target(args.passive_data, args.passive_sl)
    pca_experiment = target["pca"].to_numpy(dtype=float)
    force_experiment = target["force"].to_numpy(dtype=float)
    sem_experiment = target["sem"].to_numpy(dtype=float)
    pca_protocol = np.concatenate(([9.0], pca_experiment))
    passive_experiment = float(passive_target["force"])
    passive_sem = float(passive_target["sem"])

    static, dynamic = get_cardiac_params()
    dynamic = dynamic.copy(**FIXED_PARAMETERS)
    topology = jax.device_put(SarcTopology.create(
        nrows=4, ncols=4, static_params=static, dynamic_params=dynamic
    ))

    bounds = [(np.log10(lower), np.log10(upper)) for _, lower, upper in PARAMETER_SPECS]
    history = []

    def objective(theta):
        fitted = decode(theta)
        parameters = {**FIXED_PARAMETERS, **fitted}
        try:
            active = simulate_active_force(
                topology, pca_protocol, parameters, args.fit_replicates
            )
            model_mean_pn = active.mean(axis=1)
            passive_mean_pn = simulate_passive_force(
                topology, args.passive_sl, parameters, args.fit_replicates
            ).mean()
            model_values = np.concatenate((model_mean_pn, [passive_mean_pn]))
            experimental_values = np.concatenate((force_experiment, [passive_experiment]))
            experimental_sems = np.concatenate((sem_experiment, [passive_sem]))
            scale = optimal_scale(model_values, experimental_values, experimental_sems)
            residual = (scale * model_values - experimental_values) / experimental_sems
            value = float(np.sum(np.square(residual)))
            return value if np.isfinite(value) else 1.0e30
        except Exception as exc:
            print(f"Candidate failed: {exc}")
            return 1.0e30

    def callback(theta, convergence):
        value = objective(theta)
        history.append({
            "generation": len(history) + 1,
            "weighted_sse": value,
            "convergence": float(convergence),
            **decode(theta),
        })
        print(f"generation {len(history)}: weighted SSE={value:.4f}")
        return False

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        workers=1,
        updating="immediate",
        polish=False,
        callback=callback,
        disp=True,
    )

    fitted = decode(result.x)
    parameters = {**FIXED_PARAMETERS, **fitted}
    active = simulate_active_force(
        topology, pca_protocol, parameters, args.final_replicates
    )
    passive = simulate_passive_force(
        topology, args.passive_sl, parameters, args.final_replicates
    )
    model_mean_pn = active.mean(axis=1)
    model_sem_pn = active.std(axis=1, ddof=1) / np.sqrt(args.final_replicates)
    force_scale = optimal_scale(
        np.concatenate((model_mean_pn, [passive.mean()])),
        np.concatenate((force_experiment, [passive_experiment])),
        np.concatenate((sem_experiment, [passive_sem])),
    )
    model_mean_scaled = force_scale * model_mean_pn
    model_sem_scaled = force_scale * model_sem_pn

    fit_rows = []
    for replicate in range(args.final_replicates):
        try:
            fmin, fmax, pca50, hill = curve_fit(
                hill_pca,
                pca_experiment,
                active[:, replicate],
                p0=[0.0, float(active[:, replicate].max()), 5.7, 1.3],
                bounds=([0.0, 0.0, 4.0, 0.1], [np.inf, np.inf, 8.0, 10.0]),
                maxfev=100000,
            )[0]
            fit_rows.append({
                "replicate": replicate + 1,
                "Fmin_model_pN": fmin,
                "Fmax_model_pN": fmax,
                "Fmax_scaled": fmax * force_scale,
                "pCa50": pca50,
                "hill_coefficient": hill,
            })
        except (RuntimeError, ValueError):
            pass

    force_rows = []
    for index, pca in enumerate(pca_experiment):
        for replicate in range(args.final_replicates):
            force_rows.append({
                "pCa": pca,
                "replicate": replicate + 1,
                "active_force_model_pN": active[index, replicate],
                "active_force_scaled": active[index, replicate] * force_scale,
            })

    passive_frame = pd.DataFrame({
        "SL_um": args.passive_sl,
        "replicate": np.arange(1, args.final_replicates + 1),
        "passive_force_model_pN": passive,
        "passive_force_scaled": passive * force_scale,
        "passive_force_experiment": passive_experiment,
        "passive_force_experiment_sem": passive_sem,
    })

    fit_frame = pd.DataFrame(fit_rows)
    force_frame = pd.DataFrame(force_rows)
    summary = pd.DataFrame([{
        "target": "WT SL 2.0",
        "weighted_sse": float(result.fun),
        "force_scale_experiment_units_per_pN": force_scale,
        "n_final_successful_hill_fits": len(fit_frame),
        "Fmax_scaled_mean": fit_frame["Fmax_scaled"].mean(),
        "Fmax_scaled_sem": fit_frame["Fmax_scaled"].sem(),
        "pCa50_mean": fit_frame["pCa50"].mean(),
        "pCa50_sem": fit_frame["pCa50"].sem(),
        "hill_mean": fit_frame["hill_coefficient"].mean(),
        "hill_sem": fit_frame["hill_coefficient"].sem(),
        "passive_SL_um": args.passive_sl,
        "passive_experiment": passive_experiment,
        "passive_experiment_sem": passive_sem,
        "passive_model_scaled_mean": passive_frame["passive_force_scaled"].mean(),
        "passive_model_scaled_sem": passive_frame["passive_force_scaled"].sem(),
        **fitted,
    }])

    with pd.ExcelWriter("WT_SL20_pca_parameter_fit.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="fit_summary", index=False)
        target.to_excel(writer, sheet_name="experimental_data", index=False)
        pd.DataFrame([passive_target]).to_excel(writer, sheet_name="passive_24pct_data", index=False)
        pd.DataFrame(history).to_excel(writer, sheet_name="optimization_history", index=False)
        force_frame.to_excel(writer, sheet_name="force_replicates", index=False)
        passive_frame.to_excel(writer, sheet_name="passive_24pct_replicates", index=False)
        fit_frame.to_excel(writer, sheet_name="hill_replicates", index=False)
        pd.DataFrame([FIXED_PARAMETERS]).to_excel(writer, sheet_name="fixed_parameters", index=False)

    with open("WT_SL20_pca_fitted_parameters.json", "w", encoding="utf-8") as handle:
        json.dump({
            "target": "WT SL 2.0",
            "weighted_sse": float(result.fun),
            "force_scale_experiment_units_per_pN": force_scale,
            "fitted_parameters": fitted,
            "fixed_parameters": FIXED_PARAMETERS,
        }, handle, indent=2)

    pca_smooth = np.linspace(pca_experiment.min(), pca_experiment.max(), 300)
    mean_hill = curve_fit(
        hill_pca, pca_experiment, model_mean_scaled,
        p0=[0.0, float(model_mean_scaled.max()), 5.7, 1.3],
        bounds=([0.0, 0.0, 4.0, 0.1], [np.inf, np.inf, 8.0, 10.0]),
        maxfev=100000,
    )[0]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.errorbar(pca_experiment, force_experiment, yerr=sem_experiment,
                fmt="o", capsize=4, color="black", label="WT SL 2.0 experiment")
    ax.errorbar(pca_experiment, model_mean_scaled, yerr=model_sem_scaled,
                fmt="o", capsize=4, color="tab:blue", label="Fitted model mean")
    ax.plot(pca_smooth, hill_pca(pca_smooth, *mean_hill), color="tab:blue", linewidth=2)
    ax.invert_xaxis()
    ax.set(xlabel="pCa", ylabel="Active force (experimental units)",
           title="WT SL 2.0 pCa-force parameter fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig("WT_SL20_pca_parameter_fit.png", dpi=200)
    plt.close(fig)

    print("\nBest fitted parameters:")
    for name, value in fitted.items():
        print(f"  {name}={value:.8g}")
    print(f"Weighted SSE: {result.fun:.4f}")
    print("Wrote WT_SL20_pca_parameter_fit.xlsx")
    print("Wrote WT_SL20_pca_fitted_parameters.json")
    print("Wrote WT_SL20_pca_parameter_fit.png")


if __name__ == "__main__":
    main()
