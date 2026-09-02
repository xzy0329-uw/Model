#!/usr/bin/env python3
"""Create pCa-force reports from the fitted WT SL 2.0 parameter JSON.

Outputs:
1. WT SL 2.0 model-versus-experiment pCa-force figure.
2. WT model prediction at SL 2.0 and SL 2.3.
3. Ten SL 2.0 pCa-force curves with progressively lower titin_a and titin_b.

All replicate-level forces, Hill fits, parameter values, and summary statistics
are written to one workbook.
"""

import argparse
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


DEFAULT_FIT_FILE = Path("WT_SL20_pca_fitted_parameters.json")
DEFAULT_PCA_DATA = Path("data/pca_force.csv")
PCA_FALLBACK = Path("outputs/data/pca_force.csv")
PCA_VALUES = np.array([9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 4.5])
STEADY_WINDOW_MS = 600
DURATION_MS = 1000
DT_MS = 1.0

# Ten paired reductions from the fitted WT titin_a and titin_b values.
TITIN_SCALE_FACTORS = np.linspace(0.95, 0.50, 10)


def lattice_from_z(z_line_nm, z0=1000.0, d0=14.0, nu=0.5):
    return d0 * (z0 / z_line_nm) ** nu


def hill_pca(pca, fmin, fmax, pca50, hill):
    return fmin + (fmax - fmin) / (1.0 + 10.0 ** (hill * (pca - pca50)))


def load_experiment(path):
    if not path.exists() and PCA_FALLBACK.exists():
        path = PCA_FALLBACK
    frame = pd.read_csv(path)
    required = {"sl_um", "pca", "force", "sem"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[np.isclose(frame["sl_um"].astype(float), 2.0)].copy()
    return frame.sort_values("pca", ascending=False).reset_index(drop=True), path


def simulate_condition(topology, label, sl_um, parameters, replicates):
    z_line = sl_um * 500.0
    spacing = lattice_from_z(z_line)
    print(f"Running {label}: SL={sl_um:.2f}, d1,0={spacing:.3f} nm")
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
    raw = np.asarray(result.axial_force)[..., -STEADY_WINDOW_MS:].mean(axis=-1)
    active = raw - raw[0, :][None, :]
    return {
        "label": label,
        "sl_um": sl_um,
        "z_line_nm": z_line,
        "lattice_spacing_nm": spacing,
        "raw_force": raw,
        "active_force": active,
        "parameters": parameters,
    }


def fit_replicates(run_data, scale):
    rows = []
    active = run_data["active_force"]
    for replicate in range(active.shape[1]):
        try:
            fit = curve_fit(
                hill_pca,
                PCA_VALUES,
                active[:, replicate],
                p0=[0.0, float(active[:, replicate].max()), 5.7, 1.3],
                bounds=([0.0, 0.0, 4.0, 0.1], [np.inf, np.inf, 8.0, 10.0]),
                maxfev=100000,
            )[0]
        except (RuntimeError, ValueError):
            continue
        fmin, fmax, pca50, hill = fit
        rows.append({
            "condition": run_data["label"], "SL_um": run_data["sl_um"],
            "replicate": replicate + 1,
            "Fmin_model_pN": fmin, "Fmax_model_pN": fmax,
            "Fmax_scaled": fmax * scale, "pCa50": pca50,
            "hill_coefficient": hill,
        })
    return pd.DataFrame(rows)


def force_replicates(run_data, scale):
    rows = []
    for pca_index, pca in enumerate(PCA_VALUES):
        for replicate in range(run_data["active_force"].shape[1]):
            rows.append({
                "condition": run_data["label"], "SL_um": run_data["sl_um"],
                "z_line_nm": run_data["z_line_nm"],
                "lattice_spacing_nm": run_data["lattice_spacing_nm"],
                "pCa": pca, "replicate": replicate + 1,
                "raw_force_model_pN": run_data["raw_force"][pca_index, replicate],
                "active_force_model_pN": run_data["active_force"][pca_index, replicate],
                "active_force_scaled": run_data["active_force"][pca_index, replicate] * scale,
            })
    return pd.DataFrame(rows)


def summarize_fits(frame):
    return frame.groupby(["condition", "SL_um"], as_index=False).agg(
        n_successful_fits=("replicate", "count"),
        Fmax_model_pN_mean=("Fmax_model_pN", "mean"),
        Fmax_model_pN_sem=("Fmax_model_pN", "sem"),
        Fmax_scaled_mean=("Fmax_scaled", "mean"),
        Fmax_scaled_sem=("Fmax_scaled", "sem"),
        pCa50_mean=("pCa50", "mean"),
        pCa50_sem=("pCa50", "sem"),
        hill_mean=("hill_coefficient", "mean"),
        hill_sem=("hill_coefficient", "sem"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-file", type=Path, default=DEFAULT_FIT_FILE)
    parser.add_argument("--pca-data", type=Path, default=DEFAULT_PCA_DATA)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--sweep-replicates", type=int, default=16)
    parser.add_argument("--sweep-sl", type=float, default=2.0)
    args = parser.parse_args()

    with open(args.fit_file, encoding="utf-8") as handle:
        fit_result = json.load(handle)
    fitted = fit_result["fitted_parameters"]
    fixed = fit_result["fixed_parameters"]
    force_scale = float(fit_result["force_scale_experiment_units_per_pN"])
    parameters = {**fixed, **fitted}

    experiment, experiment_path = load_experiment(args.pca_data)
    static, dynamic = get_cardiac_params()
    dynamic = dynamic.copy(**fixed)
    effective_dynamic = dynamic.copy(**fitted)
    effective_parameter_snapshot = effective_dynamic.to_dict()
    topology = jax.device_put(SarcTopology.create(
        nrows=4, ncols=4, static_params=static, dynamic_params=dynamic
    ))

    wt_sl20 = simulate_condition(topology, "WT fitted SL 2.0", 2.0, parameters, args.replicates)
    wt_sl23 = simulate_condition(topology, "WT fitted SL 2.3", 2.3, parameters, args.replicates)
    base_runs = [wt_sl20, wt_sl23]

    sweep_runs = []
    sweep_parameter_rows = []
    for index, factor in enumerate(TITIN_SCALE_FACTORS, start=1):
        sweep_parameters = {
            **parameters,
            "titin_a": float(fitted["titin_a"] * factor),
            "titin_b": float(fitted["titin_b"] * factor),
        }
        label = f"titin sweep {index:02d} ({factor:.2f}x)"
        sweep_runs.append(simulate_condition(
            topology, label, args.sweep_sl, sweep_parameters, args.sweep_replicates
        ))
        sweep_parameter_rows.append({
            "condition": label, "scale_factor": factor,
            "titin_a": sweep_parameters["titin_a"],
            "titin_b": sweep_parameters["titin_b"],
            "titin_rest": sweep_parameters["titin_rest"],
        })

    base_fit_frame = pd.concat([fit_replicates(item, force_scale) for item in base_runs], ignore_index=True)
    base_force_frame = pd.concat([force_replicates(item, force_scale) for item in base_runs], ignore_index=True)
    sweep_fit_frame = pd.concat([fit_replicates(item, force_scale) for item in sweep_runs], ignore_index=True)
    sweep_force_frame = pd.concat([force_replicates(item, force_scale) for item in sweep_runs], ignore_index=True)
    base_summary = summarize_fits(base_fit_frame)
    sweep_summary = summarize_fits(sweep_fit_frame).merge(
        pd.DataFrame(sweep_parameter_rows)[["condition", "scale_factor", "titin_a", "titin_b", "titin_rest"]],
        on="condition", how="left",
    )

    with pd.ExcelWriter("WT_fitted_model_and_titin_sweep.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([fitted]).to_excel(writer, sheet_name="fitted_parameters", index=False)
        pd.DataFrame([fixed]).to_excel(writer, sheet_name="fixed_parameters", index=False)
        pd.DataFrame([effective_parameter_snapshot]).to_excel(
            writer, sheet_name="all_effective_parameters", index=False
        )
        pd.DataFrame([{
            "force_scale_experiment_units_per_pN": force_scale,
            "pca_experiment_file": str(experiment_path),
            "main_replicates": args.replicates,
            "sweep_replicates": args.sweep_replicates,
            "sweep_SL_um": args.sweep_sl,
        }]).to_excel(writer, sheet_name="run_settings", index=False)
        experiment.to_excel(writer, sheet_name="WT_SL20_experiment", index=False)
        base_summary.to_excel(writer, sheet_name="WT_SL20_SL23_summary", index=False)
        base_fit_frame.to_excel(writer, sheet_name="WT_SL20_SL23_hill_reps", index=False)
        base_force_frame.to_excel(writer, sheet_name="WT_SL20_SL23_force_reps", index=False)
        pd.DataFrame(sweep_parameter_rows).to_excel(writer, sheet_name="titin_sweep_parameters", index=False)
        sweep_summary.to_excel(writer, sheet_name="titin_sweep_summary", index=False)
        sweep_fit_frame.to_excel(writer, sheet_name="titin_sweep_hill_reps", index=False)
        sweep_force_frame.to_excel(writer, sheet_name="titin_sweep_force_reps", index=False)

    # Figure 1: fitted WT SL 2.0 versus experiment.
    wt20_mean = wt_sl20["active_force"].mean(axis=1) * force_scale
    wt20_sem = wt_sl20["active_force"].std(axis=1, ddof=1) / np.sqrt(args.replicates) * force_scale
    pca_smooth = np.linspace(PCA_VALUES.min(), PCA_VALUES.max(), 300)
    mean_fit = curve_fit(hill_pca, PCA_VALUES, wt20_mean,
                         p0=[0.0, wt20_mean.max(), 5.7, 1.3],
                         bounds=([0.0, 0.0, 4.0, 0.1], [np.inf, np.inf, 8.0, 10.0]),
                         maxfev=100000)[0]
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.errorbar(experiment["pca"], experiment["force"], yerr=experiment["sem"], fmt="o",
                capsize=4, color="black", label="WT SL 2.0 experiment")
    ax.errorbar(PCA_VALUES, wt20_mean, yerr=wt20_sem, fmt="o", capsize=4,
                color="tab:blue", label="Fitted WT model")
    ax.plot(pca_smooth, hill_pca(pca_smooth, *mean_fit), color="tab:blue", linewidth=2)
    ax.invert_xaxis()
    ax.set(xlabel="pCa", ylabel="Active force (experimental units)", title="WT SL 2.0: fitted model versus experiment")
    ax.legend()
    fig.tight_layout()
    fig.savefig("WT_SL20_fit_vs_experiment.png", dpi=200)
    plt.close(fig)

    # Figure 2: fitted WT model at both lengths.
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for item, color, style in [(wt_sl20, "tab:blue", "-"), (wt_sl23, "tab:green", "--")]:
        mean = item["active_force"].mean(axis=1) * force_scale
        sem = item["active_force"].std(axis=1, ddof=1) / np.sqrt(args.replicates) * force_scale
        fit = curve_fit(hill_pca, PCA_VALUES, mean, p0=[0.0, mean.max(), 5.7, 1.3],
                        bounds=([0.0, 0.0, 4.0, 0.1], [np.inf, np.inf, 8.0, 10.0]), maxfev=100000)[0]
        ax.errorbar(PCA_VALUES, mean, yerr=sem, fmt="o", capsize=4, color=color)
        ax.plot(pca_smooth, hill_pca(pca_smooth, *fit), linestyle=style, color=color,
                label=f"WT SL {item['sl_um']:.1f}: Fmax={fit[1]:.2f}, pCa50={fit[2]:.3f}")
    ax.invert_xaxis()
    ax.set(xlabel="pCa", ylabel="Active force (experimental units)", title="Fitted WT model: SL 2.0 and SL 2.3")
    ax.legend()
    fig.tight_layout()
    fig.savefig("WT_SL20_SL23_model_prediction.png", dpi=200)
    plt.close(fig)

    # Figure 3: ten titin_a/titin_b reductions at the selected SL.
    fig, ax = plt.subplots(figsize=(9, 6.5))
    cmap = plt.get_cmap("viridis")
    for index, item in enumerate(sweep_runs):
        mean = item["active_force"].mean(axis=1) * force_scale
        color = cmap(index / max(1, len(sweep_runs) - 1))
        factor = TITIN_SCALE_FACTORS[index]
        ax.plot(PCA_VALUES, mean, marker="o", color=color, label=f"a,b = {factor:.2f}x fitted")
    ax.invert_xaxis()
    ax.set(xlabel="pCa", ylabel="Active force (experimental units)",
           title=f"titin_a/titin_b reduction sweep at SL {args.sweep_sl:.1f}")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig("WT_titin_ab_downward_sweep.png", dpi=200)
    plt.close(fig)

    print("Wrote WT_fitted_model_and_titin_sweep.xlsx")
    print("Wrote WT_SL20_fit_vs_experiment.png")
    print("Wrote WT_SL20_SL23_model_prediction.png")
    print("Wrote WT_titin_ab_downward_sweep.png")


if __name__ == "__main__":
    main()
