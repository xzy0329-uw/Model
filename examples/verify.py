#!/usr/bin/env python3
"""Compare the current WT model with WT pCa-force data at SL 2.0 and 2.3.

Place this file in the repository's examples/ directory, or pass --repo-root.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def find_repo_root(explicit: Path | None) -> Path:
    candidates = [explicit, Path(__file__).resolve().parent.parent, Path.cwd()]
    for candidate in candidates:
        if candidate and (candidate / "multifil_jax").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the Model repository. Run from its root or pass --repo-root."
    )


def normalized_experiment(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sl_um, group in data.groupby("sl_um", sort=True):
        group = group.sort_values("pca").copy()
        scale = float(group["force"].max())
        if scale <= 0:
            raise ValueError(f"Experimental maximum force is not positive at SL={sl_um}.")
        group["normalized_force"] = group["force"] / scale
        group["normalized_sem"] = group["sem"] / scale
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def print_parameter_block(title: str, parameters: dict) -> None:
    print(f"\n=== {title} ===")
    width = max(len(name) for name in parameters)
    for name, value in parameters.items():
        print(f"{name:<{width}} = {value}")


def hill_curve(pca, pca50, hill_coefficient):
    """Normalized descending Hill curve in pCa coordinates."""
    return 1.0 / (1.0 + 10.0 ** (hill_coefficient * (pca - pca50)))


def fit_model_sigmoid(pca: np.ndarray, normalized_force: np.ndarray):
    """Fit a smooth Hill sigmoid to calculated model points (excluding pCa 9 baseline)."""
    mask = pca < 8.0
    fit_pca = np.asarray(pca[mask], dtype=float)
    fit_force = np.clip(np.asarray(normalized_force[mask], dtype=float), 0.0, 1.2)
    popt, _ = curve_fit(
        hill_curve,
        fit_pca,
        fit_force,
        p0=(5.7, 2.0),
        bounds=([4.0, 0.1], [7.0, 10.0]),
        maxfev=20000,
    )
    smooth_pca = np.linspace(float(fit_pca.min()), float(fit_pca.max()), 500)
    return smooth_pca, hill_curve(smooth_pca, *popt), float(popt[0]), float(popt[1])


def simulate_wt(
    repo_root: Path, data: pd.DataFrame, output_dir: Path, args
) -> dict[float, dict[str, np.ndarray]]:
    sys.path.insert(0, str(repo_root))
    from multifil_jax import run
    from multifil_jax.core.params import DynamicParams, StaticParams
    from multifil_jax.core.sarc_geometry import SarcTopology

    # Use the defaults written directly in multifil_jax/core/params.py.
    # Do not call get_cardiac_params(), because that function applies a separate
    # cardiac_overrides dictionary after DynamicParams is constructed.
    static = StaticParams()
    dynamic = DynamicParams()
    params_py_parameters = dynamic.to_dict()
    effective_parameters = params_py_parameters.copy()
    nrows = args.nrows or 2
    ncols = args.ncols or 2
    topology = SarcTopology.create(
        nrows=nrows,
        ncols=ncols,
        static_params=static,
        dynamic_params=dynamic,
    )
    topology = jax.device_put(topology)

    experimental_pca = np.sort(data["pca"].unique())[::-1]
    pca_values = np.unique(np.r_[9.0, experimental_pca])[::-1]
    duration_ms = args.duration_ms or 1000.0
    dt_ms = args.dt_ms or 1.0
    replicates = args.replicates or 10
    steady_last_ms = args.steady_last_ms or 200.0
    rng_seed = 0

    protocol = {
        "sarcomere_lengths_um": [2.0, 2.3],
        "z_line_nm": {"2.0": 1000.0, "2.3": 1150.0},
        "pCa_values": pca_values.tolist(),
        "duration_ms": duration_ms,
        "dt_ms": dt_ms,
        "steady_last_ms": steady_last_ms,
        "replicates": replicates,
        "rng_seed": rng_seed,
        "nrows": nrows,
        "ncols": ncols,
    }
    report = {
        "sources": {
            "model_parameters": "multifil_jax.core.params.DynamicParams() defaults",
            "static_parameters": "multifil_jax.core.params.StaticParams() defaults",
            "experimental_data": str(repo_root / "data" / "pca_force.csv"),
        },
        "params_py_dynamic_defaults": params_py_parameters,
        "effective_dynamic_parameters": effective_parameters,
        "static_parameters": asdict(static),
        "simulation_protocol": protocol,
    }
    parameter_json = output_dir / "WT_pCa_force_run_parameters.json"
    parameter_csv = output_dir / "WT_pCa_force_effective_parameters.csv"
    parameter_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"parameter": name, "value": value} for name, value in effective_parameters.items()]
    ).to_csv(parameter_csv, index=False)

    print_parameter_block("DynamicParams() defaults read directly from params.py", params_py_parameters)
    print_parameter_block("Effective dynamic parameters used", effective_parameters)
    print_parameter_block("Static parameters used", asdict(static))
    print_parameter_block("Simulation protocol", protocol)
    print(f"\nSaved parameter record: {parameter_json}")
    print(f"Saved parameter table:  {parameter_csv}")

    simulations = {}
    for sl_um in (2.0, 2.3):
        print(f"Simulating WT SL={sl_um:.1f} um ({replicates} replicates) ...")
        result = run(
            topology,
            pCa=pca_values.tolist(),
            z_line=sl_um * 500.0,
            duration_ms=duration_ms,
            dt=dt_ms,
            replicates=replicates,
            rng_seed=rng_seed,
            dynamic_params=effective_parameters,
        )
        n_steady = max(1, int(round(steady_last_ms / dt_ms)))
        force = np.asarray(result.axial_force)[..., -n_steady:].mean(axis=-1)
        if force.ndim == 1:
            force = force[:, None]
        elif force.ndim > 2:
            force = force.reshape(force.shape[0], -1)

        baseline_index = int(np.argmin(np.abs(pca_values - 9.0)))
        active = force - force[baseline_index, :][None, :]
        mean_active = active.mean(axis=1)
        scale = max(float(mean_active[np.argmin(pca_values)]), 1e-12)
        normalized_replicates = active / scale
        sem = (
            normalized_replicates.std(axis=1, ddof=1) / np.sqrt(normalized_replicates.shape[1])
            if normalized_replicates.shape[1] > 1
            else np.zeros(len(pca_values))
        )
        normalized_mean = normalized_replicates.mean(axis=1)
        smooth_pca, smooth_force, pca50, hill = fit_model_sigmoid(
            pca_values, normalized_mean
        )
        simulations[sl_um] = {
            "pca": pca_values,
            "normalized_force": normalized_mean,
            "normalized_sem": sem,
            "smooth_pca": smooth_pca,
            "smooth_force": smooth_force,
            "pCa50": pca50,
            "hill_coefficient": hill,
        }
        print(f"  Hill curve: pCa50={pca50:.4f}, nH={hill:.4f}")
    return simulations


def draw_condition(ax, sl_um, experiment, simulation, color):
    exp = experiment[np.isclose(experiment["sl_um"], sl_um)].sort_values("pca")
    sim = simulation[sl_um]
    ax.errorbar(
        exp["pca"], 100.0 * exp["normalized_force"],
        yerr=100.0 * exp["normalized_sem"], fmt="o", ms=6, capsize=3,
        color=color, markerfacecolor="white", markeredgewidth=1.5,
        label=f"Experiment, SL={sl_um:.1f} µm",
    )
    ax.plot(
        sim["pca"], 100.0 * sim["normalized_force"], "o", ms=4,
        color=color, alpha=0.75, label=f"Calculated model points, SL={sl_um:.1f} µm",
    )
    ax.plot(
        sim["smooth_pca"], 100.0 * sim["smooth_force"], "-", lw=2.3,
        color=color,
        label=(f"Model Hill curve, SL={sl_um:.1f} µm "
               f"(pCa₅₀={sim['pCa50']:.3f}, nH={sim['hill_coefficient']:.2f})"),
    )


def finish_axis(ax, title):
    ax.set_xlabel("pCa = −log₁₀[Ca²⁺]")
    ax.set_ylabel("Normalized active force (% maximum)")
    ax.set_title(title)
    ax.set_ylim(bottom=-5)
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--dt-ms", type=float)
    parser.add_argument("--steady-last-ms", type=float)
    parser.add_argument("--nrows", type=int)
    parser.add_argument("--ncols", type=int)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    repo_root = find_repo_root(args.repo_root)
    output_dir = (args.output_dir or repo_root / "outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = repo_root / "data" / "pca_force.csv"
    data = pd.read_csv(data_path)
    required = {"sl_um", "pca", "force", "sem"}
    if not required.issubset(data.columns):
        raise ValueError(f"{data_path} must contain columns: {sorted(required)}")
    wt_data = normalized_experiment(data[data["sl_um"].isin([2.0, 2.3])])
    simulation = simulate_wt(repo_root, wt_data, output_dir, args)

    curve_rows = []
    for sl_um, sim in simulation.items():
        curve_rows.extend(
            {"SL_um": sl_um, "pCa": pca, "normalized_force": force,
             "force_percent_max": 100.0 * force,
             "pCa50": sim["pCa50"], "hill_coefficient": sim["hill_coefficient"]}
            for pca, force in zip(sim["smooth_pca"], sim["smooth_force"])
        )
    curve_csv = output_dir / "WT_model_sigmoidal_pCa_force_curves.csv"
    pd.DataFrame(curve_rows).to_csv(curve_csv, index=False)

    fig1, ax1 = plt.subplots(figsize=(7.2, 5.4))
    draw_condition(ax1, 2.0, wt_data, simulation, "#2463A6")
    finish_axis(ax1, "WT pCa–Force: Model vs Experiment at SL=2.0 µm")
    fig1.tight_layout()
    figure1 = output_dir / "WT_SL20_pCa_force_model_vs_data.png"
    fig1.savefig(figure1, dpi=300, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(8.2, 6.0))
    draw_condition(ax2, 2.0, wt_data, simulation, "#2463A6")
    draw_condition(ax2, 2.3, wt_data, simulation, "#D14B3F")
    finish_axis(ax2, "WT pCa–Force: SL=2.0 and 2.3 µm")
    fig2.tight_layout()
    figure2 = output_dir / "WT_SL20_SL23_pCa_force_model_vs_data.png"
    fig2.savefig(figure2, dpi=300, bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close("all")
    print(f"Saved {figure1}")
    print(f"Saved {figure2}")
    print(f"Saved {curve_csv}")


if __name__ == "__main__":
    main()
