"""Joint calibration for multifil_jax passive, pCa-force, and ktr data.

Expected CSV files (headers are required):
  data/passive_length.csv : sl_um,force,sem
  data/pca_force.csv      : sl_um,pca,force,sem
  data/ktr.csv            : sl_um,pca,ktr_s,sem

The script uses a staged fit because pCa-force data alone cannot identify all
mechanisms.  Run the stages in this order and pass each JSON result to the
next stage with --start-json:

  python fit_joint_experiments.py --stage passive
  python fit_joint_experiments.py --stage pca --start-json fit_passive.json
  python fit_joint_experiments.py --stage kinetics --start-json fit_pca.json
  python fit_joint_experiments.py --stage lda --start-json fit_kinetics.json

All target simulations have LDA enabled.  The passive and pCa stages use
steady-state measurements.  The kinetics stage estimates ktr after a short
length-release/restretch protocol; set the protocol constants below to match
your experiment before interpreting that result.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PASSIVE_FILE = DATA_DIR / "passive_length.csv"
PCA_FILE = DATA_DIR / "pca_force.csv"
KTR_FILE = DATA_DIR / "ktr.csv"

# Workbook import settings.  These are used only with --workbook.  The raw
# PTP layout may have any number of preparation columns: each row is reduced
# to mean and SEM after empty cells are removed.
REFERENCE_SL_FOR_STRETCH_UM = 2.0

NROWS = 2
NCOLS = 2
DT_MS = 1.0
STEADY_DURATION_MS = 1000.0
STEADY_WINDOW_MS = 600.0
REPLICATES = 8
RNG_SEED = 12345
MAX_FIT_SOLVER_RESIDUAL_PN = 10.0

# Map experimental sarcomere length to this model's Z-line coordinate.
# Your present convention is SL 2.0 -> 1000 nm and SL 2.3 -> 1150 nm.
Z_AT_SL_2_0_NM = 1000.0
NM_Z_PER_UM_SL = 500.0
LATTICE_SPACING_NM = 14.0

# Set True if force in passive_length.csv was reported relative to its first
# length point.  Set False if it is an absolute model-comparable force.
PASSIVE_FORCE_RELATIVE_TO_FIRST_POINT = True

# pca_force.csv should contain active force.  The simulation baseline at pCa 9
# is subtracted automatically.
PCA_FORCE_IS_ACTIVE = True
# Experimental PTP force is not on the same absolute pN scale as the 2x2
# half-sarcomere model. Keep this True unless a validated force conversion is
# available; each SL pCa curve is then fit by shape rather than amplitude.
NORMALIZE_PCA_FORCE = True

# Quick-release/restretch protocol used only for the ktr stage.
# The model first equilibrates at target pCa and length, shortens one timestep,
# then returns to the original length.  Match RELEASE_NM to the experiment.
KTR_DURATION_MS = 600.0
KTR_RELEASE_AT_MS = 20
KTR_RELEASE_NM = 4.0
KTR_FIT_START_AFTER_RESTRETCH_MS = 3
KTR_FIT_END_BEFORE_END_MS = 100

# Keep this modest while debugging.  A final fit may use 80-150 iterations.
MAXITER = 24
POPSIZE = 8


def z_line_from_sl(sl_um: float) -> float:
    return Z_AT_SL_2_0_NM + NM_Z_PER_UM_SL * (float(sl_um) - 2.0)


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Expected columns: {sorted(required_columns)}")
    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if (frame["sem"] <= 0).any():
        raise ValueError(f"All SEM values in {path} must be positive.")
    return frame.sort_values(list(required_columns - {"force", "ktr_s", "sem"})).reset_index(drop=True)


def aggregate_replicate_columns(raw: pd.DataFrame, value_name: str, force_scale_to_pn: float) -> pd.DataFrame:
    """Reduce a sheet with condition in column 1 and preparations in columns 2+."""
    condition = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    replicates = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    count = replicates.count(axis=1)
    mean = replicates.mean(axis=1)
    sem = replicates.std(axis=1, ddof=1) / np.sqrt(count)
    output = pd.DataFrame({"condition": condition, value_name: mean, "sem": sem, "n": count})
    output = output.dropna(subset=["condition", value_name, "sem"])
    output = output[output["n"] >= 2].copy()
    output[value_name] *= force_scale_to_pn
    output["sem"] *= force_scale_to_pn
    return output.reset_index(drop=True)


def parse_sheet_mapping(entries: list[str] | None) -> dict[str, float]:
    """Parse repeatable CLI values like 'pCa SL2.0:2.0'."""
    mapping = {}
    for entry in entries or []:
        try:
            sheet_name, sl_text = entry.rsplit(":", 1)
            mapping[sheet_name] = float(sl_text)
        except ValueError as error:
            raise ValueError(f"Invalid sheet mapping '{entry}'. Use 'sheet name:SL'.") from error
    return mapping


def load_frames_from_workbook(
    workbook: Path,
    passive_sheet: str,
    pca_sheets: dict[str, float],
    ktr_sheets: dict[str, float],
    force_scale_to_pn: float,
    reference_sl_um: float,
) -> dict[str, pd.DataFrame]:
    """Read PTP-style wide sheets directly from the experimental workbook.

    Passive sheet: first column is % stretch, remaining columns are force.
    pCa sheet: first column is pCa, remaining columns are active force.
    ktr sheet: first column is pCa, remaining columns are ktr in s^-1.
    """
    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")
    if force_scale_to_pn <= 0:
        raise ValueError("--force-to-pn must be positive.")

    passive_raw = pd.read_excel(workbook, sheet_name=passive_sheet)
    passive = aggregate_replicate_columns(passive_raw, "force", force_scale_to_pn)
    passive = passive.rename(columns={"condition": "stretch_pct"})
    passive["sl_um"] = reference_sl_um * (1.0 + passive["stretch_pct"] / 100.0)
    passive = passive[["sl_um", "force", "sem"]].sort_values("sl_um").reset_index(drop=True)

    def load_condition_sheets(sheet_map: dict[str, float], value_name: str, scale: float) -> pd.DataFrame:
        frames = []
        for sheet_name, sl_um in sheet_map.items():
            raw = pd.read_excel(workbook, sheet_name=sheet_name)
            frame = aggregate_replicate_columns(raw, value_name, scale)
            frame = frame.rename(columns={"condition": "pca"})
            frame["sl_um"] = sl_um
            frames.append(frame[["sl_um", "pca", value_name, "sem"]])
        if not frames:
            raise ValueError("At least one --pca-sheet mapping is required with --workbook.")
        return pd.concat(frames, ignore_index=True).sort_values(["sl_um", "pca"]).reset_index(drop=True)

    pca = load_condition_sheets(pca_sheets, "force", force_scale_to_pn) if pca_sheets else None
    ktr = load_condition_sheets(ktr_sheets, "ktr_s", 1.0) if ktr_sheets else None
    return {"passive": passive, "pca": pca, "ktr": ktr}


def active_force_from_result(result, window_ms: float = STEADY_WINDOW_MS) -> np.ndarray:
    """Return one steady force per sweep point, averaged over replicates."""
    force = np.asarray(result.axial_force)
    n_window = max(1, int(window_ms / result.dt))
    steady = force[..., -n_window:].mean(axis=-1)
    if steady.ndim == 1:  # (replicates,)
        return np.array([steady.mean()])
    return steady.mean(axis=-1)  # (sweep, replicates) -> (sweep,)


def require_solver_convergence(result) -> None:
    """Reject parameter candidates that do not reach mechanical equilibrium."""
    residual = np.asarray(result.metrics["solver_residual"])
    max_residual = float(np.max(np.abs(residual)))
    if not np.isfinite(max_residual) or max_residual > MAX_FIT_SOLVER_RESIDUAL_PN:
        raise FloatingPointError(
            f"solver residual {max_residual:.3g} pN exceeds "
            f"{MAX_FIT_SOLVER_RESIDUAL_PN} pN"
        )


def force_trace_from_result(result) -> np.ndarray:
    """Mean force trace over replicates for a scalar pCa/z-line simulation."""
    force = np.asarray(result.axial_force)
    if force.ndim != 2:
        raise ValueError(f"Expected (replicates, time), got force shape {force.shape}")
    return force.mean(axis=0)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    transform: str
    bounds: tuple[float, float]

    def decode(self, x: float, base_value: float) -> float:
        if self.transform == "log_scale":
            return base_value * 10.0 ** x
        if self.transform == "negative_log_scale":
            return base_value * 10.0 ** x
        if self.transform == "additive":
            return base_value + x
        if self.transform == "log_absolute":
            return 10.0 ** x
        raise ValueError(f"Unknown transform {self.transform}")


# tm_K4 is intentionally absent: it is currently unused by transitions.py.
STAGES: dict[str, list[ParameterSpec]] = {
    "passive": [
        ParameterSpec("titin_a", "log_scale", (-0.5, 0.5)),
        ParameterSpec("titin_b", "log_scale", (-0.3, 0.3)),
        ParameterSpec("titin_rest", "additive", (-30.0, 30.0)),
    ],
    "pca": [
        ParameterSpec("tm_K1", "log_scale", (-2.0, 1.0)),
        ParameterSpec("tm_K2", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_K3", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_coop_magnitude", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_span_base", "log_scale", (-0.7, 0.7)),
        ParameterSpec("tm_span_force50", "negative_log_scale", (-0.7, 0.7)),
        ParameterSpec("tm_span_steep", "log_scale", (-0.7, 0.7)),
    ],
    "kinetics": [
        ParameterSpec("tm_k_12", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_k_23", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_k_34", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_k_41", "log_scale", (-1.5, 1.5)),
    ],
    # Joint TM-only calibration. Titin, crossbridge kinetics, and all LDA
    # parameters remain fixed at their base values. tm_K4 is unused by the
    # current transition matrix and is therefore not identifiable.
    "tm_all": [
        ParameterSpec("tm_k_12", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_k_23", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_k_34", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_k_41", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_K1", "log_scale", (-2.0, 1.0)),
        ParameterSpec("tm_K2", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_K3", "log_scale", (-1.5, 1.5)),
        ParameterSpec("tm_coop_magnitude", "log_absolute", (0.0, 1.5)),
        ParameterSpec("tm_span_base", "log_scale", (-0.7, 0.7)),
        ParameterSpec("tm_span_force50", "negative_log_scale", (-0.7, 0.7)),
        ParameterSpec("tm_span_steep", "log_scale", (-0.7, 0.7)),
    ],
    "lda": [
        ParameterSpec("xb_lda_gain", "log_absolute", (-2.0, 2.0)),
        ParameterSpec("xb_lda_strain_threshold", "log_absolute", (-1.5, 0.7)),
    ],
}


def params_from_theta(base_params, specs: list[ParameterSpec], theta: np.ndarray):
    updates = {spec.name: spec.decode(float(value), float(getattr(base_params, spec.name)))
               for spec, value in zip(specs, theta)}
    # Physiological fitting always retains the LDA mechanism.
    updates["xb_lda_enabled"] = 1.0
    return base_params.copy(**updates), updates


def run_steady(topology, static, params, pca: float, sl_um: float):
    result = run(
        topology,
        pCa=float(pca),
        z_line=z_line_from_sl(sl_um),
        lattice_spacing=LATTICE_SPACING_NM,
        duration_ms=STEADY_DURATION_MS,
        dt=DT_MS,
        replicates=REPLICATES,
        rng_seed=RNG_SEED,
        static_params=static,
        dynamic_params=params,
    )
    require_solver_convergence(result)
    return result


def simulate_passive(topology, static, params, passive: pd.DataFrame) -> np.ndarray:
    values = []
    for row in passive.itertuples(index=False):
        result = run_steady(topology, static, params, pca=9.0, sl_um=row.sl_um)
        values.append(active_force_from_result(result)[0])
    values = np.asarray(values)
    return values - values[0] if PASSIVE_FORCE_RELATIVE_TO_FIRST_POINT else values


def simulate_pca(topology, static, params, pca_data: pd.DataFrame) -> np.ndarray:
    predictions = np.empty(len(pca_data), dtype=float)
    for sl_um, group in pca_data.groupby("sl_um", sort=False):
        # Include pCa 9.0 in every group so the active-force baseline is
        # generated with the same parameter set and sarcomere length.
        pca_points = group["pca"].to_numpy(dtype=float)
        pca_with_baseline = np.concatenate([pca_points, [9.0]])
        result = run(
            topology,
            pCa=pca_with_baseline.tolist(),
            z_line=z_line_from_sl(sl_um),
            lattice_spacing=LATTICE_SPACING_NM,
            duration_ms=STEADY_DURATION_MS,
            dt=DT_MS,
            replicates=REPLICATES,
            rng_seed=RNG_SEED,
            static_params=static,
            dynamic_params=params,
        )
        require_solver_convergence(result)
        force = active_force_from_result(result)
        baseline = force[-1]
        predicted = force[:-1] - baseline if PCA_FORCE_IS_ACTIVE else force[:-1]
        predictions[group.index.to_numpy()] = predicted
    return predictions


def redevelopment_curve(time_ms: np.ndarray, ktr_s: float, f0: float, fss: float) -> np.ndarray:
    return fss - (fss - f0) * np.exp(-ktr_s * time_ms / 1000.0)


def simulate_ktr(topology, static, params, ktr_data: pd.DataFrame) -> np.ndarray:
    """Estimate ktr from a one-step length release followed by restretch.

    This is only comparable to experiment when KTR_RELEASE_NM and the timing
    match the experimental quick-release/restretch protocol.
    """
    n_steps = int(KTR_DURATION_MS / DT_MS)
    release_step = int(KTR_RELEASE_AT_MS / DT_MS)
    restretch_step = release_step + 1
    if not 1 <= release_step < n_steps - 2:
        raise ValueError("KTR_RELEASE_AT_MS must leave room for force redevelopment.")

    values = []
    for row in ktr_data.itertuples(index=False):
        z0 = z_line_from_sl(row.sl_um)
        pca_trace = np.full(n_steps, float(row.pca), dtype=np.float32)
        z_trace = np.full(n_steps, z0, dtype=np.float32)
        z_trace[release_step] = z0 - KTR_RELEASE_NM

        result = run(
            topology,
            pCa=pca_trace,
            z_line=z_trace,
            lattice_spacing=LATTICE_SPACING_NM,
            duration_ms=KTR_DURATION_MS,
            dt=DT_MS,
            replicates=REPLICATES,
            rng_seed=RNG_SEED,
            static_params=static,
            dynamic_params=params,
        )
        require_solver_convergence(result)
        trace = force_trace_from_result(result)
        start = restretch_step + int(KTR_FIT_START_AFTER_RESTRETCH_MS / DT_MS)
        stop = n_steps - int(KTR_FIT_END_BEFORE_END_MS / DT_MS)
        time_ms = np.arange(start, stop, dtype=float) * DT_MS
        time_ms -= time_ms[0]
        f0 = float(trace[start])
        fss = float(trace[-int(STEADY_WINDOW_MS / DT_MS):].mean())

        fit, _ = curve_fit(
            lambda t, ktr: redevelopment_curve(t, ktr, f0, fss),
            time_ms,
            trace[start:stop],
            p0=[5.0],
            bounds=([1e-4], [200.0]),
            maxfev=10_000,
        )
        values.append(float(fit[0]))
    return np.asarray(values)


def weighted_sse(predicted: np.ndarray, observed: np.ndarray, sem: np.ndarray) -> float:
    return float(np.sum(((predicted - observed) / sem) ** 2))


def pca_sse(predicted: np.ndarray, pca_data: pd.DataFrame) -> float:
    """Compare pCa curves, optionally removing incompatible absolute force scales."""
    observed = pca_data["force"].to_numpy(dtype=float)
    sem = pca_data["sem"].to_numpy(dtype=float)
    if not NORMALIZE_PCA_FORCE:
        return weighted_sse(predicted, observed, sem)

    total = 0.0
    for _sl_um, group in pca_data.groupby("sl_um", sort=False):
        indices = group.index.to_numpy()
        observed_group = observed[indices]
        predicted_group = predicted[indices]
        observed_scale = float(np.max(np.abs(observed_group)))
        predicted_scale = float(np.max(np.abs(predicted_group)))
        if observed_scale <= 0 or predicted_scale <= 0:
            return 1e30
        total += weighted_sse(
            predicted_group / predicted_scale,
            observed_group / observed_scale,
            sem[indices] / observed_scale,
        )
    return total


def build_objective(stage: str, topology, static, base_params, frames: dict[str, pd.DataFrame]) -> Callable[[np.ndarray], float]:
    specs = STAGES[stage]

    def objective(theta: np.ndarray) -> float:
        params, _ = params_from_theta(base_params, specs, theta)
        try:
            if stage == "passive":
                data = frames["passive"]
                pred = simulate_passive(topology, static, params, data)
                return weighted_sse(pred, data["force"].to_numpy(), data["sem"].to_numpy())
            if stage in {"pca", "lda"}:
                data = frames["pca"]
                pred = simulate_pca(topology, static, params, data)
                return pca_sse(pred, data)
            if stage == "kinetics":
                data = frames["ktr"]
                pred = simulate_ktr(topology, static, params, data)
                return weighted_sse(pred, data["ktr_s"].to_numpy(), data["sem"].to_numpy())
            if stage == "tm_all":
                pca_data = frames["pca"]
                ktr_data = frames["ktr"]
                pca_pred = simulate_pca(topology, static, params, pca_data)
                ktr_pred = simulate_ktr(topology, static, params, ktr_data)
                return (
                    pca_sse(pca_pred, pca_data)
                    + weighted_sse(ktr_pred, ktr_data["ktr_s"].to_numpy(), ktr_data["sem"].to_numpy())
                )
        except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            print(f"Invalid candidate: {error}")
            return 1e30
        raise ValueError(f"Unknown stage {stage}")

    return objective


def read_start_params(base_params, json_path: str | None):
    if json_path is None:
        return base_params
    with open(json_path, encoding="utf-8") as handle:
        values = json.load(handle)["parameter_values"]
    valid = {name: value for name, value in values.items() if hasattr(base_params, name)}
    return base_params.copy(**valid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES.keys())
    parser.add_argument("--start-json", help="Result JSON from the preceding stage.")
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--popsize", type=int, default=POPSIZE)
    parser.add_argument("--workbook", type=Path, help="Raw PTP-style Excel workbook instead of CSV files.")
    parser.add_argument("--passive-sheet", help="Workbook sheet with %% stretch in col. 1 and force replicates thereafter.")
    parser.add_argument(
        "--pca-sheet",
        action="append",
        help="Repeat for each pCa sheet, e.g. --pca-sheet 'pCa SL2.0:2.0'.",
    )
    parser.add_argument(
        "--ktr-sheet",
        action="append",
        help="Optional repeatable mapping, e.g. --ktr-sheet 'ktr SL2.0:2.0'.",
    )
    parser.add_argument(
        "--reference-sl-um",
        type=float,
        default=REFERENCE_SL_FOR_STRETCH_UM,
        help="SL before the percentage stretch in the passive sheet.",
    )
    parser.add_argument(
        "--force-to-pn",
        type=float,
        help="Multiply workbook force values by this factor to convert them to model pN.",
    )
    args = parser.parse_args()

    if args.workbook:
        if not args.passive_sheet or args.force_to_pn is None:
            parser.error("--workbook requires --passive-sheet and --force-to-pn.")
        frames = load_frames_from_workbook(
            args.workbook,
            args.passive_sheet,
            parse_sheet_mapping(args.pca_sheet),
            parse_sheet_mapping(args.ktr_sheet),
            args.force_to_pn,
            args.reference_sl_um,
        )
        if args.stage in {"pca", "lda", "tm_all"} and frames["pca"] is None:
            parser.error(f"--stage {args.stage} requires at least one --pca-sheet mapping.")
        if args.stage in {"kinetics", "tm_all"} and frames["ktr"] is None:
            parser.error(f"--stage {args.stage} requires at least one --ktr-sheet mapping.")
        if frames["ktr"] is None and args.stage not in {"kinetics", "tm_all"}:
            frames["ktr"] = load_csv(KTR_FILE, {"sl_um", "pca", "ktr_s", "sem"})
    else:
        frames = {"passive": None, "pca": None, "ktr": None}
        if args.stage == "passive":
            frames["passive"] = load_csv(PASSIVE_FILE, {"sl_um", "force", "sem"})
        elif args.stage in {"pca", "lda"}:
            frames["pca"] = load_csv(PCA_FILE, {"sl_um", "pca", "force", "sem"})
        elif args.stage == "kinetics":
            frames["ktr"] = load_csv(KTR_FILE, {"sl_um", "pca", "ktr_s", "sem"})
        elif args.stage == "tm_all":
            frames["pca"] = load_csv(PCA_FILE, {"sl_um", "pca", "force", "sem"})
            frames["ktr"] = load_csv(KTR_FILE, {"sl_um", "pca", "ktr_s", "sem"})

    static, dynamic = get_cardiac_params()
    base_params = read_start_params(dynamic, args.start_json)
    topology = jax.device_put(SarcTopology.create(
        nrows=NROWS,
        ncols=NCOLS,
        static_params=static,
        dynamic_params=base_params,
    ))

    specs = STAGES[args.stage]
    objective = build_objective(args.stage, topology, static, base_params, frames)
    result = differential_evolution(
        objective,
        bounds=[spec.bounds for spec in specs],
        strategy="best1bin",
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=0.02,
        polish=False,
        seed=RNG_SEED,
        workers=1,
        updating="immediate",
        disp=True,
    )

    fitted_params, updates = params_from_theta(base_params, specs, result.x)
    values = {name: float(getattr(fitted_params, name)) for name in fitted_params.__slots__}
    output = {
        "stage": args.stage,
        "weighted_sse": float(result.fun),
        "theta": result.x.tolist(),
        "stage_updates": updates,
        "parameter_values": values,
    }
    output_path = Path(f"fit_{args.stage}.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nBest weighted SSE: {result.fun:.4f}")
    print(f"Wrote: {output_path.resolve()}")
    print("\nUse these fitted values:")
    for name, value in updates.items():
        print(f"  {name}={value:.8g}")


if __name__ == "__main__":
    main()
