"""WT-baseline calibration and Titin perturbation analysis for multifil_jax.

Scientific design
-----------------
1. Calibrate ONE WT baseline model.
   - WT passive force-length data: full passive stretch curve.
   - WT pCa-force data: ONLY at FIT_SL_UM (default 2.0 um).
   - WT ktr data: ONLY at FIT_SL_UM (default 2.0 um).
   - WT SL 2.3 active-mechanics data are NOT used for fitting.

2. Validate the fitted WT baseline without changing parameters.
   - Predict WT pCa-force and ktr at SL 2.0 and 2.3.
   - WT SL 2.3 therefore acts as an out-of-sample LDA validation.

3. Test the Titin mechanism without fitting KO active-mechanics data.
   - Start from the frozen WT fitted parameter set.
   - Perturb ONLY titin_a, titin_b, or titin_rest.
   - Predict passive force, pCa-force, and ktr at SL 2.0 and 2.3.
   - KO data, if present, are used only as an external comparison target.
   - No KO data enter differential_evolution.

Expected directory layout
-------------------------
data/
  WT/
    passive_length.csv   # sl_um,force,sem
    pca_force.csv        # sl_um,pca,force,sem
    ktr.csv              # sl_um,pca,ktr_s,sem
  KO/
    passive_length.csv   # optional, same headers
    pca_force.csv        # optional, same headers
    ktr.csv              # optional, same headers

Typical usage
-------------
Quick WT calibration test:
  python fit_wt_titin_validation.py fit-wt --maxiter 2 --popsize 3

Full WT calibration:
  python fit_wt_titin_validation.py fit-wt --maxiter 80 --popsize 8

WT validation after fitting:
  python fit_wt_titin_validation.py validate-wt \
      --params-json results/fit_WT_baseline.json

One-at-a-time Titin perturbation scan:
  python fit_wt_titin_validation.py titin-oat \
      --params-json results/fit_WT_baseline.json

Notes
-----
- All model simulations keep xb_lda_enabled = 1.
- pCa-force is normalized by curve amplitude by default, matching the original
  fitting logic when experimental PTP force and model pN are not on the same
  absolute scale.
- The quick-release/restretch constants MUST match the experiment before ktr
  comparisons are interpreted mechanistically.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import jax
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


# =============================================================================
# User configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

WT_DIR = DATA_DIR / "WT"
KO_DIR = DATA_DIR / "KO"

WT_PASSIVE_FILE = WT_DIR / "passive_length.csv"
WT_PCA_FILE = WT_DIR / "pca_force.csv"
WT_KTR_FILE = WT_DIR / "ktr.csv"

KO_PASSIVE_FILE = KO_DIR / "passive_length.csv"
KO_PCA_FILE = KO_DIR / "pca_force.csv"
KO_KTR_FILE = KO_DIR / "ktr.csv"

# Calibration sarcomere length for active mechanics.
FIT_SL_UM = 2.0
VALIDATION_SLS_UM = (2.0, 2.3)

# Model size / simulation settings.
NROWS = 2
NCOLS = 2
DT_MS = 1.0
STEADY_DURATION_MS = 1000.0
STEADY_WINDOW_MS = 600.0
REPLICATES = 8
RNG_SEED = 12345
MAX_FIT_SOLVER_RESIDUAL_PN = 10.0

# Experimental SL -> model Z-line mapping.
Z_AT_SL_2_0_NM = 1000.0
NM_Z_PER_UM_SL = 500.0
LATTICE_SPACING_NM = 14.0

# Passive-force convention.
PASSIVE_FORCE_RELATIVE_TO_FIRST_POINT = True

# pCa-force convention.
PCA_FORCE_IS_ACTIVE = True
NORMALIZE_PCA_FORCE = True

# Quick-release/restretch protocol.
KTR_DURATION_MS = 600.0
KTR_RELEASE_AT_MS = 20
KTR_RELEASE_NM = 4.0
KTR_FIT_START_AFTER_RESTRETCH_MS = 3
KTR_FIT_END_BEFORE_END_MS = 100

# Joint loss weights. Each component is a mean squared SEM-normalized residual,
# so equal weights give approximately equal influence to passive, pCa, and ktr.
W_PASSIVE = 1.0
W_PCA = 1.0
W_KTR = 1.0

# Differential-evolution defaults.
MAXITER = 24
POPSIZE = 8

# One-at-a-time Titin perturbations.
TITIN_A_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
TITIN_B_SCALES = (0.70, 0.85, 1.00, 1.15, 1.30, 1.50)
TITIN_REST_SHIFTS_NM = (-30.0, -15.0, 0.0, 15.0, 30.0)


# =============================================================================
# Basic utilities
# =============================================================================

def z_line_from_sl(sl_um: float) -> float:
    return Z_AT_SL_2_0_NM + NM_Z_PER_UM_SL * (float(sl_um) - 2.0)


def load_csv(path: Path, required_columns: set[str], optional: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        if optional:
            return None
        raise FileNotFoundError(
            f"Missing {path}. Expected columns: {sorted(required_columns)}"
        )

    frame = pd.read_csv(path)
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if "sem" in frame.columns and (frame["sem"] <= 0).any():
        raise ValueError(f"All SEM values in {path} must be positive.")

    sort_columns = [c for c in ("sl_um", "pca") if c in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns)

    return frame.reset_index(drop=True)


def filter_sl(frame: pd.DataFrame, sl_um: float) -> pd.DataFrame:
    out = frame[np.isclose(frame["sl_um"].to_numpy(dtype=float), float(sl_um))].copy()
    if out.empty:
        raise ValueError(f"No data found at SL={sl_um:.3f} um.")
    return out.reset_index(drop=True)


def subset_available_sls(frame: pd.DataFrame | None, sls: Iterable[float]) -> pd.DataFrame | None:
    if frame is None:
        return None
    keep = np.zeros(len(frame), dtype=bool)
    values = frame["sl_um"].to_numpy(dtype=float)
    for sl in sls:
        keep |= np.isclose(values, sl)
    return frame.loc[keep].copy().reset_index(drop=True)


def safe_float(x) -> float:
    x = float(x)
    if not np.isfinite(x):
        raise FloatingPointError(f"Non-finite value encountered: {x}")
    return x


# =============================================================================
# Model result helpers
# =============================================================================

def active_force_from_result(result, window_ms: float = STEADY_WINDOW_MS) -> np.ndarray:
    """Return one steady force per sweep point, averaged over replicates."""
    force = np.asarray(result.axial_force)
    n_window = max(1, int(window_ms / result.dt))
    steady = force[..., -n_window:].mean(axis=-1)

    if steady.ndim == 1:
        # scalar sweep -> shape (replicates,)
        return np.array([steady.mean()])

    # sweep -> (sweep, replicates) -> (sweep,)
    return steady.mean(axis=-1)


def force_trace_from_result(result) -> np.ndarray:
    """Mean force trace over replicates for a scalar pCa/z-line simulation."""
    force = np.asarray(result.axial_force)
    if force.ndim != 2:
        raise ValueError(
            f"Expected force shape (replicates, time), got {force.shape}"
        )
    return force.mean(axis=0)


def require_solver_convergence(result) -> None:
    residual = np.asarray(result.metrics["solver_residual"])
    max_residual = float(np.max(np.abs(residual)))
    if not np.isfinite(max_residual) or max_residual > MAX_FIT_SOLVER_RESIDUAL_PN:
        raise FloatingPointError(
            f"solver residual {max_residual:.3g} pN exceeds "
            f"{MAX_FIT_SOLVER_RESIDUAL_PN} pN"
        )


# =============================================================================
# Parameterization
# =============================================================================

@dataclass(frozen=True)
class ParameterSpec:
    name: str
    transform: str
    bounds: tuple[float, float]

    def decode(self, x: float, base_value: float) -> float:
        if self.transform == "log_scale":
            return base_value * 10.0 ** x
        if self.transform == "negative_log_scale":
            # Useful when the baseline value is negative; multiplicative scaling
            # preserves its sign.
            return base_value * 10.0 ** x
        if self.transform == "additive":
            return base_value + x
        if self.transform == "log_absolute":
            return 10.0 ** x
        raise ValueError(f"Unknown transform: {self.transform}")


# WT baseline fit only.
# KO data never enter this parameter optimization.
WT_BASELINE_SPECS: list[ParameterSpec] = [
    # Titin / passive mechanics
    ParameterSpec("titin_a", "log_scale", (-0.5, 0.5)),
    ParameterSpec("titin_b", "log_scale", (-0.3, 0.3)),
    ParameterSpec("titin_rest", "additive", (-30.0, 30.0)),

    # Thin-filament kinetic rates
    ParameterSpec("tm_k_12", "log_scale", (-1.5, 1.5)),
    ParameterSpec("tm_k_23", "log_scale", (-1.5, 1.5)),
    ParameterSpec("tm_k_34", "log_scale", (-1.5, 1.5)),
    ParameterSpec("tm_k_41", "log_scale", (-1.5, 1.5)),

    # Thin-filament equilibrium / Ca sensitivity / cooperativity
    ParameterSpec("tm_K1", "log_scale", (-2.0, 1.0)),
    ParameterSpec("tm_K2", "log_scale", (-1.5, 1.5)),
    ParameterSpec("tm_K3", "log_scale", (-1.5, 1.5)),
    ParameterSpec("tm_coop_magnitude", "log_absolute", (0.0, 1.5)),

    # Force-dependent thin-filament activation
    ParameterSpec("tm_span_base", "log_scale", (-0.7, 0.7)),
    ParameterSpec("tm_span_force50", "negative_log_scale", (-0.7, 0.7)),
    ParameterSpec("tm_span_steep", "log_scale", (-0.7, 0.7)),
]


def params_from_theta(base_params, specs: list[ParameterSpec], theta: np.ndarray):
    updates = {
        spec.name: spec.decode(float(value), float(getattr(base_params, spec.name)))
        for spec, value in zip(specs, theta)
    }
    updates["xb_lda_enabled"] = 1.0
    return base_params.copy(**updates), updates


def params_to_dict(params) -> dict[str, float]:
    return {
        name: float(getattr(params, name))
        for name in params.__slots__
    }


def load_params_json(base_params, json_path: Path):
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    values = payload["parameter_values"]
    valid = {
        name: value
        for name, value in values.items()
        if hasattr(base_params, name)
    }
    valid["xb_lda_enabled"] = 1.0
    return base_params.copy(**valid), payload


# =============================================================================
# Simulation functions
# =============================================================================

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


def simulate_passive(
    topology,
    static,
    params,
    passive: pd.DataFrame,
) -> np.ndarray:
    values = []

    for row in passive.itertuples(index=False):
        result = run_steady(
            topology,
            static,
            params,
            pca=9.0,
            sl_um=row.sl_um,
        )
        values.append(active_force_from_result(result)[0])

    values = np.asarray(values, dtype=float)

    if PASSIVE_FORCE_RELATIVE_TO_FIRST_POINT:
        values = values - values[0]

    return values


def simulate_pca(
    topology,
    static,
    params,
    pca_data: pd.DataFrame,
) -> np.ndarray:
    """Simulate active pCa-force values for one or more sarcomere lengths."""
    predictions = np.empty(len(pca_data), dtype=float)

    for sl_um, group in pca_data.groupby("sl_um", sort=False):
        pca_points = group["pca"].to_numpy(dtype=float)

        # Same-parameter, same-SL pCa 9 baseline.
        pca_with_baseline = np.concatenate([pca_points, [9.0]])

        result = run(
            topology,
            pCa=pca_with_baseline.tolist(),
            z_line=z_line_from_sl(float(sl_um)),
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


def redevelopment_curve(
    time_ms: np.ndarray,
    ktr_s: float,
    f0: float,
    fss: float,
) -> np.ndarray:
    return fss - (fss - f0) * np.exp(-ktr_s * time_ms / 1000.0)


def simulate_ktr(
    topology,
    static,
    params,
    ktr_data: pd.DataFrame,
) -> np.ndarray:
    """Estimate ktr after one-timestep release/restretch."""
    n_steps = int(KTR_DURATION_MS / DT_MS)
    release_step = int(KTR_RELEASE_AT_MS / DT_MS)
    restretch_step = release_step + 1

    if not 1 <= release_step < n_steps - 2:
        raise ValueError(
            "KTR_RELEASE_AT_MS must leave room for redevelopment."
        )

    values = []

    for row in ktr_data.itertuples(index=False):
        z0 = z_line_from_sl(row.sl_um)

        pca_trace = np.full(
            n_steps,
            float(row.pca),
            dtype=np.float32,
        )
        z_trace = np.full(
            n_steps,
            z0,
            dtype=np.float32,
        )
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

        start = (
            restretch_step
            + int(KTR_FIT_START_AFTER_RESTRETCH_MS / DT_MS)
        )
        stop = (
            n_steps
            - int(KTR_FIT_END_BEFORE_END_MS / DT_MS)
        )

        if stop <= start + 3:
            raise ValueError("ktr fitting window is too short.")

        time_ms = np.arange(start, stop, dtype=float) * DT_MS
        time_ms -= time_ms[0]

        f0 = float(trace[start])
        n_ss = max(1, int(STEADY_WINDOW_MS / DT_MS))
        fss = float(trace[-n_ss:].mean())

        fit, _ = curve_fit(
            lambda t, ktr: redevelopment_curve(t, ktr, f0, fss),
            time_ms,
            trace[start:stop],
            p0=[5.0],
            bounds=([1e-4], [200.0]),
            maxfev=10_000,
        )

        values.append(float(fit[0]))

    return np.asarray(values, dtype=float)


# =============================================================================
# Loss functions
# =============================================================================

def weighted_mse(
    predicted: np.ndarray,
    observed: np.ndarray,
    sem: np.ndarray,
) -> float:
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    sem = np.asarray(sem, dtype=float)

    if (
        predicted.shape != observed.shape
        or observed.shape != sem.shape
    ):
        raise ValueError(
            f"Shape mismatch: pred={predicted.shape}, "
            f"obs={observed.shape}, sem={sem.shape}"
        )

    if np.any(sem <= 0):
        raise ValueError("SEM values must be positive.")

    residual = (predicted - observed) / sem
    return float(np.mean(residual ** 2))


def pca_weighted_mse(
    predicted: np.ndarray,
    pca_data: pd.DataFrame,
) -> float:
    observed = pca_data["force"].to_numpy(dtype=float)
    sem = pca_data["sem"].to_numpy(dtype=float)

    if not NORMALIZE_PCA_FORCE:
        return weighted_mse(predicted, observed, sem)

    losses = []

    for _, group in pca_data.groupby("sl_um", sort=False):
        idx = group.index.to_numpy()

        obs = observed[idx]
        pred = predicted[idx]
        sem_group = sem[idx]

        obs_scale = float(np.max(np.abs(obs)))
        pred_scale = float(np.max(np.abs(pred)))

        if obs_scale <= 0 or pred_scale <= 0:
            return 1e30

        normalized_obs = obs / obs_scale
        normalized_pred = pred / pred_scale
        normalized_sem = sem_group / obs_scale

        residual = (
            normalized_pred - normalized_obs
        ) / normalized_sem

        losses.extend((residual ** 2).tolist())

    return float(np.mean(losses))


def evaluate_losses(
    topology,
    static,
    params,
    passive_data: pd.DataFrame,
    pca_data: pd.DataFrame,
    ktr_data: pd.DataFrame,
) -> dict[str, float]:
    passive_pred = simulate_passive(
        topology, static, params, passive_data
    )
    pca_pred = simulate_pca(
        topology, static, params, pca_data
    )
    ktr_pred = simulate_ktr(
        topology, static, params, ktr_data
    )

    passive_loss = weighted_mse(
        passive_pred,
        passive_data["force"].to_numpy(dtype=float),
        passive_data["sem"].to_numpy(dtype=float),
    )
    pca_loss = pca_weighted_mse(
        pca_pred,
        pca_data,
    )
    ktr_loss = weighted_mse(
        ktr_pred,
        ktr_data["ktr_s"].to_numpy(dtype=float),
        ktr_data["sem"].to_numpy(dtype=float),
    )

    total = (
        W_PASSIVE * passive_loss
        + W_PCA * pca_loss
        + W_KTR * ktr_loss
    )

    return {
        "total": float(total),
        "passive": float(passive_loss),
        "pca": float(pca_loss),
        "ktr": float(ktr_loss),
    }


# =============================================================================
# WT calibration
# =============================================================================

def load_wt_calibration_frames() -> dict[str, pd.DataFrame]:
    passive = load_csv(
        WT_PASSIVE_FILE,
        {"sl_um", "force", "sem"},
    )
    pca_all = load_csv(
        WT_PCA_FILE,
        {"sl_um", "pca", "force", "sem"},
    )
    ktr_all = load_csv(
        WT_KTR_FILE,
        {"sl_um", "pca", "ktr_s", "sem"},
    )

    # Critical anti-leakage rule:
    # active WT calibration uses ONLY SL 2.0.
    pca_fit = filter_sl(pca_all, FIT_SL_UM)
    ktr_fit = filter_sl(ktr_all, FIT_SL_UM)

    return {
        "passive": passive,
        "pca": pca_fit,
        "ktr": ktr_fit,
        "pca_all": pca_all,
        "ktr_all": ktr_all,
    }


def build_wt_objective(
    topology,
    static,
    base_params,
    frames: dict[str, pd.DataFrame],
) -> Callable[[np.ndarray], float]:
    specs = WT_BASELINE_SPECS
    counter = {"n": 0}

    def objective(theta: np.ndarray) -> float:
        counter["n"] += 1
        params, _ = params_from_theta(
            base_params,
            specs,
            theta,
        )

        try:
            losses = evaluate_losses(
                topology,
                static,
                params,
                frames["passive"],
                frames["pca"],
                frames["ktr"],
            )

            print(
                f"eval={counter['n']:05d} | "
                f"total={losses['total']:.5g} | "
                f"passive={losses['passive']:.5g} | "
                f"pCa={losses['pca']:.5g} | "
                f"ktr={losses['ktr']:.5g}",
                flush=True,
            )

            return losses["total"]

        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            print(
                f"eval={counter['n']:05d} | invalid candidate: {error}",
                flush=True,
            )
            return 1e30

    return objective


def fit_wt(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_wt_calibration_frames()

    static, dynamic = get_cardiac_params()
    base_params = dynamic.copy(xb_lda_enabled=1.0)

    topology = jax.device_put(
        SarcTopology.create(
            nrows=NROWS,
            ncols=NCOLS,
            static_params=static,
            dynamic_params=base_params,
        )
    )

    objective = build_wt_objective(
        topology,
        static,
        base_params,
        frames,
    )

    result = differential_evolution(
        objective,
        bounds=[spec.bounds for spec in WT_BASELINE_SPECS],
        strategy="best1bin",
        maxiter=args.maxiter,
        popsize=args.popsize,
        tol=args.tol,
        polish=False,
        seed=args.seed,
        workers=1,
        updating="immediate",
        disp=True,
    )

    fitted_params, updates = params_from_theta(
        base_params,
        WT_BASELINE_SPECS,
        result.x,
    )

    best_losses = evaluate_losses(
        topology,
        static,
        fitted_params,
        frames["passive"],
        frames["pca"],
        frames["ktr"],
    )

    output = {
        "model_role": "WT baseline calibration only",
        "fit_group": "WT",
        "active_fit_sl_um": FIT_SL_UM,
        "wt_sl_2_3_used_for_fitting": False,
        "ko_used_for_fitting": False,
        "loss_weights": {
            "passive": W_PASSIVE,
            "pca": W_PCA,
            "ktr": W_KTR,
        },
        "best_loss": best_losses,
        "theta": result.x.tolist(),
        "stage_updates": {
            k: float(v) for k, v in updates.items()
        },
        "parameter_values": params_to_dict(fitted_params),
        "optimizer": {
            "maxiter": int(args.maxiter),
            "popsize": int(args.popsize),
            "tol": float(args.tol),
            "seed": int(args.seed),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
        },
        "simulation": {
            "nrows": NROWS,
            "ncols": NCOLS,
            "dt_ms": DT_MS,
            "steady_duration_ms": STEADY_DURATION_MS,
            "steady_window_ms": STEADY_WINDOW_MS,
            "replicates": REPLICATES,
            "lattice_spacing_nm": LATTICE_SPACING_NM,
            "rng_seed": RNG_SEED,
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = RESULTS_DIR / output_path

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\n=== WT BASELINE FIT COMPLETE ===")
    print(f"Best total loss: {best_losses['total']:.6g}")
    print(f"Passive loss:    {best_losses['passive']:.6g}")
    print(f"pCa loss:        {best_losses['pca']:.6g}")
    print(f"ktr loss:        {best_losses['ktr']:.6g}")
    print(f"Wrote: {output_path.resolve()}")
    print("\nFitted WT parameters:")
    for name, value in updates.items():
        print(f"  {name} = {value:.8g}")


# =============================================================================
# Prediction / validation output
# =============================================================================

def normalized_curve(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values)))
    if scale <= 0:
        return np.full_like(values, np.nan)
    return values / scale


def make_prediction_tables(
    topology,
    static,
    params,
    passive_data: pd.DataFrame | None,
    pca_data: pd.DataFrame | None,
    ktr_data: pd.DataFrame | None,
    label: str,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    if passive_data is not None:
        pred = simulate_passive(
            topology,
            static,
            params,
            passive_data,
        )
        table = passive_data.copy()
        table["predicted_force"] = pred
        table["dataset"] = label
        tables["passive"] = table

    if pca_data is not None:
        pred = simulate_pca(
            topology,
            static,
            params,
            pca_data,
        )
        table = pca_data.copy()
        table["predicted_force"] = pred

        table["observed_force_norm"] = np.nan
        table["predicted_force_norm"] = np.nan

        for _, group in table.groupby("sl_um", sort=False):
            idx = group.index.to_numpy()
            table.loc[idx, "observed_force_norm"] = normalized_curve(
                table.loc[idx, "force"].to_numpy(dtype=float)
            )
            table.loc[idx, "predicted_force_norm"] = normalized_curve(
                table.loc[idx, "predicted_force"].to_numpy(dtype=float)
            )

        table["dataset"] = label
        tables["pca"] = table

    if ktr_data is not None:
        pred = simulate_ktr(
            topology,
            static,
            params,
            ktr_data,
        )
        table = ktr_data.copy()
        table["predicted_ktr_s"] = pred
        table["dataset"] = label
        tables["ktr"] = table

    return tables


def validation_scores(
    tables: dict[str, pd.DataFrame],
) -> dict[str, float]:
    scores: dict[str, float] = {}

    if "passive" in tables:
        table = tables["passive"]
        scores["passive"] = weighted_mse(
            table["predicted_force"].to_numpy(dtype=float),
            table["force"].to_numpy(dtype=float),
            table["sem"].to_numpy(dtype=float),
        )

    if "pca" in tables:
        table = tables["pca"]
        scores["pca"] = pca_weighted_mse(
            table["predicted_force"].to_numpy(dtype=float),
            table,
        )

    if "ktr" in tables:
        table = tables["ktr"]
        scores["ktr"] = weighted_mse(
            table["predicted_ktr_s"].to_numpy(dtype=float),
            table["ktr_s"].to_numpy(dtype=float),
            table["sem"].to_numpy(dtype=float),
        )

    return scores


def write_tables(
    tables: dict[str, pd.DataFrame],
    prefix: Path,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)

    for kind, table in tables.items():
        path = prefix.parent / f"{prefix.name}_{kind}.csv"
        table.to_csv(path, index=False)
        print(f"Wrote: {path.resolve()}")


def build_model_from_json(params_json: Path):
    static, dynamic = get_cardiac_params()
    params, payload = load_params_json(
        dynamic,
        params_json,
    )
    topology = jax.device_put(
        SarcTopology.create(
            nrows=NROWS,
            ncols=NCOLS,
            static_params=static,
            dynamic_params=params,
        )
    )
    return static, params, topology, payload


def validate_wt(args) -> None:
    params_json = Path(args.params_json)
    static, params, topology, _ = build_model_from_json(params_json)

    wt_passive = load_csv(
        WT_PASSIVE_FILE,
        {"sl_um", "force", "sem"},
    )
    wt_pca = load_csv(
        WT_PCA_FILE,
        {"sl_um", "pca", "force", "sem"},
    )
    wt_ktr = load_csv(
        WT_KTR_FILE,
        {"sl_um", "pca", "ktr_s", "sem"},
    )

    wt_pca = subset_available_sls(
        wt_pca,
        VALIDATION_SLS_UM,
    )
    wt_ktr = subset_available_sls(
        wt_ktr,
        VALIDATION_SLS_UM,
    )

    tables = make_prediction_tables(
        topology,
        static,
        params,
        wt_passive,
        wt_pca,
        wt_ktr,
        label="WT",
    )

    scores = validation_scores(tables)

    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = RESULTS_DIR / prefix

    write_tables(tables, prefix)

    summary_path = prefix.parent / f"{prefix.name}_summary.json"
    summary = {
        "role": "WT validation; parameters frozen",
        "active_fit_sl_um": FIT_SL_UM,
        "validation_sls_um": list(VALIDATION_SLS_UM),
        "scores": scores,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote: {summary_path.resolve()}")


# =============================================================================
# Titin perturbation experiment
# =============================================================================

@dataclass(frozen=True)
class TitinCondition:
    name: str
    titin_a_scale: float = 1.0
    titin_b_scale: float = 1.0
    titin_rest_shift_nm: float = 0.0


def titin_oat_conditions() -> list[TitinCondition]:
    conditions = [TitinCondition("WT_baseline")]

    for scale in TITIN_A_SCALES:
        if math.isclose(scale, 1.0):
            continue
        conditions.append(
            TitinCondition(
                name=f"titin_a_x{scale:g}",
                titin_a_scale=scale,
            )
        )

    for scale in TITIN_B_SCALES:
        if math.isclose(scale, 1.0):
            continue
        conditions.append(
            TitinCondition(
                name=f"titin_b_x{scale:g}",
                titin_b_scale=scale,
            )
        )

    for shift in TITIN_REST_SHIFTS_NM:
        if math.isclose(shift, 0.0):
            continue
        sign = "plus" if shift > 0 else "minus"
        conditions.append(
            TitinCondition(
                name=f"titin_rest_{sign}{abs(shift):g}nm",
                titin_rest_shift_nm=shift,
            )
        )

    return conditions


def apply_titin_condition(wt_params, condition: TitinCondition):
    return wt_params.copy(
        titin_a=float(wt_params.titin_a) * condition.titin_a_scale,
        titin_b=float(wt_params.titin_b) * condition.titin_b_scale,
        titin_rest=float(wt_params.titin_rest) + condition.titin_rest_shift_nm,
        xb_lda_enabled=1.0,
    )


def optional_ko_frames() -> dict[str, pd.DataFrame | None]:
    passive = load_csv(
        KO_PASSIVE_FILE,
        {"sl_um", "force", "sem"},
        optional=True,
    )
    pca = load_csv(
        KO_PCA_FILE,
        {"sl_um", "pca", "force", "sem"},
        optional=True,
    )
    ktr = load_csv(
        KO_KTR_FILE,
        {"sl_um", "pca", "ktr_s", "sem"},
        optional=True,
    )

    pca = subset_available_sls(
        pca,
        VALIDATION_SLS_UM,
    )
    ktr = subset_available_sls(
        ktr,
        VALIDATION_SLS_UM,
    )

    return {
        "passive": passive,
        "pca": pca,
        "ktr": ktr,
    }


def titin_oat(args) -> None:
    params_json = Path(args.params_json)
    static, wt_params, topology, _ = build_model_from_json(params_json)

    # Use WT experimental coordinates for predictions even if KO files are absent.
    wt_passive = load_csv(
        WT_PASSIVE_FILE,
        {"sl_um", "force", "sem"},
    )
    wt_pca = subset_available_sls(
        load_csv(
            WT_PCA_FILE,
            {"sl_um", "pca", "force", "sem"},
        ),
        VALIDATION_SLS_UM,
    )
    wt_ktr = subset_available_sls(
        load_csv(
            WT_KTR_FILE,
            {"sl_um", "pca", "ktr_s", "sem"},
        ),
        VALIDATION_SLS_UM,
    )

    ko = optional_ko_frames()

    all_passive = []
    all_pca = []
    all_ktr = []
    summary_rows = []

    for i, condition in enumerate(titin_oat_conditions(), start=1):
        print(
            f"\n[{i}] Titin condition: {condition.name}",
            flush=True,
        )

        params = apply_titin_condition(
            wt_params,
            condition,
        )

        # Predictions are generated at WT experimental coordinates.
        wt_tables = make_prediction_tables(
            topology,
            static,
            params,
            wt_passive,
            wt_pca,
            wt_ktr,
            label=condition.name,
        )

        for kind, table in wt_tables.items():
            table = table.copy()
            table["condition"] = condition.name
            table["titin_a"] = float(params.titin_a)
            table["titin_b"] = float(params.titin_b)
            table["titin_rest"] = float(params.titin_rest)

            if kind == "passive":
                all_passive.append(table)
            elif kind == "pca":
                all_pca.append(table)
            elif kind == "ktr":
                all_ktr.append(table)

        row = {
            "condition": condition.name,
            "titin_a_scale": condition.titin_a_scale,
            "titin_b_scale": condition.titin_b_scale,
            "titin_rest_shift_nm": condition.titin_rest_shift_nm,
            "titin_a": float(params.titin_a),
            "titin_b": float(params.titin_b),
            "titin_rest": float(params.titin_rest),
        }

        # KO is comparison-only. These scores are descriptive and are NOT used
        # by any optimizer or to update model parameters.
        if any(v is not None for v in ko.values()):
            ko_tables = make_prediction_tables(
                topology,
                static,
                params,
                ko["passive"],
                ko["pca"],
                ko["ktr"],
                label="KO_external_comparison",
            )
            ko_scores = validation_scores(ko_tables)
            for key, value in ko_scores.items():
                row[f"KO_descriptive_{key}_wmse"] = value

        summary_rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = RESULTS_DIR / prefix

    if all_passive:
        path = prefix.parent / f"{prefix.name}_passive.csv"
        pd.concat(all_passive, ignore_index=True).to_csv(
            path,
            index=False,
        )
        print(f"Wrote: {path.resolve()}")

    if all_pca:
        path = prefix.parent / f"{prefix.name}_pca.csv"
        pd.concat(all_pca, ignore_index=True).to_csv(
            path,
            index=False,
        )
        print(f"Wrote: {path.resolve()}")

    if all_ktr:
        path = prefix.parent / f"{prefix.name}_ktr.csv"
        pd.concat(all_ktr, ignore_index=True).to_csv(
            path,
            index=False,
        )
        print(f"Wrote: {path.resolve()}")

    summary_path = prefix.parent / f"{prefix.name}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
    )
    print(f"Wrote: {summary_path.resolve()}")

    metadata_path = prefix.parent / f"{prefix.name}_metadata.json"
    metadata = {
        "analysis": "one-at-a-time Titin perturbation from frozen WT baseline",
        "ko_used_for_optimization": False,
        "wt_sl_2_3_used_for_baseline_optimization": False,
        "validation_sls_um": list(VALIDATION_SLS_UM),
        "titin_a_scales": list(TITIN_A_SCALES),
        "titin_b_scales": list(TITIN_B_SCALES),
        "titin_rest_shifts_nm": list(TITIN_REST_SHIFTS_NM),
        "note": (
            "Any KO score is descriptive external comparison only. "
            "No KO score updates parameters."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote: {metadata_path.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate WT baseline at SL2.0 active mechanics, validate WT SL2.3, "
            "and test Titin-only perturbations against external KO data."
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    fit = sub.add_parser(
        "fit-wt",
        help="Fit WT baseline using passive + SL2.0 pCa + SL2.0 ktr only.",
    )
    fit.add_argument(
        "--maxiter",
        type=int,
        default=MAXITER,
    )
    fit.add_argument(
        "--popsize",
        type=int,
        default=POPSIZE,
    )
    fit.add_argument(
        "--tol",
        type=float,
        default=0.02,
    )
    fit.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
    )
    fit.add_argument(
        "--output",
        default="fit_WT_baseline.json",
    )
    fit.set_defaults(func=fit_wt)

    validate = sub.add_parser(
        "validate-wt",
        help="Predict WT SL2.0/2.3 with a frozen fitted WT parameter set.",
    )
    validate.add_argument(
        "--params-json",
        required=True,
        type=Path,
    )
    validate.add_argument(
        "--output-prefix",
        default="WT_validation",
    )
    validate.set_defaults(func=validate_wt)

    titin = sub.add_parser(
        "titin-oat",
        help=(
            "One-at-a-time Titin perturbation from frozen WT baseline; "
            "KO is comparison-only."
        ),
    )
    titin.add_argument(
        "--params-json",
        required=True,
        type=Path,
    )
    titin.add_argument(
        "--output-prefix",
        default="titin_OAT",
    )
    titin.set_defaults(func=titin_oat)

    return parser


def main() -> None:
    print("JAX devices:", jax.devices(), flush=True)

    parser = build_parser()
    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
