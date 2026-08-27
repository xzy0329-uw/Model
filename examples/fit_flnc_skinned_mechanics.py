#!/usr/bin/env python3
"""
Fit multifil_jax parameters to FLNC skinned mechanics data.

This script is tailored to the workbook layout in:

    20260504 FLNC flwt skinned mechanics.xlsx

It reads the summary sheets for:
    - Normalized Force
    - pCa50
    - Hill Slope

and fits a small set of SRX/LDA parameters by comparing simulated force-pCa
curves at SL 2.0 and SL 2.3.

Recommended first run:

    python fit_flnc_skinned_mechanics.py summarize

Then test one model evaluation:

    python fit_flnc_skinned_mechanics.py evaluate --genotype WT

Then run a short fit:

    python fit_flnc_skinned_mechanics.py fit --genotype WT --maxiter 8 --popsize 4

Notes:
    - SL 2.0 um is mapped to half-sarcomere z_line = 1000 nm.
    - SL 2.3 um is mapped to half-sarcomere z_line = 1150 nm.
    - By default, the loss uses normalized force-pCa curves, pCa50, and Hill slope.
    - Absolute force is not used by default because the experiment and model may
      be in different force units; add a scale factor before fitting absolute force.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_EXCEL_PATH = (
    "/home/zhiyang_xue/multifil_jax/20260610 FLNC Skinned Mechanics.xlsx"
)

DEFAULT_MODEL_ROOT = "/home/zhiyang_xue/multifil_jax/multifil_jax"

PCA_VALUES = np.array([9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.0, 4.5], dtype=float)
SL_TO_Z_LINE_NM = {2.0: 1000.0, 2.3: 1150.0}


@dataclass(frozen=True)
class FitTarget:
    genotype: str 
    sl_um: float
    pca: float | None
    mean: float
    sd: float
    n: int
    source: str


def clean_numeric(value) -> float:
    """Convert Excel values like '0*' or '5.744*' to floats."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace("*", "").strip()
        if not value:
            return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def mean_sd_n(values: Iterable) -> Tuple[float, float, int]:
    arr = np.array([clean_numeric(v) for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, 0
    sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), sd, int(arr.size)


def read_force_sheet(path: str | Path, sheet_name: str) -> List[FitTarget]:
    """Read force-pCa sheets with WT/KO x SL 2.0/2.3 blocks."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    # Zero-based inclusive column ranges observed in the supplied workbook.
    blocks = [
        ("WT", 2.0, 1, 12),
        ("WT", 2.3, 14, 25),
        ("KO", 2.0, 27, 33),
        ("KO", 2.3, 35, 41),
    ]

    targets: List[FitTarget] = []
    for genotype, sl_um, col0, col1 in blocks:
        for row in range(2, raw.shape[0]):
            pca = clean_numeric(raw.iloc[row, 0])
            if not np.isfinite(pca):
                continue
            mean, sd, n = mean_sd_n(raw.iloc[row, col0 : col1 + 1])
            if n > 0:
                targets.append(
                    FitTarget(
                        genotype=genotype,
                        sl_um=sl_um,
                        pca=float(pca),
                        mean=mean,
                        sd=sd,
                        n=n,
                        source=sheet_name,
                    )
                )
    return targets


def read_summary_sheet(path: str | Path, sheet_name: str) -> List[FitTarget]:
    """Read summary sheets with rows 'SL 2.0' and 'SL 2.3'."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    blocks = [
        ("WT", 1, 12),
        ("KO", 14, 20),
    ]

    targets: List[FitTarget] = []
    for genotype, col0, col1 in blocks:
        for row in range(3, raw.shape[0]):
            label = raw.iloc[row, 0]
            if not isinstance(label, str):
                continue
            match = re.search(r"SL\s*([0-9.]+)", label)
            if not match:
                continue
            sl_um = float(match.group(1))
            mean, sd, n = mean_sd_n(raw.iloc[row, col0 : col1 + 1])
            if n > 0:
                targets.append(
                    FitTarget(
                        genotype=genotype,
                        sl_um=sl_um,
                        pca=None,
                        mean=mean,
                        sd=sd,
                        n=n,
                        source=sheet_name,
                    )
                )
    return targets


def load_targets(path: str | Path) -> Dict[str, List[FitTarget]]:
    return {
        "normalized_force": read_force_sheet(path, "Normalized Force"),
        "absolute_force": read_force_sheet(path, "Absolute Force"),
        "pca50": read_summary_sheet(path, "pCa50"),
        "hill_slope": read_summary_sheet(path, "Hill Slope"),
    }


def filter_targets(
    targets: Iterable[FitTarget],
    genotype: str,
    source: str | None = None,
) -> List[FitTarget]:
    out = [t for t in targets if t.genotype == genotype]
    if source is not None:
        out = [t for t in out if t.source == source]
    return out


def print_target_summary(targets: Dict[str, List[FitTarget]]) -> None:
    for name, rows in targets.items():
        print(f"\n{name}:")
        for genotype in ["WT", "KO"]:
            subset = [t for t in rows if t.genotype == genotype]
            if not subset:
                continue
            print(f"  {genotype}:")
            grouped = {}
            for t in subset:
                key = (t.sl_um, t.pca)
                grouped[key] = t
            for key in sorted(grouped, key=lambda x: (x[0], 999 if x[1] is None else x[1])):
                t = grouped[key]
                pca_text = "" if t.pca is None else f", pCa {t.pca:g}"
                print(
                    f"    SL {t.sl_um:g}{pca_text}: "
                    f"mean={t.mean:.5g}, sd={t.sd:.5g}, n={t.n}"
                )


def add_model_to_path(model_root: str | Path) -> None:
    model_root = str(Path(model_root).expanduser())
    if model_root not in sys.path:
        sys.path.insert(0, model_root)


def build_topology(model_root: str | Path, nrows: int, ncols: int):
    add_model_to_path(model_root)
    import jax
    from multifil_jax.core.params import get_cardiac_params
    from multifil_jax.core.sarc_geometry import SarcTopology

    static, dynamic = get_cardiac_params()
    topo = SarcTopology.create(
        nrows=nrows,
        ncols=ncols,
        static_params=static,
        dynamic_params=dynamic,
    )
    return jax.device_put(topo)


def candidate_vector_to_params(x: np.ndarray, fit_set: str) -> Dict[str, float]:
    """Map optimizer vector to multifil_jax dynamic_params."""
    if fit_set == "lda_srx":
        return {
            "xb_lda_enabled": 1.0,
            "xb_lda_gain": float(x[0]),
            "xb_lda_strain_threshold": float(x[1]),
            "xb_srx_k0": float(x[2]),
            "xb_srx_kmax": float(x[3]),
            "xb_srx_ca50": float(10.0 ** x[4]),
        }
    if fit_set == "tm_coop":
        return {
            "tm_coop_magnitude": float(x[0]),
        }
    if fit_set == "tm_coop_span":
        return {
            "tm_coop_magnitude": float(x[0]),
            "tm_span_base": float(x[1]),
            "tm_span_force50": float(x[2]),
            "tm_span_steep": float(x[3]),
        }
    raise ValueError(f"Unknown fit_set: {fit_set}")


def default_candidate_params() -> Dict[str, float]:
    return {
        "xb_lda_enabled": 1.0,
        "xb_lda_gain": 2.0,
        "xb_lda_strain_threshold": 1.0,
        "xb_srx_k0": 0.1,
        "xb_srx_kmax": 0.4,
        "xb_srx_ca50": 1e-6,
    }


def get_bounds(fit_set: str) -> List[Tuple[float, float]]:
    if fit_set == "lda_srx":
        # Last parameter is log10(xb_srx_ca50), not xb_srx_ca50 itself.
        return [
            (0.0, 10.0),    # xb_lda_gain
            (0.1, 5.0),     # xb_lda_strain_threshold
            (0.001, 0.5),   # xb_srx_k0
            (0.1, 2.0),     # xb_srx_kmax
            (-8.0, -5.0),   # log10(xb_srx_ca50)
        ]
    if fit_set == "tm_coop":
        return [
            (1.0, 500.0),   # tm_coop_magnitude
        ]
    if fit_set == "tm_coop_span":
        return [
            (1.0, 500.0),   # tm_coop_magnitude
            (10.0, 120.0),  # tm_span_base, nm
            (-30.0, 5.0),   # tm_span_force50, pN
            (0.1, 5.0),     # tm_span_steep
        ]
    raise ValueError(f"Unknown fit_set: {fit_set}")


def run_force_pca_protocol(
    topo,
    dynamic_params: Dict[str, float],
    model_root: str | Path,
    duration_ms: float,
    dt: float,
    replicates: int,
    steady_last_ms: float,
    rng_seed: int,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Run pCa curves at SL 2.0 and SL 2.3.

    Returns:
        {
            2.0: {"force": raw force by pCa, "norm_force": normalized force},
            2.3: {...},
        }
    """
    add_model_to_path(model_root)
    from multifil_jax.simulation import run

    out: Dict[float, Dict[str, np.ndarray]] = {}
    for sl_um, z_line in SL_TO_Z_LINE_NM.items():
        result = run(
            topo,
            pCa=list(PCA_VALUES),
            z_line=z_line,
            duration_ms=duration_ms,
            dt=dt,
            replicates=replicates,
            rng_seed=rng_seed,
            dynamic_params=dynamic_params,
        )

        force = np.asarray(result.axial_force)
        n_last = max(1, int(round(steady_last_ms / dt)))
        steady = force[..., -n_last:].mean(axis=-1)

        if steady.ndim == 2:
            # Shape: (pCa, replicates)
            steady = steady.mean(axis=-1)
        elif steady.ndim > 2:
            steady = steady.reshape((steady.shape[0], -1)).mean(axis=-1)

        baseline_idx = int(np.argmin(np.abs(PCA_VALUES - 9.0)))
        max_idx = int(np.argmin(np.abs(PCA_VALUES - 4.5)))

        active_force = steady - steady[baseline_idx]
        max_active_force = max(abs(float(active_force[max_idx])), 1e-12)
        norm_force = active_force / max_active_force

        out[sl_um] = {
            "force": steady,
            "active_force": active_force,
            "norm_force": norm_force,
        }
    return out


def fit_hill_curve(pca: np.ndarray, norm_force: np.ndarray) -> Tuple[float, float]:
    """Fit normalized force to a Hill curve and return pCa50, Hill slope."""
    pca = np.asarray(pca, dtype=float)
    y = np.clip(np.asarray(norm_force, dtype=float), 1e-6, 1.2)

    def hill_fn(pca_values, pca50, hill):
        return 1.0 / (1.0 + 10.0 ** (hill * (pca_values - pca50)))

    try:
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            hill_fn,
            pca,
            y,
            p0=(5.7, 3.0),
            bounds=([4.5, 0.2], [6.5, 10.0]),
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        # Fallback: interpolate pCa50, estimate slope around the midpoint.
        order = np.argsort(y)
        y_sorted = y[order]
        pca_sorted = pca[order]
        pca50 = float(np.interp(0.5, y_sorted, pca_sorted))
        idx = int(np.argmin(np.abs(y - 0.5)))
        idx0 = max(0, idx - 1)
        idx1 = min(len(pca) - 1, idx + 1)
        if idx0 == idx1:
            hill = 3.0
        else:
            dy = y[idx1] - y[idx0]
            dx = pca[idx1] - pca[idx0]
            hill = abs(float(dy / dx)) * 4.0
        return pca50, max(hill, 0.2)


def targets_by_sl(
    targets: Iterable[FitTarget],
    genotype: str,
) -> Dict[float, FitTarget]:
    return {t.sl_um: t for t in targets if t.genotype == genotype}


def normalized_force_loss(
    sim: Dict[float, Dict[str, np.ndarray]],
    targets: List[FitTarget],
    genotype: str,
) -> float:
    rows = [t for t in targets if t.genotype == genotype]
    pca_to_index = {float(p): i for i, p in enumerate(PCA_VALUES)}
    for t in rows:
        if t.sl_um not in sim or t.pca not in pca_to_index:
            continue
        pred = float(sim[t.sl_um]["norm_force"][pca_to_index[t.pca]])
        # SEM weighting with a floor prevents very small SEM rows from dominating.
        sem = t.sd / math.sqrt(t.n) if t.n > 1 else 0.05
        scale = max(sem, 0.04)
        terms.append(((pred - t.mean) / scale) ** 2)
    return float(np.mean(terms)) if terms else 0.0


def pca50_hill_losses(
    sim: Dict[float, Dict[str, np.ndarray]],
    pca50_targets: List[FitTarget],
    hill_targets: List[FitTarget],
    genotype: str,
) -> Tuple[float, float, Dict[float, Tuple[float, float]]]:
    exp_pca50 = targets_by_sl(pca50_targets, genotype)
    exp_hill = targets_by_sl(hill_targets, genotype)

    fitted: Dict[float, Tuple[float, float]] = {}
    pca50_terms = []
    hill_terms = []
    for sl_um, sim_data in sim.items():
        sim_pca50, sim_hill = fit_hill_curve(PCA_VALUES, sim_data["norm_force"])
        fitted[sl_um] = (sim_pca50, sim_hill)

        if sl_um in exp_pca50:
            pca50_terms.append(((sim_pca50 - exp_pca50[sl_um].mean) / 0.05) ** 2)
        if sl_um in exp_hill:
            hill_terms.append(((sim_hill - exp_hill[sl_um].mean) / 0.6) ** 2)

    # Explicitly emphasize length-dependent pCa50 shift.
    if 2.0 in fitted and 2.3 in fitted and 2.0 in exp_pca50 and 2.3 in exp_pca50:
        sim_delta = fitted[2.3][0] - fitted[2.0][0]
        exp_delta = exp_pca50[2.3].mean - exp_pca50[2.0].mean
        pca50_terms.append(2.0 * ((sim_delta - exp_delta) / 0.03) ** 2)

    pca50_loss = float(np.mean(pca50_terms)) if pca50_terms else 0.0
    hill_loss = float(np.mean(hill_terms)) if hill_terms else 0.0
    return pca50_loss, hill_loss, fitted


def compute_loss(
    topo,
    targets: Dict[str, List[FitTarget]],
    genotype: str,
    params: Dict[str, float],
    model_root: str | Path,
    duration_ms: float,
    dt: float,
    replicates: int,
    steady_last_ms: float,
    rng_seed: int,
    weights: Dict[str, float],
) -> Tuple[float, Dict]:
    sim = run_force_pca_protocol(
        topo=topo,
        dynamic_params=params,
        model_root=model_root,
        duration_ms=duration_ms,
        dt=dt,
        replicates=replicates,
        steady_last_ms=steady_last_ms,
        rng_seed=rng_seed,
    )

    genotypes = ["WT", "KO"] if genotype == "both" else [genotype]
    per_genotype = {}
    total_terms = []

    for genotype_name in genotypes:
        loss_curve = normalized_force_loss(
            sim,
            targets["normalized_force"],
            genotype=genotype_name,
        )
        loss_pca50, loss_hill, fitted = pca50_hill_losses(
            sim,
            targets["pca50"],
            targets["hill_slope"],
            genotype=genotype_name,
        )

        genotype_total = (
            weights["curve"] * loss_curve
            + weights["pca50"] * loss_pca50
            + weights["hill"] * loss_hill
        )
        total_terms.append(genotype_total)
        per_genotype[genotype_name] = {
            "total_loss": genotype_total,
            "curve_loss": loss_curve,
            "pca50_loss": loss_pca50,
            "hill_loss": loss_hill,
            "sim_hill_fit": {
                str(sl): {"pCa50": vals[0], "hill_slope": vals[1]}
                for sl, vals in fitted.items()
            },
        }

    total = float(np.mean(total_terms)) if total_terms else 0.0

    primary = per_genotype[genotypes[0]]

    details = {
        "total_loss": total,
        "curve_loss": primary["curve_loss"],
        "pca50_loss": primary["pca50_loss"],
        "hill_loss": primary["hill_loss"],
        "sim_hill_fit": primary["sim_hill_fit"],
        "per_genotype": per_genotype,
        "sim_norm_force": {
            str(sl): sim_data["norm_force"].tolist()
            for sl, sim_data in sim.items()
        },
        "params": params,
    }
    return float(total), details


def run_fit(args) -> None:
    targets = load_targets(args.excel)
    topo = build_topology(args.model_root, args.nrows, args.ncols)
    weights = {"curve": args.weight_curve, "pca50": args.weight_pca50, "hill": args.weight_hill}

    bounds = get_bounds(args.fit_set)
    history = []
    t0 = time.time()

    def objective(x):
        params = candidate_vector_to_params(np.asarray(x, dtype=float), args.fit_set)
        loss, details = compute_loss(
            topo=topo,
            targets=targets,
            genotype=args.genotype,
            params=params,
            model_root=args.model_root,
            duration_ms=args.duration_ms,
            dt=args.dt,
            replicates=args.replicates,
            steady_last_ms=args.steady_last_ms,
            rng_seed=args.rng_seed,
            weights=weights,
        )
        details["elapsed_s"] = time.time() - t0
        history.append(details)
        print(
            f"loss={loss:.4g} "
            f"curve={details['curve_loss']:.4g} "
            f"pCa50={details['pca50_loss']:.4g} "
            f"hill={details['hill_loss']:.4g} "
            f"params={params}",
            flush=True,
        )
        return loss

    try:
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=args.maxiter,
            popsize=args.popsize,
            seed=args.rng_seed,
            polish=args.polish,
            updating="immediate",
            workers=1,
        )
        best_x = result.x
        best_loss = float(result.fun)
    except Exception as exc:
        print(f"scipy differential_evolution unavailable or failed: {exc}")
        print("Falling back to random search.")
        rng = np.random.default_rng(args.rng_seed)
        best_x = None
        best_loss = float("inf")
        for _ in range(args.random_trials):
            x = np.array([rng.uniform(lo, hi) for lo, hi in bounds], dtype=float)
            loss = objective(x)
            if loss < best_loss:
                best_loss = loss
                best_x = x

    best_params = candidate_vector_to_params(np.asarray(best_x, dtype=float), args.fit_set)
    final_loss, final_details = compute_loss(
        topo=topo,
        targets=targets,
        genotype=args.genotype,
        params=best_params,
        model_root=args.model_root,
        duration_ms=args.duration_ms,
        dt=args.dt,
        replicates=args.replicates,
        steady_last_ms=args.steady_last_ms,
        rng_seed=args.rng_seed,
        weights=weights,
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"fit_result_{args.genotype}.json"
    history_path = output_dir / f"fit_history_{args.genotype}.json"

    payload = {
        "genotype": args.genotype,
        "excel": str(args.excel),
        "model_root": str(args.model_root),
        "best_loss": final_loss,
        "best_params": best_params,
        "details": final_details,
        "settings": vars(args),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("\nBest result")
    print(json.dumps(payload["best_params"], indent=2))
    print(f"loss={final_loss:.6g}")
    print(f"saved: {result_path}")
    print(f"history: {history_path}")


def run_evaluate(args) -> None:
    targets = load_targets(args.excel)
    topo = build_topology(args.model_root, args.nrows, args.ncols)
    weights = {"curve": args.weight_curve, "pca50": args.weight_pca50, "hill": args.weight_hill}
    params = default_candidate_params()

    if args.params_json:
        params.update(json.loads(Path(args.params_json).read_text(encoding="utf-8")))

    loss, details = compute_loss(
        topo=topo,
        targets=targets,
        genotype=args.genotype,
        params=params,
        model_root=args.model_root,
        duration_ms=args.duration_ms,
        dt=args.dt,
        replicates=args.replicates,
        steady_last_ms=args.steady_last_ms,
        rng_seed=args.rng_seed,
        weights=weights,
    )
    print(json.dumps(details, indent=2))
    print(f"\nloss={loss:.6g}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["summarize", "evaluate", "fit"])
    parser.add_argument("--excel", default=DEFAULT_EXCEL_PATH)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-dir", default="fit_results")
    parser.add_argument("--genotype", choices=["WT", "KO", "both"], default="WT")
    parser.add_argument(
        "--fit-set",
        choices=["lda_srx", "tm_coop", "tm_coop_span"],
        default="lda_srx",
        help=(
            "Parameter set to fit: lda_srx fits LDA/SRX parameters; "
            "tm_coop fits only tm_coop_magnitude; tm_coop_span also fits "
            "tm_span_base, tm_span_force50, and tm_span_steep."
        ),
    )
    parser.add_argument("--nrows", type=int, default=4)
    parser.add_argument("--ncols", type=int, default=4)
    parser.add_argument("--duration-ms", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--steady-last-ms", type=float, default=200.0)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--weight-curve", type=float, default=1.0)
    parser.add_argument("--weight-pca50", type=float, default=2.0)
    parser.add_argument("--weight-hill", type=float, default=0.5)
    parser.add_argument("--params-json", default=None)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=40)
    parser.add_argument("--polish", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "summarize":
        targets = load_targets(args.excel)
        print_target_summary(targets)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "fit":
        run_fit(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
