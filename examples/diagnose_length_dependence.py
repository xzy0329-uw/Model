#!/usr/bin/env python3
"""Diagnose length-dependent activation, geometry, and lattice-spacing effects.

Copy this file to the root of the multifil_jax repository and run:

    python diagnose_length_dependence.py

It runs SL 2.0 and SL 2.3 under four diagnostic configurations:
  1. LDA off, fixed lattice spacing
  2. LDA on,  fixed lattice spacing
  3. LDA off, Poisson lattice spacing
  4. LDA on,  Poisson lattice spacing

The output workbook separates the force-pCa fits from the mechanistic metrics,
so a reduced Fmax can be attributed to the lattice, baseline geometry, or LDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


# ---------------------------------------------------------------------------
# Experiment controls
# ---------------------------------------------------------------------------
PCA_VALUES = np.array([9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 4.5])
DURATION_MS = 1000
DT_MS = 1
REPLICATES = 32  # Raise to 100 once the diagnostic behavior is understood.
STEADY_WINDOW_MS = 600
NROWS = 4
NCOLS = 4

Z_BY_SL = {"SL 2.0": 1000.0, "SL 2.3": 1150.0}
FIXED_LATTICE_NM = 14.0
POISSON_NU = 0.5

# Keep these fixed while comparing the four conditions below.
# Do not tune TM and LDA parameters simultaneously during this diagnosis.
MODEL_UPDATES = {
    "tm_k_12": 15000.0,
    "tm_k_23": 0.40,
    "tm_k_34": 0.20,
    "tm_k_41": 0.50,
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
    "xb_lda_gain": 3.0,
    # Old strain-distance LDA fields.
    "xb_lda_strain_threshold": 0.7,
    "xb_lda_strain_scale": 1.5,
    # New force-based LDA fields, used only if they exist in DynamicParams.
    "xb_lda_force_threshold": 1.0,
    "xb_lda_force_scale": 0.5,
    # Optional backbone-preload fields, used only if they exist.
    "xb_lda_reference_z": 1000.0,
    "xb_lda_z_scale": 150.0,
    "xb_lda_preload_gain": 0.5,
}

METRIC_KEYS = [
    "frac_xb_srx",
    "frac_xb_drx",
    "frac_xb_bound",
    "actin_permissiveness",
    "frac_xb_lda_signal",
    "frac_xb_strained",
]

OUTPUT_FILE = Path("length_dependence_diagnostics.xlsx")


@dataclass(frozen=True)
class Condition:
    sl_label: str
    lda_enabled: float
    lattice_mode: str

    @property
    def label(self) -> str:
        lda = "LDA on" if self.lda_enabled else "LDA off"
        return f"{self.sl_label} | {lda} | {self.lattice_mode} lattice"


def hill_pca(pca: np.ndarray, fmin: float, fmax: float, pca50: float, nh: float) -> np.ndarray:
    return fmin + (fmax - fmin) / (1.0 + 10.0 ** (nh * (pca - pca50)))


def lattice_from_z(z_line_nm: float) -> float:
    return FIXED_LATTICE_NM * (1000.0 / z_line_nm) ** POISSON_NU


def apply_available_updates(dynamic, updates: dict[str, float]):
    """Copy only fields that exist in this local DynamicParams version."""
    available = {key: value for key, value in updates.items() if hasattr(dynamic, key)}
    skipped = sorted(set(updates) - set(available))
    if skipped:
        print("Skipped unavailable DynamicParams fields:", ", ".join(skipped))
    return dynamic.copy(**available)


def fit_one_curve(active_force: np.ndarray) -> np.ndarray:
    p0 = [0.0, float(np.max(active_force)), 5.7, 1.5]
    bounds = ([0.0, 0.0, 4.0, 0.1], [np.inf, np.inf, 8.0, 8.0])
    return curve_fit(
        hill_pca,
        PCA_VALUES,
        active_force,
        p0=p0,
        bounds=bounds,
        maxfev=100_000,
    )[0]


def summarise_result(result, condition: Condition, z_line_nm: float, lattice_nm: float):
    force = np.asarray(result.axial_force)
    steady_force = force[..., -STEADY_WINDOW_MS:].mean(axis=-1)
    # Subtract the pCa 9 force for every replicate, not merely the mean baseline.
    baseline_idx = int(np.where(np.isclose(PCA_VALUES, 9.0))[0][0])
    active_by_rep = steady_force - steady_force[baseline_idx][None, :]

    force_rows = []
    for pca_idx, pca in enumerate(PCA_VALUES):
        values = active_by_rep[pca_idx]
        force_rows.append({
            "condition": condition.label,
            "SL_um": float(condition.sl_label.split()[-1]),
            "z_line_nm": z_line_nm,
            "lattice_spacing_nm": lattice_nm,
            "pCa": pca,
            "active_force_mean_pN": float(values.mean()),
            "active_force_sem_pN": float(values.std(ddof=1) / np.sqrt(values.size)),
        })

    metric_rows = []
    for key in METRIC_KEYS:
        if key not in result.metrics:
            continue
        values = np.asarray(result.metrics[key])
        steady = values[..., -STEADY_WINDOW_MS:].mean(axis=-1)
        for pca_idx, pca in enumerate(PCA_VALUES):
            per_rep = steady[pca_idx]
            metric_rows.append({
                "condition": condition.label,
                "SL_um": float(condition.sl_label.split()[-1]),
                "z_line_nm": z_line_nm,
                "lattice_spacing_nm": lattice_nm,
                "metric": key,
                "pCa": pca,
                "mean": float(per_rep.mean()),
                "sem": float(per_rep.std(ddof=1) / np.sqrt(per_rep.size)),
            })

    fits = []
    for rep_idx in range(active_by_rep.shape[1]):
        try:
            fmin, fmax, pca50, nh = fit_one_curve(active_by_rep[:, rep_idx])
        except RuntimeError:
            continue
        fits.append({
            "condition": condition.label,
            "SL_um": float(condition.sl_label.split()[-1]),
            "z_line_nm": z_line_nm,
            "lattice_spacing_nm": lattice_nm,
            "replicate": rep_idx,
            "Fmin_pN": fmin,
            "Fmax_pN": fmax,
            "pCa50": pca50,
            "hill_coefficient": nh,
        })

    return force_rows, metric_rows, fits


def main() -> None:
    static, dynamic = get_cardiac_params()
    dynamic = apply_available_updates(dynamic, MODEL_UPDATES)

    topology = SarcTopology.create(
        nrows=NROWS,
        ncols=NCOLS,
        static_params=static,
        dynamic_params=dynamic,
    )
    topology = jax.device_put(topology)

    print(f"Topology: {topology.n_thick} thick, {topology.n_thin} thin filaments")
    print(f"Crowns: {topology.n_crowns}/thick, Sites: {topology.n_sites}/thin")

    conditions = [
        Condition(sl, lda, lattice)
        for sl in Z_BY_SL
        for lattice in ("fixed", "Poisson")
        for lda in (0.0, 1.0)
    ]

    all_force_rows: list[dict] = []
    all_metric_rows: list[dict] = []
    all_fits: list[dict] = []

    for condition in conditions:
        z_line_nm = Z_BY_SL[condition.sl_label]
        lattice_nm = (
            FIXED_LATTICE_NM
            if condition.lattice_mode == "fixed"
            else lattice_from_z(z_line_nm)
        )
        condition_dynamic = dynamic.copy(xb_lda_enabled=condition.lda_enabled)

        print(f"\nRunning: {condition.label}")
        result = run(
            topology,
            pCa=PCA_VALUES.tolist(),
            z_line=z_line_nm,
            lattice_spacing=lattice_nm,
            duration_ms=DURATION_MS,
            dt=DT_MS,
            replicates=REPLICATES,
            dynamic_params=condition_dynamic,
        )

        force_rows, metric_rows, fits = summarise_result(
            result, condition, z_line_nm, lattice_nm
        )
        all_force_rows.extend(force_rows)
        all_metric_rows.extend(metric_rows)
        all_fits.extend(fits)

    force_df = pd.DataFrame(all_force_rows)
    metric_df = pd.DataFrame(all_metric_rows)
    fit_replicates_df = pd.DataFrame(all_fits)

    fit_summary_df = (
        fit_replicates_df.groupby(
            ["condition", "SL_um", "z_line_nm", "lattice_spacing_nm"], as_index=False
        )
        .agg(
            n_successful_fits=("replicate", "count"),
            Fmax_mean_pN=("Fmax_pN", "mean"),
            Fmax_sem_pN=("Fmax_pN", lambda x: x.std(ddof=1) / np.sqrt(x.size)),
            pCa50_mean=("pCa50", "mean"),
            pCa50_sem=("pCa50", lambda x: x.std(ddof=1) / np.sqrt(x.size)),
            hill_mean=("hill_coefficient", "mean"),
            hill_sem=("hill_coefficient", lambda x: x.std(ddof=1) / np.sqrt(x.size)),
        )
    )

    # At pCa 4.5, lower bound/force in SL 2.3 indicates a geometry/overlap issue.
    high_ca_metrics = metric_df[np.isclose(metric_df["pCa"], 4.5)].copy()
    high_ca_force = force_df[np.isclose(force_df["pCa"], 4.5)].copy()

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        fit_summary_df.to_excel(writer, sheet_name="fit_summary", index=False)
        fit_replicates_df.to_excel(writer, sheet_name="fit_replicates", index=False)
        force_df.to_excel(writer, sheet_name="force_by_pCa", index=False)
        metric_df.to_excel(writer, sheet_name="metrics_by_pCa", index=False)
        high_ca_force.to_excel(writer, sheet_name="pCa_4p5_force", index=False)
        high_ca_metrics.to_excel(writer, sheet_name="pCa_4p5_metrics", index=False)

    print("\nFit summary:")
    print(fit_summary_df.to_string(index=False))
    print(f"\nWrote diagnostics to: {OUTPUT_FILE.resolve()}")
    print("\nInterpretation guide:")
    print("- Fmax lower at SL 2.3 with LDA off: geometry/overlap is the cause.")
    print("- Fmax improves when lattice is fixed: Poisson lattice compression is the cause.")
    print("- pCa50 rises only when LDA is on: LDA is producing length-dependent sensitivity.")
    print("- Lower pCa 4.5 frac_xb_bound at SL 2.3: fewer effective XB attachments.")


if __name__ == "__main__":
    main()
