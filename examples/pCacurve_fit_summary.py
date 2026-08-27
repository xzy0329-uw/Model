import jax
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from multifil_jax import get_cardiac_params, run
from multifil_jax.core.sarc_geometry import SarcTopology


def hill_pca(pca, fmin, fmax, pca50, hill_coefficient):
    return fmin + (fmax - fmin) / (
        1.0 + 10.0 ** (hill_coefficient * (pca - pca50))
    )


def standard_error(values, axis=None):
    values = np.asarray(values)
    n = values.shape[axis] if axis is not None else values.size
    if n < 2:
        return np.nan
    return np.std(values, axis=axis, ddof=1) / np.sqrt(n)


def fit_hill_curve(pca, active_force):
    """Fit one active-force pCa curve and return Fmin, Fmax, pCa50, nH."""
    return curve_fit(
        hill_pca,
        pca,
        active_force,
        p0=[
            0.0,
            max(float(np.max(active_force)), 1e-6),
            5.5,
            2.0,
        ],
        bounds=(
            [0.0, 0.0, 4.0, 0.0],
            [np.inf, np.inf, 7.0, 8.0],
        ),
        maxfev=100000,
    )


def lattice_from_z(z_line_nm, reference_z_nm=1000.0, reference_ls_nm=14.0, nu=0.5):
    return reference_ls_nm * (reference_z_nm / z_line_nm) ** nu


def summarize_fits(fit_replicates):
    def series_sem(series):
        return standard_error(series.to_numpy())

    group_columns = ["condition", "SL_um", "z_line_nm", "lattice_spacing_nm"]
    return (
        fit_replicates.groupby(group_columns, as_index=False)
        .agg(
            n_successful_fits=("replicate", "count"),
            Fmax_mean_pN=("Fmax_pN", "mean"),
            Fmax_sem_pN=("Fmax_pN", series_sem),
            pCa50_mean=("pCa50", "mean"),
            pCa50_sem=("pCa50", series_sem),
            hill_coefficient_mean=("hill_coefficient", "mean"),
            hill_coefficient_sem=("hill_coefficient", series_sem),
        )
    )


def main():
    duration_ms = 1000
    dt_ms = 1
    replicates = 100
    rng_seed = 12345
    steady_window_ms = 600
    output_file = "pCa_force_results.xlsx"

    # pCa must be supplied in the same order used for every replicate.
    pca_values = np.array([9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 4.5])

    conditions = {
        "SL 2.0": {"SL_um": 2.0, "z_line_nm": 1000.0},
        "SL 2.3": {"SL_um": 2.3, "z_line_nm": 1150.0},
    }

    static, dynamic = get_cardiac_params()
    dynamic = dynamic.copy(
    tm_k_12=15000.0,
    tm_k_23=0.40,
    tm_k_34=0.20,
    tm_k_41=0.5,

    tm_K1=15000.0,
    tm_K2=40.0,
    tm_K3=0.05,

    tm_coop_magnitude=2.0,
    tm_span_base=62.0,
    tm_span_force50=-8.0,
    tm_span_steep=0.8,

    xb_r12_coeff=250.0,
    xb_r23_coeff=0.60,
    xb_r34_coeff=0.15,
    xb_r45_coeff=0.60,
    xb_r51=0.10,
    xb_r15=0.01,

    xb_srx_k0=0.003,
    xb_r16=0.010,

    xb_lda_enabled=1.0,
    xb_lda_gain=5.0,
    
    xb_lda_lattice_gain=0.5,
    xb_lattice_reference=14.0,
    xb_lattice_binding_beta=1.0,
    )
    
    print(
    "ACTIVE:",
    float(dynamic.tm_K1),
    float(dynamic.tm_K2),
    float(dynamic.tm_coop_magnitude),
    float(dynamic.xb_lda_gain),
    )
    topology = SarcTopology.create(
        nrows=4,
        ncols=4,
        static_params=static,
        dynamic_params=dynamic,
    )
    topology = jax.device_put(topology)

    print(f"Topology: {topology.n_thick} thick, {topology.n_thin} thin filaments")
    print(f"Crowns: {topology.n_crowns}/thick, Sites: {topology.n_sites}/thin")
    print(f"Total XBs: {topology.total_xbs}")

    force_summary_rows = []
    force_replicate_rows = []
    fit_replicate_rows = []

    plt.figure(figsize=(12, 9))

    for label, condition in conditions.items():
        z_line_nm = condition["z_line_nm"]
        lattice_spacing_nm = lattice_from_z(z_line_nm)
        sl_um = condition["SL_um"]

        print(f"\nRunning condition: {label}")
        result = run(
            topology,
            pCa=pca_values.tolist(),
            z_line=z_line_nm,
            lattice_spacing=lattice_spacing_nm,
            duration_ms=duration_ms,
            dt=dt_ms,
            replicates=replicates,
            rng_seed=rng_seed,
            # Keep the cardiac parameters returned above. Do not pass a small
            # dict here, because that would replace the cardiac parameter set.
            dynamic_params=dynamic,
        )

        for key in (
            "frac_xb_srx",
            "frac_xb_drx",
            "frac_xb_bound",
            "actin_permissiveness",
            "frac_xb_lda_signal",
            "frac_xb_strained",
        ):
            if key not in result.metrics:
                continue
            values = np.asarray(result.metrics[key])
            steady_values = values[..., -steady_window_ms:].mean(axis=-1)
            print(key, steady_values.mean(axis=1))

        force = np.asarray(result.axial_force)
        steady_force = force[..., -steady_window_ms:].mean(axis=-1)
        # Shape is (n_pCa, n_replicates).

        baseline_idx = int(np.where(np.isclose(pca_values, 9.0))[0][0])
        baseline_by_replicate = steady_force[baseline_idx, :]
        active_force_by_replicate = steady_force - baseline_by_replicate[None, :]

        mean_force = steady_force.mean(axis=1)
        force_sem = standard_error(steady_force, axis=1)
        mean_active_force = active_force_by_replicate.mean(axis=1)
        active_force_sem = standard_error(active_force_by_replicate, axis=1)

        # Fit the curve through the mean force data for plotting only.
        mean_popt, _ = fit_hill_curve(pca_values, mean_active_force)
        _, mean_fmax, mean_pca50, mean_nh = mean_popt
        print(
            f"{label}: mean-curve Fmax={mean_fmax:.3f} pN, "
            f"pCa50={mean_pca50:.3f}, nH={mean_nh:.3f}"
        )

        for pca_index, pca in enumerate(pca_values):
            force_summary_rows.append({
                "condition": label,
                "SL_um": sl_um,
                "pCa": pca,
                "z_line_nm": z_line_nm,
                "lattice_spacing_nm": lattice_spacing_nm,
                "mean_force_pN": mean_force[pca_index],
                "force_sem_pN": force_sem[pca_index],
                "mean_active_force_pN": mean_active_force[pca_index],
                "active_force_sem_pN": active_force_sem[pca_index],
            })

            for replicate in range(replicates):
                force_replicate_rows.append({
                    "condition": label,
                    "SL_um": sl_um,
                    "pCa": pca,
                    "z_line_nm": z_line_nm,
                    "lattice_spacing_nm": lattice_spacing_nm,
                    "replicate": replicate,
                    "steady_force_pN": steady_force[pca_index, replicate],
                    "baseline_force_pN": baseline_by_replicate[replicate],
                    "active_force_pN": active_force_by_replicate[pca_index, replicate],
                })

        # Each replicate has its own complete pCa-force curve.
        for replicate in range(replicates):
            try:
                popt, _ = fit_hill_curve(
                    pca_values,
                    active_force_by_replicate[:, replicate],
                )
            except (RuntimeError, ValueError) as error:
                print(f"Warning: Hill fit failed for {label}, replicate {replicate}: {error}")
                continue

            fmin, fmax, pca50, hill_coefficient = popt
            fit_replicate_rows.append({
                "condition": label,
                "SL_um": sl_um,
                "z_line_nm": z_line_nm,
                "lattice_spacing_nm": lattice_spacing_nm,
                "replicate": replicate,
                "Fmin_pN": fmin,
                "Fmax_pN": fmax,
                "pCa50": pca50,
                "hill_coefficient": hill_coefficient,
            })

        pca_smooth = np.linspace(pca_values.min(), pca_values.max(), 300)
        plt.errorbar(
            pca_values,
            mean_active_force,
            yerr=active_force_sem,
            fmt="o",
            capsize=4,
            label=f"{label} simulation (mean +/- SEM)",
        )
        plt.plot(
            pca_smooth,
            hill_pca(pca_smooth, *mean_popt),
            linewidth=2,
            label=f"{label} fit, nH={mean_nh:.2f}, pCa50={mean_pca50:.2f}",
        )

    force_summary = pd.DataFrame(force_summary_rows)
    force_replicates = pd.DataFrame(force_replicate_rows)
    fit_replicates = pd.DataFrame(fit_replicate_rows)
    fit_summary = summarize_fits(fit_replicates)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        fit_summary.to_excel(writer, sheet_name="fit_summary", index=False)
        fit_replicates.to_excel(writer, sheet_name="fit_replicates", index=False)
        force_summary.to_excel(writer, sheet_name="force_summary", index=False)
        force_replicates.to_excel(writer, sheet_name="force_replicates", index=False)

    print(f"\nSaved force and fit results to {output_file}")
    print("\nFit summary:")
    print(fit_summary.to_string(index=False))

    plt.gca().invert_xaxis()
    plt.xlabel("pCa")
    plt.ylabel("Active Force (pN)")
    plt.title("pCa-Force Curve: Length Dependence Activation Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
