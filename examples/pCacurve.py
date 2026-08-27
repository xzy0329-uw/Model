import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from multifil_jax import run, get_cardiac_params, SimulationResult
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import StaticParams


def hill_pca(pca, Fmin, Fmax, pCa50, nH):
    return Fmin + (Fmax - Fmin) / (1.0 + 10.0 ** (nH * (pca - pCa50)))


static, dynamic = get_cardiac_params()

conditions = {
    "SL 2.0":  dict(z_line=1000.0),
    "SL 2.3":  dict(z_line=1150.0),
}



def lattice_from_z(z, z0=1000.0, d0=14.0, nu=0.5):
    return d0 * (z0 / z) ** nu

def lda_gain_from_z(z, z0=1000.0, base_gain=5.0, length_gain=30.0):
    extension = max(0.0, z / z0 - 1.0)
    return base_gain + length_gain * extension

def lda_threshold_from_z(z, z0=1000.0, threshold0=1.0, sensitivity=4.0):
    extension = max(0.0, z / z0 - 1.0)
    return max(0.2, threshold0 - sensitivity * extension)

topo = SarcTopology.create(
    nrows=4,
    ncols=4,
    static_params=static,
    dynamic_params=dynamic,
)
topo = jax.device_put(topo)

print(f"Topology: {topo.n_thick} thick, {topo.n_thin} thin filaments")
print(f"Crowns: {topo.n_crowns}/thick, Sites: {topo.n_sites}/thin")
print(f"Total XBs: {topo.total_xbs}")

pca_value = [9.0, 6.2, 6.0, 5.8, 5.6, 5.4, 5.2, 4.5]

fit_results = {}

summary_rows = []
replicate_rows = []
plt.figure(figsize=(12, 9))


for label, cond in conditions.items():
    z = cond["z_line"]
    lattice_spacing = lattice_from_z(z)
    print(f"\nRunning condition: {label}")

    result = run(
        topo,
        pCa=pca_value,
        z_line=z,
        lattice_spacing=lattice_spacing,
        duration_ms=1000,
        dt=1,
        replicates=100,
        rng_seed=12345,
        static_params=static,
    )
    
    
    
    print(f"\nDiagnostics for {label}")
    for key in [
        "frac_xb_srx",
        "frac_xb_drx",
        "frac_xb_bound",
        "actin_permissiveness",
        "frac_xb_lda_signal",
        "frac_xb_strained",
    ]:
        vals = np.asarray(result.metrics[key])
        steady = vals[..., -600:].mean(axis=-1)
        mean_vals = steady.mean(axis=1)
        print(key, mean_vals)

    force = np.asarray(result.axial_force)

    steady_force = force[..., -600:].mean(axis=-1)
    mean_force = steady_force.mean(axis=1)
    std_force = steady_force.std(axis=1)
    pca_array = np.array(pca_value)
    baseline_idx = np.where(np.isclose(pca_value, 9.0))[0][0]
    baseline = mean_force[baseline_idx]
    active_force = mean_force - baseline

    for i, pca in enumerate(pca_value):
        summary_rows.append({
            "condition": label,
            "SL": 2.0 if "2.0" in label else 2.3,
            "LDA": "on" if "on" in label else "off",
            "pCa": pca,
            "z_line": z,
            "lattice_spacing": lattice_spacing,
            "mean_force_pN": mean_force[i],
            "std_force_pN": std_force[i],
            "baseline_force_pN": baseline,
            "active_force_pN": active_force[i],
        })

        for rep in range(steady_force.shape[1]):
            replicate_rows.append({
                "condition": label,
                "SL": 2.0 if "2.0" in label else 2.3,
                "LDA": "on" if "on" in label else "off",
                "pCa": pca,
                "z_line": z,
                "lattice_spacing": lattice_spacing,
                "replicate": rep,
                "steady_force_pN": steady_force[i, rep],
            })

    p0 = [
        float(np.min(active_force)),
        float(np.max(active_force)),
        5.5,
        2.0,
    ]

    bounds = (
        [0.0, 0.0, 4.0, -8.0],
        [np.inf, np.inf, 7.0, 8.0],
        )

    popt, pcov = curve_fit(
        hill_pca,
        pca_array,
        active_force,
        p0=p0,
        bounds=bounds,
        maxfev=100000,
    )

    Fmin_fit, Fmax_fit, pCa50_fit, nH_fit = popt

    fit_results[label] = {
        "Fmin": Fmin_fit,
        "Fmax": Fmax_fit,
        "pCa50": pCa50_fit,
        "nH": nH_fit,
        "active_force": active_force,
        "std_force": std_force,
        "popt": popt,
    }

    print(f"{label}")
    print(f"Fmin = {Fmin_fit:.3f}")
    print(f"Fmax = {Fmax_fit:.3f}")
    print(f"pCa50 = {pCa50_fit:.3f}")
    print(f"Hill coefficient = {nH_fit:.3f}")

    pca_smooth = np.linspace(min(pca_value), max(pca_value), 300)
    force_fit = hill_pca(pca_smooth, *popt)

    plt.errorbar(
        pca_value,
        active_force,
        yerr=std_force,
        fmt="o",
        capsize=4,
        label=f"{label} simulation",
    )

    plt.plot(
        pca_smooth,
        force_fit,
        linewidth=2,
        label=f"{label} fit, nH={nH_fit:.2f}, pCa50={pCa50_fit:.2f}",
    )

plt.gca().invert_xaxis()
plt.xlabel("pCa")
plt.ylabel("Active Force (pN)")
plt.title("pCa - Force Curve: Length Dependence Activation Comparison")
plt.legend()
plt.tight_layout()
plt.show()

summary_df = pd.DataFrame(summary_rows)
replicate_df = pd.DataFrame(replicate_rows)

with pd.ExcelWriter("pCa_force_results.xlsx", engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="summary", index=False)
    replicate_df.to_excel(writer, sheet_name="replicates", index=False)

print("Saved force results to pCa_force_results.xlsx")