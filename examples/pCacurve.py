import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



from multifil_jax import run, get_cardiac_params, SimulationResult
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import StaticParams

def hill_pca(pca, Fmin, Fmax, pCa50, nH):
    return Fmin + (Fmax - Fmin) / (1.0 + 10.0 ** (nH * (pca - pCa50))) 


static, dynamic = get_cardiac_params()
topo = SarcTopology.create(
        nrows=2,   
        ncols=2, 
        static_params=static, 
        dynamic_params=dynamic)
topo = jax.device_put(topo)

print(f" Topology: {topo.n_thick} thick, {topo.n_thin} thin filaments")
print(f" Crowns: {topo.n_crowns}/thick, Sites: {topo.n_sites}/thin")
print(f" Total XBs: {topo.total_xbs}")

pca_value = [4.5, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 9.0]

result = run(
    topo,
    pCa= pca_value,
    z_line=900.0,
    duration_ms=1000,
    dt=1,
    replicates=100,
    dynamic_params={
            'xb_lda_enabled': 1.0
    }
)

force = result.axial_force

steady_force = force[..., -600:].mean(axis=-1)
mean_force = steady_force.mean(axis=1)
std_force = steady_force.std(axis=1)

baseline_idx = np.where(np.isclose(pca_value, 9.0))[0][0]
baseline = mean_force[baseline_idx]
active_force = mean_force - baseline

#Hill Co-efficient Calculation

p0 = [
    float(np.min(active_force)),
    float(np.max(active_force)),
    5.6,
    3.0
]

bounds = (
    [0.0, 0.0, 0.0, -10.0],
    [np.inf, np.inf, 10.0, 10.0]
)

popt, pcov = curve_fit(
    hill_pca,
    pca_value,
    active_force,
    p0=p0,
    bounds=bounds,
    maxfev=100000
)

Fmin_fit, Fmax_fit, pCa50_fit, nH_fit = popt

print(f"Fmin = {Fmin_fit:.3f}")
print(f"Fmax = {Fmax_fit:.3f}")
print(f"pCa50 = {pCa50_fit:.3f}")
print(f"Hill coefficient = {nH_fit:.3f}")

pca_smooth = np.linspace(min(pca_value), max(pca_value), 300)
force_fit = hill_pca(pca_smooth, *popt)

plt.figure(figsize=(12,9))
plt.errorbar(pca_value, active_force, yerr=std_force, fmt='o', capsize=4, label='Simulation data')
plt.plot(pca_smooth, force_fit, '-', linewidth=2, label='pCa Curve')

plt.gca().invert_xaxis()
plt.xlabel("pCa")
plt.ylabel("Active Force (pN)")
plt.title("pCa - Force Curve")
plt.legend()
plt.tight_layout()
plt.show()