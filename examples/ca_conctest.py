import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from multifil_jax import run, get_cardiac_params, SimulationResult
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import StaticParams

static, dynamic = get_cardiac_params()
topo = SarcTopology.create(
        nrows=4,   
        ncols=4, 
        static_params=static, 
        dynamic_params=dynamic)
topo = jax.device_put(topo)

pca_value = [100.0, 50.0, 9.0, 4.5, 4.0]

result = run(
    topo,
    pCa = pca_value,
    z_line = 900.0,
    duration_ms = 1000,
    dt = 1,
    replicates = 10,
)

force = result.axial_force

steady_force = force[..., -500:].mean(axis=-1)
mean_force = steady_force.mean(axis=1)
std_force = steady_force.std(axis=1)

plt.figure(figsize=(8,6))
plt.errorbar(pca_value, mean_force, yerr=std_force, fmt='o-', capsize=4)
plt.gca().invert_xaxis()
plt.xlabel("pCa")
plt.ylabel("Active Force (pN)")
plt.title("pCa - Force Curve")
plt.legend()
plt.tight_layout()
plt.show()