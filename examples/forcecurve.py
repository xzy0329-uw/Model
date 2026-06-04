import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit



from multifil_jax import run, get_cardiac_params, SimulationResult
from multifil_jax.core.sarc_geometry import SarcTopology
from multifil_jax.core.params import StaticParams

static, dynamic = get_cardiac_params()
topo = SarcTopology.create(
        nrows=2,   
        ncols=2, 
        static_params=static, 
        dynamic_params=dynamic)
topo = jax.device_put(topo)

result = run(
    topo,
    pCa=4.5,
    z_line=1150,
    lattice_spacing=14,
    duration_ms=1000,
    dt=1,
    replicates=3,
    )

force = result.axial_force
time = np.arange(force.shape[1]) * 1.0

plt.figure(figsize=(8,6))

for i in range(force.shape[0]):
    plt.plot(time, force[i], label=f"rep{i+1}")

plt.xlabel("Time(ms)")
plt.ylabel("Axial Force(pN)")
plt.legend()
plt.tight_layout()
plt.show()

