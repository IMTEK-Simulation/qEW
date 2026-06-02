"""Golden fixture for the C++ L-BFGS minimizer, validated against SciPy.

Builds a band-limited random field (weak disorder + stiff line tension, so the
energy is near-convex and the minimum from a flat start is well defined),
defines the qEW energy exactly as qew_model.objective (linear model,
edge-midpoint noise via a periodic grid-wrap cubic B-spline), minimises it with
SciPy's L-BFGS-B from a flat start, and stores the field, start config and
minimum energy. The C++ test loads the field, runs its own L-BFGS from the same
start, and checks the converged energy matches.

Run from this directory:  python3 gen_lbfgs_golden.py
"""

import numpy as np
import h5py
from scipy.ndimage import spline_filter, map_coordinates
from scipy.optimize import fmin_l_bfgs_b

MODE = "grid-wrap"

nx, ny = 48, 32
Lx, Ly = 2.4, 1.0
amplitude = 0.2      # weak disorder
xi = 0.25            # large correlation length -> smooth
line_tension = 3.0   # stiff -> near-convex
seed = 4

dx, dy = Lx / nx, Ly / ny

# --- band-limited random field (as FilteredNoise.__init__) ---
np.random.seed(seed)
fx = np.fft.fftfreq(nx) / dx
fy = np.fft.fftfreq(ny) / dy
Fx, Fy = np.meshgrid(fx, fy, indexing="ij")
A = np.exp(1j * np.random.uniform(0, 2 * np.pi, (nx, ny)))
A[(Fx * xi) ** 2 + (Fy * xi) ** 2 > 1] = 0
field = np.real(np.fft.ifft2(A))
field *= amplitude / np.std(field)

coef = spline_filter(field, order=3, mode=MODE)
x = np.arange(nx) * dx


def objective(h):
    """qEW energy (linear model, driving force 0), matching qew_model.objective."""
    dh = (np.roll(h, -1) - h) / dx
    line_energy = line_tension * dh**2 / 2
    xcenter = (np.roll(x, -1) + x) / 2
    hcenter = (np.roll(h, -1) + h) / 2
    gx = (xcenter / dx) % nx
    gy = (hcenter / dy) % ny
    v = map_coordinates(coef, [gx, gy], order=3, mode=MODE, prefilter=False)
    return float(np.sum(dx * (line_energy + v)))


h0 = np.full(nx, 0.5 * Ly)
h_min, e_min, info = fmin_l_bfgs_b(
    objective, h0, approx_grad=True, factr=10.0, pgtol=1e-10, maxiter=20000
)

with h5py.File("lbfgs_golden.h5", "w") as f:
    f.attrs["nx"] = nx
    f.attrs["ny"] = ny
    f.attrs["Lx"] = Lx
    f.attrs["Ly"] = Ly
    f.attrs["line_tension"] = line_tension
    f.attrs["e_min"] = e_min
    f.create_dataset("field", data=field)
    f.create_dataset("h0", data=h0)
    f.create_dataset("h_min", data=h_min)

print(f"wrote lbfgs_golden.h5  grid={nx}x{ny}  e_min={e_min:.6g}  "
      f"warnflag={info['warnflag']}  funcalls={info['funcalls']}")
