"""Generate the golden-file fixture for the C++ FilteredNoise interpolation test.

Builds a band-limited random field exactly as FilteredNoise.__init__ in
../../Overleaf/qEW/qew_core.py, then evaluates SciPy's cubic B-spline
interpolant (value + central-difference derivatives, the same scheme the Python
model uses) at a set of off-grid query points. The C++ test loads the field,
runs its own periodic FFT prefilter + analytic bicubic evaluation, and compares.

We use mode='grid-wrap' (clean period-N periodicity) to match the C++
implementation; the production code's mode='wrap' differs only negligibly for
the smooth fields used here. Run from this directory:

    python3 gen_golden.py
"""

import numpy as np
import h5py
from scipy.ndimage import spline_filter, map_coordinates

MODE = "grid-wrap"

nx, ny = 64, 48
Lx, Ly = 3.2, 1.0
amplitude = 1.7
xi_x = xi_y = 0.15
seed = 7

dx, dy = Lx / nx, Ly / ny

# --- band-limited random field (mirrors FilteredNoise.__init__) ---
np.random.seed(seed)
fx = np.fft.fftfreq(nx) / dx
fy = np.fft.fftfreq(ny) / dy
Fx, Fy = np.meshgrid(fx, fy, indexing="ij")
A = np.exp(1j * np.random.uniform(0, 2 * np.pi, (nx, ny)))
A[(Fx * xi_x) ** 2 + (Fy * xi_y) ** 2 > 1] = 0
noise = np.real(np.fft.ifft2(A))
noise *= amplitude / np.std(noise)

coef = spline_filter(noise, order=3, mode=MODE)

# --- query points, including coordinates outside the box to test periodic wrap ---
rng = np.random.default_rng(123)
M = 256
xq = rng.uniform(-Lx, 2 * Lx, M)
yq = rng.uniform(-Ly, 2 * Ly, M)
gx = (xq / dx) % nx
gy = (yq / dy) % ny

h = 1e-4  # finite-difference step in grid units, as in qew_core.FilteredNoise
def mc(a, b):
    return map_coordinates(coef, [a, b], order=3, mode=MODE, prefilter=False)

v = mc(gx, gy)
dvdx = (mc((gx + h) % nx, gy) - mc((gx - h) % nx, gy)) / (2 * h) / dx
dvdy = (mc(gx, (gy + h) % ny) - mc(gx, (gy - h) % ny)) / (2 * h) / dy

with h5py.File("noise_golden.h5", "w") as f:
    f.attrs["nx"] = nx
    f.attrs["ny"] = ny
    f.attrs["Lx"] = Lx
    f.attrs["Ly"] = Ly
    f.attrs["amplitude"] = amplitude
    f.attrs["xi_x"] = xi_x
    f.attrs["xi_y"] = xi_y
    f.create_dataset("field", data=noise)  # shape (nx, ny)
    f.create_dataset("x", data=xq)
    f.create_dataset("y", data=yq)
    f.create_dataset("v", data=v)
    f.create_dataset("dvdx", data=dvdx)
    f.create_dataset("dvdy", data=dvdy)

print(f"wrote noise_golden.h5  grid={nx}x{ny}  queries={M}  mode={MODE}")
print(f"  field std={noise.std():.6f} (target {amplitude})  v range=[{v.min():.3f},{v.max():.3f}]")
