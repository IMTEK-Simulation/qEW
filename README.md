# qEW — C++/Kokkos GPU port

A quenched Edwards–Wilkinson (qEW) dislocation-line model, built on [Kokkos](https://kokkos.org/)
so the heavy runs (energy minimization, depinning, Brownian dynamics) execute on
a GPU. It accompanies the paper *"Fractal structure, depinning, and hysteresis
of dislocations in high-entropy alloys"* (Le et al., arXiv:2410.21838).

## What it computes

A 1-D line `h(x)` relaxing in a correlated random pinning field, with either a
harmonic (`linear`) or geometric line-tension (`arclength`) elastic term:

- **Statics** — relax to a local minimum (L-BFGS), sweeping the pinning length.
- **Dynamics** — finite-temperature overdamped Langevin time integration.
- **Depinning** — the critical configuration and threshold force `f_c` via the
  Rosso–Krauth no-passing construction (Rosso & Krauth, *Phys. Rev. E* **65**,
  025101(R), 2002).

## Dependencies

All fetched automatically (CMake `FetchContent`) if not installed:

- **Kokkos 5.1.1** — on-node parallelism / GPU backend. Requires **C++20**.
- **kokkos-fft v1.1.0** — FFTs on Kokkos Views (noise setup + the L-BFGS
  preconditioner). Its backend follows the Kokkos backend: **FFTW** on the host
  (so CPU builds need **`libfftw3-dev`**), **hipFFT/rocFFT** on HIP (MI300A),
  cuFFT on CUDA.
- **HDF5 + HighFive** — *optional*, for `.h5` I/O. Auto-detected via
  `find_package(HDF5)`; without it the library still builds and the
  HDF5-dependent tests and executables are simply skipped.

No MPI, no Eigen3. (Host builds: `sudo apt-get install libfftw3-dev libhdf5-dev`.)

## Build & test

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest --output-on-failure        # 24 tests
```

The default build targets a single CPU core (Kokkos `Serial`), so `ctest` passes
on any machine. Select a different Kokkos backend at configure time:

```sh
# multi-threaded CPU (OpenMP)
cmake -B build_omp -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON
cmake --build build_omp
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=threads \
    ctest --test-dir build_omp --output-on-failure

# GPU — NVIDIA (set the arch for your card, e.g. Hopper H100 = ADA/HOPPER90)
cmake -B build_cuda -DCMAKE_BUILD_TYPE=Release \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON

# GPU — AMD MI300A (APU, gfx942, unified memory); needs ROCm/hipcc on PATH
cmake -B build_hip -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DKokkos_ENABLE_HIP=ON -DKokkos_ARCH_AMD_GFX942_APU=ON
# (discrete MI300X: drop the _APU suffix -> -DKokkos_ARCH_AMD_GFX942=ON)
```

FFTs run through kokkos-fft on the active backend (FFTW on host, hipFFT/cuFFT on
GPU) — used both for the one-time noise setup and for the per-iteration L-BFGS
preconditioner. HDF5 I/O stays on the host. All host access to device data goes
through `create_mirror_view`/`deep_copy`, so the code is correct on discrete and
unified (APU) memory.

The same kernels run on whichever backend is selected. **Tests run on the
default execution space**, so an OpenMP build runs them multi-threaded and a GPU
build runs them on the device — the suite passes in all three (Serial, OpenMP,
GPU-ready). The determinism tests double as a race / reproducibility check under
threading. Set `OMP_NUM_THREADS` to pick the thread count (and `OMP_PROC_BIND` /
`OMP_PLACES` for best performance, as Kokkos recommends).

> Note: HighFive 2.x declares `cmake_minimum_required` < 3.5, which CMake ≥ 4
> rejects; the root `CMakeLists.txt` sets `CMAKE_POLICY_VERSION_MINIMUM=3.5`
> (scoped to the HighFive fetch) to work around this.

## Running the simulations

Each executable is configured by module-level constants at the top of its
`main.cpp` (no command-line arguments), matching the Python convention.

```sh
./build/executables/qew_static/qew_static        # -> static_arclength.h5
./build/executables/qew_dynamic/qew_dynamic      # -> dynamic_arclength.h5
./build/executables/qew_depinning/qew_depinning  # -> depinning_arclength.h5
```

| Executable | Method | Mirrors |
|---|---|---|
| `qew_static` | L-BFGS minimization, sweep pinning length | `qew_static.py` |
| `qew_dynamic` | annealed Langevin (Euler–Maruyama) | `qew_dynamic.py` |
| `qew_depinning` | Rosso–Krauth force ramp toward `f_c` | `qew_depinning.py` |

## Analysis stays in Python

The executables write only the relaxed profiles `h` (+ metadata) under the same
HDF5 keys the Python scripts use. To add the analysis datasets
(`ell`/`acf`/`q`/`psd`) that the figure script reads — computed by the validated
`../Overleaf/qEW/qew_analysis.py` — run:

```sh
python3 tools/postprocess_static.py static_arclength.h5
```

The augmented file is then readable by `../Overleaf/qEW/plot.py` unchanged.
(Despite the name, `postprocess_static.py` works on any of the output files —
it processes every `.../h` dataset.)

## Source layout

```
src/
  qew_types.h          real_t, execution/memory spaces, View aliases, NoiseSample
  qew_model.h          objective() + gradient(), linear & arclength (templated on noise)
  filtered_noise.{h,cpp}  random-phase FFT noise + periodic cubic-B-spline prefilter (host);
                          device bicubic sample() with analytic derivatives
  fire.h               FIRE minimizer (fallback / cross-check)
  lbfgs.h              default static minimizer (Nocedal two-loop + Armijo + curvature skip
                       + optional inverse-Laplacian FFT preconditioner via kokkos-fft)
  dynamics.h           overdamped Langevin (Kokkos RNG)
  rosso_krauth.h       depinning: per-site velocity-zero + red-black forward relaxation
executables/<name>/    one main.cpp per simulation (+ the original `main` skeleton)
tests/                 GoogleTest suite; tests/data/ holds golden fixtures + their gen_*.py
tools/postprocess_static.py   add acf/psd to an output file (uses the Python analysis)
```

## Validation

Correctness is pinned by:

- a **gradient-consistency** test (analytic gradient vs finite-difference of the
  objective, both models);
- **golden-file** cross-checks against SciPy for the noise interpolation and for
  the L-BFGS minimum (regenerate with `tests/data/gen_*.py`);
- an **analytic depinning threshold** (washboard potential, `f_c = A·k`) for
  Rosso–Krauth;
- equipartition `⟨E⟩ = (N/2)kT` for the Langevin integrator.

**Stiff lines:** the elastic term is the ill-conditioned discrete Laplacian
(condition ~ N²), which stalls plain L-BFGS/FIRE. `qew_static` enables the
inverse-Laplacian **FFT preconditioner** (`LbfgsParams::precondition`, via
kokkos-fft `rfft`/`irfft` with reused plans), which restores fast convergence.
