# Plan: C++/Kokkos GPU port of the Python qEW model

Port the Python quenched Edwards–Wilkinson (qEW) line model in
`../Overleaf/qEW/` to C++ with [Kokkos](https://kokkos.org/) so that the heavy
runs (energy minimization, depinning, Brownian dynamics) execute on a GPU.
The port lives in this directory (`qEW/`) and reuses the existing CMake
skeleton.

## Goals and non-goals

- **Goal:** GPU-resident inner loop (gradient/objective evaluation) for the four
  Python drivers — `qew_static`, `qew_static_ensemble`, `qew_depinning`,
  `qew_dynamic` — producing `.h5` output that the existing, validated
  `../Overleaf/qEW/plot.py` and `qew_analysis.py` can read unchanged.
- **Goal:** a test suite (GoogleTest) that pins the C++ results to the published
  Python results via golden-file fixtures.
- **Non-goal:** MPI / multi-node. Single-node, single-GPU (or CPU) only. MPI has
  been removed from the CMake scripts.
- **Non-goal:** bit-exact reproduction of NumPy/SciPy RNG and interpolation;
  correctness is validated statistically and against golden tolerances.

## Dependencies

| Dependency | Status | Why |
|---|---|---|
| **Kokkos 5.1.1** | keep | On-node parallelism + GPU backend (CUDA/HIP). The only compute dependency. Requires **C++20** (project standard bumped accordingly). |
| **Eigen3** | **dropped** | Not needed — see below. Removed from `CMakeLists.txt`. |
| **MPI** | **dropped** | Out of scope. Removed from `CMakeLists.txt`, `main.cpp`, test harness. |
| kokkos-fft v1.1.0 (FetchContent) | **add** | FFTs on Kokkos Views, on whatever backend the Views live on. Used for noise generation/prefilter (setup) **and** the L-BFGS preconditioner (hot loop, on-device). Replaces pocketfft. Backends: **FFTW** on host (needs `libfftw3-dev`), **hipFFT/rocFFT** on HIP (MI300A), cuFFT on CUDA. Must be fetched *after* Kokkos. |
| HDF5 + HighFive (FetchContent, optional) | **add** | Read/write `.h5` matching the Python key strings so `plot.py` works unchanged. Guarded so the build still configures without system HDF5. |
| GoogleTest | keep | Already wired in `tests/`. |

### Why Eigen3 can be dropped

Every numerical operation in the Python code maps onto Kokkos Views and
parallel patterns — there are no dense/sparse solves, decompositions, or
eigenvalue problems. The state is a single length-`N` line `h(x)`.

| Python operation | Maps to |
|---|---|
| Element-wise stencils (`np.roll`, central differences) | Kokkos `parallel_for` over a 1-D `View` |
| Sum reductions (energy, FIRE dot products, force norms) | Kokkos `parallel_reduce` |
| Thermal forcing (`rng.standard_normal`) | `Kokkos::Random_XorShift64_Pool` |
| 2-D inverse FFT for the noise field (once, at setup) | host FFT (pocketfft / FFTW) |
| Cubic-spline prefilter + bicubic interpolation | host prefilter (setup) + `KOKKOS_INLINE_FUNCTION` evaluator (device) |
| 1-D FFT for ACF/PSD (post-processing) | host FFT, or kept in Python |

The FFT is a **setup-only** step (build the correlated noise field once); it
never appears in the hot loop, so no GPU FFT is required — only a host FFT.

### Decisions

1. **FFT backend: pocketfft** (FetchContent, header-only). Host-side, setup-only.
2. **Analysis stays in Python.** The C++ code writes `h` (and trajectories) to
   `.h5`; ACF/PSD/Hurst are computed by the existing, validated
   `../Overleaf/qEW/qew_analysis.py` + `plot.py`. The `src/analysis.{h,cpp}`
   entry below is therefore deferred indefinitely (only revisit if fully
   self-contained C++ runs become necessary).

## Source layout (extends the existing skeleton)

```
qEW/src/
  qew_types.h            # real_t=double; ExecSpace/MemSpace; View1D/View2D aliases
  filtered_noise.{h,cpp} # host: random-phase FFT noise + periodic cubic-spline prefilter
                         #       device: KOKKOS_INLINE_FUNCTION bicubic eval -> (v, dv_dx, dv_dy)
  qew_model.{h,cpp}      # objective() [parallel_reduce]; gradient() [parallel_for]
                         #   linear + arclength elastic terms, kept in sync (mirror qew_core.py)
  fire.{h,cpp}           # FIRE relaxation (general-purpose minimizer; statics)
  rosso_krauth.{h,cpp}   # variant-MC critical-configuration solver (depinning) — see below
  dynamics.{h,cpp}       # Euler–Maruyama with annealed thermal_length + Kokkos RNG
  io.{h,cpp}             # HighFive read/write, Python-compatible HDF5 keys
  analysis.{h,cpp}       # OPTIONAL host ACF/PSD/Hurst (Phase 7+, only if not deferred to Python)
qEW/executables/
  qew_static/            # FIRE sweep over pinning_length        -> static_{model}.h5
  qew_depinning/         # Rosso–Krauth + bisection on f_c        -> depinning_{model}.h5
  qew_dynamic/           # Brownian dynamics                      -> dynamic_{model}.h5
qEW/tests/               # GoogleTest, run on the default exec space (CPU and GPU)
  data/                  # golden HDF5 fixtures exported from Python
```

Configuration mirrors the Python convention: module-level constants → a small
C++ `struct Params` set in each executable's `main` (no CLI parser needed
initially). HDF5 dataset keys must match the literal format strings in
`plot.py` (`driving_force={:.1g}/pinning_length={:.3g}/h`, etc.) — these are the
only coupling between writer and reader.

## The hot loop (what actually runs on the GPU)

Both the minimizer and the dynamics reduce to repeated evaluation of the
gradient (`qew_gradient` in `qew_core.py:96`) and objective
(`qew_objective:80`). Every element is independent:

```
gradient[i] = line_force(h, i)                      # 3-point periodic stencil
            + 0.5 * (noise_dy@left_mid + noise_dy@right_mid)   # bicubic interp
            - driving_force
```

- `line_force`: `linear` → `-kappa (h[i-1]+h[i+1]-2h[i])/dx^2`; `arclength` →
  the slope-normalized difference in `qew_core.py:101-111`.
- noise terms: bicubic interpolation of the prefiltered noise grid at the
  edge midpoints `((x[i]+x[i±1])/2, (h[i]+h[i±1])/2)`, periodic-wrapped.

This is an embarrassingly parallel `parallel_for`; the objective is the same
stencil summed in a `parallel_reduce`. Keeping `objective` and `gradient`
analytically consistent is the single most important invariant (enforced by the
gradient-consistency test, mirroring `test_gradient` at `qew_core.py:126`).

## Minimizer / propagation algorithm

Three algorithms, each for the job it is correct for.

### L-BFGS — the production static minimizer (Phase 5.5, replacing FIRE)

The GPU-friendly stand-in for SciPy's `fmin_l_bfgs_b` in `qew_static` /
`qew_static_ensemble`. FIRE (below) works but its iteration count grows badly
with stiffness (lp=100 took ~190k iters); L-BFGS converges in ~hundreds.

- **Two-loop recursion** (Nocedal & Wright Alg. 7.4) for the direction
  `d = -H g`: history `s(m,N)`, `y(m,N)` as LayoutRight `View2D` ring buffers;
  the `m≈8` dot-products are `parallel_reduce`, the AXPYs `parallel_for`, scalars
  on host. Initial-Hessian scaling `H0 = (s·y)/(y·y) I`.
- **Line search: backtracking Armijo** (sufficient decrease, `c1≈1e-4`, α0=1,
  halving) — function-evaluations only (cheaper than Wolfe, which needs a
  gradient per trial; our gradient is ~2× the objective), monotone and
  non-overshooting (the observed stability), and matches the "relax to the
  nearest minimum" semantics.
- **Curvature safeguard (REQUIRED with Armijo):** skip the update when
  `s·y ≤ ε‖s‖‖y‖` (or Powell-damp). Armijo alone does *not* guarantee `s·y>0`
  (that is the Wolfe curvature condition's job); the skip keeps `H` positive
  definite. With it, Armijo+L-BFGS is robust (as in LAMMPS/ASE).
- **Unbounded** (decided): the periodic-wrap noise lets `h` roam, as in
  `qew_depinning.py`; a global offset doesn't affect roughness/ACF. (`qew_static`'s
  `[0,Ly]` box is therefore not enforced; full L-BFGS-B is avoided.)

#### FFT preconditioner (for stiff lines)

Stiff lines are dominated by the discrete Laplacian (Hessian eigenvalues
`(κ/Δx)·4 sin²(πk/N)`, condition ~ N²), which stalls unpreconditioned L-BFGS.
Precondition with the inverse elastic Hessian, diagonalised by the FFT:

```
M⁻¹ q  =  irfft( rfft(q) / ( (κ/Δx)·4 sin²(πk/N) + m ) )
```

- Replaces the scalar initial Hessian `H₀ = γI` in the two-loop recursion with
  `r = M⁻¹ q`; the `(s,y)` history then corrects `M⁻¹` toward the true `H⁻¹`.
  Everything else (Armijo, the `s·y>0` skip) is unchanged.
- `m` is a shift regularising the `k=0` zero mode (uniform translation); default
  to the smallest non-zero elastic eigenvalue (`k=1`).
- Exact for `linear`; for `arclength` it uses `κ=line_tension` (near-exact for
  stiff/flat lines, corrected by the history otherwise).
- **On-device via kokkos-fft** with *reused* `Plan`s (rfft/irfft built once,
  `execute`d each iteration — no per-iteration plan/FFTW-planning cost). 1-D real
  FFT of length N per iteration: O(N log N).
- Line-search failure → reset history to steepest descent; persistent failure →
  terminate.

### FIRE — retained fallback / cross-check

Fast Inertial Relaxation Engine (`src/fire.h`): force evals + reductions +
velocity-Verlet updates, all native Kokkos, converges on `max|force| < tol`.
Kept as a tested reference and a robustness fallback; L-BFGS is the default.

Caveat (both minimizers): the qEW energy landscape has many local minima, so
different minimizers from a flat start can settle in *different* minima.
End-to-end tests compare **statistical** observables (roughness `R₂(ℓ)`, Hurst
`H`), not the pointwise `h`.

### Rosso–Krauth — for the depinning runs

`PhysRevE.65.025101.pdf` (Rosso & Krauth 2002) describes the variant-Monte-Carlo
/ no-passing algorithm that computes the **critical depinning configuration**
`h^c` and critical force `f_c` directly, exploiting Middleton's no-passing
theorem instead of simulating the (critically slow) dynamics.

**Why it is the right tool here — not just an optimization.** The referee
request driving `qew_depinning.py` is precisely the approach to the depinning
exponent (H ≈ 1.25) and the two length scales. Energy minimizers (FIRE,
L-BFGS-B) find the **zero-force inherent structure** (statics, H ≈ 1 for the
linear model) — *not* the depinning critical state. `qew_depinning.py` works
around this with a quasi-static force ramp, but that ramp suffers from critical
slowing-down as `f → f_c` (the very problem Rosso & Krauth call "extremely
tedious"). Rosso–Krauth finds `f_c` and `h^c` rigorously and to high precision
(the paper reaches L = 2048, ζ = 0.388 ± 0.002). So for depinning it is
**scientifically better**, not merely faster.

**Algorithm sketch (adapted to our model).**
- Construct a forward-moving, non-crossing string. Advance one coordinate `i`
  at a time from `h_i` to the **closest** position where its local velocity
  `v(h_i) = -∂E/∂h_i` vanishes.
- Because the disorder is a cubic spline in `h`, the local energy is cubic in
  `h_i`, so `v(h_i) = 0` is solvable in closed form — a **quadratic** for our
  *linear* (harmonic, nearest-neighbor) elastic term. Pick the nearest forward
  root.
- Sweep until the whole string is blocked (terminate when the largest velocity
  drops below a single small threshold — the only free parameter). An outer
  **bisection on `f`** locates `f_c`.

**Is it as simple to implement as FIRE? — No, modestly more involved.** Three
points of friction, none fatal:

1. *Parallelism.* FIRE is fully data-parallel (every site updates from the
   global gradient — trivial on a GPU). The faithful Rosso–Krauth update is a
   **sequential single-site sweep** (Gauss–Seidel-like), which is not naturally
   GPU-parallel. **Good news for our case:** our elastic term is
   *nearest-neighbor* (short-range), unlike the *long-range* string in the
   paper, so a **red–black / checkerboard** sweep decouples within a color and
   maps cleanly onto Kokkos `parallel_for` while preserving the monotone
   no-crossing construction. (The paper's long-range kernel is what makes
   parallelization hard there; we avoid that.)
2. *Closed-form solve is model-dependent.* The quadratic root trick is clean for
   the **linear** model. The **arclength** model's elastic term is nonlinear in
   `h_i`, so `v(h_i) = 0` needs a cheap local 1-D Newton/bisection per site
   instead of a closed form. Still local and fast, but not the elegant quadratic.
3. *Bookkeeping.* Our noise force samples the bicubic at edge midpoints
   `(h_i+h_{i±1})/2` (a half-factor and two midpoints), and we need the spline
   cell index per site — more bookkeeping than the original per-site 1-D
   potential in the paper.

Net: the per-site math is simple (arguably simpler than FIRE's restart
bookkeeping), but the sequential→checkerboard reformulation, the per-model root
solve, and the `f_c` bisection make it a larger implementation than FIRE.

**Decision:** implement **FIRE first** (covers statics + de-risks all shared
infrastructure: model, noise, I/O, GPU parity), then add **Rosso–Krauth** for
the depinning executable. Keep the FIRE-based force-ramp available as an
independent cross-check of `f_c`.

## Dynamics

`qew_dynamic.py` is explicit Euler–Maruyama with an annealed `thermal_length`.
Port directly: `parallel_for` for the deterministic drift (`-gradient`),
`Kokkos::Random_XorShift64_Pool` for the `√(2 dt/τ)` thermal kick. No minimizer
needed.

## Test plan

Each unit test runs on the **default execution space**, so a CUDA/HIP build
exercises the GPU path automatically.

1. **Gradient consistency** (port of `test_gradient`, `qew_core.py:126`):
   finite-difference of `objective` vs. analytic `gradient`, both `linear` and
   `arclength`, random `h`. The primary correctness gate.
2. **Noise statistics:** generated field has `std == amplitude`; Fourier modes
   outside the correlation ellipse are zero; field is periodic.
3. **Bicubic interpolation:** reproduces grid values at nodes; analytic
   `dv/dx, dv/dy` match finite differences of the interpolant; matches a
   **golden** noise field + interpolated values exported from Python within
   tolerance (covers the SciPy `mode='wrap'` boundary subtlety).
4. **Objective/gradient golden test:** load a fixed `(noise, h)` exported from
   Python via HDF5; compare C++ `objective`/`gradient` to Python values.
5. **FIRE minimizer:** noise-free tilted line converges to the analytic flat
   minimum, `max|force| → 0`, energy monotonically non-increasing.
6. **Rosso–Krauth:** on a small spline potential, the closest-velocity-zero
   solve and the no-crossing/monotone advance are verified; `f_c` from the
   bisection is reproducible; `h^c` is a blocked configuration (all velocities
   ≤ threshold). Cross-check `f_c` against the FIRE force-ramp.
7. **CPU/GPU parity:** identical inputs → identical results (FP tolerance) on
   `Serial`/`OpenMP` vs. `Cuda`.
8. **Dynamics limits:** `thermal_length → 0` reduces to gradient descent (energy
   decreases); a pure-harmonic case reproduces equipartition `⟨h²⟩ = k_BT/k`.
9. **End-to-end statistical:** small grid with an *imported* Python noise field;
   C++ `R₂(ℓ)`/Hurst within statistical tolerance of `qew_static.py` (FIRE) and
   the depinning exponent trend (Rosso–Krauth).

**Golden-file strategy (tests 3, 4, 9):** a small Python helper dumps reference
`noise`, `h`, `objective`, `gradient`, and `R₂` to HDF5 fixtures committed under
`qEW/tests/data/`. This pins the C++ port to the already-published Python
results.

## CMake / build changes

- **Done:** removed MPI and Eigen3 from `CMakeLists.txt`; dropped
  `MPI::MPI_CXX`/`Eigen3::Eigen` from `src/CMakeLists.txt`; switched the test
  harness to a Kokkos-only `gtest_kokkos_main.cpp` (no MPI); removed MPI from
  `executables/main.cpp`. (`gtest_mpi_main.cpp`, `gtest-mpi-listener.hpp` are now
  unused and may be deleted.)
- **Done (Phase 1):** pocketfft (FetchContent, header-only INTERFACE target,
  header is `pocketfft_hdronly.h`); HDF5 + HighFive v2.10.0 via FetchContent,
  guarded by `find_package(HDF5)` (present on this machine, so enabled).
  HighFive 2.x needs the `CMAKE_POLICY_VERSION_MINIMUM=3.5` escape hatch under
  CMake 4. Project standard is **C++20** (Kokkos 5.x requirement).
- **To add later:** document `-DKokkos_ENABLE_CUDA=ON` (or `HIP`) as the GPU
  configure option. Default build stays CPU (`Serial`) so `ctest` passes
  everywhere.
- New executables + `add_subdirectory` entries; new test sources appended to
  `TEST_SOURCES`.

## Phased roadmap

1. **Scaffolding — DONE.** MPI/Eigen removed; Kokkos 5.1.1 + pocketfft + guarded
   HDF5/HighFive wired; `qew_types.h`; Kokkos `parallel_reduce` smoke kernel +
   pocketfft round-trip test. Green `ctest` (CPU/Serial).
2. **Model core — DONE.** `src/qew_model.h` (objective via `parallel_reduce`,
   gradient via `parallel_for`, both `linear` and `arclength`, periodic-wrap
   bookkeeping matching `qew_core.py`); `tests/test_qew_model.cpp`
   gradient-consistency test passes for both models (FD vs analytic, max diff
   < 1e-6). *Gate passed.*
3. **Noise — DONE.** `src/filtered_noise.{h,cpp}`: host random-phase FFT field
   (pocketfft) + **exact periodic FFT-based cubic B-spline prefilter** (division
   by the separable transfer function `2/3 + (1/3)cos(2πm/N)`, equivalent to
   SciPy `grid-wrap`); device `DeviceNoise::sample` does periodic bicubic with
   **analytic** derivatives (exact, unlike Python's finite-difference). Tests
   (`test_filtered_noise.cpp`): amplitude+band-limiting, node-reproduction,
   analytic-vs-FD derivatives, model gradient-consistency with real noise, and a
   SciPy **golden** cross-check (`tests/data/gen_golden.py` →
   `noise_golden.h5`, value <1e-7, derivatives <1e-4). 10/10 ctest green.
4. **FIRE — DONE.** `src/fire.h` (semi-implicit-Euler FIRE, all Kokkos
   reductions/updates) + `executables/qew_static/` (sweeps pinning length, FIRE
   from a flat start, writes `static_{model}.h5`) + `tools/postprocess_static.py`
   (Python adds `ell`/`acf`/`q`/`psd` so `plot.py` reads the file unchanged).
   Tests (`test_fire.cpp`): flat relaxation, harmonic-well minimum (known
   target), energy-decrease, determinism. 14/14 ctest green. Verified run: all 5
   pinning lengths converge (max|F|<1e-7) and Hurst rises with stiffness
   (lp=0.01→H≈0.15, lp=100→H≈0.81), matching the arclength-model trend.
   *Perf note:* FIRE iteration count grows with stiffness (lp=100: ~190k iters,
   ~1 min single-thread Serial); GPU/OpenMP builds and, if needed, a CG/L-BFGS
   path for very stiff lines are future optimisations. A tight bootstrap-ensemble
   statistical cross-check vs Python (test 9) remains for later.
5. **Dynamics — DONE.** `src/dynamics.h` (`LangevinDynamics`: Euler–Maruyama
   overdamped Langevin with `Kokkos::Random_XorShift64_Pool`; explicit drift/diff
   coefficients, `kT = diff²/(2·drift)`) + `executables/qew_dynamic/` (annealed
   thermal length, logarithmic-time snapshots → `dynamic_{model}.h5`). Tests
   (`test_dynamics.cpp`): zero-temperature step = monotone gradient descent, and
   **equipartition** `⟨E⟩ = (N/2)kT` for the quadratic case. 16/16 ctest green.
   Verified run roughens a flat line (std 0→0.155 over the anneal, mean
   conserved); `postprocess_static.py` augments the output for `plot.py`.
5.5. **L-BFGS (replace FIRE) — DONE.** `src/lbfgs.h`: Nocedal two-loop recursion
   (`m=8`, history as `std::vector<View1D>`, dot-products `parallel_reduce`,
   AXPYs `parallel_for`) + backtracking **Armijo** line search + `s·y>0` skip
   safeguard + `H0=(s·y)/(y·y)` scaling; **unbounded**. Now the default in
   `qew_static` (FIRE kept as fallback). Tests (`test_lbfgs.cpp`): flat
   relaxation, harmonic-well minimum, fast-on-quadratic (<200 iters),
   FIRE-agreement on a convex case, and a **SciPy `fmin_l_bfgs_b` golden**
   (`tests/data/gen_lbfgs_golden.py` → `lbfgs_golden.h5`; matched min energy).
   21/21 ctest green. Verified run: gives the **same Hurst exponents as FIRE**
   (lp=100→H≈0.814) with **far fewer iterations** on floppy/moderate lines
   (lp≤10: 250–1034 iters vs FIRE's 1140–18424). *Known limitation:* the very
   stiff lp=100 line is the ill-conditioned discrete Laplacian (cond ~ N²) and
   unpreconditioned L-BFGS plateaus at ~5e-6 max-force (doesn't reach 1e-6) —
   physically irrelevant for roughness (Hurst matches), but a FFT/Laplacian
   **preconditioner** is the proper fix (future work; FIRE had the same trouble,
   190k iters).
6. **Rosso–Krauth depinning — DONE.** `src/rosso_krauth.h`: single-site velocity
   (= -qew_gradient_i), numerical nearest-forward-zero finder (march < noise cell
   + bisect; serves both elastic models), and a **red-black (checkerboard)
   forward relaxation** that blocks the line or detects runaway. `qew_depinning`
   executable ramps the force (restart from previous config), brackets `f_c`,
   writes `depinning_{model}.h5`. Tests (`test_rosso_krauth.cpp`): **analytic
   washboard `f_c = A·k`** recovered to <2% by bisection, a direct equivalence
   check that `rk_site_velocity == -gradient_i`, forward/arrested blocked config,
   determinism. 25/25 ctest green. Verified run (stiff lp=5):
   blocks for f≤18, **critical slowing-down** as f→f_c (sweeps 9k→124k), depins
   at f=19 (f_c∈(18,19)); Hurst **rises toward depinning** with force
   (H: 0.55 at f=0 → 0.89 at f=18) — the referee's H→1.25 trend (asymptotic
   value needs f closer to f_c + larger grid, i.e. production runs).
7. **I/O + docs — DONE.** HDF5 keys match `plot.py` (writer/postprocess);
   `qEW/README.md` rewritten (what it computes, deps, CPU/GPU build, running the
   executables, the postprocess→plot.py workflow, source layout, validation,
   the stiff-Laplacian caveat) and the root `CLAUDE.md` C++ section updated
   (Kokkos 5.1.1/C++20, no MPI/Eigen, pocketfft + optional HDF5, file/role table,
   GPU configure flag). `qew_analysis.py` port stays deferred (analysis in
   Python by design).
8. **kokkos-fft migration — DONE.** pocketfft replaced by **kokkos-fft v1.1.0**
   everywhere: noise generation + cubic-B-spline prefilter (`filtered_noise.cpp`,
   `fft2`/`ifft2` on Views), the FFT smoke test (`fft`/`ifft`), and the
   band-limit check, all on the default execution space (host=FFTW, HIP=hipFFT).
   Fetched after Kokkos so it reuses `Kokkos::kokkos`. **Host builds need
   `libfftw3-dev`** (added to CI). 26/26 ctest green incl. the SciPy noise golden.
9. **FFT preconditioner for L-BFGS — DONE.** `lbfgs.h` has an optional
   inverse-Laplacian preconditioner (kokkos-fft `rfft`/`irfft`, reused `Plan`s)
   replacing the scalar `H₀`; enabled in `qew_static`. The shift `m` is the
   **pinning (non-elastic) curvature scale ~ dx·A/ξ²** (NOT just the zero-mode
   eigenvalue — that blows up the step on floppy lines); the executable sets it.
   Result: the stiff `lp=100` case that previously stalled (~5e-6 at 40k iters,
   190k for FIRE) now **converges in ~29 iterations**; the full sweep runs in
   ~5 s (was ~60–110 s). Same Hurst as before in the stiff regime. Tests:
   `Lbfgs.PreconditionerOnStiffLine` (stiff converges fast, fewer iters than
   unpreconditioned); the SciPy golden uses it too.
