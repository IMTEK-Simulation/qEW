# Optimization potential & known issues

Technical notes from a code review of the qEW port (June 2026). Items are
ordered by expected impact.

> **Status:** items 1–12 are addressed on this branch — see `fix_plan.md`
> for the phase breakdown and measured numbers. Item 12's GPU benefit
> (RNG lock relief) still needs verification on the CUDA runner; the minor
> items 13–15 are won't-fix unless profiling says otherwise. File references point at the current `main`-line
sources. Nothing here changes physics; everything preserves the documented
analytic consistency between `objective()` and `gradient()` and the golden-file
contracts.

The hot path, per the benchmark (`executables/benchmark`), is the bicubic noise
gather inside `gradient()` / `objective()` — memory-bandwidth-bound (16
coefficient loads + ~70 FLOPs per `DeviceNoise::sample()`). Most of the wins
below attack either (a) the number of `sample()` calls, or (b) per-iteration
kernel-launch / device-sync overhead in the minimizers.

## Bugs / correctness hazards

1. **Red–black sweep races for odd `n`** — `src/rosso_krauth.h:147`
   (`if ((i & 1) != color) return;`). On a periodic ring with odd `n`, sites
   `0` and `n-1` have the same parity but are nearest neighbours, so the same
   colour pass updates both while each reads the other's `h`. All current
   configs use even `nx`, so this is latent — but nothing enforces it.
   *Fix:* assert `n % 2 == 0` in `rk_block()` (cheap), or fall back to a
   3-colour sweep for odd `n`.

2. **Inconsistent pinning-length → line-tension map across run drivers** —
   `src/qew_runs.h`: `static_sweep()` and `depinning_ramp()` used
   `line_tension = pinning_length^1.5 * amplitude`, while `dynamic_anneal()`
   used `(pinning_length / xi)^1.5 * amplitude` — the same struct-field name
   meant two different physical quantities. **Resolved:** the paper
   (arXiv:2410.21838, eq. (10)) controls the model through the dimensionless
   Larkin ratio `lambda_p/xi = (Gamma/U0)^{2/3}`, so all drivers now take a
   `pinning_length_over_xi` field and share `Gamma = A * (lambda_p/xi)^{3/2}`;
   each driver's mapping is pinned by a unit test. HDF5 key strings are
   unchanged and their values carry the ratio (already true for static and
   depinning output; dynamic output previously recorded the physical length).

3. **Floating-point loop counter in the force ramp** —
   `src/qew_runs.h:214`: `for (real_t f = 0; f <= c.f_max; f += c.f_step)`.
   Exact for the default `f_step = 1.0`, but for a non-representable step
   (e.g. `0.1`) accumulated rounding can drop or shift the final step and
   perturb the HDF5 key strings (`driving_force=%.4g`). *Fix:* integer index,
   `f = k * f_step`.

4. **No default `CMAKE_BUILD_TYPE`** — root `CMakeLists.txt` never sets a
   fallback, so configuring without `-DCMAKE_BUILD_TYPE=Release` silently
   produces an unoptimized (`-O0`) build of a bandwidth-bound HPC code. *Fix:*
   default to `Release` when unset (warn). Cosmetic: the project is still
   named `my-fantastic-simulation-code` (line 5), and `hello.cpp` /
   `kokkos_smoketest.cpp` scaffolding is still compiled into the production
   `lib` target.

## Optimization potential

### Minimizer (L-BFGS, `src/lbfgs.h`)

5. **Redundant objective evaluation every iteration** — line 200:
   `phi0 = objective(h, ...)` is recomputed at the start of each line search,
   but `h` is exactly the point accepted by the previous iteration's line
   search, whose `phi` is already known. Carrying the accepted value forward
   removes one full objective (one `sample()` per site + reduction) per
   iteration — when the first trial step is accepted (the common,
   preconditioned case) this *halves* the objective-evaluation cost.

6. **Fuse the trial step into the objective reduce** — each Armijo trial does
   a `parallel_for` (write `h = x0 + a*d`, line 212) plus a `parallel_reduce`
   (objective). Evaluating the objective directly at `x0(i) + a*d(i)` inside
   the reduce, committing `h` only on acceptance, saves one kernel launch and
   `n` stores per trial and removes the `deep_copy(h, x0)` undo path
   (line 224) and the `deep_copy(x0, h)` save (line 199).

7. **Fewer reductions / device syncs per iteration** —
   - `sy`, `ss`, `yy` (lines 245–247) are three separate `parallel_reduce`s
     over the same two vectors; fuse into one reduce with a 3-component
     reducer. The `lbfgs_sy` `parallel_for` (line 240) can be folded in too.
   - The two-loop recursion issues `2m` blocking dot products per iteration,
     each a device→host sync; for the production `n = 4096` a GPU run is
     latency- rather than bandwidth-limited here. The `axpy` can be fused
     with the next loop's dot (`q -= a*y` and `s·q` in one kernel).
   - History update: `deep_copy(S[row], s_tmp)` / `deep_copy(Y[row], y_tmp)`
     (lines 256–257) copy two full vectors per iteration; `std::swap`-ing the
     `View`s into the slot is free. (The FFT plans are bound to `q`/`d`, not
     `s_tmp`/`y_tmp`, so swapping is safe.)

### Force kernel (`src/qew_model.h`)

8. **Each noise edge is sampled twice in `gradient()`** — site `i` samples the
   midpoints of edges `(i-1,i)` and `(i,i+1)` (lines 105–106), so every edge
   is evaluated by both adjacent sites: `2n` bicubic gathers for `n` distinct
   edges. A two-pass scheme — edge kernel writing `dv_dy` (and the arclength
   slope term) per edge into a scratch view, then a cheap combine kernel —
   does `n` gathers + `~3n` doubles of scratch traffic instead of `2n`
   gathers (16 loads each). Expect close to 2× on the dominant kernel.
   (`objective()` already samples each edge exactly once; `rk_site_velocity`
   re-evaluates at varying `h` and cannot share.)

9. **`sample()` computes derivatives the caller discards** —
   `objective()` uses only `s.v`, yet `DeviceNoise::sample()`
   (`src/filtered_noise.h:67`) always accumulates `dv_dgx` *and* `dv_dgy` and
   `bspline_weights()` always fills `dw[4]`. `gradient()` never uses `dv_dx`
   anywhere in the codebase. Splitting into value-only / value+dy variants
   (template tag or separate methods) trims ~⅓–½ of the FLOPs in the 4×4
   inner loop. Gain is modest where memory-bound, but the Armijo line search
   is objective-only and benefits directly.

10. **Integer modulo in the inner loops** — each `sample()` does 8 `%`
    operations for the neighbour wrap (`src/filtered_noise.h:79–80`), plus the
    stencils' `(i±1) % n`; integer division is expensive on GPUs. After the
    positive wrap, `ix ∈ [0, nx]` (the `gx == nx` rounding edge case is
    currently absorbed by `ix % nx`), so conditional add/subtract wraps are
    sufficient — keep an explicit clamp for that edge case. Power-of-two
    grids could use bit masks.

### Rosso–Krauth (`src/rosso_krauth.h`)

11. **Redundant mean-`h` reduction per sweep** — `rk_block()` runs two
    reductions per sweep (lines 161–167): the advance sum and a fresh mean of
    `h`. Since every move is forward, `mean_h = mean_h0 + Σ_sweeps
    total_adv / n`; tracking the cumulative advance eliminates the second
    reduction. Sweeps run up to `max_sweeps = 2·10⁵`, so this halves the
    per-sweep reduction/sync count.

### Langevin dynamics (`src/dynamics.h`)

12. **RNG pool acquisition per site per step** — `pool.get_state()` /
    `free_state()` inside the `parallel_for` (lines 48–50) acquires a locked
    state per site per step. With the benchmark's `nx = 262144` on a GPU the
    default-sized `Random_XorShift64_Pool` serializes on lock contention.
    Alternatives: size the pool to the launch width, or pre-fill a normal
    deviate view per step with `Kokkos::fill_random` and fuse the update.
    Also, the `gradient()` call inside `step()` inherits item 8's 2× edge
    duplication.

### Minor / one-time

13. **FIRE** (`src/fire.h`) — `nv2`/`nf2` are two separate reductions
    (lines 103–108) and the v-step (line 79) could fuse with the power
    reduction (line 85). Low priority: FIRE is the fallback minimizer.
14. **Noise setup** (`src/filtered_noise.cpp`) — phase generation, mean and
    variance are serial host loops over `nx·ny`; `build_coefficients()` stages
    through three complex 2-D buffers + host mirrors. One-time cost
    (`setup ≈` seconds in the benchmark), only worth touching if many seeds
    are ever swept in one run.
15. **`to_host_vec`** (`src/qew_runs.h:28`) — extra element-wise copy from the
    mirror into a `std::vector`; could `memcpy` from the mirror when layouts
    match. Negligible at current snapshot rates.

## Suggested order of attack

Items 5–7 (L-BFGS overhead) and 8 (edge-duplication) are independent,
low-risk, and cover the production `qew_static` runtime; the
gradient-consistency and golden-file tests pin their correctness. Item 11 is
a five-line change covering `qew_depinning`. Item 1 should be fixed (or
asserted away) before anyone runs an odd-`n` line. Items 2 and 4 are
correctness-of-use issues rather than speed, and are cheap insurance.
