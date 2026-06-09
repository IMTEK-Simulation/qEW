# Implementation plan for `optimization_notes.md`

> **Status (June 2026): all phases 0–4 are implemented on this branch.**
> Phase 1 was settled by the paper (arXiv:2410.21838 eq. (10):
> `lambda_p/xi = (Gamma/U0)^{2/3}`, varied by scanning Gamma at fixed
> disorder): all drivers now take the dimensionless ratio as
> `pinning_length_over_xi` and share `Gamma = A (lambda_p/xi)^{3/2}`, with
> per-driver mapping tests; line tensions are numerically unchanged.
> Measured on the benchmark (Serial): Phase 2 took the end-to-end
> preconditioned L-BFGS relaxation (n = 4096, arclength) from 5.73 s / 12893
> objective evals to 4.77 s / 10424 at an identical 2470 iterations; Phase 3
> took it to 3.55 s (38% total) and the hot kernels (n = 262144) from
> grad 33.5 / dyn 52.4 ms to 26.5 / 34.4 ms, with the L-BFGS trajectory
> bit-identical throughout. Phase 4's RK change removes one of two
> reductions per sweep; the Langevin RNG chunking is serial-neutral and its
> GPU benefit still needs the CUDA runner (see below). Phase 5 remains
> won't-fix unless profiling says otherwise.

Companion to [`optimization_notes.md`](optimization_notes.md); item numbers
below refer to that document. Work is grouped into five phases that are
independently land-able, ordered so that correctness fixes go first and each
perf change is benchmarked in isolation. Every phase ends with the full
`ctest` suite on a Serial **and** an OpenMP build (the threaded run doubles as
the race check), plus a before/after run of `executables/benchmark` for the
perf phases.

Ground rules:

- The gradient-consistency test and the golden-file tests are the correctness
  gates; they must pass unchanged (same tolerances) in every phase.
- Tests that assert `n_func_evals` counts will be updated *deliberately* in
  Phase 2, in the same commit as the behaviour change, with the new expected
  count derived by hand.
- One commit (or small commit series) per phase, so a regression bisects to a
  phase.

## Phase 0 — Safety & build hygiene (items 1, 3, 4)

Small, riskless, do first.

1. **Odd-`n` guard** (`rosso_krauth.h`): in `rk_block()`, before launching
   kernels, `throw std::invalid_argument` (host code) when `n % 2 != 0`,
   citing the red-black neighbour conflict. A 3-colour sweep for odd `n` is
   *not* worth the complexity until someone needs odd lines. Add a death/throw
   test.
2. **Integer ramp counter** (`qew_runs.h::depinning_ramp`): iterate
   `for (int k = 0; k * f_step <= f_max; ++k)` with `f = k * f_step`. The
   default `f_step = 1.0` produces bit-identical forces, so no test or HDF5
   key changes.
3. **CMake**: default `CMAKE_BUILD_TYPE=Release` when unset (with a STATUS
   message); rename the project to `qew`. Leave `hello.cpp` /
   `kokkos_smoketest.cpp` in place for now — they are referenced by tests and
   removing them is cosmetic churn; fold into a later cleanup if desired.

## Phase 1 — Decision point: line-tension convention (item 2)

Blocked on information outside the repo: the `../Overleaf/qEW/qew_*.py`
sources. **Needs Lars to confirm** which convention each Python driver uses.
Then, in one commit:

- Rename the field in whichever config struct holds the ratio to
  `pinning_length_over_xi` (or convert at the call site so all three drivers
  use one formula).
- Add a tiny unit test per driver pinning the `pinning_length →
  line_tension` mapping, so the convention can never silently diverge again.

Nothing later depends on this; it can land any time.

## Phase 2 — L-BFGS overhead (items 5–7)

All inside `lbfgs.h`, in three commits, benchmarking `qew_static`-shaped runs
(preconditioned, n = 4096) after each:

1. **Carry the objective forward** (item 5). Hoist one
   `phi = objective(h, ...)` above the iteration loop; on line-search
   acceptance set `phi = phi_accepted`; delete the per-iteration `phi0`
   recompute. The line-search-failure `continue` path restores `h = x0`, so
   the carried value stays valid there too. Arithmetic is unchanged →
   golden L-BFGS test passes bit-for-bit; only `n_func_evals` assertions in
   `test_lbfgs.cpp` need recounting.
2. **Fuse trial step into the objective** (item 6). Add a local helper
   `trial_objective(x0, d, alpha, p, noise)` — the `objective()` reduce with
   `h(i)` replaced by `x0(i) + alpha * d(i)` (and the same for the `ip`
   neighbour, which is the identical FP expression, so results stay
   bit-identical). Commit `h` with a single `parallel_for` only on
   acceptance; the `deep_copy(x0, h)` save and `deep_copy(h, x0)` undo both
   disappear. Keep `objective()` itself untouched (public API, used by
   tests/benchmark).
3. **Reduction fusion + history swap** (item 7).
   - One `parallel_reduce` with three reducers computing `sy`, `ss`, `yy`
     while also writing `s_tmp`/`y_tmp` (replaces one `parallel_for` + three
     reduces).
   - `std::swap(S[row], s_tmp)` / `std::swap(Y[row], y_tmp)` instead of
     `deep_copy` (safe: the FFT plans are bound to `q`/`d` only).
   - Fold `q -= a*y` into the subsequent dot where profiling on a GPU build
     shows the sync count matters; otherwise skip — this is the only
     speculative sub-item, and it slightly obfuscates the two-loop recursion.

## Phase 3 — Hot-kernel work reduction (items 8, 9, 10)

The order matters: land 10 (mechanical, bit-identical) first so the bigger
rewrites diff cleanly against it.

1. **Modulo removal** (item 10, `filtered_noise.h::sample`,
   `qew_model.h`, `rosso_krauth.h`): replace `(i±1) % n` stencil wraps with
   compare-and-adjust; in `sample()`, clamp the `gx == nx` rounding edge case
   explicitly (`if (ix >= nx) { ix -= nx; gx -= nx; }` before computing
   `fx`), then build the four neighbour indices with conditional wraps.
   Same arithmetic → golden noise-interpolation test must pass bit-for-bit.
2. **Value-only / dy-only sampling** (item 9). Add to `DeviceNoise`:
   `sample_value(x, y)` (no `dw`, no derivative accumulators) and keep
   `sample()` as-is. In `qew_model.h`, route calls through C++20
   `requires`-detected helpers (`noise_value(noise, x, y)` falls back to
   `noise.sample(x, y).v` for the analytic test stubs, so no test functor
   has to change). Split `bspline_weights` into weights-only and
   weights+derivatives variants. `objective()` switches to the value path.
3. **Edge-pass gradient** (item 8). Introduce
   `struct GradientWorkspace { View1D edge; }` (sized lazily to `n`).
   `gradient(h, p, noise, grad, ws)` becomes two kernels:
   - edge kernel: for edge `(i, i+1)` store
     `edge(i) = 0.5 * dv_dy(midpoint)` — plus, for the arclength model, the
     normalized slope `dh/sqrt(1+dh²)`, packed into a second view or a 2-D
     workspace, since both adjacent sites reuse it;
   - combine kernel: `grad(i) = dx * (line_force(from edge slopes or h) +
     edge(im) + edge(i) - f)`.
   Keep a back-compat `gradient(h, p, noise, grad)` overload that allocates a
   workspace internally (used by tests); `lbfgs_minimize`, `fire_minimize`,
   and `LangevinDynamics` own a persistent workspace. Floating-point sums
   re-associate slightly (`0.5*(sl+sr)` becomes `0.5*sl + 0.5*sr`), so the
   gradient-consistency test (tolerance-based) is the gate, and the golden
   L-BFGS comparison tolerance may need a documented epsilon bump — verify
   it stays within the existing tolerance first.
   `rk_site_velocity` is intentionally untouched (single-site re-evaluation
   cannot share edges).

Acceptance: benchmark `grad` ms/eval roughly halves on OpenMP; `obj`
improves measurably from item 9; no test tolerance loosened beyond what is
justified in the commit message.

## Phase 4 — Rosso–Krauth & dynamics (items 11, 12)

1. **Drop the per-sweep mean reduction** (item 11): track
   `cum_adv += total_adv` and compute `res.mean_h = mean_h0 + cum_adv / n`.
   Bit-identical apart from summation order of the mean; the RK tests
   (washboard `f_c`, runaway) are tolerance-based and unaffected.
2. **Langevin RNG** (item 12): measure first. On the OpenMP build the pool
   is per-thread and fine; the concern is GPU lock contention. Run the
   benchmark's `dyn` timing on the CUDA CI runner; only if `dyn − grad`
   overhead is significant, switch to a pre-generated normal-deviate view
   filled by a dedicated kernel where each thread owns one pool state for the
   whole row-chunk (avoids per-element lock churn), fused with the update
   step. The equipartition test gates correctness; exact trajectories may
   change (different draw order) — that test is statistical, so no fixture
   updates expected.

## Phase 5 — Optional cleanups (items 13–15)

FIRE reduction fusion, host-side noise-setup loops, `to_host_vec` memcpy.
Do only if profiling ever shows them; otherwise close as won't-fix and note
that in `optimization_notes.md`.

## Sequencing summary

| Phase | Contents | Risk | Blocked on |
|---|---|---|---|
| 0 | odd-`n` guard, ramp counter, CMake | none | — |
| 1 | line-tension convention | none | Python sources (Lars) |
| 2 | L-BFGS: carry φ, fused trial, fused reduces, view swap | low | — |
| 3 | modulo removal → sample variants → edge-pass gradient | medium | — |
| 4 | RK mean tracking; Langevin RNG (measure-first) | low | GPU runner for 12 |
| 5 | FIRE / setup / copy cleanups | — | profiling evidence |

Phases 0, 2, 3, 4 are mutually independent except that Phase 3's edge-pass
should rebase on Phase 2's line-search changes (both touch the `lbfgs.h`
call sites). Suggested landing order: 0 → 2 → 3 → 4, with 1 whenever the
Python check happens.
