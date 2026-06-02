#ifndef QEW_ROSSO_KRAUTH_H
#define QEW_ROSSO_KRAUTH_H

#include <Kokkos_Core.hpp>

#include "qew_model.h"
#include "qew_types.h"

// Rosso-Krauth depinning solver (PhysRevE.65.025101.pdf): instead of simulating
// the (critically slow) dynamics, construct the blocked/critical configuration
// directly by a no-passing, forward-only relaxation -- advance each site to its
// NEAREST forward velocity-zero, sweep until the whole line is blocked.
//
// The site velocity is exactly v_i = -dE/dh_i (the qew_gradient term for site i),
// so a blocked configuration (all v_i = 0) is a metastable state of the SAME
// energy as the minimizer. With nearest-neighbour coupling the velocity of site
// i depends only on h_{i-1}, h_i, h_{i+1}, so a red-black (checkerboard) sweep
// updates a whole colour in parallel while its neighbours are fixed -- a
// Gauss-Seidel-by-colour scheme that preserves the monotone construction and
// maps onto Kokkos/GPU. (The paper's hard, long-range case does not decouple;
// ours, being short-range, does.)
//
// For a given driving force f the construction either BLOCKS (a metastable state
// exists, f < f_c) or RUNS AWAY (the line advances without bound, f > f_c). An
// outer bisection / force ramp over f then locates f_c and the critical line.
//
// The nearest forward zero is found numerically (march in steps < the noise
// cell, then bisect) rather than by the paper's closed-form quadratic, so the
// same code serves both the linear and arclength elastic models.
namespace qew {

struct RkParams {
    real_t dstep = 0;          // march step for the forward-zero search (< cell)
    real_t max_advance = 0;    // cap on a single site's forward move per sweep
    real_t root_tol = 1e-10;   // bisection tolerance on h
    real_t sweep_tol = 1e-9;   // blocked when the sweep's total advance < this
    real_t runaway = 0;        // unblocked once mean(h) advances this far
    int max_sweeps = 200000;
};

struct RkResult {
    bool blocked = false;
    bool runaway = false;
    int sweeps = 0;
    real_t total_advance = 0;  // advance in the last sweep
    real_t mean_h = 0;
};

// Single-site velocity v_i(h_i) with neighbours fixed: exactly -qew_gradient_i.
template <class Noise>
KOKKOS_INLINE_FUNCTION real_t rk_site_velocity(int i, int n, real_t dx, real_t lt,
                                               real_t f, Model model, real_t hi,
                                               real_t hl, real_t hr,
                                               const Noise &noise) {
    const int im = (i - 1 + n) % n;
    const int ip = (i + 1) % n;

    real_t line_force;
    if (model == Model::Linear) {
        line_force = -lt * (hl + hr - static_cast<real_t>(2) * hi) / (dx * dx);
    } else {
        const real_t dhr = (hr - hi) / dx;
        const real_t dhl = (hi - hl) / dx;
        line_force = lt *
                     (dhl / Kokkos::sqrt(static_cast<real_t>(1) + dhl * dhl) -
                      dhr / Kokkos::sqrt(static_cast<real_t>(1) + dhr * dhr)) /
                     dx;
    }

    const real_t xcl = static_cast<real_t>(im + i) * static_cast<real_t>(0.5) * dx;
    const real_t hcl = (hl + hi) * static_cast<real_t>(0.5);
    const real_t xcr = static_cast<real_t>(i + ip) * static_cast<real_t>(0.5) * dx;
    const real_t hcr = (hi + hr) * static_cast<real_t>(0.5);
    const real_t dvl = noise.sample(xcl, hcl).dv_dy;
    const real_t dvr = noise.sample(xcr, hcr).dv_dy;

    const real_t grad =
        dx * (line_force + static_cast<real_t>(0.5) * (dvl + dvr) - f);
    return -grad;  // velocity
}

// Nearest forward position h' >= hi with v(h') = 0 (a stable forward
// equilibrium). Returns hi unchanged if the site is already blocked (v <= 0).
// Sets found=false (and returns the marched-to position) if no zero is reached
// within max_advance -- i.e. the site is running.
template <class Noise>
KOKKOS_INLINE_FUNCTION real_t rk_forward_zero(int i, int n, real_t dx, real_t lt,
                                              real_t f, Model model, real_t hi,
                                              real_t hl, real_t hr,
                                              const Noise &noise,
                                              const RkParams &rp, bool &found) {
    const real_t v0 = rk_site_velocity(i, n, dx, lt, f, model, hi, hl, hr, noise);
    if (v0 <= 0) {
        found = true;
        return hi;  // already blocked; no forward move
    }
    real_t a = hi;
    real_t advanced = 0;
    while (advanced < rp.max_advance) {
        const real_t b = a + rp.dstep;
        const real_t vb =
            rk_site_velocity(i, n, dx, lt, f, model, b, hl, hr, noise);
        if (vb <= 0) {  // sign change in [a, b] -> stable zero, bisect
            real_t lo = a, hicur = b;
            for (int k = 0; k < 60 && (hicur - lo) > rp.root_tol; ++k) {
                const real_t mid = static_cast<real_t>(0.5) * (lo + hicur);
                const real_t vm =
                    rk_site_velocity(i, n, dx, lt, f, model, mid, hl, hr, noise);
                if (vm > 0)
                    lo = mid;
                else
                    hicur = mid;
            }
            found = true;
            return static_cast<real_t>(0.5) * (lo + hicur);
        }
        a = b;
        advanced += rp.dstep;
    }
    found = false;
    return a;  // running: advanced by ~max_advance without finding a zero
}

// Construct the blocked configuration for the force in `p.driving_force`,
// advancing `h` forward in place. Returns whether it blocked or ran away.
template <class Noise>
RkResult rk_block(const View1D &h, const Params &p, const Noise &noise,
                  const RkParams &rp) {
    const int n = static_cast<int>(h.extent(0));
    const real_t dx = p.physical_size / static_cast<real_t>(n);
    const real_t lt = p.line_tension;
    const real_t f = p.driving_force;
    const Model model = p.model;

    View1D adv("rk_adv", n);

    real_t mean_h0 = 0;
    Kokkos::parallel_reduce(
        "rk_mean0", n,
        KOKKOS_LAMBDA(const int i, real_t &acc) { acc += h(i); }, mean_h0);
    mean_h0 /= static_cast<real_t>(n);

    RkResult res;
    for (int sweep = 0; sweep < rp.max_sweeps; ++sweep) {
        for (int color = 0; color < 2; ++color) {
            Kokkos::parallel_for(
                "rk_sweep", n, KOKKOS_LAMBDA(const int i) {
                    if ((i & 1) != color) return;
                    const int im = (i - 1 + n) % n;
                    const int ip = (i + 1) % n;
                    bool found = false;
                    const real_t hnew =
                        rk_forward_zero(i, n, dx, lt, f, model, h(i), h(im),
                                        h(ip), noise, rp, found);
                    adv(i) = hnew - h(i);  // >= 0 (forward only)
                    h(i) = hnew;
                });
        }

        real_t total_adv = 0;
        Kokkos::parallel_reduce(
            "rk_adv_sum", n,
            KOKKOS_LAMBDA(const int i, real_t &acc) { acc += adv(i); }, total_adv);
        real_t mean_h = 0;
        Kokkos::parallel_reduce(
            "rk_mean", n,
            KOKKOS_LAMBDA(const int i, real_t &acc) { acc += h(i); }, mean_h);
        mean_h /= static_cast<real_t>(n);

        res.sweeps = sweep + 1;
        res.total_advance = total_adv;
        res.mean_h = mean_h;

        if (mean_h - mean_h0 > rp.runaway) {  // depinned: advanced too far
            res.runaway = true;
            res.blocked = false;
            return res;
        }
        if (total_adv < rp.sweep_tol) {  // converged: line is blocked
            res.blocked = true;
            res.runaway = false;
            return res;
        }
    }
    // ran out of sweeps without a clear verdict
    res.blocked = false;
    res.runaway = false;
    return res;
}

}  // namespace qew

#endif  // QEW_ROSSO_KRAUTH_H
