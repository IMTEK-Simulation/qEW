#ifndef QEW_LBFGS_H
#define QEW_LBFGS_H

#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

#include "qew_model.h"
#include "qew_types.h"

// Limited-memory BFGS energy minimisation, the production replacement for FIRE
// in the static qEW runs (FIRE is kept as a fallback). GPU-friendly: the Nocedal
// two-loop recursion is a handful of dot-products (parallel_reduce) and AXPYs
// (parallel_for) over length-N vectors with the small m-vector history; the
// scalars live on the host.
//
// Line search: backtracking ARMIJO (sufficient decrease) -- function
// evaluations only (cheaper than Wolfe, whose curvature test needs a gradient
// per trial), monotone and non-overshooting. Armijo alone does NOT guarantee
// the curvature condition s.y > 0, so we add the standard safeguard: SKIP the
// (s,y) update whenever s.y <= eps |s||y|, which keeps the implicit inverse
// Hessian positive definite. Initial-Hessian scaling H0 = (s.y)/(y.y) I.
//
// Unbounded: the periodic-wrap noise lets h roam freely (cf. qew_depinning.py),
// so no box constraint / L-BFGS-B machinery is needed.
namespace qew {

struct LbfgsParams {
    int m = 8;                     // history length
    real_t ftol = 1e-6;            // converged when max_i |dE/dh_i| < ftol
    int max_iter = 10000;
    real_t c1 = 1e-4;              // Armijo sufficient-decrease constant
    real_t ls_shrink = 0.5;        // backtracking factor
    int max_ls = 30;               // max backtracking steps
    real_t curvature_eps = 1e-12;  // skip pair if s.y <= eps |s||y|

    // FFT preconditioner for stiff lines: use the inverse elastic (Laplacian)
    // Hessian as the initial Hessian H0 (diagonalised by the FFT). Crushes the
    // condition ~ N^2 that otherwise stalls L-BFGS on stiff lines.
    bool precondition = false;
    // Shift added to the elastic eigenvalues: physically the pinning
    // (non-elastic) curvature scale (~ dx*A/xi^2). It regularises the k=0 zero
    // mode AND makes the preconditioner degrade to a well-scaled scalar step on
    // floppy lines (where elastic curvature is tiny). <=0 -> auto (smallest
    // non-zero elastic eigenvalue), which is only adequate when elastic
    // curvature dominates everywhere (stiff lines); set it explicitly otherwise.
    real_t precond_shift = 0;
};

struct LbfgsResult {
    int iterations = 0;
    real_t max_force = 0;
    bool converged = false;
    long n_func_evals = 0;
    long n_curvature_skips = 0;  // (s,y) pairs rejected by the curvature safeguard
    long n_ls_restarts = 0;      // line-search failures recovered by history reset
};

namespace detail {

inline real_t lbfgs_dot(int n, const View1D &a, const View1D &b) {
    real_t s = 0;
    Kokkos::parallel_reduce(
        "lbfgs_dot", n,
        KOKKOS_LAMBDA(const int i, real_t &acc) { acc += a(i) * b(i); }, s);
    return s;
}

inline real_t lbfgs_max_abs(int n, const View1D &a) {
    real_t s = 0;
    Kokkos::parallel_reduce(
        "lbfgs_maxabs", n,
        KOKKOS_LAMBDA(const int i, real_t &acc) {
            const real_t v = Kokkos::fabs(a(i));
            if (v > acc) acc = v;
        },
        Kokkos::Max<real_t>(s));
    return s;
}

inline void lbfgs_axpy(int n, real_t c, const View1D &x, const View1D &y) {
    Kokkos::parallel_for(
        "lbfgs_axpy", n, KOKKOS_LAMBDA(const int i) { y(i) += c * x(i); });
}

}  // namespace detail

// Relax `h` in place. Returns iteration count, final max|force|, convergence
// flag, and the total number of objective evaluations.
template <class Noise>
LbfgsResult lbfgs_minimize(const View1D &h, const Params &p, const Noise &noise,
                           const LbfgsParams &lp) {
    using detail::lbfgs_axpy;
    using detail::lbfgs_dot;
    using detail::lbfgs_max_abs;

    const int n = static_cast<int>(h.extent(0));
    const int m = lp.m;

    View1D g("lbfgs_g", n), g_new("lbfgs_gnew", n), d("lbfgs_d", n);
    View1D q("lbfgs_q", n), s_tmp("lbfgs_s", n), y_tmp("lbfgs_y", n);
    GradientWorkspace gws;
    std::vector<View1D> S(m), Y(m);
    for (int k = 0; k < m; ++k) {
        S[k] = View1D("lbfgs_S", n);
        Y[k] = View1D("lbfgs_Y", n);
    }
    std::vector<real_t> rho(m, 0.0);
    std::vector<int> slots;  // chronological row indices, back() = newest
    real_t gamma = 1.0;      // H0 scaling

    // ---- optional FFT (inverse-Laplacian) preconditioner ----
    // H0 = M^{-1} with M the elastic Hessian, diagonalised by the FFT:
    //   M^{-1} q = irfft( rfft(q) / ((kappa/dx) 4 sin^2(pi k/N) + shift) ).
    // Plans are built once and reused (execute) each iteration.
    ExecSpace exec;
    const int nh = n / 2 + 1;
    using CView1D = Kokkos::View<Kokkos::complex<real_t> *, MemSpace>;
    CView1D qhat("lbfgs_qhat", lp.precondition ? nh : 0);
    View1D eig("lbfgs_eig", lp.precondition ? nh : 0);
    std::optional<KokkosFFT::Plan<ExecSpace, View1D, CView1D>> rfft_plan;
    std::optional<KokkosFFT::Plan<ExecSpace, CView1D, View1D>> irfft_plan;
    if (lp.precondition) {
        const real_t kappa = p.line_tension;
        const real_t dx = p.physical_size / static_cast<real_t>(n);
        const real_t pi = static_cast<real_t>(3.14159265358979323846);
        real_t shift = lp.precond_shift;
        if (shift <= 0) {  // regularise k=0 with the smallest non-zero elastic mode
            const real_t s1 = std::sin(pi / static_cast<real_t>(n));
            shift = (kappa / dx) * 4 * s1 * s1;
        }
        const real_t kap = kappa, dxx = dx, sh = shift;
        const int N = n;
        Kokkos::parallel_for(
            "lbfgs_eig", nh, KOKKOS_LAMBDA(const int k) {
                const real_t s = Kokkos::sin(pi * static_cast<real_t>(k) /
                                             static_cast<real_t>(N));
                eig(k) = (kap / dxx) * 4 * s * s + sh;
            });
        rfft_plan.emplace(exec, q, qhat, KokkosFFT::Direction::forward, -1);
        irfft_plan.emplace(exec, qhat, d, KokkosFFT::Direction::backward, -1);
    }

    gradient(h, p, noise, g, gws);
    LbfgsResult res;
    int ls_fail_streak = 0;
    real_t phi = 0;  // objective at the current iterate, carried across
                     // iterations (each accepted trial evaluates it anyway)

    for (int it = 0; it < lp.max_iter; ++it) {
        const real_t gmax = lbfgs_max_abs(n, g);
        if (gmax < lp.ftol) {
            res.iterations = it;
            res.max_force = gmax;
            res.converged = true;
            return res;
        }

        // ---- two-loop recursion: d = -H g ----
        Kokkos::deep_copy(q, g);
        const int hs = static_cast<int>(slots.size());
        std::vector<real_t> la(hs);
        for (int idx = hs - 1; idx >= 0; --idx) {
            const int row = slots[idx];
            const real_t a = rho[row] * lbfgs_dot(n, S[row], q);
            la[idx] = a;
            lbfgs_axpy(n, -a, Y[row], q);  // q -= a y
        }
        if (lp.precondition) {
            // d = M^{-1} q : rfft(q) -> qhat; qhat /= eig; irfft(qhat) -> d.
            KokkosFFT::execute(*rfft_plan, q, qhat);
            Kokkos::parallel_for(
                "lbfgs_precond", nh,
                KOKKOS_LAMBDA(const int k) { qhat(k) = qhat(k) / eig(k); });
            KokkosFFT::execute(*irfft_plan, qhat, d);
        } else {
            const real_t g0 = slots.empty() ? static_cast<real_t>(1) : gamma;
            Kokkos::parallel_for(
                "lbfgs_scale", n, KOKKOS_LAMBDA(const int i) { d(i) = g0 * q(i); });
        }
        for (int idx = 0; idx < hs; ++idx) {
            const int row = slots[idx];
            const real_t b = rho[row] * lbfgs_dot(n, Y[row], d);
            lbfgs_axpy(n, la[idx] - b, S[row], d);  // d += (a-b) s
        }
        Kokkos::parallel_for(
            "lbfgs_neg", n, KOKKOS_LAMBDA(const int i) { d(i) = -d(i); });

        // Ensure a descent direction; otherwise fall back to steepest descent.
        real_t dphi0 = lbfgs_dot(n, g, d);
        if (dphi0 >= 0) {
            slots.clear();
            Kokkos::parallel_for(
                "lbfgs_sd", n, KOKKOS_LAMBDA(const int i) { d(i) = -g(i); });
            dphi0 = -lbfgs_dot(n, g, g);
        }

        // ---- Armijo backtracking line search ----
        // Trials are evaluated at h + alpha*d without writing h; the step is
        // committed only on acceptance, so failure needs no saved copy/undo.
        if (it == 0) {  // later iterations reuse the accepted trial value
            phi = objective(h, p, noise);
            ++res.n_func_evals;
        }
        const real_t phi0 = phi;  // objective at the current iterate
        // alpha0 = 1 for L-BFGS / preconditioned steps (well-scaled direction);
        // for an unpreconditioned first steepest-descent step, scale by 1/|g|.
        real_t alpha = static_cast<real_t>(1);
        if (!lp.precondition && slots.empty()) {
            alpha = static_cast<real_t>(1) /
                    std::max(static_cast<real_t>(1), std::sqrt(lbfgs_dot(n, g, g)));
        }
        bool ok = false;
        for (int ls = 0; ls < lp.max_ls; ++ls) {
            const real_t phi_trial = objective(h, d, alpha, p, noise);
            ++res.n_func_evals;
            if (phi_trial <= phi0 + lp.c1 * alpha * dphi0) {
                phi = phi_trial;
                ok = true;
                break;
            }
            alpha *= lp.ls_shrink;
        }
        if (!ok) {
            if (slots.empty() || ++ls_fail_streak > 2) {
                res.iterations = it;
                res.max_force = gmax;
                res.converged = false;
                return res;
            }
            slots.clear();  // reset history, retry as steepest descent
            ++res.n_ls_restarts;
            continue;
        }
        ls_fail_streak = 0;

        // Commit the accepted step.
        const real_t al = alpha;
        Kokkos::parallel_for(
            "lbfgs_step", n, KOKKOS_LAMBDA(const int i) { h(i) += al * d(i); });

        // ---- curvature pair s = alpha d, y = g_new - g ----
        // One fused kernel writes the pair and reduces all three scalars.
        gradient(h, p, noise, g_new, gws);
        real_t sy = 0, ss = 0, yy = 0;
        Kokkos::parallel_reduce(
            "lbfgs_sy", n,
            KOKKOS_LAMBDA(const int i, real_t &asy, real_t &ass, real_t &ayy) {
                const real_t s = al * d(i);
                const real_t y = g_new(i) - g(i);
                s_tmp(i) = s;
                y_tmp(i) = y;
                asy += s * y;
                ass += s * s;
                ayy += y * y;
            },
            sy, ss, yy);
        if (sy > lp.curvature_eps * std::sqrt(ss * yy)) {  // skip safeguard
            int row;
            if (static_cast<int>(slots.size()) < m) {
                row = static_cast<int>(slots.size());
            } else {
                row = slots.front();
                slots.erase(slots.begin());
            }
            // Swap the pair into the history slot (s_tmp/y_tmp become the
            // slot's old buffers, reused as scratch next iteration).
            std::swap(S[row], s_tmp);
            std::swap(Y[row], y_tmp);
            rho[row] = static_cast<real_t>(1) / sy;
            gamma = sy / yy;
            slots.push_back(row);
        } else {
            ++res.n_curvature_skips;  // non-positive curvature: keep H0 pos. def.
        }
        std::swap(g, g_new);  // g <- g_new; old g becomes scratch
    }

    res.iterations = lp.max_iter;
    res.max_force = lbfgs_max_abs(n, g);
    res.converged = false;
    return res;
}

}  // namespace qew

#endif  // QEW_LBFGS_H
