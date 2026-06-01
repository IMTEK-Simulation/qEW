#ifndef QEW_FIRE_H
#define QEW_FIRE_H

#include <Kokkos_Core.hpp>

#include "qew_model.h"
#include "qew_types.h"

// FIRE (Fast Inertial Relaxation Engine, Bitzek et al. 2006) energy
// minimisation, replacing SciPy's L-BFGS-B for the static qEW runs. Everything
// is force evaluations + reductions + element-wise updates, so the whole
// relaxation runs on the default execution space (GPU on a CUDA/HIP build).
//
// Semi-implicit Euler variant: integrate v += dt F; mix v toward F; advance
// h += dt v; adapt dt from the power P = F·v. Converges to a local minimum
// (max_i |dE/dh_i| < ftol). NB: the qEW landscape is multi-minimum, so the
// reached minimum need not coincide with L-BFGS-B's -- compare statistics, not
// pointwise h.
namespace qew {

struct FireParams {
    real_t dt_max = 0;       // stability-limited max timestep (REQUIRED, > 0)
    real_t dt_init = 0;      // initial timestep; defaults to 0.1*dt_max if <= 0
    real_t ftol = 1e-6;      // converged when max_i |dE/dh_i| < ftol
    int max_iter = 100000;
    int n_min = 5;           // steps with P>0 before dt may grow
    real_t f_inc = 1.1;
    real_t f_dec = 0.5;
    real_t alpha_start = 0.1;
    real_t f_alpha = 0.99;
};

struct FireResult {
    int iterations = 0;
    real_t max_force = 0;
    bool converged = false;
};

// Relax `h` in place. Returns iteration count, final max|force|, and whether the
// convergence criterion was met within max_iter.
template <class Noise>
FireResult fire_minimize(const View1D &h, const Params &p, const Noise &noise,
                         const FireParams &fp) {
    const int n = static_cast<int>(h.extent(0));
    View1D grad("fire_grad", n);
    View1D v("fire_v", n);
    Kokkos::deep_copy(v, static_cast<real_t>(0));

    real_t dt = (fp.dt_init > 0) ? fp.dt_init : static_cast<real_t>(0.1) * fp.dt_max;
    real_t alpha = fp.alpha_start;
    int n_pos = 0;

    auto max_abs_grad = [&]() {
        real_t m = 0;
        Kokkos::parallel_reduce(
            "fire_maxforce", n,
            KOKKOS_LAMBDA(const int i, real_t &acc) {
                const real_t a = Kokkos::fabs(grad(i));
                if (a > acc) acc = a;
            },
            Kokkos::Max<real_t>(m));
        return m;
    };

    gradient(h, p, noise, grad);

    FireResult res;
    for (int it = 0; it < fp.max_iter; ++it) {
        const real_t max_f = max_abs_grad();
        if (max_f < fp.ftol) {
            res.iterations = it;
            res.max_force = max_f;
            res.converged = true;
            return res;
        }

        // v += dt * F, with F = -grad.
        Kokkos::parallel_for(
            "fire_vstep", n,
            KOKKOS_LAMBDA(const int i) { v(i) -= dt * grad(i); });

        // Power P = F·v = -grad·v.
        real_t P = 0;
        Kokkos::parallel_reduce(
            "fire_power", n,
            KOKKOS_LAMBDA(const int i, real_t &acc) { acc += -grad(i) * v(i); }, P);

        if (P > 0) {
            if (n_pos > fp.n_min) {
                dt = Kokkos::fmin(dt * fp.f_inc, fp.dt_max);
                alpha *= fp.f_alpha;
            }
            ++n_pos;
        } else {
            Kokkos::deep_copy(v, static_cast<real_t>(0));
            dt *= fp.f_dec;
            alpha = fp.alpha_start;
            n_pos = 0;
        }

        // Mix velocity toward the force direction: v = (1-a) v + a |v|/|F| F.
        real_t nv2 = 0, nf2 = 0;
        Kokkos::parallel_reduce(
            "fire_nv", n,
            KOKKOS_LAMBDA(const int i, real_t &acc) { acc += v(i) * v(i); }, nv2);
        Kokkos::parallel_reduce(
            "fire_nf", n,
            KOKKOS_LAMBDA(const int i, real_t &acc) { acc += grad(i) * grad(i); }, nf2);
        const real_t nf = Kokkos::sqrt(nf2);
        if (nf > 0) {
            const real_t a = alpha;
            const real_t scale = alpha * Kokkos::sqrt(nv2) / nf;  // a |v|/|F|
            Kokkos::parallel_for(
                "fire_mix", n, KOKKOS_LAMBDA(const int i) {
                    v(i) = (static_cast<real_t>(1) - a) * v(i) - scale * grad(i);
                });
        }

        // Advance positions and refresh the force.
        Kokkos::parallel_for(
            "fire_xstep", n,
            KOKKOS_LAMBDA(const int i) { h(i) += dt * v(i); });
        gradient(h, p, noise, grad);
    }

    res.iterations = fp.max_iter;
    res.max_force = max_abs_grad();
    res.converged = false;
    return res;
}

}  // namespace qew

#endif  // QEW_FIRE_H
