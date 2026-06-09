#ifndef QEW_MODEL_H
#define QEW_MODEL_H

#include <Kokkos_Core.hpp>

#include "qew_types.h"

// Energy and force of a 1-D quenched Edwards-Wilkinson line h(x) on a periodic
// grid. Direct port of qew_objective / qew_gradient in
// ../../Overleaf/qEW/qew_core.py. The two MUST stay analytically consistent:
// gradient() is the exact gradient of objective(); the gradient-consistency
// test (test_qew_model.cpp) enforces this, mirroring qew_core.test_gradient.
//
// The disorder is supplied as a device-callable functor `Noise` providing
//     KOKKOS_INLINE_FUNCTION NoiseSample sample(real_t x, real_t y) const;
// so the same kernels run on CPU or GPU. Phase 2 uses an analytic stub; Phase 3
// swaps in FilteredNoise (bicubic interpolation of an FFT-generated field).
namespace qew {

enum class Model { Linear, Arclength };

struct Params {
    real_t physical_size = 1.0;   // box length L_x; grid spacing dx = L_x / N
    real_t line_tension = 1.0;    // kappa
    real_t driving_force = 0.0;   // f
    Model model = Model::Linear;
};

namespace detail {

// Shared objective body: the line position enters only through the
// device-callable functor `at(i)`, so the plain evaluation and the
// line-search trial evaluation share one analytic definition.
template <class Noise, class At>
real_t objective_eval(int n, const Params &p, const Noise &noise, const At &at) {
    const real_t dx = p.physical_size / static_cast<real_t>(n);
    const real_t lt = p.line_tension;
    const real_t f = p.driving_force;
    const Model model = p.model;

    real_t total = 0.0;
    Kokkos::parallel_reduce(
        "qew::objective", n,
        KOKKOS_LAMBDA(const int i, real_t &acc) {
            const int ip = (i + 1) % n;  // periodic forward neighbour
            const real_t hi = at(i);
            const real_t hip = at(ip);
            const real_t dh_dx = (hip - hi) / dx;

            real_t line_energy;
            if (model == Model::Linear) {
                line_energy = lt * dh_dx * dh_dx * static_cast<real_t>(0.5);
            } else {
                line_energy = lt * (Kokkos::sqrt(static_cast<real_t>(1) +
                                                 dh_dx * dh_dx) -
                                    static_cast<real_t>(1));
            }

            // Edge midpoint, matching python xcenter/hcenter (periodic wrap).
            const real_t xc = static_cast<real_t>(i + ip) * static_cast<real_t>(0.5) * dx;
            const real_t hc = (hip + hi) * static_cast<real_t>(0.5);
            const NoiseSample s = noise.sample(xc, hc);

            acc += dx * (line_energy + s.v - f * hi);
        },
        total);
    return total;
}

struct LineAt {
    View1D h;
    KOKKOS_INLINE_FUNCTION real_t operator()(int i) const { return h(i); }
};

struct TrialAt {
    View1D x0, d;
    real_t alpha;
    KOKKOS_INLINE_FUNCTION real_t operator()(int i) const {
        return x0(i) + alpha * d(i);
    }
};

}  // namespace detail

// Total energy E[h] = sum_i dx * (line_energy_i + V_i - f h_i), where the
// elastic and noise terms are evaluated on the forward edge (i, i+1) exactly as
// in qew_objective. Runs as a parallel_reduce on the default execution space.
template <class Noise>
real_t objective(const View1D &h, const Params &p, const Noise &noise) {
    return detail::objective_eval(static_cast<int>(h.extent(0)), p, noise,
                                  detail::LineAt{h});
}

// E at the line-search trial point x0 + alpha*d, evaluated without
// materialising it -- saves a kernel launch and n stores per trial, and the
// caller commits (or discards) the step afterwards.
template <class Noise>
real_t objective(const View1D &x0, const View1D &d, real_t alpha,
                 const Params &p, const Noise &noise) {
    return detail::objective_eval(static_cast<int>(x0.extent(0)), p, noise,
                                  detail::TrialAt{x0, d, alpha});
}

// Force grad_i = dE/dh_i. Elastic term: harmonic Laplacian (linear) or
// slope-normalized difference (arclength); noise term: half the sum of dV/dy at
// the left edge (i-1, i) and right edge (i, i+1). Exact port of qew_gradient,
// run as a parallel_for. `grad` must be pre-allocated with the same extent as h.
template <class Noise>
void gradient(const View1D &h, const Params &p, const Noise &noise,
              const View1D &grad) {
    const int n = static_cast<int>(h.extent(0));
    const real_t dx = p.physical_size / static_cast<real_t>(n);
    const real_t lt = p.line_tension;
    const real_t f = p.driving_force;
    const Model model = p.model;

    Kokkos::parallel_for(
        "qew::gradient", n, KOKKOS_LAMBDA(const int i) {
            const int ip = (i + 1) % n;          // right neighbour
            const int im = (i - 1 + n) % n;      // left neighbour

            real_t line_force;
            if (model == Model::Linear) {
                const real_t d2h = h(im) + h(ip) - static_cast<real_t>(2) * h(i);
                line_force = -lt * d2h / (dx * dx);
            } else {
                const real_t dhr = (h(ip) - h(i)) / dx;  // right slope
                const real_t dhl = (h(i) - h(im)) / dx;  // left slope
                line_force =
                    lt *
                    (dhl / Kokkos::sqrt(static_cast<real_t>(1) + dhl * dhl) -
                     dhr / Kokkos::sqrt(static_cast<real_t>(1) + dhr * dhr)) /
                    dx;
            }

            // dV/dy at the left and right edge midpoints (periodic wrap matches
            // python xcenter_left/right and hcenter_left/right).
            const real_t xcl = static_cast<real_t>(im + i) * static_cast<real_t>(0.5) * dx;
            const real_t hcl = (h(im) + h(i)) * static_cast<real_t>(0.5);
            const real_t xcr = static_cast<real_t>(i + ip) * static_cast<real_t>(0.5) * dx;
            const real_t hcr = (h(ip) + h(i)) * static_cast<real_t>(0.5);
            const NoiseSample sl = noise.sample(xcl, hcl);
            const NoiseSample sr = noise.sample(xcr, hcr);

            grad(i) = dx * (line_force +
                            static_cast<real_t>(0.5) * (sl.dv_dy + sr.dv_dy) - f);
        });
}

}  // namespace qew

#endif  // QEW_MODEL_H
