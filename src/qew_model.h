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

// Noise access points: prefer the cheap single-quantity samplers when the
// functor provides them (DeviceNoise does); analytic test noises only supply
// the full sample() and fall back to it. Keeps one call site per kernel.
template <class Noise>
KOKKOS_INLINE_FUNCTION real_t noise_value(const Noise &noise, real_t x, real_t y) {
    if constexpr (requires { noise.sample_value(x, y); })
        return noise.sample_value(x, y);
    else
        return noise.sample(x, y).v;
}

template <class Noise>
KOKKOS_INLINE_FUNCTION real_t noise_dvdy(const Noise &noise, real_t x, real_t y) {
    if constexpr (requires { noise.sample_dy(x, y); })
        return noise.sample_dy(x, y);
    else
        return noise.sample(x, y).dv_dy;
}

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
            const int ip = (i + 1 == n) ? 0 : i + 1;  // periodic forward neighbour
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
            acc += dx * (line_energy + noise_value(noise, xc, hc) - f * hi);
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

// Scratch for the two-pass gradient: per-edge quantities computed once and
// shared by the two adjacent sites (the dominant bicubic gather would
// otherwise run twice per edge). Own one of these in loops that call
// gradient() repeatedly so the allocations are reused; the workspace-free
// overload below allocates on the fly.
struct GradientWorkspace {
    View1D edge_dvdy;   // dV/dy at the midpoint of edge (i, i+1)
    View1D edge_slope;  // arclength only: dh/sqrt(1+dh^2) on edge (i, i+1)

    void ensure(int n, Model model) {
        if (static_cast<int>(edge_dvdy.extent(0)) != n)
            edge_dvdy = View1D("qew_edge_dvdy", n);
        if (model == Model::Arclength &&
            static_cast<int>(edge_slope.extent(0)) != n)
            edge_slope = View1D("qew_edge_slope", n);
    }
};

// Force grad_i = dE/dh_i. Elastic term: harmonic Laplacian (linear) or
// slope-normalized difference (arclength); noise term: half the sum of dV/dy at
// the left edge (i-1, i) and right edge (i, i+1). Exact port of qew_gradient.
// `grad` must be pre-allocated with the same extent as h.
//
// Two passes: edge kernel (one noise gather + one slope per edge), then a
// cheap per-site combine. Site i's left edge is edge i-1, its right edge is
// edge i; the edge midpoints match python xcenter/hcenter (periodic wrap), so
// the result is identical to the former one-pass version that sampled every
// edge twice.
template <class Noise>
void gradient(const View1D &h, const Params &p, const Noise &noise,
              const View1D &grad, GradientWorkspace &ws) {
    const int n = static_cast<int>(h.extent(0));
    const real_t dx = p.physical_size / static_cast<real_t>(n);
    const real_t lt = p.line_tension;
    const real_t f = p.driving_force;
    const Model model = p.model;

    ws.ensure(n, model);
    const View1D edvdy = ws.edge_dvdy;
    const View1D eslope = ws.edge_slope;
    const bool arc = (model == Model::Arclength);

    Kokkos::parallel_for(
        "qew::gradient_edges", n, KOKKOS_LAMBDA(const int i) {
            const int ip = (i + 1 == n) ? 0 : i + 1;
            const real_t xc =
                static_cast<real_t>(i + ip) * static_cast<real_t>(0.5) * dx;
            const real_t hc = (h(ip) + h(i)) * static_cast<real_t>(0.5);
            edvdy(i) = detail::noise_dvdy(noise, xc, hc);
            if (arc) {
                const real_t dh = (h(ip) - h(i)) / dx;
                eslope(i) =
                    dh / Kokkos::sqrt(static_cast<real_t>(1) + dh * dh);
            }
        });

    Kokkos::parallel_for(
        "qew::gradient", n, KOKKOS_LAMBDA(const int i) {
            const int ip = (i + 1 == n) ? 0 : i + 1;  // right neighbour
            const int im = (i == 0) ? n - 1 : i - 1;  // left neighbour

            real_t line_force;
            if (model == Model::Linear) {
                const real_t d2h = h(im) + h(ip) - static_cast<real_t>(2) * h(i);
                line_force = -lt * d2h / (dx * dx);
            } else {
                line_force = lt * (eslope(im) - eslope(i)) / dx;
            }

            grad(i) = dx * (line_force +
                            static_cast<real_t>(0.5) *
                                (edvdy(im) + edvdy(i)) - f);
        });
}

template <class Noise>
void gradient(const View1D &h, const Params &p, const Noise &noise,
              const View1D &grad) {
    GradientWorkspace ws;
    gradient(h, p, noise, grad, ws);
}

}  // namespace qew

#endif  // QEW_MODEL_H
