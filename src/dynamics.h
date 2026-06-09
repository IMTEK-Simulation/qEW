#ifndef QEW_DYNAMICS_H
#define QEW_DYNAMICS_H

#include <cstdint>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "qew_model.h"
#include "qew_types.h"

// Overdamped Langevin (Brownian) dynamics by Euler-Maruyama, ported from
// ../../Overleaf/qEW/qew_dynamic.py. One step advances the line by a
// deterministic drift down the energy gradient plus a Gaussian thermal kick:
//
//     h_i += -drift_coeff * dE/dh_i  +  diff_coeff * N(0,1).
//
// The stationary distribution is exp(-E/kT) with kT = diff_coeff^2 / (2 drift_coeff).
// qew_dynamic.py parameterises these by an (annealed) thermal length L_T:
//     drift_coeff = L_T * dt / (tau * A),  diff_coeff = L_T * sqrt(2 dt / tau),
// which gives kT = A * L_T. Keeping the two coefficients explicit here keeps the
// integrator independent of that particular parameterisation (and lets a test
// set diff_coeff = 0 to recover pure gradient descent). The thermal noise uses
// Kokkos' parallel RNG so the whole step runs on the GPU for a CUDA/HIP build.
namespace qew {

class LangevinDynamics {
public:
    using Pool = Kokkos::Random_XorShift64_Pool<ExecSpace>;

    LangevinDynamics(int n, std::uint64_t seed)
        : grad_("langevin_grad", n), pool_(seed) {}

    template <class Noise>
    void step(const View1D &h, const Params &p, const Noise &noise,
              real_t drift_coeff, real_t diff_coeff) {
        gradient(h, p, noise, grad_, ws_);
        const View1D grad = grad_;
        Pool pool = pool_;
        const real_t a = drift_coeff;
        const real_t b = diff_coeff;
        const int n = static_cast<int>(h.extent(0));
        if (b > 0) {
            // One RNG state per lane, each covering a contiguous block: a
            // pool acquire per chunk instead of per site (the per-site
            // acquire is lock traffic that serialises on wide GPUs).
            const int n_lanes = (n < 8192) ? n : 8192;
            const int chunk = (n + n_lanes - 1) / n_lanes;
            Kokkos::parallel_for(
                "langevin_step", n_lanes, KOKKOS_LAMBDA(const int t) {
                    auto gen = pool.get_state();
                    const int lo = t * chunk;
                    const int hi = (lo + chunk < n) ? lo + chunk : n;
                    for (int i = lo; i < hi; ++i)
                        h(i) += -a * grad(i) +
                                b * static_cast<real_t>(gen.normal());
                    pool.free_state(gen);
                });
        } else {  // zero temperature: pure gradient descent, no RNG at all
            Kokkos::parallel_for(
                "langevin_step", n,
                KOKKOS_LAMBDA(const int i) { h(i) += -a * grad(i); });
        }
    }

private:
    View1D grad_;
    GradientWorkspace ws_;
    Pool pool_;
};

}  // namespace qew

#endif  // QEW_DYNAMICS_H
