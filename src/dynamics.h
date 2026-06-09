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
        const bool thermal = (b > 0);
        Kokkos::parallel_for(
            "langevin_step", static_cast<int>(h.extent(0)),
            KOKKOS_LAMBDA(const int i) {
                real_t xi = 0;
                if (thermal) {
                    auto gen = pool.get_state();
                    xi = static_cast<real_t>(gen.normal());
                    pool.free_state(gen);
                }
                h(i) += -a * grad(i) + b * xi;
            });
    }

private:
    View1D grad_;
    GradientWorkspace ws_;
    Pool pool_;
};

}  // namespace qew

#endif  // QEW_DYNAMICS_H
