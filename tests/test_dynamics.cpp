#include "dynamics.h"

#include <cmath>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "qew_model.h"
#include "qew_types.h"

using namespace qew;

namespace {

// Harmonic trough V = 1/2 k (y - y0)^2 (no x dependence). With a quadratic
// elastic term (linear model) the total energy is a positive-definite quadratic
// form, so equipartition gives <E> = (N/2) kT exactly.
struct HarmonicWell {
    real_t k = 4.0;
    real_t y0 = 0.0;
    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t, real_t y) const {
        const real_t d = y - y0;
        return NoiseSample{static_cast<real_t>(0.5) * k * d * d, 0, k * d};
    }
};

View1D random_line(int n, std::uint64_t seed, real_t lo, real_t hi) {
    View1D h("h", n);
    auto hh = Kokkos::create_mirror_view(h);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<real_t> d(lo, hi);
    for (int i = 0; i < n; ++i) hh(i) = d(rng);
    Kokkos::deep_copy(h, hh);
    return h;
}

}  // namespace

// Test 8a: with the thermal term off, a step is gradient descent -- for a small
// enough drift the energy decreases monotonically.
TEST(Dynamics, ZeroTemperatureIsGradientDescent) {
    const int n = 64;
    Params p;
    p.physical_size = n;  // dx = 1
    p.line_tension = 1.0;
    p.driving_force = 0.0;
    p.model = Model::Linear;
    HarmonicWell well{4.0, 0.0};

    View1D h = random_line(n, 1, -0.5, 0.5);
    LangevinDynamics dyn(n, /*seed=*/0);

    const real_t drift = 0.05;  // < 2/lambda_max for a contractive step
    real_t e_prev = objective(h, p, well);
    for (int s = 0; s < 2000; ++s) {
        dyn.step(h, p, well, drift, /*diff_coeff=*/0.0);
        const real_t e = objective(h, p, well);
        EXPECT_LE(e, e_prev + 1e-12) << "energy increased at step " << s;
        e_prev = e;
    }
    EXPECT_LT(e_prev, 1e-6);  // relaxed to the (zero) minimum
}

// Test 8b: at finite temperature the sampled mean energy obeys equipartition,
// <E> = (N/2) kT, with kT = diff^2 / (2 drift). Validates the integrator's
// fluctuation-dissipation balance.
TEST(Dynamics, EquipartitionMeanEnergy) {
    const int n = 16;
    Params p;
    p.physical_size = n;  // dx = 1
    p.line_tension = 1.0;
    p.driving_force = 0.0;
    p.model = Model::Linear;  // quadratic energy
    HarmonicWell well{4.0, 0.0};

    const real_t drift = 0.02;
    const real_t kT = 0.05;
    const real_t diff = std::sqrt(2 * drift * kT);  // => kT = diff^2/(2 drift)

    View1D h("h", n);
    Kokkos::deep_copy(h, static_cast<real_t>(0));
    LangevinDynamics dyn(n, /*seed=*/2024);

    const int n_equil = 40000;
    const int n_sample = 360000;
    for (int s = 0; s < n_equil; ++s) dyn.step(h, p, well, drift, diff);

    double sum_e = 0;
    for (int s = 0; s < n_sample; ++s) {
        dyn.step(h, p, well, drift, diff);
        sum_e += objective(h, p, well);
    }
    const double mean_e = sum_e / n_sample;
    const double expected = 0.5 * n * kT;  // (N/2) kT
    EXPECT_NEAR(mean_e, expected, 0.1 * expected) << "mean E = " << mean_e;
}
