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

// Test 8c: the driving force enters the drift. At zero temperature a flat line
// in a harmonic trough at y0 plus a constant force f relaxes (elastic term
// vanishes for a flat line) to the shifted equilibrium where the site velocity
// -dE/dh = -(k(y-y0) - f) is zero, i.e. y = y0 + f/k. Exercises the f term in
// the integrated gradient (the existing tests all use f = 0).
TEST(Dynamics, DrivingForceShiftsEquilibrium) {
    const int n = 32;
    const real_t k = 4.0, y0 = 0.2, f = 0.6;
    Params p;
    p.physical_size = n;  // dx = 1
    p.line_tension = 1.0;
    p.driving_force = f;
    p.model = Model::Linear;
    HarmonicWell well{k, y0};

    View1D h("h", n);
    Kokkos::deep_copy(h, static_cast<real_t>(0));
    LangevinDynamics dyn(n, /*seed=*/0);

    const real_t drift = 0.05;
    for (int s = 0; s < 5000; ++s)
        dyn.step(h, p, well, drift, /*diff_coeff=*/0.0);

    auto hh = Kokkos::create_mirror_view(h);
    Kokkos::deep_copy(hh, h);
    const real_t y_eq = y0 + f / k;
    real_t max_dev = 0;
    for (int i = 0; i < n; ++i) max_dev = std::max(max_dev, std::abs(hh(i) - y_eq));
    EXPECT_LT(max_dev, 1e-6);
}

// Test 8d: with diff_coeff == 0 the RNG branch is skipped entirely, so the step
// is purely deterministic -- two integrators seeded DIFFERENTLY must produce
// identical trajectories. Pins that the thermal kick is gated on diff_coeff > 0
// (a regression here would silently inject noise into the supposedly
// zero-temperature gradient-descent path the static cross-checks rely on).
TEST(Dynamics, ZeroDiffusionIgnoresSeed) {
    const int n = 48;
    Params p;
    p.physical_size = n;
    p.line_tension = 1.0;
    p.driving_force = 0.3;
    p.model = Model::Linear;
    HarmonicWell well{4.0, 0.1};

    View1D h1 = random_line(n, /*seed=*/5, -0.5, 0.5);
    View1D h2 = random_line(n, /*seed=*/5, -0.5, 0.5);  // same start
    LangevinDynamics d1(n, /*seed=*/111);
    LangevinDynamics d2(n, /*seed=*/999);  // different RNG seed

    for (int s = 0; s < 50; ++s) {
        d1.step(h1, p, well, 0.05, /*diff_coeff=*/0.0);
        d2.step(h2, p, well, 0.05, /*diff_coeff=*/0.0);
    }

    auto a = Kokkos::create_mirror_view(h1);
    auto b = Kokkos::create_mirror_view(h2);
    Kokkos::deep_copy(a, h1);
    Kokkos::deep_copy(b, h2);
    for (int i = 0; i < n; ++i) EXPECT_EQ(a(i), b(i));
}
