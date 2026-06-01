#include "fire.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "qew_model.h"
#include "qew_types.h"

using namespace qew;

namespace {

// No disorder: the energy is pure line tension, minimised by any flat line.
struct ZeroNoise {
    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t, real_t) const { return NoiseSample{0, 0, 0}; }
};

// A smooth harmonic trough at y0 (no x dependence). Combined with line tension
// the unique minimiser is the flat line h_i = y0, a nontrivial known target.
struct HarmonicWell {
    real_t k = 1.0;
    real_t y0 = 0.5;
    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t, real_t y) const {
        const real_t d = y - y0;
        return NoiseSample{static_cast<real_t>(0.5) * k * d * d, 0, k * d};
    }
};

View1D make_line(const std::vector<real_t> &vals) {
    View1D h("h", vals.size());
    auto hh = Kokkos::create_mirror_view(h);
    for (std::size_t i = 0; i < vals.size(); ++i) hh(i) = vals[i];
    Kokkos::deep_copy(h, hh);
    return h;
}

std::vector<real_t> to_host(const View1D &h) {
    auto hh = Kokkos::create_mirror_view(h);
    Kokkos::deep_copy(hh, h);
    std::vector<real_t> out(h.extent(0));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = hh(i);
    return out;
}

}  // namespace

// Test 5a: pure line tension relaxes a perturbed line to a flat profile.
TEST(Fire, RelaxesToFlatLine) {
    const int n = 128;
    const real_t dx = 1.0;
    Params p;
    p.physical_size = n * dx;  // dx = 1
    p.line_tension = 1.0;
    p.driving_force = 0.0;
    p.model = Model::Linear;

    std::vector<real_t> init(n);
    for (int i = 0; i < n; ++i) init[i] = 0.5 + 0.3 * std::cos(4 * M_PI * i / n);
    View1D h = make_line(init);

    FireParams fp;
    fp.dt_max = 0.1 * std::sqrt(dx / p.line_tension);
    fp.ftol = 1e-9;
    const FireResult r = fire_minimize(h, p, ZeroNoise{}, fp);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.max_force, 1e-9);
    const auto hf = to_host(h);
    const real_t span = *std::max_element(hf.begin(), hf.end()) -
                        *std::min_element(hf.begin(), hf.end());
    EXPECT_LT(span, 1e-6);  // flat
}

// Test 5b: line tension + harmonic trough converges to the known minimum h=y0.
TEST(Fire, ConvergesToHarmonicMinimum) {
    const int n = 96;
    const real_t dx = 1.0;
    Params p;
    p.physical_size = n * dx;
    p.line_tension = 1.0;
    p.driving_force = 0.0;
    p.model = Model::Linear;

    HarmonicWell well{2.0, 0.5};

    std::vector<real_t> init(n, 0.0);  // start far from y0
    View1D h = make_line(init);

    FireParams fp;
    fp.dt_max = 0.1 * std::sqrt(dx / p.line_tension);
    fp.ftol = 1e-9;
    const FireResult r = fire_minimize(h, p, well, fp);

    EXPECT_TRUE(r.converged);
    const auto hf = to_host(h);
    real_t max_dev = 0;
    for (real_t val : hf) max_dev = std::max(max_dev, std::abs(val - well.y0));
    EXPECT_LT(max_dev, 1e-6);
}

// Test 5c: minimisation does not increase the energy.
TEST(Fire, LowersEnergy) {
    const int n = 96;
    const real_t dx = 1.0;
    Params p;
    p.physical_size = n * dx;
    p.line_tension = 1.0;
    p.driving_force = 0.0;
    p.model = Model::Arclength;

    HarmonicWell well{1.5, 0.6};
    std::vector<real_t> init(n);
    std::mt19937 rng(1);
    std::uniform_real_distribution<real_t> d(0.0, 1.0);
    for (int i = 0; i < n; ++i) init[i] = d(rng);
    View1D h = make_line(init);

    const real_t e0 = objective(h, p, well);
    FireParams fp;
    fp.dt_max = 0.1 * std::sqrt(dx / p.line_tension);
    fp.ftol = 1e-9;
    fire_minimize(h, p, well, fp);
    const real_t e1 = objective(h, p, well);
    EXPECT_LE(e1, e0 + 1e-12);
}

// Test 7 (parity/determinism): identical inputs give identical results. On a
// CUDA/HIP build this also exercises the GPU path; here it pins reproducibility.
TEST(Fire, Deterministic) {
    const int n = 64;
    const real_t dx = 1.0;
    Params p;
    p.physical_size = n * dx;
    p.line_tension = 1.0;
    p.driving_force = 0.0;
    p.model = Model::Linear;
    HarmonicWell well{1.0, 0.5};

    std::vector<real_t> init(n);
    for (int i = 0; i < n; ++i) init[i] = 0.3 * std::sin(2 * M_PI * i / n);

    FireParams fp;
    fp.dt_max = 0.1 * std::sqrt(dx / p.line_tension);
    fp.ftol = 1e-10;

    View1D h1 = make_line(init);
    View1D h2 = make_line(init);
    const FireResult r1 = fire_minimize(h1, p, well, fp);
    const FireResult r2 = fire_minimize(h2, p, well, fp);

    EXPECT_EQ(r1.iterations, r2.iterations);
    const auto a = to_host(h1), b = to_host(h2);
    for (int i = 0; i < n; ++i) EXPECT_EQ(a[i], b[i]);
}
