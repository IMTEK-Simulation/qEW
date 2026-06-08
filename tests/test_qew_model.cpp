#include "qew_model.h"

#include <algorithm>
#include <cmath>
#include <random>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "qew_types.h"

// Port of qew_core.test_gradient: check that the analytic gradient() equals the
// finite-difference of objective() for both elastic models. A smooth analytic
// noise (with exact dV/dy) is used so this isolates the assembly/chain-rule
// bookkeeping; interpolation accuracy is tested separately in Phase 3.

using namespace qew;

namespace {

// Smooth, device-callable pinning potential with exact partial derivatives.
struct AnalyticNoise {
    real_t amp = 0.3;
    real_t kx = 0.7;
    real_t ky = 1.3;

    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t x, real_t y) const {
        const real_t v = amp * Kokkos::sin(kx * x) * Kokkos::cos(ky * y);
        const real_t dv_dx = amp * kx * Kokkos::cos(kx * x) * Kokkos::cos(ky * y);
        const real_t dv_dy = -amp * ky * Kokkos::sin(kx * x) * Kokkos::sin(ky * y);
        return NoiseSample{v, dv_dx, dv_dy};
    }
};

// Spatially constant pinning potential V == v0 (zero derivatives). Lets us write
// the total energy of a flat line in closed form, independent of the assembly.
struct ConstNoise {
    real_t v0 = 0;
    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t, real_t) const { return NoiseSample{v0, 0, 0}; }
};

void check_gradient(Model model) {
    const int n = 64;

    Params p;
    p.physical_size = static_cast<real_t>(n);  // dx = 1 keeps magnitudes O(1)
    p.line_tension = 1.0;
    p.driving_force = 0.37;
    p.model = model;
    const AnalyticNoise noise;

    View1D h("h", n);
    auto h_host = Kokkos::create_mirror_view(h);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<real_t> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) h_host(i) = dist(rng);
    Kokkos::deep_copy(h, h_host);

    // Analytic gradient.
    View1D grad("grad", n);
    gradient(h, p, noise, grad);
    auto grad_host = Kokkos::create_mirror_view(grad);
    Kokkos::deep_copy(grad_host, grad);

    // Central finite difference of the objective, one coordinate at a time.
    const real_t eps = 1e-6;
    real_t max_diff = 0.0;
    for (int j = 0; j < n; ++j) {
        const real_t saved = h_host(j);

        h_host(j) = saved + eps;
        Kokkos::deep_copy(h, h_host);
        const real_t e_plus = objective(h, p, noise);

        h_host(j) = saved - eps;
        Kokkos::deep_copy(h, h_host);
        const real_t e_minus = objective(h, p, noise);

        h_host(j) = saved;  // restore
        Kokkos::deep_copy(h, h_host);

        const real_t fd = (e_plus - e_minus) / (2 * eps);
        max_diff = std::max(max_diff, std::abs(fd - grad_host(j)));
    }

    EXPECT_LT(max_diff, 1e-6)
        << "model=" << (model == Model::Linear ? "linear" : "arclength");
}

}  // namespace

TEST(QewModel, GradientMatchesFiniteDifferenceLinear) {
    check_gradient(Model::Linear);
}

TEST(QewModel, GradientMatchesFiniteDifferenceArclength) {
    check_gradient(Model::Arclength);
}

// Absolute energy of a FLAT line in a constant potential is known in closed
// form: every edge slope is zero (no line-tension energy in either model), so
//   E = sum_i dx * (v0 - f h_i) = L * (v0 - f c).
// This pins the absolute scale of objective() -- the dx weighting, the inclusion
// of the noise value, and the SIGN of the driving-force term -- for BOTH models
// (the gradient-consistency test above is blind to all three). The linear case
// is otherwise only checked indirectly via the SciPy L-BFGS golden; the
// arclength energy value is checked nowhere else.
TEST(QewModel, ObjectiveFlatLineConstantPotential) {
    const int n = 50;
    const real_t L = 3.0;
    const real_t c = 0.7;      // flat line height
    const real_t v0 = 1.25;    // constant potential value
    const real_t f = 0.4;      // driving force

    View1D h("h", n);
    Kokkos::deep_copy(h, c);

    for (const Model model : {Model::Linear, Model::Arclength}) {
        Params p;
        p.physical_size = L;
        p.line_tension = 2.3;  // irrelevant for a flat line, but should be ignored
        p.driving_force = f;
        p.model = model;

        const real_t e = objective(h, p, ConstNoise{v0});
        const real_t expected = L * (v0 - f * c);
        EXPECT_NEAR(e, expected, 1e-12)
            << "model=" << (model == Model::Linear ? "linear" : "arclength");
    }
}

// A single-mode line h_i = a sin(2 pi i / n) in the LINEAR model with zero noise
// and zero force has a closed-form discrete elastic energy:
//   E = sum_i dx * (1/2) lt ((h_{i+1}-h_i)/dx)^2
//     = (lt a^2 n / (2 dx)) * (1 - cos(2 pi / n))       [periodic, exact].
// Pins the linear line-tension term's value (not just its gradient) on a
// non-trivial profile.
TEST(QewModel, ObjectiveSinusoidLinearElastic) {
    const int n = 32;
    const real_t L = static_cast<real_t>(n);  // dx = 1
    const real_t dx = L / n;
    const real_t lt = 1.7;
    const real_t a = 0.9;
    const real_t two_pi = 2 * 3.14159265358979323846;

    View1D h("h", n);
    auto hh = Kokkos::create_mirror_view(h);
    for (int i = 0; i < n; ++i) hh(i) = a * std::sin(two_pi * i / n);
    Kokkos::deep_copy(h, hh);

    Params p;
    p.physical_size = L;
    p.line_tension = lt;
    p.driving_force = 0.0;
    p.model = Model::Linear;

    const real_t e = objective(h, p, ConstNoise{0.0});
    const real_t expected =
        (lt * a * a * n / (2 * dx)) * (1 - std::cos(two_pi / n));
    EXPECT_NEAR(e, expected, 1e-10);
}
