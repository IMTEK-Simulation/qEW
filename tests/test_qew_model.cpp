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
