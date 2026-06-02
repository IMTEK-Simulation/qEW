#include "lbfgs.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "fire.h"
#include "qew_model.h"
#include "qew_types.h"

#ifdef QEW_WITH_HDF5
#include <highfive/H5File.hpp>

#include "filtered_noise.h"
#endif

using namespace qew;

namespace {

struct ZeroNoise {
    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t, real_t) const { return NoiseSample{0, 0, 0}; }
};

// Line tension + harmonic trough at y0: unique minimiser is the flat line y0.
struct HarmonicWell {
    real_t k = 2.0;
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

Params linear_params(int n) {
    Params p;
    p.physical_size = n;  // dx = 1
    p.line_tension = 1.0;
    p.driving_force = 0.0;
    p.model = Model::Linear;
    return p;
}

}  // namespace

// Pure line tension relaxes to a flat profile.
TEST(Lbfgs, RelaxesToFlatLine) {
    const int n = 128;
    Params p = linear_params(n);
    std::vector<real_t> init(n);
    for (int i = 0; i < n; ++i) init[i] = 0.5 + 0.3 * std::cos(4 * M_PI * i / n);
    View1D h = make_line(init);

    LbfgsParams lp;
    lp.ftol = 1e-9;
    const LbfgsResult r = lbfgs_minimize(h, p, ZeroNoise{}, lp);

    EXPECT_TRUE(r.converged);
    const auto hf = to_host(h);
    const real_t span = *std::max_element(hf.begin(), hf.end()) -
                        *std::min_element(hf.begin(), hf.end());
    EXPECT_LT(span, 1e-6);
}

// Converges to the known harmonic minimum h = y0.
TEST(Lbfgs, ConvergesToHarmonicMinimum) {
    const int n = 96;
    Params p = linear_params(n);
    HarmonicWell well{2.0, 0.5};
    View1D h = make_line(std::vector<real_t>(n, 0.0));

    LbfgsParams lp;
    lp.ftol = 1e-9;
    const LbfgsResult r = lbfgs_minimize(h, p, well, lp);

    EXPECT_TRUE(r.converged);
    const auto hf = to_host(h);
    real_t max_dev = 0;
    for (real_t v : hf) max_dev = std::max(max_dev, std::abs(v - well.y0));
    EXPECT_LT(max_dev, 1e-6);
}

// On a quadratic problem L-BFGS converges in few iterations (vs FIRE's many).
TEST(Lbfgs, FastOnQuadratic) {
    const int n = 128;
    Params p = linear_params(n);
    HarmonicWell well{1.0, 0.3};
    View1D h = make_line(std::vector<real_t>(n, 0.0));

    LbfgsParams lp;
    lp.ftol = 1e-8;
    const LbfgsResult r = lbfgs_minimize(h, p, well, lp);

    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.iterations, 200);  // L-BFGS is fast on quadratics
}

// L-BFGS and FIRE reach the same minimum on a convex problem.
TEST(Lbfgs, AgreesWithFire) {
    const int n = 96;
    Params p = linear_params(n);
    HarmonicWell well{1.5, 0.4};

    std::vector<real_t> init(n);
    std::mt19937 rng(3);
    std::uniform_real_distribution<real_t> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) init[i] = dist(rng);

    View1D h_lbfgs = make_line(init);
    LbfgsParams lp;
    lp.ftol = 1e-9;
    lbfgs_minimize(h_lbfgs, p, well, lp);

    View1D h_fire = make_line(init);
    FireParams fp;
    fp.dt_max = 0.1 * std::sqrt(1.0 / p.line_tension);
    fp.ftol = 1e-9;
    fire_minimize(h_fire, p, well, fp);

    const auto a = to_host(h_lbfgs), b = to_host(h_fire);
    real_t max_diff = 0;
    for (int i = 0; i < n; ++i) max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    EXPECT_LT(max_diff, 1e-5);
}

#ifdef QEW_WITH_HDF5
// Golden cross-check against SciPy's L-BFGS-B on the real FilteredNoise energy:
// starting from the same configuration, the C++ minimum energy matches SciPy's.
TEST(Lbfgs, MatchesScipyGolden) {
    HighFive::File file(QEW_LBFGS_GOLDEN_PATH, HighFive::File::ReadOnly);
    int nx = 0, ny = 0;
    double Lx = 0, Ly = 0, line_tension = 0, e_min = 0;
    file.getAttribute("nx").read(nx);
    file.getAttribute("ny").read(ny);
    file.getAttribute("Lx").read(Lx);
    file.getAttribute("Ly").read(Ly);
    file.getAttribute("line_tension").read(line_tension);
    file.getAttribute("e_min").read(e_min);

    std::vector<std::vector<double>> field2d;
    file.getDataSet("field").read(field2d);
    std::vector<real_t> field(static_cast<std::size_t>(nx) * ny);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            field[static_cast<std::size_t>(i) * ny + j] = field2d[i][j];

    std::vector<real_t> h0;
    file.getDataSet("h0").read(h0);

    FilteredNoise noise(field.data(), nx, ny, Lx, Ly);
    Params p;
    p.physical_size = Lx;
    p.line_tension = line_tension;
    p.driving_force = 0.0;
    p.model = Model::Linear;

    View1D h = make_line(h0);
    LbfgsParams lp;
    lp.ftol = 1e-8;
    const LbfgsResult r = lbfgs_minimize(h, p, noise.device_noise(), lp);
    EXPECT_TRUE(r.converged);

    const real_t e_cpp = objective(h, p, noise.device_noise());
    // Same energy functional, same start, near-convex (weak disorder, stiff
    // line): both minimisers reach the same basin.
    EXPECT_NEAR(e_cpp, e_min, 1e-5 * std::abs(e_min) + 1e-6);
}
#endif
