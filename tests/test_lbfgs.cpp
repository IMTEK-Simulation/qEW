#include "lbfgs.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "filtered_noise.h"
#include "fire.h"
#include "qew_model.h"
#include "qew_types.h"

#ifdef QEW_WITH_HDF5
#include <highfive/H5File.hpp>
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

// L-BFGS on the ARCLENGTH model: a flat line in a harmonic trough at y0 is still
// the unique minimiser (the arclength elastic term vanishes for a flat line), so
// it converges to h = y0. Every other L-BFGS test uses the linear model; this
// exercises the arclength branch of objective()/gradient() through the solver.
TEST(Lbfgs, ConvergesToHarmonicMinimumArclength) {
    const int n = 96;
    Params p = linear_params(n);
    p.model = Model::Arclength;
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

// Capped iterations report converged=false (the failure path callers branch on).
// All other L-BFGS tests assert the success path.
TEST(Lbfgs, ReportsNonConvergence) {
    const int n = 96;
    Params p = linear_params(n);
    HarmonicWell well{2.0, 0.5};
    View1D h = make_line(std::vector<real_t>(n, 0.0));

    LbfgsParams lp;
    lp.ftol = 1e-12;  // unreachable in one iteration
    lp.max_iter = 1;
    const LbfgsResult r = lbfgs_minimize(h, p, well, lp);

    EXPECT_FALSE(r.converged);
    EXPECT_GE(r.max_force, lp.ftol);
}

// Curvature safeguard (the `sy <= eps |s||y|` skip branch). By Cauchy-Schwarz
// sy <= |s||y| = sqrt(ss*yy), so an enormous curvature_eps makes the acceptance
// test sy > eps*sqrt(ss*yy) fail for EVERY pair: no (s,y) is ever stored. The
// solver must then degrade gracefully to (scaled) steepest descent and still
// reach the minimum -- the safeguard keeps the implicit inverse Hessian positive
// definite rather than corrupting the solve. A convex problem never skips on its
// own, so this is the only way to exercise the branch.
TEST(Lbfgs, CurvatureSkipForcedByLargeEps) {
    const int n = 16;
    Params p = linear_params(n);  // small, well-conditioned -> SD converges
    HarmonicWell well{2.0, 0.5};
    View1D h = make_line(std::vector<real_t>(n, 0.0));

    LbfgsParams lp;
    lp.ftol = 1e-8;
    lp.max_iter = 20000;
    lp.curvature_eps = 1e30;  // reject every curvature pair
    const LbfgsResult r = lbfgs_minimize(h, p, well, lp);

    EXPECT_GT(r.n_curvature_skips, 0);
    EXPECT_TRUE(r.converged);
    const auto hf = to_host(h);
    real_t max_dev = 0;
    for (real_t v : hf) max_dev = std::max(max_dev, std::abs(v - well.y0));
    EXPECT_LT(max_dev, 1e-6);
}

// Line-search give-up branch: max_ls = 0 means the backtracking loop runs zero
// trials, so the very first (steepest-descent) step "fails". With no history to
// fall back on, the solver reports non-convergence immediately (iteration 0)
// rather than looping or dereferencing empty history.
TEST(Lbfgs, LineSearchGivesUpWithNoHistory) {
    const int n = 32;
    Params p = linear_params(n);
    HarmonicWell well{2.0, 0.5};
    View1D h = make_line(std::vector<real_t>(n, 0.0));

    LbfgsParams lp;
    lp.ftol = 1e-9;
    lp.max_ls = 0;
    const LbfgsResult r = lbfgs_minimize(h, p, well, lp);

    EXPECT_FALSE(r.converged);
    EXPECT_EQ(r.iterations, 0);
    EXPECT_EQ(r.n_ls_restarts, 0);  // never got far enough to build history
}

// Line-search RESTART branch: when a unit L-BFGS step fails the line search but
// history exists, the solver discards its history and retries as steepest
// descent instead of giving up or dereferencing empty history.
//
// Trigger: a rough, strong-noise landscape on which a unit L-BFGS step
// overshoots a noise minimum by ~16x -- it needs ~5 Armijo backtracks to
// recover -- combined with max_ls clamped to 3. Once history is built, that
// step exhausts its 3 trials and falls into the restart branch. The overshoot
// is decisive (mls=1,2,3 all fire it; the line search only stops failing at
// mls>=5), so the branch is reached with margin rather than on an FP knife-edge.
TEST(Lbfgs, LineSearchRestartsOnStepFailure) {
    const int nx = 64, ny = 64;
    const real_t Lx = 3.2, Ly = 1.0;
    FilteredNoise noise(nx, ny, Lx, Ly, /*amplitude=*/2.0, 0.15, 0.15, /*seed=*/3);

    Params p;
    p.physical_size = Lx;
    p.line_tension = 0.5;
    p.driving_force = 0.0;
    p.model = Model::Linear;

    View1D h = make_line(std::vector<real_t>(nx, 0.5 * Ly));
    LbfgsParams lp;
    lp.ftol = 1e-6;
    lp.max_iter = 50000;
    lp.max_ls = 3;  // smaller than the ~5 backtracks the overshoot needs
    const LbfgsResult r = lbfgs_minimize(h, p, noise.device_noise(), lp);
    EXPECT_GT(r.n_ls_restarts, 0);  // the restart branch executed

    // Control: with the normal backtracking budget the SAME problem never needs
    // a restart and converges -- confirming the restart is induced by the clamp
    // (a genuine safety net) and not a routine code path.
    View1D h2 = make_line(std::vector<real_t>(nx, 0.5 * Ly));
    LbfgsParams lp2 = lp;
    lp2.max_ls = 30;
    const LbfgsResult r2 = lbfgs_minimize(h2, p, noise.device_noise(), lp2);
    EXPECT_TRUE(r2.converged);
    EXPECT_EQ(r2.n_ls_restarts, 0);
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

// The FFT preconditioner makes a stiff line (ill-conditioned discrete Laplacian,
// condition ~ N^2) converge in few iterations, where the unpreconditioned solver
// stalls. Uses the real FilteredNoise energy (needs kokkos-fft, not HDF5).
TEST(Lbfgs, PreconditionerOnStiffLine) {
    const int nx = 256, ny = 64;
    const real_t Lx = 2.56, Ly = 1.0;  // dx = 0.01 -> stiff Laplacian
    FilteredNoise noise(nx, ny, Lx, Ly, /*amplitude=*/0.3, 0.2, 0.2, /*seed=*/5);

    Params p;
    p.physical_size = Lx;
    p.line_tension = 50.0;  // strongly elastic-dominated
    p.driving_force = 0.0;
    p.model = Model::Linear;

    // Preconditioned: should converge quickly.
    View1D h = make_line(std::vector<real_t>(nx, 0.5 * Ly));
    LbfgsParams opt;
    opt.ftol = 1e-6;
    opt.max_iter = 5000;
    opt.precondition = true;
    const LbfgsResult r = lbfgs_minimize(h, p, noise.device_noise(), opt);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.iterations, 300);

    // Unpreconditioned from the same start needs many more iterations; confirm
    // the preconditioner is a large win.
    View1D h2 = make_line(std::vector<real_t>(nx, 0.5 * Ly));
    LbfgsParams opt2 = opt;
    opt2.precondition = false;
    const LbfgsResult r2 = lbfgs_minimize(h2, p, noise.device_noise(), opt2);
    EXPECT_LT(r.iterations, r2.iterations);
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
    lp.precondition = true;  // stiff golden field: preconditioner reaches ftol
                             // robustly (same minimum, hence same energy).
    const LbfgsResult r = lbfgs_minimize(h, p, noise.device_noise(), lp);
    EXPECT_TRUE(r.converged);

    const real_t e_cpp = objective(h, p, noise.device_noise());
    // Same energy functional, same start, near-convex (weak disorder, stiff
    // line): both minimisers reach the same basin.
    EXPECT_NEAR(e_cpp, e_min, 1e-5 * std::abs(e_min) + 1e-6);
}
#endif
