#include "qew_runs.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "qew_format.h"
#include "qew_types.h"

#ifdef QEW_WITH_HDF5
#include <highfive/H5File.hpp>

#include "qew_io.h"
#endif

// Coverage for the simulation executables, whose orchestration was factored into
// qew::static_sweep / dynamic_anneal / depinning_ramp (src/qew_runs.h) and whose
// HDF5 layout lives in qew::write_* (src/qew_io.h). The mains are now thin glue
// over these; here we drive the logic on tiny problems with cheap analytic
// noise, and (with HDF5) round-trip the writers to pin the dataset-key contract
// with the Python analysis scripts.

using namespace qew;

namespace {

struct ZeroNoise {
    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t, real_t) const { return NoiseSample{0, 0, 0}; }
};

// Washboard V = A cos(k y), no x dependence: a flat line is a single particle
// with analytic depinning threshold f_c = A k.
struct SinusoidNoise {
    real_t A = 0.1;
    real_t k = 2.0 * 3.14159265358979323846;  // lambda = 1
    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t, real_t y) const {
        return NoiseSample{A * Kokkos::cos(k * y), 0, -A * k * Kokkos::sin(k * y)};
    }
};

}  // namespace

// ---- fmt(): the HDF5 key formatter --------------------------------------
TEST(Fmt, MatchesPrintfSpecs) {
    EXPECT_EQ(fmt("%.1g", 0.0), "0");
    EXPECT_EQ(fmt("%.3g", 0.1), "0.1");
    EXPECT_EQ(fmt("%.3g", 100.0), "100");
    EXPECT_EQ(fmt("%.6g", 1.0), "1");
    EXPECT_EQ(fmt("%.4g", 0.8), "0.8");
}

// ---- static_sweep -------------------------------------------------------
TEST(Executables, StaticSweepShapeAndMapping) {
    StaticConfig cfg;
    cfg.model = Model::Linear;
    cfg.amplitude = 1.3;
    cfg.xi = 0.1;
    cfg.Lx = 1.0;
    cfg.Ly = 1.0;
    cfg.nx = 16;
    cfg.ny = 8;
    cfg.pinning_lengths_over_xi = {0.1, 1.0, 4.0};
    cfg.ftol = 1e-8;
    cfg.max_iter = 2000;

    const auto profiles = static_sweep(cfg, ZeroNoise{});

    ASSERT_EQ(profiles.size(), cfg.pinning_lengths_over_xi.size());
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        const auto &prof = profiles[i];
        EXPECT_EQ(prof.pinning_length_over_xi, cfg.pinning_lengths_over_xi[i]);
        // Larkin inversion Gamma = amplitude * (lambda_p/xi)^{3/2}
        // (arXiv:2410.21838 eq. (10)); shared by all three drivers.
        EXPECT_NEAR(prof.line_tension,
                    std::pow(cfg.pinning_lengths_over_xi[i], 1.5) * cfg.amplitude,
                    1e-12);
        ASSERT_EQ(prof.h.size(), static_cast<std::size_t>(cfg.nx));
        // Zero noise, zero force: the flat start is already the minimum.
        EXPECT_TRUE(prof.result.converged);
        const double span = *std::max_element(prof.h.begin(), prof.h.end()) -
                            *std::min_element(prof.h.begin(), prof.h.end());
        EXPECT_LT(span, 1e-9);
        for (double v : prof.h) EXPECT_NEAR(v, 0.5 * cfg.Ly, 1e-9);
    }
}

// ---- dynamic_anneal -----------------------------------------------------
TEST(Executables, DynamicAnnealTimestepAndSnapshots) {
    DynamicConfig cfg;
    cfg.model = Model::Linear;
    cfg.amplitude = 1.0;
    cfg.xi = 0.1;
    cfg.pinning_length_over_xi = 0.1;
    cfg.Lx = 1.0;
    cfg.Ly = 1.0;
    cfg.nx = 32;
    cfg.ny = 8;
    cfg.tau = 1.0;
    cfg.tl_start = 0.01;
    cfg.tl_end = 0.0001;
    cfg.total_time = 200.0;

    const DynamicResult res = dynamic_anneal(cfg, ZeroNoise{});

    // dt = min(0.1, tau/10 * min(fac, fac^2)), fac = xi / max(tl_start, tl_end).
    const real_t fac = cfg.xi / std::max(cfg.tl_start, cfg.tl_end);
    const real_t dt_expected =
        std::min(static_cast<real_t>(0.1), cfg.tau / 10 * std::min(fac, fac * fac));
    EXPECT_NEAR(res.dt, dt_expected, 1e-12);
    // Same Larkin map as the static driver: Gamma = A * (lambda_p/xi)^{3/2}.
    EXPECT_NEAR(res.line_tension,
                std::pow(cfg.pinning_length_over_xi, 1.5) * cfg.amplitude, 1e-12);

    // Snapshots are logarithmically spaced: first near 10*dt, each ~10x the last.
    ASSERT_GE(res.snapshots.size(), 2u);
    EXPECT_NEAR(res.snapshots.front().t, 10 * res.dt, res.dt);
    for (std::size_t i = 1; i < res.snapshots.size(); ++i) {
        EXPECT_GT(res.snapshots[i].t, res.snapshots[i - 1].t);
        EXPECT_NEAR(res.snapshots[i].t / res.snapshots[i - 1].t, 10.0, 0.5);
    }
    for (const auto &snap : res.snapshots) {
        ASSERT_EQ(snap.h.size(), static_cast<std::size_t>(cfg.nx));
        for (double v : snap.h) EXPECT_TRUE(std::isfinite(v));
    }
    EXPECT_LE(res.snapshots.back().t, res.total_time);
}

// ---- depinning_ramp -----------------------------------------------------
TEST(Executables, DepinningRampBracketsCriticalForce) {
    const SinusoidNoise noise;
    const real_t fc_exact = noise.A * noise.k;     // = 0.2*pi ~ 0.628
    const real_t lambda = 2 * 3.14159265358979323846 / noise.k;

    DepinningConfig cfg;
    cfg.model = Model::Linear;
    cfg.amplitude = 1.0;
    cfg.pinning_length_over_xi = 1.0;
    cfg.Lx = 8.0;
    cfg.Ly = 1.0;
    cfg.nx = 8;
    cfg.ny = 8;
    cfg.f_step = 0.2;
    cfg.f_max = 2.0;
    cfg.rp.dstep = lambda / 100;
    cfg.rp.max_advance = lambda;
    cfg.rp.runaway = 3 * lambda;
    cfg.rp.root_tol = 1e-11;
    cfg.rp.sweep_tol = 1e-8;
    cfg.rp.max_sweeps = 20000;

    const DepinningResult res = depinning_ramp(cfg, noise);

    // Same Larkin map as the other drivers: Gamma = A * (lambda_p/xi)^{3/2}.
    EXPECT_NEAR(res.line_tension,
                std::pow(cfg.pinning_length_over_xi, 1.5) * cfg.amplitude, 1e-12);

    ASSERT_FALSE(res.steps.empty());
    EXPECT_TRUE(res.steps.front().result.blocked);  // f = 0 blocks
    EXPECT_EQ(res.steps.front().f, 0.0);

    // The ramp stops at the first unblocked force, which brackets f_c from above.
    const DepinningStep &last = res.steps.back();
    EXPECT_FALSE(last.result.blocked);
    EXPECT_TRUE(last.result.runaway);
    EXPECT_TRUE(last.h.empty());  // no profile stored for the runaway force
    EXPECT_NEAR(res.f_c_upper_bound, last.f, 1e-9);

    EXPECT_GT(res.f_c_upper_bound, 0.0);
    EXPECT_LE(res.f_c_upper_bound, cfg.f_max);
    // f_c lies in (f_c_upper_bound - f_step, f_c_upper_bound].
    EXPECT_LE(res.f_c_upper_bound - cfg.f_step, fc_exact);
    EXPECT_LE(fc_exact, res.f_c_upper_bound + 1e-9);

    // Every stored (blocked) step carries a full profile; the runaway one alone
    // is profile-free.
    for (const auto &st : res.steps) {
        if (st.result.blocked)
            EXPECT_EQ(st.h.size(), static_cast<std::size_t>(cfg.nx));
    }
}

#ifdef QEW_WITH_HDF5
namespace {

bool has_dataset(HighFive::File &f, const std::string &path) {
    return f.exist(path) && f.getObjectType(path) == HighFive::ObjectType::Dataset;
}

}  // namespace

// The writers' dataset KEYS are a contract with the Python analysis scripts.
// Round-trip each writer and assert the exact key strings + values.
TEST(Executables, WriteStaticKeysAndValues) {
    StaticConfig cfg;
    cfg.Lx = 4.0;
    cfg.driving_force = 0.0;

    std::vector<StaticProfile> profiles(1);
    profiles[0].pinning_length_over_xi = 0.1;
    profiles[0].line_tension = 0.5;
    profiles[0].h = {1.0, 2.0, 3.0, 4.0};

    const std::string fname = "test_write_static.h5";
    {
        HighFive::File file(fname, HighFive::File::Truncate);
        write_static(file, cfg, profiles);
    }
    HighFive::File file(fname, HighFive::File::ReadOnly);

    double Lx = 0;
    file.getDataSet("physical_size").read(Lx);
    EXPECT_DOUBLE_EQ(Lx, cfg.Lx);

    const std::string prefix = "driving_force=0/pinning_length=0.1";
    ASSERT_TRUE(has_dataset(file, prefix + "/h"));
    ASSERT_TRUE(has_dataset(file, prefix + "/pinning_length"));
    std::vector<double> h;
    file.getDataSet(prefix + "/h").read(h);
    EXPECT_EQ(h, profiles[0].h);
    double lp = 0;
    file.getDataSet(prefix + "/pinning_length").read(lp);
    EXPECT_DOUBLE_EQ(lp, 0.1);
}

TEST(Executables, WriteDynamicKeysAndValues) {
    DynamicConfig cfg;
    cfg.Lx = 2.0;
    cfg.driving_force = 0.0;
    cfg.pinning_length_over_xi = 0.1;

    DynamicResult res;
    res.line_tension = 2.0;
    res.snapshots.push_back(DynamicSnapshot{static_cast<real_t>(1.0), {5.0, 6.0}});

    const std::string fname = "test_write_dynamic.h5";
    {
        HighFive::File file(fname, HighFive::File::Truncate);
        write_dynamic(file, cfg, res);
    }
    HighFive::File file(fname, HighFive::File::ReadOnly);

    const std::string prefix = "timestep=1/driving_force=0/line_tension=2";
    ASSERT_TRUE(has_dataset(file, prefix + "/h"));
    ASSERT_TRUE(has_dataset(file, prefix + "/pinning_length"));
    std::vector<double> h;
    file.getDataSet(prefix + "/h").read(h);
    EXPECT_EQ(h, (std::vector<double>{5.0, 6.0}));
}

TEST(Executables, WriteDepinningSkipsRunawayAndWritesFc) {
    DepinningConfig cfg;
    cfg.Lx = 8.0;
    cfg.pinning_length_over_xi = 5.0;

    DepinningResult res;
    res.f_c_upper_bound = 0.8;
    DepinningStep blocked;
    blocked.f = 0.0;
    blocked.result.blocked = true;
    blocked.h = {0.5, 0.5, 0.5};
    DepinningStep runaway;
    runaway.f = 0.8;
    runaway.result.blocked = false;
    runaway.result.runaway = true;  // h intentionally empty
    res.steps = {blocked, runaway};

    const std::string fname = "test_write_depinning.h5";
    {
        HighFive::File file(fname, HighFive::File::Truncate);
        write_depinning(file, cfg, res);
    }
    HighFive::File file(fname, HighFive::File::ReadOnly);

    ASSERT_TRUE(has_dataset(file, "driving_force=0/h"));
    ASSERT_TRUE(has_dataset(file, "driving_force=0/pinning_length"));
    // The runaway force stores no profile.
    EXPECT_FALSE(file.exist("driving_force=0.8/h"));

    double fc = 0;
    ASSERT_TRUE(has_dataset(file, "f_c_upper_bound"));
    file.getDataSet("f_c_upper_bound").read(fc);
    EXPECT_DOUBLE_EQ(fc, 0.8);
}
#endif  // QEW_WITH_HDF5
