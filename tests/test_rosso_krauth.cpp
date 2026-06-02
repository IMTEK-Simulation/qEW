#include "rosso_krauth.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "filtered_noise.h"
#include "qew_model.h"
#include "qew_types.h"

using namespace qew;

namespace {

// Sinusoidal pinning, no x dependence: V = A cos(k y). For a flat line the
// elastic term vanishes, so the model reduces to a single particle driven
// through a washboard potential, whose depinning threshold is the maximum
// pinning force, f_c = max_y dV/dy = A k.  (k = 2*pi/lambda)
struct SinusoidNoise {
    real_t A = 0.1;
    real_t k = 2.0 * 3.14159265358979323846;  // lambda = 1
    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t, real_t y) const {
        return NoiseSample{A * Kokkos::cos(k * y), 0, -A * k * Kokkos::sin(k * y)};
    }
};

View1D flat_line(int n, real_t value) {
    View1D h("h", n);
    Kokkos::deep_copy(h, value);
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

// Strong analytic check: bisection on f recovers the single-particle depinning
// threshold f_c = A k for the washboard potential.
TEST(RossoKrauth, RecoversAnalyticCriticalForce) {
    const int n = 16;
    const SinusoidNoise noise;
    const real_t fc_exact = noise.A * noise.k;  // = 0.2*pi
    const real_t lambda = 2 * 3.14159265358979323846 / noise.k;

    Params p;
    p.physical_size = n;  // dx = 1 (irrelevant: flat line, no elastic)
    p.line_tension = 1.0;
    p.model = Model::Linear;

    RkParams rp;
    rp.dstep = lambda / 200;
    rp.max_advance = lambda;
    rp.runaway = 3 * lambda;
    rp.root_tol = 1e-12;
    rp.sweep_tol = 1e-11;
    rp.max_sweeps = 100000;

    auto blocks = [&](real_t f) {
        p.driving_force = f;
        View1D h = flat_line(n, 0.0);
        return rk_block(h, p, noise, rp).blocked;
    };

    // Bisection: f_lo blocks, f_hi runs away.
    real_t f_lo = 0.0, f_hi = 2 * fc_exact;
    ASSERT_TRUE(blocks(f_lo));
    ASSERT_FALSE(blocks(f_hi));
    for (int it = 0; it < 40; ++it) {
        const real_t fm = 0.5 * (f_lo + f_hi);
        if (blocks(fm))
            f_lo = fm;
        else
            f_hi = fm;
    }
    const real_t fc = 0.5 * (f_lo + f_hi);
    EXPECT_NEAR(fc, fc_exact, 0.02 * fc_exact);
}

// Direct equivalence: the per-site velocity the solver uses equals -dE/dh_i from
// qew_model for ARBITRARY configurations (not just blocked ones). This pins the
// hand-written rk_site_velocity to the finite-difference-validated gradient().
TEST(RossoKrauth, SiteVelocityMatchesGradient) {
    const int nx = 64, ny = 64;
    const real_t Lx = 3.2, Ly = 1.0;
    FilteredNoise noise(nx, ny, Lx, Ly, /*amplitude=*/0.7, 0.2, 0.2, /*seed=*/13);
    const DeviceNoise dn = noise.device_noise();

    for (const Model model : {Model::Linear, Model::Arclength}) {
        Params p;
        p.physical_size = Lx;
        p.line_tension = 1.3;
        p.driving_force = 0.4;
        p.model = model;

        View1D h("h", nx);
        auto hh = Kokkos::create_mirror_view(h);
        std::mt19937 rng(21);
        std::uniform_real_distribution<real_t> dist(0.1 * Ly, 0.9 * Ly);
        for (int i = 0; i < nx; ++i) hh(i) = dist(rng);
        Kokkos::deep_copy(h, hh);

        View1D grad("grad", nx);
        gradient(h, p, dn, grad);

        const int n = nx;
        const real_t dx = Lx / nx;
        const real_t lt = p.line_tension;
        const real_t f = p.driving_force;
        View1D vel("vel", nx);
        Kokkos::parallel_for(
            "rk_vel", nx, KOKKOS_LAMBDA(const int i) {
                const int im = (i - 1 + n) % n;
                const int ip = (i + 1) % n;
                vel(i) = rk_site_velocity(i, n, dx, lt, f, model, h(i), h(im),
                                          h(ip), dn);
            });

        real_t max_diff = 0;
        Kokkos::parallel_reduce(
            "cmp", nx,
            KOKKOS_LAMBDA(const int i, real_t &acc) {
                const real_t diff = Kokkos::fabs(vel(i) - (-grad(i)));
                if (diff > acc) acc = diff;
            },
            Kokkos::Max<real_t>(max_diff));
        EXPECT_LT(max_diff, 1e-12)
            << "model=" << (model == Model::Linear ? "linear" : "arclength");
    }
}

// The blocked configuration is forward of the start (monotone construction) and
// has no site with positive velocity (v_i = -dE/dh_i <= 0 everywhere).
TEST(RossoKrauth, BlockedConfigIsForwardAndArrested) {
    const int nx = 64, ny = 64;
    const real_t Lx = 3.2, Ly = 1.0;
    FilteredNoise noise(nx, ny, Lx, Ly, /*amplitude=*/0.5, 0.2, 0.2, /*seed=*/7);
    const DeviceNoise dn = noise.device_noise();

    Params p;
    p.physical_size = Lx;
    p.line_tension = 1.0;
    p.driving_force = 0.0;
    p.model = Model::Linear;

    View1D h = flat_line(nx, 0.5 * Ly);
    const auto h0 = to_host(h);

    RkParams rp;
    rp.dstep = 0.5 * (Ly / ny);
    rp.max_advance = Ly;
    rp.runaway = 3 * Ly;
    rp.root_tol = 1e-11;
    rp.sweep_tol = 1e-9;
    rp.max_sweeps = 200000;

    const RkResult r = rk_block(h, p, dn, rp);
    EXPECT_TRUE(r.blocked);

    const auto hf = to_host(h);
    for (int i = 0; i < nx; ++i) EXPECT_GE(hf[i], h0[i] - 1e-12);  // forward only

    // No site has positive velocity: max_i (-dE/dh_i) <= tol.
    View1D grad("grad", nx);
    gradient(h, p, dn, grad);
    real_t max_v = 0;
    Kokkos::parallel_reduce(
        "max_v", nx,
        KOKKOS_LAMBDA(const int i, real_t &acc) {
            const real_t v = -grad(i);
            if (v > acc) acc = v;
        },
        Kokkos::Max<real_t>(max_v));
    EXPECT_LT(max_v, 1e-6);
}

// Determinism: same force and start give an identical blocked configuration.
TEST(RossoKrauth, Deterministic) {
    const int nx = 48, ny = 48;
    const real_t Lx = 2.4, Ly = 1.0;
    FilteredNoise noise(nx, ny, Lx, Ly, 0.5, 0.2, 0.2, /*seed=*/9);
    const DeviceNoise dn = noise.device_noise();

    Params p;
    p.physical_size = Lx;
    p.line_tension = 1.5;
    p.driving_force = 0.3;
    p.model = Model::Arclength;

    RkParams rp;
    rp.dstep = 0.5 * (Ly / ny);
    rp.max_advance = Ly;
    rp.runaway = 5 * Ly;
    rp.root_tol = 1e-11;
    rp.sweep_tol = 1e-9;
    rp.max_sweeps = 200000;

    View1D h1 = flat_line(nx, 0.5 * Ly);
    View1D h2 = flat_line(nx, 0.5 * Ly);
    rk_block(h1, p, dn, rp);
    rk_block(h2, p, dn, rp);

    const auto a = to_host(h1), b = to_host(h2);
    for (int i = 0; i < nx; ++i) EXPECT_EQ(a[i], b[i]);
}
