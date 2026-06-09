#ifndef QEW_RUNS_H
#define QEW_RUNS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "dynamics.h"
#include "lbfgs.h"
#include "qew_model.h"
#include "qew_types.h"
#include "rosso_krauth.h"

// Orchestration of the three production simulations, factored out of the
// executables' main() so it can be unit-tested without HDF5 (the I/O is split
// into qew_io.h). Each routine is templated on the `Noise` functor, exactly like
// the kernels it drives, so tests can pass a cheap analytic potential instead of
// paying for a FilteredNoise FFT setup. The loop bodies are a faithful copy of
// the original mains -- behaviour (timestep heuristic, anneal schedule, force
// ramp + f_c bracketing) is unchanged.
//
// Pinning-length convention (Le et al., arXiv:2410.21838, eq. (10)): the Larkin
// length obeys lambda_p / xi = (Gamma / U0)^{2/3}, varied by holding the
// disorder (amplitude U0, correlation length xi) fixed and scanning the line
// tension Gamma. All drivers therefore take the DIMENSIONLESS ratio
// lambda_p/xi and invert it as
//     Gamma = amplitude * (lambda_p/xi)^{3/2}.
// The ratio is also what the HDF5 "pinning_length" keys/values carry, matching
// the paper's labelling (lambda_p/xi = 0.1 floppy ... 100 stiff).
namespace qew {

inline std::vector<double> to_host_vec(const View1D &h) {
    auto hh = Kokkos::create_mirror_view(h);
    Kokkos::deep_copy(hh, h);
    std::vector<double> out(h.extent(0));
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = hh(i);
    return out;
}

// ---------------------------------------------------------------------------
// Static: L-BFGS relaxation sweeping the pinning length (cf. qew_static.py).
// ---------------------------------------------------------------------------
struct StaticConfig {
    Model model = Model::Arclength;
    real_t amplitude = 1.0;
    real_t xi = 0.1;            // correlation length; MUST match the noise's xi
    real_t Lx = 32.0;
    real_t Ly = 1.0;
    int nx = 4096;
    int ny = 512;               // noise y-resolution (used only at noise setup)
    real_t driving_force = 0.0;
    // lambda_p/xi ratios to sweep, floppy (<1) to stiff (>1).
    std::vector<double> pinning_lengths_over_xi = {0.01, 0.1, 1.0, 10.0, 100.0};
    real_t ftol = 1e-6;
    int max_iter = 5000;
};

struct StaticProfile {
    double pinning_length_over_xi = 0;
    real_t line_tension = 0;
    std::vector<double> h;
    LbfgsResult result;
};

template <class Noise>
std::vector<StaticProfile> static_sweep(const StaticConfig &c, const Noise &noise) {
    std::vector<StaticProfile> out;
    out.reserve(c.pinning_lengths_over_xi.size());
    for (double lp : c.pinning_lengths_over_xi) {
        Params p;
        p.physical_size = c.Lx;
        p.line_tension = std::pow(lp, static_cast<real_t>(1.5)) * c.amplitude;
        p.driving_force = c.driving_force;
        p.model = c.model;

        View1D h("h", c.nx);
        Kokkos::deep_copy(h, static_cast<real_t>(0.5) * c.Ly);  // flat start

        LbfgsParams opt;
        opt.ftol = c.ftol;
        opt.max_iter = c.max_iter;
        // Inverse-Laplacian FFT preconditioner with the pinning-curvature shift
        // ~ dx*A/xi^2 (degrades to a well-scaled scalar step on floppy lines).
        opt.precondition = true;
        opt.precond_shift = static_cast<real_t>(0.5) * (c.Lx / c.nx) * c.amplitude /
                            (c.xi * c.xi);
        const LbfgsResult r = lbfgs_minimize(h, p, noise, opt);

        StaticProfile prof;
        prof.pinning_length_over_xi = lp;
        prof.line_tension = p.line_tension;
        prof.h = to_host_vec(h);
        prof.result = r;
        out.push_back(std::move(prof));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Dynamic: annealed overdamped Langevin (cf. qew_dynamic.py).
// ---------------------------------------------------------------------------
struct DynamicConfig {
    Model model = Model::Arclength;
    real_t amplitude = 1.0;
    real_t xi = 0.1;            // correlation length; MUST match the noise's xi
    real_t pinning_length_over_xi = 0.1;  // lambda_p/xi (floppy)
    real_t driving_force = 0.0;
    real_t Lx = 2.0;
    real_t Ly = 1.0;
    int nx = 2048;
    int ny = 1024;
    std::uint64_t seed = 1;
    real_t tau = 1.0;
    real_t tl_start = 0.01;
    real_t tl_end = 0.0001;
    real_t total_time = 1000.0;
};

struct DynamicSnapshot {
    real_t t = 0;
    std::vector<double> h;
};

struct DynamicResult {
    real_t dt = 0;
    real_t total_time = 0;
    real_t line_tension = 0;
    std::vector<DynamicSnapshot> snapshots;
};

template <class Noise>
DynamicResult dynamic_anneal(const DynamicConfig &c, const Noise &noise) {
    const real_t tau = c.tau;
    // qew_dynamic.py timestep heuristic.
    const real_t fac = c.xi / std::max(c.tl_start, c.tl_end);
    const real_t dt = std::min(static_cast<real_t>(0.1),
                               tau / 10 * std::min(fac, fac * fac));
    const real_t line_tension =
        std::pow(c.pinning_length_over_xi, static_cast<real_t>(1.5)) * c.amplitude;

    Params p;
    p.physical_size = c.Lx;
    p.line_tension = line_tension;
    p.driving_force = c.driving_force;
    p.model = c.model;

    View1D h("h", c.nx);
    Kokkos::deep_copy(h, static_cast<real_t>(0.5) * c.Ly);  // flat start
    LangevinDynamics dyn(c.nx, c.seed + 1);

    DynamicResult res;
    res.dt = dt;
    res.total_time = c.total_time;
    res.line_tension = line_tension;

    real_t t = 0;
    real_t next_save = 10 * dt;
    real_t tl = c.tl_start;
    while (t < c.total_time) {
        const real_t drift = tl * dt / (tau * c.amplitude);
        const real_t diff = tl * std::sqrt(2 * dt / tau);
        dyn.step(h, p, noise, drift, diff);

        tl = c.tl_start + (c.tl_end - c.tl_start) * (t / c.total_time);  // anneal
        t += dt;
        if (t > next_save - dt / 2) {
            res.snapshots.push_back(DynamicSnapshot{t, to_host_vec(h)});
            next_save *= 10;
        }
    }
    return res;
}

// ---------------------------------------------------------------------------
// Depinning: Rosso-Krauth force ramp toward f_c (cf. qew_depinning.py).
// ---------------------------------------------------------------------------
struct DepinningConfig {
    Model model = Model::Arclength;
    real_t amplitude = 1.0;
    real_t xi = 0.1;            // correlation length; MUST match the noise's xi
    real_t pinning_length_over_xi = 5.0;  // lambda_p/xi (stiff)
    real_t Lx = 8.0;
    real_t Ly = 1.0;
    int nx = 1024;
    int ny = 256;
    real_t f_step = 1.0;
    real_t f_max = 40.0;
    RkParams rp;  // caller fills dstep/max_advance/runaway/tolerances
};

struct DepinningStep {
    real_t f = 0;
    RkResult result;
    std::vector<double> h;  // populated only when result.blocked
};

struct DepinningResult {
    real_t line_tension = 0;
    real_t f_c_upper_bound = -1;  // first force with no blocked state, else -1
    std::vector<DepinningStep> steps;
};

template <class Noise>
DepinningResult depinning_ramp(const DepinningConfig &c, const Noise &noise) {
    const real_t line_tension =
        std::pow(c.pinning_length_over_xi, static_cast<real_t>(1.5)) * c.amplitude;

    Params p;
    p.physical_size = c.Lx;
    p.line_tension = line_tension;
    p.model = c.model;

    View1D h("h", c.nx);
    Kokkos::deep_copy(h, static_cast<real_t>(0.5) * c.Ly);  // flat start

    DepinningResult res;
    res.line_tension = line_tension;

    // Integer counter: accumulating f += f_step drifts for non-representable
    // steps and can drop the final force (and perturb the HDF5 key strings).
    for (int k = 0; static_cast<real_t>(k) * c.f_step <= c.f_max; ++k) {
        const real_t f = static_cast<real_t>(k) * c.f_step;
        p.driving_force = f;
        const RkResult r = rk_block(h, p, noise, c.rp);  // restart from previous

        DepinningStep st;
        st.f = f;
        st.result = r;
        if (r.blocked) {
            st.h = to_host_vec(h);
        } else {
            res.f_c_upper_bound = f;  // first unblocked force brackets f_c
        }
        res.steps.push_back(std::move(st));
        if (!r.blocked) break;
    }
    return res;
}

}  // namespace qew

#endif  // QEW_RUNS_H
