// Quasi-static depinning of the qEW line via the Rosso-Krauth construction,
// the C++/GPU port of ../../Overleaf/qEW/qew_depinning.py. Ramps the driving
// force; at each force the blocked (critical) configuration is built directly by
// the no-passing forward relaxation, restarting from the previous one. The ramp
// stops when no blocked state exists (the line runs away) -- that brackets f_c.
// Profiles are written to depinning_{model}.h5; analysis stays in Python.
//
// The ramp lives in qew::depinning_ramp (src/qew_runs.h) and the HDF5 layout in
// qew::write_depinning (src/qew_io.h); both are unit-tested. This main is
// configuration + glue: it seeds the config with the paper's defaults, then lets
// command-line options override any of them so parameter sweeps need no rebuild
// (run with --help for the list).

#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "../cli.h"
#include "filtered_noise.h"
#include "qew_io.h"
#include "qew_runs.h"
#include "qew_types.h"

using namespace qew;

namespace {
const char *kUsage =
    "qew_depinning -- Rosso-Krauth force ramp toward the threshold f_c.\n"
    "All options take a value; defaults match qew_depinning.py.\n\n"
    "  --model           linear|arclength      (default arclength)\n"
    "  --amplitude       disorder amplitude U0  (default 1)\n"
    "  --xi              correlation length     (default 0.1)\n"
    "  --pinning_length  lambda_p/xi (stiff)    (default 5)\n"
    "  --Lx --Ly         system size            (default 8, 1)\n"
    "  --nx --ny         resolution             (default 1024, 256)\n"
    "  --f_step          force ramp increment   (default 1)\n"
    "  --f_max           force ramp cap         (default 40)\n"
    "  --seed            noise RNG seed          (default 1)\n"
    "  --output          HDF5 filename          (default depinning_<model>.h5)\n"
    "  RK solver knobs (defaults derived from Ly/ny):\n"
    "  --dstep --max_advance --runaway --root_tol --sweep_tol --max_sweeps\n";

int run(int argc, char **argv) {
    cli::Args args(argc, argv);
    if (args.help) {
        std::fputs(kUsage, stdout);
        return 0;
    }

    // --- configuration: paper defaults, overridable on the command line ---
    DepinningConfig cfg;
    const std::string model_name = args.str("model", "arclength");
    if (model_name == "linear") {
        cfg.model = Model::Linear;
    } else if (model_name == "arclength") {
        cfg.model = Model::Arclength;
    } else {
        throw std::runtime_error("--model must be 'linear' or 'arclength', got '" +
                                 model_name + "'");
    }
    cfg.amplitude = args.dbl("amplitude", 1.0);
    cfg.xi = args.dbl("xi", 0.1);                       // correlation length
    cfg.pinning_length_over_xi = args.dbl("pinning_length", 5.0);  // stiff
    cfg.Lx = args.dbl("Lx", 8.0);
    cfg.Ly = args.dbl("Ly", 1.0);
    cfg.nx = args.integer("nx", 1024);
    cfg.ny = args.integer("ny", 256);
    cfg.f_step = args.dbl("f_step", 1.0);  // ramp from 0 up until the line depins
    cfg.f_max = args.dbl("f_max", 40.0);
    // RK solver knobs: defaults derived from Ly/ny (read above), each overridable.
    cfg.rp.dstep =
        args.dbl("dstep", static_cast<double>(0.5) * (cfg.Ly / cfg.ny));
    cfg.rp.max_advance = args.dbl("max_advance", cfg.Ly);
    cfg.rp.runaway = args.dbl("runaway", 2.0 * cfg.Ly);  // two periods unblocked
    cfg.rp.root_tol = args.dbl("root_tol", 1e-10);
    cfg.rp.sweep_tol = args.dbl("sweep_tol", 1e-8);
    cfg.rp.max_sweeps = args.integer("max_sweeps", 200000);
    const std::uint64_t seed = args.u64("seed", 1);
    const std::string fname =
        args.str("output", std::string("depinning_") + model_name + ".h5");
    args.require_all_used();

    FilteredNoise noise(cfg.nx, cfg.ny, cfg.Lx, cfg.Ly, cfg.amplitude, cfg.xi,
                        cfg.xi, seed);
    const DepinningResult res = depinning_ramp(cfg, noise.device_noise());

    std::printf("model=%s  lambda_p/xi=%.3g  line_tension=%.4g\n",
                model_name.c_str(),
                static_cast<double>(cfg.pinning_length_over_xi),
                static_cast<double>(res.line_tension));
    for (const DepinningStep &st : res.steps) {
        std::printf("f=%7.3f  sweeps=%-7d mean_h=%8.4f  %s\n",
                    static_cast<double>(st.f), st.result.sweeps,
                    static_cast<double>(st.result.mean_h),
                    st.result.blocked
                        ? "blocked"
                        : (st.result.runaway ? "DEPINNED" : "timeout"));
    }
    if (res.f_c_upper_bound >= 0)
        std::printf("Line depinned near f=%.3f; stopping ramp.\n",
                    static_cast<double>(res.f_c_upper_bound));

    HighFive::File file(fname, HighFive::File::Truncate);
    write_depinning(file, cfg, res);
    std::printf("wrote %s  (run tools/postprocess_static.py to add acf/psd)\n",
                fname.c_str());
    return 0;
}
}  // namespace

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    int rc = 0;
    try {
        rc = run(argc, argv);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "qew_depinning: %s\n\n", e.what());
        std::fputs(kUsage, stderr);
        rc = 2;
    }
    Kokkos::finalize();
    return rc;
}
