// Static qEW: L-BFGS energy minimisation sweeping the pinning length, the
// C++/GPU port of ../../Overleaf/qEW/qew_static.py. Relaxes a flat line to a
// local minimum at zero driving force for each pinning length and writes the
// profiles to static_{model}.h5. Analysis (ACF/PSD/Hurst) stays in Python: run
// tools/postprocess_static.py on the output to add the datasets plot.py reads.
//
// The sweep itself lives in qew::static_sweep (src/qew_runs.h) and the HDF5
// layout in qew::write_static (src/qew_io.h); both are unit-tested. This main is
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
    "qew_static -- L-BFGS relaxation, sweep over pinning length.\n"
    "All options take a value; defaults match qew_static.py.\n\n"
    "  --model           linear|arclength      (default arclength)\n"
    "  --amplitude       disorder amplitude U0  (default 1)\n"
    "  --xi              correlation length     (default 0.1)\n"
    "  --Lx --Ly         system size            (default 32, 1)\n"
    "  --nx --ny         resolution             (default 4096, 512)\n"
    "  --driving_force   external force         (default 0)\n"
    "  --pinning_lengths comma list of lambda_p/xi (default 0.01,0.1,1,10,100)\n"
    "  --ftol            L-BFGS tolerance       (default 1e-6)\n"
    "  --max_iter        L-BFGS iteration cap   (default 5000)\n"
    "  --seed            noise RNG seed         (default 1)\n"
    "  --output          HDF5 filename          (default static_<model>.h5)\n";

int run(int argc, char **argv) {
    cli::Args args(argc, argv);
    if (args.help) {
        std::fputs(kUsage, stdout);
        return 0;
    }

    // --- configuration: paper defaults, overridable on the command line ---
    StaticConfig cfg;
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
    cfg.xi = args.dbl("xi", 0.1);  // correlation length
    cfg.Lx = args.dbl("Lx", 32.0);
    cfg.Ly = args.dbl("Ly", 1.0);
    cfg.nx = args.integer("nx", 4096);
    cfg.ny = args.integer("ny", 512);
    cfg.driving_force = args.dbl("driving_force", 0.0);
    // lambda_p/xi: floppy (0.01) through stiff (100), cf. the paper's sweep.
    cfg.pinning_lengths_over_xi =
        args.dbl_list("pinning_lengths", {0.01, 0.1, 1.0, 10.0, 100.0});
    cfg.ftol = args.dbl("ftol", 1e-6);  // ample for roughness analysis
    cfg.max_iter = args.integer("max_iter", 5000);
    const std::uint64_t seed = args.u64("seed", 1);
    const std::string fname =
        args.str("output", std::string("static_") + model_name + ".h5");
    args.require_all_used();

    FilteredNoise noise(cfg.nx, cfg.ny, cfg.Lx, cfg.Ly, cfg.amplitude, cfg.xi,
                        cfg.xi, seed);
    const std::vector<StaticProfile> profiles =
        static_sweep(cfg, noise.device_noise());

    for (const StaticProfile &prof : profiles) {
        std::printf("lp/xi=%-6.3g lt=%-10.4g iters=%-7d max|F|=%.3e  %s\n",
                    prof.pinning_length_over_xi,
                    static_cast<double>(prof.line_tension),
                    prof.result.iterations,
                    static_cast<double>(prof.result.max_force),
                    prof.result.converged ? "converged" : "NOT converged");
    }

    HighFive::File file(fname, HighFive::File::Truncate);
    write_static(file, cfg, profiles);
    std::printf(
        "wrote %s  (run tools/postprocess_static.py to add acf/psd for "
        "plot.py)\n",
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
        std::fprintf(stderr, "qew_static: %s\n\n", e.what());
        std::fputs(kUsage, stderr);
        rc = 2;
    }
    Kokkos::finalize();
    return rc;
}
