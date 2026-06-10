// Brownian (overdamped Langevin) dynamics of the qEW line, the C++/GPU port of
// ../../Overleaf/qEW/qew_dynamic.py. Integrates h with Euler-Maruyama under an
// annealed thermal length L_T and saves snapshots at logarithmically spaced
// times to dynamic_{model}.h5. Analysis stays in Python (postprocess + plot.py).
//
// The anneal loop lives in qew::dynamic_anneal (src/qew_runs.h) and the HDF5
// layout in qew::write_dynamic (src/qew_io.h); both are unit-tested. This main
// is configuration + glue: it seeds the config with the paper's defaults, then
// lets command-line options override any of them so parameter sweeps need no
// rebuild (run with --help for the list).

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
    "qew_dynamic -- annealed overdamped Langevin dynamics.\n"
    "All options take a value; defaults match qew_dynamic.py.\n\n"
    "  --model           linear|arclength      (default arclength)\n"
    "  --amplitude       disorder amplitude U0  (default 1)\n"
    "  --xi              correlation length     (default 0.1)\n"
    "  --pinning_length  lambda_p/xi            (default 0.1)\n"
    "  --driving_force   external force         (default 0)\n"
    "  --Lx --Ly         system size            (default 2, 1)\n"
    "  --nx --ny         resolution             (default 2048, 1024)\n"
    "  --tau             damping time           (default 1)\n"
    "  --tl_start        anneal thermal length, start (default 0.01)\n"
    "  --tl_end          anneal thermal length, end   (default 0.0001)\n"
    "  --total_time      integration time       (default 1000; production ~1e6)\n"
    "  --seed            noise/RNG seed          (default 1)\n"
    "  --output          HDF5 filename          (default dynamic_<model>.h5)\n";

int run(int argc, char **argv) {
    cli::Args args(argc, argv);
    if (args.help) {
        std::fputs(kUsage, stdout);
        return 0;
    }

    // --- configuration: paper defaults, overridable on the command line ---
    DynamicConfig cfg;
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
    // lambda_p/xi = 0.1 (floppy).
    cfg.pinning_length_over_xi = args.dbl("pinning_length", 0.1);
    cfg.driving_force = args.dbl("driving_force", 0.0);
    cfg.Lx = args.dbl("Lx", 2.0);
    cfg.Ly = args.dbl("Ly", 1.0);
    cfg.nx = args.integer("nx", 2048);
    cfg.ny = args.integer("ny", 1024);
    cfg.seed = args.u64("seed", 1);
    cfg.tau = args.dbl("tau", 1.0);
    cfg.tl_start = args.dbl("tl_start", 0.01);
    cfg.tl_end = args.dbl("tl_end", 0.0001);
    // Production qew_dynamic.py uses total_time = 1e6; default kept short so the
    // demo finishes quickly. Raise --total_time for a real anneal.
    cfg.total_time = args.dbl("total_time", 1000.0);
    const std::string fname =
        args.str("output", std::string("dynamic_") + model_name + ".h5");
    args.require_all_used();

    FilteredNoise noise(cfg.nx, cfg.ny, cfg.Lx, cfg.Ly, cfg.amplitude, cfg.xi,
                        cfg.xi, cfg.seed);
    const DynamicResult res = dynamic_anneal(cfg, noise.device_noise());

    std::printf("dt=%.3g  total_time=%.3g  steps=%ld  line_tension=%.4g\n",
                static_cast<double>(res.dt),
                static_cast<double>(res.total_time),
                static_cast<long>(res.total_time / res.dt),
                static_cast<double>(res.line_tension));
    for (const DynamicSnapshot &snap : res.snapshots)
        std::printf("  saved t=%.6g\n", static_cast<double>(snap.t));

    HighFive::File file(fname, HighFive::File::Truncate);
    write_dynamic(file, cfg, res);
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
        std::fprintf(stderr, "qew_dynamic: %s\n\n", e.what());
        std::fputs(kUsage, stderr);
        rc = 2;
    }
    Kokkos::finalize();
    return rc;
}
