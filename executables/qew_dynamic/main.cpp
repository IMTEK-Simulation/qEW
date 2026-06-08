// Brownian (overdamped Langevin) dynamics of the qEW line, the C++/GPU port of
// ../../Overleaf/qEW/qew_dynamic.py. Integrates h with Euler-Maruyama under an
// annealed thermal length L_T and saves snapshots at logarithmically spaced
// times to dynamic_{model}.h5. Analysis stays in Python (postprocess + plot.py).
//
// The anneal loop lives in qew::dynamic_anneal (src/qew_runs.h) and the HDF5
// layout in qew::write_dynamic (src/qew_io.h); both are unit-tested. This main
// is just configuration + glue, matching the Python module-constant convention.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "filtered_noise.h"
#include "qew_io.h"
#include "qew_runs.h"
#include "qew_types.h"

using namespace qew;

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // --- configuration (cf. qew_dynamic.py module constants) ---
        DynamicConfig cfg;
        cfg.model = Model::Arclength;
        const char *model_name = "arclength";
        cfg.amplitude = 1.0;
        cfg.xi = 0.1;  // correlation length
        cfg.pinning_length = 0.01;
        cfg.driving_force = 0.0;
        cfg.Lx = 2.0;
        cfg.Ly = 1.0;
        cfg.nx = 2048;
        cfg.ny = 1024;
        cfg.seed = 1;
        cfg.tau = 1.0;
        cfg.tl_start = 0.01;
        cfg.tl_end = 0.0001;
        // Production qew_dynamic.py uses total_time = 1e6; kept short here so the
        // demo finishes quickly. Increase for a real anneal.
        cfg.total_time = 1000.0;

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

        const std::string fname = std::string("dynamic_") + model_name + ".h5";
        HighFive::File file(fname, HighFive::File::Truncate);
        write_dynamic(file, cfg, res);
        std::printf(
            "wrote %s  (run tools/postprocess_static.py to add acf/psd)\n",
            fname.c_str());
    }
    Kokkos::finalize();
    return 0;
}
