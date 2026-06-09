// Static qEW: L-BFGS energy minimisation sweeping the pinning length, the
// C++/GPU port of ../../Overleaf/qEW/qew_static.py. Relaxes a flat line to a
// local minimum at zero driving force for each pinning length and writes the
// profiles to static_{model}.h5. Analysis (ACF/PSD/Hurst) stays in Python: run
// tools/postprocess_static.py on the output to add the datasets plot.py reads.
//
// The sweep itself lives in qew::static_sweep (src/qew_runs.h) and the HDF5
// layout in qew::write_static (src/qew_io.h); both are unit-tested. This main is
// just the configuration + glue, matching the Python module-constant convention.

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
        // --- configuration (cf. qew_static.py module constants) ---
        StaticConfig cfg;
        cfg.model = Model::Arclength;
        const char *model_name = "arclength";
        cfg.amplitude = 1.0;
        cfg.xi = 0.1;  // correlation length
        cfg.Lx = 32.0;
        cfg.Ly = 1.0;
        cfg.nx = 4096;
        cfg.ny = 512;
        cfg.driving_force = 0.0;
        cfg.pinning_lengths = {0.01, 0.1, 1.0, 10.0, 100.0};
        cfg.ftol = 1e-6;  // ample for roughness analysis
        cfg.max_iter = 5000;
        const std::uint64_t seed = 1;

        FilteredNoise noise(cfg.nx, cfg.ny, cfg.Lx, cfg.Ly, cfg.amplitude, cfg.xi,
                            cfg.xi, seed);
        const std::vector<StaticProfile> profiles =
            static_sweep(cfg, noise.device_noise());

        for (const StaticProfile &prof : profiles) {
            std::printf("lp=%-6.3g lt=%-10.4g iters=%-7d max|F|=%.3e  %s\n",
                        prof.pinning_length, static_cast<double>(prof.line_tension),
                        prof.result.iterations,
                        static_cast<double>(prof.result.max_force),
                        prof.result.converged ? "converged" : "NOT converged");
        }

        const std::string fname = std::string("static_") + model_name + ".h5";
        HighFive::File file(fname, HighFive::File::Truncate);
        write_static(file, cfg, profiles);
        std::printf(
            "wrote %s  (run tools/postprocess_static.py to add acf/psd for "
            "plot.py)\n",
            fname.c_str());
    }
    Kokkos::finalize();
    return 0;
}
