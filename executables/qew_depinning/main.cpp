// Quasi-static depinning of the qEW line via the Rosso-Krauth construction,
// the C++/GPU port of ../../Overleaf/qEW/qew_depinning.py. Ramps the driving
// force; at each force the blocked (critical) configuration is built directly by
// the no-passing forward relaxation, restarting from the previous one. The ramp
// stops when no blocked state exists (the line runs away) -- that brackets f_c.
// Profiles are written to depinning_{model}.h5; analysis stays in Python.
//
// The ramp lives in qew::depinning_ramp (src/qew_runs.h) and the HDF5 layout in
// qew::write_depinning (src/qew_io.h); both are unit-tested. This main is just
// configuration + glue, matching the Python module-constant convention.

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
        // --- configuration (cf. qew_depinning.py) ---
        DepinningConfig cfg;
        cfg.model = Model::Arclength;
        const char *model_name = "arclength";
        cfg.amplitude = 1.0;
        cfg.xi = 0.1;             // correlation length
        cfg.pinning_length = 5.0;  // lambda_p/xi (stiff)
        cfg.Lx = 8.0;
        cfg.Ly = 1.0;
        cfg.nx = 1024;
        cfg.ny = 256;
        cfg.f_step = 1.0;  // force ramp: from 0 upward until the line depins
        cfg.f_max = 40.0;
        cfg.rp.dstep = static_cast<real_t>(0.5) * (cfg.Ly / cfg.ny);
        cfg.rp.max_advance = cfg.Ly;
        cfg.rp.runaway = 2 * cfg.Ly;  // advanced two periods without blocking
        cfg.rp.root_tol = 1e-10;
        cfg.rp.sweep_tol = 1e-8;
        cfg.rp.max_sweeps = 200000;
        const std::uint64_t seed = 1;

        FilteredNoise noise(cfg.nx, cfg.ny, cfg.Lx, cfg.Ly, cfg.amplitude, cfg.xi,
                            cfg.xi, seed);
        const DepinningResult res = depinning_ramp(cfg, noise.device_noise());

        std::printf("model=%s  lambda_p/xi=%.3g  line_tension=%.4g\n", model_name,
                    static_cast<double>(cfg.pinning_length),
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

        const std::string fname = std::string("depinning_") + model_name + ".h5";
        HighFive::File file(fname, HighFive::File::Truncate);
        write_depinning(file, cfg, res);
        std::printf("wrote %s  (run tools/postprocess_static.py to add acf/psd)\n",
                    fname.c_str());
    }
    Kokkos::finalize();
    return 0;
}
