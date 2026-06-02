// Quasi-static depinning of the qEW line via the Rosso-Krauth construction,
// the C++/GPU port of ../../Overleaf/qEW/qew_depinning.py. Ramps the driving
// force; at each force the blocked (critical) configuration is built directly by
// the no-passing forward relaxation, restarting from the previous one. The ramp
// stops when no blocked state exists (the line runs away) -- that brackets f_c.
// Profiles are written to depinning_{model}.h5; analysis stays in Python.
//
// Configuration is by the constants below, matching the Python convention.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>

#include "filtered_noise.h"
#include "qew_model.h"
#include "qew_types.h"
#include "rosso_krauth.h"

using namespace qew;

namespace {
std::string fmt(const char *spec, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), spec, v);
    return std::string(buf);
}
}  // namespace

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // --- configuration (cf. qew_depinning.py) ---
        const Model model = Model::Arclength;
        const char *model_name = "arclength";
        const real_t amplitude = 1.0;
        const real_t xi = 0.1;             // correlation length
        const real_t pinning_length = 5.0;  // lambda_p/xi (stiff)
        const real_t Lx = 8.0, Ly = 1.0;
        const int nx = 1024, ny = 256;
        const std::uint64_t seed = 1;

        const real_t line_tension =
            std::pow(pinning_length, static_cast<real_t>(1.5)) * amplitude;

        // Force ramp: from 0 upward until the line depins.
        const real_t f_step = 1.0;
        const real_t f_max = 40.0;

        FilteredNoise noise(nx, ny, Lx, Ly, amplitude, xi, xi, seed);
        const DeviceNoise dn = noise.device_noise();

        Params p;
        p.physical_size = Lx;
        p.line_tension = line_tension;
        p.model = model;

        RkParams rp;
        rp.dstep = static_cast<real_t>(0.5) * (Ly / ny);
        rp.max_advance = Ly;
        rp.runaway = 2 * Ly;  // advanced two periods without blocking = depinned
        rp.root_tol = 1e-10;
        rp.sweep_tol = 1e-8;
        rp.max_sweeps = 200000;

        const std::string fname = std::string("depinning_") + model_name + ".h5";
        HighFive::File file(fname, HighFive::File::Truncate);
        file.createDataSet("physical_size", static_cast<double>(Lx));

        std::printf("model=%s  lambda_p/xi=%.3g  line_tension=%.4g\n", model_name,
                    static_cast<double>(pinning_length),
                    static_cast<double>(line_tension));

        View1D h("h", nx);
        Kokkos::deep_copy(h, static_cast<real_t>(0.5) * Ly);  // flat start

        real_t f_c = -1;
        for (real_t f = 0; f <= f_max; f += f_step) {
            p.driving_force = f;
            const RkResult r = rk_block(h, p, dn, rp);

            std::printf("f=%7.3f  sweeps=%-7d mean_h=%8.4f  %s\n",
                        static_cast<double>(f), r.sweeps,
                        static_cast<double>(r.mean_h),
                        r.blocked ? "blocked"
                                  : (r.runaway ? "DEPINNED" : "timeout"));

            if (r.blocked) {
                auto hh = Kokkos::create_mirror_view(h);
                Kokkos::deep_copy(hh, h);
                std::vector<double> hvec(nx);
                for (int i = 0; i < nx; ++i) hvec[i] = hh(i);

                const std::string prefix = "driving_force=" + fmt("%.4g", f);
                file.createDataSet(prefix + "/h", hvec);
                file.createDataSet(prefix + "/pinning_length",
                                   static_cast<double>(pinning_length));
            } else {
                f_c = f;  // first force with no blocked state brackets f_c
                std::printf("Line depinned near f=%.3f; stopping ramp.\n",
                            static_cast<double>(f));
                break;
            }
        }
        if (f_c >= 0) file.createDataSet("f_c_upper_bound", static_cast<double>(f_c));
        std::printf("wrote %s  (run tools/postprocess_static.py to add acf/psd)\n",
                    fname.c_str());
    }
    Kokkos::finalize();
    return 0;
}
