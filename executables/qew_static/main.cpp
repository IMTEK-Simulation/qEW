// Static qEW: FIRE energy minimisation sweeping the pinning length, the C++/GPU
// port of ../../Overleaf/qEW/qew_static.py. Relaxes a flat line to a local
// minimum at zero driving force for each pinning length and writes the profiles
// to static_{model}.h5. Analysis (ACF/PSD/Hurst) stays in Python: run
// tools/postprocess_static.py on the output to add the datasets plot.py reads.
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
#include "lbfgs.h"
#include "qew_model.h"
#include "qew_types.h"

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
        // --- configuration (cf. qew_static.py module constants) ---
        const Model model = Model::Arclength;
        const char *model_name = "arclength";
        const real_t amplitude = 1.0;
        const real_t xi = 0.1;  // correlation length
        const real_t Lx = 32.0, Ly = 1.0;
        const int nx = 4096, ny = 512;
        const real_t driving_force = 0.0;
        const std::vector<double> pinning_lengths = {0.01, 0.1, 1.0, 10.0, 100.0};
        const std::uint64_t seed = 1;

        FilteredNoise noise(nx, ny, Lx, Ly, amplitude, xi, xi, seed);
        const DeviceNoise dn = noise.device_noise();

        const std::string fname = std::string("static_") + model_name + ".h5";
        HighFive::File file(fname, HighFive::File::Truncate);
        file.createDataSet("physical_size", static_cast<double>(Lx));

        for (double lp : pinning_lengths) {
            Params p;
            p.physical_size = Lx;
            p.line_tension = std::pow(lp, 1.5) * amplitude;
            p.driving_force = driving_force;
            p.model = model;

            View1D h("h", nx);
            Kokkos::deep_copy(h, static_cast<real_t>(0.5) * Ly);  // flat start

            LbfgsParams opt;
            opt.ftol = 1e-6;  // ample for roughness analysis
            opt.max_iter = 5000;
            // Inverse-Laplacian FFT preconditioner: removes the stiff lines'
            // ill-conditioning (cond ~ N^2). The shift is the pinning
            // (non-elastic) curvature scale ~ dx*A/xi^2, so the preconditioner
            // degrades to a well-scaled scalar step on floppy lines (where the
            // elastic term is negligible) and is the true Laplacian inverse on
            // stiff ones.
            opt.precondition = true;
            opt.precond_shift = static_cast<real_t>(0.5) * (Lx / nx) * amplitude / (xi * xi);
            const LbfgsResult r = lbfgs_minimize(h, p, dn, opt);

            auto hh = Kokkos::create_mirror_view(h);
            Kokkos::deep_copy(hh, h);
            std::vector<double> hvec(nx);
            for (int i = 0; i < nx; ++i) hvec[i] = hh(i);

            const std::string prefix = "driving_force=" + fmt("%.1g", driving_force) +
                                       "/pinning_length=" + fmt("%.3g", lp);
            file.createDataSet(prefix + "/pinning_length", lp);
            file.createDataSet(prefix + "/h", hvec);

            std::printf("lp=%-6.3g lt=%-10.4g iters=%-7d max|F|=%.3e  %s\n", lp,
                        static_cast<double>(p.line_tension), r.iterations,
                        static_cast<double>(r.max_force),
                        r.converged ? "converged" : "NOT converged");
        }
        std::printf(
            "wrote %s  (run tools/postprocess_static.py to add acf/psd for "
            "plot.py)\n",
            fname.c_str());
    }
    Kokkos::finalize();
    return 0;
}
