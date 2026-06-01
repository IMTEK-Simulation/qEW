// Brownian (overdamped Langevin) dynamics of the qEW line, the C++/GPU port of
// ../../Overleaf/qEW/qew_dynamic.py. Integrates h with Euler-Maruyama under an
// annealed thermal length L_T and saves snapshots at logarithmically spaced
// times to dynamic_{model}.h5. Analysis stays in Python (postprocess + plot.py).
//
// Configuration is by the constants below, matching the Python convention.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>

#include "dynamics.h"
#include "filtered_noise.h"
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
        // --- configuration (cf. qew_dynamic.py module constants) ---
        const Model model = Model::Arclength;
        const char *model_name = "arclength";
        const real_t amplitude = 1.0;
        const real_t xi = 0.1;             // correlation length
        const real_t pinning_length = 0.01;
        const real_t driving_force = 0.0;
        const real_t Lx = 2.0, Ly = 1.0;
        const int nx = 2048, ny = 1024;
        const std::uint64_t seed = 1;

        const real_t tau = 1.0;
        const real_t tl_start = 0.01;
        const real_t tl_end = 0.0001;
        // qew_dynamic.py timestep heuristic.
        const real_t fac = xi / std::max(tl_start, tl_end);
        const real_t dt = std::min(static_cast<real_t>(0.1),
                                   tau / 10 * std::min(fac, fac * fac));
        // Production qew_dynamic.py uses total_time = 1e6; kept short here so the
        // demo finishes quickly. Increase for a real anneal.
        const real_t total_time = 1000.0;

        const real_t line_tension =
            std::pow(pinning_length / xi, static_cast<real_t>(1.5)) * amplitude;

        FilteredNoise noise(nx, ny, Lx, Ly, amplitude, xi, xi, seed);
        const DeviceNoise dn = noise.device_noise();

        Params p;
        p.physical_size = Lx;
        p.line_tension = line_tension;
        p.driving_force = driving_force;
        p.model = model;

        View1D h("h", nx);
        Kokkos::deep_copy(h, static_cast<real_t>(0.5) * Ly);  // flat start
        LangevinDynamics dyn(nx, seed + 1);

        const std::string fname = std::string("dynamic_") + model_name + ".h5";
        HighFive::File file(fname, HighFive::File::Truncate);
        file.createDataSet("physical_size", static_cast<double>(Lx));

        std::printf("dt=%.3g  total_time=%.3g  steps=%ld  line_tension=%.4g\n",
                    static_cast<double>(dt), static_cast<double>(total_time),
                    static_cast<long>(total_time / dt),
                    static_cast<double>(line_tension));

        auto save = [&](real_t t) {
            auto hh = Kokkos::create_mirror_view(h);
            Kokkos::deep_copy(hh, h);
            std::vector<double> hvec(nx);
            for (int i = 0; i < nx; ++i) hvec[i] = hh(i);
            const std::string prefix = "timestep=" + fmt("%.6g", t) +
                                       "/driving_force=" + fmt("%.1g", driving_force) +
                                       "/line_tension=" + fmt("%.3g", line_tension);
            file.createDataSet(prefix + "/pinning_length", pinning_length);
            file.createDataSet(prefix + "/h", hvec);
            std::printf("  saved t=%.6g\n", static_cast<double>(t));
        };

        real_t t = 0;
        real_t next_save = 10 * dt;
        real_t tl = tl_start;
        while (t < total_time) {
            const real_t drift = tl * dt / (tau * amplitude);
            const real_t diff = tl * std::sqrt(2 * dt / tau);
            dyn.step(h, p, dn, drift, diff);

            tl = tl_start + (tl_end - tl_start) * (t / total_time);  // anneal
            t += dt;
            if (t > next_save - dt / 2) {
                save(t);
                next_save *= 10;
            }
        }
        std::printf(
            "wrote %s  (run tools/postprocess_static.py to add acf/psd)\n",
            fname.c_str());
    }
    Kokkos::finalize();
    return 0;
}
