// Thread-scaling benchmark for the qEW hot-loop kernels. Times the parallel
// kernels (gradient, objective, Langevin step) on a large line, excluding the
// one-time noise setup. Run at several OMP_NUM_THREADS to measure scaling:
//
//   for t in 1 2 4 8 16; do OMP_NUM_THREADS=$t ./benchmark; done

#include <cstdio>

#include <Kokkos_Core.hpp>

#include "dynamics.h"
#include "filtered_noise.h"
#include "lbfgs.h"
#include "qew_model.h"
#include "qew_types.h"

using namespace qew;

int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int nx = 262144;   // line length (the parallel dimension)
        const int ny = 64;       // noise y-resolution (setup only)
        const real_t Lx = 256.0, Ly = 1.0;
        const int reps = 200;

        const int nthreads = ExecSpace().concurrency();

        Kokkos::Timer timer;
        FilteredNoise noise(nx, ny, Lx, Ly, 1.0, 0.1, 0.1, /*seed=*/1);
        const double t_setup = timer.seconds();
        const DeviceNoise dn = noise.device_noise();

        Params p;
        p.physical_size = Lx;
        p.line_tension = 1.0;
        p.driving_force = 0.1;
        p.model = Model::Arclength;

        View1D h("h", nx), grad("grad", nx);
        Kokkos::deep_copy(h, static_cast<real_t>(0.5) * Ly);

        // Warm up (first launch pays one-off costs).
        gradient(h, p, dn, grad);
        Kokkos::fence();

        // gradient(): parallel_for over the line (2 bicubic evals/site).
        timer.reset();
        for (int k = 0; k < reps; ++k) gradient(h, p, dn, grad);
        Kokkos::fence();
        const double t_grad = timer.seconds();

        // objective(): parallel_reduce over the line.
        timer.reset();
        volatile real_t sink = 0;
        for (int k = 0; k < reps; ++k) sink += objective(h, p, dn);
        Kokkos::fence();
        const double t_obj = timer.seconds();
        (void)sink;

        // Langevin step: gradient + RNG kick (parallel).
        LangevinDynamics dyn(nx, /*seed=*/7);
        timer.reset();
        for (int k = 0; k < reps; ++k) dyn.step(h, p, dn, 0.01, 0.01);
        Kokkos::fence();
        const double t_dyn = timer.seconds();

        std::printf(
            "threads=%2d  setup=%6.3fs  grad=%7.3f ms/ea  obj=%7.3f ms/ea  "
            "dyn=%7.3f ms/ea\n",
            nthreads, t_setup, 1e3 * t_grad / reps, 1e3 * t_obj / reps,
            1e3 * t_dyn / reps);

        // End-to-end L-BFGS relaxation shaped like a qew_static run
        // (preconditioned, flat start). Tracks the minimiser-loop overheads
        // (line-search objective evals, two-loop recursion, history updates)
        // that the per-kernel timings above cannot see.
        const int snx = 4096, sny = 512;
        const real_t sLx = 32.0, sLy = 1.0, sxi = 0.1;
        FilteredNoise snoise(snx, sny, sLx, sLy, 1.0, sxi, sxi, /*seed=*/1);
        Params sp;
        sp.physical_size = sLx;
        sp.line_tension = 1.0;  // lambda_p/xi = 1
        sp.driving_force = 0.0;
        sp.model = Model::Arclength;
        View1D hs("hs", snx);
        Kokkos::deep_copy(hs, static_cast<real_t>(0.5) * sLy);
        LbfgsParams sopt;
        sopt.ftol = 1e-6;
        sopt.max_iter = 5000;
        sopt.precondition = true;
        sopt.precond_shift =
            static_cast<real_t>(0.5) * (sLx / snx) / (sxi * sxi);
        timer.reset();
        const LbfgsResult sres =
            lbfgs_minimize(hs, sp, snoise.device_noise(), sopt);
        const double t_lbfgs = timer.seconds();
        std::printf(
            "lbfgs(n=%d, precond): %6.3fs  iters=%d  feval=%ld  %s\n", snx,
            t_lbfgs, sres.iterations, sres.n_func_evals,
            sres.converged ? "converged" : "NOT converged");
    }
    Kokkos::finalize();
    return 0;
}
