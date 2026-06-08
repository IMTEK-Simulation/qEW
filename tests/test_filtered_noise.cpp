#include "filtered_noise.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <KokkosFFT.hpp>

#include "qew_model.h"
#include "qew_types.h"

#ifdef QEW_WITH_HDF5
#include <highfive/H5File.hpp>
#endif

using namespace qew;

namespace {

// Evaluate the device noise at a list of points via a portable parallel_for
// (runs on the GPU for a CUDA/HIP build), returning value and both derivatives.
void sample_points(const DeviceNoise &noise, const std::vector<real_t> &xs,
                   const std::vector<real_t> &ys, std::vector<real_t> &v,
                   std::vector<real_t> &dvdx, std::vector<real_t> &dvdy) {
    const int m = static_cast<int>(xs.size());
    View1D X("X", m), Y("Y", m), V("V", m), DX("DX", m), DY("DY", m);
    auto xh = Kokkos::create_mirror_view(X);
    auto yh = Kokkos::create_mirror_view(Y);
    for (int i = 0; i < m; ++i) {
        xh(i) = xs[i];
        yh(i) = ys[i];
    }
    Kokkos::deep_copy(X, xh);
    Kokkos::deep_copy(Y, yh);

    Kokkos::parallel_for(
        "sample", m, KOKKOS_LAMBDA(const int i) {
            const NoiseSample s = noise.sample(X(i), Y(i));
            V(i) = s.v;
            DX(i) = s.dv_dx;
            DY(i) = s.dv_dy;
        });

    v.resize(m);
    dvdx.resize(m);
    dvdy.resize(m);
    auto vh = Kokkos::create_mirror_view(V);
    auto dxh = Kokkos::create_mirror_view(DX);
    auto dyh = Kokkos::create_mirror_view(DY);
    Kokkos::deep_copy(vh, V);
    Kokkos::deep_copy(dxh, DX);
    Kokkos::deep_copy(dyh, DY);
    for (int i = 0; i < m; ++i) {
        v[i] = vh(i);
        dvdx[i] = dxh(i);
        dvdy[i] = dyh(i);
    }
}

}  // namespace

// Test 2: the generated field has the requested amplitude (population std) and
// is band-limited (Fourier modes outside the correlation ellipse vanish).
TEST(FilteredNoise, AmplitudeAndBandLimiting) {
    const int nx = 128, ny = 96;
    const real_t Lx = 4.0, Ly = 1.0, amplitude = 2.0, xi_x = 0.2, xi_y = 0.1;
    FilteredNoise noise(nx, ny, Lx, Ly, amplitude, xi_x, xi_y, /*seed=*/42);

    const auto &f = noise.field();
    double mean = 0;
    for (double val : f) mean += val;
    mean /= f.size();
    double var = 0;
    for (double val : f) var += (val - mean) * (val - mean);
    const double sd = std::sqrt(var / f.size());
    EXPECT_NEAR(sd, amplitude, 1e-10);

    // Forward FFT (kokkos-fft) and confirm masked modes are ~zero.
    using cplx = Kokkos::complex<double>;
    Kokkos::View<cplx **, MemSpace> A("A", nx, ny), Ahat("Ahat", nx, ny);
    auto Ah = Kokkos::create_mirror_view(A);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Ah(i, j) = cplx(f[static_cast<std::size_t>(i) * ny + j], 0);
    Kokkos::deep_copy(A, Ah);
    ExecSpace exec;
    KokkosFFT::fft2(exec, A, Ahat);
    exec.fence();
    auto Hh = Kokkos::create_mirror_view(Ahat);
    Kokkos::deep_copy(Hh, Ahat);

    double max_mag = 0, max_masked = 0;
    const double dx = Lx / nx, dy = Ly / ny;
    for (int i = 0; i < nx; ++i) {
        const int ksx = (i < (nx + 1) / 2) ? i : i - nx;
        const double fx = ksx / (double)nx / dx;
        for (int j = 0; j < ny; ++j) {
            const int ksy = (j < (ny + 1) / 2) ? j : j - ny;
            const double fy = ksy / (double)ny / dy;
            const double re = Hh(i, j).real(), im = Hh(i, j).imag();
            const double mag = std::sqrt(re * re + im * im);
            max_mag = std::max(max_mag, mag);
            if ((fx * xi_x) * (fx * xi_x) + (fy * xi_y) * (fy * xi_y) > 1.0)
                max_masked = std::max(max_masked, mag);
        }
    }
    EXPECT_LT(max_masked, 1e-8 * max_mag);
}

// Test 3a: the bicubic interpolant reproduces the sample values at grid nodes
// (the defining property of a correct prefilter + B-spline evaluation).
TEST(FilteredNoise, ReproducesValuesAtNodes) {
    const int nx = 64, ny = 48;
    const real_t Lx = 3.2, Ly = 1.0;
    FilteredNoise noise(nx, ny, Lx, Ly, 1.5, 0.15, 0.15, /*seed=*/3);
    const auto &f = noise.field();
    const real_t dx = noise.dx(), dy = noise.dy();

    std::vector<real_t> xs, ys;
    std::vector<std::pair<int, int>> nodes;
    for (int i = 0; i < nx; i += 5) {
        for (int j = 0; j < ny; j += 5) {
            xs.push_back(i * dx);
            ys.push_back(j * dy);
            nodes.emplace_back(i, j);
        }
    }
    std::vector<real_t> v, dvdx, dvdy;
    sample_points(noise.device_noise(), xs, ys, v, dvdx, dvdy);

    double max_err = 0;
    for (std::size_t k = 0; k < nodes.size(); ++k) {
        const double ref = f[static_cast<std::size_t>(nodes[k].first) * ny + nodes[k].second];
        max_err = std::max(max_err, std::abs(v[k] - ref));
    }
    EXPECT_LT(max_err, 1e-9);
}

// Test 3b: analytic derivatives agree with a central finite difference of the
// interpolated value (internal consistency of dv_dx / dv_dy).
TEST(FilteredNoise, AnalyticDerivativesMatchFiniteDifference) {
    const int nx = 64, ny = 48;
    const real_t Lx = 3.2, Ly = 1.0;
    FilteredNoise noise(nx, ny, Lx, Ly, 1.5, 0.15, 0.15, /*seed=*/5);
    const real_t dx = noise.dx(), dy = noise.dy();
    const real_t ex = 1e-5 * dx, ey = 1e-5 * dy;

    std::vector<real_t> xs, ys;
    std::mt19937 rng(99);
    std::uniform_real_distribution<real_t> ux(0, Lx), uy(0, Ly);
    for (int k = 0; k < 100; ++k) {
        xs.push_back(ux(rng));
        ys.push_back(uy(rng));
    }

    std::vector<real_t> v0, dvdx, dvdy, vxp, vxm, vyp, vym, junk;
    sample_points(noise.device_noise(), xs, ys, v0, dvdx, dvdy);

    std::vector<real_t> xp = xs, xm = xs, yp = ys, ym = ys;
    for (std::size_t k = 0; k < xs.size(); ++k) {
        xp[k] += ex; xm[k] -= ex; yp[k] += ey; ym[k] -= ey;
    }
    sample_points(noise.device_noise(), xp, ys, vxp, junk, junk);
    sample_points(noise.device_noise(), xm, ys, vxm, junk, junk);
    sample_points(noise.device_noise(), xs, yp, vyp, junk, junk);
    sample_points(noise.device_noise(), xs, ym, vym, junk, junk);

    double max_dx = 0, max_dy = 0;
    for (std::size_t k = 0; k < xs.size(); ++k) {
        max_dx = std::max(max_dx, std::abs((vxp[k] - vxm[k]) / (2 * ex) - dvdx[k]));
        max_dy = std::max(max_dy, std::abs((vyp[k] - vym[k]) / (2 * ey) - dvdy[k]));
    }
    EXPECT_LT(max_dx, 1e-5);
    EXPECT_LT(max_dy, 1e-5);
}

// Test 4 (integration): the real FilteredNoise plugged into qew_model keeps
// objective() and gradient() consistent. Because the noise derivative is now
// analytic (exact), the agreement is as tight as for the analytic stub -- and
// tighter than the Python original, whose noise derivative is itself an FD.
TEST(FilteredNoise, GradientConsistentWithModel) {
    const int nx = 64, ny = 64;
    const real_t Lx = 3.2, Ly = 1.0;
    FilteredNoise noise(nx, ny, Lx, Ly, /*amplitude=*/0.5, 0.3, 0.3, /*seed=*/11);
    const DeviceNoise dn = noise.device_noise();

    for (const Model model : {Model::Linear, Model::Arclength}) {
        Params p;
        p.physical_size = Lx;
        p.line_tension = 0.5;
        p.driving_force = 0.1;
        p.model = model;

        View1D h("h", nx);
        auto h_host = Kokkos::create_mirror_view(h);
        std::mt19937 rng(7);
        std::uniform_real_distribution<real_t> dist(0.2 * Ly, 0.8 * Ly);
        for (int i = 0; i < nx; ++i) h_host(i) = dist(rng);
        Kokkos::deep_copy(h, h_host);

        View1D grad("grad", nx);
        gradient(h, p, dn, grad);
        auto grad_host = Kokkos::create_mirror_view(grad);
        Kokkos::deep_copy(grad_host, grad);

        const real_t eps = 1e-6;
        real_t max_diff = 0;
        for (int j = 0; j < nx; ++j) {
            const real_t saved = h_host(j);
            h_host(j) = saved + eps;
            Kokkos::deep_copy(h, h_host);
            const real_t e_plus = objective(h, p, dn);
            h_host(j) = saved - eps;
            Kokkos::deep_copy(h, h_host);
            const real_t e_minus = objective(h, p, dn);
            h_host(j) = saved;
            Kokkos::deep_copy(h, h_host);
            max_diff = std::max(max_diff,
                                std::abs((e_plus - e_minus) / (2 * eps) - grad_host(j)));
        }
        EXPECT_LT(max_diff, 1e-5)
            << "model=" << (model == Model::Linear ? "linear" : "arclength");
    }
}

// The pinning field is periodic in BOTH directions, and sample() must wrap any
// query coordinate -- including negative ones and many periods out -- back into
// the fundamental cell. The depinning and dynamics runs drive h forward through
// many periods of Ly (and the noise x is the edge-midpoint, which the periodic
// h advances past Lx), so correct wrapping is load-bearing for the physics. No
// other test queries outside [0,Lx) x [0,Ly); this pins it directly.
TEST(FilteredNoise, PeriodicWrapAcrossManyCells) {
    const int nx = 48, ny = 40;
    const real_t Lx = 2.5, Ly = 1.3;
    FilteredNoise noise(nx, ny, Lx, Ly, 1.0, 0.25, 0.25, /*seed=*/17);
    const DeviceNoise dn = noise.device_noise();

    // Base query points strictly inside the cell.
    std::vector<real_t> xs, ys;
    std::mt19937 rng(123);
    std::uniform_real_distribution<real_t> ux(0, Lx), uy(0, Ly);
    const int m = 60;
    for (int k = 0; k < m; ++k) {
        xs.push_back(ux(rng));
        ys.push_back(uy(rng));
    }
    std::vector<real_t> v0, dx0, dy0;
    sample_points(dn, xs, ys, v0, dx0, dy0);

    // Same points shifted by integer numbers of periods (including negative and
    // large offsets) must reproduce value AND both derivatives bit-for-tight.
    const std::pair<int, int> shifts[] = {
        {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {3, -2}, {-5, 7}, {-13, -11}};
    for (const auto &sh : shifts) {
        std::vector<real_t> xss = xs, yss = ys;
        for (int k = 0; k < m; ++k) {
            xss[k] += sh.first * Lx;
            yss[k] += sh.second * Ly;
        }
        std::vector<real_t> v, dvx, dvy;
        sample_points(dn, xss, yss, v, dvx, dvy);
        double mv = 0, mdx = 0, mdy = 0;
        for (int k = 0; k < m; ++k) {
            mv = std::max(mv, std::abs(v[k] - v0[k]));
            mdx = std::max(mdx, std::abs(dvx[k] - dx0[k]));
            mdy = std::max(mdy, std::abs(dvy[k] - dy0[k]));
        }
        EXPECT_LT(mv, 1e-9) << "value not periodic for shift (" << sh.first
                            << "," << sh.second << ")";
        EXPECT_LT(mdx, 1e-9) << "dv/dx not periodic for shift (" << sh.first
                             << "," << sh.second << ")";
        EXPECT_LT(mdy, 1e-9) << "dv/dy not periodic for shift (" << sh.first
                             << "," << sh.second << ")";
    }
}

#ifdef QEW_WITH_HDF5
// Test 3c (golden): prefilter Python's noise_on_grid and check the interpolated
// value/derivatives against SciPy's cubic B-spline at the same query points.
TEST(FilteredNoise, MatchesScipyGolden) {
    HighFive::File file(QEW_GOLDEN_PATH, HighFive::File::ReadOnly);
    int nx = 0, ny = 0;
    double Lx = 0, Ly = 0;
    file.getAttribute("nx").read(nx);
    file.getAttribute("ny").read(ny);
    file.getAttribute("Lx").read(Lx);
    file.getAttribute("Ly").read(Ly);

    std::vector<std::vector<double>> field2d;
    file.getDataSet("field").read(field2d);  // shape (nx, ny)
    std::vector<real_t> field(static_cast<std::size_t>(nx) * ny);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            field[static_cast<std::size_t>(i) * ny + j] = field2d[i][j];

    std::vector<real_t> xs, ys, v_ref, dvdx_ref, dvdy_ref;
    file.getDataSet("x").read(xs);
    file.getDataSet("y").read(ys);
    file.getDataSet("v").read(v_ref);
    file.getDataSet("dvdx").read(dvdx_ref);
    file.getDataSet("dvdy").read(dvdy_ref);

    FilteredNoise noise(field.data(), nx, ny, Lx, Ly);
    std::vector<real_t> v, dvdx, dvdy;
    sample_points(noise.device_noise(), xs, ys, v, dvdx, dvdy);

    double max_v = 0, max_dx = 0, max_dy = 0;
    for (std::size_t k = 0; k < xs.size(); ++k) {
        max_v = std::max(max_v, std::abs(v[k] - v_ref[k]));
        max_dx = std::max(max_dx, std::abs(dvdx[k] - dvdx_ref[k]));
        max_dy = std::max(max_dy, std::abs(dvdy[k] - dvdy_ref[k]));
    }
    EXPECT_LT(max_v, 1e-7) << "value mismatch vs SciPy";
    // Python derivatives are themselves finite differences (step 1e-4 grid
    // units), so compare a little more loosely against our analytic ones.
    EXPECT_LT(max_dx, 1e-4) << "dv/dx mismatch vs SciPy";
    EXPECT_LT(max_dy, 1e-4) << "dv/dy mismatch vs SciPy";
}
#endif
