#include "filtered_noise.h"

#include <cmath>
#include <cstddef>
#include <random>

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

namespace qew {

namespace {

using cplx = Kokkos::complex<real_t>;
using CView2D = Kokkos::View<cplx **, MemSpace>;

// numpy fftfreq(n)[k] in cycles per sample: k/n for k < (n+1)/2, else (k-n)/n.
inline real_t fftfreq(int k, int n) {
    const int ks = (k < (n + 1) / 2) ? k : k - n;
    return static_cast<real_t>(ks) / static_cast<real_t>(n);
}

}  // namespace

FilteredNoise::FilteredNoise(int nx, int ny, real_t Lx, real_t Ly,
                             real_t amplitude, real_t xi_x, real_t xi_y,
                             std::uint64_t seed)
    : nx_(nx), ny_(ny), Lx_(Lx), Ly_(Ly),
      dx_(Lx / nx), dy_(Ly / ny), field_(static_cast<std::size_t>(nx) * ny) {
    const real_t two_pi = static_cast<real_t>(2) * std::acos(static_cast<real_t>(-1));

    // Random unit-amplitude phases, masked to the (possibly anisotropic)
    // correlation ellipse (fx*xi_x)^2 + (fy*xi_y)^2 <= 1. Built on the host (RNG)
    // then uploaded.
    CView2D A("noise_A", nx, ny);
    auto Ah = Kokkos::create_mirror_view(A);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<real_t> phase(0, two_pi);
    for (int i = 0; i < nx; ++i) {
        const real_t fx = fftfreq(i, nx) / dx_;
        for (int j = 0; j < ny; ++j) {
            const real_t fy = fftfreq(j, ny) / dy_;
            const real_t ph = phase(rng);
            if ((fx * xi_x) * (fx * xi_x) + (fy * xi_y) * (fy * xi_y) >
                static_cast<real_t>(1)) {
                Ah(i, j) = cplx(0, 0);
            } else {
                Ah(i, j) = cplx(std::cos(ph), std::sin(ph));
            }
        }
    }
    Kokkos::deep_copy(A, Ah);

    // Inverse FFT -> real-space field (kokkos-fft backward norm == numpy
    // ifft2 normalisation 1/(nx*ny)).
    CView2D B("noise_B", nx, ny);
    ExecSpace exec;
    KokkosFFT::ifft2(exec, A, B);
    exec.fence();

    // Real part, then scale to the requested amplitude (population std; the mean
    // / DC term is left in place, matching the Python version).
    auto Bh = Kokkos::create_mirror_view(B);
    Kokkos::deep_copy(Bh, B);
    const std::size_t n = static_cast<std::size_t>(nx) * ny;
    real_t mean = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) mean += Bh(i, j).real();
    mean /= static_cast<real_t>(n);
    real_t var = 0;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j) {
            const real_t d = Bh(i, j).real() - mean;
            var += d * d;
        }
    const real_t scale = amplitude / std::sqrt(var / static_cast<real_t>(n));
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            field_[static_cast<std::size_t>(i) * ny + j] = Bh(i, j).real() * scale;

    build_coefficients();
}

FilteredNoise::FilteredNoise(const real_t *field, int nx, int ny, real_t Lx,
                             real_t Ly)
    : nx_(nx), ny_(ny), Lx_(Lx), Ly_(Ly),
      dx_(Lx / nx), dy_(Ly / ny),
      field_(field, field + static_cast<std::size_t>(nx) * ny) {
    build_coefficients();
}

void FilteredNoise::build_coefficients() {
    // Periodic cubic B-spline prefilter, done in Fourier space:
    //   c = IFFT2( FFT2(field) / (Kx_hat ⊗ Ky_hat) ),  K_hat[m] = 2/3 + (1/3)cos(2πm/N),
    // the exact solve of the circular interpolation system (1/6,2/3,1/6)*c = s.
    ExecSpace exec;
    const int nx = nx_, ny = ny_;
    const real_t two_pi = static_cast<real_t>(2) * std::acos(static_cast<real_t>(-1));

    CView2D S("prefilt_S", nx, ny);
    auto Sh = Kokkos::create_mirror_view(S);
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            Sh(i, j) = cplx(field_[static_cast<std::size_t>(i) * ny + j], 0);
    Kokkos::deep_copy(S, Sh);

    CView2D Shat("prefilt_Shat", nx, ny);
    KokkosFFT::fft2(exec, S, Shat);  // forward (unnormalized)

    Kokkos::parallel_for(
        "prefilt_divide",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(const int i, const int j) {
            const real_t kx = static_cast<real_t>(2) / 3 +
                              Kokkos::cos(two_pi * i / nx) / static_cast<real_t>(3);
            const real_t ky = static_cast<real_t>(2) / 3 +
                              Kokkos::cos(two_pi * j / ny) / static_cast<real_t>(3);
            Shat(i, j) = Shat(i, j) / (kx * ky);
        });

    CView2D Sc("prefilt_Sc", nx, ny);
    KokkosFFT::ifft2(exec, Shat, Sc);  // inverse (1/(nx*ny))

    View2D coef("noise_coef", nx, ny);
    Kokkos::parallel_for(
        "prefilt_real",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nx, ny}),
        KOKKOS_LAMBDA(const int i, const int j) { coef(i, j) = Sc(i, j).real(); });
    exec.fence();

    device_noise_.coef = coef;
    device_noise_.nx = nx_;
    device_noise_.ny = ny_;
    device_noise_.dx = dx_;
    device_noise_.dy = dy_;
}

}  // namespace qew
