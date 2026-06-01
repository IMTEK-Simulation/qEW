#include "filtered_noise.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>

#include "pocketfft_hdronly.h"

namespace qew {

namespace {

using cpx = std::complex<real_t>;

// numpy fftfreq(n)[k] in cycles per sample: k/n for k < (n+1)/2, else (k-n)/n.
inline real_t fftfreq(int k, int n) {
    const int ks = (k < (n + 1) / 2) ? k : k - n;
    return static_cast<real_t>(ks) / static_cast<real_t>(n);
}

// 2-D complex FFT, in place. forward=true matches numpy fft (exp(-i...));
// forward=false matches numpy ifft (exp(+i...)). `fct` is the scaling applied
// to every output element.
void fft2_inplace(std::vector<cpx> &a, int nx, int ny, bool forward, real_t fct) {
    const pocketfft::shape_t shape{static_cast<std::size_t>(nx),
                                   static_cast<std::size_t>(ny)};
    const pocketfft::stride_t stride{
        static_cast<std::ptrdiff_t>(ny * sizeof(cpx)),
        static_cast<std::ptrdiff_t>(sizeof(cpx))};
    const pocketfft::shape_t axes{0, 1};
    pocketfft::c2c(shape, stride, stride, axes, forward, a.data(), a.data(), fct);
}

}  // namespace

FilteredNoise::FilteredNoise(int nx, int ny, real_t Lx, real_t Ly,
                             real_t amplitude, real_t xi_x, real_t xi_y,
                             std::uint64_t seed)
    : nx_(nx), ny_(ny), Lx_(Lx), Ly_(Ly),
      dx_(Lx / nx), dy_(Ly / ny), field_(static_cast<std::size_t>(nx) * ny) {
    const real_t two_pi = static_cast<real_t>(2) * std::acos(static_cast<real_t>(-1));

    // Random unit-amplitude phases, then keep only modes inside the (possibly
    // anisotropic) correlation ellipse (fx*xi_x)^2 + (fy*xi_y)^2 <= 1.
    std::vector<cpx> a(static_cast<std::size_t>(nx) * ny);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<real_t> phase(static_cast<real_t>(0), two_pi);
    for (int i = 0; i < nx; ++i) {
        const real_t fx = fftfreq(i, nx) / dx_;
        for (int j = 0; j < ny; ++j) {
            const real_t fy = fftfreq(j, ny) / dy_;
            const real_t ph = phase(rng);  // draw for every mode (C-order)
            const std::size_t idx = static_cast<std::size_t>(i) * ny + j;
            if ((fx * xi_x) * (fx * xi_x) + (fy * xi_y) * (fy * xi_y) >
                static_cast<real_t>(1)) {
                a[idx] = cpx(0, 0);
            } else {
                a[idx] = std::exp(cpx(0, ph));
            }
        }
    }

    // Inverse FFT -> real-space field (numpy ifft2 normalisation 1/(nx*ny)).
    fft2_inplace(a, nx, ny, /*forward=*/false,
                 static_cast<real_t>(1) / (static_cast<real_t>(nx) * ny));

    // Real part, then scale to the requested amplitude (population std, matching
    // numpy np.std; the mean / DC term is left in place exactly as in Python).
    const std::size_t n = static_cast<std::size_t>(nx) * ny;
    real_t mean = 0;
    for (std::size_t k = 0; k < n; ++k) {
        field_[k] = a[k].real();
        mean += field_[k];
    }
    mean /= static_cast<real_t>(n);
    real_t var = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const real_t d = field_[k] - mean;
        var += d * d;
    }
    const real_t std_dev = std::sqrt(var / static_cast<real_t>(n));
    const real_t scale = amplitude / std_dev;
    for (std::size_t k = 0; k < n; ++k) field_[k] *= scale;

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
    // Periodic cubic B-spline prefilter. The interpolation condition is the
    // circular convolution  (1/6, 2/3, 1/6) * c = s  on each axis, which in
    // Fourier space is a division by the (real, separable) transfer function
    //   K_hat[m] = 2/3 + (1/3) cos(2 pi m / N).
    // So c = IFFT2( FFT2(s) / (Kx_hat ⊗ Ky_hat) ), exact to machine precision.
    const std::size_t n = static_cast<std::size_t>(nx_) * ny_;
    std::vector<cpx> s(n);
    for (std::size_t k = 0; k < n; ++k) s[k] = cpx(field_[k], 0);

    fft2_inplace(s, nx_, ny_, /*forward=*/true, static_cast<real_t>(1));

    const real_t two_pi = static_cast<real_t>(2) * std::acos(static_cast<real_t>(-1));
    std::vector<real_t> kx(nx_), ky(ny_);
    for (int i = 0; i < nx_; ++i) {
        kx[i] = static_cast<real_t>(2) / 3 +
                std::cos(two_pi * i / nx_) / static_cast<real_t>(3);
    }
    for (int j = 0; j < ny_; ++j) {
        ky[j] = static_cast<real_t>(2) / 3 +
                std::cos(two_pi * j / ny_) / static_cast<real_t>(3);
    }
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            s[static_cast<std::size_t>(i) * ny_ + j] /= (kx[i] * ky[j]);
        }
    }

    fft2_inplace(s, nx_, ny_, /*forward=*/false,
                 static_cast<real_t>(1) / static_cast<real_t>(n));

    View2D coef("noise_coef", nx_, ny_);
    auto h = Kokkos::create_mirror_view(coef);
    for (int i = 0; i < nx_; ++i) {
        for (int j = 0; j < ny_; ++j) {
            h(i, j) = s[static_cast<std::size_t>(i) * ny_ + j].real();
        }
    }
    Kokkos::deep_copy(coef, h);

    device_noise_.coef = coef;
    device_noise_.nx = nx_;
    device_noise_.ny = ny_;
    device_noise_.dx = dx_;
    device_noise_.dy = dy_;
}

}  // namespace qew
