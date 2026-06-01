#ifndef QEW_FILTERED_NOISE_H
#define QEW_FILTERED_NOISE_H

#include <cstdint>
#include <vector>

#include <Kokkos_Core.hpp>

#include "qew_types.h"

// Correlated random pinning field, ported from FilteredNoise in
// ../../Overleaf/qEW/qew_core.py.
//
// Setup (host): a band-limited periodic field is built by masking the Fourier
// modes of a random-phase spectrum (inverse FFT, pocketfft) and normalised to a
// target amplitude. The field is then converted to periodic cubic B-spline
// coefficients (an exact FFT-based prefilter) and uploaded to the device.
//
// Evaluation (device): periodic bicubic B-spline interpolation returns the
// value and BOTH partial derivatives analytically -- unlike the Python version,
// which finite-differences the interpolant. The analytic derivative is the
// exact gradient of the interpolated value, so it plugs into qew_model's
// objective()/gradient() and keeps them consistent to machine precision.
//
// Boundary handling: we use the clean periodic prefilter (period N, equivalent
// to SciPy's mode='grid-wrap'), which is the correct choice for an FFT-periodic
// field. The Python production code passed mode='wrap' (period N-1 overlap); the
// two differ only negligibly for the smooth band-limited fields used here.
namespace qew {

// Cubic B-spline interpolation weights w[0..3] for the four neighbours
// {i-1, i, i+1, i+2} at fractional position t in [0,1), and their derivatives
// dw[0..3] with respect to t (i.e. d/d(grid coordinate)).
KOKKOS_INLINE_FUNCTION
void bspline_weights(real_t t, real_t w[4], real_t dw[4]) {
    const real_t t2 = t * t;
    const real_t t3 = t2 * t;
    const real_t omt = static_cast<real_t>(1) - t;

    w[0] = omt * omt * omt / static_cast<real_t>(6);
    w[1] = (static_cast<real_t>(4) - static_cast<real_t>(6) * t2 +
            static_cast<real_t>(3) * t3) / static_cast<real_t>(6);
    w[2] = (static_cast<real_t>(1) + static_cast<real_t>(3) * t +
            static_cast<real_t>(3) * t2 - static_cast<real_t>(3) * t3) /
           static_cast<real_t>(6);
    w[3] = t3 / static_cast<real_t>(6);

    dw[0] = -omt * omt / static_cast<real_t>(2);
    dw[1] = (-static_cast<real_t>(12) * t + static_cast<real_t>(9) * t2) /
            static_cast<real_t>(6);
    dw[2] = (static_cast<real_t>(3) + static_cast<real_t>(6) * t -
             static_cast<real_t>(9) * t2) / static_cast<real_t>(6);
    dw[3] = static_cast<real_t>(3) * t2 / static_cast<real_t>(6);
}

// Device-callable view of the prefiltered field. Copyable by value into Kokkos
// kernels (the coefficient View is a reference-counted device handle), so it
// plays the `Noise` role expected by qew_model.
struct DeviceNoise {
    View2D coef;            // cubic B-spline coefficients, shape (nx, ny)
    int nx = 0;
    int ny = 0;
    real_t dx = 1;          // physical grid spacing in x
    real_t dy = 1;          // physical grid spacing in y

    KOKKOS_INLINE_FUNCTION
    NoiseSample sample(real_t x, real_t y) const {
        // Physical -> grid coordinates, wrapped into [0, n) (positive modulo).
        real_t gx = x / dx;
        real_t gy = y / dy;
        gx -= Kokkos::floor(gx / static_cast<real_t>(nx)) * static_cast<real_t>(nx);
        gy -= Kokkos::floor(gy / static_cast<real_t>(ny)) * static_cast<real_t>(ny);

        const int ix = static_cast<int>(Kokkos::floor(gx));
        const int iy = static_cast<int>(Kokkos::floor(gy));
        const real_t fx = gx - static_cast<real_t>(ix);
        const real_t fy = gy - static_cast<real_t>(iy);

        const int ixn[4] = {(ix - 1 + nx) % nx, ix % nx, (ix + 1) % nx, (ix + 2) % nx};
        const int iyn[4] = {(iy - 1 + ny) % ny, iy % ny, (iy + 1) % ny, (iy + 2) % ny};

        real_t wx[4], dwx[4], wy[4], dwy[4];
        bspline_weights(fx, wx, dwx);
        bspline_weights(fy, wy, dwy);

        real_t v = 0;
        real_t dv_dgx = 0;
        real_t dv_dgy = 0;
        for (int a = 0; a < 4; ++a) {
            real_t row = 0;       // sum_b c * wy[b]
            real_t row_dy = 0;    // sum_b c * dwy[b]
            for (int b = 0; b < 4; ++b) {
                const real_t c = coef(ixn[a], iyn[b]);
                row += c * wy[b];
                row_dy += c * dwy[b];
            }
            v += wx[a] * row;
            dv_dgx += dwx[a] * row;
            dv_dgy += wx[a] * row_dy;
        }

        NoiseSample s;
        s.v = v;
        s.dv_dx = dv_dgx / dx;  // grid -> physical derivative
        s.dv_dy = dv_dgy / dy;
        return s;
    }
};

class FilteredNoise {
public:
    // Generate a band-limited random field. `xi_x`, `xi_y` are the correlation
    // lengths (pass the same value twice for the isotropic case).
    FilteredNoise(int nx, int ny, real_t Lx, real_t Ly, real_t amplitude,
                  real_t xi_x, real_t xi_y, std::uint64_t seed);

    // Build from an existing real-space field (row-major, length nx*ny). Used by
    // golden-file tests to prefilter Python's noise_on_grid directly, decoupling
    // the interpolation test from RNG differences.
    FilteredNoise(const real_t *field, int nx, int ny, real_t Lx, real_t Ly);

    DeviceNoise device_noise() const { return device_noise_; }

    const std::vector<real_t> &field() const { return field_; }  // noise_on_grid
    int nx() const { return nx_; }
    int ny() const { return ny_; }
    real_t dx() const { return dx_; }
    real_t dy() const { return dy_; }

private:
    void build_coefficients();  // FFT prefilter field_ -> upload to device

    int nx_;
    int ny_;
    real_t Lx_;
    real_t Ly_;
    real_t dx_;
    real_t dy_;
    std::vector<real_t> field_;  // host noise_on_grid (row-major nx*ny)
    DeviceNoise device_noise_;
};

}  // namespace qew

#endif  // QEW_FILTERED_NOISE_H
