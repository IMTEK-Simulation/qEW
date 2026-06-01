#include "fft_util.h"

#include <algorithm>
#include <complex>
#include <vector>

#include "pocketfft_hdronly.h"

namespace qew {

double fft_roundtrip_error(std::size_t n) {
    using cpx = std::complex<double>;

    std::vector<cpx> in(n), fwd(n), back(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Arbitrary deterministic signal.
        in[i] = cpx(static_cast<double>(i % 7) - 3.0,
                    static_cast<double>((2 * i) % 5) - 2.0);
    }

    const pocketfft::shape_t shape{n};
    const pocketfft::stride_t stride{static_cast<std::ptrdiff_t>(sizeof(cpx))};
    const pocketfft::shape_t axes{0};

    pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD, in.data(),
                   fwd.data(), 1.0);
    pocketfft::c2c(shape, stride, stride, axes, pocketfft::BACKWARD, fwd.data(),
                   back.data(), 1.0 / static_cast<double>(n));

    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::abs(back[i] - in[i]));
    }
    return max_err;
}

}  // namespace qew
