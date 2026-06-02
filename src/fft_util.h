#ifndef QEW_FFT_UTIL_H
#define QEW_FFT_UTIL_H

#include <cstddef>

namespace qew {

// Smoke test for kokkos-fft: forward then inverse complex FFT of a fixed signal
// on the default execution space, returning the maximum absolute round-trip
// error. A correctly linked kokkos-fft (FFTW on host, hipFFT/cuFFT on GPU)
// returns a value near machine epsilon.
double fft_roundtrip_error(std::size_t n);

}  // namespace qew

#endif  // QEW_FFT_UTIL_H
