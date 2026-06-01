#ifndef QEW_FFT_UTIL_H
#define QEW_FFT_UTIL_H

#include <cstddef>

namespace qew {

// Phase-1 smoke test for pocketfft: forward then inverse complex-to-complex
// 1-D FFT of a fixed signal, returning the maximum absolute round-trip error.
// A correctly linked pocketfft returns a value near machine epsilon. The real
// noise-field FFT lands in src/filtered_noise.cpp in Phase 3.
double fft_roundtrip_error(std::size_t n);

}  // namespace qew

#endif  // QEW_FFT_UTIL_H
