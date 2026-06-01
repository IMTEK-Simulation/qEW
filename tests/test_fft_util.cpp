#include "fft_util.h"

#include <gtest/gtest.h>

// Phase-1 smoke test: pocketfft forward/inverse round-trip recovers the input.
// Confirms the pocketfft dependency is fetched, included and links correctly.
TEST(FftUtil, RoundTripRecoversInput) {
    EXPECT_LT(qew::fft_roundtrip_error(64), 1e-12);
    EXPECT_LT(qew::fft_roundtrip_error(100), 1e-12);  // non-power-of-two
    EXPECT_LT(qew::fft_roundtrip_error(1024), 1e-12);
}
