#include "kokkos_smoketest.h"

#include <gtest/gtest.h>

// Phase-1 smoke test: a Kokkos parallel_reduce through the static library.
// On a GPU build this runs on the device, confirming the toolchain end to end.
TEST(KokkosSmoke, ParallelSum) {
    // sum_{i=0}^{n-1} i = n(n-1)/2
    EXPECT_DOUBLE_EQ(qew::parallel_sum(100), 4950.0);
    EXPECT_DOUBLE_EQ(qew::parallel_sum(1), 0.0);
    EXPECT_DOUBLE_EQ(qew::parallel_sum(0), 0.0);
}
