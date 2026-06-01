#ifndef QEW_KOKKOS_SMOKETEST_H
#define QEW_KOKKOS_SMOKETEST_H

namespace qew {

// Sum 0..n-1 via a Kokkos parallel_reduce on the default execution space.
// Phase-1 smoke test: proves the Kokkos toolchain links and a kernel runs
// through the static library (and, on a GPU build, on the device).
double parallel_sum(int n);

}  // namespace qew

#endif  // QEW_KOKKOS_SMOKETEST_H
