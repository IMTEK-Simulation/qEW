#include "kokkos_smoketest.h"

#include "qew_types.h"

namespace qew {

double parallel_sum(int n) {
    double total = 0.0;
    Kokkos::parallel_reduce(
        "qew::parallel_sum", n,
        KOKKOS_LAMBDA(int i, double &acc) { acc += static_cast<double>(i); },
        total);
    return total;
}

}  // namespace qew
