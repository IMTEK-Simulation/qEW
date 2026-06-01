#ifndef QEW_TYPES_H
#define QEW_TYPES_H

#include <Kokkos_Core.hpp>

// Shared type aliases for the qEW port. The line state h(x) is a 1-D array; the
// noise grid is 2-D. Views live in the default execution space's memory, so a
// CUDA/HIP build keeps them on the GPU and a CPU build keeps them in host RAM.
namespace qew {

using real_t = double;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;

// Line state h(x) and per-site fields.
using View1D = Kokkos::View<real_t *, MemSpace>;
using HostView1D = View1D::host_mirror_type;

// Noise grid and spline coefficients.
using View2D = Kokkos::View<real_t **, MemSpace>;
using HostView2D = View2D::host_mirror_type;

// Value and partial derivatives of the pinning potential at a point. Returned by
// any device-callable noise functor (the analytic test stub, or FilteredNoise).
struct NoiseSample {
    real_t v;      // V(x, y)
    real_t dv_dx;  // dV/dx
    real_t dv_dy;  // dV/dy
};

}  // namespace qew

#endif  // QEW_TYPES_H
