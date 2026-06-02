#include "fft_util.h"

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

#include "qew_types.h"

namespace qew {

double fft_roundtrip_error(std::size_t n) {
    using cview = Kokkos::View<Kokkos::complex<real_t> *, MemSpace>;
    const int m = static_cast<int>(n);

    cview x("x", n), xhat("xhat", n), xinv("xinv", n);
    auto xh = Kokkos::create_mirror_view(x);
    for (int i = 0; i < m; ++i) {
        xh(i) = Kokkos::complex<real_t>(static_cast<real_t>(i % 7) - 3,
                                        static_cast<real_t>((2 * i) % 5) - 2);
    }
    Kokkos::deep_copy(x, xh);

    ExecSpace exec;
    KokkosFFT::fft(exec, x, xhat);    // forward (unnormalized)
    KokkosFFT::ifft(exec, xhat, xinv);  // inverse (1/n) -> round-trip identity
    exec.fence();

    real_t max_err = 0;
    Kokkos::parallel_reduce(
        "fft_roundtrip", m,
        KOKKOS_LAMBDA(const int i, real_t &acc) {
            const Kokkos::complex<real_t> d = xinv(i) - x(i);
            const real_t a = Kokkos::sqrt(d.real() * d.real() + d.imag() * d.imag());
            if (a > acc) acc = a;
        },
        Kokkos::Max<real_t>(max_err));
    return max_err;
}

}  // namespace qew
