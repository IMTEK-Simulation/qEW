// GoogleTest entry point that brackets the test run with Kokkos
// initialize/finalize, so tests may launch kernels on the default execution
// space (Serial/OpenMP on CPU, or CUDA/HIP on GPU builds). No MPI.

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int result = RUN_ALL_TESTS();
    Kokkos::finalize();
    return result;
}
