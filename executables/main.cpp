#include "hello.h"
#include <iostream>
#include <filesystem>
#include <mpi.h>
#include <Kokkos_Core.hpp>


int main(int argc, char *argv[]) {
    int rank = 0, size = 1;

    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);

    // Retrieve process infos
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello I am rank " << rank << " of " << size << "\n";

    if (rank == 0)
      hello_world();

    auto input_path = "./simulation_test_input.txt";

    if (not std::filesystem::exists(input_path))
      std::cerr << "warning: could not find input file " << input_path << "\n";

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
