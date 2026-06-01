#include "hello.h"
#include <iostream>
#include <Kokkos_Core.hpp>


int main(int argc, char *argv[]) {
    Kokkos::initialize(argc, argv);

    {
        std::cout << "qEW skeleton\n";
        Kokkos::print_configuration(std::cout);
        hello_world();
    }

    Kokkos::finalize();

    return 0;
}
