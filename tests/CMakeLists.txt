find_package(GTest)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

set(TEST_SOURCES
    test_hello_world.cpp
    test_kokkos_smoke.cpp
    test_fft_util.cpp
    test_qew_model.cpp
    test_filtered_noise.cpp
    test_fire.cpp
    test_dynamics.cpp
    test_lbfgs.cpp
    test_rosso_krauth.cpp
    test_executables.cpp
)

# Kokkos-aware GoogleTest entry point (initializes/finalizes Kokkos around the
# test run). No MPI. gtest_mpi_main.cpp / gtest-mpi-listener.hpp are unused and
# may be deleted.
list(APPEND TEST_SOURCES gtest_kokkos_main.cpp)

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE lib GTest::gtest)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Golden-file cross-check against SciPy (test_filtered_noise.cpp) needs HDF5.
if(QEW_WITH_HDF5)
    target_compile_definitions(tests PRIVATE
        QEW_WITH_HDF5
        QEW_GOLDEN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/noise_golden.h5"
        QEW_LBFGS_GOLDEN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/lbfgs_golden.h5")
    target_link_libraries(tests PRIVATE HighFive)
endif()

include(GoogleTest)
# Enumerate tests at `ctest` time (PRE_TEST), not at build time. The test binary
# calls Kokkos::initialize(), so on a GPU build the default POST_BUILD discovery
# would have to start the device *during the build* -- which fails on a GPU-less
# login node (and can time out on slow first HIP init), surfacing as the
# "tests_NOT_BUILT" sentinel. PRE_TEST defers listing to when you run ctest on a
# GPU node. The longer timeout covers slow device initialization.
gtest_discover_tests(tests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 120)
