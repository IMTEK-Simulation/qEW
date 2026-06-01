cmake_minimum_required(VERSION 3.14)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()
project(my-fantastic-simulation-code LANGUAGES C CXX VERSION 0.1)

# Kokkos 5.x requires C++20 or newer.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Dependency: Kokkos (on-node parallelism / GPU). No MPI, no Eigen3 — the qEW
# line model needs only element-wise stencils, reductions and RNG, all of which
# map directly onto Kokkos Views and parallel patterns.
find_package(Kokkos 5.1.1)
if(NOT Kokkos_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      kokkos
      URL https://github.com/kokkos/kokkos/archive/refs/tags/5.1.1.tar.gz
    )
    set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "Disable Kokkos tests" FORCE)
    set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Kokkos examples" FORCE)
    FetchContent_MakeAvailable(kokkos)
endif()

# pocketfft: header-only C++ FFT (the same code NumPy/SciPy vendor). Host-side
# only — used once at setup to build the correlated noise field, never in the
# hot loop, so no GPU FFT is required. The 'cpp' branch ships a single header
# (pocketfft_hdr.h) and no CMakeLists, so we wrap it in an INTERFACE target.
include(FetchContent)
FetchContent_Declare(
  pocketfft
  GIT_REPOSITORY https://gitlab.mpcdf.mpg.de/mtr/pocketfft.git
  GIT_TAG cpp
)
FetchContent_MakeAvailable(pocketfft)
if(NOT TARGET pocketfft)
    add_library(pocketfft INTERFACE)
    target_include_directories(pocketfft INTERFACE ${pocketfft_SOURCE_DIR})
    # The one-time setup FFT does not need pocketfft's threading layer; disabling
    # it avoids a pthread link dependency.
    target_compile_definitions(pocketfft INTERFACE POCKETFFT_NO_MULTITHREADING)
endif()

# HDF5 I/O (optional): needed to read/write .h5 matching the Python key strings
# and to load golden-file test fixtures. Guarded so the project still configures
# and `ctest` passes on machines without HDF5. Analysis itself stays in Python.
find_package(HDF5 QUIET COMPONENTS C)
if(HDF5_FOUND)
    FetchContent_Declare(
      HighFive
      GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
      GIT_TAG v2.10.0
    )
    set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    # HighFive 2.x declares cmake_minimum_required < 3.5, which CMake 4 rejects
    # outright. The policy-min override (scoped to this fetch) tells CMake to
    # treat it as requesting 3.5. Drop this once a stable HighFive 3.x is tagged.
    set(_qew_saved_policy_min "${CMAKE_POLICY_VERSION_MINIMUM}")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_MakeAvailable(HighFive)
    set(CMAKE_POLICY_VERSION_MINIMUM "${_qew_saved_policy_min}")
    set(QEW_WITH_HDF5 ON)
    message(STATUS "HDF5 found: HighFive I/O enabled")
else()
    set(QEW_WITH_HDF5 OFF)
    message(STATUS "HDF5 not found: HighFive I/O disabled (analysis is done in Python)")
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(executables)
add_subdirectory(tests)
