add_library(lib STATIC
    hello.cpp
    kokkos_smoketest.cpp
    fft_util.cpp
    filtered_noise.cpp
)
target_include_directories(lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lib PUBLIC Kokkos::kokkos KokkosFFT::fft)
