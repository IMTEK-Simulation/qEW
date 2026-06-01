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

set(TEST_SOURCES test_hello_world.cpp)

list(APPEND TEST_SOURCES gtest_mpi_main.cpp)

add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests PRIVATE lib GTest::gtest)
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})


include(GoogleTest)
gtest_discover_tests(tests)
