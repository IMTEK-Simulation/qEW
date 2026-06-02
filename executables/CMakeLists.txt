add_executable(main main.cpp)
target_link_libraries(main PRIVATE lib)

# Production executables that write HDF5 output need HighFive.
if(QEW_WITH_HDF5)
    add_subdirectory(qew_static)
    add_subdirectory(qew_dynamic)
    add_subdirectory(qew_depinning)
endif()
