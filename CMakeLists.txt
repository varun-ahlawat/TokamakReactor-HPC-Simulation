cmake_minimum_required(VERSION 3.16)
project(TokamakDisruptionSim
    VERSION 1.0.0
    DESCRIPTION "High-Performance Tokamak Plasma Disruption Simulation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Optional HDF5
find_package(HDF5 COMPONENTS CXX)

# Core library sources
set(CORE_SOURCES
    src/core/grid.cpp
    src/core/field.cpp
    src/core/time_integrator.cpp
)

# Physics sources
set(PHYSICS_SOURCES
    src/physics/mhd_solver.cpp
    src/physics/thermal_quench.cpp
    src/physics/current_quench.cpp
    src/physics/runaway_electrons.cpp
    src/physics/disruption_mitigation.cpp
)

# Coupling sources
set(COUPLING_SOURCES
    src/coupling/multi_physics_coupler.cpp
)

# I/O sources
set(IO_SOURCES
    src/io/diagnostics.cpp
    src/io/data_writer.cpp
)

# Utility sources
set(UTILS_SOURCES
    src/utils/physical_constants.cpp
)

# Main simulation library
add_library(tokamak_lib STATIC
    ${CORE_SOURCES}
    ${PHYSICS_SOURCES}
    ${COUPLING_SOURCES}
    ${IO_SOURCES}
    ${UTILS_SOURCES}
)

target_include_directories(tokamak_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(tokamak_lib PUBLIC
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

if(HDF5_FOUND)
    target_compile_definitions(tokamak_lib PUBLIC HAS_HDF5)
    target_link_libraries(tokamak_lib PUBLIC ${HDF5_CXX_LIBRARIES})
    target_include_directories(tokamak_lib PUBLIC ${HDF5_INCLUDE_DIRS})
endif()

# Main executable
add_executable(tokamak_sim src/main.cpp)
target_link_libraries(tokamak_sim PRIVATE tokamak_lib)

# Field dumper for visualization
add_executable(dump_fields src/dump_fields.cpp)
target_link_libraries(dump_fields PRIVATE tokamak_lib)

# Tests
enable_testing()
add_executable(run_tests tests/test_main.cpp)
target_link_libraries(run_tests PRIVATE tokamak_lib)
add_test(NAME TokamakTests COMMAND run_tests)

# Physics validation tests (litmus test for physical correctness)
add_executable(run_validation_tests tests/test_physics_validation.cpp)
target_link_libraries(run_validation_tests PRIVATE tokamak_lib)
add_test(NAME PhysicsValidation COMMAND run_validation_tests)
