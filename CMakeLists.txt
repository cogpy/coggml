cmake_minimum_required(VERSION 3.5)
project(cogpy-hpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library: INTERFACE target
add_library(cog INTERFACE)
target_include_directories(cog INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Unified test suite (all modules)
add_executable(test_all test/test_all.cpp)
target_link_libraries(test_all PRIVATE cog)

# Ternary quantization test
add_executable(test_tq_log2_3 test/test_tq_log2_3.cpp)
target_link_libraries(test_tq_log2_3 PRIVATE cog)

# Multi-model inference engine test
add_executable(test_multi_model test/test_multi_model.cpp)
target_link_libraries(test_multi_model PRIVATE cog)

# Extended coverage test suite
add_executable(test_coverage test/test_coverage.cpp)
target_link_libraries(test_coverage PRIVATE cog)

enable_testing()
add_test(NAME cogpy_tests       COMMAND test_all)
add_test(NAME tq_log2_3_tests   COMMAND test_tq_log2_3)
add_test(NAME multi_model_tests COMMAND test_multi_model)
add_test(NAME coverage_tests    COMMAND test_coverage)

# CogPy demo (all 14 modules)
add_subdirectory(examples/cogpy)

# Install headers
install(DIRECTORY include/cog DESTINATION include)
