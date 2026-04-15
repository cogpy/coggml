# examples/cogpy/CMakeLists.txt
# CogPy header-only library demo
#
# Build:
#   cd /path/to/coggml && mkdir build && cd build && cmake .. && make cogpy_demo

cmake_minimum_required(VERSION 3.5)

add_executable(cogpy_demo cogpy_demo.cpp)
target_link_libraries(cogpy_demo PRIVATE cog)
