set(TARGET opencog-simple)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE ggml common-ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
