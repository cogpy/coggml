set(TARGET opencog-simple)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE ggml common-ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET opencog-reasoning)
add_executable(${TARGET} reasoning_example.cpp)
target_link_libraries(${TARGET} PRIVATE ggml common-ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET opencog-advanced)
add_executable(${TARGET} advanced_features.cpp)
target_link_libraries(${TARGET} PRIVATE ggml common-ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET opencog-hebbian)
add_executable(${TARGET} hebbian_learning.cpp)
target_link_libraries(${TARGET} PRIVATE ggml common-ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
