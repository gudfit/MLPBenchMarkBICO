cmake_minimum_required(VERSION 3.18)
project(MLPBenchMarkBICO LANGUAGES CXX CUDA)
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CMAKE_CUDA_ARCHITECTURES 89)  # Single-arch for Ada (SM 8.9)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Generate dispatcher header
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/Evaluator_dispatcher.h
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_dispatcher.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_dispatcher.py
    COMMENT "Generating CUDA kernel dispatcher..."
)

add_custom_target(GenerateDispatcher ALL DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Evaluator_dispatcher.h)

add_executable(mlp_benchmark
    main.cu
    Evaluator.cu
    BICOExplorer.cu
)
target_sources(mlp_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Evaluator_dispatcher.h)
add_dependencies(mlp_benchmark GenerateDispatcher)

target_link_libraries(mlp_benchmark PRIVATE CUDA::cudart)
set_property(TARGET mlp_benchmark PROPERTY CXX_STANDARD 17)
set_property(TARGET mlp_benchmark PROPERTY CUDA_STANDARD 17)
target_compile_options(mlp_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        --generate-line-info
    >
)
target_include_directories(mlp_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
message(STATUS "CUDA architecture set to: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "")
message(STATUS "Build instructions:")
message(STATUS " mkdir -p build && cd build")
message(STATUS " cmake ..")
message(STATUS " make -j")
message(STATUS "")
message(STATUS "Run with:")
message(STATUS " ./mlp_benchmark")
