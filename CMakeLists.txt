cmake_minimum_required(VERSION 3.18)
project(glu_mlp_bico LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(mlp_benchmark main.cu)

target_link_libraries(mlp_benchmark PRIVATE CUDA::cudart)
set_property(TARGET mlp_benchmark PROPERTY CXX_STANDARD 17)
set_property(TARGET mlp_benchmark PROPERTY CUDA_STANDARD 17)

target_compile_options(mlp_benchmark PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

message(STATUS "Set CUDA architecture to: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "To build, run: rm -rf build && mkdir build && cd build && cmake .. && make")
message(STATUS "To run, execute: ./mlp_benchmark")
