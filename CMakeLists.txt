cmake_minimum_required(VERSION 3.18)
project(glu_mlp_bico LANGUAGES CXX CUDA)

# Set policy to avoid warnings about CUDA_ARCHITECTURES
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA architecture (RTX 5090 would likely be SM90 or similar)
# For now, let's use a more recent architecture
set(CMAKE_CUDA_ARCHITECTURES 90)

# Enable extended constexpr support for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Enable fast math for better performance (trade precision for speed)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

# Add executable
add_executable(mlp_benchmark
    main.cu
    bico_config.cpp
    glu_kernels.h
    bico_config.h
)

# Link against CUDA runtime
target_link_libraries(mlp_benchmark PRIVATE CUDA::cudart)

# Set C++ and CUDA standards
set_property(TARGET mlp_benchmark PROPERTY CXX_STANDARD 17)
set_property(TARGET mlp_benchmark PROPERTY CUDA_STANDARD 17)

# Additional compiler flags for CUDA
target_compile_options(mlp_benchmark PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        --generate-line-info
    >
)

# Add include directories if needed (for future headers)
target_include_directories(mlp_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Build instructions
message(STATUS "CUDA architecture set to: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "")
message(STATUS "Build instructions:")
message(STATUS "  mkdir -p build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j")
message(STATUS "")
message(STATUS "Run with:")
message(STATUS "  ./mlp_benchmark")
