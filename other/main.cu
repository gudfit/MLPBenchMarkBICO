#include "BICOExplorer.h"
#include "Evaluator.h"
#include "KernelConfig.h"
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
// Helper macro for checking CUDA calls
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

// Updated: Hardcoded search space matching supported dispatcher configs (sync with generate_dispatcher.py)
std::vector<KernelConfig> generate_search_space() {
    std::vector<KernelConfig> space = {
        // Original 5 + 10 more (matching CONFIGURATIONS in Python script)
        {16, 16, 16, 16, 16},
        {32, 32, 32, 32, 8},
        {64, 32, 16, 32, 16},
        {32, 64, 16, 64, 8},
        {128, 16, 8, 16, 32},
        {64, 64, 16, 16, 64},  // The failing config (now supported)
        {32, 32, 32, 32, 32},
        {16, 32, 16, 32, 16},
        {16, 64, 16, 64, 16},
        {64, 32, 16, 16, 64},
        {32, 16, 16, 16, 32},
        {16, 16, 8, 16, 16},
        {64, 16, 8, 16, 64},
        {32, 64, 8, 32, 32},
        {128, 64, 8, 64, 16},
        // Add more here to match expanded CONFIGURATIONS (up to 60 if desired)
    };
    std::cout << "Generated a search space of " << space.size() << " valid configurations (supported by dispatcher)." << std::endl;
    return space;
}

int main() {
  const int M = 1024;
  const int K = 4096;
  const int N = 12288;
  std::cout << "Matrix dimensions: " << M << " x " << K << " * " << K << " x "
            << N << std::endl;
  // Generate random data on host
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &val : h_A)
    val = dist(rng);
  for (auto &val : h_B)
    val = dist(rng);
  // Allocate device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice));
  // 1. Generate the Exploration Space (C) programmatically
  std::vector<KernelConfig> search_space = generate_search_space();
  // 2. Create the Evaluator
  Evaluator evaluator(d_A, d_B, d_C, M, N, K);
  // 3. Create the Explorer
  BICOExplorer explorer(search_space, evaluator);
  // 4. Define the Budget (Lambda) and run the exploration
  int budget = 50; // Evaluate a subset of the generated space
  explorer.explore(std::min(budget, static_cast<int>(search_space.size())));
  // Cleanup
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return 0;
}
