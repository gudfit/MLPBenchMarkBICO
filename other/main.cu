#include "BICOExplorer.h"
#include "Evaluator.h"
#include "KernelConfig.h"
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

std::vector<KernelConfig> generate_search_space() {
  std::vector<KernelConfig> space;
  std::vector<int> tile_m_opts = {16, 32, 64, 128};
  std::vector<int> tile_n_opts = {16, 32, 64, 128};
  std::vector<int> tile_k_opts = {8, 16, 32};
  std::vector<int> block_dim_x_opts = {16, 32, 64};

  for (int tm : tile_m_opts) {
    for (int tn : tile_n_opts) {
      for (int tk : tile_k_opts) {
        for (int bx : block_dim_x_opts) {
          if (tn % bx != 0)
            continue;
          int by = tm;
          if (bx * by > 1024)
            continue;
          size_t shared_needed = (tm * tk + tk * tn) * sizeof(float);
          if (shared_needed > 48000)
            continue;
          space.push_back({tm, tn, tk, bx, by});
        }
      }
    }
  }
  std::cout << "Generated a search space of " << space.size()
            << " valid configurations." << std::endl;
  return space;
}

int main() {
  const int M = 1024;
  const int K = 4096;
  const int N = 12288;
  std::cout << "Matrix dimensions: " << M << " x " << K << " * " << K << " x "
            << N << std::endl;
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &val : h_A)
    val = dist(rng);
  for (auto &val : h_B)
    val = dist(rng);
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float),
                        cudaMemcpyHostToDevice));
  std::vector<KernelConfig> search_space = generate_search_space();
  Evaluator evaluator(d_A, d_B, d_C, M, N, K);
  BICOExplorer explorer(search_space, evaluator);
  int budget = 50;
  explorer.explore(std::min(budget, static_cast<int>(search_space.size())));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return 0;
}
