#include "BICOExplorer.h"
#include "Evaluator.h"
#include "Heuristic.h"
#include "KernelConfig.h"
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip> // Added for std::setprecision and std::fixed
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

std::vector<KernelConfig> load_search_space_from_file(const std::string &filename) {
  std::vector<KernelConfig> space;
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    std::cerr << "FATAL: Could not open configuration file: " << filename << std::endl;
    exit(1);
  }
  KernelConfig config;
  while (infile >> config.TILE_M >> config.TILE_N >> config.TILE_K >>
         config.BLOCK_DIM_X >> config.BLOCK_DIM_Y) {
    if (config.isValid()) {
      space.push_back(config);
    } else {
      std::cerr << "Warning: Invalid configuration skipped: " << config.toString() << std::endl;
    }
  }
  std::cout << "Loaded a search space of " << space.size()
            << " valid configurations from " << filename << "." << std::endl;
  return space;
}

int main() {
  const int M = 1024;
  const int K = 4096;
  const int N = 12288;
  std::cout << "Matrix dimensions: " << M << " x " << K << " * " << K << " x " << N
            << std::endl;
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &val : h_A) val = dist(rng);
  for (auto &val : h_B) val = dist(rng);
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
  std::cout << "\n===== [Phase 1: Predictive BICO Exploration] =====\n";
  std::vector<KernelConfig> full_search_space = load_search_space_from_file("configurations.txt");
  std::vector<KernelConfig> guided_search_space = predictive_search(full_search_space);
  std::cout << "Predictive model has ranked " << guided_search_space.size() << " configurations.\n";
  // Save all ranked configurations to a text file
  std::ofstream ranked_file("ranked_configurations.txt");
  if (ranked_file.is_open()) {
    ranked_file << "Ranked Configurations by Reuse Score (Phase 1 Predictive BICO Exploration):\n\n";
    for (size_t i = 0; i < guided_search_space.size(); ++i) {
      double score = calculate_reuse_score(guided_search_space[i]);
      ranked_file << "#" << (i+1) << ": " << guided_search_space[i].toString()
                  << " (Score: " << std::fixed << std::setprecision(2) << score << ")\n";
    }
    ranked_file.close();
    std::cout << "All ranked configurations saved to ranked_configurations.txt\n";
  } else {
    std::cerr << "Warning: Could not open ranked_configurations.txt for writing.\n";
  }
  std::cout << "Top 5 most promising candidates:\n";
  for (int i = 0; i < 5 && i < guided_search_space.size(); ++i) {
    std::cout << " #" << i + 1 << ": " << guided_search_space[i].toString()
              << " (Reuse Score: " << std::fixed << std::setprecision(2)
              << calculate_reuse_score(guided_search_space[i]) << ")\n";
  }
  std::vector<KernelConfig> random_search_space = full_search_space;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(random_search_space.begin(), random_search_space.end(), g);
  std::cout << "\n===== [Phase 2: Empirical Head-to-Head Exploration] =====\n";
  int budget = 20;
  Evaluator evaluator(d_A, d_B, d_C, M, N, K);
  std::cout << "\n--- [Running BICO-GUIDED Explorer] ---\n";
  BICOExplorer explorer_guided("GUIDED", guided_search_space, evaluator, false);
  explorer_guided.explore(budget);
  std::cout << "\n--- [Running RANDOM (Baseline) Explorer] ---\n";
  BICOExplorer explorer_random("RANDOM", random_search_space, evaluator, true);
  explorer_random.explore(budget);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  return 0;
}
