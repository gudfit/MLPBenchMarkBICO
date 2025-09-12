#include "glu_kernels.h"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// Helper to verify results
void verify_results(const std::vector<half> &ref, const std::vector<half> &res,
                    const std::string &name) {
  bool correct = true;
  float max_err = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    float r = __half2float(ref[i]);
    float s = __half2float(res[i]);
    float err = std::abs(r - s);
    if (err > 1e-2f) {
      correct = false;
      std::cerr << "Verification FAILED for " << name << " at index " << i
                << ". Ref: " << r << ", Res: " << s << " (err=" << err << ")"
                << std::endl;
      break;
    }
    max_err = std::max(max_err, err);
  }
  if (correct)
    std::cout << "Verification PASSED for " << name << " (max err: " << max_err
              << ")" << std::endl;
}

int main() {
  // Problem Size
  const int M = 1024;        // Batch size * sequence length
  const int K_hidden = 4096; // Hidden dimension
  const int N_inter = 12288; // Intermediate (FFN) dimension

  std::cout << "Problem Size: A(" << M << "x" << K_hidden << ") @ W("
            << K_hidden << "x" << N_inter << ") -> down (" << N_inter << "x"
            << K_hidden << ")" << std::endl;

  // Allocate Host Memory
  std::vector<half> h_A(M * K_hidden);
  std::vector<half> h_W_up(K_hidden * N_inter);
  std::vector<half> h_W_gate(K_hidden * N_inter);
  std::vector<half> h_W_down(N_inter * K_hidden);
  std::vector<half> h_output_baseline(M * K_hidden);
  std::vector<half> h_output_fused(M * K_hidden);

  std::mt19937 rng(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &val : h_A)
    val = __float2half(dist(rng));
  for (auto &val : h_W_up)
    val = __float2half(dist(rng));
  for (auto &val : h_W_gate)
    val = __float2half(dist(rng));
  for (auto &val : h_W_down)
    val = __float2half(dist(rng));

  // Allocate Device Memory
  half *d_A, *d_W_up, *d_W_gate, *d_W_down, *d_output;
  half *d_up_proj, *d_gate_proj, *d_gated_result;
  CUDA_CHECK(cudaMalloc(&d_A, M * K_hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_W_up, K_hidden * N_inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_W_gate, K_hidden * N_inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_W_down, N_inter * K_hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_output, M * K_hidden * sizeof(half)));

  // Intermediate buffers
  CUDA_CHECK(cudaMalloc(&d_up_proj, M * N_inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_gate_proj, M * N_inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_gated_result, M * N_inter * sizeof(half)));

  // Copy Data to Device
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K_hidden * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W_up, h_W_up.data(),
                        K_hidden * N_inter * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W_gate, h_W_gate.data(),
                        K_hidden * N_inter * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_W_down, h_W_down.data(),
                        N_inter * K_hidden * sizeof(half),
                        cudaMemcpyHostToDevice));

  // Benchmarking Setup
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float ms_baseline = 0, ms_fused = 0;

  dim3 warp_block(32, 1);
  dim3 grid_up((N_inter + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
  dim3 grid_elem((M * N_inter + 255) / 256);
  dim3 block_elem(256);
  dim3 grid_down((K_hidden + WMMA_K - 1) / WMMA_K, (M + WMMA_M - 1) / WMMA_M);

  size_t fused_smem =
      (WMMA_M * BLOCK_K_FUSED + 2ULL * BLOCK_K_FUSED * WMMA_N) * sizeof(half) +
      2ULL * TILE_ELEMENTS * sizeof(float) + TILE_ELEMENTS * sizeof(half);

  // BENCHMARK 1: Baseline 3-Kernel

  std::cout << "\n--- Running Baseline (3 Kernels) ---" << std::endl;
  CUDA_CHECK(cudaEventRecord(start));
  glu_kernel1_up_gate_gemm<<<grid_up, warp_block>>>(
      d_A, d_W_up, d_W_gate, d_up_proj, d_gate_proj, M, N_inter, K_hidden);
  glu_kernel2_elementwise_swiglu<<<grid_elem, block_elem>>>(
      d_up_proj, d_gate_proj, d_gated_result, M, N_inter);
  glu_kernel3_down_gemm<<<grid_down, warp_block>>>(
      d_gated_result, d_W_down, d_output, M, N_inter, K_hidden);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_baseline, start, stop));
  std::cout << "Baseline Latency: " << ms_baseline << " ms" << std::endl;
  CUDA_CHECK(cudaMemcpy(h_output_baseline.data(), d_output,
                        M * K_hidden * sizeof(half), cudaMemcpyDeviceToHost));

  // BENCHMARK 2: Fused Up+SwiGLU + Down

  std::cout << "\n--- Running Fused (Up+SwiGLU + Down) ---" << std::endl;
  dim3 grid_fused((N_inter + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
  std::cout << "Fused kernel requesting " << fused_smem / 1024.0
            << " KB of shared memory." << std::endl;
  CUDA_CHECK(cudaEventRecord(start));
  glu_fused_up_gate_swiglu<<<grid_fused, warp_block, fused_smem>>>(
      d_A, d_W_up, d_W_gate, d_gated_result, M, N_inter, K_hidden);
  glu_kernel3_down_gemm<<<grid_down, warp_block>>>(
      d_gated_result, d_W_down, d_output, M, N_inter, K_hidden);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&ms_fused, start, stop));
  std::cout << "Fused Latency: " << ms_fused << " ms" << std::endl;

  CUDA_CHECK(cudaMemcpy(h_output_fused.data(), d_output,
                        M * K_hidden * sizeof(half), cudaMemcpyDeviceToHost));

  // Verification and Cleanup
  std::cout << "\n--- Verification ---" << std::endl;
  verify_results(h_output_baseline, h_output_fused, "Fused vs. Baseline");

  std::cout << "\n--- Performance Summary ---" << std::endl;
  if (ms_fused > 0)
    std::cout << "Speedup from Fusion: " << (ms_baseline / ms_fused) << "x"
              << std::endl;

  cudaFree(d_A);
  cudaFree(d_W_up);
  cudaFree(d_W_gate);
  cudaFree(d_W_down);
  cudaFree(d_output);
  cudaFree(d_up_proj);
  cudaFree(d_gate_proj);
  cudaFree(d_gated_result);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
