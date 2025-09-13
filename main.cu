#include "bico_config.h"
#include "glu_kernels.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// Helper function to verify results between reference and test implementations
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

Evaluator createEvaluator(const half *d_A, const half *d_W_up,
                          const half *d_W_gate, half *d_gated_result, int M,
                          int N, int K, half *d_W_down, half *d_output) {
  constexpr size_t half_size = sizeof(half);
  constexpr size_t static_smem_bytes =
      2 * TILE_ELEMENTS * sizeof(float) + TILE_ELEMENTS * half_size;
  return [=](const BudgetConfig &config) -> KernelResult {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    int block_k = 64;
    int num_warps = 4;
    float smem_kb = config.smem_budget;
    int max_smem_elems = static_cast<int>(smem_kb * 1024 / half_size);
    int wmma_elems = block_k * WMMA_N;
    int available_elems = max_smem_elems - 2 * wmma_elems;
    int effective_block_k = block_k;
    if (config.occ_budget < 0.5f)
      effective_block_k = std::min(block_k * 2, available_elems / WMMA_M);

    if (smem_kb <= 48 && config.occ_budget <= 0.4f) {
      block_k = 64;
      num_warps = 2;
    } else if (smem_kb > 96 && config.occ_budget <= 0.3f) {
      block_k = 128;
      num_warps = 4;
    } else {
      block_k = 64;
      num_warps = 4;
    }
    wmma_elems = block_k * WMMA_N;
    available_elems = max_smem_elems - 2 * wmma_elems;
    effective_block_k = block_k;
    if (config.occ_budget < 0.5f)
      effective_block_k =
          std::min(block_k * 2, std::max(0, available_elems) / WMMA_M);

    size_t dynamic_smem_bytes =
        (static_cast<size_t>(WMMA_M) * effective_block_k +
         2 * static_cast<size_t>(effective_block_k) * WMMA_N) *
        half_size;
    size_t total_smem_bytes = dynamic_smem_bytes + static_smem_bytes;

    int dev_id;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));
    if (total_smem_bytes > static_cast<size_t>(prop.sharedMemPerBlock)) {
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
      return KernelResult(config, 0, false);
    }

    dim3 grid_fused((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block_size(32 * num_warps);
    float latency_ms = 0;

    try {
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(start));
      for (int i = 0; i < 5; ++i) {
        if (smem_kb <= 48 && config.occ_budget <= 0.4f) {
          glu_fused_up_gate_swiglu_budget<64, 2>
              <<<grid_fused, block_size, dynamic_smem_bytes>>>(
                  d_A, d_W_up, d_W_gate, d_gated_result, M, N, K,
                  static_cast<int>(smem_kb), config.occ_budget);
        } else if (smem_kb > 96 && config.occ_budget <= 0.3f) {
          glu_fused_up_gate_swiglu_budget<128, 4>
              <<<grid_fused, block_size, dynamic_smem_bytes>>>(
                  d_A, d_W_up, d_W_gate, d_gated_result, M, N, K,
                  static_cast<int>(smem_kb), config.occ_budget);
        } else {
          glu_fused_up_gate_swiglu_budget<64, 4>
              <<<grid_fused, block_size, dynamic_smem_bytes>>>(
                  d_A, d_W_up, d_W_gate, d_gated_result, M, N, K,
                  static_cast<int>(smem_kb), config.occ_budget);
        }
      }
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      CUDA_CHECK(cudaEventElapsedTime(&latency_ms, start, stop));
      latency_ms /= 5.0f;

      dim3 grid_down((K + WMMA_K - 1) / WMMA_K, (M + WMMA_M - 1) / WMMA_M);
      glu_kernel3_down_gemm<<<grid_down, dim3(32)>>>(d_gated_result, d_W_down,
                                                     d_output, M, N, K);

      CUDA_CHECK(cudaDeviceSynchronize());
    } catch (...) {
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
      return KernelResult(config, 0, false);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return KernelResult(config, latency_ms, true);
  };
}

int main() {
  const int M = 1024;
  const int K_hidden = 4096;
  const int N_inter = 12288;
  std::cout << "Problem Size: A(" << M << "x" << K_hidden << ") @ W("
            << K_hidden << "x" << N_inter << ") -> down (" << N_inter << "x"
            << K_hidden << ")" << std::endl;
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
  half *d_A, *d_W_up, *d_W_gate, *d_W_down, *d_output;
  half *d_up_proj, *d_gate_proj, *d_gated_result;
  CUDA_CHECK(cudaMalloc(&d_A, M * K_hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_W_up, K_hidden * N_inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_W_gate, K_hidden * N_inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_W_down, N_inter * K_hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_output, M * K_hidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_up_proj, M * N_inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_gate_proj, M * N_inter * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_gated_result, M * N_inter * sizeof(half)));
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
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float ms_baseline = 0;
  dim3 warp_block(32, 1);
  dim3 grid_up((N_inter + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
  dim3 grid_elem((M * N_inter + 255) / 256);
  dim3 block_elem(256);
  dim3 grid_down((K_hidden + WMMA_K - 1) / WMMA_K, (M + WMMA_M - 1) / WMMA_M);
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
  auto evaluator = createEvaluator(d_A, d_W_up, d_W_gate, d_gated_result, M,
                                   N_inter, K_hidden, d_W_down, d_output);
  BICOExplorer explorer(evaluator);
  std::cout << "\n--- Exploring Budget Frontier ---" << std::endl;
  explorer.exploreFrontier();
  BudgetConfig best_config = explorer.findBestConfig();
  std::cout << "Best configuration: " << best_config << std::endl;
  std::cout << "\n--- Final Run with Best Configuration ---" << std::endl;
  KernelResult final_result = evaluator(best_config);

  if (final_result.valid) {
    std::cout << "Final latency: " << final_result.latency_ms << " ms"
              << std::endl;
    CUDA_CHECK(cudaMemcpy(h_output_fused.data(), d_output,
                          M * K_hidden * sizeof(half), cudaMemcpyDeviceToHost));

    std::cout << "\n--- Verification ---" << std::endl;
    verify_results(h_output_baseline, h_output_fused, "Optimized vs. Baseline");
    std::cout << "\n--- Performance Summary ---" << std::endl;
    if (final_result.latency_ms > 0)
      std::cout << "Speedup from Optimization: "
                << (ms_baseline / final_result.latency_ms) << "x" << std::endl;
  } else {
    std::cout << "Best configuration is invalid!" << std::endl;
  }
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_W_up));
  CUDA_CHECK(cudaFree(d_W_gate));
  CUDA_CHECK(cudaFree(d_W_down));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_up_proj));
  CUDA_CHECK(cudaFree(d_gate_proj));
  CUDA_CHECK(cudaFree(d_gated_result));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
