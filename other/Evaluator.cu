#include "Evaluator.h"
#include "Evaluator_dispatcher.h"
#include <iostream>

Evaluator::Evaluator(float *d_A_, const float *d_B_, float *d_C_, int M_,
                     int N_, int K_)
    : d_A(d_A_), d_B(d_B_), d_C(d_C_), M(M_), N(N_), K(K_) {}

double Evaluator::evaluate(const KernelConfig &config) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  launch_kernel_with_config(config, d_C, d_A, d_B, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(start));
  const int num_runs = 100;
  for (int i = 0; i < num_runs; ++i)
    launch_kernel_with_config(config, d_C, d_A, d_B, M, N, K);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return static_cast<double>(milliseconds) / num_runs;
}
