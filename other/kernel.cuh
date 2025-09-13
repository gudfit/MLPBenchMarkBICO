#pragma once
#include <cuda_runtime.h>

template <int TILE_M, int TILE_N, int TILE_K, int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void matrix_multiply_kernel(float *C, const float *A, const float *B,
                                       int M, int N, int K) {
  static_assert(BLOCK_DIM_X >= TILE_N, "BLOCK_DIM_X must be at least TILE_N");
  static_assert(BLOCK_DIM_Y >= TILE_M, "BLOCK_DIM_Y must be at least TILE_M");
  extern __shared__ float smem[];
  float *As = smem;
  float *Bs = smem + TILE_M * TILE_K;
  const int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;
  const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y;
  const int c_row = block_row * TILE_M + threadIdx.y;
  const int c_col = block_col * TILE_N + threadIdx.x;
  float accumulator = 0.0f;
  for (int k_tile_idx = 0; k_tile_idx < K; k_tile_idx += TILE_K) {
    for (int i = thread_idx; i < TILE_M * TILE_K; i += num_threads) {
      const int load_row = i / TILE_K;
      const int load_col = i % TILE_K;
      const int gmem_row = block_row * TILE_M + load_row;
      const int gmem_col = k_tile_idx + load_col;
      if (gmem_row < M && gmem_col < K)
        As[load_row * TILE_K + load_col] = A[gmem_row * K + gmem_col];
      else
        As[load_row * TILE_K + load_col] = 0.0f;
    }
    for (int i = thread_idx; i < TILE_K * TILE_N; i += num_threads) {
      const int load_row = i / TILE_N;
      const int load_col = i % TILE_N;
      const int gmem_row = k_tile_idx + load_row;
      const int gmem_col = block_col * TILE_N + load_col;
      if (gmem_row < K && gmem_col < N)
        Bs[load_row * TILE_N + load_col] = B[gmem_row * N + gmem_col];
      else
        Bs[load_row * TILE_N + load_col] = 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < TILE_K; ++k)
      accumulator +=
          As[threadIdx.y * TILE_K + k] * Bs[k * TILE_N + threadIdx.x];
    __syncthreads();
  }
  if (c_row < M && c_col < N)
    C[c_row * N + c_col] = accumulator;
}
