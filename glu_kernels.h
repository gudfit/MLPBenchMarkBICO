#pragma once

// 确保在包含mma.h之前定义必要的宏
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#if defined(__CUDACC__)
#define __CUDACC_VER_MAJOR__ __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MINOR__ __CUDACC_VER_MINOR__
#define __CUDACC_VER_BUILD__ __CUDACC_VER_BUILD__
#endif

#include <mma.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Shared memory size for compute in baseline and fused
constexpr int TILE_ELEMENTS = WMMA_M * WMMA_N;

// Baseline kernel 1: Up and Gate projection using simple WMMA (one warp per tile)
__global__ void glu_kernel1_up_gate_gemm(const half* A, const half* W_up, const half* W_gate, half* up_proj, half* gate_proj, int M, int N, int K) {
    int tile_m = blockIdx.y * WMMA_M;
    int tile_n = blockIdx.x * WMMA_N;

    if (tile_m >= M || tile_n >= N) return;

    // Fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag_up;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_up;
    nvcuda::wmma::fill_fragment(acc_frag_up, 0.0f);
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag_gate;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag_gate;
    nvcuda::wmma::fill_fragment(acc_frag_gate, 0.0f);

    __shared__ float sh_acc_up[TILE_ELEMENTS];
    __shared__ float sh_acc_gate[TILE_ELEMENTS];

    int num_tiles_k = (K + WMMA_K - 1) / WMMA_K;
    for (int tile_k = 0; tile_k < num_tiles_k; ++tile_k) {
        const half* a_tile = A + tile_m * K + tile_k * WMMA_K;
        nvcuda::wmma::load_matrix_sync(a_frag, a_tile, K);

        const half* b_up_tile = W_up + tile_k * WMMA_K * N + tile_n;
        nvcuda::wmma::load_matrix_sync(b_frag_up, b_up_tile, N);

        const half* b_gate_tile = W_gate + tile_k * WMMA_K * N + tile_n;
        nvcuda::wmma::load_matrix_sync(b_frag_gate, b_gate_tile, N);

        nvcuda::wmma::mma_sync(acc_frag_up, a_frag, b_frag_up, acc_frag_up);
        nvcuda::wmma::mma_sync(acc_frag_gate, a_frag, b_frag_gate, acc_frag_gate);
    }

    // Store to shared (packed row-major)
    nvcuda::wmma::store_matrix_sync(sh_acc_up, acc_frag_up, WMMA_N, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(sh_acc_gate, acc_frag_gate, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    // Convert and write to global (strided)
    int global_offset = tile_m * N + tile_n;
    int tid = threadIdx.x;
    for (int i = tid; i < TILE_ELEMENTS; i += blockDim.x) {
        int r = i / WMMA_N;
        int c = i % WMMA_N;
        up_proj[global_offset + r * N + c] = __float2half(sh_acc_up[i]);
        gate_proj[global_offset + r * N + c] = __float2half(sh_acc_gate[i]);
    }
}

// Baseline kernel 2: Elementwise SwiGLU
__global__ void glu_kernel2_elementwise_swiglu(const half* up_proj, const half* gate_proj, half* gated_result, int M, int N) {
    int total_elements = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float gate_val = __half2float(gate_proj[idx]);
        float up_val = __half2float(up_proj[idx]);
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
        float swish_gate = gate_val * sigmoid_gate;
        gated_result[idx] = __float2half(swish_gate * up_val);
    }
}

// Baseline kernel 3: Down projection using simple WMMA
__global__ void glu_kernel3_down_gemm(const half* gated_result, const half* W_down, half* final_output, int M, int N, int K) {
    int tile_m = blockIdx.y * WMMA_M;
    int tile_n = blockIdx.x * WMMA_K;

    if (tile_m >= M || tile_n >= K) return;

    // Fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_K, WMMA_K, WMMA_K, half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_K, WMMA_K, float> acc_frag;
    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    __shared__ float sh_acc[TILE_ELEMENTS];

    int num_tiles_k = (N + WMMA_K - 1) / WMMA_K;
    for (int tile_k = 0; tile_k < num_tiles_k; ++tile_k) {
        const half* a_tile = gated_result + tile_m * N + tile_k * WMMA_K;
        nvcuda::wmma::load_matrix_sync(a_frag, a_tile, N);

        const half* b_tile = W_down + tile_k * WMMA_K * K + tile_n;
        nvcuda::wmma::load_matrix_sync(b_frag, b_tile, K);

        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // Store to shared (packed row-major)
    nvcuda::wmma::store_matrix_sync(sh_acc, acc_frag, WMMA_K, nvcuda::wmma::mem_row_major);
    __syncthreads();

    // Convert and write to global (strided)
    int global_offset = tile_m * K + tile_n;
    int tid = threadIdx.x;
    for (int i = tid; i < TILE_ELEMENTS; i += blockDim.x) {
        int r = i / WMMA_K;
        int c = i % WMMA_K;
        final_output[global_offset + r * K + c] = __float2half(sh_acc[i]);
    }
}

// Fused kernel: Up + Gate + SwiGLU with shared memory (single tile per block for simplicity)
constexpr int BLOCK_K_FUSED = 128;

__global__ void glu_fused_up_gate_swiglu(const half* A, const half* W_up, const half* W_gate, half* gated_result, int M, int N, int K) {
    extern __shared__ half smem[];
    half* sA = smem;
    half* sW_up = sA + WMMA_M * BLOCK_K_FUSED;
    half* sW_gate = sW_up + BLOCK_K_FUSED * WMMA_N;

    // Compute shared
    __shared__ float sh_up[TILE_ELEMENTS];
    __shared__ float sh_gate[TILE_ELEMENTS];
    __shared__ half sh_final[TILE_ELEMENTS];

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_up;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_gate;
    nvcuda::wmma::fill_fragment(acc_up, 0.0f);
    nvcuda::wmma::fill_fragment(acc_gate, 0.0f);

    int block_row = blockIdx.y * WMMA_M;
    int block_col = blockIdx.x * WMMA_N;
    if (block_row >= M || block_col >= N) return;

    int tid = threadIdx.x;
    for (int bk = 0; bk < K; bk += BLOCK_K_FUSED) {
        // Load A to shared (M tile x K block)
        for (int i = tid; i < WMMA_M * BLOCK_K_FUSED; i += blockDim.x) {
            int row = i / BLOCK_K_FUSED;
            int col = i % BLOCK_K_FUSED;
            if (block_row + row < M && bk + col < K) {
                sA[i] = A[(block_row + row) * K + bk + col];
            } else {
                sA[i] = __float2half(0.0f);
            }
        }
        // Load W_up and W_gate to shared (K block x N tile, row-major)
        for (int i = tid; i < BLOCK_K_FUSED * WMMA_N; i += blockDim.x) {
            int row = i / WMMA_N;
            int col = i % WMMA_N;
            if (bk + row < K && block_col + col < N) {
                sW_up[i] = W_up[(bk + row) * N + block_col + col];
                sW_gate[i] = W_gate[(bk + row) * N + block_col + col];
            } else {
                sW_up[i] = __float2half(0.0f);
                sW_gate[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int ki = 0; ki < BLOCK_K_FUSED; ki += WMMA_K) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag_up, b_frag_gate;
            nvcuda::wmma::load_matrix_sync(a_frag, sA + ki, BLOCK_K_FUSED);
            nvcuda::wmma::load_matrix_sync(b_frag_up, sW_up + ki * WMMA_N, WMMA_N);
            nvcuda::wmma::load_matrix_sync(b_frag_gate, sW_gate + ki * WMMA_N, WMMA_N);

            nvcuda::wmma::mma_sync(acc_up, a_frag, b_frag_up, acc_up);
            nvcuda::wmma::mma_sync(acc_gate, a_frag, b_frag_gate, acc_gate);
        }
        __syncthreads();
    }

    // Store to shared (packed)
    nvcuda::wmma::store_matrix_sync(sh_up, acc_up, WMMA_N, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(sh_gate, acc_gate, WMMA_N, nvcuda::wmma::mem_row_major);
    __syncthreads();

    // Compute SwiGLU elementwise
    for (int i = tid; i < TILE_ELEMENTS; i += blockDim.x) {
        float gate_val = sh_gate[i];
        float up_val = sh_up[i];
        float sigmoid_gate = 1.0f / (1.0f + expf(-gate_val));
        float swish_gate = gate_val * sigmoid_gate;
        sh_gate[i] = swish_gate * up_val;  // Reuse sh_gate for result
    }
    __syncthreads();

    // Convert to half in shared
    for (int i = tid; i < TILE_ELEMENTS; i += blockDim.x) {
        sh_final[i] = __float2half(sh_gate[i]);
    }
    __syncthreads();

    // Load to half fragment (packed)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> final_frag;
    nvcuda::wmma::load_matrix_sync(final_frag, sh_final, WMMA_N, nvcuda::wmma::mem_row_major);

    // Store to global (strided)
    int out_offset = block_row * N + block_col;
    nvcuda::wmma::store_matrix_sync(gated_result + out_offset, final_frag, N, nvcuda::wmma::mem_row_major);
}
