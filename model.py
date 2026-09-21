"""
Matrix Multiplication Algorithms in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - matmul_cpu
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

# Step 2 - max_abs_diff
#include <cmath>

float max_abs_diff(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

# Step 3 - matmul_naive_kernel
#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_matmul_naive(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    matmul_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 4 - matmul_coalesced_kernel
#include <cuda_runtime.h>

__global__ void matmul_coalesced_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_matmul_coalesced(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_coalesced_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 5 - time_launch_ms
#include <cuda_runtime.h>

typedef void (*matmul_launch_fn)(const float*, const float*, float*, int, int, int);

float time_launch_ms(matmul_launch_fn launch, const float* dA, const float* dB,
                     float* dC, int M, int N, int K, int iters) {
    // Warm-up launch: absorbs module loading and first-touch costs.
    launch(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        launch(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_ms / static_cast<float>(iters);
}

double matmul_gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * static_cast<double>(M) * N * K;
    double seconds = static_cast<double>(ms) * 1e-3;
    return flops / seconds / 1e9;
}

# Step 6 - matmul_tiled_kernel
#include <cuda_runtime.h>

constexpr int TILE_SMEM = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SMEM][TILE_SMEM];
    __shared__ float Bs[TILE_SMEM][TILE_SMEM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SMEM + tx;
    int row = blockIdx.y * TILE_SMEM + ty;

    float sum = 0.0f;

    int num_tiles = (K + TILE_SMEM - 1) / TILE_SMEM;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_SMEM;

        // Load one element of As: A[row, k0 + tx]
        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Load one element of Bs: B[k0 + ty, col]
        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SMEM; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_tiled(const float* A, const float* B, float* C,
                         int M, int N, int K) {
    dim3 block(TILE_SMEM, TILE_SMEM);
    dim3 grid((N + TILE_SMEM - 1) / TILE_SMEM, (M + TILE_SMEM - 1) / TILE_SMEM);
    matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 7 - matmul_tiled_1d_kernel
#include <cuda_runtime.h>

constexpr int R1_BM = 64, R1_BN = 64, R1_BK = 8, R1_TM = 8;

__global__ void matmul_tiled_1d_kernel(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
    __shared__ float As[R1_BM * R1_BK];   // 64 x 8
    __shared__ float Bs[R1_BK * R1_BN];   // 8 x 64

    int tid = threadIdx.x;                     // 0 .. 511

    // Output ownership: one column, 8 consecutive rows.
    int col       = tid % R1_BN;               // 0 .. 63
    int row_group = (tid / R1_BN) * R1_TM;     // 0, 8, 16, ..., 56

    // Load positions within the tiles.
    int a_row = tid / R1_BK;                   // 0 .. 63
    int a_col = tid % R1_BK;                   // 0 .. 7
    int b_row = tid / R1_BN;                   // 0 .. 7
    int b_col = tid % R1_BN;                   // 0 .. 63

    float acc[R1_TM];
    #pragma unroll
    for (int i = 0; i < R1_TM; ++i) acc[i] = 0.0f;

    int block_row = blockIdx.y * R1_BM;
    int block_col = blockIdx.x * R1_BN;

    int num_tiles = (K + R1_BK - 1) / R1_BK;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * R1_BK;

        // One guarded load per thread into As (A block_row..+63, k0..k0+7)
        int g_a_row = block_row + a_row;
        int g_a_col = k0 + a_col;
        As[a_row * R1_BK + a_col] =
            (g_a_row < M && g_a_col < K) ? A[g_a_row * K + g_a_col] : 0.0f;

        // One guarded load per thread into Bs (B k0..k0+7, block_col..+63)
        int g_b_row = k0 + b_row;
        int g_b_col = block_col + b_col;
        Bs[b_row * R1_BN + b_col] =
            (g_b_row < K && g_b_col < N) ? B[g_b_row * N + g_b_col] : 0.0f;

        __syncthreads();

        // Register blocking: one Bs read feeds R1_TM FMAs.
        #pragma unroll
        for (int k = 0; k < R1_BK; ++k) {
            float b = Bs[k * R1_BN + col];
            #pragma unroll
            for (int i = 0; i < R1_TM; ++i) {
                acc[i] += As[(row_group + i) * R1_BK + k] * b;
            }
        }

        __syncthreads();
    }

    // Guarded stores of the 8 accumulators.
    #pragma unroll
    for (int i = 0; i < R1_TM; ++i) {
        int g_row = block_row + row_group + i;
        int g_col = block_col + col;
        if (g_row < M && g_col < N) {
            C[g_row * N + g_col] = acc[i];
        }
    }
}

void launch_matmul_tiled_1d(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    dim3 block(512);
    dim3 grid((N + R1_BN - 1) / R1_BN, (M + R1_BM - 1) / R1_BM);
    matmul_tiled_1d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 8 - matmul_tiled_2d_kernel (not yet solved)
# TODO: implement

# Step 9 - matmul_vectorized_kernel (not yet solved)
# TODO: implement

# Step 10 - matmul_double_buffered_kernel (not yet solved)
# TODO: implement

# Step 11 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 12 - matmul_batched_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_splitk_kernel (not yet solved)
# TODO: implement

# Step 14 - gemv_kernel (not yet solved)
# TODO: implement

# Step 15 - matmul_bias_relu_kernel (not yet solved)
# TODO: implement

# Step 16 - matrix_addsub_kernel (not yet solved)
# TODO: implement

# Step 17 - strassen_one_level (not yet solved)
# TODO: implement

# Step 18 - csr_spmm_kernel (not yet solved)
# TODO: implement

# Step 19 - matmul_lower_triangular_kernel (not yet solved)
# TODO: implement

# Step 20 - matmul_dispatch (not yet solved)
# TODO: implement

