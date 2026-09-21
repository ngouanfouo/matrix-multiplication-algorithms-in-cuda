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

# Step 8 - matmul_tiled_2d_kernel
#include <cuda_runtime.h>

constexpr int R2_BM = 64, R2_BN = 64, R2_BK = 8, R2_TM = 4, R2_TN = 4;

__global__ void matmul_tiled_2d_kernel(const float* A, const float* B, float* C,
                                       int M, int N, int K) {
    __shared__ float As[R2_BM * R2_BK];   // 64 x 8 = 512
    __shared__ float Bs[R2_BK * R2_BN];   // 8 x 64 = 512

    int tid = threadIdx.x;                       // 0 .. 255
    int thread_row = (tid / 16) * R2_TM;         // 0, 4, ..., 60
    int thread_col = (tid % 16) * R2_TN;         // 0, 4, ..., 60

    int block_row = blockIdx.y * R2_BM;
    int block_col = blockIdx.x * R2_BN;

    float acc[R2_TM][R2_TN];
    #pragma unroll
    for (int i = 0; i < R2_TM; ++i)
        #pragma unroll
        for (int j = 0; j < R2_TN; ++j)
            acc[i][j] = 0.0f;

    int num_tiles = (K + R2_BK - 1) / R2_BK;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * R2_BK;

        // Load A tile: 64 x 8, two elements per thread
        for (int i = tid; i < R2_BM * R2_BK; i += 256) {
            int r = i / R2_BK;
            int c = i % R2_BK;
            int gr = block_row + r;
            int gc = k0 + c;
            As[i] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        // Load B tile: 8 x 64, two elements per thread
        for (int i = tid; i < R2_BK * R2_BN; i += 256) {
            int r = i / R2_BN;
            int c = i % R2_BN;
            int gr = k0 + r;
            int gc = block_col + c;
            Bs[i] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        // Compute the thread's 4 x 4 sub-tile
        #pragma unroll
        for (int k = 0; k < R2_BK; ++k) {
            float regA[R2_TM];
            float regB[R2_TN];

            #pragma unroll
            for (int i = 0; i < R2_TM; ++i)
                regA[i] = As[(thread_row + i) * R2_BK + k];

            #pragma unroll
            for (int j = 0; j < R2_TN; ++j)
                regB[j] = Bs[k * R2_BN + thread_col + j];

            #pragma unroll
            for (int i = 0; i < R2_TM; ++i)
                #pragma unroll
                for (int j = 0; j < R2_TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    // Guarded 4 x 4 store
    #pragma unroll
    for (int i = 0; i < R2_TM; ++i) {
        int gr = block_row + thread_row + i;
        #pragma unroll
        for (int j = 0; j < R2_TN; ++j) {
            int gc = block_col + thread_col + j;
            if (gr < M && gc < N) {
                C[gr * N + gc] = acc[i][j];
            }
        }
    }
}

void launch_matmul_tiled_2d(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + R2_BN - 1) / R2_BN, (M + R2_BM - 1) / R2_BM);
    matmul_tiled_2d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 9 - matmul_vectorized_kernel
#include <cuda_runtime.h>

constexpr int V_BM = 64, V_BN = 64, V_BK = 8, V_TM = 4, V_TN = 4;

__global__ void matmul_vectorized_kernel(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    // A tile stored transposed: As[k * V_BM + m] = A[block_row + m][k0 + k]
    __shared__ float As[V_BK * V_BM];
    // B tile stored row-major with 16-byte alignment for float4 access
    __shared__ __align__(16) float Bs[V_BK * V_BN];

    int tid = threadIdx.x;                          // 0 .. 255
    int thread_row = (tid / 16) * V_TM;             // 0, 4, ..., 60
    int thread_col = (tid % 16) * V_TN;             // 0, 4, ..., 60
    int block_row  = blockIdx.y * V_BM;
    int block_col  = blockIdx.x * V_BN;

    float acc[V_TM][V_TN];
    #pragma unroll
    for (int i = 0; i < V_TM; ++i)
        #pragma unroll
        for (int j = 0; j < V_TN; ++j)
            acc[i][j] = 0.0f;

    int num_tiles = (K + V_BK - 1) / V_BK;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * V_BK;

        if (tid < 128) {
            // --- A tile: 64 x 8, one float4 per thread, stored transposed ---
            int r  = tid / 2;
            int c  = (tid % 2) * 4;
            int gr = block_row + r;
            int gc = k0 + c;

            float4 a4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gr < M && gc < K) {
                a4 = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
            }
            As[(c + 0) * V_BM + r] = a4.x;
            As[(c + 1) * V_BM + r] = a4.y;
            As[(c + 2) * V_BM + r] = a4.z;
            As[(c + 3) * V_BM + r] = a4.w;

            // --- B tile: 8 x 64, one float4 per thread ---
            int br  = tid / 16;
            int bc  = (tid % 16) * 4;
            int gbr = k0 + br;
            int gbc = block_col + bc;

            float4 b4 = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gbr < K && gbc < N) {
                b4 = *reinterpret_cast<const float4*>(&B[gbr * N + gbc]);
            }
            *reinterpret_cast<float4*>(&Bs[br * V_BN + bc]) = b4;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < V_BK; ++k) {
            float regA[V_TM];
            #pragma unroll
            for (int i = 0; i < V_TM; ++i)
                regA[i] = As[k * V_BM + thread_row + i];

            float4 regB = *reinterpret_cast<const float4*>(&Bs[k * V_BN + thread_col]);

            #pragma unroll
            for (int i = 0; i < V_TM; ++i) {
                acc[i][0] += regA[i] * regB.x;
                acc[i][1] += regA[i] * regB.y;
                acc[i][2] += regA[i] * regB.z;
                acc[i][3] += regA[i] * regB.w;
            }
        }

        __syncthreads();
    }

    // Guarded float4 stores: column guard is required when N < block_col + 64.
    // N % 4 == 0 and thread_col % 4 == 0, so gc < N implies gc + 3 < N.
    #pragma unroll
    for (int i = 0; i < V_TM; ++i) {
        int gr = block_row + thread_row + i;
        int gc = block_col + thread_col;
        if (gr < M && gc < N) {
            float4 out = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4*>(&C[gr * N + gc]) = out;
        }
    }
}

void launch_matmul_vectorized(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    // Caller guarantees K % 4 == 0 and N % 4 == 0.
    dim3 block(256);
    dim3 grid((N + V_BN - 1) / V_BN, (M + V_BM - 1) / V_BM);
    matmul_vectorized_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 10 - matmul_double_buffered_kernel
#include <cuda_runtime.h>

constexpr int D_BM = 64, D_BN = 64, D_BK = 8, D_TM = 4, D_TN = 4;

__global__ void matmul_double_buffered_kernel(const float* A, const float* B, float* C,
                                              int M, int N, int K) {
    __shared__ float As[2][D_BM * D_BK];   // 2 stages of 64 x 8
    __shared__ float Bs[2][D_BK * D_BN];   // 2 stages of 8 x 64

    int tid = threadIdx.x;                       // 0 .. 255
    int thread_row = (tid / 16) * D_TM;          // 0, 4, ..., 60
    int thread_col = (tid % 16) * D_TN;          // 0, 4, ..., 60
    int block_row  = blockIdx.y * D_BM;
    int block_col  = blockIdx.x * D_BN;

    // Each thread owns two elements of the 512-element A tile and two of B.
    // Flat index = tid and tid + 256; A uses (r = i / 8, c = i % 8),
    // B uses (r = i / 64, c = i % 64).
    int a_r0 = tid / D_BK;              int a_c0 = tid % D_BK;
    int a_r1 = (tid + 256) / D_BK;      int a_c1 = (tid + 256) % D_BK;
    int b_r0 = tid / D_BN;              int b_c0 = tid % D_BN;
    int b_r1 = (tid + 256) / D_BN;      int b_c1 = (tid + 256) % D_BN;

    float acc[D_TM][D_TN];
    #pragma unroll
    for (int i = 0; i < D_TM; ++i)
        #pragma unroll
        for (int j = 0; j < D_TN; ++j)
            acc[i][j] = 0.0f;

    int num_tiles = (K + D_BK - 1) / D_BK;

    // -------- Prologue: load tile 0 into stage 0 --------
    {
        int gr0 = block_row + a_r0, gc0 = a_c0;
        int gr1 = block_row + a_r1, gc1 = a_c1;
        As[0][a_r0 * D_BK + a_c0] = (gr0 < M && gc0 < K) ? A[gr0 * K + gc0] : 0.0f;
        As[0][a_r1 * D_BK + a_c1] = (gr1 < M && gc1 < K) ? A[gr1 * K + gc1] : 0.0f;

        int gbr0 = b_r0, gbc0 = block_col + b_c0;
        int gbr1 = b_r1, gbc1 = block_col + b_c1;
        Bs[0][b_r0 * D_BN + b_c0] = (gbr0 < K && gbc0 < N) ? B[gbr0 * N + gbc0] : 0.0f;
        Bs[0][b_r1 * D_BN + b_c1] = (gbr1 < K && gbc1 < N) ? B[gbr1 * N + gbc1] : 0.0f;
    }
    __syncthreads();

    // -------- Steady state --------
    int stage = 0;
    for (int t = 0; t < num_tiles; ++t) {
        // 1) Prefetch tile t+1 into registers (still global memory, hidden
        //    behind the FMAs below).
        float pa0 = 0.f, pa1 = 0.f, pb0 = 0.f, pb1 = 0.f;
        bool has_next = (t + 1) < num_tiles;
        if (has_next) {
            int kn = (t + 1) * D_BK;

            int gr0 = block_row + a_r0, gc0 = kn + a_c0;
            int gr1 = block_row + a_r1, gc1 = kn + a_c1;
            pa0 = (gr0 < M && gc0 < K) ? A[gr0 * K + gc0] : 0.0f;
            pa1 = (gr1 < M && gc1 < K) ? A[gr1 * K + gc1] : 0.0f;

            int gbr0 = kn + b_r0, gbc0 = block_col + b_c0;
            int gbr1 = kn + b_r1, gbc1 = block_col + b_c1;
            pb0 = (gbr0 < K && gbc0 < N) ? B[gbr0 * N + gbc0] : 0.0f;
            pb1 = (gbr1 < K && gbc1 < N) ? B[gbr1 * N + gbc1] : 0.0f;
        }

        // 2) Compute from the current stage.
        #pragma unroll
        for (int k = 0; k < D_BK; ++k) {
            float regA[D_TM];
            float regB[D_TN];
            #pragma unroll
            for (int i = 0; i < D_TM; ++i)
                regA[i] = As[stage][(thread_row + i) * D_BK + k];
            #pragma unroll
            for (int j = 0; j < D_TN; ++j)
                regB[j] = Bs[stage][k * D_BN + thread_col + j];
            #pragma unroll
            for (int i = 0; i < D_TM; ++i)
                #pragma unroll
                for (int j = 0; j < D_TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
        }

        // 3) Publish the prefetched registers into the free stage.
        int next_stage = 1 - stage;
        if (has_next) {
            As[next_stage][a_r0 * D_BK + a_c0] = pa0;
            As[next_stage][a_r1 * D_BK + a_c1] = pa1;
            Bs[next_stage][b_r0 * D_BN + b_c0] = pb0;
            Bs[next_stage][b_r1 * D_BN + b_c1] = pb1;
        }

        // 4) One barrier: separates this iteration's compute from the next
        //    iteration's writes into the same stage, and this iteration's
        //    writes into next_stage from the next iteration's reads.
        __syncthreads();
        stage = next_stage;
    }

    // -------- Epilogue: guarded 4 x 4 store --------
    #pragma unroll
    for (int i = 0; i < D_TM; ++i) {
        int gr = block_row + thread_row + i;
        #pragma unroll
        for (int j = 0; j < D_TN; ++j) {
            int gc = block_col + thread_col + j;
            if (gr < M && gc < N)
                C[gr * N + gc] = acc[i][j];
        }
    }
}

void launch_matmul_double_buffered(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + D_BN - 1) / D_BN, (M + D_BM - 1) / D_BM);
    matmul_double_buffered_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

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

