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

# Step 11 - matmul_nt_kernel
#include <cuda_runtime.h>

constexpr int TILE_NT = 16;

__global__ void matmul_nt_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    __shared__ float As[TILE_NT][TILE_NT];
    __shared__ float Bs[TILE_NT][TILE_NT];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_NT + ty;    // m index in C
    int col = blockIdx.x * TILE_NT + tx;    // n index in C

    float sum = 0.0f;

    int num_tiles = (K + TILE_NT - 1) / TILE_NT;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_NT;

        // A tile: A[row][k0 + tx]
        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // B tile: B[block_col + ty][k0 + tx] — B is N x K row-major
        int b_row = blockIdx.x * TILE_NT + ty;
        int b_col = k0 + tx;
        Bs[ty][tx] = (b_row < N && b_col < K) ? B[b_row * K + b_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_NT; ++k) {
            sum += As[ty][k] * Bs[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_nt(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 block(TILE_NT, TILE_NT);
    dim3 grid((N + TILE_NT - 1) / TILE_NT, (M + TILE_NT - 1) / TILE_NT);
    matmul_nt_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 12 - matmul_batched_kernel
#include <cuda_runtime.h>

constexpr int TILE_BATCH = 16;

__global__ void matmul_batched_kernel(const float* A, const float* B, float* C,
                                      int M, int N, int K) {
    __shared__ float As[TILE_BATCH][TILE_BATCH];
    __shared__ float Bs[TILE_BATCH][TILE_BATCH];

    // Select this block's batch element.
    const float* Ab = A + (size_t)blockIdx.z * M * K;
    const float* Bb = B + (size_t)blockIdx.z * K * N;
    float*       Cb = C + (size_t)blockIdx.z * M * N;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_BATCH + ty;   // m index in C
    int col = blockIdx.x * TILE_BATCH + tx;   // n index in C

    float sum = 0.0f;

    int num_tiles = (K + TILE_BATCH - 1) / TILE_BATCH;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_BATCH;

        // A tile: Ab[row][k0 + tx]
        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? Ab[row * K + a_col] : 0.0f;

        // B tile: Bb[k0 + ty][col]
        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? Bb[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_BATCH; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        Cb[row * N + col] = sum;
    }
}

void launch_matmul_batched(const float* A, const float* B, float* C,
                           int M, int N, int K, int batch) {
    dim3 block(TILE_BATCH, TILE_BATCH);
    dim3 grid((N + TILE_BATCH - 1) / TILE_BATCH,
              (M + TILE_BATCH - 1) / TILE_BATCH,
              batch);
    matmul_batched_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# Step 13 - matmul_splitk_kernel
#include <cuda_runtime.h>

constexpr int TILE_SPLITK = 16;

__global__ void matmul_splitk_kernel(const float* A, const float* B, float* C,
                                     int M, int N, int K, int k_per_split) {
    __shared__ float As[TILE_SPLITK][TILE_SPLITK];
    __shared__ float Bs[TILE_SPLITK][TILE_SPLITK];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SPLITK + ty;
    int col = blockIdx.x * TILE_SPLITK + tx;

    // This block's slice of K.
    int k_begin = blockIdx.z * k_per_split;
    int k_end   = min(K, k_begin + k_per_split);

    float sum = 0.0f;

    // Walk the K slice in 16-wide tiles.
    for (int k0 = k_begin; k0 < k_end; k0 += TILE_SPLITK) {
        // A tile: A[row][k0 + tx]
        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K && a_col < k_end)
                         ? A[row * K + a_col] : 0.0f;

        // B tile: B[k0 + ty][col]
        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && b_row < k_end && col < N)
                         ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SPLITK; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

void launch_matmul_splitk(const float* A, const float* B, float* C,
                          int M, int N, int K, int splits) {
    // C accumulates partials across z, so it must start at zero.
    cudaMemset(C, 0, (size_t)M * N * sizeof(float));

    int k_per_split = (K + splits - 1) / splits;

    dim3 block(TILE_SPLITK, TILE_SPLITK);
    dim3 grid((N + TILE_SPLITK - 1) / TILE_SPLITK,
              (M + TILE_SPLITK - 1) / TILE_SPLITK,
              splits);
    matmul_splitk_kernel<<<grid, block>>>(A, B, C, M, N, K, k_per_split);
}

# Step 14 - gemv_kernel
#include <cuda_runtime.h>

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int K) {
    int warp_id = threadIdx.x / 32;   // 0..3 within the block
    int lane    = threadIdx.x % 32;

    int row = blockIdx.x * 4 + warp_id;

    // Rows past M do nothing. The row is uniform across the warp, so this
    // early return is safe and keeps the shuffle reduction warp-uniform.
    if (row >= M) return;

    const float* row_ptr = A + (size_t)row * K;

    // Each lane strides through the row with stride 32 (coalesced: a warp
    // load covers 32 consecutive floats = 128 bytes).
    float sum = 0.0f;
    for (int k = lane; k < K; k += 32) {
        sum += row_ptr[k] * x[k];
    }

    // Butterfly reduction across the 32 lanes.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }

    // After the reduction every lane holds the full row sum; lane 0 stores.
    if (lane == 0) {
        y[row] = sum;
    }
}

void launch_gemv(const float* A, const float* x, float* y, int M, int K) {
    // 128 threads = 4 warps = 4 rows per block.
    dim3 block(128);
    dim3 grid((M + 3) / 4);
    gemv_kernel<<<grid, block>>>(A, x, y, M, K);
}

# Step 15 - matmul_bias_relu_kernel
#include <cuda_runtime.h>

constexpr int TILE_EPI = 16;

__global__ void matmul_bias_relu_kernel(const float* A, const float* B,
                                        const float* bias, float* C,
                                        int M, int N, int K) {
    __shared__ float As[TILE_EPI][TILE_EPI];
    __shared__ float Bs[TILE_EPI][TILE_EPI];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_EPI + ty;   // m index in C
    int col = blockIdx.x * TILE_EPI + tx;   // n index in C

    float sum = 0.0f;

    int num_tiles = (K + TILE_EPI - 1) / TILE_EPI;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_EPI;

        // A tile: A[row][k0 + tx]
        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // B tile: B[k0 + ty][col]
        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_EPI; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Fused epilogue: bias add + ReLU, applied while the value is in a register.
    if (row < M && col < N) {
        float v = sum + bias[col];
        C[row * N + col] = v > 0.0f ? v : 0.0f;
    }
}

void launch_matmul_bias_relu(const float* A, const float* B, const float* bias,
                             float* C, int M, int N, int K) {
    dim3 block(TILE_EPI, TILE_EPI);
    dim3 grid((N + TILE_EPI - 1) / TILE_EPI, (M + TILE_EPI - 1) / TILE_EPI);
    matmul_bias_relu_kernel<<<grid, block>>>(A, B, bias, C, M, N, K);
}

# Step 16 - matrix_addsub_kernel
#include <cuda_runtime.h>

__global__ void matrix_addsub_kernel(const float* X, int ldx,
                                     const float* Y, int ldy,
                                     float* Z, int ldz,
                                     int rows, int cols, float sign) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // column
    int r = blockIdx.y * blockDim.y + threadIdx.y;   // row

    if (r < rows && c < cols) {
        Z[r * ldz + c] = X[r * ldx + c] + sign * Y[r * ldy + c];
    }
}

void launch_matrix_addsub(const float* X, int ldx,
                          const float* Y, int ldy,
                          float* Z, int ldz,
                          int rows, int cols, float sign) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    matrix_addsub_kernel<<<grid, block>>>(X, ldx, Y, ldy, Z, ldz, rows, cols, sign);
}

# Step 17 - strassen_one_level
#include <cuda_runtime.h>

void strassen_one_level(const float* A, const float* B, float* C, int n) {
    int h = n / 2;

    // --- Quadrant pointers (leading dimension n) ---
    const float* A11 = A;
    const float* A12 = A + h;
    const float* A21 = A + (size_t)h * n;
    const float* A22 = A + (size_t)h * n + h;

    const float* B11 = B;
    const float* B12 = B + h;
    const float* B21 = B + (size_t)h * n;
    const float* B22 = B + (size_t)h * n + h;

    float* C11 = C;
    float* C12 = C + h;
    float* C21 = C + (size_t)h * n;
    float* C22 = C + (size_t)h * n + h;

    // --- Scratch buffers: two h x h operand buffers, seven h x h products ---
    size_t bytes = (size_t)h * h * sizeof(float);
    float *S1, *S2;
    float *P1, *P2, *P3, *P4, *P5, *P6, *P7;
    cudaMalloc(&S1, bytes);
    cudaMalloc(&S2, bytes);
    cudaMalloc(&P1, bytes);
    cudaMalloc(&P2, bytes);
    cudaMalloc(&P3, bytes);
    cudaMalloc(&P4, bytes);
    cudaMalloc(&P5, bytes);
    cudaMalloc(&P6, bytes);
    cudaMalloc(&P7, bytes);

    // Leading dimension of every scratch buffer is h (contiguous h x h).

    // --- P1 = (A11 + A22)(B11 + B22) ---
    launch_matrix_addsub(A11, n, A22, n, S1, h, h, h,  1.0f);
    launch_matrix_addsub(B11, n, B22, n, S2, h, h, h,  1.0f);
    launch_matmul_tiled(S1, S2, P1, h, h, h);

    // --- P2 = (A21 + A22) B11 ---
    launch_matrix_addsub(A21, n, A22, n, S1, h, h, h,  1.0f);
    launch_matrix_addsub(B11, n, B11, n, S2, h, h, h,  0.0f);   // copy B11
    launch_matmul_tiled(S1, S2, P2, h, h, h);

    // --- P3 = A11 (B12 - B22) ---
    launch_matrix_addsub(A11, n, A11, n, S1, h, h, h,  0.0f);   // copy A11
    launch_matrix_addsub(B12, n, B22, n, S2, h, h, h, -1.0f);
    launch_matmul_tiled(S1, S2, P3, h, h, h);

    // --- P4 = A22 (B21 - B11) ---
    launch_matrix_addsub(A22, n, A22, n, S1, h, h, h,  0.0f);   // copy A22
    launch_matrix_addsub(B21, n, B11, n, S2, h, h, h, -1.0f);
    launch_matmul_tiled(S1, S2, P4, h, h, h);

    // --- P5 = (A11 + A12) B22 ---
    launch_matrix_addsub(A11, n, A12, n, S1, h, h, h,  1.0f);
    launch_matrix_addsub(B22, n, B22, n, S2, h, h, h,  0.0f);   // copy B22
    launch_matmul_tiled(S1, S2, P5, h, h, h);

    // --- P6 = (A21 - A11)(B11 + B12) ---
    launch_matrix_addsub(A21, n, A11, n, S1, h, h, h, -1.0f);
    launch_matrix_addsub(B11, n, B12, n, S2, h, h, h,  1.0f);
    launch_matmul_tiled(S1, S2, P6, h, h, h);

    // --- P7 = (A12 - A22)(B21 + B22) ---
    launch_matrix_addsub(A12, n, A22, n, S1, h, h, h, -1.0f);
    launch_matrix_addsub(B21, n, B22, n, S2, h, h, h,  1.0f);
    launch_matmul_tiled(S1, S2, P7, h, h, h);

    // --- Assemble C quadrants in place ---
    // C11 = P1 + P4 - P5 + P7
    launch_matrix_addsub(P1, h, P4, h, C11, n, h, h,  1.0f);
    launch_matrix_addsub(C11, n, P5, h, C11, n, h, h, -1.0f);
    launch_matrix_addsub(C11, n, P7, h, C11, n, h, h,  1.0f);

    // C12 = P3 + P5
    launch_matrix_addsub(P3, h, P5, h, C12, n, h, h,  1.0f);

    // C21 = P2 + P4
    launch_matrix_addsub(P2, h, P4, h, C21, n, h, h,  1.0f);

    // C22 = P1 - P2 + P3 + P6
    launch_matrix_addsub(P1, h, P2, h, C22, n, h, h, -1.0f);
    launch_matrix_addsub(C22, n, P3, h, C22, n, h, h,  1.0f);
    launch_matrix_addsub(C22, n, P6, h, C22, n, h, h,  1.0f);

    // --- Free scratch ---
    cudaFree(S1); cudaFree(S2);
    cudaFree(P1); cudaFree(P2); cudaFree(P3); cudaFree(P4);
    cudaFree(P5); cudaFree(P6); cudaFree(P7);
}

# Step 18 - csr_spmm_kernel
#include <cuda_runtime.h>

__global__ void csr_spmm_kernel(const int* row_ptr, const int* col_idx,
                                const float* vals, const float* B,
                                float* C, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;   // output column
    int m = blockIdx.y * blockDim.y + threadIdx.y;   // output row

    if (m < M && n < N) {
        int start = row_ptr[m];
        int end   = row_ptr[m + 1];

        float acc = 0.0f;
        for (int j = start; j < end; ++j) {
            int k = col_idx[j];                      // column of A / row of B
            acc += vals[j] * B[k * N + n];
        }
        C[m * N + n] = acc;
    }
}

void launch_csr_spmm(const int* row_ptr, const int* col_idx, const float* vals,
                     const float* B, float* C, int M, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    csr_spmm_kernel<<<grid, block>>>(row_ptr, col_idx, vals, B, C, M, N);
}

# Step 19 - matmul_lower_triangular_kernel
#include <cuda_runtime.h>

constexpr int TILE_TRI = 16;

__global__ void matmul_lower_triangular_kernel(const float* A, const float* B,
                                               float* C, int M, int N) {
    __shared__ float As[TILE_TRI][TILE_TRI];
    __shared__ float Bs[TILE_TRI][TILE_TRI];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_TRI + ty;   // m index in C (and row of A)
    int col = blockIdx.x * TILE_TRI + tx;   // n index in C

    float sum = 0.0f;

    // A is M x M, so K = M. Only K-tiles t = 0 .. blockIdx.y can be nonzero:
    // block row `by` touches columns of A up to (by+1)*TILE_TRI - 1, and
    // tiles to the right of the diagonal block are entirely above the diagonal.
    int last_tile = blockIdx.y;
    for (int t = 0; t <= last_tile; ++t) {
        int k0 = t * TILE_TRI;

        // A tile: A[row][k0 + tx], zeroed above the diagonal.
        int a_col = k0 + tx;
        bool in_lower = (a_col <= row);     // diagonal condition
        As[ty][tx] = (row < M && a_col < M && in_lower)
                         ? A[row * M + a_col] : 0.0f;

        // B tile: B[k0 + ty][col]
        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < M && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_TRI; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_matmul_lower_triangular(const float* A, const float* B, float* C,
                                    int M, int N) {
    dim3 block(TILE_TRI, TILE_TRI);
    dim3 grid((N + TILE_TRI - 1) / TILE_TRI, (M + TILE_TRI - 1) / TILE_TRI);
    matmul_lower_triangular_kernel<<<grid, block>>>(A, B, C, M, N);
}

# Step 20 - matmul_dispatch (not yet solved)
# TODO: implement

