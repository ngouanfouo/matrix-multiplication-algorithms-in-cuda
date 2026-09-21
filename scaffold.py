"""
Matrix Multiplication Algorithms in CUDA scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

// scaffold.cu - benchmark every matrix multiplication algorithm of the project on a real GPU.
// The learner's kernels and launchers are concatenated above; main only drives them.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

static void fill_uniform(float* a, int n, unsigned seed) {
    unsigned s = seed * 2654435761u + 12345u;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        a[i] = ((s >> 8) % 2000) / 1000.0f - 1.0f;
    }
}

static void bench(const char* name, matmul_launch_fn launch, const float* dA, const float* dB, float* dC,
                  int M, int N, int K, const float* ref, std::vector<float>& host) {
    float ms = time_launch_ms(launch, dA, dB, dC, M, N, K, 10);
    cudaMemcpy(host.data(), dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("  %-18s %8.3f ms  %8.1f GFLOP/s  max|err| %.1e\n", name, ms, matmul_gflops(M, N, K, ms), max_abs_diff(host.data(), ref, M * N));
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %.0f GB/s peak bandwidth)\n", prop.name, prop.multiProcessorCount, 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    // ---- 1. The optimization ladder on a 512^3 product ----
    const int M = 512, N = 512, K = 512;
    std::vector<float> hA((size_t)M * K), hB((size_t)K * N), ref((size_t)M * N), host((size_t)M * N);
    fill_uniform(hA.data(), M * K, 1);
    fill_uniform(hB.data(), K * N, 2);
    matmul_cpu(hA.data(), hB.data(), ref.data(), M, N, K);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, (size_t)M * K * 4); cudaMalloc(&dB, (size_t)K * N * 4); cudaMalloc(&dC, (size_t)M * N * 4);
    cudaMemcpy(dA, hA.data(), (size_t)M * K * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), (size_t)K * N * 4, cudaMemcpyHostToDevice);
    printf("\n%dx%dx%d, fp32, 10 timed launches each:\n", M, N, K);
    bench("naive", launch_matmul_naive, dA, dB, dC, M, N, K, ref.data(), host);
    bench("coalesced", launch_matmul_coalesced, dA, dB, dC, M, N, K, ref.data(), host);
    bench("tiled 16x16", launch_matmul_tiled, dA, dB, dC, M, N, K, ref.data(), host);
    bench("tiled 1D regs", launch_matmul_tiled_1d, dA, dB, dC, M, N, K, ref.data(), host);
    bench("tiled 2D regs", launch_matmul_tiled_2d, dA, dB, dC, M, N, K, ref.data(), host);
    bench("vectorized", launch_matmul_vectorized, dA, dB, dC, M, N, K, ref.data(), host);
    bench("double buffered", launch_matmul_double_buffered, dA, dB, dC, M, N, K, ref.data(), host);

    // ---- 2. Strassen and the triangular shortcut on the same matrices ----
    {
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        strassen_one_level(dA, dB, dC, M);
        cudaEventRecord(s);
        for (int i = 0; i < 5; ++i) strassen_one_level(dA, dB, dC, M);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms, s, e); ms /= 5;
        cudaMemcpy(host.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        printf("  %-18s %8.3f ms  %8.1f GFLOP/s* max|err| %.1e   (*counted as 2MNK; 7 tiled products of 256^3 plus 18 additions)\n",
               "strassen 1 level", ms, matmul_gflops(M, N, K, ms), max_abs_diff(host.data(), ref.data(), M * N));
        std::vector<float> hL((size_t)M * M), refL((size_t)M * N);
        for (int r = 0; r < M; ++r) for (int c = 0; c < M; ++c) hL[(size_t)r * M + c] = (c <= r) ? hA[(size_t)r * K + c] : 0.0f;
        matmul_cpu(hL.data(), hB.data(), refL.data(), M, N, M);
        float* dL; cudaMalloc(&dL, (size_t)M * M * 4);
        cudaMemcpy(dL, hL.data(), (size_t)M * M * 4, cudaMemcpyHostToDevice);
        launch_matmul_lower_triangular(dL, dB, dC, M, N); cudaDeviceSynchronize();
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) launch_matmul_lower_triangular(dL, dB, dC, M, N);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e); ms /= 10;
        cudaMemcpy(host.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost);
        float ms_dense = time_launch_ms(launch_matmul_tiled, dL, dB, dC, M, N, M, 10);
        printf("  %-18s %8.3f ms  vs tiled on the same triangular A %8.3f ms  max|err| %.1e\n", "lower triangular", ms, ms_dense, max_abs_diff(host.data(), refL.data(), M * N));
        cudaFree(dL); cudaEventDestroy(s); cudaEventDestroy(e);
    }

    // ---- 3. Shapes that need their own kernel ----
    printf("\nshape-specific kernels:\n");
    {
        const int Mq = 64, Nq = 64, Kq = 4096;
        std::vector<float> qa((size_t)Mq * Kq), qb((size_t)Kq * Nq), qref((size_t)Mq * Nq), qhost((size_t)Mq * Nq);
        fill_uniform(qa.data(), Mq * Kq, 3); fill_uniform(qb.data(), Kq * Nq, 4);
        matmul_cpu(qa.data(), qb.data(), qref.data(), Mq, Nq, Kq);
        float *qA, *qB, *qC; cudaMalloc(&qA, (size_t)Mq * Kq * 4); cudaMalloc(&qB, (size_t)Kq * Nq * 4); cudaMalloc(&qC, (size_t)Mq * Nq * 4);
        cudaMemcpy(qA, qa.data(), (size_t)Mq * Kq * 4, cudaMemcpyHostToDevice); cudaMemcpy(qB, qb.data(), (size_t)Kq * Nq * 4, cudaMemcpyHostToDevice);
        float ms_plain = time_launch_ms(launch_matmul_double_buffered, qA, qB, qC, Mq, Nq, Kq, 10);
        cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
        launch_matmul_splitk(qA, qB, qC, Mq, Nq, Kq, 8); cudaDeviceSynchronize();
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) launch_matmul_splitk(qA, qB, qC, Mq, Nq, Kq, 8);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms_split; cudaEventElapsedTime(&ms_split, s, e); ms_split /= 10;
        cudaMemcpy(qhost.data(), qC, (size_t)Mq * Nq * 4, cudaMemcpyDeviceToHost);
        printf("  64x64x4096: double buffered %.3f ms (16 blocks) vs split-K x8 %.3f ms (128 blocks), max|err| %.1e\n", ms_plain, ms_split, max_abs_diff(qhost.data(), qref.data(), Mq * Nq));
        const int Mv = 4096, Kv = 4096;
        std::vector<float> va((size_t)Mv * Kv), vx(Kv), vref(Mv), vhost(Mv);
        fill_uniform(va.data(), Mv * Kv, 5); fill_uniform(vx.data(), Kv, 6);
        matmul_cpu(va.data(), vx.data(), vref.data(), Mv, 1, Kv);
        float *vA, *vX, *vY; cudaMalloc(&vA, (size_t)Mv * Kv * 4); cudaMalloc(&vX, Kv * 4); cudaMalloc(&vY, Mv * 4);
        cudaMemcpy(vA, va.data(), (size_t)Mv * Kv * 4, cudaMemcpyHostToDevice); cudaMemcpy(vX, vx.data(), Kv * 4, cudaMemcpyHostToDevice);
        float ms_gemm = time_launch_ms(launch_matmul_coalesced, vA, vX, vY, Mv, 1, Kv, 10);
        launch_gemv(vA, vX, vY, Mv, Kv); cudaDeviceSynchronize();
        cudaEventRecord(s);
        for (int i = 0; i < 10; ++i) launch_gemv(vA, vX, vY, Mv, Kv);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms_gemv; cudaEventElapsedTime(&ms_gemv, s, e); ms_gemv /= 10;
        cudaMemcpy(vhost.data(), vY, Mv * 4, cudaMemcpyDeviceToHost);
        double gbps = (double)Mv * Kv * 4 / (ms_gemv * 1e-3) / 1e9;
        printf("  4096x4096 times a vector: coalesced GEMM %.3f ms vs warp-per-row GEMV %.3f ms (%.0f GB/s), max|err| %.1e\n", ms_gemm, ms_gemv, gbps, max_abs_diff(vhost.data(), vref.data(), Mv));
        cudaFree(qA); cudaFree(qB); cudaFree(qC); cudaFree(vA); cudaFree(vX); cudaFree(vY);
        cudaEventDestroy(s); cudaEventDestroy(e);
    }

    // ---- 4. The dispatcher's decisions ----
    printf("\nmatmul_dispatch decisions:\n");
    const char* paths[4] = {"gemv", "split-K", "vectorized", "double buffered"};
    int shapes[4][3] = {{4096, 1, 4096}, {32, 32, 2048}, {512, 512, 512}, {33, 65, 17}};
    for (int i = 0; i < 4; ++i) {
        int m = shapes[i][0], n = shapes[i][1], k = shapes[i][2];
        std::vector<float> a((size_t)m * k), b((size_t)k * n), r((size_t)m * n), h((size_t)m * n);
        fill_uniform(a.data(), m * k, 7 + i); fill_uniform(b.data(), k * n, 8 + i);
        matmul_cpu(a.data(), b.data(), r.data(), m, n, k);
        float *xa, *xb, *xc; cudaMalloc(&xa, (size_t)m * k * 4); cudaMalloc(&xb, (size_t)k * n * 4); cudaMalloc(&xc, (size_t)m * n * 4);
        cudaMemcpy(xa, a.data(), (size_t)m * k * 4, cudaMemcpyHostToDevice); cudaMemcpy(xb, b.data(), (size_t)k * n * 4, cudaMemcpyHostToDevice);
        int path = matmul_dispatch(xa, xb, xc, m, n, k);
        cudaDeviceSynchronize();
        cudaMemcpy(h.data(), xc, (size_t)m * n * 4, cudaMemcpyDeviceToHost);
        printf("  %4d x %4d x %4d -> %-15s max|err| %.1e\n", m, n, k, paths[path], max_abs_diff(h.data(), r.data(), m * n));
        cudaFree(xa); cudaFree(xb); cudaFree(xc);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}

