# Matrix Multiplication Algorithms in CUDA

Write general matrix multiplication in CUDA a dozen different ways and measure every one on a real GPU. Climb the optimization ladder from a naive kernel through coalescing, shared-memory tiling, register blocking, float4 loads and double buffering, cover batched and split-K shapes, then finish with a shape-aware dispatcher and a GFLOP/s table.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** matmul_cpu
- [x] **2.** max_abs_diff
- [x] **3.** matmul_naive_kernel
- [x] **4.** matmul_coalesced_kernel
- [x] **5.** time_launch_ms
- [x] **6.** matmul_tiled_kernel
- [x] **7.** matmul_tiled_1d_kernel
- [x] **8.** matmul_tiled_2d_kernel
- [x] **9.** matmul_vectorized_kernel
- [x] **10.** matmul_double_buffered_kernel
- [x] **11.** matmul_nt_kernel
- [x] **12.** matmul_batched_kernel
- [x] **13.** matmul_splitk_kernel
- [x] **14.** gemv_kernel
- [x] **15.** matmul_bias_relu_kernel
- [x] **16.** matrix_addsub_kernel
- [x] **17.** strassen_one_level
- [ ] **18.** csr_spmm_kernel
- [ ] **19.** matmul_lower_triangular_kernel
- [ ] **20.** matmul_dispatch

---

Built on Deep-ML.
