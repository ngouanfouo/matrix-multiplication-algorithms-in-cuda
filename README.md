# Matrix Multiplication Algorithms in CUDA

Write general matrix multiplication in CUDA a dozen different ways and measure every one on a real GPU. Climb the optimization ladder from a naive kernel through coalescing, shared-memory tiling, register blocking, float4 loads and double buffering, cover batched and split-K shapes, then finish with a shape-aware dispatcher and a GFLOP/s table.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** matmul_cpu
- [ ] **2.** max_abs_diff
- [ ] **3.** matmul_naive_kernel
- [ ] **4.** matmul_coalesced_kernel
- [ ] **5.** time_launch_ms
- [ ] **6.** matmul_tiled_kernel
- [ ] **7.** matmul_tiled_1d_kernel
- [ ] **8.** matmul_tiled_2d_kernel
- [ ] **9.** matmul_vectorized_kernel
- [ ] **10.** matmul_double_buffered_kernel
- [ ] **11.** matmul_nt_kernel
- [ ] **12.** matmul_batched_kernel
- [ ] **13.** matmul_splitk_kernel
- [ ] **14.** gemv_kernel
- [ ] **15.** matmul_bias_relu_kernel
- [ ] **16.** matrix_addsub_kernel
- [ ] **17.** strassen_one_level
- [ ] **18.** csr_spmm_kernel
- [ ] **19.** matmul_lower_triangular_kernel
- [ ] **20.** matmul_dispatch

---

Built on Deep-ML.
