# matrix-multiplication-algorithms-in-cuda
Write general matrix multiplication in CUDA a dozen different ways and measure every one on a real GPU. Climb the optimization ladder from a naive kernel through coalescing, shared-memory tiling, register blocking, float4 loads and double buffering, cover batched and split-K shapes, then finish with a shape-aware dispatcher and a GFLOP/s table.
