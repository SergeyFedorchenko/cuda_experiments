# CUDA Matrix Multiply (Naive vs Tiled)

Minimal CUDA C project with:
- `gemm_naive`: straightforward i-k-j loop on GPU.
- `gemm_tiled`: shared-memory tiling with `BLOCK x BLOCK` tiles.

## Build
```bash
make ARCH=sm_86 BLOCK=32    # set your GPU arch; e.g., sm_75, sm_86, sm_90
