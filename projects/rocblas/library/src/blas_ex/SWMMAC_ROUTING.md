# SWMMAC Routing Guide — rocBLAS Integration

## Activation

```bash
# Activate SWMMAC INT4 path (default: off, uses Tensile GEMM)
export ROCBLAS_SWMMAC_INT4=1

# Optional: select precision family
export ROCBLAS_SWMMAC_PRECISION=int4   # int4 | int8 | fp16 | bf16 | fp8 | mxfp4
```

## Build Integration

`rocblas_swmmac.cpp` is compiled as part of `librocblas.so` via
`library/src/CMakeLists.txt` → `rocblas_ex_source_no_tensile`.

No separate compilation step needed. Already added to build in this commit.

## Architecture Dispatch

```
rocblas_gemm_ex() or rocblas_gemm_batched_ex()
    │
    ├── ROCBLAS_SWMMAC_INT4=0 (default)
    │       └── standard Tensile GEMM path
    │
    └── ROCBLAS_SWMMAC_INT4=1
            └── rocblas_swmmac_launch()
                    │
                    ├── at=160 (INT4)  → StaggeredPipeline 16-chain kernel
                    ├── at=162 (INT8)  → 8-chain kernel
                    ├── at=150 (FP16)  → 8-chain outer-product kernel
                    ├── at=168 (BF16)  → 8-chain DOE-calibrated kernel
                    ├── at=171 (FP8)   → 2-chain kernel
                    ├── at=172 (BF8)   → 2-chain kernel
                    └── at=170 (MXFP4) → INT4 kernel (scale in conv layer)
```

## Precision Details

| at | Precision | ops/inst | Chain | VGPR | HW_SCALE |
|----|-----------|----------|-------|------|----------|
| 160 | INT4 (K=64) | 32768 | 16 | 19 | 1.0 |
| 162 | INT8 (K=32) | 16384 | 8 | 14 | 1.0 |
| 150 | FP16 (K=32) | 16384 | 8 | 22 | 0.25 |
| 168 | BF16 (K=32) | 16384 | 8 | 22 | 1/29.266 |
| 171 | FP8 (K=32) | 16384 | 2 | 14 | 1.0 |
| 170 | MXFP4 (K=64) | 32768 | 16 | 19 | UE8M0 |

## Hardware Defect Workaround

All kernels implement **wave-level** cooperative work-claiming via
`__builtin_amdgcn_readfirstlane` to work around RDNA4 HWXDL Silent Drop
hardware defect (see `test/swmmac/DISCOVERY.md` for details).

## Performance (gfx1200, 32 CUs, 2780 MHz)

| Precision | Peak (TOPS/TFLOPS) | vs Tensile Baseline |
|-----------|-------------------|---------------------|
| INT4 | 4326 | +456% |
| INT8 | — | — |
| FP16 | — | — |
