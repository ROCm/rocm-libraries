# FP32/FP64 Fixed-Point Emulation

This directory implements FP32/FP64 GEMM emulation via **Ozaki Scheme II**
(Ozaki, Uchino, Imamura, [arXiv:2504.08009](https://arxiv.org/abs/2504.08009)),
using **INT8 tensor cores** and the **Chinese Remainder Theorem (CRT)**.  On
hardware where INT8 peak throughput greatly exceeds FP64/FP32 throughput, this
can outperform native DGEMM/SGEMM while producing results that are roughly as
accurate as — and sometimes more accurate than — the native path.

- **Supported hardware:** AMD Instinct MI350X and MI355X (gfx950).
- **Supported types:** FP64 (`HIP_R_64F`) and FP32 (`HIP_R_32F`).
- **Restrictions:** non-batched (`batch_count = 1`) only; default epilogue only.

For the complete user-facing guide (env vars, per-matmul descriptor attributes,
workspace sizing, `hipblaslt-bench` examples), see
[`docs/how-to/fp-emulation.rst`](../../../../../../docs/how-to/fp-emulation.rst).

## Algorithm pipeline

Each emulated GEMM runs through the following GPU stages:

1. **Preliminary shift + INT8 extraction** — compute a per-row/column shift
   from the input magnitudes and extract the high INT8 slice of each operand.
2. **Preliminary INT8 GEMM** — a single INT8 tensor-core GEMM whose result
   drives the shift refinement.
3. **Shift refinement + ADP** — refine the per-row/column shifts and run
   Automatic Dynamic Precision (ADP), which selects the minimum number of CRT
   moduli (INT8 GEMM passes) needed to reach the target accuracy for the given
   input's dynamic range.
4. **Multi-modulus scaling** — extract the per-modulus INT8 residue slices for
   the batched GEMM passes.
5. **Batched INT8 GEMMs** — one INT8 GEMM per modulus (chunked to fit the
   workspace).
6. **CRT reconstruction** — accumulate the per-modulus INT32 results (via a
   double-double / TwoSum accumulator) and finalize to the output type.

If the input contains Inf/NaN (with detection enabled) or its dynamic range
exceeds the CRT capacity for the maximum moduli count, the call falls back to
native GEMM automatically.

## File layout

| File | Contents |
| --- | --- |
| `emulation.cpp` | Host orchestration: performance model, workspace/budget logic, kernel launch helpers, the recursive `emulated_gemm_impl`, the public `fp64EmulatedGemm` / `fp32EmulatedGemm` entry points, and env-var / descriptor-attribute resolution. Compiled as HIP. |
| `include/emulation.hpp` | Public host-side interface (intentionally free of HIP device types so it can be included from plain C++ units). Declares the public entry points, the decision/settings structs, and the env-var parsers. |
| `include/tables.hpp` | Compile-time CRT constant tables: the moduli, `UInt256` big-integer helper, and the derived `qpi`, `P_dd`, `inv_P`, `log2P`, `neg_mod`, `inv_mod`, `pad`, etc. |
| `include/kernel_common.hpp` | Shared device helpers (`floor_log2`, `warp_reduce_max_abs`, `block_reduce_max`, `adp_atomicMaxF`) and kernel tuning constants (`OZ2_PRELIM_*`, `OZ2_SCALE_*`). |
| `include/prelim_kernels.hpp` | Stage 1 — preliminary shift + INT8 extraction kernels (`accu_prelim_A/B_T/N_kernel`). |
| `include/shift_refine_kernels.hpp` | Stage 3 — shift refinement + ADP kernels (`refine_sftA_partial_kernel`, `col_max_kernel`, `adp_reduce_A/B_kernel`, `refine_sftA_apply_kernel`, `refine_sftB_kernel`). |
| `include/scale_kernels.hpp` | Stage 4 — multi-modulus scaling kernels (`scale_A/B_T/N_kernel`). |
| `include/reconstruct_kernels.hpp` | Stage 6 — CRT accumulation + finalization kernels (`chunk_accum_kernel`, `accum_finalize_kernel`). |

The kernel headers are header-only because the kernels are templates
instantiated by the host-side launch helpers in `emulation.cpp`; they all live
in `namespace FixedPointEmulation` and are `__global__ static` (internal
linkage) to keep the symbols hidden.

## Build

This code is built as a standalone `hipblaslt-fixed-point-emulation` **OBJECT**
library (see `CMakeLists.txt`) and linked `PRIVATE` into both the `hipblaslt`
shared library and the `hipblaslt-test` harness.  This keeps the internal
symbols hidden from the release `.so` while still letting the tests call the
internal entry points directly.  `emulation.cpp` must be compiled as HIP.

## Configuration

Emulation can be enabled and tuned two ways:

**1. Environment variables** (process-wide default):

```bash
export HIPBLASLT_EMULATE_DOUBLE_PRECISION=1   # enable FP64 emulation
export HIPBLASLT_EMULATE_SINGLE_PRECISION=1   # enable FP32 emulation
```

**2. Per-matmul descriptor attributes** (per call; take precedence over the
env vars for that specific matmul):

```c
int32_t v = emulation_enabled ? 1 : 0;
hipblasLtMatmulDescSetAttribute(
    desc, HIPBLASLT_MATMUL_DESC_EMULATION_ENABLED_EXT, &v, sizeof(v));
```

Other per-call knobs (strategy, precision target, special-values mask) are also
available as `HIPBLASLT_MATMUL_DESC_EMULATION_*_EXT` attributes.

See [`docs/how-to/fp-emulation.rst`](../../../../../../docs/how-to/fp-emulation.rst)
for the full list of settings, the setting-precedence rules, and workspace
requirements.
