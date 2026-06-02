# AICK-1346: Unified MX FLATMM GroupedGemm Implementation

## Context

Qwen3 and DeepSeek-V3 training for the Advance AI day requires a grouped GEMM
interface for MX data types (FP8, FP6, FP4). The existing `MXFlatmmKernel` handles
single-problem MX FLATMM on both gfx950 and gfx1250, and `GroupedFlatmmKernel`
handles grouped (non-MX) FLATMM via a persistent tile-loop. The gap: there is no
kernel that combines **MX FLATMM pipelines with grouped dispatch**. This plan fills
that gap by creating `GroupedMXFlatmmKernel`.

A separate `MxGroupedGemmKernel` already exists for standard GEMM pipelines
(CompTDMV1/V2), but it uses `UniversalGemmKernelArgs` and cannot drive FLATMM
pipelines which require the flat B tensor layout and FLATMM-specific scale windows.

## Approach: Follow the GroupedFlatmmKernel pattern

The existing `GroupedFlatmmKernel` (`include/ck_tile/ops/flatmm/kernel/grouped_flatmm_kernel.hpp`)
provides the exact dispatch pattern we need:

1. Persistent tile-loop iterates over groups
2. Per-group `FlatmmKernelArgs` is constructed on-device from arrays of pointers
3. Delegates to `FlatmmKernel::operator(impl_kargs, block_linear_idx)`

The new kernel does the same but inherits from `MXFlatmmKernel` instead of
`FlatmmKernel`, so the per-group dispatch calls `MXFlatmmKernel::operator()` which
handles MX scale windows and the MX FLATMM pipeline.

## Files to create

### 1. `include/ck_tile/ops/flatmm/kernel/grouped_mx_flatmm_kernel.hpp`

New kernel struct `GroupedMXFlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>`:

- **Inherits from** `MXFlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>`
- **Host args**: Reuse the existing `GroupedFlatmmHostArgs<ScaleM, ScaleN, NumDTensor>` from
  `grouped_flatmm_kernel.hpp:19-77` — it already has arrays of per-group pointers including
  `ScaleM*` and `ScaleN*` arrays
- **`operator()`**: Persistent tile-loop that mirrors `GroupedFlatmmKernel::operator()` at
  line 350-388, but constructs `FlatmmKernelArgs` with scale pointers from `kargs.scale_m[group_idx]`
  / `kargs.scale_n[group_idx]` and delegates to `MXFlatmmKernel::operator(impl_kargs, block_linear_idx)`
- **`GridSize()`**: Persistent grid size computation (same pattern as `GroupedFlatmmKernel::GridSize`)
- **`GetName()`**: Returns `"mx_grouped_flatmm_..."` 
- **`MakeKernelArgs()`**: Pass-through (same as `GroupedFlatmmKernel::MakeKernelArgs`)

The dispatch loop pattern (from `GroupedFlatmmKernel` line 350-388):
```cpp
for(group_idx = 0; group_idx < group_count; ++group_idx) {
    group_block_cnt = TilePartitioner::GridSize(M[group_idx], N[group_idx]);
    while(block_linear_idx < group_block_cnt) {
        // Build per-group FlatmmKernelArgs with scale_m[group_idx], scale_n[group_idx]
        underlying_kernel(impl_kargs, block_linear_idx);
        block_linear_idx += total_block_cnt;
    }
    block_linear_idx -= group_block_cnt;
}
```

### 2. Update `include/ck_tile/ops/flatmm.hpp`

Add `#include "ck_tile/ops/flatmm/kernel/grouped_mx_flatmm_kernel.hpp"`.

### 3. Add test files to existing `test/ck_tile/grouped_gemm_mx/` directory

New files (prefix `test_grouped_gemm_mx_flatmm`):

- **`test_grouped_gemm_mx_flatmm_util.hpp`** — Test fixture extending `TestMXFlatmmBase`
  from `test/ck_tile/flatmm/test_mx_flatmm_base.hpp`. Key additions:
  - Generate multiple (M, N, K) problems per test case
  - Allocate per-group A, B_shuffled, scale_A, scale_B, E buffers
  - Pre-shuffle weights and scales per group (reuse `preShuffleWeight` and
    `preShuffleScale` from `test_mx_flatmm_base.hpp`)
  - Run CPU reference per group, concatenate results, compare
  - **Configurable tensor initialization** via a `constexpr` template parameter
    (matching the `init_method` pattern from `example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc:103-122`):
    - `init_method == 0`: random values — `FillUniformDistribution{0.0, 1.0}` for A,
      `{-0.5, 0.5}` for B, `{-2.0, 2.0}` for scales
    - `init_method == 1`: constant values — A=2.0, B=0.5, scale_a=0.5, scale_b=2.0
    - Selected at compile time via a constexpr parameter in the test fixture, so
      individual test types can specify their init method in the type list
  - Use `mx_flatmm_init_proxy_t<T>` for fp6 initialization (proxy through
    `pk_fp6_t` then memcpy to `pk_fp6x16_t`), following the pattern at
    `run_mx_flatmm.inc:8-10,93-96,130+`
- **`test_grouped_gemm_mx_flatmm.cpp`** — Test type definitions covering:
  - FP8xFP8, FP4xFP4, FP6xFP6, FP8xFP4, FP4xFP8 (matching existing MX FLATMM test coverage)
  - Both GFX950 and GFX1250 traits (selected at compile time via arch traits)
- **`test_grouped_gemm_mx_flatmm_ut_cases.inc`** — Test cases with various group counts
  and problem sizes
- Update existing **`CMakeLists.txt`** in `grouped_gemm_mx/` to add the new test target
  with `GPU_TARGETS` gating for gfx950/gfx1250

## Key design decisions

1. **Persistent dispatch only** (no binary search variant). The `GroupedFlatmmKernel`
   already uses persistent dispatch exclusively, and the FLATMM workloads are
   large enough that persistent kernels provide better GPU utilization.

2. **Reuse `GroupedFlatmmHostArgs`** rather than creating a new host args struct.
   It already has `ScaleM*` and `ScaleN*` arrays for per-group scale pointers.

3. **Single kernel class** works for both gfx950 and gfx1250 because the pipeline
   type is a template parameter. Architecture selection happens at the
   traits/instantiation level, not in the kernel itself.
   - `MXFlatmmPipelineAGmemBGmemCRegV1` — tested on **both gfx950 and gfx1250**
   - `WeightPreshufflePipelineAGmemBGmemCRegTDM` — tested on **gfx1250 only** (TDM is gfx1250-specific)

4. **No split-K initially** (`k_batch=1`). The existing `GroupedFlatmmKernel` asserts
   `k_batch == 1`, and the persistent kernel doesn't support split-K.

## Patterns to reuse (with file paths)

| What | From | Notes |
|------|------|-------|
| Persistent tile-loop dispatch | `grouped_flatmm_kernel.hpp:350-388` | Core dispatch pattern |
| `GroupedFlatmmHostArgs` struct | `grouped_flatmm_kernel.hpp:19-77` | Host args with scale arrays |
| `MXFlatmmKernel::operator()` | `mx_flatmm_kernel.hpp:389-425` | Per-group GEMM execution |
| `preShuffleWeight()` | `test/ck_tile/flatmm/test_mx_flatmm_base.hpp:22-60` | B tensor pre-shuffle |
| `preShuffleScale()` | `test/ck_tile/flatmm/mx_flatmm_arch_traits.hpp` | Scale pre-shuffle |
| `TestMXFlatmmBase` fixture | `test/ck_tile/flatmm/test_mx_flatmm_base.hpp:66+` | Test base class |
| CMakeLists arch gating | `test/ck_tile/flatmm/CMakeLists.txt` | Build system pattern |

## Verification

1. **Unit tests**: Run on gfx950 and gfx1250 (or simulator) with FP8/FP6/FP4 data types,
   varying group counts (1, 2, 4, 8) and problem sizes
2. **Correctness**: Compare against per-group CPU reference (`reference_mx_gemm`) for each
   group independently
3. **Build**: Verify compilation with both arch traits via `/ck-build`
4. **Smoke test**: Use existing `smoke_test_mx.sh` pattern if applicable
