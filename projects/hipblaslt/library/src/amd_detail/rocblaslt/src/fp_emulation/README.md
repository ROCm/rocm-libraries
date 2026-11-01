<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

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
| `include/tables.hpp` | Compile-time CRT constant tables: the moduli, `UInt256` big-integer helper, and the derived `qpi`, `P_dd`, `inv_P`, `crt_capacity_bits`, `neg_mod`, `inv_mod`, `limb_weight`, `pad`, etc. |
| `include/kernel_common.hpp` | Shared device helpers (`floor_log2`, `warp_reduce_max_abs`, `block_reduce_max`, `adp_atomicMaxF`), the integer residue helpers used by stages 4 and 6 (`split_int88`, `int8_residue`, `balanced_residue_i32`), and kernel tuning constants (`OZ2_PRELIM_*`, `OZ2_SCALE_*`). |
| `include/prelim_kernels.hpp` | Stage 1 — preliminary shift + INT8 extraction kernels (`accu_prelim_A/B_T/N_kernel`). |
| `include/shift_refine_kernels.hpp` | Stage 3 — shift refinement + ADP kernels (`refine_sftA_partial_kernel`, `col_max_kernel`, `adp_reduce_A/B_kernel`, `refine_sftA_apply_kernel`, `refine_sftB_kernel`). |
| `include/scale_kernels.hpp` | Stage 4 — multi-modulus scaling kernels (`scale_A/B_T/N_kernel`). |
| `include/reconstruct_kernels.hpp` | Stage 6 — CRT accumulation + finalization kernels (`chunk_accum_kernel`, `accum_finalize_kernel`). |

The kernel headers are header-only because the kernels are templates
instantiated by the host-side launch helpers in `emulation.cpp`; they all live
in `namespace FixedPointEmulation` and are `__global__ static` (internal
linkage) to keep the symbols hidden.

## Adding support for a new device

Emulation is enabled only on devices listed in `hw_params_by_pci_id` in
`get_perf_model_params()` (`emulation.cpp`). There is no other device gate:
adding a row enables emulation and the performance model for that device.

1. **PCI device ID.** Read it with
   `hipDeviceGetAttribute(&id, hipDeviceAttributePciChipId, dev)`. Only the
   low 16 bits are used. Add one row per SKU variant.
2. **Measure on the target device:**
   - `BW`: achievable HBM bandwidth in bytes/s, from a large streaming copy.
   - `F64`, `F32`: peak native DGEMM and SGEMM throughput in flop/s, from
     large square `hipblaslt-bench` runs.
   - `I8`: peak INT8 GEMM throughput in op/s (INT8 inputs, INT32 output,
     `TN` layout, matching what the emulation path calls).
   - Overheads in seconds:
     - `L_kernel`: empty kernel launch.
     - `L_matmul`: launch overhead of a small `hipblasLtMatmul`.
     - `L_memset`: `hipMemsetAsync`.
     - `L_sync`: `hipStreamSynchronize` round trip.
     - `L_host`: per-call host overhead of the emulation path.
3. **Compute the table row** (field order `ai_fp64, ai_fp32, ai_int8,
   ovh_kernel, ovh_matmul, ovh_memset, ovh_sync, ovh_host`):
   - `ai_fp64 = F64 / BW`, `ai_fp32 = F32 / BW`, `ai_int8 = I8 / BW`.
   - `ovh_X = L_X * BW` for each overhead.

   Two significant digits are enough. Only ratios between fields affect the
   emulate-vs-native and split decisions, so all values share one scale.
4. **Kernel efficiencies.** The `eff_*` factors in `PerfModelKernelEffs` are
   shared by all devices and were calibrated on gfx950. On a different
   architecture, re-check them: each factor is the phase's raw model time
   (bandwidth-limited time, without the efficiency divisor) divided by its
   measured time from the `t_*_ms` columns of the profiling CSV.
5. **Validate.** Run a size sweep with `HIPBLASLT_EMULATION_PROFILE=<file>`
   and `HIPBLAS_EMULATION_STRATEGY=eager`, and time the same sizes with native
   GEMM. `pred_total_rel` should track the measured emulated/native time
   ratio, especially near 1.0, where the decision flips.
6. **Update docs and tests.** Add the device to "Supported hardware" above
   and in `docs/how-to/fp-emulation.rst`. The emulation gtests detect support
   through the same table, so run
   `hipblaslt-test --gtest_filter='*FixedPointEmulation*'` on the new device.

## Profiling CSV

With `HIPBLASLT_EMULATION_PROFILE=<file>`, each emulated call appends one row:

- `t_*_ms` and `cpu_*_ms`: measured GPU-event and host wall-clock times in
  milliseconds, summed over all sub-GEMMs when the problem is split.
- `pred_*_rel`: performance-model predictions, as a fraction of the
  predicted native GEMM cost. `pred_total_rel < 1` means emulation is
  predicted to be faster than native.

## Build

This code is built as a standalone `hipblaslt-fixed-point-emulation` **OBJECT**
library (see `CMakeLists.txt`) and linked `PRIVATE` into both the `hipblaslt`
shared library and the `hipblaslt-test` harness.  This keeps the internal
symbols hidden from the release `.so` while still letting the tests call the
internal entry points directly.  `emulation.cpp` must be compiled as HIP.

## Configuration

### Requesting emulation

Emulation is requested through the **compute type** of the matmul descriptor:

| Compute type | Behaviour |
| --- | --- |
| `HIPBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT` | FP64 GEMM requested to use fixed-point emulation (A/B/C/D and scale type `HIP_R_64F`). |
| `HIPBLAS_COMPUTE_32F_EMULATED_FIXEDPOINT` | FP32 GEMM requested to use fixed-point emulation (A/B/C/D and scale type `HIP_R_32F`). |
| `HIPBLAS_COMPUTE_64F` / `HIPBLAS_COMPUTE_32F` | Native GEMM, unless `HIPBLAS_EMULATE_DOUBLE_PRECISION=1` / `HIPBLAS_EMULATE_SINGLE_PRECISION=1` is set ("transparent mode"). |

An emulated compute type *requests* emulation. Whether a call is actually
emulated still depends on the strategy, device support, `batch_count = 1`,
the epilogue, and the input data; otherwise it runs as a native GEMM.

Internally the compute types map to `rocblaslt_compute_f64_emulated_fixedpoint`
(200) and `rocblaslt_compute_f32_emulated_fixedpoint` (201), whose values must
match the `hipblasComputeType_t` enumerators.

### Environment variables (process-wide defaults)

| Variable | Values | Meaning |
| --- | --- | --- |
| `HIPBLAS_EMULATE_DOUBLE_PRECISION` | `0` / `1` | Transparently emulate plain `HIPBLAS_COMPUTE_64F` GEMMs. |
| `HIPBLAS_EMULATE_SINGLE_PRECISION` | `0` / `1` | Transparently emulate plain `HIPBLAS_COMPUTE_32F` GEMMs. |
| `HIPBLAS_EMULATION_STRATEGY` | `performant` (default) / `eager` | `performant` emulates only when the performance model predicts a speedup; `eager` always emulates when supported. |
| `HIPBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK` | hex bitmask, default `0x3` | Bit 0 = detect Inf, bit 1 = detect NaN. `0` skips detection (unsupported values silently corrupt the output). |
| `HIPBLAS_FIXEDPOINT_EMULATION_FP64_MANTISSA_BIT_COUNT` | `1`..`52`, default `52` | ADP precision target for FP64. Fewer bits mean fewer moduli and faster execution. Takes precedence over the generic variable below. |
| `HIPBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT` | `1`..`52`, default `52` | Same as the FP64 variable, used only when that one is unset; FP64 only. CUDA-compatible name (`CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT`), but sets the ADP target rather than a fixed bit count. |
| `HIPBLAS_FIXEDPOINT_EMULATION_FP32_MANTISSA_BIT_COUNT` | `1`..`23`, default `23` | ADP precision target for FP32. |
| `HIPBLASLT_EMULATION_NUM_MODULI` | `2`..`20` | **Testing only.** Overrides the ADP-selected moduli count; accuracy is not guaranteed below the ADP choice. |
| `HIPBLASLT_EMULATION_PROFILE` | file path | Append per-call profiling CSV rows to this file. |

### Per-matmul settings (emulation descriptor)

Per-matmul settings live in a separate `hipblasLtEmulationDesc_t`, attached to
the matmul descriptor with `HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR`.
Attaching it does **not** enable emulation; it only tunes GEMMs that are already
emulated through the compute type or the env vars.

```c
hipblasLtEmulationDesc_t emul;
hipblasLtEmulationDescCreate(&emul);

int32_t strategy = HIPBLAS_EMULATION_STRATEGY_EAGER;
hipblasLtEmulationDescSetAttribute(
    emul, HIPBLASLT_EMULATION_DESC_STRATEGY, &strategy, sizeof(strategy));

int32_t bits = 26; /* ~1e-8 relative accuracy */
hipblasLtEmulationDescSetAttribute(
    emul, HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT, &bits, sizeof(bits));

hipblasLtMatmulDescSetAttribute(
    matmul_desc, HIPBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR, &emul, sizeof(emul));

/* ... hipblasLtMatmul(...) ... */

hipblasLtEmulationDescDestroy(emul); /* only after all matmuls using matmul_desc */
```

Emulation descriptor attributes (`hipblasLtEmulationDescAttributes_t`):

| Attribute | Type | Meaning |
| --- | --- | --- |
| `HIPBLASLT_EMULATION_DESC_STRATEGY` | `int32_t` (`hipblasEmulationStrategy_t`) | `DEFAULT` (0) = inherit env var, `PERFORMANT` (1), `EAGER` (2). |
| `HIPBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT` | `int32_t` bitmask | Bit 0 = Inf, bit 1 = NaN; same layout as the env var: `0` = none, `1` = Inf, `2` = NaN, `3` = both, `0xFFFF` (default) = both. Values match CUDA's `cudaEmulationSpecialValuesSupport`. Stored internally as a `uint32_t`. |
| `HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT` | `int32_t` | ADP precision target in mantissa bits; `0` = inherit env var. |
| `HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL` | `int32_t` | Not used at the moment; the value is stored and returned by Get, but has no effect. Default `0`. |
| `HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET` | `int32_t` | Not used at the moment; the value is stored and returned by Get, but has no effect. Default `0`. |
| `HIPBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER` | `int32_t*` | Not used at the moment; the pointer is stored as-is (never dereferenced) and returned by Get, but has no effect. Default `NULL`. |

The matmul descriptor stores only a pointer to the emulation descriptor, so the
emulation descriptor must outlive every matmul call that uses it. Set the
attribute to `NULL` to detach it.

### Setting precedence

Emulation descriptor attributes > environment variables > built-in defaults.
A `DEFAULT` / `0` value on an emulation descriptor attribute falls back to the
corresponding environment variable. See `fixedPointEmulationDecision()` in
`emulation.cpp` for the full resolution logic.

### Workspace

Query the required workspace with `hipblasLtEmulationWorkspaceSize()` and pass
it as `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`. It returns `0` when emulation
is not applicable (unsupported device or type, or `batch_count != 1`).

See [`docs/how-to/fp-emulation.rst`](../../../../../../docs/how-to/fp-emulation.rst)
for the complete user-facing guide and `hipblaslt-bench` examples.
