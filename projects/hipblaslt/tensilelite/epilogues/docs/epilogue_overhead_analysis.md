# Epilogue Overhead Analysis: subtile MXfp8 MT192x256 on gfx950

Problem: 8192×8192×1×8192, FP8 TN, MT192x256, MI16x16x128, DepthU=256, StreamK=3, PGR=1, PLR=1, DirectToLds=1, WGM=4.

| Kernel | time-us | TFlops | % of 2612T peak |
|---|---|---|---|
| No epilogue (F8BS) | 584 µs | 1880 | 72% |
| PRMS + dyn-quant (F8F8S PRMS_RA) | 934 µs | 1180 | 45% |
| **Gap** | **~350 µs** | | |

---

## Root causes

### What is provably identical (isolated to the epilogue)

Both kernels share the exact same GEMM core:

| Metric | No-epilogue | PRMS+dyn-quant |
|---|---|---|
| MFMA instruction count | 491 | 491 |
| VGPRs / SGPRs | 256 / 98 | 255 / 92 |
| VGPR/SGPR spills | 0 / 0 | 0 / 0 |
| LDS (group_segment) | 128 KiB | 128 KiB |
| Occupancy | 1 WG/CU | 1 WG/CU |

Occupancy reduction, register spilling, and LDS pressure are **not the cause**.

---

### Cause 1: Heavy, fully-exposed epilogue (~220–280 µs)

The PRMS kernel executes serially after the MFMA loop:

| Extra work | Instruction |
|---|---|
| Read residual tensor (bf16) | `buffer_load_short_d16` |
| fp32 RMS + dyn-quant chain | `v_mul/max/fma/add_f32` + cross-lane `ds_bpermute` reductions |
| Format conversion: D (fp8) | `v_cvt_pk_fp8_f32` |
| Format conversion: residualOut (bf16) | `v_cvt_pk_bf16_f32` |
| Write quantized fp8 D | `buffer_store_byte` |
| Write bf16 residualOut | `buffer_store_short` |
| Write MX scale tensor | `buffer_store_byte` |

That is 1 extra global read + 2 extra global stores + a full RMS/quant math chain, all fully exposed
after the loop. At 1-wave/SIMD (LDS-bound at 128 KiB) there is no second wave to hide latency.

Extra HBM traffic: ~128 MiB residual read + 128 MiB residualOut write + 2 MiB scale = **~258 MiB**.
At HBM bandwidth this accounts for only 35–90 µs; the math and conversion chain dominates.

Note: "Partial" in PRMS means this kernel writes a partial bf16 result for a *downstream* RMS kernel —
there is no cross-tile atomic reduction inside this kernel (0 `s_atomic`/`global_atomic` found in `.co`).

---

### Cause 2: StreamK abandoned — `StreamKForceDPOnly=1` (~60–100 µs)

- No-epilogue kernel: full StreamK — 18 `SK_Fixup`/`SK_Partials`/`GW_B1`/`GW_B0_MB` labels in disassembly.
- PRMS kernel: `SKFDPO1` — 0 such labels, degenerates to pure data-parallel.

The row-wise RMS normalization epilogue is incompatible with StreamK K-splitting: CUs that each
hold a K-partial cannot independently compute a row's RMS and then merge. So the kernel gives up
StreamK's tail load-balancing entirely.

For a 1376-tile / 256-CU grid (5.375 tiles/CU, non-integer) this creates real tail waste on the
compute phase itself.

---

### What is NOT the cause

- No register spilling (0 spills in `.co` amdhsa notes for both kernels).
- No cross-CU atomic serialization (0 `s_atomic`/`global_atomic` in the PRMS kernel).
- No occupancy change (same VGPR count and LDS footprint).
- The CSV `mem-*-bytes` columns in the benchmark logs are misleading: they count StreamK fp32
  workspace (~384 MiB) for the no-epilogue run but omit the epilogue tensors for the PRMS run.

---

## Approximate attribution of the 350 µs

| Cause | Estimated cost |
|---|---|
| Exposed epilogue memory + compute (dominant) | 220–280 µs |
| Loss of StreamK tail load-balancing | 60–100 µs |
| Raw extra HBM bandwidth | 35–90 µs (overlaps with first row) |

Compute floor (ideal MFMA only at 2612 TFlops peak): ~421 µs.
The no-epilogue kernel is already at 72% peak efficiency; the epilogue regression is essentially
all overhead layered on top.

---

## Evidence

| Artifact | Path |
|---|---|
| No-epilogue benchmark log | `out.log` |
| PRMS benchmark log | `prms-out.log` |
| PRMS YAML (epilogue params, SKFDPO=1 at line 131) | `Tensile/Tests/common/gemm/gfx950/gemm_partial_rms_residual_mxfp8quant_residualout_scaled_mxfp8_k1.yaml` |
| PRMS code object | `tensile-out-prms/1_BenchmarkProblems/Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_H_PRMS_RA_UserArgs_00/00_Final/caches/e8c541e606c8/source/library/gfx950/TensileLibrary_gfx950.co` |
