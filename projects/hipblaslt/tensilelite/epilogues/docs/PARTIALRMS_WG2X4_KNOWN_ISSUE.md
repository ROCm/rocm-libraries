# Known issue: PartialRMS WG2x4 (8-wave) partialBuf miscompute

## Status
Open. The **WG2x4** wave-group configs (`MIWaveGroup = [2, 4]`, i.e. `wg_m=2` and
`wg_n=4`, an 8-wave / 512-thread workgroup) have been **removed** from
`gemm_partial_rms_residual_mxfp8quant_residualout_scaled_mxfp8_k1.yaml` because
they fail full CPU validation (`NumElementsToValidate: -1`). All other wave-group
tiles (WG 1x1, 2x1, 1x2, 2x2, 1x4) pass. Re-add the WG2x4 block once the root
cause is fixed.

## Impact
WG2x4 was the **top performer** (`MI=[16,16,128,1,1,4,2,2,4]`, MT128x128, PGR2 at
`8192^3`: ~1.14 PFlops). Removing it costs peak throughput for this kernel until
the bug is fixed. This is a **correctness** issue, not a perf regression in the
remaining configs.

## Affected configs (removed)
Groups 1 and 2 (MT1=128), the WG 2x4 block, all PGR in {0,1,2}:

```
- [16, 16, 128, 1, 1,  2,  2, 2, 4]  # MT0=64   MT1=128
- [16, 16, 128, 1, 1,  4,  2, 2, 4]  # MT0=128  MT1=128
- [16, 16, 128, 1, 1,  6,  2, 2, 4]  # MT0=192  MT1=128
- [16, 16, 128, 1, 1,  8,  2, 2, 4]  # MT0=256  MT1=128
```

Group 3 (MT1=256) is unaffected — its configs are WG2x2 (`[...,2,2]`), not WG2x4.

## Symptom
- Only the `partialBuf` output (the per-token PartialRMS Sigma-x^2 partial sums)
  fails. The GEMM `D` (fp8) and `residualOut` (bf16 of `H = GEMM + residual`) both
  validate correctly.
- The error is a **deterministic ~1.3% overcount** of Sigma-x^2 on the real tokens
  (e.g. GPU `4.0457e9` vs reference `3.9920e9`), consistent across runs.
- It is **data-dependent**: with constant/uniform inputs (exact dot products) the
  same WG2x4 kernel **passes**; only random/varying GEMM values trigger the failure.
- It is **not** near-threshold noise: it fails on every run with the same
  per-token error ratio.

## Discriminator (isolated experimentally)
The failure requires `wg_m>1` **and** `wg_n=4` (an 8-wave workgroup), independent
of `mma_n`:

| MIWaveGroup | waves | result |
|-------------|-------|--------|
| 2x1, 2x2    | 2, 4  | PASS   |
| 1x4         | 4     | PASS   |
| **2x4**     | **8** | **FAIL** |

`wg_m=2, wg_n=2, mma_n=2` passes and `wg_m=2, wg_n=4, mma_n=4` fails, so `mma_n`
is not the axis — the 8-wave workgroup is.

## Ruled out
- **Not** the recent bf16 store/load vectorization commit
  (`067aeda87ec`): forcing the scalar fallback paths
  (`useWideResidual=False`, `useWideBf16Store=False`) still fails identically.
- **Not** fp8-specific: with constant fp8 A/B the WG2x4 kernel passes. The bf16
  PartialRMS tests (`gemm_partial_rms_residualout_bf16_*`) simply never contained
  a WG2x4 config, so the bug was latent, not fp8-related.
- **Not** the residual-add: it fails with residual zeroed (`--init-residual 0`).
- **Not** an LDS overflow: gfx950 has 160 KB LDS; the reduction scratch is ~4-8 KB.

## Investigation notes for the fix
- The Subtile PartialRMS epilogue emitter (`Tensile/Components/Subtile/
  SubtilePartialRMSEmit.py`) generates **byte-for-byte identical** assembly for the
  passing 4-wave `MT64x64` and the failing 8-wave `MT64x128` kernels — both the
  fused-pass Sigma-x^2 squaring and the entire cross-wave LDS reduction
  (`_crossWaveReduceFree0`, address formula `writeAddr = Serial * laneSlotBytes`,
  sibling reads, all three `s_barrier`s with correct `s_waitcnt`). The only
  runtime difference is the wave count (4 vs 8).
- Because `residualOut` (= bf16 of the same `H` used for Sigma-x^2) validates
  exactly, `H` is correct to bf16 precision; a ~1.3% Sigma-x^2 error is
  mathematically inconsistent with squaring that same `H`, which points to the
  8-wave reduction picking up wrong/extra data at runtime rather than a logic
  error in the emitter's arithmetic.
- **Attempted fix that did NOT resolve it**: moving the cross-wave reduction
  scratch to a dedicated LDS region above the GEMM main-loop region (so it cannot
  alias the GEMM/persistent-loop LDS) plus an explicit LDS fence at reduction
  entry. This still left WG2x4 failing, so LDS aliasing with the GEMM tile region
  is not the (complete) root cause. That change was reverted.
- Remaining leads to investigate: 8-wave (2 wavefronts/SIMD on gfx950) LDS
  coherence / `s_barrier` semantics for this workgroup size; StreamK persistent-
  loop LDS-reuse boundary; or per-wave partial corruption specific to the 8-wave
  register allocation. The GEMM accumulator itself is ruled out (H is correct).

## Reproduction
Minimal (~1 min): a benchmark YAML with a WG2x4 config plus WG2x2/WG1x4 controls
at problem size `[4096, 4, 1, 4096]`, `NumElementsToValidate: -1`, run with:

```
source ~/.tensile/bin/activate && LD_LIBRARY_PATH=/opt/rocm/lib \
  Tensile/bin/Tensile <repro>.yaml tensile-out > repro.log 2>&1
grep -aoE ',(PASSED|FAILED),' repro.log | sort | uniq -c
```

Per-element device-vs-reference values are visible via the client flags
`ValidationMaxToPrint` and `ValidationPrintValids` (the failing tensor is
`[partialBuf]`).
