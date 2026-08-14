# MX-Scaled FP8 Subtile GEMM — MIWaveGroup[0]=4 (WG4x1) Base-GEMM Correctness Bug

**Date:** 2026-08-20 (re-verified)
**GPU:** gfx950
**Status:** Root cause confirmed to be in the BASE MX-scaled fp8 subtile GEMM
(not the PartialRMS epilogue). Guard `_validateSubtileMXWaveGroup` has been
REMOVED from `Solution.py`; the 14 failures are therefore currently exposed.

## Summary

The base MX-scaled fp8 subtile GEMM produces **incorrect accumulators** when
`MIWaveGroup[0] == 4` (the asymmetric WG4x1 wave group). This is a defect in the
base GEMM, **independent of the PartialRMS fused epilogue**. The
`gemm_partial_rms_scaled_mxfp8_k1.yaml` failures (the `MIWT4_16 / WG64_4_1`
solution, all sizes, PGR=0 and PGR=1) are downstream symptoms: PartialRMS
consumes the already-wrong accumulator, so both the `[d]` output and the
`[partialBuf]` reduction come out wrong.

An earlier note claimed this had been "refined" to a PartialRMS-only /
mxfp8-only epilogue bug. That refinement was wrong. Fresh isolation on
2026-08-20 reproduced the failure in the base GEMM with a clean numerical
signature (see below); the PartialRMS emitter is a faithful consumer.

## Empirical evidence (2026-08-20, guard removed)

Runs via `Tensile/bin/Tensile` with `NumElementsToValidate: -1`.

1. **mxfp8 PartialRMS** (`gemm_partial_rms_scaled_mxfp8_k1.yaml`): 127 PASSED /
   14 FAILED. The only failing solution is `MT256x256x256 / MIWT4_16 /
   WG64_4_1` (MIWaveGroup[4,1]), all 7 sizes x PGR=0/1.

2. **mxfp8 BASE GEMM, no PartialRMS** (`/tmp/base_wg4x1_nopartialrms.yaml`,
   tiles `[16,16,128,1,1,4,16,4,1]` vs control `[16,16,128,1,1,8,8,2,2]`):
   the WG4x1 tile FAILS 5/6 (256x256/K256/PGR1 borderline-passes); the
   MIWaveGroup[2,2] control passes 6/6. The `[d]` output (no gamma, no
   reduction) is wrong by a **(n-1)/n scaling** where n = K / DepthU:
   - K=256 (n=2 DepthU iters): device ≈ 0.48 x reference (≈ 1/2)
   - K=512 (n=4 DepthU iters): device ≈ 0.74 x reference (≈ 3/4)

   This is the fingerprint of **one DepthU K-iteration's contribution being
   dropped/corrupted**. It cannot originate in the epilogue (which runs once,
   after the K-loop), and it scales with the K-loop trip count.

3. **bf16 PartialRMS** (`gemm_partial_rms_bf16_k1.yaml`, contains many WG4x1
   tiles incl. `MT384x256 [6,16] WG4x1`, `MT512x128 [8,8] WG4x1`,
   `MT64x512 [1,32] WG4x1`): 486 PASSED / 0 FAILED. bf16 has no MX scale and
   uses MI_K=32; the WG4x1 PartialRMS epilogue path is correct.

## The discriminator is wg_m=4, NOT mma_m=4

All mxfp8 solutions grouped by (MIWaveTile, WorkGroup) with wg_m = WG0/16:

| MIWaveTile | WorkGroup | MIWaveGroup (wg_m,wg_n) | verdict |
|-----------|-----------|-------------------------|---------|
| MIWT4_16  | WG64_4_1  | (4,1)                   | FAIL    |
| MIWT4_2   | WG32_8_1  | (2,2)                   | PASS    |
| MIWT4_2   | WG16_16_1 | (1,4)                   | PASS    |
| MIWT4_4   | WG32_8_1  | (2,2)                   | PASS    |
| MIWT4_4   | WG16_16_1 | (1,4)                   | PASS    |
| MIWT4_8   | WG32_8_1  | (2,2)                   | PASS    |
| MIWT8_8   | WG32_8_1  | (2,2)                   | PASS    |
| MIWT16_4  | WG16_16_1 | (1,4)                   | PASS    |
| MIWT10_8/6_8/8_6/8_10 | WG32_8_1 | (2,2)          | PASS    |

Tiles with `mma_m = MIWaveTile[0] = 4` (MIWT4_2, MIWT4_4, MIWT4_8) PASS when
`wg_m in {1,2}`. Only `wg_m = 4` fails. So the discriminator is
`MIWaveGroup[0] == 4`, independent of `mma_m` and independent of PartialRMS.

## Where the real fix lives

The bug is in the base MX-scaled fp8 K-loop for `wg_m=4` — one DepthU
iteration's scale/data does not land. Both PGR=0 (single-partition) and PGR=1
fail, so it is not solely a prefetch race. Prime suspects, in order:

- `Tensile/Components/Subtile/LogicalScheduler.py` — subtile K-loop scheduling:
  partition sizing, per-iteration scale/data GR->LR placement, and wait-count
  logic for `wg_m=4`.
- `Tensile/Components/Subtile/SubtileScaleEmit.py` —
  `_applyScaleWavePartitionLROffset` and the scale LDS partition/stride: for
  MXSA, `partitionIndex = waveId % wg_m` with
  `totalScaleBytes = (lrGlobalSubtileGrid[0] // wg_m) * lrGlobalSubtileGrid[1]
  * lrSubtileSize`. Verify all 4 M-wave partitions are populated and read for
  every DepthU iteration.
- `Tensile/Components/Subtile/Kernel.py` `emitMfmaCode` — the per-K-step scale
  group/selector (`scaleGroupA`, `sAsel`) for the second MI_K step (mmak=1).

## Reproduction

Base GEMM (no PartialRMS) isolation — fastest, removes the epilogue variable:

```bash
cd /home/fmoracor/rocm-libraries/projects/hipblaslt/tensilelite
source ~/.tensile/bin/activate
LD_LIBRARY_PATH=/opt/rocm/lib PYTHONPATH=. \
python Tensile/bin/Tensile /tmp/base_wg4x1_nopartialrms.yaml /tmp/base_wg4x1_out \
  > /tmp/base_wg4x1_out.log 2>&1
grep -aE ',(PASSED|FAILED),' /tmp/base_wg4x1_out.log
```

Minimal repro YAML: plain MX-scaled fp8 TN GEMM (`DataType: F8`,
`DestDataType: b`, `ComputeDataType: s`, `MXBlockA/B: 32`, `TransposeA: True`,
`UseSubtileImpl: True`, `StreamK: 3`, `StreamKForceDPOnly: 1`, `DepthU: 256`,
`ScheduleIterAlg: 3`, `PrefetchGlobalRead: [0,1]`) forking:

```
- [16, 16, 128, 1, 1,  4, 16, 4, 1]  # MT256x256, MIWaveGroup[4,1]  (FAILS)
- [16, 16, 128, 1, 1,  8,  8, 2, 2]  # MT256x256, MIWaveGroup[2,2]  (control, PASSES)
```

over sizes `[256,256,1,256]`, `[512,512,1,512]`, `[1024,1024,1,512]`.

The PartialRMS-level repro is
`Tensile/Tests/common/gemm/gfx950/gemm_partial_rms_scaled_mxfp8_k1.yaml`.

## Guard status

`_validateSubtileMXWaveGroup` was REMOVED from
`Tensile/SolutionStructs/Solution.py` on 2026-08-20 (both the function and its
call site next to `_validateSubtileGRKPartition`). With the guard gone the 14
broken configs are re-exposed and will fail validation / can leak into
generated logic.

Decision pending: either (a) re-instate a guard rejecting MX-scaled subtile
solutions with `MIWaveGroup[0] > 2` — and broaden it beyond `UsePartialRMS`,
since the base GEMM is affected regardless of the epilogue — or (b) fix the
base-GEMM `wg_m=4` K-accumulation per the suspects above and then keep the
guard removed.
