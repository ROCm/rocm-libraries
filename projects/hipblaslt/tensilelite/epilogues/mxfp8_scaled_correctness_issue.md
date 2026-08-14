# MX-Scaled FP8 PartialRMS Kernel Correctness Issue

**Date:** 2026-08-19
**GPU:** gfx950
**Branch:** `users/fabianmcg/epilogues`
**Discovered during:** Origami library-logic generation for epilogue benchmark YAMLs

---

## Affected kernels

Two benchmark YAMLs produce kernels that fail validation in every solution and
every tested problem size:

| YAML | Tensile type tag | FAILED / total |
|------|-----------------|----------------|
| `gemm_partial_rms_scaled_mxfp8_k1.yaml` | `F8BS_MXAE8B32_MXBE8B32_BH_PRMS` | 76 / 76 |
| `gemm_partial_rms_residual_scaled_mxfp8_k1.yaml` | `F8BS_MXAE8B32_MXBE8B32_BH_PRMS_RA` | 52 / 52 |

The four mxfp8-quant variants (`mxfp8quant_bf16`, `mxfp8quant_scaled_mxfp8`,
`residual_mxfp8quant_bf16`, `residual_mxfp8quant_scaled_mxfp8`) passed
validation with 0 failures.

---

## Failure signature

Every solution that was validated returned a large fraction of incorrect output
values. The error is not sporadic — it is consistent across all tested tile
sizes and problem sizes.

### Error rates

**`gemm_partial_rms_scaled_mxfp8_k1`** (5 problem sizes, 11 solutions each, 76 FAILED):

| Problem size (M, N, Batch, K) | Incorrect / total | Error rate |
|-------------------------------|-------------------|-----------|
| (320, 320, 1, 256) | 93 904 / 102 400 | 91.7 % |
| (512, 512, 1, 512) | 241 226 / 262 144 | 92.0 % |
| (1024, 1024, 1, 512) | 939 740 / 1 048 576 | 89.6 % |
| (256, 2048, 1, 256) | 493 061 / 524 288 | 94.0 % |
| (384, 256, 1, 256) | 90 533 / 98 304 | 92.1 % |
| (128, 512, 1, 256) | 60 317 / 65 536 | 92.0 % |
| (256, 256, 1, 512) | 240 239 / 262 144 | 91.6 % |

**`gemm_partial_rms_residual_scaled_mxfp8_k1`** (5 problem sizes, 52 FAILED):

| Problem size (M, N, Batch, K) | Incorrect / total | Error rate |
|-------------------------------|-------------------|-----------|
| (320, 320, 1, 256) | 95 726 / 102 400 | 93.5 % |
| (512, 512, 1, 512) | 230 739 / 262 144 | 88.0 % |
| (256, 2048, 1, 256) | 490 643 / 524 288 | 93.6 % |
| (128, 512, 1, 256) | 61 829 / 65 536 | 94.3 % |
| (256, 256, 1, 512) | 228 609 / 262 144 | 87.2 % |

### Tile sizes tested (representative, `scaled_mxfp8_k1`)

`MT64x256x256`, `MT128x128x256`, `MT128x256x256`, `MT192x256x256`,
`MT256x256x256`, `MT320x256x256` — all fail at the same ~90 % rate
regardless of tile shape or `EPS`/`PGR` sub-flags.

---

## What is correct vs. incorrect

The affected problem type is:

```
DataType:    F8 (fp8, OCP e4m3)
DataTypeA:   F8
DataTypeB:   F8
DestDataType: B (bf16)
ComputeDataType: S (fp32)
UsePartialRMS: True
PartialRMSResidualAdd: False / True (both variants)
PartialRMSQuant: False
DQuantType: None
UseScaleAB: 'B'          ← MX block scales on both A and B
MXBlockA/B: 32
DataTypeMXSA/B: E8 (OCP e8m0 exponent-only scale)
```

The distinguishing feature compared to the passing variants is `UseScaleAB: 'B'`
with `MXBlockA/B: 32` — i.e. MX block-scale factors applied to both A and B
inputs. The `mxfp8quant` variants (which pass) use `UseScaleAB: ''` and
produce an MX-scaled output instead.

---

## Hypotheses

The ~90 % error rate and the consistency across tile sizes suggest a systematic
data-layout or scale-application bug rather than a numerical precision issue.
Likely candidates:

1. **Scale layout mismatch** — the MX scale tensor for A or B is read with the
   wrong stride or transposition, poisoning most of the accumulation lanes.
   A similar layout issue was previously fixed for the dynquant scale path
   (see `79cccda6` — "fix mxfp8 dynquant scale layout for scaled GEMM
   chaining"); the same class of bug may affect the PRMS path.

2. **Scale broadcast dimension** — MX block size 32 along the K dimension
   requires one scale per 32 K elements. If the epilogue reads the scale with
   K/32 rounded incorrectly when K is not a multiple of 32, or applies the
   scale to the wrong dimension, ~7/8 or more of elements would be wrong.

3. **Missing or mismatched `UseScaleAB` branch in the PartialRMS epilogue
   emitter** — `Components/CustomSchedule.py` or the epilogue code path may
   not yet handle `UseScaleAB == 'B'` with `UsePartialRMS == True`
   simultaneously.

---

## Impact

- The Origami library-logic files for these two problem types were generated
  and are structurally valid, but the underlying kernels produce incorrect
  output. They should not be used in production until the correctness issue is
  resolved.
- The four `mxfp8quant` variants are unaffected and their logic files are
  correct.
- The `bf16` (non-MX) PartialRMS variants are unaffected.

---

## Reproduction

```bash
# From projects/hipblaslt
source ~/.tensile/bin/activate

PYTHONPATH=tensilelite \
python tensilelite/Tensile/Tensile.py \
  tensilelite/epilogues/yaml/gemm_partial_rms_scaled_mxfp8_k1.yaml \
  /tmp/repro_scaled_mxfp8 \
  2>&1 | tee /tmp/repro_scaled_mxfp8.log
```

Validation failures appear in Phase 1 (BenchmarkProblems) for every solution.

---

## Suggested next steps

1. Compare the PartialRMS epilogue emitter (`Components/CustomSchedule.py` or
   equivalent) against the working `mxfp8quant` path to find where
   `UseScaleAB` is handled — check for missing branches or wrong
   `IndexAssignmentsMXSA/B` usage.
2. Cross-reference the dynquant scale-layout fix (`79cccda6`) — the fix for
   scaled GEMM chaining may need a parallel change in the PRMS accumulation
   path.
3. Add a minimal unit test with a small (e.g. 64×64×32) problem to isolate
   whether the error is in the scale load, the accumulation, or the
   write-back.
