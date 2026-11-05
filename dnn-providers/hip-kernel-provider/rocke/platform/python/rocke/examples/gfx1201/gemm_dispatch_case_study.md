<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Size-aware FP16 GEMM dispatch — gfx1201 case study

How the gfx1201 FP16 RCR `universal_gemm` dispatcher came to choose a tile by
output area and reduction length, instead of one 64x32x16 tile for every shape.

Per [`platform/AGENTS.md`](../../../../AGENTS.md) §Compliance this document
records **methodology, levers and relative ratios only**: every number below is
a ratio between configurations measured in the same process on the same
device; no absolute latency or throughput appears here. Measured scope: gfx1201,
fp16 inputs and output with fp32 accumulation, RCR layout (`C = A * B^T`), the
`mem` pipeline, the shapes listed under [Results](#results). Baseline: the
`rdna_wmma_default` candidate (64x32x16, 2x1 waves) as shipped before this
change, on the same build. Provenance: rocKE `develop` at `90ef5ad9489` plus this
change, kernels compiled with the ROCm 10.2 SDK (LLVM 24) through comgr.

## Table of contents

- [What was investigated](#what-was-investigated)
- [The instrument](#the-instrument)
- [Finding 1: one tile size for every shape leaves large GEMMs slow](#finding-1-one-tile-size-for-every-shape-leaves-large-gemms-slow)
- [Finding 2: workgroup count decides the crossover](#finding-2-workgroup-count-decides-the-crossover)
- [Finding 3: a 32-deep K step helps large outputs](#finding-3-a-32-deep-k-step-helps-large-outputs)
- [Finding 4: the bottleneck is the K loop, not the tile](#finding-4-the-bottleneck-is-the-k-loop-not-the-tile)
- [The rule](#the-rule)
- [Rejected levers](#rejected-levers)
- [Correctness gate](#correctness-gate)
- [Results](#results)
- [Caveats](#caveats)
- [Replay](#replay)
- [Keep / revert / defer decisions](#keep--revert--defer-decisions)

## What was investigated

Before this change, the gfx1201 FP16 dispatcher returned the same 64x32x16
tile for every problem, with a 32x32x16 tile as the only fallback. The
question was whether larger tiles that the same `universal_gemm` emitter
already builds for gfx1201 should be selected for some shapes, and by what
rule, without changing the emitter, the pipeline, or any other architecture.

## The instrument

The development sweep timed eight 16x16x16-WMMA tile/warp layouts at K steps
of 16 and 32, plus the existing default and fallback, on 50 shapes: squares
from 128 to 4096, rectangular M and N from 128 to 2048, short and long K,
dimensions not divisible by 128, a K not divisible by 32, and public model
projection and MoE-expert shapes. Every configuration was verified before its
timing counted, the shape order was shuffled per repeat, and two independent
runs were compared.

This document's [Results](#results) come from the in-tree dispatcher sweep,
[`rocke.benchmark.gemm.fp16_rcr_sweep`](../../benchmark/gemm/fp16_rcr_sweep.py),
which builds every registered candidate a shape admits, launches each through
`rocke.run_manifest` with `--verify`, and records one timing per variant. It was
run three times in separate processes; the tables report medians of the three.

## Finding 1: one tile size for every shape leaves large GEMMs slow

From an output area of about one million elements upward, 128-wide tiles were
consistently faster than the 64x32x16 default, and the gap grew with the
problem size. On the smallest and skinniest shapes the default and the 32x32
fallback stayed competitive, and the larger tiles were slower.

## Finding 2: workgroup count decides the crossover

The crossover tracked the number of output tiles against the device's compute
units rather than any single dimension: a 128x128 tile needs enough output tiles
to occupy the device before its higher reuse pays off. Output area (`M*N`) is
therefore the gate, exposed as the derived request dimension
`output_elements`, so one rule covers square, wide and tall shapes alike.

## Finding 3: a 32-deep K step helps large outputs

Above about two million output elements the K=32 tiles (128x64x32 and
64x128x32) beat the K=16 128x128 tile; between one and two million the
128x128x16 tile with 4x1 waves was the more reliable choice. A K=32 tile needs
`K % 32 == 0`; large shapes whose K is only a multiple of 16 keep the
128x128x16 tile through a separate lower-priority candidate rather than falling
back to the default. Very short reductions (below K = 256) gained little from
any larger tile and stay on the default.

## Finding 4: the bottleneck is the K loop, not the tile

Thread traces of the `mem` pipeline show each K iteration waiting on the
current tile's global loads before any matrix work: load, wait, store to LDS,
barrier, LDS read, barrier, WMMA. Larger tiles help because each wait is
amortized over more WMMA instructions, not because the wait goes away. A
software-pipelined WMMA K loop that keeps the next tile's loads in flight
during the current tile's math is the larger follow-up; it is out of scope
here.

## The rule

Four gfx1201-only candidates, each a fixed spec (`mem` pipeline, `intrawave`
scheduler, `default` epilogue, wave32, no padding). The band values are named
constants beside `_CDNA_MFMA_FP16` in
[`dispatch/gemm/fp16_rcr.py`](../../dispatch/gemm/fp16_rcr.py):

| priority | candidate | tile | waves | shape gate |
|---|---|---|---|---|
| 10 | `gfx1201_wmma_128x128x16` | 128x128x16 | 4x1 | `_GFX1201_MEDIUM_SHAPES` |
| 11 | `gfx1201_wmma_128x64x32` | 128x64x32 | 2x1 | `_GFX1201_K32_SHAPES` |
| 12 | `gfx1201_wmma_64x128x32` | 64x128x32 | 1x2 | `_GFX1201_K32_SHAPES` |
| 13 | `gfx1201_wmma_128x128x16_large` | 128x128x16 | 4x1 | `_GFX1201_LARGE_SHAPES` |
| 14 | `rdna_wmma_default` (unchanged) | 64x32x16 | 2x1 | none |
| 20 | `rdna_wmma_32x32` (unchanged) | 32x32x16 | 2x2 | none |

Exact tile divisibility stays in each candidate's support predicate, so a shape
a tile cannot divide falls through to the next candidate. Existing pins keep
their specs and hashes, and gfx11-generic, gfx1200 and every CDNA target keep
their previous selections.

## Rejected levers

- **128x256 tiles** (2x4, 1x8, 8x1 waves): competitive at some large shapes but
  never consistently ahead of the smaller K=32 tiles; not registered.
- **K step 64:** slower than K=32 on every shape tried.
- **Issuing a tile's global loads before its LDS stores** (an emitter change,
  prototyped and measured): it changed the compiled loop of only one
  configuration, helped some shapes and slowed others, and did not address the
  wait in [Finding 4](#finding-4-the-bottleneck-is-the-k-loop-not-the-tile).
  Reverted; superseded by the pipelined-loop follow-up.
- **Using the 32x32 fallback for small shapes by rule:** its advantage at the
  smallest shapes was within launch-overhead noise in the development sweep
  and inconsistent across runs; small shapes keep the existing default.

## Correctness gate

Every timed point was verified on the device by `rocke.run_manifest --verify`:
the full `M x N` output against a NumPy fp32 reference rounded to fp16, inputs
small signed integers from a fixed seed, tolerance `1e-2` absolute plus `1e-2`
relative; a nonzero bad count fails the run. In the three replay runs every
variant reported zero bad elements. Separately, the old and new selections were
compared with signed non-integer and cancellation-heavy inputs against a CPU
float64 reference on 84 shapes, including shapes not used to set the bands; old
and new outputs were bit-identical on every shape. The in-tree GPU test
[`test_fp16_rcr_runtime.py`](../../../../tests/dispatch/dispatch_tests/gemm/test_fp16_rcr_runtime.py)
launches each new candidate on gfx1201 and checks its full output.

## Results

Median of three in-tree sweep runs on one gfx1201 device. "vs default" is the
default candidate's time over the selected candidate's time on the same shape
(higher is faster). "best / selected" is how far the selected candidate is from
the fastest candidate the shape admits (1.00 means it is the fastest).

| shape (M x N x K) | selected candidate | vs default | best / selected |
|---|---|---|---|
| 512 x 512 x 512 | `rdna_wmma_default` | 1.00x | 1.21x (`rdna_wmma_32x32`) |
| 128 x 1280 x 2560 (MoE expert) | `rdna_wmma_default` | 1.00x | 1.13x (`rdna_wmma_32x32`) |
| 1024 x 1024 x 128 | `rdna_wmma_default` | 1.00x | 1.00x |
| 768 x 768 x 768 | `rdna_wmma_default` | 1.00x | 1.00x |
| 1024 x 1024 x 1024 | `gfx1201_wmma_128x128x16` | 2.04x | 1.00x |
| 1216 x 1024 x 1024 (M not divisible by 128) | `gfx1201_wmma_64x128x32` | 1.62x | 1.00x |
| 1536 x 1536 x 1536 | `gfx1201_wmma_128x64x32` | 2.86x | 1.00x |
| 2048 x 2048 x 2048 | `gfx1201_wmma_128x64x32` | 3.06x | 1.02x (`gfx1201_wmma_64x128x32`) |
| 2048 x 2048 x 1040 (K not divisible by 32) | `gfx1201_wmma_128x128x16_large` | 1.87x | 1.00x |
| 2048 x 2560 x 6144 (projection) | `gfx1201_wmma_128x64x32` | 3.05x | 1.00x |
| 2048 x 6144 x 2560 (projection) | `gfx1201_wmma_128x64x32` | 3.15x | 1.00x |
| 4096 x 4096 x 4096 | `gfx1201_wmma_128x64x32` | 3.85x | 1.00x |

Faster at these measured configurations; shapes below the bands are unchanged
by design. The development sweep and the separate 84-shape check agreed with
these factors in direction and order of magnitude.

## Caveats

- One gfx1201 device. gfx11-generic (gfx1151) and gfx1200 were not measured
  and keep the previous selection.
- `rocke.run_manifest` launches through the per-call runtime path, which adds
  host overhead that dominates the smallest kernels: run-to-run spread on the
  512^3, 768^3 and K = 128 points reached tens of percent, so the first four
  rows above are not evidence either way. Shapes in the bands spread by under
  5 percent across runs.
- Warm-cache timing: the same input buffers are reused across launches.
- The two losses in the table (the 32x32 fallback at 512^3 and at the MoE
  expert shape) are left as they are; see [Rejected levers](#rejected-levers).
- The bands encode the device's compute-unit count implicitly through output
  area; a gfx12 device with a different count would need its own measurement.

## Replay

From `platform/`, on a gfx1201 device, with the ROCm toolchain on `PATH`:

```bash
export PYTHONPATH=python
for run in 1 2 3; do
  python -m rocke.benchmark.gemm.fp16_rcr_sweep --arch gfx1201 \
    --output-dir sweep_run$run --compile --run --parallel 4 \
    --warmup-iters 20 --timed-iters 100 \
    --shape 512,512,512:sq512:verify \
    --shape 128,1280,2560:expert_up:verify \
    --shape 1024,1024,128:k128:verify \
    --shape 768,768,768:sq768:verify \
    --shape 1024,1024,1024:sq1024:verify \
    --shape 1216,1024,1024:m_not_128:verify \
    --shape 1536,1536,1536:sq1536:verify \
    --shape 2048,2048,2048:sq2048:verify \
    --shape 2048,2048,1040:k_not_32:verify \
    --shape 2048,2560,6144:proj_out:verify \
    --shape 2048,6144,2560:proj_in:verify \
    --shape 4096,4096,4096:sq4096:verify
done
```

Each run writes `gemm_fp16_rcr_sweep.json` and `.csv` with one record per
(shape, admitted candidate), including the verify line; the selected candidate
for a shape is `dispatch_gemm_fp16(GemmRequest(M, N, K, arch="gfx1201"))`. Stop
other GPU work during the runs.

## Keep / revert / defer decisions

- **Keep:** the four size-gated gfx1201 candidates and their bands; the derived
  `output_elements` dimension; the unchanged default and fallback for shapes
  below the bands.
- **Revert:** issuing a tile's loads before its stores as a standalone emitter
  change.
- **Defer:** a software-pipelined WMMA K loop (the measured bottleneck);
  measuring gfx1151 and gfx1200 before extending the bands to them; revisiting
  the small-shape fallback with a launch path that excludes host overhead.
