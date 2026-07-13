# HSTU Attention Backward — Rocke case study (gfx950)

A replayable optimization case study for the Rocke HSTU attention **backward**
kernels (`dv`, `dk`, `dq`) on gfx950 (MI355X, CDNA4), benchmarked against the
FlyDSL reference on AITER branch `dlejeune/flydsl_hsta_bwd`.

- Math: [`ALGORITHM.md`](ALGORITHM.md)
- Kernel architecture + future optimizations: [`ARCHITECTURE.md`](ARCHITECTURE.md)
- Kernels: `library/kernels/common/hstu_attention_bwd.py`
- Shared math helpers: `platform/python/rocke/helpers/hstu_attention.py`
- Benchmark/parity harness:
  `library/benchmarks/gfx950/attention/benchmark_hstu_bwd_flydsl_rocke.py`

## Goal

Implement the HSTU backward in Rocke and beat FlyDSL (and eventually AOTriton)
on gfx950, with exact numerical parity as a hard gate at every step.

## Result summary

Optimization arc on shape `bf16, H=4, head_dim=128, hidden_dim=128, batch=120,
max_seq_len=512` (tuned CSV index 6), parity vs FlyDSL exact throughout
(`max_abs ~1e-6..1e-8`, `mean_abs ~1e-10..1e-13`):

| stage | total ms | vs FlyDSL |
|-------|---------:|----------:|
| scalar baseline (warp-distributed) | 770.5 | 0.001× |
| all dot products → MFMA GEMMs | 17.5 | 0.055× |
| + causal streamed-range limiting | 10.2 | 0.093× |
| + tiled multi-wave CTAs, swizzled-LDS staging, GEMM1-frag reuse across d-chunks | 1.62 | 0.586× |
| + direct C→A fragment reuse (drop `lds_gate` + 1 barrier), `waves_per_eu` plumbing | 1.54 | 0.62× |

Net: **~500× faster than the scalar baseline**, gap to FlyDSL closed from ~1000×
to ~1.3–1.6×.

On shape index 9 (`H=8`, with the tuned `waves_per_eu`) the split gets even
closer and **dq overtakes FlyDSL**:

| kernel | flydsl ms | rocke ms | speedup (fly/rocke) |
|--------|----------:|---------:|--------------------:|
| dv | 0.305 | 0.677 | 0.45× |
| dk | 0.711 | 0.888 | 0.80× |
| dq | 0.781 | 0.708 | **1.10×** |

Caveat: the `waves_per_eu` values come from the gfx942-tuned CSV; `dv`'s `wpe=1`
hurts on gfx950 (dv wants high occupancy), which is why dv regresses on shape 9.
gfx950-specific `wpe` tuning is tracked in [`ARCHITECTURE.md`](ARCHITECTURE.md) §8.

## What made the difference

1. **Scalar → MFMA (44×).** The first MFMA revision still recomputed the QK /
   dO·V dot products per lane; replacing them with real `mfma_f32_16x16x16`
   GEMMs (score, dA, and the output reduction) removed all `fmuladd`-based dot
   products. Verified via IR: `fmuladd = 0`, 32 (dv) / 48 (dk,dq) MFMAs per iter.
2. **Causal range limiting (1.7×).** Skip fully-masked streamed tiles at tile
   granularity (pure-causal only).
3. **Tiling + LDS staging + fragment reuse (6.3×).** The dominant win. Enlarge
   the owned tile to `block_m × out_dim` with multi-wave CTAs so each streamed
   tile is loaded **once** and amortized across many 16×16 sub-tiles; stage the
   two streamed operands through swizzled LDS; compute the score/gate fragment
   once per (own, stream) pair and reuse it across all `out_dim/16` output
   chunks. See [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. **Direct C→A fragment reuse + occupancy hint.** Emitting GEMM1 as
   `C[m=stream, n=own]` lets the MFMA C→A transpose feed GEMM2's `A[own, stream]`
   directly — removing the `lds_gate` round-trip and one barrier (3→2
   barriers/iter, matching FlyDSL).

## Reproduce

Environment (gfx950 / MI355X). FlyDSL requires its pinned `flydsl==0.2.2`; the
project venv at `/workspace/FlyDSL_CKTile_conv_comparison/FlyDSL/.venv` was used
(the AITER branch rejects other FlyDSL versions). The AITER branch is checked out
at `~/aiter-flydsl_hsta_bwd`.

```bash
# List the tuned-CSV shapes:
$FLYDSL_VENV/bin/python \
  library/benchmarks/gfx950/attention/benchmark_hstu_bwd_flydsl_rocke.py \
  --aiter-root ~/aiter-flydsl_hsta_bwd --list-shapes

# Parity + benchmark one shape (dense-exact lengths, tiled Rocke path):
$FLYDSL_VENV/bin/python \
  library/benchmarks/gfx950/attention/benchmark_hstu_bwd_flydsl_rocke.py \
  --aiter-root ~/aiter-flydsl_hsta_bwd --shape-index 6 --warmup 2 --iters 5 --dense-exact

# Compare against the simple one-16x16-tile-per-CTA MFMA path:
#   add --no-tiled
```

The harness reads shapes/tile-configs from
`aiter/ops/flydsl/hstu_attention_bwd_tuned.csv`, generates HSTU inputs the same
way as the FlyDSL tests, builds both backends, prints per-kernel and total
timings plus `max_abs`/`mean_abs` parity.

Codegen (no GPU) regression tests:

```bash
PYTHONPATH=../../../../platform/python:. python3 -m pytest \
  tests/test_attention_builds.py -k hstu -q
```

## Correctness gate

Every optimization was validated by comparing the three Rocke gradients directly
against the FlyDSL kernel outputs on-device before accepting the change. The
gate held at `max_abs ~1e-6..1e-8` throughout (bf16 fast-math SiLU recompute
noise), consistent with the FlyDSL test tolerances (`dv` 2e-2, `dk`/`dq` 3e-2
against the fp32 torch oracle).

## Provenance

The FlyDSL ISA gap analysis that guided the tiling/LDS/pipeline work was produced
by reverse-engineering the generated gfx950 ISA of all three FlyDSL backward
kernels (single streamed loop, resident-operand hoisting, `buffer_load_lds`
prefetch with `(8,3)` swizzle, register-prefetch of the second operand, 2
barriers/iter, deferred scalar stores). The remaining gap and the concrete next
steps are enumerated in [`ARCHITECTURE.md`](ARCHITECTURE.md) §8.
