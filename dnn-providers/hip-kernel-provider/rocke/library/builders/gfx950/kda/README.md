# gfx950 chunkwise KDA

This directory contains the spec-driven builders and correctness harnesses for
chunkwise gated delta-rule linear-attention prefill:

- `kda_chunk_prep.py` builds the six state-independent per-chunk tiles.
- `kda_chunk_split.py` materializes those tiles, then runs the serial state scan.
- `kda_chunk_fused.py` builds and consumes the tiles in one workgroup.

The builder-local performance path is the split composition. The fused path is
retained as an alternative schedule and as an independent cross-check. Both
builders accept an optional non-zero initial state and can store the final
state.

The dispatcher exposes all three emitted kernels. Its `auto` policy remains
the fused kernel because one dispatch result cannot yet represent the split
path's workspace and ordered pair of launches. Callers selecting the split
path must request and launch `chunk_prep` followed by `chunk_scan`.

## Run the builders

From `rocke/library` with the editable `rocke` and library packages available:

```bash
python builders/gfx950/kda/kda_chunk_prep.py
python builders/gfx950/kda/kda_chunk_split.py
python builders/gfx950/kda/kda_chunk_fused.py
```

Use `--no-check` for benchmarking only and `--shapes BxHxT,...` to select
benchmark shapes. The checks compare against independent float64 references.

The reusable benchmark scenario runs both compositions:

```bash
python -m benchmarks.gfx950.kda.benchmark_chunkwise --path both
```

The split benchmark selects the scan geometry from `batch * heads`, matching
the dispatcher.

## Optimized scan schedule

The standalone scan is software-pipelined without a second LDS tile set:

- V loads are issued before publishing the state mirror and computing `Z`, so
  those operations cover the V-memory latency.
- Chunk zero is staged before the loop. A C32 scan with at least two waves
  issues the next chunk's materialized tiles while the current chunk computes,
  holds them in registers, and commits them only after the current LDS reads
  retire.
- The SA16 schedule delays that tile issue until V has retired, avoiding
  competition between the larger prefetch burst and the current residual.
- One decay value is loaded per lane and state-column tile, then reused for all
  accumulator slots owned by that lane.

`KdaChunkScanSpec.prefetch_tiles` defaults on. Setting it to `False` retains the
immediate staging path and adds `nopf` to the kernel name for controlled A/B
testing. C16 and single-wave schedules keep immediate staging automatically
because the prefetched vectors would cost more occupancy or tail traffic than
they hide.

For the Kimi K3 contract (`bf16`, `DK=DV=128`, `C=32`),
`tuned_kda_chunk_scan_spec(workgroups)` selects:

- four value bands with a 128-thread SA16 scan through 96 recurrence streams;
- two value bands with a 256-thread SA16 scan through 192 streams;
- the unsplit 256-thread SA32 scan above 192 streams.

The prep kernel stays on its independent 256-thread schedule. Value splitting
changes only scan ownership and grid size; every workgroup writes a disjoint
V/state band.

## Dispatcher

`dispatch.kda` registers the fused kernel and both ordered split phases:

```python
from dispatch.kda import KdaRequest, dispatch_kda

common = dict(
    batch=1,
    num_heads=12,
    seqlen=4096,
    head_k=128,
    head_v=128,
    chunk_size=32,
    dtype="bf16",
    arch="gfx950",
)

prep = dispatch_kda(KdaRequest(**common, algorithm="chunk_prep"))
scan = dispatch_kda(KdaRequest(**common, algorithm="chunk_scan"))

assert prep.candidate.name == "kda_gfx950_chunk_prep"
assert scan.candidate.name == "kda_gfx950_chunk_scan"
assert scan.spec.value_splits == 4
assert scan.grid == (1 * 12 * 4, 1, 1)
```

The two requests produce compatible materialized-tile layouts even though the
prep and scan block sizes differ. Launch `prep` and then `scan` on the same
stream. An unqualified request still returns `kda_gfx950_chunk_fused`.

### Raw beta ABI

The framework-facing raw prep kernel has separate compile-time beta ABIs.
BF16 is the default; `fp32_beta_dtype=True` preserves direct FP32 input. Their
kernel names carry `bbf16` and `bfp32`, respectively.

Both variants accept a strided `[B,T,H]` projection view:
`beta_stride_batch`, `beta_stride_token`, and `beta_stride_head` are element
strides in the raw prep signature. The BF16 variant extends each load to FP32;
both variants then apply sigmoid in FP32.

Prepared chunk-packed prep and the fused kernel retain their original FP32
beta ABI. Raw callers pass the original tensor and its strides without a
host-side dtype conversion or contiguous materialization.

## Tests

The CPU lane validates admission rules and compiles all three builders through
comgr when available:

```bash
python -m pytest \
  tests/test_kda_chunkwise_spec.py \
  tests/dispatch/kda/test_gfx950_wiring.py
```

The GPU lane requires a gfx950 device and checks the tile oracle, both
compositions, non-zero initial state, value splits, and split/fused agreement:

```bash
python -m pytest tests/test_kda_chunkwise_gfx950_numeric.py -m gpu
```

KDA is a family in the shared CPU-only IR parity harness. From the `rocke/`
root, re-bless or verify all representative families with:

```bash
PYTHONPATH=platform/python:library \
  python platform/tests/instances/rocke_ir_parity_harness.py \
    --write platform/tests/golden/rocke_representative_ir_sha256.json
PYTHONPATH=platform/python:library \
  python -m pytest platform/tests/test_rocke_ci_static.py \
    -k ir_cases_match_golden_sha256
```
