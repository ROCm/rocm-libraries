# gfx950 chunkwise KDA

This directory contains the spec-driven builders and correctness harnesses for
chunkwise gated delta-rule linear-attention prefill:

- `kda_chunk_prep.py` builds the six state-independent per-chunk tiles.
- `kda_chunk_split.py` materializes those tiles, then runs the serial state scan.
- `kda_chunk_fused.py` builds and consumes the tiles in one workgroup.

The split path is the default composition. The fused path is retained as an
alternative schedule and as an independent cross-check. Both builders accept
an optional non-zero initial state and can store the final state.

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

This family is currently builder-only. It is intentionally not registered in
the dispatcher.

## Tests

The CPU lane validates admission rules and compiles all three builders through
comgr when available:

```bash
python -m pytest tests/test_kda_chunkwise_spec.py
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
