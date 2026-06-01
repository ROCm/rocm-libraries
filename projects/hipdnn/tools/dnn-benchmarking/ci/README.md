# dnn-benchmarking CI helpers

Small stdlib-only scripts invoked by the presubmit workflow
`.github/workflows/hipdnn-benchmark-ci.yml`. They live with the tool they guard
so they version with it and are unit-testable without GitHub Actions.

## `check_results.py`

Tool-health gate. Reads a `results.json` (from `python -m dnn_benchmarking
--output`) and exits non-zero unless `metadata.pass_combinations > 0` and
`metadata.error_combinations == 0`. This catches the silent case where every
graph/engine combination is *skipped* (e.g. a wrong plugin path) — which the
tool itself reports as exit 0.

```bash
python3 ci/check_results.py results.json
```

It is **not** a performance or correctness gate; it only asserts the benchmark
tool ran end-to-end without breaking.

Unit tests: `tests/unit/ci/test_check_results.py` (CPU-only, no `gpu` marker).

## `emit_provenance.py`

Writes a `provenance.json` sidecar from GitHub Actions environment variables
(commit SHA, PR number, run URL, TheRock artifact pin, GPU family, graph glob,
warmup/iters). `results.json` describes the host/GPU/software environment but
not the CI identity; the sidecar makes an archived result self-contained.

```bash
python3 ci/emit_provenance.py provenance.json
```

Missing env vars are recorded as `null` — provenance never blocks the job.
