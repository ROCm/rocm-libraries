# CK DSL Benchmark

`rocke.benchmark` owns evidence collection: sweep planning, compile records,
runtime correctness, benchmark timing, and JSON/CSV output. Dispatch selection
lives in `rocke.dispatch`.

## Layout

```text
rocke/benchmark/
  __init__.py
  summary.py                 # generic repeated-run summaries for manifests
  gemm/
    fp16_rcr_sweep.py        # dispatcher-backed FP16 RCR GEMM sweep harness
    tests/
      test_fp16_rcr_sweep.py
      test_fp16_rcr_multigpu.py
```

## GEMM FP16 RCR Sweep

Plan only:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python -m rocke.benchmark.gemm.fp16_rcr_sweep \
  --output-dir /tmp/rocke_gemm_sweep \
  --shape '128,128,32:small:true'
```

Compile one selected variant and run it:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python -m rocke.benchmark.gemm.fp16_rcr_sweep \
  --output-dir /tmp/rocke_gemm_sweep \
  --shape '128,128,32:small:true' \
  --spec-id cdna_cshuffle_default \
  --compile --run \
  --warmup-iters 1 --timed-iters 3
```

Run the representative shape set:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python -m rocke.benchmark.gemm.fp16_rcr_sweep \
  --output-dir /tmp/rocke_gemm_sweep \
  --compile --run \
  --parallel 2
```

The JSON output uses schema:

```text
ck.dsl.benchmark.gemm.fp16_rcr_sweep/v1
```

The document contains:

- `config`: sweep request metadata;
- `variants`: supported dispatcher variants keyed by `KernelId`;
- `filtered`: rejected candidates with reasons;
- `builds`: HSACO/manifest build records;
- `runs`: `run_manifest` correctness/benchmark records.

## Run Tests

No-GPU sweep planning tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python -m unittest discover \
  -s dnn-providers/hip-kernel-provider/rocKE/Python/rocke/benchmark/gemm/tests \
  -p 'test_fp16_rcr_sweep.py'
```

GPU-gated multi-GPU sweep smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python -m unittest \
  dnn-providers/hip-kernel-provider/rocKE/Python/rocke/benchmark/gemm/tests/test_fp16_rcr_multigpu.py
```

All GEMM benchmark tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocKE/Python \
  ~/atom-venv/bin/python -m unittest discover \
  -s dnn-providers/hip-kernel-provider/rocKE/Python/rocke/benchmark/gemm/tests \
  -p 'test*.py'
```

## Onboard A New Benchmark Harness

1. Add a dedicated operator folder under `rocke/benchmark/<operator>/`.
2. Keep one harness per dispatch case, for example:
   - `gemm/fp16_rcr_sweep.py`
   - `conv/fwd_nhwc_krsc_sweep.py`
   - `attention/paged_kv_sweep.py`
3. Build sweep plans from dispatcher requests and registered candidates. Do not
   duplicate support predicates in benchmark code.
4. Use `KernelId.cache_key` as the benchmark identity.
5. Record filtered variants, build failures, run failures, correctness failures,
   and successful timings in the same JSON document.
6. Keep correctness and performance policy explicit:
   - small shapes may use CPU reference checks;
   - large shapes should use launch-only, GPU reference, sampled checks, or
     nightly full validation;
   - benchmark runs should report median/spread before gating performance.
7. Add tests under `rocke/benchmark/<operator>/tests/`.

## Multi-GPU Sweeps

The current GEMM harness supports parallel run lanes through
`run_sweep_variants(..., parallel=N)`. Benchmark lanes are still simple worker
threads that launch `run_manifest` subprocesses; future work can replace this
with a per-GPU scheduler that pins one process/HIP context per device and records
device metadata in every run record.
