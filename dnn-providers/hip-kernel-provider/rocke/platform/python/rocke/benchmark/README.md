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
  moe/
    fused_mega_fp8_dispatch.py   # dispatcher-driven fused-MoE fp8 benchmark
    tests/
      test_fused_mega_fp8_dispatch.py
```

## GEMM FP16 RCR Sweep

Plan only:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  ~/atom-venv/bin/python -m rocke.benchmark.gemm.fp16_rcr_sweep \
  --output-dir /tmp/rocke_gemm_sweep \
  --shape '128,128,32:small:true'
```

Compile one selected variant and run it:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  ~/atom-venv/bin/python -m rocke.benchmark.gemm.fp16_rcr_sweep \
  --output-dir /tmp/rocke_gemm_sweep \
  --shape '128,128,32:small:true' \
  --spec-id cdna_cshuffle_default \
  --compile --run \
  --warmup-iters 1 --timed-iters 3
```

Run the representative shape set:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
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

## MoE Fused-Mega FP8 Dispatch Bench

`moe/fused_mega_fp8_dispatch.py` sweeps the token count through
`dispatch_moe_plan` and runs whatever comes back. Token count is a selection
knob for this family, so the default set spans the fused band (one launch), the
split band (two launches), and both `tile_m` bands -- a routing regression
shows up as a changed kernel, a changed launch count, or a failed check.

This harness must run on a **torch-free** interpreter: importing torch before
the first Comgr compile resolves Comgr against torch's bundled LLVM and changes
codegen. It asserts torch is absent, and numpy is the only dependency it needs.

Routing decisions only, no GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  python3 -m rocke.benchmark.moe.fused_mega_fp8_dispatch --plan-only
```

Build, run, check and time the default shapes:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  python3 -m rocke.benchmark.moe.fused_mega_fp8_dispatch \
  --json /tmp/rocke_moe_dispatch.json
```

A short smoke run on a shared GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  python3 -m rocke.benchmark.moe.fused_mega_fp8_dispatch \
  --tokens 1,32 --warmup 20 --iters 10 --check sample
```

Correctness is a numpy f32 model of exactly the operands the kernel consumes:
every token up to `--check auto`'s oracle limit, a random exact-checked sample
of tokens above it, and a finiteness check over the whole output either way.
The reported `check` column says which was used. The JSON output uses schema:

```text
ck.dsl.benchmark.moe.fused_mega_fp8_dispatch/v1
```

## Run Tests

No-GPU sweep planning tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  ~/atom-venv/bin/python -m unittest discover \
  -s dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/benchmark/gemm/tests \
  -p 'test_fp16_rcr_sweep.py'
```

GPU-gated multi-GPU sweep smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  ~/atom-venv/bin/python -m unittest \
  dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/benchmark/gemm/tests/test_fp16_rcr_multigpu.py
```

No-GPU MoE dispatch-bench tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  python3 -m unittest \
  rocke.benchmark.moe.tests.test_fused_mega_fp8_dispatch
```

All GEMM benchmark tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=dnn-providers/hip-kernel-provider/rocke/platform/python \
  ~/atom-venv/bin/python -m unittest discover \
  -s dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/benchmark/gemm/tests \
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
