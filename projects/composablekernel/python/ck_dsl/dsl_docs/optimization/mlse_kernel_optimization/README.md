# MLSE Kernel Optimization Notes And Tools

This directory vendors CK DSL-relevant optimization runbooks and helper scripts
from:

```text
/workspace/mlse-tools-internal/performance/kernel_optimization
```

The files are copied here so CK DSL optimization guidance remains available with
the `ck_dsl` docs. They are reference material, not part of the runtime package.

## Skills

`skills/` contains the CK DSL/profiling-relevant runbooks:

- `gemm-optimization-ckdsl.md`
- `lds-optimization-ckdsl.md`
- `prefetch-data-load-ckdsl.md`
- `capture-kernel-trace-ckdsl.md`
- `kernel-trace-analysis.md`
- `empirical-case-studies.md`
- `kernel-launch-guide.md`
- `bisect-perf-regression.md`

FlyDSL-only and environment-management skills were intentionally not copied.

## Helper Scripts

`tools/` contains the helper scripts most useful for CK DSL benchmarking and
post-processing:

- `stage1_benchmark/`
  - `_ua_shape_utils.py`
  - `benchmark_ckdsl_unified_attention.py`
  - `benchmark_triton_unified_attention.py`
- `stage3_extract_isa/`
  - `count_instructions.py`
  - `extract_isa.py`
  - `compare_ua_hsacos.py`
- `stage4_analyze/`
  - `analyze_prefetch_efficiency.py`
  - `analyze_lds_conflicts.py`
  - `parse_kernel_trace.py`
- `stage5_compare/`
  - `compare_rocprof_stats.py`
- `utils/`
  - `compare_isa.py`
  - `extract_ck_dsl_isa.py`
  - `profile_register_usage.py`
  - `rocm_tools.py`

Some scripts assume the original MLSE repository layout. When running them from
this vendored location, set `PYTHONPATH` explicitly to the relevant tool
subdirectories and CK DSL Python root.

Example:

```bash
cd /workspace/rocm-libraries/projects/composablekernel

PYTHONPATH="python:python/ck_dsl/dsl_docs/optimization/mlse_kernel_optimization/tools/stage1_benchmark" \
  python/ck_dsl/.venv/bin/python \
  python/ck_dsl/dsl_docs/optimization/mlse_kernel_optimization/tools/stage1_benchmark/benchmark_ckdsl_unified_attention.py \
  --shapes /workspace/mlse-tools-internal/performance/kernel_optimization/tests/aiter_ua_prefill2d_allbf16.json \
  --dtype bf16 \
  --limit 1
```

## Profiling Note

ATT trace analysis requires `rocprof-trace-decoder`. If it is unavailable, use
the PMC profiling path documented in `skills/capture-kernel-trace-ckdsl.md`
instead of relying on `code.json` instruction-level traces.
