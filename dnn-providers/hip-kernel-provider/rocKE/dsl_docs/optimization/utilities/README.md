# Kernel Optimization Notes And Tools

The files are copied here so CK DSL optimization guidance remains available with
the `rocke` docs. They are reference material, not part of the runtime package.

## Skills

`skills/` contains the CK DSL/profiling-relevant runbooks:

- `gemm-optimization-rocke.md`
- `lds-optimization-rocke.md`
- `prefetch-data-load-rocke.md`
- `capture-kernel-trace-rocke.md`
- `kernel-trace-analysis.md`
- `empirical-case-studies.md`
- `kernel-launch-guide.md`
- `bisect-perf-regression.md`

## Helper Scripts

`tools/` contains the helper scripts most useful for CK DSL benchmarking and
post-processing:

- `dsl_probes/` (rocke-native, no GPU launch required for most probes)
  - `probe_occupancy.py`           — `llvm-readelf --notes` → VGPR/AGPR/SGPR/LDS occupancy estimate
  - `probe_isa_inspect.py`         — `llvm-objdump` → opcode-class histogram (MFMA, ds_read, waitcnt, …)
  - `probe_intrinsic_counts.py`    — lowered LLVM IR → AMDGCN intrinsic histogram
  - `probe_lowering_compare.py`    — LLVM-direct vs HIP-debug backend HSACO compare
  - `probe_config_sweep.py`        — generic `dataclasses.replace` sweep over a spec dataclass
  - `probe_targeted_bench.py`      — direct-CUDA-event bench across a list of shapes
  - `probe_rocprof_single.py`      — single-process rocprof-friendly harness
  - `README.md`                    — when-to-use index
- `stage1_benchmark/`
  - `_ua_shape_utils.py`
  - `benchmark_rocke_unified_attention.py`
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
  - `extract_rocke_isa.py`
  - `profile_register_usage.py`
  - `rocm_tools.py`

The `dsl_probes/` folder is the "step 3: inspect the artifact before
chasing performance" tier from the optimization runbook. It is purely
DSL-side (compile + lower + readelf + objdump) so it works without a
GPU attached and runs in well under one second per variant. The
`stage{1,3,4,5}` tiers complement it with rocprof- and trace-based
analysis once a real launch is in flight.

Some scripts assume the original MLSE repository layout. When running them from
this vendored location, set `PYTHONPATH` explicitly to the relevant tool
subdirectories and CK DSL Python root.

Example:

```bash
cd <repo>/dnn-providers/hip-kernel-provider/rocKE

PYTHONPATH="Python:dsl_docs/optimization/mlse_kernel_optimization/tools/stage1_benchmark" \
  Python/rocke/.venv/bin/python \
  dsl_docs/optimization/mlse_kernel_optimization/tools/stage1_benchmark/benchmark_rocke_unified_attention.py \
  --shapes dsl_docs/optimization/utilities/tools/stage1_benchmark/tests/aiter_ua_prefill2d_allbf16.json \
  --dtype bf16 \
  --limit 1
```

## Profiling Note

ATT trace analysis requires `rocprof-trace-decoder`. If it is unavailable, use
the PMC profiling path documented in `skills/capture-kernel-trace-rocke.md`
instead of relying on `code.json` instruction-level traces.
