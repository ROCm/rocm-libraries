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
  - `benchmark_rocke_unified_attention.py` (moved to `rocke/library/builders/common/benchmark_rocke_unified_attention.py`)
  - `benchmark_triton_unified_attention.py`
- `stage2_capture/`
  - `capture_att_trace.py` — one-command `rocprofv3 --att` capture: preflights the
    decoder, discovers the kernel name, decodes, and reports each dispatch
- `wavescope/` — capture and post-process traces for the WaveScope viewer
  - `capture_wavescope_trace.py` — wraps `stage2_capture` with the source-location
    env and the sidecar step, so the trace opens with Python correlation
  - `emit_inline_frames.py` — `inline_frames.json` sidecar recovering the inlining
    call stack rocprofv3 flattens away
  - `README.md`                    — install the extension, capture, and read a trace
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

The `benchmark_rocke_unified_attention` benchmark moved from the platform
`stage1_benchmark/` directory to the library. Run it with `rocke/library` on
`PYTHONPATH` from the provider root:

```bash
PYTHONPATH=rocke/library python3 -m builders.common.benchmark_rocke_unified_attention \
  --shapes rocke/library/builders/gfx950/attention/aiter_ua_prefill2d_allbf16.json \
  --dtype bf16 \
  --limit 1
```

## Profiling Note

ATT trace analysis requires `rocprof-trace-decoder`, which does not ship with ROCm
and is not in the ROCm apt repository — install it separately from
<https://github.com/ROCm/rocprof-trace-decoder>. If it is unavailable, use the PMC
profiling path documented in `skills/capture-kernel-trace-rocke.md` instead of
relying on `code.json` instruction-level traces.

A decoded `ui_output_*_dispatch_*` folder can be read interactively in the
**WaveScope** viewer (per-wave timeline over the ISA listing, occupancy, bottleneck
rules) as well as by `stage4_analyze/parse_kernel_trace.py`. WaveScope also carries a
two-way annotation protocol — an agent writes `annotations.json` into the dispatch
folder and the developer replies with `notes.json` — so a bottleneck claim can be
visually verified and argued with rather than just read. See Step 6 of
`skills/capture-kernel-trace-rocke.md`.

The `Source` column of `code.json` is empty unless the kernel was built with
`ROCKE_DEBUG_LOC=1`, which makes the lowering emit DWARF line tables and gives
every instruction the Python line that authored it. Without it all trace
analysis is at ISA level. See
`tools/wavescope/README.md` for the capture, and
`dsl_docs/architecture/wavescope_integration.md` for how the inline call stack
above that line is recovered.
