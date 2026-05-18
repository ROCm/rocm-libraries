# Testing And Debugging Guide

This page covers how to run the `ck_dsl` test suites, how to build and validate an instance from scratch, and how to debug common failure modes.

## Repo Layout For Testing

```text
python/test/test_ck_dsl.py            # static unit suite (no GPU required)
python/test/test_ck_dsl_examples.py   # end-to-end example harness (HIP required)
example/ck_tile/dsl/<N>_*/gen.py      # one generator per example, plus expected.json
python/ck_dsl/examples/               # Python-owned example generators
python/ck_dsl/examples/attention/parity_unified_attention.py   # attention parity harness
python/ck_dsl/examples/ck_tile_parity.py                       # small-op parity harness
```

## Environment

Use the Python interpreter from your ROCm development environment:

```text
python
```

Required environment:

- ROCm 7.x with `libamd_comgr` and `libamdhip64` discoverable by the dynamic linker.
- Python 3.12 with `torch` built for ROCm.
- AMDGPU device visible to HIP (e.g. MI300X, MI325X, MI350X, MI355X for the gfx950 default ISA).

Verify quickly:

```bash
python - <<'PY'
import sys
print("python", sys.version)
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY
```

`torch.cuda.is_available() == True` is required for everything below except the static unit suite.

## Static Unit Tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  python python/test/test_ck_dsl.py
```

In-process. Tests:

- `TestCoreIR`: IR construction, LLVM lowering text shape, static_for, scf_if, optimize_kernel CSE, pointer attr metadata, s_waitcnt encoding.
- `TestTransforms`: coordinate-transform DAG, conv `(m, k) -> NHWC` offset and validity SSA generation.
- `TestAnalysis`: `analyze_llvm_ir`, `analyze_hsaco`, `parse_isa`, `parse_resources`, `compare_variant_reports`, `summarize_runs`.
- `TestHelpers`: atoms catalog, `WarpGrid`, `CoalescedTileLoader`, `AsyncTileLoader`, `LdsLayout`, `SchedulePolicy`, `SoftwarePipeline`, `make_gemm_manifest`.
- `TestInstances`: end-to-end build smoke for `build_universal_gemm`, `build_implicit_gemm_conv`, `build_direct_conv_4c`, `build_direct_conv_16c`, all attention builders.

Expected runtime: ~2 seconds. The validation pass for this doc tree had `231 tests, OK`.

These tests do not require a GPU. They prove IR builds, LLVM text shape, and helpers' static contracts.

## Generated Example Harness

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python \
  python python/test/test_ck_dsl_examples.py
```

Discovers every `example/ck_tile/dsl/<N>_*/gen.py` with an `expected.json`. For each:

1. Subprocess: `gen.py --output-dir <tmp>` builds HSACO + manifest.
2. Asserts one HSACO file with non-zero size.
3. Subprocess: `python -m ck_dsl.run_manifest <hsaco> <manifest> --shape ... --verify`.
4. Asserts `max_abs_diff = 0` for bit-exact kernels or `bad = 0` for tolerance kernels.
5. Asserts measured TFLOPS / GB/s pass any declared lower bound in `expected.json`.

Skips runtime stages cleanly when HIP is not available.

Expected runtime: ~3-4 minutes.

## Manual Build + Verify One Instance

```bash
cd <composablekernel-checkout>

# Build the implicit-GEMM conv example.
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
PYTHONPATH=python python \
    -m ck_dsl.examples.bake_off_implicit_gemm --output-dir "$OUT_DIR"

# Inspect what was emitted.
ls "$OUT_DIR"
# ck_dsl_ex08_bake_off_implicit_gemm_*.hsaco
# manifest.json
# (and .ir.txt / .ll if write_ir_text / write_llvm_text are on)

# Run + verify.
PYTHONPATH=python python \
    -m ck_dsl.run_manifest "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
```

The runner prints `verify max_abs_diff=... bad=K/N` and `Perf: <ms>, <TFlops>, <GB/s>`.

## Distribution Demos

```bash
PYTHONPATH=python python \
  python/ck_dsl/examples/distribution_reduce_demo.py --M 32 --N 4096

PYTHONPATH=python python \
  python/ck_dsl/examples/distribution_2d_add_demo.py --H 64 --W 128
```

These exercise the distribution-driven `load_tile` / `store_tile` path end-to-end (build HSACO + launch + verify vs torch reference).

## Small-Op Parity

```bash
PYTHONPATH=python python \
  python/ck_dsl/examples/ck_tile_parity.py --op all
```

Runs every shipped small-op against a torch reference with per-op tolerance gates. Exit non-zero if any kernel exceeds its tolerance.

## Attention Parity

```bash
export AITER_PATH=<aiter-checkout>
PYTHONPATH="python:${AITER_PATH}" \
  python \
  python/ck_dsl/examples/attention/parity_unified_attention.py \
  --attempts 10 --warmup 5 \
  --paths auto,2d,3d \
  --set default \
  --report "${OUT_DIR:-$(mktemp -d)}"/parity.json
```

Requires AITER for the Triton baseline and the reference attention path. Set
`AITER_PATH` to that checkout before running the harness. The harness:

1. Builds AITER unified-attention inputs (paged KV, GQA).
2. Runs Triton with the matching kernel path.
3. Runs CK DSL via `run_unified_attention_torch(..., backend=...)`.
4. Compares both to AITER's `ref_paged_attn`.
5. Times each lane on a single HIP queue, the same timer for both backends.

The harness writes a JSON report. Use the `--scenario` filter for targeted reruns. See `examples/attention/README.md` for the canonical 5-run methodology.

## Benchmark + Sweep

Build many configs in parallel:

```bash
PYTHONPATH=python python \
  example/ck_tile/dsl/07_gemm_universal_sweep/gen.py \
  --output-dir "$OUT_DIR" --subset compute --parallel 16
```

Benchmark each with median + spread:

```bash
PYTHONPATH=python python \
  -m ck_dsl.sweep_bench "$OUT_DIR"/sweep_manifest.json \
  --attempts 3 --csv "$OUT_DIR"/results.csv
```

The benchmark driver writes a CSV: one row per `(kernel, shape)` with `median_tflops, min_tflops, max_tflops, spread_pct, max_abs_diff`. Use `benchmark_manifest(..., attempts=5, discard_first=True, verify_first=True)` for one manifest.

## Inspecting Generated Code

```python
from ck_dsl import (
    IRBuilder, F16, PtrType, compile_kernel,
    analyze_llvm_ir, analyze_hsaco,
)

kernel = build_universal_gemm(spec)
art = compile_kernel(kernel, capture_ir_text=True)

print("MLIR-style IR:")
print(art.ir_text)

print("\nLLVM IR text (first 200 lines):")
print("\n".join(art.llvm_text.splitlines()[:200]))

ir_stats = analyze_llvm_ir(art.llvm_text)
print(ir_stats.as_dict())

from pathlib import Path
out = Path.cwd() / "k.hsaco"
out.write_bytes(art.hsaco)
hsaco_stats = analyze_hsaco(
    str(out),
)
print(hsaco_stats.isa.as_dict())
print(hsaco_stats.resources.as_dict())
```

Inspect `art.llvm_text` for missing intrinsic declarations, wrong types, missing waitcnt, etc. Inspect `hsaco_stats.resources` for VGPR / SGPR / LDS surprises.

## Comparing Two Variants Of A Kernel

```python
from ck_dsl import VariantReport, compare_variant_reports

baseline = VariantReport.from_artifact(art_baseline)
variant  = VariantReport.from_artifact(art_variant)

diff = compare_variant_reports(baseline, variant)
print(diff)
```

Reports MFMA delta, vector store delta, VGPR delta, LDS delta — the runbook-required evidence that a lever moved the intended primitive.

## Debugging Patterns

### "Compiles but wrong output, small max_abs"

- Suspect MFMA lane packing first (especially K-packed atoms).
- Suspect epilogue `lane_to_output` second.
- Suspect missing `s_waitcnt(vmcnt=0)` on async paths third.
- Try the same kernel with `epilogue="default"` to isolate the cshuffle path.

### "Compiles but wrong output, large max_abs"

- Likely structural: wrong tile geometry, wrong descriptor offset, accumulator not zeroed.
- Test on adversarial shapes: prime-ish M/N/K, very small batches, masks that exercise tails.
- Materialize the implicit-GEMM A via `build_img2col` and check it matches a numpy reference.

### "Hangs at launch"

- Almost always a barrier inside divergent control flow, or a missing `s_waitcnt(vmcnt=0)` followed by `b.sync()` on async paths.
- Check `kernel.attrs["max_workgroup_size"]` matches `LaunchConfig.block[0]`.

### "Faults at launch"

- Pointer arg sized wrong: `gemm_args_signature()` is `ptr (8) ptr (8) ptr (8) i32 (4) i32 (4) i32 (4)`. Mismatched pack order fails.
- Stride passed in bytes vs elements (or vice versa).
- Pointer not aligned to declared `align=N`.

### "Slow but correct"

- Run `analyze_llvm_ir(art.llvm_text)` and confirm the intended MFMA shape / vector store / async LDS load appears.
- Run `analyze_hsaco(...)` and check VGPR / SGPR / LDS.
- Benchmark with `benchmark_manifest(..., attempts=5)` and check `spread_pct` — high spread means cold cache or scheduler instability, not a permanent regression.
- Compare against the empirical results in `RUNBOOK_COMPLIANCE.md` if the kernel is one of the bake-offs.

### "Async kernel only fails under benchmark pressure"

- Missing `s_waitcnt(vmcnt=0)` between issue and consume.
- Workspace tensor reused across iterations without a `Runtime.wait_stream` between.
- `KernelLauncher` constructed with stale signature.

## CI-Style Smoke

A two-minute smoke for a clean clone:

```bash
cd <composablekernel-checkout>
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=python
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"

python python/test/test_ck_dsl.py                       # static
python -m ck_dsl.examples.bake_off_implicit_gemm \
    --output-dir "$OUT_DIR"
python -m ck_dsl.run_manifest \
    "$OUT_DIR"/*.hsaco "$OUT_DIR"/manifest.json --verify
python python/ck_dsl/examples/ck_tile_parity.py --op elementwise
```

If all four pass, the build, COMGR, HIP module, launcher, manifest, and at least one production instance work end-to-end.
