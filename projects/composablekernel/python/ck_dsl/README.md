# ck_dsl — Python authoring layer for Composable Kernel Tile

A Python DSL for authoring high-performance CK Tile-style GPU kernels on
AMDGPU (gfx940/gfx942/gfx950). Drops straight to AMDGPU LLVM IR and
in-process HSACO via `libamd_comgr` — no hipcc, no MLIR pipeline, no
template instantiation. Typical kernel codegen: **5-100 ms** total wall
time per instance (warm comgr).

This is the same authoring surface CK Tile exposes in C++:
  - `tensor_descriptor` with `unmerge`, `embed`, `merge`, `pad`
  - `tile_distribution_encoding` for warp/lane partitioning
  - `tile_window`, `load_tile`, `store_tile`, `gemm_tile`
  - MFMA atoms keyed by `(in_dtype, m, n, k)`
  - `cshuffle_epilogue`, `compv4` pipelines, `sched_group_barrier` hints

… but in Python, with SSA values for runtime coords and a runtime IR
that lowers in milliseconds instead of seconds.

## What's in here

```text
python/ck_dsl/
├── __init__.py               # top-level package re-exports
├── __main__.py               # `python -m ck_dsl` (lists discoverable entry points)
├── transforms.py             # CK-style coordinate-transform DAG
│                             #   (TensorDescriptor + Transform subclasses)
├── run_manifest.py           # Python-native (hsaco + manifest.json) runner
├── sweep.py                  # parallel build-on-the-fly sweep driver
├── sweep_bench.py            # benchmark driver consuming sweep manifests
│
├── core/                     # SSA IR + lowering passes (the DSL itself)
│   ├── ir.py                 #   Type, Value, Op, Region, IRBuilder, KernelDef
│   ├── ir_print.py           #   print_ir(): MLIR-style textual dump
│   ├── passes.py             #   constant fold, CSE, dead pure-op removal
│   ├── lower_llvm.py         #   lower_kernel_to_llvm() AMDGPU LLVM IR
│   ├── lower_hip.py          #   lower_kernel_to_hip() HIP C++ text (debug)
│   └── __init__.py
│
├── runtime/                  # in-process HSACO build + HIP module + launch
│   ├── comgr.py              #   ctypes over libamd_comgr (HSACO build)
│   ├── hip_module.py         #   ctypes over libamdhip64 (Runtime, Module,
│   │                         #     Function, Event); owns the per-stream
│   │                         #     args-buffer keep-alive queue
│   ├── torch_module.py       #   torch-tensor arg packing + resolve_stream
│   │                         #     (back-compat launch_torch_kernel shim)
│   ├── launcher.py           #   KernelLauncher / PipelineLauncher /
│   │                         #     WorkspacePool / DeviceMem / time_launches
│   │                         #     -- the canonical launch primitives
│   └── __init__.py
│
├── helpers/                  # high-level kernel-authoring helpers
│   ├── atoms.py              #   MfmaAtom (16x16x16, 16x16x32, 32x32x8,
│   │                         #     32x32x16, 4x4x4 fp16)
│   ├── geometry.py           #   WarpGrid: tid/lane/warp_m/warp_n decomp
│   ├── loads.py              #   CoalescedTileLoader, AsyncTileLoader
│   ├── layouts.py            #   LdsLayout: K-pad / packed async guardrails
│   ├── schedule.py           #   SchedulePolicy: sched_group_barrier policy
│   ├── pipeline.py           #   SoftwarePipeline: ping-pong staging
│   ├── epilogues.py          #   DirectEpilogue, CShuffleEpilogue
│   ├── compile.py            #   compile_kernel() one-shot IR->HSACO
│   ├── manifest.py           #   make_gemm_manifest, make_conv_manifest
│   └── __init__.py
│
├── analysis/                 # static inspection: LLVM IR + HSACO/ISA
│   ├── ir.py                 #   analyze_llvm_ir()
│   ├── isa.py                #   analyze_hsaco(), parse_isa(),
│   │                         #     parse_resources()
│   └── __init__.py
│
├── benchmark/                # repeated-run benchmark summaries
│   ├── summary.py            #   benchmark_manifest(), summarize_runs()
│   └── __init__.py
│
├── instances/                # parametric kernel instance builders
│   ├── gemm_universal.py     #   universal GEMM (matches CK dispatcher's
│   │                         #     config schema 1:1)
│   ├── conv_implicit_gemm.py #   NHWC × KRSC -> NHWK (bake-off 1)
│   ├── conv_direct_grouped.py#   grouped direct conv (bake-off 2: 16c + 4c)
│   └── __init__.py
│
└── examples/                 # maintained Python-owned example generators
    ├── bake_off_implicit_gemm.py
    ├── bake_off_direct_conv_16c.py
    └── bake_off_direct_conv_4c.py

example/ck_tile/dsl/          # CMake integration (each <N>_*/gen.py wraps a
│                             # generator in ck_dsl.examples or instances/)
├── common/launcher.cpp       #   optional C++ host launcher (Python runner
│                             #     is the primary path)
├── 06_gemm_universal_cshuffle/  # universal-builder hero GEMM (cshuffle)
├── 07_gemm_universal_sweep/  # full dispatcher-config sweep
├── 08_bake_off_implicit_gemm/# implicit-GEMM conv (~248 / ~280 TFLOPS;
│                             # see results.md)
├── 09_bake_off_direct_conv_16c/  # direct grouped conv 16c (~210 TFLOPS)
└── 10_bake_off_direct_conv_4c/   # direct grouped conv 4c (~48.8 TFLOPS)
```

## Hello, GEMM (12 lines)

```python
from ck_dsl import compile_kernel
from ck_dsl.instances import (
    TileSpec, TraitSpec, UniversalGemmSpec, build_universal_gemm,
)

spec = UniversalGemmSpec(
    name="my_gemm",
    tile=TileSpec(tile_m=128, tile_n=128, tile_k=32,
                  warp_m=2, warp_n=2, warp_k=1,
                  warp_tile_m=32, warp_tile_n=32, warp_tile_k=16),
    trait=TraitSpec(pipeline="compv4", scheduler="intrawave",
                    epilogue="cshuffle"),
)
kernel = build_universal_gemm(spec)
artifact = compile_kernel(kernel)
# artifact.hsaco -> HSA Code Object bytes ready for hipModuleLoadData
# artifact.timings -> {ir_build, ir_lower_llvm, comgr_*, total} (ms)
```

## Hello, convolution

```python
from ck_dsl.instances.conv_implicit_gemm import (
    ConvProblem, ImplicitGemmConvSpec, build_implicit_gemm_conv,
)

prob = ConvProblem(
    N=8, Hi=56, Wi=56, C=64,
    K=64, R=3, S=3,
    sH=1, sW=1, pH=1, pW=1, dH=1, dW=1,
)
spec = ImplicitGemmConvSpec(
    problem=prob,
    tile_m=64, tile_n=64, tile_k=64,
    warp_m=2, warp_n=2,
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    pipeline="mem",       # single-buffer LDS
    epilogue="cshuffle",  # wide vec-8 buffer_store_dwordx4
)
kernel = build_implicit_gemm_conv(spec)
```

Performance on this kernel: **280 TFLOPS** in HIP-graph mode on MI300X,
**248 TFLOPS** per-launch. Beats CK Tile's `cktile_fixed_lean` by ~12%
on the same bake-off problem (`N=8`, `Hi=Wi=56`, `C=K=64`, 3x3, pad=1).

## Convolution addressing — the coordinate-transform DAG

The cleanest part of CK Tile's convolution authoring surface, ported
to Python verbatim. See `python/ck_dsl/TRANSFORM_DAG.md` for the full
walkthrough.

```python
from ck_dsl.transforms import TensorDescriptor, unmerge, embed, pad

# A is NHWC. We want to access it as a (m, k) implicit-GEMM matrix,
# where m = N*Ho*Wo (output spatial flatten) and k = R*S*C (filter
# flatten). The DAG that maps (m, k) -> NHWC linear offset:

desc = (
    TensorDescriptor
        .naive("A_nhwc", lengths=[N, Hi, Wi, C],
               coord_names=["n", "hi", "wi", "c"])
        .transform(
            unmerge("m",  into=["n", "ho", "wo"],  dims=[N, Ho, Wo]),
            embed (["ho", "r"], "hi", strides=[sH, dH], offset=-pH,
                   lo=0, hi=Hi),
            embed (["wo", "s"], "wi", strides=[sW, dW], offset=-pW,
                   lo=0, hi=Wi),
            unmerge("k",  into=["r", "s", "c"],   dims=[R, S, C]),
            pad   ("r",  lo=0, hi=R),       # partial-K-tile guard
            pad   ("s",  lo=0, hi=S),       # partial-K-tile guard
        )
)

# Now in the kernel body, just hand it (m, k) and get back the
# NHWC linear offset + the in-bounds predicate:
a_off_elems, a_valid = desc.offset(b, m=m_val, k=k_val)
```

The descriptor produces, at IR build time, the same SSA dataflow
that hand-written kernels emit by integer arithmetic. The win is
that the algebraic structure is preserved — the kernel author never
writes `(m / (Ho*Wo)) % N` by hand, and any change to the problem
(e.g., stride/dilation/pad) edits the *descriptor*, not the
addressing math inside the K-loop.

## Helpers (high-level kernel-authoring layer)

`python/ck_dsl/helpers/` is the "shortest path" surface for kernel
authors who don't want to compose every primitive from scratch. See
`python/ck_dsl/helpers/README.md` for the full reference.

  - `MfmaAtom` (`helpers/atoms.py`) — one dataclass per MFMA shape, with
    per-lane widths, `emit(b, a, b, c)` dispatch, and the per-lane
    output `lane_to_output(b, lane, i) -> (row_off, col_off)` so the
    epilogue can compute storage addresses without re-deriving the AMD
    lane layout each time.
  - `WarpGrid` (`helpers/geometry.py`) — block geometry + lane/warp/tid
    decomposition packed into one immutable view.
  - `CoalescedTileLoader` (`helpers/loads.py`) — coalesced global→LDS
    copy plan parameterised over tile geometry and vec width.
  - `AsyncTileLoader` (`helpers/loads.py`) — direct DRAM→LDS via
    `raw_ptr_buffer_load_lds` (runbook §6.3) for the compv4 overlap.
  - `DirectEpilogue` (`helpers/epilogues.py`) — per-lane vec stores
    using the atom's `lane_to_output` mapping.
  - `CShuffleEpilogue` (`helpers/epilogues.py`) — LDS-stage acc + wide
    vec global stores (runbook §9.3).
  - `compile_kernel(kernel)` (`helpers/compile.py`) — one-shot
    `IR → LLVM IR → HSACO` wrapper returning a `KernelArtifact`.
  - `make_gemm_manifest`, `make_conv_manifest`
    (`helpers/manifest.py`) — emit the standard manifest.json schema
    the launcher and runner consume.

## Analysis and Benchmarking

The runbook says to inspect the generated program before trusting a
performance story. The reusable surface is:

```python
from ck_dsl import analyze_llvm_ir, analyze_hsaco, benchmark_manifest

# Fast static check on the LLVM IR text from compile_kernel(...):
ir_stats = analyze_llvm_ir(artifact.llvm_text)
assert ir_stats.async_buffer_load_lds_calls > 0

# Final ISA/resource check on the emitted HSACO:
hsaco_stats = analyze_hsaco(
    hsaco_path,
    objdump="/opt/rocm/llvm/bin/llvm-objdump",
    readelf="/opt/rocm/llvm/bin/llvm-readelf",
)
print(hsaco_stats.isa.as_dict())
print(hsaco_stats.resources.as_dict())

# Hygienic repeated timing:
summary = benchmark_manifest(manifest_path, attempts=7,
                             verify_first=True, discard_first=True)
print(summary.median_tflops, summary.spread_pct)
```

Use this trio whenever a new DSL feature claims a speedup: prove the
intended primitive lowered (`analyze_llvm_ir`), prove the final ISA has
the expected instruction/resource footprint (`analyze_hsaco`), then
benchmark with median/spread (`benchmark_manifest`).

## Unified Attention Status

`ck_dsl` now has the full compiler/runtime stack for AITER's
`unified_attention` *and* ships two tuned MFMA kernels: a tiled 2D
single-warp kernel and a tiled 3D split-KV pipeline (segment +
reduce). The 3D path is the one we ship in production; it beats AITER's
Triton `unified_attention` on every covered scenario.

### Apples-to-apples vs Triton (MI355X / gfx950, ROCm 7.0.1)

All numbers are the average of 10 timed iterations after 5 warmup
launches, produced by
`python/ck_dsl/examples/attention/parity_unified_attention.py`.

#### Each backend's own selector (production-relevant)

CK DSL's dispatcher always chooses the 3D split-KV path when supported.
Triton's dispatcher uses its own `use_2d_kernel(...)` heuristic, so
"tri-path" below shows which kernel Triton actually launched per
scenario.

| Scenario                  | tri-auto | ck-auto  | speedup | tri-path | max_abs(CK vs ref) |
|---------------------------|---------:|---------:|--------:|---------:|-------------------:|
| decode_d128_b16           |  87.1us  |  36.8us  | **2.36x** | 3d | 1.83e-4 |
| decode_d128_b64           |  84.8us  |  33.6us  | **2.53x** | 3d | 1.83e-4 |
| decode_d256_b16           |  86.2us  |  35.5us  | **2.43x** | 3d | 1.22e-4 |
| prefill_d128_b16          |  52.5us  |  34.7us  | **1.51x** | 2d | 1.95e-3 |
| mixed_d128_b16            |  97.5us  |  33.5us  | **2.91x** | 3d | 9.77e-4 |
| sliding_d128_b16          |  63.5us  |  34.8us  | **1.82x** | 2d | 3.05e-4 |
| softcap_d128_b16          |  84.1us  |  34.4us  | **2.45x** | 3d | 1.22e-4 |
| bf16_decode_d128_b64      |  85.2us  |  36.6us  | **2.33x** | 3d | 9.77e-4 |
| alibi_decode_d128_b16     |  85.7us  |  35.9us  | **2.39x** | 3d | 9.77e-4 |
| alibi_mixed_d128_b16      |  87.8us  |  35.6us  | **2.47x** | 3d | 1.95e-3 |
| qq_bias_prefill_d128_b16  |  73.4us  |  34.8us  | **2.11x** | 2d | 1.95e-3 |

`max_abs(CK vs ref)` is the worst per-element error against the AITER
`ref_paged_attn` reference — on every scenario CK DSL matches Triton
bit-for-bit (`max_abs(CK vs Triton) == 0` once both are cast back to
`fp16`/`bf16`).

#### 3D vs 3D (force both backends to use the split-KV kernel)

The CK DSL 3D kernel is the apples-to-apples winner on 10 of 11
scenarios; same split-KV algorithm as Triton's 3D, just hand-tuned
with CK Tile lessons. `alibi_mixed_d128_b16` is launch-overhead-bound
on this GPU (one of its sequences is only 5 query tokens / 18 KV
tokens) and varies 3-4x across attempts; rerun the harness for a
stable median.

| Scenario                  | tri-3d   | ck-3d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           |  84.1us  |  33.8us  | **2.49x** |
| decode_d128_b64           |  83.3us  |  33.9us  | **2.46x** |
| decode_d256_b16           |  83.9us  |  33.6us  | **2.50x** |
| prefill_d128_b16          |  94.7us  |  50.7us  | **1.87x** |
| mixed_d128_b16            |  83.5us  |  33.2us  | **2.51x** |
| sliding_d128_b16          |  84.2us  |  34.4us  | **2.45x** |
| softcap_d128_b16          |  84.7us  |  36.1us  | **2.35x** |
| bf16_decode_d128_b64      |  85.2us  |  33.0us  | **2.58x** |
| alibi_decode_d128_b16     |  83.1us  |  33.5us  | **2.48x** |
| alibi_mixed_d128_b16      |  87.7us  |  33.8us  | **2.59x** |
| qq_bias_prefill_d128_b16  |  85.6us  |  35.5us  | **2.41x** |

#### 2D vs 2D (force both backends to use the single-warp kernel)

CK DSL's 2D kernel is a single-warp, single-CTA-per-(qblock, kv_head)
design that we keep as a fallback for problems that 3D's segment
pipeline doesn't (yet) cover. It is **not** competitive with Triton's
multi-warp 2D kernel and is never selected by `backend="auto"`. We
include this table only for completeness.

| Scenario                  | tri-2d   | ck-2d    | speedup |
|---------------------------|---------:|---------:|--------:|
| decode_d128_b16           | 113.9us  | 641.6us  |  0.18x  |
| prefill_d128_b16          |  51.0us  | 598.3us  |  0.09x  |
| mixed_d128_b16            |  55.5us  | 630.3us  |  0.09x  |
| sliding_d128_b16          |  52.6us  | 601.0us  |  0.09x  |
| softcap_d128_b16          | 100.3us  | 685.3us  |  0.15x  |
| bf16_decode_d128_b64      |  89.4us  | 610.6us  |  0.15x  |
| alibi_decode_d128_b16     | 124.9us  | 627.2us  |  0.20x  |
| qq_bias_prefill_d128_b16  |  56.1us  | 633.5us  |  0.09x  |

`alibi_mixed_d128_b16` produces NaN on the CK 2D path (known
limitation; the 2D path is single-warp and is not exercised by `auto`).

### Reproducing

```bash
cd /workspace/rocm-libraries-streaming/projects/composablekernel
sudo -n env \
  "PYTHONPATH=$(pwd)/python:/workspace/aiter" \
  /workspace/dsl_bake_off/venv/bin/python3 \
  python/ck_dsl/examples/attention/parity_unified_attention.py \
  --attempts 10 --warmup 5 \
  --report ck/dsl/unified_attention_parity.json
```

The script:
- forces Triton onto the 2D/3D kernel per row by monkey-patching its
  `use_2d_kernel(...)` heuristic,
- forces CK DSL onto the matching path via
  `run_unified_attention_torch(..., backend="tiled"/"3d"/"auto")`,
- compares both backends' outputs to the AITER `ref_paged_attn`
  reference and to each other.

See
[`python/ck_dsl/examples/attention/README.md`](examples/attention/README.md)
for the full table and methodology.

### Compiler / runtime pieces backing the kernels

- f32 math, `exp2`, reciprocal, softcap, `sitofp`, fp8e4m3->f32, typed
  / masked loads (with OOB-safe index clamp), vector reductions, and
  runtime branches in `core/`.
- Launcher abstractions in `runtime/launcher.py`: `KernelLauncher`
  (one compiled HSACO + loaded HIP module + function entry, cached
  per problem-shape), `PipelineLauncher` (multi-stage kernels on
  one stream -- split-KV attention's seg+reduce, k-fixup GEMM, etc.),
  `WorkspacePool` (long-lived torch workspace tensors), `DeviceMem`
  (RAII over `hipMalloc/hipFree` for numpy / manifest flows), and
  `time_launches` (the only event-creating timing primitive). These
  capture what CK Tile's `fmha_bwd_launcher`, FlyDSL's
  `_TorchReduceWrapper`, and Triton's `JITFunction` all do
  structurally; see `runtime/__init__.py` for the rationale.
- Args-buffer keep-alive queue (`Runtime._pending_args`) and stream
  resolution (`resolve_stream`) in `runtime/hip_module.py` /
  `runtime/torch_module.py` handle the
  `HIP_LAUNCH_PARAM_BUFFER_POINTER` lifetime race and the torch
  caching-allocator stream desync that previously surfaced as
  intermittent `max_abs` drift in benchmark harnesses. See
  [`ck/dsl/unified_attention_creative_results.md`](../../ck/dsl/unified_attention_creative_results.md)
  for the investigation.
- Attention helpers (`PagedKvDescriptor`, `OnlineSoftmaxState`, the
  AITER 2D / 3D config selectors) in `helpers/attention.py`.
- Tiled MFMA kernels in `instances/attention_tiled_2d.py` and
  `instances/attention_tiled_3d.py`; plus the scalar correctness
  kernels in `instances/attention_unified.py` used as oracles.
- Coverage: causal + sliding window + softcap + sinks + ALiBi +
  QQ-bias on `fp16`/`bf16`, `head_size in {128, 256}`,
  `block_size in {16, 64}`. FP8 K/V cache + output scale/clamp is the
  next coverage step (see
  [`ck/dsl/unified_attention_results.md`](../../ck/dsl/unified_attention_results.md)).

## Compiler-Layer Optimizations

Direct LLVM IR emission is fast, but the DSL must own the high-level
optimizations that MLIR/CK templates would otherwise do before LLVM:

  - `IRBuilder.static_for(...)` and `IRBuilder.unroll(...)` mark intentional
    compile-time loops; they emit straight-line IR, not `scf.for`.
  - SSA `Value` objects raise on Python truthiness, so `if value:` cannot
    silently become a host branch. Use `IRBuilder.static_if(bool, ...)` for
    compile-time decisions and `IRBuilder.scf_if(value)` for runtime branches.
  - `ck_dsl.core.passes.optimize_kernel(...)` performs conservative
    constant folding, CSE, and dead pure-op removal. It never removes loads,
    stores, barriers, async copies, or MFMAs.
  - Kernel params support pointer metadata (`noalias`, `readonly`,
    `writeonly`, `align`, `dereferenceable`) and LLVM lowering preserves it.
  - `LdsLayout`, `SchedulePolicy`, and `SoftwarePipeline` lift LDS layout,
    scheduler hints, and prologue/steady-state/epilogue construction into
    reusable helpers.

## Runbook compliance

Each runbook section in
`/workspace/.cursor/skills/gpu-op-optimization-runbook/OPTIMIZATION_RUNBOOK.md`
maps to a DSL primitive:

| Runbook section | DSL primitive |
|---|---|
| §2.2 problem definition | `ConvProblem`, `UniversalGemmSpec` |
| §4.1 algorithm mapping  | `instances/gemm_universal.py`, `instances/conv_implicit_gemm.py` |
| §5 work decomposition   | `TileSpec`, `WarpGrid` (in progress) |
| §6.1 buffer rsrc        | `buffer_rsrc` IR op (with DW3=0x27000 flags) |
| §6.2 vec global stores  | `global_store_vN_f16`, `buffer_store_vN_f16` |
| §6.3 async DMA          | `async_buffer_load_lds`, `async_buffer_load_lds_addr` |
| §7.2 MFMA atoms         | `MfmaAtom` (16x16x16, 16x16x32, 32x32x8, 32x32x16, 4x4x4) |
| §7.3 sched hints        | `sched_group_barrier`, `s_setprio`, `s_waitcnt` |
| §9.3 cshuffle epilogue  | `epilogue="cshuffle"` in `UniversalGemmSpec`, `ImplicitGemmConvSpec` |
| §11 ISA inspection      | `compile_kernel(...).hsaco` -> write to disk -> `llvm-objdump -d` |
| §12 autotuning          | `ck_dsl.sweep` parallel build + cache + bench drivers |

See `ck/dsl/ck_dsl_current_results.md` for empirical perf numbers from
the runbook-driven optimisation pass.

## Why this exists

Three forces:

1. **Authoring productivity.** CK Tile in C++ requires deep template
   meta-programming and hipcc round-trips (200-2000 ms per kernel
   instance). Python authoring with in-process LLVM compilation cuts
   the inner loop to ~30-50 ms (or ~5 ms warm).

2. **Algebraic surface.** The CK Tile coordinate-transform DAG and the
   `tile_distribution_encoding` are powerful but tedious to wire up in
   templates. In Python they're concrete data objects, and the
   transforms compose naturally over SSA values.

3. **Non-bijective ops as first-class.** Layout-algebra DSLs (CuTe and
   its derivatives) are elegant for bijective layouts (GEMM, plain
   batched matmul) but awkward for non-bijective ones (convolution
   in/out maps, attention masks, scatter-gather). CK Tile's transform
   DAG handles those natively, and so does this DSL — see the
   implicit-GEMM bake-off result.

## Tests, sweeps, examples

```sh
# unit tests (34 passing in test_ck_dsl.py; 3 pre-existing failures in
# test_gen_instances need a CK library path not present in this workspace)
PYTHONPATH=python python3 -m pytest python/test/test_ck_dsl.py

# build + bench one example (Python-native runner, no C++ host launch)
PYTHONPATH=python python3 -m ck_dsl.examples.bake_off_implicit_gemm \
    --output-dir /tmp/ex08
sudo -n env PYTHONPATH=python python3 -m ck_dsl.run_manifest \
    /tmp/ex08/*.hsaco /tmp/ex08/manifest.json --verify

# parallel build-on-the-fly sweep over the dispatcher's hero subset
PYTHONPATH=python python3 example/ck_tile/dsl/07_gemm_universal_sweep/gen.py \
    --output-dir /tmp/sweep --subset compute --parallel 16
PYTHONPATH=python python3 -m ck_dsl.sweep_bench /tmp/sweep/sweep_manifest.json \
    --attempts 3 --csv /tmp/sweep/results.csv
```
