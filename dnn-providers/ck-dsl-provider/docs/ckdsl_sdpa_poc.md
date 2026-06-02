# CK DSL Dispatcher SDPA POC (ALMIOPEN-2002)

Developer guide for the forward-SDPA path in the `ck-dsl-provider`: a hipDNN provider that
drives an **unmodified** CK DSL unified paged attention kernel, selecting the kernel's perf
config via the CK-Tile dispatcher's ML heuristic (`FmhaMLHeuristic::predict_tflops`).

**Status:** validated — numerically correct on gfx950 (8/8), perf characterized. See
"Status & results" below. Target arch **gfx950**; all non-GPU layers run on gfx90a.

> Hard rule: **torch is not a dependency** of the provider, the generation/compile path, or any
> in-tree test. The PyTorch perf baseline (below) is an *external* benchmark run from a separate
> venv.

## Pipeline

```
hipDNN SDPA-fwd graph
  → SdpaAdapter::buildSpec        honest capability gate (accept broad-but-safe / decline cleanly)
  → SdpaFwdPlanBuilder::buildPlan enumerate knob combos → FmhaMLHeuristic argmax (analytic fallback)
  → compile_service._compile_sdpa_fwd_unified  build UnifiedAttention2DTiledSpec → comgr → gfx950 HSACO
  → SdpaFwdPlan::execute          marshal block_tables/cu_seqlens/seqused → 18-slot ABI → launch
```

---

## Quickstart

All paths are relative to the worktree root `worktrees/rocm-libraries/<branch>/`.

### 1. Configure + build (superbuild)

```bash
# One-time: a venv with pybind11 (embedded interp) + lightgbm (scorer backend).
python3 -m venv .venv
.venv/bin/pip install pybind11 lightgbm

# Configure the superbuild for hipdnn + ck-dsl-provider. CMAKE_HIP_ARCHITECTURES
# defaults to "gfx90a;gfx950" if unset (the provider enables the HIP language).
cmake --preset ck-dsl-provider -B build-superbuild \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_INSTALL_PREFIX="$PWD/../../../WIP/worktrees/rocm-libraries/<branch>/superbuild/install"

ninja -C build-superbuild
```

The `ck-dsl-provider` preset = `ROCM_LIBS_ENABLE_COMPONENTS "hipdnn;ck-dsl-provider"`. LightGBM is
found via the venv's site-packages (`lightgbm/lib/lib_lightgbm.so`) and linked onto the plugin.

### 2. Run tests

Set the runtime lib path once (HIP + the venv's LightGBM):

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$PWD/.venv/lib/python3.12/site-packages/lightgbm/lib:$LD_LIBRARY_PATH
```

**Host C++ unit tests (no GPU needed except the GPU-gated subset):**
```bash
ninja -C build-superbuild ck_dsl_provider_unit_tests      # NOTE: ck-dsl-provider-unit-check (ctest) does NOT rebuild the binary
./build-superbuild/bin/ck_dsl_provider_unit_tests
```

**Torch-free kernel *generation* matrix (CPU-only, runs anywhere; cross-compiles to gfx950 HSACO):**
```bash
.venv/bin/python -m pytest dnn-providers/ck-dsl-provider/python/tests/   # conftest self-bootstraps sys.path
```

**GPU plumbing proof (any AMD GPU, incl. gfx90a — uses a fake 18-arg kernel):**
```bash
./build-superbuild/bin/ck_dsl_provider_unit_tests --gtest_filter='*SdpaFwdFakeLaunch*'
```

**gfx950-only (skip elsewhere): correctness, perf, oracle.** Use `HIPDNN_LOG_LEVEL=info` to see
perf lines.
```bash
B=./build-superbuild/bin/ck_dsl_provider_integration_tests
$B --gtest_filter='*SdpaFwdFp16*'                          # correctness vs CPU reference (8 cases)
HIPDNN_LOG_LEVEL=info $B --gtest_filter='*SdpaFwdPerf*'    # dense throughput sweep (TFLOPS)
HIPDNN_LOG_LEVEL=info $B --gtest_filter='*Paged*:*Varlen*' # paged + varlen throughput (perf-only, no oracle)
HIPDNN_LOG_LEVEL=info $B --gtest_filter='*Oracle*'         # oracle config sweep (heuristic vs best); slow
$B --gtest_filter='*BuildTime*'                            # first-use / JIT-compile latency (prints [BuildTime] lines)
```

### 3. PyTorch perf baseline (external — separate venv, torch NOT in the hipDNN tree)

```bash
python3 -m venv torch-bench-venv
torch-bench-venv/bin/pip install --index-url https://rocm.nightlies.amd.com/v2/gfx950-dcgpu/ torch numpy
torch-bench-venv/bin/python torch_sdpa_bench.py    # mirrors the C++ perf shapes + FLOPS denominator
```
The `torch_sdpa_bench.py` script lives in the workspace WIP area (kept out of the repo by design);
it times `F.scaled_dot_product_attention(is_causal=True)` with the same shapes/denominator as
`IntegrationGpuCkDslSdpaFwdPerf` so the TFLOPS are directly comparable.

---

## Utilities we created

**Provider (C++) — `dnn-providers/ck-dsl-provider/src/`:**
- `adapters/sdpa/SdpaPerfKnobs.hpp` — POD of the tiled-2D kernel perf knobs.
- `adapters/sdpa/SdpaCandidateSelector.{hpp,cpp}` — `supportsTiled2d` C++ mirror, knob-first
  enumerator, knobs→`FmhaKernelKey` mapping, injectable-score `selectArgmax`, analytic fallback,
  `selectPerfKnobs`.
- `adapters/sdpa/SdpaScorer.{hpp,cpp}` — HIP-free pimpl over `FmhaMLHeuristic` (LightGBM
  `predict_tflops`); `.cpp` is the only HIP TU.
- `adapters/sdpa/SdpaAdapter.{hpp,cpp}` — the honest capability gate.
- `adapters/sdpa/SdpaMarshalling.{hpp,cpp}` — host marshalling (block_tables / cu_seqlens /
  seqused) for dense-degenerate / paged / varlen.
- `engines/sdpa/SdpaFwdPlanBuilder.cpp` — the scoring call; `engines/sdpa/SdpaFwdPlan.cpp` — the
  18-slot ABI launch (dense path wired; paged/varlen TODO).
- `graph/GraphSignature.cpp` — SDPA cache key (folds knobs + problem lanes).

**Compile service (Python) — `dnn-providers/ck-dsl-provider/python/ck_dsl_provider/compile_service.py`:**
- `_compile_sdpa_fwd_unified` — the unified op_kind path (problem+spec → `build_unified_attention_2d_tiled`
  → comgr → 18-slot arg_schema + grid).
- `compile_sdpa_fwd_fake` — trivial 18-arg kernel for the GPU plumbing test.

**Tests / harnesses:**
- `tests/SdpaCandidateSelectorTest.cpp`, `tests/SdpaAdapterTest.cpp`, `tests/SdpaMarshallingTest.cpp`,
  `tests/SdpaScorerTest.cpp`, `tests/GraphSignatureTest.cpp` — host unit coverage.
- `python/tests/test_sdpa_generation.py` — torch-free generation matrix (HSACO/ABI/grid/IR).
- `tests/SdpaFwdFakeLaunchTest.cpp` — full execute() plumbing proof on any GPU (fake kernel).
- `integration_tests/IntegrationGpuCkDslSdpaFwdFp16.cpp` — gfx950 correctness vs CPU reference.
- `integration_tests/IntegrationGpuCkDslSdpaFwdPerf.cpp` — gfx950 throughput sweep (dense), plus
  the perf-only `Paged_Identity` / `Paged_Scatter` / `Varlen_Mixed` cases.
- `integration_tests/IntegrationGpuCkDslSdpaFwdOracle.cpp` — gfx950 oracle config sweep.
- `integration_tests/IntegrationGpuCkDslSdpaFwdBuildTime.cpp` — gfx950 first-use / JIT-compile
  latency probe (separates one-time warmup, per-shape cold compile, and JitCache hit).

---

## Status & results (MI350X / gfx950)

- **Correctness:** 8/8 — fp16 & bf16 × head {64,128,256} × {MHA,GQA}, causal, vs CPU reference.
  (Dense path only — there is no CPU reference for paged/varlen, so those are perf-only.)
- **Perf vs PyTorch flash** (aligned hipEvent harness, full `4·B·Hq·S²·D`): dispatcher pick
  ~0.31–0.52× of flash, plateauing ~250 TFLOPS.
- **Paged + varlen (wired, perf-only):** the unified kernel is always paged; real-paged binds the
  graph's `Page_table_K` device buffer directly to the block-table slot, varlen reads per-sequence
  lengths via a small in-`execute()` D2H. Measured (B4 GQA D128 S2048): `Paged_Identity` **245
  TFLOPS** (contiguous table — parity with the dense B4/S2048 246, confirming the passthrough
  plumbing), `Paged_Scatter` **245** (reverse-permutation table — block-table indirection adds no
  measurable overhead at this shape), `Varlen_Mixed` **228** (bf16, lengths {S,S/2,S,S/4}, actual-
  token denominator). No correctness oracle exists for these paths.
- **Oracle sweep (key finding):** the best enumerated config is **~2× faster than the
  heuristic's pick** at S8192 (503 vs 247 TFLOPS), reaching ~0.64–0.73× of flash. The heuristic
  picks the *smallest* config (nw1/mw16) for every shape → the gap is a **config-selection**
  problem, not a kernel ceiling.
- **Scoring-query fidelity fix + dispatcher-vs-analytic (decisive):** the heuristic was querying the
  model out-of-distribution (pipeline pinned to the OOV `qr_pagedkv`, stub `tile_k0`, paged flag the
  fwd model never saw). We made the scoring query trained-faithful (score `qr_async`, real
  `k0/k1/n1/k0max`, dense scoring flag) — a scoring-only change (kernel + JIT cache unchanged). **It
  did NOT change the pick:** the heuristic still selects nw1/mw16 for every shape, *including
  in-family* bf16/d64/gqa8, where it is actually **worse than the analytic baseline** (0.90×). Oracle
  vs heuristic vs analytic:

  | shape | heuristic | oracle-best | oracle/heur | analytic | heur/analytic |
  |---|---|---|---|---|---|
  | fp16 S8192 D128 | 247 | 503 (nw4/mw32/mfma32) | 2.03× | 216 | 1.15× |
  | fp16 S2048 D128 | 227 | 324 (same) | 1.42× | 192 | 1.19× |
  | bf16 in-family D64 S2048 | 290 | 452 (same) | 1.56× | 322 | **0.90×** |

  Conclusion: the gfx950 model was trained on CK-Tile **assembly** fwd kernels and **does not
  transfer** to the CK DSL unified kernel — it ranks small tiles highest when our kernel wants large
  tiles. A faithful query is necessary (and a prerequisite for any working model) but **insufficient**.
  The real lever is a **retrain/autotune on CK-DSL-measured TFLOPS** (the oracle harness already
  produces exactly that data; oracle-best is the same nw4/mw32/mfma32 config across all shapes tested).

### First-use latency (JIT / `buildPlan`)

`buildPlan` (capability gate + dispatcher scoring + comgr JIT compile + `HipModule` load) is paid
at plan-build time, **separate from** the per-launch kernel time the TFLOPS numbers report
(`execute()` is ~0.3 ms at S2048 to ~4.5 ms at S8192). Measured by `*BuildTime*`:

| phase | cost | what it is |
|-------|------|------------|
| First call in the process | **~5.9 s** | one-time warmup: LightGBM model load (~11 MB) + comgr / embedded-CPython init + first compile |
| Steady-state cold compile (each **new** shape) | **~105–115 ms** | comgr lowering+compile of the gfx950 HSACO — effectively shape-independent |
| JitCache hit (repeat shape) | **~2 ms** | module is cached by signature; this is just the dispatcher **re-scoring**, which runs on every `buildPlan` |

Implications for users: the first SDPA call in a process eats ~6 s of one-time init; each distinct
shape then costs ~110 ms to JIT once; thereafter plan-build is ~2 ms — and that residual is
re-scoring (enumerate + LightGBM predict), not recompile, so it's cacheable if it ever matters.
The compile cost amortizes after a handful of launches per shape.

### Open follow-ups
1. **CK-DSL-native heuristic retrain (the lever — now confirmed required).** The scoring query is
   trained-faithful but the gfx950 CK-Tile-assembly model does not transfer to the DSL kernel
   (still picks nw1/mw16; ≈/worse than analytic). Retrain a model on CK-DSL-measured TFLOPS — the
   oracle sweep already emits per-(shape,config) TFLOPS, the exact training signal. Closes the ~2×.
2. `num_warps=8` + large-tile comgr CODEGEN failure (21/51 oracle configs); tighten the
   enumerator's `supportsTiled2d` mirror so it doesn't emit non-compilable configs.
3. Kernel feature gaps (further work needed): non-causal mode; LSE output (paged kernel emits none).
4. Real-paged + varlen `execute()` launch branches are now wired and perf-measured (dense,
   real-paged, varlen, paged+varlen). A CPU reference for paged/varlen (to add *correctness*
   coverage beyond perf) remains future work.

Full technical writeup + perf tables: workspace `Plans/almiopen-2002-writeup-DRAFT.md`.
