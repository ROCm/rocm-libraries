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
HIPDNN_LOG_LEVEL=info $B --gtest_filter='*SdpaFwdPerf*'    # throughput sweep (TFLOPS)
HIPDNN_LOG_LEVEL=info $B --gtest_filter='*Oracle*'         # oracle config sweep (heuristic vs best); slow
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
- `integration_tests/IntegrationGpuCkDslSdpaFwdPerf.cpp` — gfx950 throughput sweep.
- `integration_tests/IntegrationGpuCkDslSdpaFwdOracle.cpp` — gfx950 oracle config sweep.

---

## Status & results (MI350X / gfx950)

- **Correctness:** 8/8 — fp16 & bf16 × head {64,128,256} × {MHA,GQA}, causal, vs CPU reference.
- **Perf vs PyTorch flash** (aligned hipEvent harness, full `4·B·Hq·S²·D`): dispatcher pick
  ~0.31–0.52× of flash, plateauing ~250 TFLOPS.
- **Oracle sweep (key finding):** the best enumerated config is **2.05× faster than the
  heuristic's pick** at S8192 (504 vs 246 TFLOPS), reaching ~0.64–0.73× of flash. The heuristic
  degenerately picks the *smallest* config for every shape → the gap is a **config-selection**
  problem (retrain/autotune), not a kernel ceiling.

### Open follow-ups
1. CK-DSL autotuner + heuristic retrain over realistic shapes (closes the ~2×).
2. `num_warps=8` + large-tile comgr CODEGEN failure (21/51 oracle configs); tighten the
   enumerator's `supportsTiled2d` mirror so it doesn't emit non-compilable configs.
3. Kernel feature gaps (raise with Vidya): non-causal mode; LSE output (paged kernel emits none).
4. Real-paged / varlen `execute()` launch branches (dense-degenerate is wired + verified).

Full technical writeup + perf tables: workspace `Plans/almiopen-2002-writeup-DRAFT.md`.
