# ck-dsl-provider

A hipDNN engine plugin that executes **ck_dsl-generated** kernels with **no Python at
runtime**. It mirrors `ck-fmha-provider`, but the kernel backend is the bundled, dependency-light
`ck_dsl_runtime` (HIP + `libamd_comgr` only) instead of the CK-Tile C++ template dispatcher.

See the design RFC: `ck_dsl_hipdnn_rfc.md`.

## Architecture (the transparent 5-stage component model)

```
hipDNN graph ─▶ CkDslParamParser  (stage 1: graph → ck_dsl::Problem)
            ─▶ ck_dsl::Dispatcher (stage 2: Problem → cache_key, FirstFit)
            ─▶ ck_dsl::ArtifactStore (stage 3: cache_key → shipped .hsaco/.ll)
            ─▶ ck_dsl::Compiler (stage 4: comgr .ll→HSACO, only if not prebuilt)
            ─▶ ck_dsl::Kernel   (stage 5: pack kernarg + hipModuleLaunchKernel)
```

- **Path (a) "Fast Mode":** a prebuilt `.hsaco` ships → load + launch, ~0 cold start.
- **Path (b) "JIT fallback":** only a `.ll` ships → `libamd_comgr` compiles it in-process
  (~9 ms, cached) → launch. No Python, no subprocess.

`ck_dsl_runtime` is independently buildable + GPU-tested (`runtime/tests/test_runtime.cpp`).

## Layout
```
ck-dsl-provider/
├── CMakeLists.txt
├── cmake/Dependencies.cmake          # imports the hipDNN SDK from HIPDNN_ROOT
├── runtime/                          # ck_dsl_runtime (header-only, hip + comgr)
│   ├── include/ck_dsl_runtime/{json,manifest,comgr,kernel,artifact_store,dispatcher,runtime}.hpp
│   ├── tests/test_runtime.cpp        # standalone path-a + path-b GPU test
│   └── CMakeLists.txt
└── src/
    ├── CkDslPluginPublic.cpp         # EnginePluginImpl.inl entry point
    ├── CkDslContainer.{hpp,cpp}      # registers CK_DSL_GEMM_ENGINE
    ├── CkDslHandle.{hpp,cpp}         # arch detect + ArtifactStore + Dispatcher
    ├── CkDslContext.hpp / CkDslSettings.hpp
    └── engines/
        ├── CkDslParamParser.{hpp,cpp}        # Matmul graph → Problem
        ├── CkDslGemmEngine.{hpp,cpp}
        └── plans/CkDslGemmPlan{,Builder}.{hpp,cpp}
```

## Build

The build dir should be on a **local** filesystem (NFS makes gcc/comgr pathologically slow).

```bash
# 1) runtime library standalone (optional; proves the core on GPU)
cmake -S runtime -B /var/tmp/ckdsl_rt -G Ninja \
  -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_PREFIX_PATH=/opt/rocm
ninja -C /var/tmp/ckdsl_rt
/var/tmp/ckdsl_rt/ck_dsl_runtime_test <bundle_dir> 512 512 256 gfx950

# 2) the hipDNN plugin
cmake -S . -B /var/tmp/ckdsl_prov -G Ninja \
  -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DHIPDNN_ROOT="${HIPDNN_ROOT:-<repo>/projects/hipdnn}"
ninja -C /var/tmp/ckdsl_prov          # -> libck_dsl_provider_plugin.so
```

## Run

The plugin discovers its kernel bundle from `CK_DSL_KERNEL_LIB_PATH` (a directory of shipped
`manifest.json` + `.hsaco`/`.ll`, e.g. produced by `ck_dsl` `write_artifact` / the offline
generator). Install the `.so` into hipDNN's plugin dir and point `HIPDNN_PLUGIN_PATH` at it, then
run any SDPA/matmul graph — same flow as `ck-fmha-provider`'s EndToEnd demo.

## Environment variables

Main provider flags: `CK_DSL_KERNEL_LIB_PATH` (prebuilt HSACO bundle dir; leave
empty to force C-JIT), `HIPDNN_PLUGIN_PATH` (plugin location), `CK_DSL_C_JIT=1`
(generate kernels from C source at runtime), `CK_DSL_ALLOW_ENGINE_MISMATCH=1`
(downgrade the stale-bundle freshness check to a warning). The DSL build toolchain
also honours `CK_DSL_LLVM_FLAVOR` (`llvm22`/`llvm20`). **Full list of every
flag:** [`dsl_docs/reference/env_flags.md`](../../projects/composablekernel/python/ck_dsl/dsl_docs/reference/env_flags.md).

## Status

| Piece | State |
|---|---|
| `ck_dsl_runtime` (manifest, comgr, kernel, store, dispatcher) | **done + GPU-verified** (path a & b bit-exact) |
| GEMM engine (CK_DSL_GEMM_ENGINE): parser, plan builder, plan, container, handle, plugin C-ABI | **done; builds; dlopen smoke pass** |
| SDPA/FMHA engine (CK_DSL_ATTENTION_ENGINE): SdpaAttributes parser, engine, plan builder, plan | **done; builds; plugin exposes 2 engines; dlopen smoke pass** |
| SDPA dense→paged metadata synthesis (block_tables/seq_lens/query_start_len) + numerical verify | TODO (ck_dsl unified attention is paged-KV; structure + arg mapping + grid wired) |
| **GEMM end-to-end through hipDNN graph** (`integration_tests/EndToEndGemmDemo.cpp`) | **done; `graph.build()`+`graph.execute()` → `max_abs_diff=0` PASS** |
| **SDPA end-to-end through hipDNN graph** (`integration_tests/EndToEndSdpaDemo.cpp`) | **done; B=1 BSHD; dense→paged metadata synthesized in-plan; `max_abs_diff=4e-4` PASS** vs C++ causal reference |
| **Conv (fwd) end-to-end through hipDNN graph** (`integration_tests/EndToEndConvDemo.cpp`) | **done; implicit-GEMM NHWC×KRSC→NHWK; `max_abs_diff=3.8e-6` PASS** vs C++ reference |
| Kernel bundle shipped **in-provider** (`kernels/gfx950/{gemm*,attn,conv}/`) | done — HSACO + manifest + `.ll` per kernel |
| **Trained-model kernel selection** (LightGBM, mirrors CK Tile ML heuristic) | **done; `CK_DSL_ML_MODEL_PATH` → ranks by predicted TFLOPS; shape-dependent, differs from FirstFit; GEMM e2e PASS with model active** |

## Kernel selection (FirstFit → per-op trained models)

`ck_dsl_runtime::Dispatcher` supports two strategies (mirroring the CK Tile dispatcher):
- **FirstFit** (default): supported candidates by priority / largest tile.
- **Heuristic**: `ck_dsl_runtime/ml_heuristic.hpp::DslMlHeuristic` loads **per-op LightGBM models**
  and ranks candidates by **predicted TFLOPS**. Feature extraction dispatches by op, using the
  *exact feature layouts* of CK Tile's models — so CK Tile's trained models are reused directly:
  - **GEMM** → 72-feature `gemm_universal_fp16_gfx950` model.
  - **SDPA/FMHA** → 68-feature `fmha_fwd_gfx950` model (correctly separates prefill vs decode).
  - **Conv** → the GEMM model on the implicit-GEMM `(M=N·Ho·Wo, N=K, K=R·S·C)` problem.

The provider installs the heuristic when `CK_DSL_ML_MODEL_DIR` points at a directory of per-op models
(`<dir>/gemm/model_tflops.lgbm`, `<dir>/fmha/model_tflops.lgbm`); otherwise it uses FirstFit. Models
are loaded purely through the **LightGBM C API** (`LGBM_BoosterCreateFromModelfile` /
`PredictForMat` / `Free`) — **no Python at runtime** (training is offline). The plugin links
`lib_lightgbm`. Models ship in-provider under `kernels/<arch>/models/{gemm,fmha}/`. A standalone
demo (`ml_test.cpp`) prints per-op predicted TFLOPS.
| SDPA B>1 BSHD / BHSD-transpose / paged-KV-graph passthrough; conv shape-generalization | follow-on |
| Other op engines (moe, norm) | scaffolding TODO — mechanical per-family repetition |
| Offline `KernelLibraryGen` (enumerate ck_dsl.dispatch → per-arch bundle) | TODO (gemm `gen.py` already emits a valid 1-kernel bundle) |
| General matmul B-layout (vs ck_dsl RCR) | RCR assumed; stride-based detection TODO |
```
