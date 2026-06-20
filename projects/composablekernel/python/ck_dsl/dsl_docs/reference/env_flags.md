# Environment Variable Reference

Every environment variable the CK DSL, its C++ engine, the hipDNN provider, and
the tooling read. The **core** flags are the ones most users need; the rest are
build/CI, integration, or experimental kernel-development knobs that are off by
default. For setup and the most common flags in context, see
[`../development/setup_guide.md`](../development/setup_guide.md).

> Set on Linux with `export NAME=value`; on Windows with `set NAME=value`.

## Core engine

| Variable | Values (default) | Purpose |
|---|---|---|
| `CK_DSL_BACKEND` | `cpp` \| `python` \| `both` (**cpp**) | Which engine lowers Python-authored kernels. `cpp` = C++ engine (auto-falls back to Python if `ckc_engine` isn't built); `python` = native lowerer; `both` = run both and assert byte-identical (the differential check). |
| `CK_DSL_LLVM_FLAVOR` | `llvm22` \| `llvm20` (auto) | Force the LLVM IR flavor (datalayout/intrinsics). Auto: PyTorch's bundled ROCm version → `/opt/rocm` → `llvm22`. Set `llvm22` when your `comgr` is 7.2 but `/opt/rocm` is older, to avoid a `COMPILE_SOURCE_TO_BC` rejection. |
| `CK_DSL_CPP_STRICT` | `1` (unset) | Make `cpp` backend **raise** instead of silently falling back to Python when `ckc_engine` is unavailable. |
| `CK_DSL_DEBUG` | `1` (unset) | Verbose engine diagnostics during build/lowering. |
| `CK_DSL_TIME` | `1` (unset) | Print phase timings for the build/lower/compile pipeline. |
| `CK_DSL_USE_SUDO` | `1` (unset) | Benchmark/sweep harness launches kernels via `sudo -n -E` (for boxes where the user lacks GPU device-group access). |

## GEMM dispatch

| Variable | Values (default) | Purpose |
|---|---|---|
| `CK_DSL_GEMM_SPLIT_K` | `auto` \| `off` \| `<n>` (**auto**) | Split-K degree for CDNA UniversalGemm dispatch. `auto` runs the selection heuristic (engages split-K only for skinny / tall-N decode shapes whose grid leaves the CU-rich device idle; `1` for shapes that already fill it, keeping the default/square path byte-identical). `off` (or `0`/`1`) forces split-K disabled. `<n>` (>= 2) forces that degree, snapped down to the largest factor that evenly slices K. When engaged, the dispatcher launches a `(N_tiles, M_tiles, split_k)` grid and the caller must zero-init an f32 `[M, N]` workspace before launch and cast it back to the output dtype after. |

## hipDNN provider

| Variable | Values (default) | Purpose |
|---|---|---|
| `CK_DSL_C_JIT` | `1` (unset) | Provider generates kernels from C source at runtime (C-JIT) instead of loading prebuilt HSACOs. |
| `CK_DSL_PROVIDER_C_JIT` | CMake `ON`/`OFF` | Build-time switch compiling the provider's C-JIT path. |
| `CK_DSL_ALLOW_ENGINE_MISMATCH` | `1` (unset) | Downgrade the engine build-id freshness check from a hard error to a warning on a stale/mismatched kernel bundle (default: fail loudly). |
| `CK_DSL_KERNEL_LIB_PATH` | path (unset) | Directory of the prebuilt HSACO kernel bundle the provider loads (Fast mode); leave empty to force C-JIT. |
| `CK_DSL_ML_MODEL_DIR` | path (unset) | Directory of the ML heuristic models used for kernel selection. |
| `HIPDNN_PLUGIN_PATH` | path (unset) | Where hipDNN locates the ck-dsl-provider plugin. |

## Tooling, CI & remote

| Variable | Purpose |
|---|---|
| `CKC_PARITY_BUILD` / `CKC_PARITY_EMIT` | The binding-parity harness's `.so` dir and prebuilt-emitter dir — **must be the same fresh build** (mixing reports false mismatches). |
| `CKC_PARITY_ALLOW_STALE` | Allow the parity harness to run with a build-id mismatch (debug only). |
| `CKDSL_CI_RUN_DIR` | Out-of-tree directory for local/cluster CI runs (keeps build products off the source tree). |
| `CKDSL_ROOT` | Override the detected repository root. |
| `CKDSL_REMOTE_HOST` / `CKDSL_REMOTE_USER` / `CKDSL_REMOTE_KEY` / `CKDSL_REMOTE_STAGE` | Multi-arch remote-test orchestrator: SSH host/user/key and the staging dir. |
| `CKDSL_SLURM_EXTRA` | Extra SLURM flags appended by the multi-arch submitter. |
| `CKDSL_DOCKER_IMAGE` / `CKDSL_DOCKER_EXTRA_FLAGS` | Container image and extra flags for containerized runs. |
| `ROCM_PATH` | ROCm install prefix override (when not `/opt/rocm`). |
| `LLVM_OBJDUMP` / `LLVM_READELF` | Paths to the LLVM tools the ISA/resource probes shell out to. |

## Case-study-specific flags

Some flags are specific to one example/case study and are documented there, not
here: the `ATOM_USE_DSL_GEMM` / `ATOM_USE_DSL_ATTENTION` / `ATOM_DSL_GEMM_MAX_M`
/ `ATOM_DSL_GEMM_DEBUG` torch.compile-routing flags live in
`examples/gfx950/qwen3_30b_a3b/README.md`, and `AITER_PATH` (an external-baseline
checkout) is documented in the example READMEs that use it (e.g.
`examples/gfx950/attention/README.md`, `examples/gfx950/moe/README.md`).

## Experimental kernel-development knobs (off by default)

These tune or diagnose specific kernel paths. **Leave unset** unless you are
working on that path; they are read at the relevant builder/instance and several
are diagnostics that intentionally change emission. None affect the default build.

| Family | Variables | Area |
|---|---|---|
| MoE / preshuffle | `CK_DSL_PRESHUFFLE_W_DOWN`, `CK_DSL_PRESHUFFLE_W_GATE_UP_PACKED`, `CK_DSL_PRESHUFFLE_W_GATE_UP_INTERLEAVED`, `CK_DSL_ACTIVE_TILE_SKIP_GEMMS` | MoE weight layout / active-tile-skip |
| FP8 MoE mega-kernel | `CKDSL_FP8_AGPR_MFMA_DOWN`, `CKDSL_FP8_MFMA_CLUSTER`, `CKDSL_FP8_MFMA_NOP`, `CKDSL_FP8_SCHED`, `CKDSL_FP8_XCD`, `CKDSL_FP8_X_DTLA` | FP8 MoE scheduling / MFMA experiments |
| matmul-nbits | `CKDSL_NBITS_DEBUG` | quantized-GEMM diagnostics |
| Warp-specialized pipeline | `CK_WSP3_ASYNC`, `CK_WSP3_DEPTH`, `CK_WSP3_DRAIN`, `CK_WSP3_SCHED` | producer/consumer GEMM pipeline params |
| Swizzle study | `CK_SWZ_L`, `CK_SWZ_R`, `CK_SWZ_W` | LDS swizzle levels (left/right/write) |
| Attention | `CK_DSL_ATTENTION_COMPILE_BACKEND`, `CK_DSL_ATTENTION_AGPR_ALLOC_ZERO` | attention compile backend / AGPR alloc |
| gfx942 attention tuning | `HIPDNN_GFX942_NUM_WARPS`, `HIPDNN_GFX942_WAVES_PER_EU`, `HIPDNN_GFX942_IGLP`, `HIPDNN_GFX942_Q_DIRECT`, `HIPDNN_GFX942_Q_MAJOR_GRID`, `HIPDNN_GFX942_K_LDSSEQ`, `HIPDNN_GFX942_K_SLICED_RING`, `HIPDNN_GFX942_KV_CACHE_POLICY`, `HIPDNN_GFX942_GLOBAL_LOAD_LDS_K`, `HIPDNN_GFX942_SWIZZLE_VLDS`, `HIPDNN_GFX942_FLASH_WIDE` | experimental gfx942 FMHA levers |
| gfx942 V-transpose-store diagnostics | `HIPDNN_GFX942_CFV`, `HIPDNN_GFX942_CFV_CK_VLDS`, `HIPDNN_GFX942_CFV_SCALAR_READ`, `HIPDNN_GFX942_CFV_STORE`, `HIPDNN_GFX942_CFV_STORE_PREZERO`, `HIPDNN_GFX942_CFV_STORE_SCALAR_LOAD`, `HIPDNN_GFX942_CFV_STORE_SCATTER`, `HIPDNN_GFX942_CFV_STORE_SEPOFF`, `HIPDNN_GFX942_CFV_STORE_SPLIT` | gfx942 cfv-store debug toggles (some intentionally change emission) |

A flag not listed here that you find in the source is, by definition, an
internal experimental knob — treat it as off-by-default and read its call site.
