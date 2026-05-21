# CK DSL hipDNN Provider — Session Handoff

**Resume on:** node with gfx950 (MI350-series) GPU access.

## Where things stand

Branch: `users/dahawkin/ck-dsl-provider`, cut from `users/vanantha/ck-dsl-prototype`.
HEAD: `0fcec098917`.

```
0fcec098917 [CK DSL] Hold GIL while releasing CompileServiceBridge module ref.
2fa22bc9c8e [CK DSL] Add Python compile-service bridge (M1 step I-3).
e980348fcab [CK DSL] Embed CPython interpreter in provider plugin (M1 step I-2).
73116d68404 [CK DSL] Add provider skeleton (M1 step I-1).
bf7546ed99e [CK DSL] Add hipDNN provider implementation plan.   ← branch base
```

Working tree clean. `WIP/` is untracked and contains logs + the prep findings doc.

## Plan progress (per plan v0.9 §6.2)

| Step | Status |
|---|---|
| Prep P-1 … P-7 | done, synthesis in `WIP/prep_findings/PREP_FINDINGS.md` |
| I-1 skeleton | ✅ committed |
| I-2 embedded interpreter | ✅ committed |
| I-3 compile-service bridge | ✅ committed (+ reviewer GIL-dtor fix) |
| I-4 KernelArtifact / HipModule round-trip | **next — needs GPU** |
| I-5 JitCache | needs GPU |
| I-6 ConvImplicitGemmAdapter + Spec | host-only, can run pre-GPU |
| I-7 PlanBuilder | needs GPU |
| I-8 Plan::execute | needs GPU |
| I-9 PerfMeasurement | needs GPU |
| I-10 Integration test | needs GPU |
| I-11 CI / pre-commit clean | terminal step |

## Hardware constraint discovered this session

**ck_dsl is gfx950-only.** Defaults in `runtime/comgr.py:210`, `helpers/compile.py:68,82,129`, and `examples/bake_off_implicit_gemm.py:44` all hardcode `amdgcn-amd-amdhsa--gfx950`. The DSL emits gfx950-specific instructions broadly (MFMA 32×32×16 fp16, `ds_swizzle_b32`, `v_permlane32_swap_b32`, `ds_read_b64_tr_b{8,16}`, scaled FP8/BF8 converts), not just MFMA hot paths. M1 hardware target is **MI350-series**. No fallback to MI300/MI250.

## What I-4 needs to do (next step)

Per plan §6.2 step I-4 — KernelArtifact / HipModule round-trip from a known blob:

1. **Python side** — extend `dnn-providers/ck-dsl-provider/python/ck_dsl_provider/compile_service.py` with a `compile_smoke()` that uses `ck_dsl.helpers.compile.compile_kernel` to produce a trivial HSACO (no MFMA, but isa still gfx950) and returns bytes + minimal launch metadata.
2. **C++ side** under `dnn-providers/ck-dsl-provider/src/runtime/`:
   - `KernelArtifact.{hpp,cpp}` — struct per P-1 recommendation in `WIP/prep_findings/PREP_FINDINGS.md`.
   - `HipModule.{hpp,cpp}` — RAII wrapper around `hipModule_t` + `hipFunction_t`.
   - `LaunchAbi.{hpp,cpp}` — schema-driven arg packing (replaces the launcher.cpp's hardcoded-per-kind packing). The pack signature recommendation is in P-1.
3. Extend `CompileServiceBridge` with a `compileSmoke()` method that calls into the Python side and returns a `KernelArtifact`.
4. Unit test: load the HSACO via `hipModuleLoadData`, get the kernel function, launch over a 1-element buffer, verify no `hipError`. Gate on `hipGetDeviceCount() > 0` so the host-only CI lane stays green.

After I-4 → I-7 inherits the runtime layer; I-6 (adapter) is independent and can run in parallel.

## Files the next session should read first

Listed in dependency order (fastest on-ramp):

1. `WIP/STATUS.md` (this file)
2. `WIP/prep_findings/PREP_FINDINGS.md` — all design decisions + the P-1 `KernelArtifact` shape that I-4 will instantiate
3. `projects/composablekernel/python/ck_dsl/dsl_docs/hipdnn_provider/plan.md` v0.9 — especially §6.2 step I-4 and §6.5 risk register
4. The four existing commits' diffs — for the established patterns (GIL discipline, sys.path injection, test conventions)

## Gotchas to carry forward

- **CMake `Python3` discovery** must keep the pin to `/usr/bin/python3` with `Python3_FIND_STRATEGY=LOCATION` — otherwise a uv-managed Python in `~/.local` hijacks it. Already in `dnn-providers/ck-dsl-provider/CMakeLists.txt`.
- **cmake-lint docstring convention** is `# ` line comments *immediately preceding* `function(...)`, not `#[==[ ]==]` and not inside the body. VersionUtils.cmake style.
- **pybind11 dtor with `py::object` members** must acquire the GIL. Pattern in `CompileServiceBridge::~CompileServiceBridge()` is the template — replicate for any future class that holds `py::object`/`py::module_`/`py::dict` members.
- **ck_dsl has no `__version__`** and **no `pyproject.toml`** — cache-key uses git SHA (already baked into the version header via `CkDslProviderVersion.cmake`); package discovery uses CMake-baked `sys.path` prepend (`ckdsl_provider_paths.h`).
- **Engine naming is per-op** (v0.9 amendment): `CkDslConvImplicitGemmEngine`, not `CkDslEngine`. M5+ adds siblings, not a refactor.
