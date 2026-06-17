# PR #8503 Update — 2026-06-16

## execute() wiring + integration test stubs

Two remaining issues from Brian's review are now addressed:

### Issue 1 FIXED: `launch_flash2_v7()` now connected to `execute()`

New files added:

| File | Purpose |
|------|---------|
| `HipFlash2KernelUtils.hpp` | `HipModuleGuard` + `loadKernelModule` + `launchFlash2Kernel` — mirrors `SdpaKernelUtils.hpp` |
| `HipFlash2FwdPlan.hpp/cpp` | `IPlan::execute()` maps UID→ptr, fills `Flash2KernelArgs`, dispatches via `hipModuleLaunchKernel` |
| `HipFlash2FwdPlanBuilder_v2.hpp` / `HipFlash2FwdPlanBuilder.cpp` | Full `IPlanBuilder`: `isApplicable`, `buildPlan` loads arch `.co` via `hipModuleLoad`, creates `HipFlash2FwdPlan` |
| `HipFlash2Engine.cpp` (fixed) | `initializeExecutionContext` now calls `executionContext.setExecutionSettings(Settings{})` then `pb->buildPlan(...)` — matching `AsmSdpaEngine` pattern exactly |

`buildPlan()` flow:
1. `getDeviceString()` → `"gfx942"` or `"gfx950"`
2. `flash2CoPath(archId)` → `${HIP_FLASH2_KERNEL_DIR}/hip_flash2_fwd_{arch}.co`
3. `loadKernelModule(coPath, "flash2_v7_hipdnn_d128")` → `HipModuleGuard`
4. `executionContext.setPlan(make_unique<HipFlash2FwdPlan>(kernel, params))`

### Issue 2: Integration test golden reference graph descriptors added

Under `dnn-providers/integration-tests/golden_reference_data/quick/SdpaFwd/bhsd/fp16/`:
- `hd128_causal_mha/Small/` — B=2 H=32 S=2048 D=128 causal (validated: MI300X 71.27 TFLOPS, MI325X 78.98 TFLOPS, MI355X 153.83 TFLOPS)
- `hd128_causal_gqa4/Small/` — B=2 H_q=32 H_kv=8 S=4096 D=128 GQA-4 causal
- `hd64_causal/Small/` — B=2 H=32 S=2048 D=64 causal

### Outstanding: DVC binary tensor data

`.tensors.dvc` manifests are placeholder stubs. Actual `.tensor{N}.bin` files need
`generate_sdpa_fwd_golden.py --dtype fp16` run on Alola (PyTorch>=2.12+rocm, `SDPBackend.MATH`)
and `dvc push`. Will Daryl's GPU reference work cover this, or should we run on Alola?
