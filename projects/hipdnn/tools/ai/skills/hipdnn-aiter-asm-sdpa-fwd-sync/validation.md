# Build and validation

Run these after `apply`, `verify` and the engine edits.

## 1. Build

Build the hip-kernel-provider with the `hipdnn-superbuild` skill (`../hipdnn-superbuild/SKILL.md`). Reconfigure first: the engine's CMake finds kernel files with globs that run only at configure time. The build regenerates the kernel registries and the `.kpack` archives; a packing failure stops the build.

## 2. Unit tests

Run the hip-kernel-provider unit tests with the `hipdnn-superbuild-test` skill (`../hipdnn-superbuild-test/SKILL.md`):

- Every update: the `TestAsmKpack*`, `TestAsmKernelPath`, `TestSdpaModuleCache`, `TestSdpaKernargRanges` and `TestAsmSdpaEngine` suites.
- Forward modules: the `TestSdpaFwd*` suites.
- Backward modules: the `TestSdpaBwd*` suites, among them `TestSdpaBwdOdoKernelArgs`, `TestSdpaBwdDqdkdvKernelArgs` and `TestSdpaBwdPostKernelArgs` (argument layout), `TestSdpaBwdRegistryLookup` (registry lookups per arch), `TestSdpaBwdPostStage`, `TestSdpaBwdPsskOffStrides`, `TestSdpaBwdPaddedHeadDims`, `TestSdpaBwdPlanGridMath` and `TestSdpaBwdPlanBuilder`.

`TestAsmKpackLoading` needs no GPU: it opens the built archive of every arch in `ASM_SDPA_GPU_ARCHS`, so it checks packing for all of them on any machine. `TestSdpaFwdPlanBuilder.IsApplicableAvailableKernels`, the backward builder tests that need a device, and the other device tests check only the device's own arch, or skip. Check the GPU with `rocminfo` before saying a machine has no AMD GPU.

## 3. GPU validation per arch

Every arch whose objects changed, and every arch whose launch code changed, needs the GPU tests of each updated direction to pass on a GPU of that arch. A port of shared host code (argument struct, packing, grid, dispatch) changes the launch code of every arch that vendors the module. Both directions' tests live in `dnn-providers/hip-kernel-provider/src/integration_tests/asm_sdpa_engine/`.

### Forward

`IntegrationGpuSdpaForward.cpp`: `IntegrationGpuSdpaFwdBf16` (instances `Smoke`, `SmokeFp8` and `SmokeStats`, built from the kernel registry) and `IntegrationGpuSdpaFwdShapeSweepBf16`. Filter on `*IntegrationGpuSdpaFwd*`.

- The registry cases check every kept row numerically: batch-mode rows against the CPU reference, group-mode rows against the CPU ragged reference. They use only equal query and key-value head counts, FNUZ fp8 inputs and the attention scale 1/sqrt(hdim_q).
- A numerical failure on a new kernel is not always a launch bug. Some kernels round intermediate values to bf16 where others keep fp32: the gfx950 hd256 kernels round Q × scale × log2(e) to bf16, so their error grows with the scale. Before changing launch code, emulate the kernel's rounding (read its disassembly) against the reference at the test's shape and scale.
- A rule the diff added for a case the registry does not exercise, such as GQA, needs a shape-sweep case.

### Backward

`IntegrationGpuSdpaBackward.cpp`: `IntegrationGpuSdpaBwdBf16` and `IntegrationGpuSdpaBwdFp16` (instances `Smoke` and `Comprehensive`), checked against a CPU backward reference. Filter on `*IntegrationGpuSdpaBwd*`.

- The cases come from fixed lists (`getSdpaBwdSmokeTestCases` and `getSdpaBwdComprehensiveTestCases`), not from the registry. A changed object is covered only where a case reaches it, and a new row, head dim, mask or accumulator is covered only after a case for it is added. List the kept rows no case reaches.
- The Verified Kernel Matrix in `asm/asm_kernels/README.md` and `K_VERIFIED_KERNELS` in `plans/SdpaBwdPlanBuilder.cpp` record which DQDKDV rows were calibrated against the CPU reference, keyed by arch, dtype, row hdim_q and row hdim_v. `isVerifiedDispatch` counts only psskddv rows run at their own head dims. Change the table and the array together, and only for rows whose cases passed on that arch.
- Group-mode (varlen) rows are not exercised by these cases. Say so in the report when no other check covers them.

### Where to run them

- If the local GPU is one of the updated archs, run them locally with `hipdnn-superbuild-test`.
- For the others, use the `hipdnn-trigger-ci` skill (`../hipdnn-trigger-ci/SKILL.md`) with the matching GPU families. It dry-runs first; dispatch only after the user approves the exact command. The branch must be pushed first, which needs the user's approval too.
- Report each arch and direction as passed, failed, or not run. Never call a kernel validated on an arch where it was not run.

## 4. Cross-provider suite

Use the `hipdnn-integration-testing` skill (`../hipdnn-integration-testing/SKILL.md`) for the `ASM_SDPA_ENGINE` lane of `dnn-providers/integration-tests`. Its forward and backward bundles carry `.support.json` claims per arch:

- A removed kernel: remove the claims that relied on it, or the lane reports a broken claim.
- A new arch or kept row: add claims only for bundles that passed on that arch.
- An unclaimed pass (`unclaimed_support`) means the engine now serves a bundle no claim names. Add the claim after review.

## 5. Checks before the PR

- `verify` exits 0.
- `pre-commit run --files` on the changed files passes.
- No Jira ID appears in code, comments, CSVs or provenance files.
