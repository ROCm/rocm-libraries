# Build and validation

Run these after `apply`, `verify` and the engine edits.

## 1. Build

Build the hip-kernel-provider with the `hipdnn-superbuild` skill (`../hipdnn-superbuild/SKILL.md`). Reconfigure first: the engine's CMake finds kernel files with globs that run only at configure time. The build regenerates the kernel registries and the `.kpack` archives; a packing failure stops the build.

## 2. Unit tests

Run the hip-kernel-provider unit tests with the `hipdnn-superbuild-test` skill (`../hipdnn-superbuild-test/SKILL.md`), at least the `TestSdpaFwd*`, `TestAsmKpack*`, `TestAsmKernelPath`, `TestSdpaModuleCache`, `TestSdpaKernargRanges` and `TestAsmSdpaEngine` suites.

`TestAsmKpackLoading` needs no GPU: it opens the built archive of every arch in `ASM_SDPA_GPU_ARCHS`, so it checks packing for all of them on any machine. `TestSdpaFwdPlanBuilder.IsApplicableAvailableKernels` and the other device tests check only the device's own arch, or skip. Check the GPU with `rocminfo` before saying a machine has no AMD GPU.

## 3. GPU validation per arch

Every arch whose objects changed needs the forward GPU tests in `dnn-providers/hip-kernel-provider/src/integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp` to pass on a GPU of that arch: `IntegrationGpuSdpaFwdBf16` (instances `Smoke`, `SmokeFp8` and `SmokeStats`, built from the kernel registry) and `IntegrationGpuSdpaFwdShapeSweepBf16`. Filter on `*IntegrationGpuSdpaFwd*`.

- The registry cases validate every kept batch-mode row numerically, with equal query and key-value head counts and FNUZ fp8 inputs. They skip numerical validation of group-mode rows, so a new or changed group-mode kernel needs another check, such as a cross-provider bundle with ragged tensors. Say so in the report when there is none.
- A rule the diff added for a case the registry does not exercise, such as GQA, needs a shape-sweep case.

Where to run them:

- If the local GPU is one of the updated archs, run them locally with `hipdnn-superbuild-test`.
- For the others, use the `hipdnn-trigger-ci` skill (`../hipdnn-trigger-ci/SKILL.md`) with the matching GPU families. It dry-runs first; dispatch only after the user approves the exact command. The branch must be pushed first, which needs the user's approval too.
- Report each arch as passed, failed, or not run. Never call a kernel validated on an arch where it was not run.

## 4. Cross-provider suite

Use the `hipdnn-integration-testing` skill (`../hipdnn-integration-testing/SKILL.md`) for the `ASM_SDPA_ENGINE` lane of `dnn-providers/integration-tests`. Its forward bundles carry `.support.json` claims per arch:

- A removed kernel: remove the claims that relied on it, or the lane reports a broken claim.
- A new arch or kept row: add claims only for bundles that passed on that arch.
- An unclaimed pass (`unclaimed_support`) means the engine now serves a bundle no claim names. Add the claim after review.

## 5. Checks before the PR

- `verify` exits 0.
- `pre-commit run --files` on the changed files passes.
- No Jira ID appears in code, comments, CSVs or provenance files.
