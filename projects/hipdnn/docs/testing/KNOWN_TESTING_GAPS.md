# Known Testing Gaps

This file lists known limits in current hipDNN testing.

## Integration test bundles

Integration test bundles are off by default. See [Bundles are opt-in at runtime](https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/integration-tests/README.md#bundles-are-opt-in-at-runtime-for-now).

## Sanitizers and platforms

- Pull-request runs of the [ASAN workflow](../../../../.github/workflows/therock-multi-arch-ci-asan.yml) build sanitizer binaries but do not run their tests. Manual runs may test when a matching runner is available.
- The [device-ASAN nightly](../../../../.github/workflows/therock-multi-arch-ci-asan-nightly.yml) uses runner settings managed outside this repository. On 2026-08-17, `gfx94X` had a test runner, `gfx950` was build-only, and `gfx125X` had no supported ASAN build.
- Standalone ASAN support depends on the OS and GPU. Windows host ASAN does not provide the same coverage as Linux device ASAN. Some sample and provider convolution tests are still excluded because of upstream failures. See [sanitizer configuration](../../cmake/Sanitizers.cmake) and [build guidance](../Building.md#address-sanitizer-build).
- TSAN builds on Linux, but hipDNN has no verified TSAN CI job.
- The [release multi-arch workflow](../../../../.github/workflows/therock-multi-arch-ci.yml) is manual only.

## Code coverage

The 80% coverage number is a goal, not a merge requirement. Local [coverage targets](../../CMakeLists.txt) create reports but do not enforce a minimum. The repository has a hipDNN [Codecov configuration](../../../../codecov.yml), but checked-in workflows do not prove that uploads or coverage limits are required.

## Tested GPUs and supported GPUs

- A configured build does not prove that tests ran on a physical GPU. Some Linux targets are build-only, and external runner settings choose the actual hardware. See [TheRock CI](../../../../.github/workflows/therock-ci.yml), [Linux test workflow](../../../../.github/workflows/therock-ci-linux.yml), and [superbuild workflow](../../../../.github/workflows/hipdnn-superbuild-ci.yml).
- Passing CI on one GPU does not prove support for every GPU. Support can depend on the operation, engine, build options, libraries, and runtime checks. Use [hipDNN operation support](../OperationSupport.md) and provider documentation for support claims.

## Tracked provider test exceptions

Provider TOML and category files contain the current skips and tolerances. The issues below track known exceptions.

- **Shared integration tests:** BF16 RMSNorm reference cases are skipped while [#10560](https://github.com/ROCm/rocm-libraries/issues/10560) is open.
- **MIOpen:** [MIOPEN_ENGINE.toml](../../../../dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml) contains current tolerances and GPU-specific skips. Related issues: [#6979](https://github.com/ROCm/rocm-libraries/issues/6979), [#8029](https://github.com/ROCm/rocm-libraries/issues/8029), [#8030](https://github.com/ROCm/rocm-libraries/issues/8030), and [#6864](https://github.com/ROCm/rocm-libraries/issues/6864).
- **hipBLASLt:** [HIPBLASLT_ENGINE.toml](../../../../dnn-providers/hipblaslt-provider/config/HIPBLASLT_ENGINE.toml) contains the BF16 fused-matmul tolerance and gfx12 FP16 MatmulBias skips tracked by [#8033](https://github.com/ROCm/rocm-libraries/issues/8033).
- **HIP-kernel:** [HIP_MLOPS_ENGINE.toml](../../../../dnn-providers/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml) and [ASM_SDPA_ENGINE.toml](../../../../dnn-providers/hip-kernel-provider/config/ASM_SDPA_ENGINE.toml) contain current skips and tolerances. [#10497](https://github.com/ROCm/rocm-libraries/issues/10497) tracks disabled provider-mode rocKE Python tests.

## Failures, skips, and retries

- An enabled test failure fails its workflow. GitHub settings outside this repository decide whether that workflow blocks a merge.
- A green workflow means only that its selected tests passed after allowed skips, filters, tolerance changes, and retries. It does not mean every test ran. The [auto-retry workflow](../../../../.github/workflows/auto-retry-failed.yml) covers TheRock CI and two TheRock nightlies, but not superbuild or other jobs.

## Automated performance checks

hipDNN has no automated GPU performance gate. [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking#readme) can run workloads and report engine timing, but it does not save every workload and build detail and is not used by CI. No checked-in policy defines baselines, normal noise, reruns, or who owns regression triage.
