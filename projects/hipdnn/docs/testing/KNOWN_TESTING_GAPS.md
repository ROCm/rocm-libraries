# Known Testing Gaps

This file lists known limits in current hipDNN testing.

## Integration test bundles

Integration test bundles are off by default. See [Bundles are opt-in at runtime](https://github.com/ROCm/rocm-libraries/blob/develop/dnn-providers/integration-tests/README.md#bundles-are-opt-in-at-runtime-for-now).

## Sanitizers and platforms

- Pull-request runs of the [ASAN workflow](../../../../.github/workflows/therock-multi-arch-ci-asan.yml) build sanitizer binaries but do not run their tests. Manual runs may test when a matching runner is available.
- The [device-ASAN nightly](../../../../.github/workflows/therock-multi-arch-ci-asan-nightly.yml) uses runner settings managed outside this repository. On 2026-08-17, `gfx94X` had a test runner, `gfx950` was build-only, and `gfx125X` had no supported ASAN build.
- Standalone ASAN differs by platform. Windows uses host ASAN and disables STL container annotations because the x64 toolset lacks `stl_asan.lib`; heap, stack, global, and use-after-free checks remain enabled. Linux also enables device ASAN with `HSA_XNACK=1` and currently targets `gfx908`, `gfx90a`, and `gfx942`. See [sanitizer configuration](../../cmake/Sanitizers.cmake).
- Standalone ASAN disables four sample tests on Windows and Linux: convolution forward, data-gradient, weight-gradient, and serialization round-trip. Linux also disables three fused or deterministic convolution samples. See [sample registration](../../samples/CMakeLists.txt).
- Shared and MIOpen-provider convolution tests call `SKIP_IF_ASAN()` for known lower-level failures: a rocBLAS/Tensile heap-buffer-overflow on `gfx90a` and a CK ASAN stall on `gfx942`. See the [shared convolution test](../../../../dnn-providers/integration-tests/src/integration-tests/conv/IntegrationGpuConvForward.cpp) and [MIOpen provider test](../../../../dnn-providers/miopen-provider/integration_tests/IntegrationGpuBenchmarkingKnob.cpp).
- TSAN builds on Linux, but hipDNN has no verified TSAN CI job.

## Code coverage

The 80% coverage number is a goal, not a merge requirement. Local [coverage targets](../../CMakeLists.txt) create reports but do not enforce a minimum. The repository has a hipDNN [Codecov configuration](../../../../codecov.yml), but checked-in workflows do not prove that uploads or coverage limits are required.

## Static analysis

- Linux superbuild CI runs `clang-tidy` during compilation, but the Windows job disables it.
- The repository-wide [`clang-tidy` workflow](../../../../.github/workflows/clang-tidy.yml) does not include hipDNN. hipDNN therefore has no separate cross-platform static-analysis job.

## Tested GPUs and supported GPUs

- A configured build does not prove that tests ran on a physical GPU. Some Linux targets are build-only, and external runner settings choose the actual hardware. See [TheRock CI](../../../../.github/workflows/therock-ci.yml), [Linux test workflow](../../../../.github/workflows/therock-ci-linux.yml), and [superbuild workflow](../../../../.github/workflows/hipdnn-superbuild-ci.yml).
- Passing CI on one GPU does not prove support for every GPU. Support can depend on the operation, engine, build options, libraries, and runtime checks. Use [hipDNN operation support](../OperationSupport.md) and provider documentation for support claims.

## Automated performance checks

hipDNN has no automated GPU performance gate. [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking#readme) can run workloads and report engine timing, but it does not save every workload and build detail and is not used by CI. No checked-in policy defines baselines, normal noise, reruns, or who owns regression triage. WIP to get a weekly performance regression run ALMIOPEN-1908.
