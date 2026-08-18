# Known Testing Gaps

This document records current limitations in our testing.

## Integration bundle execution

- Bundle tests are the preferred authoring format but remain runtime opt-in; checked-in provider CMake and workflows do not enable them. Track current mechanics in the [provider integration guide](../../../../dnn-providers/integration-tests/README.md), [shared categories](../../../../dnn-providers/integration-tests/test_categories.yaml), and provider category files.

## Sanitizers and platforms

- Sanitizer builds and test execution are asymmetric. The [ASAN workflow](../../../../.github/workflows/therock-multi-arch-ci-asan.yml) provides opt-in PR and manual builds: PR-triggered sanitizer runs are build-only, while manual dispatch may test where sandbox mapping exists. The [device-ASAN nightly](../../../../.github/workflows/therock-multi-arch-ci-asan-nightly.yml) uses the same external mapping. As verified on 2026-08-17, `gfx94X` has a sandbox test assignment, `gfx950` is build-only, and `gfx125X` lacks a supported ASAN build variant; this external mapping can change.
- Standalone ASAN has architecture and platform constraints, and Windows host instrumentation is not equivalent to Linux device-ASAN coverage. Some standalone sample and provider convolution cases remain excluded because of upstream failures. [Sanitizer configuration](../../cmake/Sanitizers.cmake), [build guidance](../Building.md#address-sanitizer-build), and test registration remain the sources of truth.
- TSAN is a Linux build capability, but no hipDNN TSAN CI lane or verified CI test execution is checked in. Release multi-arch PR automation is also not active; its [workflow](../../../../.github/workflows/therock-multi-arch-ci.yml) is manual-only, defaulting to Linux `gfx94X`, `gfx950` and Windows `gfx110X` when inputs are empty.

## Coverage enforcement

- The 80% coverage goal remains aspirational as an acceptance gate. Local [coverage targets](../../CMakeLists.txt) generate reports but do not enforce a threshold; the monorepo [Codecov configuration](../../../../codecov.yml) declares a hipDNN target, but checked-in workflows do not verify upload, required-status, or component-level enforcement. Required GitHub checks are external to this repository.

## Tested versus supported architectures

- CI build matrices, executed test matrices, and runtime support are not interchangeable. Some configured Linux families are build-only, superbuild runner names do not prove physical GPU execution, and TheRock runner assignment comes from external mutable configuration. Current workflow evidence is in [TheRock CI](../../../../.github/workflows/therock-ci.yml), its [Linux test workflow](../../../../.github/workflows/therock-ci-linux.yml), and the [superbuild workflow](../../../../.github/workflows/hipdnn-superbuild-ci.yml).
- No centralized architecture-support contract is checked into this repository. Capability varies by operation, engine, build option, dependency, and runtime heuristic; external provider policy may also exist. Use [hipDNN operation support](../OperationSupport.md) and provider-owned support/configuration documents for narrow claims. Passing CI on one device must not be generalized into a support guarantee.

## Provider workaround tracking

Provider-owned TOML and category files are canonical. This index intentionally does not copy tolerance values or test filters; linked issues provide tracking for known exceptions.

- **Shared integration harness:** Standard BF16 RMSNorm reference cases are excluded while [issue #10560](https://github.com/ROCm/rocm-libraries/issues/10560) is open.
- **MIOpen provider:** [MIOPEN_ENGINE.toml](../../../../dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml) records current tolerance exceptions and architecture-specific skips. Related tracking includes [#6979](https://github.com/ROCm/rocm-libraries/issues/6979), [#8029](https://github.com/ROCm/rocm-libraries/issues/8029), [#8030](https://github.com/ROCm/rocm-libraries/issues/8030), and [#6864](https://github.com/ROCm/rocm-libraries/issues/6864).
- **hipBLASLt provider:** [HIPBLASLT_ENGINE.toml](../../../../dnn-providers/hipblaslt-provider/config/HIPBLASLT_ENGINE.toml) records the BF16 fused-matmul tolerance exception and gfx12 FP16 MatmulBias skips tracked by [#8033](https://github.com/ROCm/rocm-libraries/issues/8033).
- **HIP-kernel provider:** [HIP_MLOPS_ENGINE.toml](../../../../dnn-providers/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml) and [ASM_SDPA_ENGINE.toml](../../../../dnn-providers/hip-kernel-provider/config/ASM_SDPA_ENGINE.toml) record current skips and tolerance exceptions. Disabled provider-mode rocKE Python tests are tracked by [#10497](https://github.com/ROCm/rocm-libraries/issues/10497).

## Failures, exclusions, and retries

- Enabled test failures fail their owning workflows; no checked hipDNN lane is explicitly advisory. Whether a workflow status is required for merge is controlled outside the repository and is not verified here.
- A green workflow means its configured tests passed after declared skips, tolerance overrides, architecture filters, and any allowed retries. It does not mean every registered or supported case ran. Provider exceptions are visible in the configs above. The [auto-retry workflow](../../../../.github/workflows/auto-retry-failed.yml) covers only TheRock CI and two named TheRock nightlies from default-branch configuration; superbuild and other lanes are outside it.

## Performance regression signal

- No checked automated hipDNN performance-regression signal runs representative workloads, compares a controlled baseline, attributes regressions, and gates changes. [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking#readme) provides manual workload execution and engine-name/ID/version-attributed timing summaries with an optional plugin path, but lacks complete run/workload provenance and currently excludes CI use. No checked-in policy defines baseline selection, environment pinning, noise/rerun handling, or triage ownership; external lab or team policy remains unverified.
