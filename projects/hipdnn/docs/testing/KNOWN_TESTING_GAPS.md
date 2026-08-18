# Known Testing Gaps

This document records current limitations in our testing.

## Pre-submit scope and integration tiers

- Core tier names do not currently represent increasing scope: [`quick` matches every core test](../../test_categories.yaml), so `standard`, `comprehensive`, and `full` select the same core set through cascading labels. TheRock PR selection normally requests `standard`, while the separate [superbuild PR workflow](../../../../.github/workflows/hipdnn-superbuild-ci.yml) runs unfiltered CTest. There is therefore no single consistent “pre-submit subset.”
- Bundle tests are the preferred authoring format but remain runtime opt-in; checked-in provider CMake and workflows do not enable them. The integration project's internal GPU-reference/unit `comprehensive` and `full` category blocks are disabled. MIOpen and hipBLASLt external suites still define those tiers, while HIP-kernel external suites are temporarily standard-only. MIOpen's `quick` and `standard` bundle patterns remain commented out. Track current mechanics in the [provider integration guide](../../../../dnn-providers/integration-tests/README.md), [shared categories](../../../../dnn-providers/integration-tests/test_categories.yaml), and provider category files.

## Unit-test classification debt

- Checked-in core and provider `unit` binaries do not yet fully match the GPU-free isolation model in [Testing Strategy](TESTING_STRATEGY.md#unit-testing-gpu-free-dependency-isolation). Backend, Data SDK, Test SDK, and provider binaries include controlled dynamic loading, HIP memory/device paths, real dependency handles, runtime compilation, or embedded kernels. No-device runs skip affected cases; those skips are missing observations, not GPU-free unit coverage.

## Sanitizers and platforms

- Sanitizer builds and test execution are asymmetric. The [ASAN workflow](../../../../.github/workflows/therock-multi-arch-ci-asan.yml) provides opt-in PR and manual builds: PR-triggered sanitizer runs are build-only, while manual dispatch may test where sandbox mapping exists. The [device-ASAN nightly](../../../../.github/workflows/therock-multi-arch-ci-asan-nightly.yml) uses the same external mapping. As verified on 2026-08-17, `gfx94X` has a sandbox test assignment, `gfx950` is build-only, and `gfx125X` lacks a supported ASAN build variant; this external mapping can change.
- Standalone ASAN has architecture and platform constraints, and Windows host instrumentation is not equivalent to Linux device-ASAN coverage. Some standalone sample and provider convolution cases remain excluded because of upstream failures. [Sanitizer configuration](../../cmake/Sanitizers.cmake), [build guidance](../Building.md#address-sanitizer-build), and test registration remain the sources of truth.
- TSAN is a Linux build capability, but no hipDNN TSAN CI lane or verified CI test execution is checked in. Release multi-arch PR automation is also not active; its [workflow](../../../../.github/workflows/therock-multi-arch-ci.yml) is manual-only, defaulting to Linux `gfx94X`, `gfx950` and Windows `gfx110X` when inputs are empty.

## Coverage enforcement

- The 80% coverage goal remains aspirational as an acceptance gate. Local [coverage targets](../../CMakeLists.txt) generate reports but do not enforce a threshold; the monorepo [Codecov configuration](../../../../codecov.yml) declares a hipDNN target, but checked-in workflows do not verify upload, required-status, or component-level enforcement. Required GitHub checks are external to this repository.

## Tested versus supported architectures

- CI build matrices, executed test matrices, and runtime support are not interchangeable. Some configured Linux families are build-only, superbuild runner names do not prove physical GPU execution, and TheRock runner assignment comes from external mutable configuration. Current workflow evidence is in [TheRock CI](../../../../.github/workflows/therock-ci.yml), its [Linux test workflow](../../../../.github/workflows/therock-ci-linux.yml), and the [superbuild workflow](../../../../.github/workflows/hipdnn-superbuild-ci.yml).
- No centralized architecture-support contract is checked into this repository. Capability varies by operation, engine, build option, dependency, and runtime heuristic; external provider policy may also exist. Use [hipDNN operation support](../OperationSupport.md) and provider-owned support/configuration documents for narrow claims. Passing CI on one device must not be generalized into a support guarantee.

## Provider workaround debt

Provider-owned TOML and category files are canonical. This index intentionally does not copy tolerance values or test filters.

- **Shared integration harness:** The integration project's internal GPU-reference/unit `comprehensive` and `full` category blocks are disabled, and standard BF16 RMSNorm reference cases are excluded while [issue #10560](https://github.com/ROCm/rocm-libraries/issues/10560) is open. Provider external categories differ: MIOpen and hipBLASLt define comprehensive/full, while HIP-kernel currently exposes a temporary standard-only mapping. See [shared categories](../../../../dnn-providers/integration-tests/test_categories.yaml).
- **MIOpen provider:** [MIOPEN_ENGINE.toml](../../../../dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml) contains current batch-normalization and convolution tolerance exceptions plus architecture-specific Conv-Bias-Activation skips. The CBA skip cites open [#6979](https://github.com/ROCm/rocm-libraries/issues/6979); WRW rationale cites closed [#8029](https://github.com/ROCm/rocm-libraries/issues/8029) and [#8030](https://github.com/ROCm/rocm-libraries/issues/8030), whose mitigations remain encoded. [Integration categories](../../../../dnn-providers/miopen-provider/test_categories_integration.yaml) exclude Windows gfx110X and use elevated timeouts; linked open [#6864](https://github.com/ROCm/rocm-libraries/issues/6864) documents broader/Linux runner variance rather than that Windows-specific setting.
- **hipBLASLt provider:** [HIPBLASLT_ENGINE.toml](../../../../dnn-providers/hipblaslt-provider/config/HIPBLASLT_ENGINE.toml) contains a BF16 fused-matmul tolerance exception and skips matching gfx12 FP16 MatmulBias fixtures for open [#8033](https://github.com/ROCm/rocm-libraries/issues/8033)'s transA=T/transB=T bias-epilogue no-algorithm case. Within the wired `*Matmul*` selection, unsupported graphs default to GTest skip; manual `--fail-on-unsupported` converts that condition to failure.
- **HIP-kernel provider:** [HIP_MLOPS_ENGINE.toml](../../../../dnn-providers/hip-kernel-provider/config/HIP_MLOPS_ENGINE.toml) carries global test-workaround skips for pure-BF16 RMSNorm backward fixtures and large layer-normalization cases; these are not declared unsupported operations. [ASM_SDPA_ENGINE.toml](../../../../dnn-providers/hip-kernel-provider/config/ASM_SDPA_ENGINE.toml) applies backward-specific tolerance exceptions because the common resolver uses forward-tuned values. External integration currently exposes a temporary standard-only mapping of Smoke cases. Three provider-mode installed rocKE Python CTest entries remain disabled while open [#10497](https://github.com/ROCm/rocm-libraries/issues/10497) tracks re-enablement; C++ rocKE smoke tests remain enabled.

## Failures, exclusions, and retries

- Enabled test failures fail their owning workflows; no checked hipDNN lane is explicitly advisory. Whether a workflow status is required for merge is controlled outside the repository and is not verified here.
- A green workflow means its configured tests passed after declared skips, tolerance overrides, architecture filters, and any allowed retries. It does not mean every registered or supported case ran. Provider exceptions are visible in the configs above. The [auto-retry workflow](../../../../.github/workflows/auto-retry-failed.yml) covers only TheRock CI and two named TheRock nightlies from default-branch configuration; superbuild and other lanes are outside it.

## Performance regression signal

- No checked automated hipDNN performance-regression signal runs representative workloads, compares a controlled baseline, attributes regressions, and gates changes. [dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking#readme) provides manual workload execution and engine-name/ID/version-attributed timing summaries with an optional plugin path, but lacks complete run/workload provenance and currently excludes CI use. No checked-in policy defines baseline selection, environment pinning, noise/rerun handling, or triage ownership; external lab or team policy remains unverified.
