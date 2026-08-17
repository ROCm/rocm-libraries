# hipDNN Testing

This guide answers the immediate contributor question: “I just made a change. What do I do before I push?”
For project context, start with the [README](../README.md), [design overview](./Design.md), and [contribution guide](../CONTRIBUTING.md).
Detailed test models and coverage responsibilities live in [Testing Strategy](./TESTING_STRATEGY.md); tracked limitations live in [Known Gaps](./KNOWN_GAPS.md).

## Component Overview

hipDNN is a graph-based DNN execution library with Windows and Linux support. The frontend, backend, and plugin separation allows each layer to be unit tested independently, while integration tests validate cross-layer behavior.

hipDNN core includes the backend, frontend, Data SDK, FlatBuffers SDK, Plugin SDK, and Test SDK.
Core unit and API suites use in-tree test plugins, so core-only changes can be validated without building a production provider.

Provider plugins are separate projects under `dnn-providers/`.
Changes that affect plugin loading, graph execution, operation support, or provider-facing contracts need validation with the affected provider and its integration tests. Prefer the superbuild; an installed standalone-provider workflow is also supported.
See [Plugin Development](./PluginDevelopment.md) for the plugin boundary and the [provider integration README](../../../dnn-providers/integration-tests/README.md) for integration mechanics and tiers.

## Development Workflow

Use the path matching your change. For local pre-push validation, run the applicable component's `standard` label, then widen coverage when risk requires it. Core `standard` currently selects the same tests as the other core tiers; provider label contents differ.
Build setup, supported presets, target names, and platform prerequisites are owned by [Building hipDNN](./Building.md).

### Core-only standalone path

Use this path when changes stay within hipDNN core and do not alter provider or cross-project behavior.
From `projects/hipdnn/`:

```bash
cmake --preset release
cmake --build build/release
ctest --test-dir build/release -L standard --output-on-failure
```

During iteration, use `ctest -R <regex>` or a direct GoogleTest filter after building to narrow the run.
Before pushing, return to the complete core `standard` tier.

### Provider or integration superbuild path

Use this path for provider code, provider-visible contracts, real graph execution, or cross-project integration changes.
From the repository root, choose the narrowest provider preset that covers the change. Use `hipdnn-providers` for MIOpen plus hipBLASLt, or `hipdnn-providers-all` when HIP-kernel is also affected:

```bash
cmake --preset hipdnn-providers -DROCM_LIBS_ENABLE_ROOT_CTEST=ON
cmake --build build
ctest --test-dir build -L standard --output-on-failure
```

Root CTest registration requires `ROCM_LIBS_ENABLE_ROOT_CTEST=ON` at configure time.
Provider-specific presets and other superbuild choices are listed in [Building hipDNN](./Building.md#superbuild).

`ctest` only runs binaries already present in the build tree; it does not compile changed sources or refresh stale test binaries.
Always build first after a source change. Generated hipDNN/provider category-check targets and provider external-integration check targets are also safe because they build their declared test dependencies before running their fixed scope.

Follow [Coding Style and Naming Guidelines](./CodingStyleAndNamingGuidelines.md) and complete the non-test contribution checks in [CONTRIBUTING](../CONTRIBUTING.md) before pushing.

## Unit Testing Strategy

Unit tests should isolate one component and exercise observable behavior, boundaries, and failure paths.
Keep them deterministic and fast; replace external dependencies with existing fakes or mocks.
Add a regression test for a defect and prove it fails without the fix.

Place tests beside the component’s existing suite rather than creating a second test structure.
Use GoogleTest patterns already present in that suite.
GPU-dependent tests must remain runnable on machines without a device by using the project’s established skip mechanism.

Do not duplicate naming or parameterization rules here.
Use [Coding Style and Naming Guidelines](./CodingStyleAndNamingGuidelines.md) for naming and [Testing Strategy](./TESTING_STRATEGY.md) for detailed layer selection and coverage guidance.

## Integration Testing Strategy

Core black-box tests validate public backend and frontend behavior with controlled in-tree plugins.
Use them when a change crosses component boundaries but does not require numerical validation by a production provider.

Real-provider integration tests execute graphs through provider plugins and compare results with a reference.
Use them for operation correctness, provider selection, engine behavior, and device-dependent paths.
The shared provider integration suite is the normal home for “this graph runs and produces the right result”; provider-local tests cover behavior specific to one provider.

hipDNN owns routing, API, serialization, and plugin-contract quality.
Each provider owns its operation correctness and supported-architecture coverage; a core-only pass is not evidence that provider behavior is correct.
Detailed responsibility and environment matrices belong in [Testing Strategy](./TESTING_STRATEGY.md), while bundle authoring and tier mechanics belong in the [provider integration README](../../../dnn-providers/integration-tests/README.md).

## Pre-submit / CI gates

Use the applicable component's `standard` label as the local pre-push convention, noting that core tiers currently collapse to one set and superbuild CI runs unfiltered CTest. Local checks do not replace CI.
Choose scope from the two paths above, record the exact commands and outcomes in the pull request, and never mark skipped or unrun checks as passed.

Checked-in workflows are the source of truth for requested operating systems, runner labels, build options, static-analysis settings, and test commands. Actual runner assignment, external TheRock configuration, and required GitHub checks must be confirmed from the current workflow run and repository settings.
Executable workflow details live in [`hipdnn-superbuild-ci.yml`](../../../.github/workflows/hipdnn-superbuild-ci.yml), and review routing lives in [`CODEOWNERS`](../../../.github/CODEOWNERS).
Follow the repository contribution guidance in [CONTRIBUTING](../CONTRIBUTING.md).

If a required check fails, fix the cause or document a maintainer-approved exception.
Do not weaken, disable, or skip a test merely to make a gate pass.
Known differences between intended policy and current enforcement are indexed in [Known Gaps](./KNOWN_GAPS.md).

## ASAN/TSAN/sanitizer coverage

Sanitizer runs are risk-based additions to the normal `standard` validation.
Run ASAN for changes involving ownership, allocation, descriptor lifetime, serialization, plugin loading, or failure cleanup.
Run TSAN for changes involving threads, shared state, callbacks, logging concurrency, or synchronization.

ASAN and TSAN are separate, mutually exclusive build configurations.
TSAN is supported only on Linux and checks host-side races; ASAN support depends on platform and GPU architecture, so some GPU tests are skipped in ASAN builds.
For ASAN, use [Building hipDNN: Address Sanitizer Build](./Building.md#address-sanitizer-build). For TSAN, use [Building hipDNN: Thread Sanitizer Build](./Building.md#thread-sanitizer-build). Run the applicable `standard` label from the sanitizer build.

A clean sanitizer run supplements functional assertions; it does not prove numerical correctness or replace provider integration coverage.
See [Known Gaps](./KNOWN_GAPS.md) for current sanitizer automation and coverage limitations.
