# hipDNN Testing

- **Status:** Draft
- **Owner:** @ROCm/hipdnn-core
- **Technical Lead:** Brian Harrison
- **Last Updated:** 2026-08-17

This guide tells contributors which tests to run before pushing a change.
For general project information, see the [README](../../README.md), [design overview](../Design.md), and [contribution guide](../../CONTRIBUTING.md).
For detailed test design and ownership, see [Testing Strategy](./TESTING_STRATEGY.md). For current limits, see [Known Testing Gaps](./KNOWN_TESTING_GAPS.md).

## Component Overview

hipDNN is a graph-based DNN execution library with Windows and Linux support. The frontend, backend, and plugin separation allows each layer to be unit tested independently, while integration tests validate cross-layer behavior.

hipDNN core includes the backend, frontend, Data SDK, FlatBuffers SDK, Plugin SDK, and Test SDK.
Core unit and API tests use test plugins from this repository. You can test core-only changes without building a production provider.

Production provider plugins live under `dnn-providers/`.
Changes to plugin loading, graph execution, operation support, or provider-facing behavior must be tested with the affected provider. Prefer the superbuild; you can also test an installed standalone provider.
See [Plugin Development](../PluginDevelopment.md) for the plugin boundary and the [provider integration README](../../../../dnn-providers/integration-tests/README.md) for provider test setup and categories.

## Development Workflow

Choose the path that matches your change. Before pushing, run the affected component's `standard` tests.
See [Building hipDNN](../Building.md) for setup, presets, targets, and platform requirements.

### Core-only standalone path

Use this path when changes stay within hipDNN core and do not alter provider or cross-project behavior.
From `projects/hipdnn/`:

```bash
cmake --preset release
cmake --build build/release
ctest --test-dir build/release -L standard --output-on-failure
```

While working, use `ctest -R <regex>` or a GoogleTest filter to run only the tests you need.
Before pushing, run the full core `standard` category.

### Provider or integration superbuild path

Use this path for provider changes, provider-facing core changes, real graph execution, or changes that span projects.
From the repository root, use the smallest preset that includes the affected provider. Use `hipdnn-providers` for MIOpen and hipBLASLt, or `hipdnn-providers-all` when HIP-kernel is also affected:

```bash
cmake --preset hipdnn-providers -DROCM_LIBS_ENABLE_ROOT_CTEST=ON
cmake --build build
ctest --test-dir build -L standard --output-on-failure
```

Root-level CTest only sees these tests when CMake was configured with `ROCM_LIBS_ENABLE_ROOT_CTEST=ON`.
See [Building hipDNN](../Building.md#superbuild) for other provider presets.

`ctest` does not build changed code. Always build before running tests.

Follow [Coding Style and Naming Guidelines](../CodingStyleAndNamingGuidelines.md) and complete the non-test contribution checks in [CONTRIBUTING](../../CONTRIBUTING.md) before pushing.

## Unit Testing Strategy

Unit tests should test one component and its visible behavior, edge cases, and failures.
Keep them fast and repeatable. Replace outside dependencies with existing fakes or mocks.
For a bug fix, add a test that fails before the fix and passes after it.

Put new tests beside the existing tests for that component.
Follow the GoogleTest patterns already used there.
If a unit test needs a GPU, call `SKIP_IF_NO_DEVICES()` so it skips cleanly on CPU-only machines.

Use [Coding Style and Naming Guidelines](../CodingStyleAndNamingGuidelines.md) for test names. Use [Testing Strategy](./TESTING_STRATEGY.md) to choose the right test layer.

## Integration Testing Strategy

Core integration tests check the public backend and frontend with controlled test plugins.
Use them when a change crosses component boundaries but does not need a production provider.

Provider integration tests run graphs through a real provider and compare results with a reference.
Use them to test operation results, provider and engine selection, and GPU behavior.
Put shared graph tests in the provider-agnostic suite. Put behavior that belongs to one provider in that provider's own tests.
Each provider runs the shared `hipdnn_integration_tests` graph set separately with its own plugin and engine.

hipDNN owns API, graph, routing, serialization, and plugin-lifecycle tests.
Each provider owns its operation results and supported GPU coverage. Passing core tests does not prove that a provider works correctly.
See [Testing Strategy](./TESTING_STRATEGY.md) for ownership details and the [provider integration README](../../../../dnn-providers/integration-tests/README.md) for provider test setup.

## Performance & Benchmarking

hipDNN sends graphs to provider engines; providers run most kernels. Every performance result must name the provider and engine so a slowdown can be traced to hipDNN, the provider adapter, or the underlying library.

Use [ROCm dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking#readme) for manual graph benchmarks. Record the workload revision, graph inputs, GPU, provider, engine, plugin version or artifact, and software build. Compare only with a baseline from the same GPU architecture.

Keep GPU execution time separate from host submission time. See [Testing Strategy](./TESTING_STRATEGY.md#performance-and-benchmarking) for details. There is no automated performance gate today; see [Known Testing Gaps](./KNOWN_TESTING_GAPS.md#automated-performance-checks).

## Pre-submit / CI gates

Run the affected component's `standard` tests before pushing. Local tests do not replace CI.
Record the commands you ran and their results in the pull request. Do not mark skipped or unrun checks as passed.

Workflow files are the source of truth for CI platforms, runners, build options, and test commands. Runner assignments and required GitHub checks may be configured outside this repository, so confirm them in the current workflow run and repository settings.
See [`hipdnn-superbuild-ci.yml`](../../../../.github/workflows/hipdnn-superbuild-ci.yml) for the superbuild workflow and [`CODEOWNERS`](../../../../.github/CODEOWNERS) for review routing.
Complete the non-test checks in [CONTRIBUTING](../../CONTRIBUTING.md).

If a required check fails, fix it or document an exception approved by a maintainer.
Do not disable or skip a test only to make a check pass.
See [Known Testing Gaps](./KNOWN_TESTING_GAPS.md) for limits in current automation and policy.

## Static Analysis

hipDNN uses `clang-tidy` to find common C++ and HIP problems. Linux builds enable it by default; Windows builds disable it by default because it adds significant build time.

The Linux hipDNN superbuild CI enables `clang-tidy`; the Windows job disables it. For local checks, build with `-DENABLE_CLANG_TIDY=ON` or run the `hipdnn-tidy` target when it is available. See [Building hipDNN: Clang Tools](../Building.md#clang-tools) and [Clang-Tidy Rules](../CodingStyleAndNamingGuidelines.md#151-clang-tidy-rules).

Static analysis does not replace unit or integration tests.

## ASAN/TSAN/sanitizer coverage

Sanitizers are extra checks for risky changes; they do not replace normal `standard` tests.
Run ASAN for changes involving memory ownership, allocation, object lifetime, serialization, plugin loading, or cleanup after failures.
Run TSAN for changes involving threads, shared state, callbacks, logging, or synchronization.

ASAN and TSAN use separate build configurations and cannot be enabled together.
TSAN supports Linux host code only. ASAN support depends on the platform and GPU, so some GPU tests skip under ASAN.
For ASAN commands, see [Building hipDNN: Address Sanitizer Build](../Building.md#address-sanitizer-build). For TSAN, configure a standalone Linux build with `-DBUILD_THREAD_SANITIZER=ON`. Build first, then run the affected `standard` tests.

A clean sanitizer run checks memory or thread safety only. It does not prove numerical correctness or replace provider integration tests.
See [Known Testing Gaps](./KNOWN_TESTING_GAPS.md) for current sanitizer limits.
