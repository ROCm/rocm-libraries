# hipDNN Testing Strategy

## Purpose

This document explains what each test layer should check, who owns it, and what current CI results do and do not prove.

Other documents own the detailed instructions:
- [Testing](./TESTING.md) tells contributors what to run before pushing.
- [Building](../Building.md) has build, coverage, sanitizer, and test commands.
- [Coding Style and Naming Guidelines](../CodingStyleAndNamingGuidelines.md#11-test-naming-guidelines) has test naming rules.
- The [provider integration test guide](../../../../dnn-providers/integration-tests/README.md) explains bundles, provider setup, and categories.
- Workflow files and [CODEOWNERS](../../../../.github/CODEOWNERS) define CI jobs and review routing.
- [Known Testing Gaps](./KNOWN_TESTING_GAPS.md) lists work that is missing or incomplete. A listed gap is not a current requirement.

**Current state** describes what the repository does today. **Strategy** describes the target design. Future work belongs in [Known Testing Gaps](./KNOWN_TESTING_GAPS.md) until it is implemented.

## Principles

1. **Test the smallest useful part.** Put a test at the lowest layer that can check the behavior without copying production logic.
2. **Classify tests by what they depend on.** Unit tests replace outside dependencies. Integration tests use real components, plugins, runtimes, or devices.
3. **Keep unit tests GPU-free.** Tests that need a GPU, runtime compilation, or kernel execution belong in integration or performance testing.
4. **Separate routing from numerical results.** hipDNN owns graph handling and plugin calls. Providers own whether an engine produces correct results on a device.
5. **A skip is not a pass.** A skipped test gives no result for that configuration.
6. **CI does not define product support.** A green job proves only that its selected tests passed in that environment.
7. **Make failures easy to trace.** Name the layer and engine, use repeatable inputs, and avoid duplicate tests that report the same failure.

## Unit Tests Should Not Need a GPU

### Target Design

**Strategy:** A unit test checks one production unit and replaces dependencies outside that unit with fakes, stubs, or mocks:

- backend unit tests replace production providers and GPU work;
- frontend unit tests replace the backend C API;
- SDK unit tests do not run providers or kernels;
- provider adapter unit tests replace library calls, runtime compilation, and GPU execution.

Unit tests should not need a GPU. Behavior that needs a real HIP runtime, production plugin, library handle, runtime compiler, or kernel belongs in integration testing.

### Who Tests What

| Layer | What it checks | What the test replaces |
|---|---|---|
| Backend | State, descriptors, lifetime, errors, plugin coordination | Production providers and GPU work |
| Frontend | Graph building and backend API calls | Backend implementation |
| Data and FlatBuffers SDKs | Data models, serialization, logging, utilities | Consumers, providers, and GPU work |
| Plugin SDK | Provider helpers and base behavior | Real provider engines |
| Test SDK | Reference results, comparisons, shared test tools | Production plugins and GPU work |
| Provider adapter | Graph translation, support checks, configuration, errors | Library or kernel execution and GPU |

Public backend and frontend tests are integration tests because they cross the public API boundary, even when they use controlled test plugins.

## Integration and Provider Tests

### Test Layers

| Layer | What it runs | What it proves | What it does not prove |
|---|---|---|---|
| Public backend API | Public C API with controlled plugins | API lifetime, descriptors, errors, and plugin management | Numerical results from a real provider |
| Frontend-to-backend | Frontend API through the backend with a controlled plugin | Graph conversion, routing, execution flow, and returned results | Kernel correctness on a GPU |
| Provider-local integration | One provider with its real libraries and GPU | Provider-specific support, errors, engine behavior, and repeatability | Behavior of another provider |
| Shared provider integration | Provider-agnostic `hipdnn_integration_tests` with one plugin, one engine, and reference results | End-to-end graph execution and numerical results for that engine, device, and problem | Full operation or GPU support |
| Library or kernel tests | MIOpen, hipBLASLt, rocKE, or another lower-level library | Behavior owned by that library | hipDNN routing unless the test runs through hipDNN |

Use the shared suite for graph execution and result checks. Use provider-local tests for provider-specific errors, options, engine selection, compilation, or behavior that a graph bundle cannot express. See the [provider integration test guide](../../../../dnn-providers/integration-tests/README.md) for setup and authoring rules.

The executable contains one shared graph set. Each provider runs it separately with its own plugin and engine; one run never loads or compares several providers.

Reference execution gives the expected result, but it does not prove that a provider supports the graph. The provider must accept the graph, choose an engine, and run it before results are compared. See the [CPU Graph Executor design](../rfcs/0001_CpuGraphExecutorDesign.md).

### Ownership

| Item | Owner |
|---|---|
| Public API, graph serialization, routing, plugin lifetime, and frontend/backend data | hipDNN core tests |
| Shared graphs, reference comparison, tolerances, and provider-agnostic test harness | Shared provider integration code |
| Provider graph translation, support checks, engine binding, skips, tolerances, and adapter errors | Provider tests and configuration |
| Kernel selection and results below the provider adapter | Lower-level library tests and provider integration tests |
| GPU-specific support checks | Provider or engine runtime checks |
| Review routing | [CODEOWNERS](../../../../.github/CODEOWNERS) |

CODEOWNERS chooses reviewers; it does not define product support or who owns performance regressions. Keep the provider and engine name in failures, then find the lowest layer that still fails.

### Product Support and CI Coverage Are Different

Support claims and CI results come from different sources:

1. **Support sources** are operation-support docs, build options, runtime checks, and explicit exclusions.
2. **CI results** show that one job ran selected tests on a specific OS, GPU, and configuration.

One cannot replace the other. A green job does not prove support for every engine, operation, shape, datatype, option, or GPU. A missing job does not prove lack of support. A family name such as `gfx94X` is not a specific tested GPU unless the runner identifies one.

Provider capability sources:

- [MIOpen provider operation support](../../../../dnn-providers/miopen-provider/docs/OperationSupport.md)
- [hipBLASLt provider operation support](../../../../dnn-providers/hipblaslt-provider/docs/OperationSupport.md)
- [HIP-kernel provider engine architecture and build flags](../../../../dnn-providers/hip-kernel-provider/README.md#architecture)

These sources are not combined into one support table because support can depend on build options, library behavior, execution direction, runtime checks, and exclusions.

**Current state:** Shared integration tests prefer data-driven bundles, but checked-in provider builds and workflows do not enable them. Bundles available in the repository must not be reported as tests that ran.

## Current CI Workflows

### Test Categories

The public test categories are `quick`, `standard`, `comprehensive`, and `full`. Provider category mechanics belong to the [provider integration test guide](../../../../dnn-providers/integration-tests/README.md).

In core hipDNN, `quick` matches every core test and the higher categories inherit the same set. This is intentional behavior, not a coverage gap or a request to differentiate the categories.

### Checked on 2026-08-17

This table shows repository settings, not a support list. An enabled test job may still skip GPU tests, and external runner settings can change the exact hardware.

| Workflow | When it runs | Requested systems | What runs |
|---|---|---|---|
| [TheRock CI](../../../../.github/workflows/therock-ci.yml) | Pull requests, selected pushes, manual runs | Linux `gfx94X`, `gfx950`, `gfx125X`; Windows `gfx1151` | Linux package tests are disabled for `gfx950` and `gfx125X`. Other test jobs still depend on external runner assignments. |
| [hipDNN Superbuild CI](../../../../.github/workflows/hipdnn-superbuild-ci.yml) | Matching pull requests | Linux `gfx94X-dcgpu`/`gfx942`; Windows `gfx1151` | Root CTest and frontend wheel tests run. A runner name does not prove every GPU test ran instead of skipping. |
| [Legacy TheRock Nightly](../../../../.github/workflows/therock-ci-nightly.yml) | Daily and manual | Linux `gfx94X`, `gfx950`; Windows `gfx1151` | Scheduled runs use `comprehensive`; Linux `gfx950` is build-only. Other tests depend on external runners. |
| [Release Multi-Arch Nightly](../../../../.github/workflows/therock-multi-arch-ci-nightly.yml) | Daily and manual | Linux `gfx94X`, `gfx950`; Windows `gfx1151` | External settings choose the exact test runners. |
| [Release Multi-Arch CI](../../../../.github/workflows/therock-multi-arch-ci.yml) | Manual only | Defaults to Linux `gfx94X`, `gfx950`; Windows `gfx110X` | It is not an automatic pull-request check. |

Update this table when workflow triggers, GPU families, runner assignments, path filters, or test categories change. Workflow files are always the source of truth.

### Test Failures and Merge Rules

**Current state:** An enabled hipDNN test job fails when one of its selected tests fails. A skipped or unselected job is not a failure, but it is also not proof that the tests passed.

GitHub settings outside this repository decide which jobs block a merge. Do not claim that every listed workflow blocks merging unless those settings were checked.

## Sanitizers and Coverage

### Who Owns Sanitizer Failures

Sanitizers check only the code they instrument. hipDNN owns sanitizer errors in core code. Providers own errors in their adapters. Errors in lower-level libraries must be reported to those library owners. A provider test without sanitizer instrumentation does not check memory safety.

See [Building: Address Sanitizer Build](../Building.md#address-sanitizer-build) for supported commands.

**Current state:**

- Standalone ASAN supports Linux and Windows, but repository settings do not prove that every platform and test combination is clean.
- Standalone TSAN supports Linux host code only. No verified hipDNN TSAN CI job exists.
- The [opt-in ASAN workflow](../../../../.github/workflows/therock-multi-arch-ci-asan.yml) builds sanitizer variants for labeled pull requests and manual runs. Pull requests build only; manual runs may test when a runner is available.
- The [ASAN nightly workflow](../../../../.github/workflows/therock-multi-arch-ci-asan-nightly.yml) runs device ASAN on Linux. On 2026-08-17, `gfx94X` had a test runner, `gfx950` was build-only, and `gfx125X` had no supported ASAN build. External settings can change.

ASAN does not test every pull request, provider, or GPU. See [Known Testing Gaps](./KNOWN_TESTING_GAPS.md) for missing coverage.

### Coverage

LLVM coverage reports are run manually; see [Building](../Building.md). The 80% coverage number is a goal, not a CI requirement.

Line coverage shows which code ran. It does not prove correct behavior, provider support, GPU coverage, or numerical accuracy.

## Performance and Benchmarking

### Goal

Performance tests measure graph execution and host submission time. They are separate from correctness tests, and every result must include its test environment.

Use the public [ROCm dnn-benchmarking project](https://github.com/ROCm/dnn-benchmarking#readme) for commands, result formats, and workloads. This document explains how hipDNN teams should read those results.

### Workloads and Source Details

Workloads are stored as DVC pointer files under `Workloads/**/*.tar.gz.dvc`. Record the pointer revision and selected graph inputs because benchmark output does not save them automatically.

Use fixed, reviewable workloads. Record any change to graph structure, shapes, datatypes, inputs, or engine selection.

### What Each Time Measures

Keep these times separate:

- **GPU graph time:** device time for all GPU work started by the graph. A graph may run several kernels.
- **Host submission time:** host time used to enqueue the graph. It does not include later GPU completion.
- **Total elapsed time:** may include setup, allocation, correctness checks, synchronization, or cleanup. State exactly what it includes.

Finish warmups before timing. Record warmup count, timed run count, synchronization method, timing method, and summary statistic. Current output does not save all of these fields.

### Comparing Results

- Compare with a baseline from the same GPU architecture. Prefer the same GPU model, OS, runtime, power settings, library versions, provider build, engine, workload, and benchmark options.
- Record provider, engine name and ID, plugin version or artifact, and feature flags. Some of this information must still be saved outside the benchmark output.
- Keep host and GPU statistics separate. Save raw samples separately when distribution analysis is needed.
- Re-run suspicious results in a controlled environment before calling them regressions. Do not compare performance when correctness tests fail.
- Set pass/fail limits in the owning experiment or release policy after measuring normal variation. This document defines no universal limit.

### Who Owns Performance

| Layer | Responsibility |
|---|---|
| Workload and benchmark tool | Inputs, timing code, result format, statistics, and repeatability |
| hipDNN core | Graph building, serialization, provider dispatch, API overhead, and engine identity |
| Provider adapter | Support checks, graph translation, engine selection, plugin overhead, and provider options |
| Lower-level library or kernel | Kernel choice, compilation, workspace use, and GPU execution |
| CI or lab | Hardware, system software, power settings, baselines, and comparable runs |

CODEOWNERS chooses reviewers but does not assign performance triage. First reproduce the result, decide whether the change is in host or GPU time, and then find the lowest layer that still shows it.

**Current state:** dnn-benchmarking reports engine name, ID, version, and an optional plugin path. It does not save all workload, build, timing, or feature details. No checked-in hipDNN CI job compares GPU performance with a baseline.
