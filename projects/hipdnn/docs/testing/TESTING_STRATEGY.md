# hipDNN Testing Strategy

## Purpose and Scope

This document defines the validation model and responsibility boundaries for hipDNN maintainers. It explains what each test layer is expected to prove, how provider and GPU coverage are interpreted, and how current automation relates to product capability.

It does not own build commands, test naming rules, provider integration mechanics, or GitHub merge policy:
- [Testing](./TESTING.md) is the concise contributor action guide.
- [Building](../Building.md) owns build, coverage, sanitizer, and test-target commands.
- [Coding Style and Naming Guidelines](../CodingStyleAndNamingGuidelines.md#11-test-naming-guidelines) owns test naming.
- The [provider integration test guide](../../../../dnn-providers/integration-tests/README.md) owns bundle authoring, provider configuration, and tier mechanics.
- Checked-in workflow files and [CODEOWNERS](../../../../.github/CODEOWNERS) are the executable sources for CI and review routing.
- [Known Testing Gaps](./KNOWN_TESTING_GAPS.md) records missing automation, incomplete rollout, and unresolved policy. A gap is not a current requirement or a claim of existing enforcement.

Statements labeled **Current state** describe repository behavior verified on the last-updated date. **Strategy** statements define the intended responsibility model. Future policy belongs in [Known Testing Gaps](./KNOWN_TESTING_GAPS.md) until implementation and enforcement exist.

## Principles

1. **Test the smallest meaningful boundary.** A test belongs at the lowest layer that can observe the contract without reproducing production internals.
2. **Classify by dependency boundary, not directory alone.** Unit tests isolate outward dependencies; integration tests exercise real boundaries between components, plugins, runtimes, or devices.
3. **Keep unit validation GPU-free.** Device availability, runtime compilation, kernel execution, and numerical comparison belong in integration or performance layers.
4. **Separate routing correctness from numerical correctness.** hipDNN core owns graph/API transport and plugin interaction. Providers and their underlying libraries own whether an engine can execute a problem correctly on a device.
5. **Treat skips as missing observations.** A skipped test can be appropriate for an inapplicable configuration, but it is not passing evidence for that configuration.
6. **Do not infer product support from CI.** A CI lane is a dated observation on one configured environment. Support must come from an explicit capability contract and runtime applicability checks.
7. **Keep failures actionable.** Tests should identify the layer and engine under test, use deterministic inputs where practical, and avoid non-blocking duplicates of the same signal.

## Unit Testing: GPU-Free Dependency Isolation

### Target Model

**Strategy:** A unit test owns one production unit and replaces dependencies outside that unit with fakes, stubs, or mocks. The target dependency boundaries are:

- backend unit tests replace production providers and device work while validating descriptors, handles, error propagation, graph extensions, and internal utilities;
- frontend unit tests isolate the backend C API while validating graph construction, nodes, attributes, and frontend control flow;
- Data SDK, FlatBuffers SDK, Plugin SDK, and Test SDK unit tests keep provider and kernel execution outside their boundary;
- provider adapter unit tests replace dependency-library calls, runtime compilation, and device execution while validating translation, applicability, configuration, and error paths.

In this target model, GPU hardware is not required to complete a unit suite. Logic that only becomes meaningful with a real HIP runtime, loaded production plugin, dependency handle, runtime compiler, or dispatched kernel belongs in integration testing.

### Current Classification Debt

**Current state:** Checked-in `unit` binaries do not yet fully conform to the target model:

- backend unit tests load controlled shared test plugins and include HIP device paths;
- Data SDK and Test SDK unit binaries include device-memory cases guarded by `SKIP_IF_NO_DEVICES()`;
- provider unit binaries mix isolated adapter tests with real MIOpen or hipBLASLt handles, runtime compilation, embedded kernels, and device-gated cases.

Those tests can complete on no-device runners by skipping affected cases, but each skip is a missing observation. Their current classification does not make device-dependent behavior part of the target unit-test model. This migration debt is tracked in [Known Testing Gaps](./KNOWN_TESTING_GAPS.md).

### Target Unit-Test Ownership

| Layer | Primary contract | Outward dependencies replaced |
|---|---|---|
| Backend | Internal state, descriptors, lifecycle, error handling, plugin coordination | Production providers and device work |
| Frontend | Graph construction and backend API translation | Backend implementation |
| Data and FlatBuffers SDKs | Data-model, serialization, logging, utility behavior | Consumers, providers, and device work |
| Plugin SDK | Provider interface helpers and base behavior | Concrete provider engines |
| Test SDK | Host references, comparisons, and shared test utilities | Production plugins and device work |
| Provider adapter | Graph translation, engine applicability, configuration, failure mapping | Underlying library/kernel execution and GPU |

Public backend and frontend contracts are not unit contracts merely because a controlled provider is used. The main public backend and frontend targets are black-box integration tests because they cross the published API boundary.

## Integration and Provider Validation

### Validation Layers

| Layer | Boundary exercised | What it proves | What it does not prove |
|---|---|---|---|
| Public backend API | Consumer through exported backend C API, usually with controlled plugins | API lifecycle, descriptor behavior, error mapping, and plugin-management contract | Numerical correctness of a real provider |
| Frontend-to-backend | Frontend C++ API through backend with controlled provider behavior | Graph lowering, descriptor creation, execution flow, and result transport | Kernel correctness on an ASIC |
| Provider-local integration | Provider plugin plus real dependencies/device | Adapter-specific applicability, unsupported paths, engine behavior, determinism, and cases not reducible to graph-output comparison | Cross-provider consistency by itself |
| Shared provider integration | `hipdnn_integration_tests`, a selected plugin/engine, and reference execution | End-to-end graph execution and numerical comparison for the observed engine/device/problem | Complete operation or architecture support |
| Dependency-library tests | MIOpen, hipBLASLt, rocKE, or other kernel-library boundary | Library/kernel behavior owned below the provider adapter | hipDNN graph routing unless exercised through hipDNN |

The shared suite is the default home for “execute this graph on this engine and compare its outputs” coverage. Provider-local C++ tests remain appropriate for adapter-specific failure paths, feature switches, determinism, compilation, engine selection, and other behavior that cannot be represented as a graph-output bundle. The [provider integration test guide](../../../../dnn-providers/integration-tests/README.md) owns that authoring decision and all bundle/tier mechanics.

Reference execution provides an expected numerical result; it does not establish provider applicability. A provider must first declare the graph applicable, select an engine, and execute it. Tests must preserve that sequence so a reference implementation cannot mask selection or translation failures. See the [CPU Graph Executor design](../rfcs/0001_CpuGraphExecutorDesign.md) for the host reference architecture.

### Responsibility Boundaries

| Artifact or behavior | Validation responsibility |
|---|---|
| Public API, graph serialization, routing, plugin lifecycle, and frontend/backend data transport | hipDNN core tests |
| Shared graph cases, reference comparison, tolerance plumbing, and cross-provider harness | Shared provider integration infrastructure |
| Provider graph translation, applicability, engine binding, provider-specific skips/tolerances, and adapter failures | Provider adapter tests and configuration |
| Kernel selection and numerical behavior below the adapter | Underlying dependency and kernel-library tests, plus end-to-end provider observations |
| Architecture-specific applicability | Provider/engine capability contract and runtime checks |
| Review routing for changes | Checked-in [CODEOWNERS](../../../../.github/CODEOWNERS) |

CODEOWNERS establishes review routing, not product support, on-call duty, or a performance-regression service-level agreement. A failure crossing layers should retain provider and engine identity, then be reduced to the lowest failing boundary before ownership is assigned.

### Declared Capability Is Not Observed CI Coverage

Provider and engine evidence comes from two independent classes of sources:

1. **Declared capability sources** include operation-support documentation, build-feature state, runtime architecture/problem checks, and explicit exclusions. No single synchronized capability contract currently combines all of them.
2. **Observed coverage** records that a configured build or test job ran on a dated lane. It includes workflow, OS, architecture family or device, enabled features, selected engine, test filter/tier, skips, and result.

Neither record substitutes for the other. A green lane does not prove every engine, operation, shape, datatype, feature flag, or device variant is supported. An absent lane does not prove lack of support. Family aliases such as `gfx94X` also must not be presented as a specific tested ASIC unless the runner configuration identifies one.

Provider capability sources:

- [MIOpen provider operation support](../../../../dnn-providers/miopen-provider/docs/OperationSupport.md)
- [hipBLASLt provider operation support](../../../../dnn-providers/hipblaslt-provider/docs/OperationSupport.md)
- [HIP-kernel provider engine architecture and build flags](../../../../dnn-providers/hip-kernel-provider/README.md#architecture)

The strategy deliberately does not synthesize those fragmented sources into one support matrix. Provider build flags, dependency heuristics, direction-specific engine checks, and runtime exclusions require source-specific qualification and versioning.

**Current state:** Data-driven bundles are the preferred authoring format in the shared integration project. Checked-in provider CMake and workflows do not pass `--allow-bundles` or set `HIPDNN_TEST_ALLOW_BUNDLES`; an external environment could still opt in. Existing wired C++ integration tests remain the checked-in external-check signal. Bundle presence must not be reported as executed provider coverage.

## CI Model and Dated Workflow Snapshot

### Interpretation

CTest semantic labels (`unit` and `integration`) describe test boundaries. Tier labels describe selection breadth. Core hipDNN and provider integration projects maintain separate category definitions; provider tier behavior belongs to the [provider integration test guide](../../../../dnn-providers/integration-tests/README.md).

**Current state:** In core hipDNN, the wildcard membership starts in `quick` and cascades through all higher labels. Therefore `quick`, `standard`, `comprehensive`, and `full` currently select the same core hipDNN tests. No broader core coverage may be claimed for `standard` or higher labels until [the category definition](../../test_categories.yaml) changes.

### Snapshot Verified 2026-08-17

This table records checked-in workflow configuration, not a support matrix. “Test job enabled” means automation schedules the test stage; device-dependent cases can still skip, and mutable external runner configuration can affect the exact hardware.

| Workflow | Trigger and selection | Configured build families/device | Observed test stage |
|---|---|---|---|
| [TheRock CI](../../../../.github/workflows/therock-ci.yml) | Active on PRs, selected pushes, and manual dispatch; PR default is `standard`, with path selection and labels able to skip or override it | Wrapper requests Linux `gfx94X`, `gfx950`, `gfx125X`; Windows `gfx1151` | Checked Linux logic suppresses package tests for `gfx950` and `gfx125X`. `gfx94X` and Windows `gfx1151` test execution still depends on mutable runner assignment. |
| [hipDNN Superbuild CI](../../../../.github/workflows/hipdnn-superbuild-ci.yml) | Path-filtered PR workflow; unfiltered root CTest | Linux target family `gfx94X-dcgpu` with configured device `gfx942`; Windows `gfx1151` | Root CTest and frontend-wheel tests run on generic scale-runner labels. Device selection and runner naming do not prove that every GPU case executes rather than skips. |
| [Legacy TheRock Nightly](../../../../.github/workflows/therock-ci-nightly.yml) | Scheduled daily and manually dispatchable; scheduled runs select `comprehensive` | Linux `gfx94X`, `gfx950`; Windows `gfx1151` | Checked Linux logic makes `gfx950` build-only. `gfx94X` and Windows `gfx1151` test jobs depend on mutable runner assignment. |
| [Release Multi-Arch Nightly](../../../../.github/workflows/therock-multi-arch-ci-nightly.yml) | Scheduled daily and manually dispatchable; delegates setup and runner selection to TheRock | Linux `gfx94X`, `gfx950`; Windows `gfx1151` | Exact functional runner assignment is externally configured and must be checked in the workflow result. |
| [Release Multi-Arch CI](../../../../.github/workflows/therock-multi-arch-ci.yml) | Manual dispatch only; its PR trigger is commented out | Empty inputs fall back to Linux `gfx94X`, `gfx950` and Windows `gfx110X` | It is not a current automatic presubmit lane. |

The workflow snapshot must be refreshed when workflow triggers, architecture families, runner assignments, path selection, or category definitions change. Workflow YAML remains authoritative over this dated explanation.

### Failure and Merge-Gate Semantics

**Current state:** Enabled hipDNN test jobs fail their workflow when their selected tests fail. The inspected hipDNN paths do not define an informational test class or a `continue-on-error` duplicate of the test signal. A skipped or unselected job is not a failure and is not evidence that its tests passed.

Workflow failure behavior does not establish GitHub merge policy. Required checks, rulesets, and branch protection are configured outside these files. Documentation may say that an enabled failure makes its workflow fail; it must not say that every listed workflow blocks merge unless the external repository policy is separately verified.

## Sanitizers and Coverage

### Sanitizer Responsibility

Sanitizers validate memory and concurrency behavior at the layer they instrument. hipDNN core owns sanitizer-clean core code; provider adapters own adapter findings; dependency findings must be reduced and routed to the dependency owner. A provider integration pass without sanitizer instrumentation is not memory-safety evidence.

[Building: Address Sanitizer Build](../Building.md#address-sanitizer-build) owns supported local configuration and execution commands.

**Current state:**

- Standalone ASAN configuration supports Linux and Windows. Repository-static configuration does not establish that every platform/test combination is currently clean.
- Standalone TSAN support is implemented for Linux, is mutually exclusive with ASAN, and has no verified hipDNN CI workflow.
- The [opt-in ASAN workflow](../../../../.github/workflows/therock-multi-arch-ci-asan.yml) provides sanitizer builds for label-enabled PRs and manual dispatches. PR-triggered sanitizer runs are build-only. Manual dispatch uses full ASAN and may run tests where sandbox mapping exists.
- The [ASAN nightly workflow](../../../../.github/workflows/therock-multi-arch-ci-asan-nightly.yml) configures Linux device-side ASAN. In the external runner configuration verified on 2026-08-17, `gfx94X` has a sandbox test assignment, `gfx950` is build-only, and `gfx125X` lacks a supported ASAN build variant. This mapping is mutable; workflow results remain authoritative.

These facts do not justify a blanket claim that ASAN tests every PR, every provider, or every configured architecture. Missing sanitizer automation and platform cleanliness remain in [Known Testing Gaps](./KNOWN_TESTING_GAPS.md).

### Coverage Responsibility

LLVM source-based coverage targets exist and are run manually; [Building](../Building.md) owns their commands and report locations. The project has an aspirational 80% code-coverage target. No checked-in hipDNN workflow currently enforces an 80% repository or per-component floor, so coverage must not be described as a PR acceptance gate.

Line coverage measures execution, not behavioral completeness, provider capability, device breadth, or numerical quality. Maintainers should use uncovered branches to guide focused tests, while judging integration completeness through explicit graph, engine, and environment observations.

## Performance and Benchmarking

### Purpose

Performance testing detects changes in graph execution and host submission without conflating provider engines, devices, or timing boundaries. It is separate from correctness testing: correctness is a prerequisite, while a performance result is a measured observation tied to an environment.

The public [ROCm dnn-benchmarking project](https://github.com/ROCm/dnn-benchmarking#readme) defines and implements current benchmark commands, result schema, and workload handling. Its CODEOWNERS routes review; hipDNN documentation defines interpretation and responsibility boundaries, not duplicate mechanics.

### Workloads and Provenance

The current workload catalog is represented by individual DVC pointer files under `Workloads/**/*.tar.gz.dvc`, including model and microbenchmark families. A benchmark record must identify the exact pointer revision and selected graph inputs, but current output does not capture that provenance automatically; record it externally. Do not refer to a single aggregate workload archive as the current catalog.

Workloads should be deterministic, reviewable, and small enough to attribute. Changes to graph structure, shapes, datatypes, validation inputs, or engine selection create a new comparison context and must be recorded with the result.

### Timing Boundaries

Benchmark reports must keep these measurements separate:

- **GPU graph event span:** device elapsed time between GPU events bracketing all work dispatched by the graph. A graph may dispatch multiple kernels, so this is not “single-kernel time.”
- **Host submission time:** host time spent enqueueing the graph in the staged timing path. It excludes subsequent device completion and must not be labeled end-to-end latency.
- **Broader elapsed time:** setup, allocation, preparation, correctness work, synchronization, or teardown may be included by outer or fallback timers. Such values require their exact boundary and must not be compared with pure host-submission samples.

Untimed warmups must complete and queued warmup work must be drained before timed samples begin. Warmup count, timed-iteration count, synchronization method, timing path, and summary statistic belong in result metadata. Current final artifacts retain summary statistics but do not serialize those run-boundary fields, so record them externally until the schema does.

### Comparison Principles

- Compare only against a baseline from the same GPU architecture. Prefer the same device model, OS, runtime, clocks/power policy, dependency versions, provider/plugin build, engine, workload revision, and benchmark configuration.
- Attribute every result to loaded provider and engine identity, engine ID, plugin version or artifact, and feature flags. Current output records engine name, ID, version, and an optional plugin path; artifact revision/hash and general feature flags require external recording. “hipDNN performance” without engine attribution is not actionable.
- Preserve separate host-submission and GPU-event summary statistics. Current final output does not preserve raw sample distributions, so any distribution-level analysis requires separate artifacts.
- Re-run suspicious changes in a controlled environment before assigning a regression. Correctness failures invalidate performance comparisons.
- Define acceptance thresholds in the owning experiment or release policy after measuring noise. This strategy intentionally defines no universal percentage, statistic, or rerun threshold.

### Layered Ownership

| Layer | Performance responsibility |
|---|---|
| Workload catalog and benchmark harness | Input provenance, timing implementation, result schema, statistics, and reproducibility |
| hipDNN core | Graph construction/serialization, provider dispatch, API overhead, and engine identity propagation |
| Provider adapter | Applicability, graph translation, engine/config selection, plugin overhead, and provider feature flags |
| Underlying library or kernel implementation | Kernel selection, compilation, workspace behavior, and device execution |
| CI or lab environment | Hardware identity, system software, clock/power controls, baseline storage, and run comparability |

CODEOWNERS can route a code review but does not by itself assign regression triage or baseline approval. Triage should first reproduce the result, identify whether movement is in host submission or GPU event span, then reduce it to the owning layer.

**Current state:** dnn-benchmarking can produce engine-name/ID/version-attributed timing summaries with an optional plugin path, but complete workload, build, timing-path, and feature provenance is absent. Its checked automation does not run an on-GPU hipDNN performance-regression gate. No checked-in automated baseline, gate policy, or triage assignment was found; external lab or team policy remains unverified.
