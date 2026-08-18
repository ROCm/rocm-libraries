# hipDNN Testing Strategy

This document describes how hipDNN is tested. It keeps the original split between unit tests, integration tests, and performance tests, then adds current links and requirements.

For commands, see [Testing](./TESTING.md) and [Building](../Building.md). For test names, see [Coding Style and Naming Guidelines](../CodingStyleAndNamingGuidelines.md). For known limits, see [Known Testing Gaps](./KNOWN_TESTING_GAPS.md).

## Principles

- Test the smallest useful part.
- Keep unit tests fast and independent. GPU use is allowed when required, but GPU-dependent tests must call `SKIP_IF_NO_DEVICES()`.
- Use integration tests for public APIs and real boundaries between components, providers, libraries, runtimes, or devices.
- hipDNN tests own graph routing and plugin communication. Providers own operation results on supported GPUs.
- A skipped test gives no result for that configuration.
- A green CI job proves only that its selected tests passed in that environment; it does not define full product support.

## 1. White Box Testing (Unit Tests)

Unit tests check the internal behavior of one hipDNN component.

### Component Comparison

| Component | Location | Purpose | GPU use | Platforms |
|---|---|---|---|---|
| Backend | `backend/tests/` | Test backend implementation details | Usually none | Windows and Linux |
| Frontend | `frontend/tests/` | Test frontend implementation details | Usually none | Windows and Linux |
| Data SDK | `data_sdk/tests/` | Test Data SDK implementation details | Usually none | Windows and Linux |
| FlatBuffers SDK | `flatbuffers_sdk/tests/` | Test serialization and generated data helpers | Usually none | Windows and Linux |
| Plugin SDK | `plugin_sdk/tests/` | Test Plugin SDK helpers and base classes | Usually none | Windows and Linux |
| Test SDK | `test_sdk/tests/` | Test reference code and shared test tools | Usually none | Windows and Linux |
| Provider | `dnn-providers/<name>/tests/` | Test one provider's internal behavior | Minimal and fast | Windows and Linux |

Unit tests should avoid GPU use when practical, but they may use a GPU when the behavior cannot be tested otherwise. Every GPU-dependent unit test must call `SKIP_IF_NO_DEVICES()` so it skips cleanly on CPU-only machines.

### Test Areas by Component

#### Backend

- Descriptors
- Plugin loading and management
- Error handling
- Backend utilities
- Handles
- Graph extensions

#### Frontend

- Attributes
- Nodes
- Graph construction and flow
- Frontend utilities

#### SDKs

- Data objects and serialization
- Logging
- API helpers and base classes
- CPU reference implementations
- Test and validation utilities

#### Providers

- Graph translation
- Support checks
- Engine configuration
- Error handling
- Provider-specific utilities

### Unit Test Requirements

- Keep tests fast and repeatable.
- Use existing fakes, stubs, or mocks for outside dependencies when practical.
- Put tests beside the component's current test suite.
- Use `SKIP_IF_NO_DEVICES()` for every GPU-dependent test.
- Add a regression test for each bug fix.
- Treat the 80% code-coverage number as a goal, not a merge requirement.

## 2. Integration Tests

Integration tests check public APIs or real boundaries between hipDNN components, providers, libraries, runtimes, and devices.

### Black Box API Tests

Black box tests use public interfaces instead of internal implementation details.

#### Backend API Tests

| Attribute | Details |
|---|---|
| Location | `tests/backend/` |
| Purpose | Check the public hipDNN backend API |
| Requirements | Use public headers, controlled test plugins, and `SKIP_IF_NO_DEVICES()` for GPU paths |
| Platforms | Windows and supported Linux distributions |
| Frequency | Run for matching pull requests |

These tests cover descriptor APIs, handles, execution, plugin management, graph extensions, and logging.

### End-to-End Integration Tests

| Test type | Location | Purpose | GPU use |
|---|---|---|---|
| Frontend-to-backend | `tests/frontend/` | Check graph creation, descriptor conversion, routing, and execution flow | Optional; GPU paths must skip without a device |
| Provider integration | `dnn-providers/<name>/integration_tests/` | Check behavior specific to one provider | Required for GPU behavior |
| Shared provider integration | `dnn-providers/integration-tests/` | Run shared graphs through one provider and compare results with a reference | Required for numerical validation |

#### Frontend-to-Backend Tests

- Use controlled test plugins.
- Check graph creation and execution APIs.
- Check frontend-to-backend descriptor conversion.
- Do not claim provider numerical correctness from controlled-plugin results.

#### Provider Tests

- Each provider keeps tests for its own errors, support checks, engine behavior, determinism, and tuning controls.
- Test every GPU family the provider claims to support.
- Use provider category and configuration files as the source of truth.

#### Shared Provider Integration Tests

`hipdnn_integration_tests` is provider-agnostic. It contains one shared graph set. Each provider runs the executable independently with its own plugin and engine; one run never loads or compares several providers.

Use the shared suite for graph execution and numerical comparison against the CPU reference executor. See the [provider integration test guide](../../../../dnn-providers/integration-tests/README.md) for bundles, provider configuration, and categories.

### Graph Validation

The CPU Graph Executor provides expected results for integration tests. A provider must first accept the graph, choose an engine, and run it. The test then compares the provider result with the reference result. See [CPU Graph Executor Design](../rfcs/0001_CpuGraphExecutorDesign.md).

## Validation Responsibilities and GPU Coverage

| Component | Responsibility |
|---|---|
| hipDNN core | Public APIs, graph data, routing, plugin lifetime, and frontend/backend flow |
| Shared provider integration | Shared graph cases, reference results, tolerances, and the provider-agnostic harness |
| Providers | Graph translation, support checks, engine selection, provider skips and tolerances, and operation results on supported GPUs |
| Lower-level libraries and kernels | Kernel selection, compilation, workspace use, and device execution |
| CI or lab | Runners, hardware, system software, and repeatable test environments |

A green hipDNN core job proves that its selected routing and API tests passed. It does not prove that every provider, operation, shape, datatype, engine, or GPU is supported. Provider support comes from provider documentation and runtime support checks.

[CODEOWNERS](../../../../.github/CODEOWNERS) chooses reviewers; it does not define product support.

## Key Quality Concerns

- **Frontend-to-plugin data path:** hipDNN must pass the right tensors, attributes, and scalars to the selected provider and return its results correctly.
- **Operation results:** Provider integration tests must compare graph results with a reference result.
- **Public API behavior:** Backend C and frontend C++ APIs must keep their documented behavior.
- **Windows and Linux:** GPU-dependent tests must skip cleanly when no device is available.
- **Memory and thread safety:** Use ASAN and TSAN for the code they support.
- **Static analysis:** Use `clang-tidy` to catch common C++ and HIP problems.

## 3. General Testing Requirements

### Test Categories

The public test categories are `quick`, `standard`, `comprehensive`, and `full`. In core hipDNN, `quick` currently matches every core test and higher categories inherit that set. Provider categories may differ; see the [provider integration test guide](../../../../dnn-providers/integration-tests/README.md).

### Code Coverage

- Enable coverage with `-DHIPDNN_ENABLE_COVERAGE=ON`.
- Build the `coverage` target to create reports under `coverage-report/`.
- The 80% coverage number is a goal, not an enforced merge rule.
- Coverage shows which code ran; it does not prove correct behavior or GPU support.

### GPU Requirements

- Tests must run on CPU-only machines.
- GPU-dependent tests must call `SKIP_IF_NO_DEVICES()`.
- A skip is not a pass for that GPU configuration.
- Tests should use every available GPU only when the test's purpose requires it.

### Static Analysis

hipDNN uses `clang-tidy`. Linux builds and Linux superbuild CI enable it; Windows builds and Windows superbuild CI disable it. Local developers can use `-DENABLE_CLANG_TIDY=ON` or the `hipdnn-tidy` target. See [Building: Clang Tools](../Building.md#clang-tools).

### Sanitizers

Standalone ASAN tests are run manually. Pull-request ASAN jobs build host-ASAN binaries but do not run their tests. Manual workflow runs and the scheduled nightly can run ASAN tests where matching GPU runners are available.

TSAN supports Linux host code. ASAN and TSAN use separate build configurations. See [Testing](./TESTING.md#asantsansanitizer-coverage), [Building](../Building.md#address-sanitizer-build), and [Known Testing Gaps](./KNOWN_TESTING_GAPS.md#sanitizers-and-platforms).

### CI Results

An enabled test failure fails its workflow. A skipped or unselected job is not proof that tests passed. GitHub settings outside this repository decide which jobs block a merge.

### Flaky Tests

Tests should give a reliable result. Fix or remove a test that fails intermittently; do not leave it as a non-blocking failure.

## 4. Performance Testing

Use [ROCm dnn-benchmarking](https://github.com/ROCm/dnn-benchmarking#readme) for manual graph benchmarks.

- Record workload revision, graph inputs, GPU, provider, engine, plugin version or artifact, and software build.
- Compare only with a baseline from the same GPU architecture and, when possible, the same system setup.
- Keep GPU graph time separate from host submission time.
- Re-run suspicious results before calling them regressions.
- Do not compare performance when correctness tests fail.

hipDNN has no checked-in automated GPU performance gate. See [Known Testing Gaps](./KNOWN_TESTING_GAPS.md#automated-performance-checks).
