# hipDNN Testing Strategy

This document outlines the comprehensive testing strategy for hipDNN, covering unit tests (white box testing), integration tests (black box testing, api tests, and end to end tests), and performance/benchmarking.

Please refer to the coding standards in [Coding Style and Naming Guidelines](../CodingStyleAndNamingGuidelines.md) to see test naming conventions we follow.

---

## 1. White Box Testing (Unit Tests) ⬜

White box tests focus on internal implementation details of hipDNN components.

### Component Comparison

| Component | Location | Purpose | GPU Testing | Environments |
|-----------|----------|---------|-------------|--------------|
| **Backend** | `backend/tests/` | Test internal implementation of hipDNN backend | Minimal/None | Windows & Linux |
| **Frontend** | `frontend/tests/` | Test internal implementation of hipDNN frontend | Minimal/None | Windows & Linux |
| **Data SDK** | `data_sdk/tests/` | Test internal implementation of Data SDK | Minimal/None | Windows & Linux |
| **Plugin SDK** | `plugin_sdk/tests/` | Test internal implementation of Plugin SDK | Minimal/None | Windows & Linux |
| **Test SDK** | `test_sdk/tests/` | Test internal implementation of Test SDK | Minimal/None | Windows & Linux |
| **Provider** | `dnn-providers/<name>/tests/` | Test internal implementation of a specific provider | Minimal & fast | Windows & Linux |

Note: If a test depends on the GPU then it needs to be marked with `SKIP_IF_NO_DEVICES()` so tests run and pass correctly on CPU only machines.
---

### Test Categories by Component

#### Backend
- Descriptors
- Plugin system
- Error handling
- Backend utilities
- Handle
- Graph extensions

#### Frontend
- Attribute
- Node
- Graph construction & flow
- Frontend utilities

#### Data SDK
- Data objects
- Logging
- SDK utilities

#### Plugin SDK
- Plugin API utilities
- Engine base classes

#### Test SDK
- CPU reference implementations
- Test utilities
- Validation helpers

#### Plugin
- TBD based on plugin implementation
- See the recommended implementation in [Plugin Development](../PluginDevelopment.md#implementation-details)

### Common Requirements
- **Mocking**: Use GMOCK for mocking dependencies
- **Execution**: Fast execution required (Time limits enabled in TheRock CI)
- **Isolation**: Use stubbed/mocked implementations for dependencies
- **GPU Operations**: Must be marked with `SKIP_IF_NO_DEVICES()`
- **Coverage**: Each component should maintain >80% code coverage

---

## 2. Integration Tests

### Black Box Integration Tests ⬛

Black box tests validate the public API without knowledge of internal implementation.  These are a type of integration test.

#### Backend API Tests

| Attribute | Details |
|-----------|---------|
| **Location** | `tests/backend/` |
| **Purpose** | Validate API of hipDNN backend works as expected |
| **Requirements** | • Test only public interfaces from `backend/include/`<br>• Use stubbed plugins for controlled testing<br>• Fast running<br>• GPU operations marked with `SKIP_IF_NO_DEVICES()` |
| **Environments** | Windows & supported Linux distros |
| **Frequency** | Run on each PR |

##### Test Categories
- Descriptor APIs (create, get/set properties, destroy)
  - Engine API
  - Engine config API
  - Engine heuristic API
  - Execution plan API
  - Handle API
  - Variant pack API
  - Graph API
  - Graph extension API for serialized graph structures
- Backend execute API
- Plugin management extension API
- Logging extension API

---

### End to End Integration Tests 🧩

Integration tests validate end-to-end functionality across components.

#### End to End Integration Test Comparison

| Test Type | Location | Purpose | GPU Required | Test Speed | Environments |
|-----------|----------|---------|--------------|------------|--------------|
| **Frontend-Backend** | `tests/frontend/` | Validate end-to-end hipDNN functionality | No - mark GPU ops with `SKIP_IF_NO_DEVICES()` | Fast | Windows & Linux |
| **Provider Integration** | `dnn-providers/<name>/integration_tests/` | Validate end-to-end graph support for a provider | Yes - required for validation | Can be slower | Windows & Linux |
| **External Integration** | `dnn-providers/integration-tests/` | Run graphs through a hipDNN provider plugin and compare results against a reference executor, across providers | Yes - required for validation | Can be slower | Windows & Linux |

#### Test Requirements by Type

##### Frontend-Backend
- Use fake plugins for controlled behavior
- No accuracy/solution validation (stubbed)
- Test graph creation and execution API
- Test backend descriptor creation from frontend
- Test execution flow validation

##### Provider Integration
- Validate correctness and graph support
- Each provider maintains its own test suite
- Test on all ASICs supported by the provider
- Tests are divided into two categories described by the prefix argument passed to INSTANTIATE_TEST_SUITE_P
  - **Smoke** - These tests are designed to test features using the smallest possible shape and run quickly (combined smoke test run time must be under 5 mins)
  - **Full** - These tests can contain regression shapes, large shapes, or slow shapes

##### External Integration
- A shared, cross-provider harness (`hipdnn_integration_tests`) that runs graphs through a hipDNN provider plugin and compares the results against a reference executor, rather than living inside any one provider
- Under a superbuild the provider plugin is discovered automatically; standalone, pass `--test-article /path/to/libmiopen_plugin.so`
- Tiered by the `INSTANTIATE_TEST_SUITE_P` prefix (`Smoke`/`Standard`/`Comprehensive`/`Full`); tiers cascade so each higher tier includes the lower ones, and any suite without a tier prefix runs in smoke
- See the [external integration tests README](../../../../dnn-providers/integration-tests/README.md) for adding operations and shapes

### Graph Validation
We use reference implementations via the CPU Graph Executor to validate correctness of graph execution in integration tests. See the [CPU Graph Executor Design Document](../rfcs/0001_CpuGraphExecutorDesign.md) for more details.

---

## Validation Responsibility and GPU Coverage

hipDNN is primarily a routing library: the frontend API hands a graph to a provider plugin that knows how to run it. That shapes what "GPU coverage" means here, and splits validation responsibility between hipDNN and its providers:

- **hipDNN library** is largely GPU-agnostic. It does not implement per-architecture kernels, so it does not own a per-`gfx` correctness matrix. Its own on-device testing is the **backend/end-to-end integration tests** that confirm hipDNN marshals the right data to the plugin and returns the plugin's results faithfully - i.e. that the routing and data path are correct, not that an operation is numerically correct on a given ASIC.
- **Operation correctness** across providers is validated by the cross-provider suite in `dnn-providers/integration-tests/`, which runs graphs through a provider plugin and compares the results against a reference executor. This is where `gfx`-specific accuracy coverage lives, so ASIC coverage is effectively delegated to the providers: the suite exercises whatever GPU is present, and each provider is responsible for the architectures it supports.

The cross-provider suite is the default place to add "does this graph run and verify on this engine" tests (authored as data-driven bundles); a provider's own `integration_tests/` directory is reserved for cases that are *not* just running a graph, such as unsupported/error paths, determinism, and benchmarking knobs. For the specifics (bundle formats, tiers, how to add tests, per-provider configuration), see the [integration tests README](../../../../dnn-providers/integration-tests/README.md).

What hipDNN itself validates directly, by OS:

| Configuration | Validated | Notes |
|---------------|-----------|-------|
| Linux | Yes | Full library + integration test suites |
| Windows | Yes | Same suites; some GPU/HIP tests are unsupported on Windows, so it validates a subset |
| Specific `gfx` targets | Delegated to providers | On-device tests exercise whatever GPU is present; per-ASIC operation correctness is each provider's responsibility |

> [!NOTE]
> CI runs the smoke and standard test tiers as its per-change gate. The exact tier-to-cadence mapping and which tiers gate versus run nightly are still evolving and subject to change; see the [integration tests README](../../../../dnn-providers/integration-tests/README.md) for the current tier definitions.

## Key Quality Concerns

The following are the highest-priority correctness guarantees, and how each is validated:

- **Frontend-to-plugin data path** - hipDNN must pass the correct tensors, attributes, and scalars to the selected provider and return the plugin's results faithfully. Validated by the backend/end-to-end integration tests on a real device.
- **Numerical correctness of operations** - the results of executed graphs must be correct. Validated by the cross-provider integration suite, which runs graphs through a provider plugin and compares the output against a reference executor.
- **Public API compatibility** - the backend C API and frontend C++ API are the contract for consumers. Validated by the black-box API tests (`tests/backend/`) and frontend tests.
- **Cross-platform (Windows and Linux)** - the library must build and behave correctly on both. Validated by running the suites on both OSes, with GPU-dependent tests guarded by `SKIP_IF_NO_DEVICES()`.
- **Memory safety** - no leaks or invalid accesses. Validated by sanitizer runs (currently a manual process; see [Testing § Address Sanitizer](../Testing.md#address-sanitizer)).

---

## 3. General Testing Requirements

### Code Coverage
- **Tool**: LLVM source-based coverage (`-fprofile-instr-generate -fcoverage-mapping`, then `llvm-profdata` + `llvm-cov`). Enable with `-DHIPDNN_ENABLE_COVERAGE=ON` and build the `coverage` target; reports land in `coverage-report/` (an lcov-format `coverage.info` is also exported).
- **Target**: 80% overall coverage
- **Component Target**: Each sub-section should be above 80% individually
- **Enforcement**: Coverage must remain above 80% for PRs to be accepted

> [!NOTE]
> This 80% target is a **code-coverage** measure (how many lines the tests execute). For how behavior and configuration coverage is divided between hipDNN and the providers, see [Validation Responsibility and GPU Coverage](#validation-responsibility-and-gpu-coverage).

### Test Environment Compatibility

Tests must work in the following environments:

| Environment Type | Supported Methods |
|-----------------|-------------------|
| **CLI Build Environment** | `ninja hipdnn-check`, `ninja hipdnn-check-verbose` (standalone also has the unprefixed `check`) |
| **IDE** | Visual Studio Code and extensions like TestMate |
| **Artifacts** | • Installed testing artifacts<br>• Running built test executables |
| **Operating System** | • Windows<br>• Supported Linux distros |

> [!TIP]
> `ninja hipdnn-unit-check` runs fast, isolated unit and API tests (also: `hipdnn-unit-check-verbose`).<br>
> `ninja hipdnn-integration-check` runs slower, end-to-end integration tests (also: `hipdnn-integration-check-verbose`).<br>
> In a standalone `projects/hipdnn` build the unprefixed aliases (`check`, `unit-check`, `integration-check`, ...) also exist. See [Testing](../Testing.md#ctest-vs-check-targets) for the full ctest/check-target mapping.

### GPU Requirements
- **Without GPU**: All GPU tests must be skippable (warnings, not errors)
- **With GPU**: Tests should detect and utilize available GPU resources
- **Platform Support**: Windows & supported Linux distributions

### Green CI 🟩

For each PR, the latest commit must pass every CI pipeline listed in the [Test Plan](./TestPlan.md#prerequisites).

### Flaky Tests

Every test in the suite is expected to be a reliable signal. hipDNN does not quarantine flaky tests: a test that fails intermittently is either **fixed** or **removed**, never left in place as a non-blocking exception.

## 3. Performance Testing

See the [Roadmap](../Roadmap.md#testing-and-performance) for status of the upcoming performance benchmarking project, which will track performance of hipDNN and installed plugins across a broad set of graphs.
