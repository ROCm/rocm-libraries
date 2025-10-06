# hipDNN Testing Strategy

This document outlines the comprehensive testing strategy for hipDNN, covering unit tests(white box testing), integration tests (black box testing, api tests, and end to end tests), and performance/benchmarking.

Please refer to the coding standards in [Coding Style and Naming Guidelines](../CodingStyleAndNamingGuidelines.md) to see test naming conventions we follow.

---

## 1. White Box Testing (Unit Tests) ⬜

White box tests focus on internal implementation details of hipDNN components.

### Component Comparison

| Component | Location | Purpose | GPU Testing | Environments |
|-----------|----------|---------|-------------|--------------|
| **Backend** | `backend/tests/` | Test internal implementation of hipDNN backend | Minimal/None | Windows & Linux |
| **Frontend** | `frontend/tests/` | Test internal implementation of hipDNN frontend | Minimal/None | Windows & Linux |
| **SDK** | `sdk/tests/` | Test internal implementation of hipDNN SDK | Minimal/None expected | Windows & Linux |
| **Plugin** | `plugins/<name>/tests/` | Test internal implementation of specific plugin | Minimal & fast | Windows & Linux |

Note: If a test depends on the GPU then it needs to be marked with `SKIP_IF_NO_DEVICE()` so tests run and pass correctly on CPU only machines.
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

#### SDK
- Plugins
- Data objects
- Logging
- SDK utilities
- Reference implementations

#### Plugin
- TBD based on plugin implementation
- See the recommended implementation in [Plugin Development](../PluginDevelopment.md#implementation-details)

### Common Requirements
- **Mocking**: Use GMOCK for mocking dependencies
- **Execution**: Fast execution required
- **Isolation**: Use stubbed/mocked implementations for dependencies
- **GPU Operations**: Must be marked with `SKIP_IF_NO_DEVICE()`
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
| **Requirements** | • Test only public interfaces from `backend/include/`<br>• Use stubbed plugins for controlled testing<br>• Fast running<br>• GPU operations marked with `SKIP_IF_NO_DEVICE()` |
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
| **Frontend-Backend** | `tests/frontend/` | Validate end-to-end hipDNN functionality | No - mark GPU ops with `SKIP_IF_NO_DEVICE()` | Fast | Windows & Linux |
| **Plugin Integration** | `plugins/<name>/integration_tests/` | Validate end-to-end graph support for plugin | Yes - required for validation | Can be slower | Windows & Linux |

#### Test Requirements by Type

##### Frontend-Backend
- Use fake plugins for controlled behavior
- No accuracy/solution validation (stubbed)
- Test graph creation and execution API
- Test backend descriptor creation from frontend
- Test execution flow validation

##### Plugin Integration
- Validate correctness and graph support
- Each plugin maintains its own test suite
- Test on all ASICs supported by the plugin

---

## 3. General Testing Requirements

### Code Coverage 
- **Target**: 80% overall coverage
- **Component Target**: Each sub-section should be above 80% individually
- **Enforcement**: Coverage must remain above 80% for PRs to be accepted

### Test Environment Compatibility

Tests must work in the following environments:

| Environment Type | Supported Methods |
|-----------------|-------------------|
| **CLI Build Environment** | `ninja check`, `ninja check_ctest` |
| **IDE** | Visual Studio Code and extensions like TestMate |
| **Artifacts** | • Installed testing artifacts<br>• Running built test executables |
| **Operating System** | • Windows<br>• Supported Linux distros |

> [!TIP]
> `ninja unit-check` runs fast, isolated unit and API tests.
> `ninja integration-check` runs slower, end-to-end integration tests.

### GPU Requirements
- **Without GPU**: All GPU tests must be skippable (warnings, not errors)
- **With GPU**: Tests should detect and utilize available GPU resources
- **Platform Support**: Windows & supported Linux distributions

### Green CI 🟩

For each PR, the latest commit must pass every CI pipeline listed in the [Test Plan](./TestPlan.md#prerequisites).

## 3. Performance Testing

See the [Roadmap](../Roadmap.md#testing-and-performance) for status of the upcoming performance benchmarking project, which will track performance of hipDNN and installed plugins across a broad set of graphs.
