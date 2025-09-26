# Coding Style & Naming Guidelines

This document defines the canonical project-wide coding and test naming conventions.

## Table of Contents

- [1. Naming Summary](#1-naming-summary)
- [2. File & Class Naming](#2-file--class-naming)
- [3. Functions](#3-functions)
  - [3.1 Unused Function Arguments](#31-unused-function-arguments)
- [4. Variables](#4-variables)
- [5. Members](#5-members)
- [6. Globals](#6-globals)
- [7. Interfaces](#7-interfaces)
- [8. Enums](#8-enums)
- [9. Constants](#9-constants)
- [10. Namespaces](#10-namespaces)
- [11. Test Naming Guidelines](#11-test-naming-guidelines)
  - [11.1 Keywords](#111-keywords)
  - [11.2 Unit Tests](#112-unit-tests)
    - [Naming Examples](#naming-examples)
    - [File Naming](#file-naming)
  - [11.3 Integration Tests](#113-integration-tests)
    - [Naming Examples](#naming-examples-1)
    - [File Naming](#file-naming-1)
  - [11.4 Test Case Naming](#114-test-case-naming)
  - [11.5 Rationale](#115-rationale)
- [12. Examples](#12-examples)
  - [Class & File](#class--file)
  - [Interface](#interface)
  - [Constant & Enum](#constant--enum)
  - [Test (gtest)](#test-gtest)
- [13. Decision Checklist](#13-decision-checklist)
- [14. Deviation Process](#14-deviation-process)
- [15. Automated Tooling](#15-automated-tooling)
  - [15.1 Clang-Tidy Rules](#151-clang-tidy-rules)
  - [15.2 Test Naming Enforcement Tool](#152-test-naming-enforcement-tool)

## 1. Naming Summary

| Kind | Format | Example |
|------|--------|---------|
| Class / Struct | PascalCase | TensorDescriptor |
| Interface | I + PascalCase | ITensorView |
| Function | camelCase | buildGraph() |
| Variable (local / parameter) | camelCase | workspaceSize |
| Private Member variable | _camelCase | _cachedPlan |
| Static variable | s_camelCase | s_engineCount |
| Global variable | g_camel_case | g_global_state |
| Constant / Macro | UPPER_CASE | MAX_WORKSPACE_BYTES |
| Enum Type | PascalCase | EngineMode |
| Enum Value | UPPER_SNAKE | ENGINE_MODE_DEFAULT |

## 2. File & Class Naming

- Prefer the filename to match the primary class it contains: `ExecutionPlan.hpp`, `ExecutionPlan.cpp`.
- Utility collections (no single dominant class) may use a name that makes sense to describe the contents (e.g. `Error.hpp`).
- Interfaces: prefix `I`, e.g. `IAllocator.hpp`.
- Keep one major class per file when practical.
- Avoid excessively long filenames; lean on directory structure for grouping.

## 3. Functions

- Use descriptive action-oriented verbs: `createPlan`, `finalizeConfig`, `launchKernels`.

### 3.1 Unused Function Arguments

- Prefer use of `[[maybe_unused]]` rather than commenting-out argument names or using std::ignore
  - Exception is for arguments that *shall not* be used, such as the case in a legacy version of a method that has an argument that is no longer relevant and shouldn't be used.  In this case, comment-out the argument name and leave a comment indicating the reason.

## 4. Variables

- Favor clarity over abbreviation: prefer `intermediateSize` to `intSz`.

## 5. Members

- Private / protected data members: prefix single underscore `_` then camelCase (`_opGraph`).
- Static data members: `s_camelCase`.
- Exposed constants inside a class: `static constexpr` UPPER_CASE.
- Plain structs whose intent is a passive aggregate (all or mostly public data) DO NOT prefix member names with `_`; just use camelCase.
  - Rationale: underscores communicate encapsulation; POD-style structs are transparent.

Example:

```cpp
struct TensorExtent {
    int n;
    int c;
    int h;
    int w;
};
```

If later you add invariants or non-trivial behavior, consider converting to a class and applying the underscore rule to newly private members.

## 6. Globals

- Avoid unless absolutely required; prefix `g_` to make visibility explicit.

## 7. Interfaces

- Naming: `IInterfaceName`.
- Keep pure abstract; avoid data members.

## 8. Enums

- Enum type name: PascalCase (e.g. `EngineMode`, `ConvolutionMode`).
- Enumerator names: UPPER_SNAKE (`ENGINE_MODE_DEFAULT`, `ENGINE_MODE_DETERMINISTIC`).
- When mirroring external APIs, keep exact enumerator spellings.

## 9. Constants

- UPPER_CASE with optional single underscores: `DEFAULT_ALIGNMENT`, `MAX_TENSOR_RANK`.
- Prefer `constexpr` over macros when possible.

## 10. Namespaces

- lower_snake_case with single underscores
- Most code should fit generally within a few namespaces
  - `hipdnn_<component>`: (eg. hipdnn_frontend) Contains all basic code required for the component
    - `utilities`: Contains code that can aid and assist in using component code
    - `test_utilities`: Contains code that can aid and assist in testing component code

## 11. Test Naming Guidelines

GoogleTest reserves underscores in test suite and test names for future expansion. Current repository names with underscores risk future incompatibility; we proactively constrain test suite naming.

Rules below apply ONLY to the TestSuite name (first parameter of `TEST` / `TEST_F` / `TEST_P`). The TestCase (second parameter) can be descriptive but should still avoid the reserved keywords where noted.  When writing parameterized tests the prefix(parameter name is `InstantiationName`) in the `INSTANTIATE_TEST_SUITE_P` macro can be left blank or used for another purpose that make sense for that test.

### 11.1 Keywords

- **Integration**: Only for integration tests, always first if present
- **Test**: Mainly for unit tests, always first if test is not an Integration test.
- **Gpu**: Optional after Integration or Test but before suite name if the test needs Gpu support.
- **Datatypes**: Bfp16, Fp16, Fp32. Always last if present.

### 11.2 Unit Tests

In most cases unit style tests should be named so the directly mirror the class under test.  If the class is named `MyClass`, then the test suite should be named `TestMyClass`.  In general these kinds of tests should try to avoid using anything that requires Gpu support.  This is not always possible, in the cases where Gpu support is required, the test suite should be named `TestGpuMyClass`.

#### Naming Examples

```cpp
TestBackendLogger
TestHandle
TestGpuHandle
TestBatchnormBwdPlan
TestBatchnormBwdPlanFp32
TestBatchnormBwdPlanFp16
```

#### File Naming
The test file name should mirror the primary test suite it contains.  For example, if the main test in a suite is `TestMyClass`, the file should be named `TestMyClass.cpp`.  That same file may also contain `TestGpuMyClass` but it is not the primary test suite so the file name does not need to reflect it.

### 11.3 Integration Tests

See [TestingStrategy.md](testing/TestingStrategy.md) for more information on integration tests. Integration tests should be named to reflect the feature or component under test.

#### Naming Examples

```cpp
IntegrationGpuBatchnormBackwardNchwFp32
IntegrationGpuBatchnormBackwardNchwBfp16
IntegrationGpuBatchnormBackwardNchwFp16
IntegrationGraphFusion
```

#### File Naming
For integration tests, the main test suite might be named `IntegrationGpuFeatureX` but have several child suites like `IntegrationGpuFeatureXFp32` and `IntegrationGpuFeatureXBfp16`. The parent suite name is the primary suite, so the file name should be `IntegrationGpuFeatureX.cpp`. 

### 11.4 Test Case Naming

May be richly descriptive `HandlesLargeStride`, `RejectsMismatchedLayouts` or very simple `Correctness`, `Accuracy`. Avoid duplicating suite-level keywords (`Integration`, `Gpu`, datatype tokens) redundantly inside the test case name.  The test case name is the preferred place to list the shape/layout variant being tested (e.g. `Nchw`, `Nhwc`).  In general there should not be duplication across the suite name and test case name.  Otherwise the naming of the test case is entirely up to the developer.

### 11.5 Rationale

- Ordering enforces quick visual parsing (environment → scope → subject → specialization).
- Avoid underscores to remain future-proof with gtest evolution.
- Suffix datatype to emphasize functional context before precision variant.
- Consistent pattern simplifies filtering (e.g. `--gtest_filter=*Gpu*Fp32`).

## 12. Examples

### Class & File

File: `ExecutionPlan.hpp`
```cpp
class ExecutionPlan {
public:
    static constexpr int MAX_STEPS = 8;

    explicit ExecutionPlan(int initialSteps);
    void buildGraph();
    int stepCount() const;

private:
    int _stepCount;
    bool _isFinalized;
};
```

### Interface

```cpp
class IAllocator {
public:
    virtual ~IAllocator() = default;
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
};
```

### Constant & Enum

```cpp
enum EngineMode {
    ENGINE_MODE_DEFAULT = 0,
    ENGINE_MODE_DETERMINISTIC = 1
};

constexpr size_t MAX_WORKSPACE_BYTES = 1ull << 32;
```

### Test (gtest)

```cpp
TEST(IntegrationGpuGraphFusionFp32, FusesThreeSequentialOps) {
    // ...
}
TEST(IntegrationGpuGraphFusionBfp16, FusesThreeSequentialOps) {
    // ...
}

TEST(TestExecutionPlan, BuildsGraphCorrectly) {
    // ...
}
```

## 13. Decision Checklist

When adding new code, verify:
- Names follow the table in Section 1.
- File name matches main class (or is a justified utility collection).
- Test suite names follow ordering & allowed tokens.
- Layout tokens appear before datatype tokens when both used.
- No stray underscores in test suite names.

## 14. Deviation Process

If an external API or standard library interop forces divergence (e.g., fixed enum value names), document the exception with a brief comment near the declaration.

## 15. Automated Tooling

The repository includes automated tooling to enforce coding standards and maintain consistency across the codebase.

### 15.1 Clang-Tidy Rules

The project uses clang-tidy to automatically enforce many of the coding style guidelines defined in this document. The configuration can be found in `.clang-tidy` at the repository root.

The CI pipeline automatically runs clang-tidy on all pull requests to ensure compliance before merging.

### 15.2 Test Naming Enforcement Tool

*[Placeholder: A dedicated test naming enforcement tool is planned to automatically validate that all test names follow the conventions outlined in Section 10]*

This tool will:
- Parse all test files and extract TEST/TEST_F suite names
- Validate ordering of keywords (Integration, Gpu, Feature, Layout, Datatype)
- Check for prohibited underscores in test suite names
- Generate reports on non-compliant test names
- Integrate with CI to block merges with invalid test names

---

Adhering to these rules and utilizing the automated tooling maintains readability, consistency, and tooling friendliness across the codebase.
