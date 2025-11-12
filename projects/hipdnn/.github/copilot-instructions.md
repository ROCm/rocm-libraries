# Copilot Rules for hipDNN Project

## C++ Code Style

### Naming Conventions
- Use CamelCase for class and struct names (e.g., `BatchNormTestCase`, `SimpleTensorBundle`)
- Use camelBack for functions, variables, and private members (e.g., `setupEnvironment()`, `tensorData`)
- Prefix private members with underscore (e.g., `_handle`, `_testData`)

### File Headers
- Always add copyright header to all source files:
  ```cpp
  // Copyright © Advanced Micro Devices, Inc., or its affiliates.
  // SPDX-License-Identifier:  MIT
  ```
- For header files (.h, .hpp), add `#pragma once` immediately after copyright

### Code Practices
- Use `auto` when initializing with a cast to avoid duplicating the type name
  ```cpp
  // Good
  auto tensor = static_cast<float*>(data);
  // Avoid
  float* tensor = static_cast<float*>(data);
  ```
- Use auto when initializing variables, unless the type is not obvious.
- always use {}'s for if/for/while bodies, even if single line
  ```cpp
    // Good
    if(condition)
    {
        doSomething();
    }
    // Avoid
    if(condition) doSomething();
  ```

### Build System
- Always use CMake for managing C/C++ dependencies
- When discussing dependencies or build configuration, provide CMake-based solutions

### Serialization
- Use Flatbuffers for serialization needs
- Provide Flatbuffer schema definitions when creating serializable data structures

### Testing
- Use Google Test (gtest) framework for all C/C++ tests
- Never generate a main() function in test files - gtest provides its own
- Use TEST(), TEST_F(), or TEST_P() macros as appropriate

#### Test Naming Guidelines

Rules below apply ONLY to the TestSuite name (first parameter of `TEST` / `TEST_F` / `TEST_P`). The TestCase (second parameter) can be descriptive but should still avoid the reserved keywords where noted.
TestCase should be PascalCase.

**Ordering & Composition (left → right):**

1. Optional `Integration` prefix for integration tests.
2. Optional `Gpu` (immediately after `Integration` if both apply) for GPU-required tests.
3. Core Feature / Subject under test (PascalCase, no underscores).
4. Optional Datatype token (`Bfp16`, `Fp16`, `Fp32`) at the end.

Omit any category that does not apply.

### 10.1 Keywords (reserved positions)

- **Integration** (only for integration tests, always first if present).
- **Gpu** (always first unless preceded by Integration).
- **Datatypes**: Bfp16, Fp16, Fp32.

### 10.2 Unit Tests

In most cases unit style tests should be named so the directly mirror the class under test.  If the class is named `MyClass`, then the test suite should be named `TestMyClass`.  In general these kinds of tests should try to avoid using anything that requires Gpu support.  This is not always possible, in the cases where Gpu support is required, the test suite should be named `GpuTestMyClass`.

### 10.3 Valid Examples

```cpp
IntegrationGpuConvolutionPlannerNchwFp32
GpuTestActivationKernelNchwFp32
GpuTestExecutionPlanBuilderFp32
GpuTestExecutionPlanBuilderNchw
IntegrationGraphFusion
TestConvolutionHeuristicsFp32
TestConvolutionHeuristics
```

## Example Code Structure

```cpp
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once  // For header files only

#include <memory>
#include <vector>

// Class names in PascalCase
class MyTestClass
{
public:
    // Public methods in camelCase
    void setupTest();
    int getValue() const;

private:
    // Private members prefixed with _
    int _testValue;
    std::vector<float> _data;
};

// Functions in camelCase
void processData(const MyTestClass& obj);
```

## Test File Example

```cpp
// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

class MyTestFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup code
    }

private:
    int _testData;
};

TEST_F(MyTestFixture, testSomething)
{
    // Test implementation
}
