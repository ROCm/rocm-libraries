# StinkyTofu Tests

This directory contains all tests for the StinkyTofu library.

## Directory Structure

```
tests/
+-- CMakeLists.txt           # Test build configuration
+-- StinkyTofuTestMain.cpp   # Main test entry point (GTest runner)
+-- unit/                    # Unit tests
    +-- WaitCntInsertionTest.cpp
    +-- IntrusiveListTest.cpp
    +-- PassManagerTest.cpp
    +-- StinkyAsmEmitterTest.cpp
```

## Building Tests

From the build directory:

```bash
# Build all test executables
make tests

# Or build specific test executable
make unit_tests
```

## Running Tests

### Run All Tests

```bash
# Run unit tests directly
./tests/unit_tests

# Run all tests using CTest
ctest

# Run only unit tests using CTest label
ctest -L unit_tests
```

### Run Specific Test Suites

Using GTest filters directly:

```bash
# Run all AsmEmitter tests
./tests/unit_tests --gtest_filter="AsmEmitterTest.*"

# Run specific test
./tests/unit_tests --gtest_filter="AsmEmitterTest.EmitWithCycleInfo"

# Run multiple test suites
./tests/unit_tests --gtest_filter="AsmEmitterTest.*:PassManagerFlowTest.*"
```

Using CTest:

```bash
# Run all unit tests (by label)
ctest -L unit_tests

# Run tests matching pattern
ctest -R "AsmEmitter"

# Run tests NOT matching pattern
ctest -E "IntrusiveList"

# Combine label and pattern
ctest -L unit_tests -R "AsmEmitter"
```

### Other CTest Options

```bash
# List all tests without running
ctest -N

# List tests with a specific label
ctest -L unit_tests -N

# Verbose output
ctest -V

# Show output only on failure
ctest --output-on-failure
```

## Test Organization

### Unit Tests (`unit/`)

Unit tests verify individual components in isolation:
- **PassManagerTest**: Tests for the pass manager infrastructure
- **IntrusiveListTest**: Tests for the intrusive list data structure
- **WaitCntInsertionTest**: Tests for the configurable wait count pass
- **StinkyAsmEmitterTest**: Tests for the assembly emitter

## Adding New Tests

### Adding a Unit Test

1. Create your test file in `tests/unit/`:
   ```cpp
   #include <gtest/gtest.h>
   #include "stinkytofu/core/stinkytofu.hpp"

   TEST(MyNewTest, BasicTest) {
       // Your test code
       EXPECT_EQ(1, 1);
   }
   ```

2. Add the file to `tests/CMakeLists.txt`:
   ```cmake
   set(UNIT_TEST_FILES
       unit/PassManagerTest.cpp
       unit/IntrusiveListTest.cpp
       unit/WaitCntInsertionTest.cpp
       unit/asm/StinkyAsmEmitterTest.cpp
       unit/MyNewTest.cpp  # Add your new test here
   )
   ```

3. Rebuild:
   ```bash
   make unit_tests
   # or build all test executables
   make tests
   ```

4. Run your new tests:
   ```bash
   ./tests/unit_tests --gtest_filter="MyNewTest.*"
   ctest -L unit_tests -R "MyNewTest"
   ```

## Test Conventions

- Use descriptive test names: `TEST(Component, WhatItDoes)`
- One test file per component or module
- Use test fixtures for shared setup/teardown
- Keep tests focused and independent
- Use `EXPECT_*` for non-fatal assertions
- Use `ASSERT_*` for fatal assertions (test stops on failure)

## Test Categories

### Current Test Types

- **Unit Tests** (`unit/`) - Tests for individual components in isolation
  - Executable: `unit_tests`
  - Label: `unit_tests`
  - Command: `./tests/unit_tests` or `ctest -L unit_tests`

### Adding New Test Types

To add a new test category (e.g., integration tests, benchmarks), follow this pattern in `CMakeLists.txt`:

```cmake
# Integration Tests (example)
set(INTEGRATION_TEST_FILES
    integration/YourIntegrationTest.cpp
)
list(TRANSFORM INTEGRATION_TEST_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

add_executable(integration_tests ${TEST_MAIN} ${INTEGRATION_TEST_FILES})
target_link_libraries(integration_tests PUBLIC ${TEST_LINK_LIBS})
target_include_directories(integration_tests PRIVATE ${TEST_INCLUDE_DIRS})
add_dependencies(integration_tests tablegen_generated)

gtest_discover_tests(integration_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 60
    PROPERTIES LABELS "integration_tests"
)

# Add to the 'tests' target
add_dependencies(tests integration_tests)
```

This ensures:
1. Each test type has its own executable
2. Labels are correctly applied per test type
3. `make tests` builds all test executables
4. Tests can be filtered by type using labels

## Python Tests

Python tests use pytest and are automatically integrated into CTest when `STINKYTOFU_BUILD_PYTHON=ON`.

### Running Python Tests via CTest

```bash
# Build with Python support
cmake -DSTINKYTOFU_BUILD_PYTHON=ON ..
make

# Run all Python tests (fast: ~0.8s for 23 tests)
ctest -L python

# Run specific test file
ctest -R python.test_ir_basic

# Show output on failure
ctest -L python --output-on-failure

# Verbose mode (shows all output)
ctest -L python -V
```

### Running Individual Tests Directly with Pytest

For debugging specific test cases, use pytest directly with proper PYTHONPATH:

```bash
# Set PYTHONPATH to find the built Python module
export PYTHONPATH=<build_dir>/lib

# Run one specific test
pytest python_module/tests/test_ir_basic.py::TestMemoryManagement::test_module_destruction -v

# Run all tests in a class
pytest python_module/tests/test_ir_basic.py::TestMemoryManagement -v

# Run tests matching a pattern
pytest python_module/tests/test_ir_basic.py -k "memory" -v

# Enable print statements and debugger
pytest python_module/tests/test_ir_basic.py::test_name -v -s --pdb
```

### Python Test Organization

- Each `test_*.py` file = one CTest entry
- Tests are in `python_module/tests/`
- CTest suppresses output by default (use `--output-on-failure` or `-V` to see details)
