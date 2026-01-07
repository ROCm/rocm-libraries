# StinkyTofu Python Module Testing Guide

This guide covers testing the StinkyTofu Python module using **pytest**.

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Set PYTHONPATH and run all tests
cd /path/to/rocm-libraries/shared/stinkytofu
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -v
```

## Running Tests

### Run All Tests
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -v
```

### Run Specific Test File
```bash
# Run basic tests
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/test_basic.py -v

# Run MFMA tests
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/test_mfma.py -v

# Run gfx1250-specific instruction tests
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/test_gfx1250.py -v
```

### Run Specific Test
```bash
# Run a specific test from test_basic.py
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/test_basic.py::test_composite_instruction -v

# Run a specific test from test_mfma.py
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/test_mfma.py::TestBasicMFMA::test_mfma_bf16_gfx942 -v
```

### Run Tests with Output (-s shows print statements)
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -v -s
```

## Using Markers

Tests are organized with markers for easy filtering:

### Run Only Composite Instruction Tests
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -m "composite" -v
```

### Run Only Architecture-Specific Tests
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -m "architecture" -v
```

### Run Only MFMA Instruction Tests
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -m "mfma" -v
```

### Run Only Sparse Matrix Tests
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -m "sparse" -v
```

### Exclude Slow Tests
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -m "not slow" -v
```

## Keyword Filtering

### Run Tests Matching a Keyword
```bash
# Run all tests with "valu" in the name
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -k "valu" -v

# Run all tests for gfx942
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -k "gfx942" -v
```

## Coverage Reports

### Generate HTML Coverage Report
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ \
    --cov=stinkytofu \
    --cov-report=html \
    --cov-report=term
```

View the report: `open htmlcov/index.html`

### Terminal Coverage Report
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ \
    --cov=stinkytofu \
    --cov-report=term-missing
```

## Parallel Execution

Run tests in parallel for faster execution:

```bash
# Auto-detect number of CPUs
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -n auto

# Use specific number of workers
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -n 4
```

## Test Fixtures

### Available Fixtures

- **`gfx942_builder`**: StinkyTofu instance for gfx942 (MI300A/MI300X)
- **`gfx950_builder`**: StinkyTofu instance for gfx950
- **`gfx1250_builder`**: StinkyTofu instance for gfx1250
- **`any_builder`**: Parametrized fixture that tests across all architectures

### Using Fixtures in Tests

```python
def test_my_feature(gfx942_builder):
    """Test using the gfx942 fixture."""
    st = gfx942_builder
    module = st.createIRList("my_test")
    # ... test code ...
```

## Parametrized Tests

The `test_architecture_support` test demonstrates parametrization:

```python
@pytest.mark.parametrize("arch,arch_name", [
    ([9, 4, 2], "gfx942"),
    ([9, 5, 0], "gfx950"),
    ([12, 5, 0], "gfx1250"),
])
def test_architecture_support(arch, arch_name):
    # This test runs 3 times, once for each architecture
    st = StinkyAsmIR(arch)
    # ... test code ...
```

## Test Organization

```
python_module/tests/
+-- conftest.py          # Shared fixtures and pytest configuration
+-- pytest.ini           # pytest configuration file
+-- requirements.txt     # Test dependencies
+-- test_basic.py        # Basic functionality tests
+-- test_mfma.py         # MFMA/WMMA/SMFMA instruction tests
+-- test_gfx1250.py      # gfx1250-specific instruction tests
+-- testing.md           # This file
```

## Writing New Tests

### Basic Test Structure

```python
def test_my_feature(gfx942_builder):
    """Clear description of what this test validates."""
    st = gfx942_builder
    module = st.createIRList("test_name")

    # Create instructions
    module.add(st.VAddU32(vgpr(0), vgpr(1), vgpr(2), "test"))

    # Get assembly
    asm = module.emitAssembly(emit_comments=True)

    # Assertions
    assert "v_add_u32" in asm.lower()
```

### Adding Markers

```python
@pytest.mark.slow
@pytest.mark.composite
def test_complex_feature(gfx942_builder):
    """Test that takes longer to run."""
    # ... test code ...
```

## Debugging Failed Tests

### Run with More Verbose Output
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -vv
```

### Show Local Variables on Failure
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ -l
```

### Drop into Debugger on Failure
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ --pdb
```

### Run Last Failed Tests
```bash
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ --lf
```

## CI/CD Integration

Example for continuous integration:

```bash
#!/bin/bash
set -e

# Build the module
cmake --build build --target stinkytofu_python

# Run tests with coverage
PYTHONPATH=build/lib:$PYTHONPATH pytest python_module/tests/ \
    --cov=stinkytofu \
    --cov-report=xml \
    --cov-report=term \
    --junitxml=test-results.xml

# Check coverage threshold (e.g., 80%)
coverage report --fail-under=80
```

## Debug vs Release Builds

Some tests (specifically architecture error handling tests in `test_gfx1250.py`) are skipped in debug builds because assertions fire before exceptions can be caught. These tests require a release build with assertions disabled (`-DCMAKE_BUILD_TYPE=Release`).

In debug builds, you'll see:
```
SKIPPED [1] ... Error handling tests require release build (assertions disabled)
```

This is expected and correct behavior - debug assertions help catch bugs during development.

## Additional Resources

- pytest documentation: https://docs.pytest.org/
- pytest markers: https://docs.pytest.org/en/stable/how-to/mark.html
- pytest fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html
- pytest parametrize: https://docs.pytest.org/en/stable/how-to/parametrize.html

