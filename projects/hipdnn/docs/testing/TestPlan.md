# hipDNN Test Plan

This document outlines the test plan for hipDNN, covering test execution procedures and expectations. If needed, reference the [Quick Start Guide](../Building.md#quick-start-guide) to prepare a local environment.

> [!IMPORTANT]
> ⚠️ **All prerequisites and tests in this document must pass for a successful release.**

---

## Prerequisites

### Test Case 1: CI Is Green 🟩

Existing checks should be running automatically on all PRs pre-merge and on `develop` branch post-merge.

| CI Stage | Description |
|----------|-------------|
| `static-analysis` | Runs linting and static analysis tools to detect code issues early |
| `precheckin` | Runs unit & integration tests |
| `codecov` | Checks code coverage requirements |
| `debug` | Runs pre-checkin checks in a debug build |

### Test Case 2: Documentation is Current 🕒

Verify that all documentation is up to date:

1. Check version numbers throughout the documentation
2. Review instructions, explanations, and wording for clarity and accuracy
4. Verify changelog is complete and correct

> See the documentation listed in the [README](../../README.md#documentation) to identify relevant areas.

---

## Regular Tests

### Test Case 1: Run the Automated Tests ⚙️

Run as follows:
```bash
cmake ..
ninja check_ctest
```   

#### Expected Results

- **Test Status**: All tests should pass
- **GPU Test Behavior**:
  - **Without GPU**: All GPU tests should skip gracefully without failures
  - **With GPU**: Plugin integration tests may skip if the GPU is not supported
    - Skipped tests should provide clear messages indicating lack of ASIC support
- **Plugin Support**: ASIC-specific coverage is determined by individual plugins and is not a global hipDNN requirement

---

## ASAN Enabled Tests

### Test Case 1: Run the Automated Tests with ASAN Enabled 🚨

Run as follows:
```bash
cmake .. -DBUILD_ADDRESS_SANITIZER=ON
ninja check_ctest
```

#### Expected Results

- **Test Status**: All tests should pass
- **GPU Test Behavior**: All GPU tests will be skipped due to ASAN being enabled
- **Memory Safety**: No memory leaks or violations should be detected
