# Test Runner Base Class

This directory contains the base class for component test runners in rocm-libraries.

## Overview

The `BaseTestRunner` class abstracts common patterns found across all component test runners, reducing code duplication and making it easier to maintain consistent behavior.

## Architecture

```
rocm-libraries/
├── scripts/test_runners/
│   ├── base_test_runner.py    # Shared base class
│   └── README.md               # This file
└── projects/
    ├── rocrand/
    │   └── test_runner.py      # Inherits from BaseTestRunner
    ├── rocsolver/
    │   └── test_runner.py      # Inherits from BaseTestRunner
    └── ...
```

## Common Patterns Abstracted

The base class handles:

- **Environment variable handling**: `THEROCK_BIN_DIR`, `TEST_TYPE`, `SHARD_INDEX`, `TOTAL_SHARDS`, `AMDGPU_FAMILIES`, `RUNNER_OS`, `OUTPUT_ARTIFACTS_DIR`
- **GTest sharding setup**: Converts 1-indexed GitHub Actions shards to 0-indexed GTest shards
- **ROCM_PATH setup**: Sets `ROCM_PATH` for runtime kernel compilation
- **Logging**: Consistent logging format across all runners
- **Command execution**: Standard subprocess execution with proper error handling
- **Test directory resolution**: Default logic with override capability

## Usage

### Basic Pattern

Each component creates a `test_runner.py` in its project directory:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# Import base class
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "test_runners"))
from base_test_runner import BaseTestRunner


class MyComponentTestRunner(BaseTestRunner):
    def __init__(self):
        super().__init__(component_name="mycomponent")
        # Component-specific configuration
        self.parallel_jobs = 8

    def build_command(self):
        # Must implement: return list of command args
        return ["ctest", "--test-dir", self.get_test_directory()]


if __name__ == "__main__":
    runner = MyComponentTestRunner()
    sys.exit(runner.run())
```

### Customization Points

#### 1. Quick Test Filters

Override `get_quick_test_filters()` to define quick test patterns:

```python
def get_quick_test_filters(self):
    return ["*quick*", "*smoke*", "-*slow*"]
```

#### 2. Custom Environment Setup

Override `setup_environment()` to add component-specific env vars:

```python
def setup_environment(self):
    super().setup_environment()  # Call base implementation
    self.environ_vars["MY_CUSTOM_VAR"] = "value"
    if self.amdgpu_families == "gfx1151":
        self.environ_vars["SPECIAL_FLAG"] = "1"
```

#### 3. Custom Test Directory

Override `get_test_directory()` for non-standard layouts:

```python
def get_test_directory(self):
    return f"{self.therock_bin_dir}/custom/path/{self.component_name}"
```

#### 4. Additional Validation

Override `validate_environment()` to check component-specific requirements:

```python
def validate_environment(self):
    super().validate_environment()  # Call base validation
    if not os.getenv("MY_REQUIRED_VAR"):
        self.logger.error("MY_REQUIRED_VAR is required")
        sys.exit(1)
```

## Test Runner Patterns

### Pattern 1: CTest + GTEST_FILTER

Used by: rocRAND, rocPRIM

```python
def build_command(self):
    cmd = ["ctest", "--test-dir", self.get_test_directory(), ...]
    
    if self.is_quick_test():
        self.environ_vars["GTEST_FILTER"] = ":".join(self.get_quick_test_filters())
    
    return cmd
```

### Pattern 2: Direct GTest Binary

Used by: rocSOLVER, rocBLAS

```python
def build_command(self):
    cmd = [f"{self.therock_bin_dir}/mycomponent-test"]
    
    if self.is_quick_test():
        filter_str = ":".join(self.get_quick_test_filters())
        cmd.append(f"--gtest_filter={filter_str}")
    
    return cmd
```

### Pattern 3: CTest with Exclude Regex

Used by: rocWMMA

```python
def build_command(self):
    cmd = ["ctest", "--test-dir", self.get_test_directory()]
    
    # Exclude specific tests
    if self.tests_to_ignore:
        cmd.extend(["--exclude-regex", "|".join(self.tests_to_ignore)])
    
    return cmd
```

### Pattern 4: Custom Binary with Arguments

Used by: rocFFT, rocBLAS

```python
def build_command(self):
    cmd = [f"{self.therock_bin_dir}/mycomponent-test"]
    
    if self.is_quick_test():
        cmd.extend(["--smoketest"])
    else:
        cmd.extend(["--gtest_filter=-*multi_gpu*", "--test_prob", "0.02"])
    
    return cmd
```

## Available Properties

All test runners have access to:

- `self.component_name` - Component name
- `self.therock_bin_dir` - THEROCK_BIN_DIR env var
- `self.therock_dir` - Root of TheRock repository
- `self.test_type` - TEST_TYPE env var (default: "full")
- `self.amdgpu_families` - AMDGPU_FAMILIES env var
- `self.runner_os` - RUNNER_OS env var (lowercased)
- `self.shard_index` - SHARD_INDEX env var (1-indexed)
- `self.total_shards` - TOTAL_SHARDS env var
- `self.output_artifacts_dir` - OUTPUT_ARTIFACTS_DIR env var
- `self.environ_vars` - Copy of os.environ for subprocess
- `self.logger` - Logger instance

## Available Methods

- `is_quick_test()` - Returns True if TEST_TYPE == "quick"
- `setup_gtest_sharding()` - Sets GTEST_SHARD_INDEX and GTEST_TOTAL_SHARDS
- `setup_rocm_path()` - Sets ROCM_PATH from THEROCK_BIN_DIR parent
- `setup_environment()` - Called before running tests
- `get_test_directory()` - Returns test directory path
- `validate_environment()` - Validates required env vars
- `get_quick_test_filters()` - Returns quick test patterns (override this)
- `build_command()` - Builds test command (must override)
- `run()` - Main entry point that executes the test

## Benefits

### Code Reduction

- **Before**: ~100-130 lines per test runner
- **After**: ~40-70 lines per test runner (40-60% reduction)

### Consistency

- All runners use the same environment variable handling
- Consistent logging format
- Standard error handling
- Uniform sharding implementation

### Maintainability

- Bug fixes in one place benefit all components
- Easy to add new features (e.g., timeout handling)
- Clear contract via abstract base class
- Type hints and documentation

## Migration Guide

To migrate an existing test runner:

1. Create `projects/<component>/test_runner.py`
2. Import `BaseTestRunner` from `scripts/test_runners/base_test_runner.py`
3. Create a class inheriting from `BaseTestRunner`
4. Move component-specific configuration to `__init__()`
5. Move test filter lists to `get_quick_test_filters()`
6. Move command construction to `build_command()`
7. Move any custom env setup to `setup_environment()`
8. Remove boilerplate: env var reading, logging setup, sharding, etc.

## Examples

See:
- `projects/rocrand/test_runner.py` - CTest + GTEST_FILTER pattern
- `projects/rocsolver/test_runner.py` - Direct GTest binary pattern
