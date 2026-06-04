# ROCm Libraries CTest Integration Architecture

This directory contains the shared CTest integration files for organizing and executing tests across ROCm library projects using YAML-based test categorization.

## Directory Structure

```
shared/ctest/
├── README.md                      # This file - architecture documentation
├── TestCategories.cmake           # CMake module for test category integration
└── parse_test_categories.py       # Python parser for YAML to CMake conversion
```

**Files:**
- [TestCategories.cmake](./TestCategories.cmake) - CMake module with `apply_test_category_labels()` function
- [parse_test_categories.py](./parse_test_categories.py) - Python parser for YAML to CMake conversion

## Architecture Overview

The CTest integration provides a flexible, maintainable system for organizing tests into categories with support for platform-specific and GPU-specific test exclusions.

### **Core Components**

#### 1. **test_categories.yaml** (Project-specific)
Located in each project's test directory (e.g., `projects/miopen/test/gtest/test_categories.yaml`).

Defines test organization:
- Test categories with patterns and labels
- Test exclusions
- Timeout settings per category

#### 2. **parse_test_categories.py** (Shared)
Python script that:
- Parses YAML configuration files
- Detects runtime environment (OS, GPU architecture)
- Builds gtest filter strings with positive and negative patterns
- Generates CMake test registration code with proper exclusion syntax

#### 3. **TestCategories.cmake** (Shared)
CMake module providing:
- `apply_test_category_labels()` function for projects
- Python interpreter detection
- Error handling and fallback mechanisms

## Execution Flow

```mermaid
flowchart TD
    A[Project CMakeLists.txt] -->|include| B[TestCategories.cmake]
    A -->|call| C[apply_test_category_labels<br/>target, yaml, workdir]

    B -->|find_package| D[Python3]
    C -->|execute_process| E[parse_test_categories.py]

    E -->|read| F[test_categories.yaml]
    E -->|detect| G[OS & GPU Architecture]
    E -->|build| H[Filter Strings<br/>positive-negative]
    E -->|generate| I[CMake Code]

    I -->|write to| J[Generated CMake<br/>build/test_categories.cmake]

    J -->|add_test| K[Test Registration]
    J -->|set_tests_properties| L[Labels & Timeouts]

    K --> M[CTest Execution]
    L --> M

    M -->|ctest -L category -LE ex_gpu| N[Run without gpu exclusions]
    M -->|ctest -L category -L ex_gpu_gfx1150| O[Run on Specific GPU with exclusions]

    style A fill:#e1f5ff,color:#000
    style E fill:#fff4e1,color:#000
    style J fill:#e8f5e9,color:#000
    style M fill:#f3e5f5,color:#000
```

##  YAML Configuration Format

### **Basic Structure**

```yaml
test_categories:
  category_name:
    description: "Human-readable description"
    test_patterns:
      - "*pattern1*"
      - "*pattern2*"
    exclude:
      - "*pattern_to_exclude*"
    exclude_windows:
      - "*linux_only_tests*"
    exclude_linux:
      - "*windows_only_tests*"
    labels:
      - "quick"
      - "label2"

exclude_gpu:
  # Common pattern definitions using YAML anchors for reusability
  common_patterns: &common_patterns
    - "*pattern1*"
    - "*pattern2*"
    - "*pattern3*"

  exclude_gpu_gfx11X:
    test_patterns: *common_patterns  # Reuse common patterns
    labels:
      - "quick"
      - "standard"
      - "comprehensive"
      - "full"
      - "ex_gpu_gfx11X"

  exclude_gpu_gfx1150:
    test_patterns:
      - "*specific_pattern*"
    labels:
      - "quick"
      - "ex_gpu_gfx1150"

execution_settings:
  default_timeout: 300
  timeout_multiplier: 1    # Multiplier for all timeouts (1, 1.5, 1.75, 2, etc.)
  # environment: { VAR1: "val1", VAR2: "val2" }   # optional; applied to all category tests
  category_timeouts:
    quick: 300
    standard: 1800
```

**Timeout Configuration:**
- `default_timeout`: Base timeout for categories not explicitly listed (in seconds)
- `timeout_multiplier`: Global multiplier applied to all timeouts (default: 1)
  - Use values like `1.5`, `1.75`, `2` to extend timeouts where needed
- `category_timeouts`: Timeouts for specific categories (before multiplier is applied)

**Environment (optional):** Under `execution_settings`, an `environment` map sets env vars for all category tests (e.g. `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`). Keys and values are strings; they are passed to CTest as `ENVIRONMENT "VAR1=val1;VAR2=val2"`.

**Extra arguments (optional, per-category):** A category may set `extra_args` to a list (or single string) of additional command-line arguments that the parser appends to the test command after `--gtest_filter=...`. Useful for projects whose test binary accepts runtime flags beyond gtest filtering (for example, a flag to select a reduced subset of tests, or to scale a sampling/iteration parameter). Each entry is shell-quoted with `shlex.quote`, so values with spaces or shell metacharacters are preserved as a single argument by CTest.

```yaml
test_categories:
  quick:
    test_patterns:
      - "*"
    extra_args:           # appended after --gtest_filter
      - "--quick-subset"
    labels:
      - "quick"
  standard:
    test_patterns:
      - "*"
    exclude:
      - "*excluded_pattern*"
    extra_args:           # multi-token args are passed verbatim
      - "--sample-probability"
      - "0.02"
    labels:
      - "standard"
```

`extra_args` flow through identically to category tests and to GPU-exclusion test variants, in both the build-tree CTest definitions and the install-tree `CTestTestfile.cmake`.

### **Enhanced Structure (Optional Fields)**

All fields below are **optional** and can be added incrementally. Teams can use them for richer test documentation and enable future capabilities like AI-assisted test selection:

```yaml
test_categories:
  category_name:
    # Required fields (same as base)
    description: "Human-readable description"
    test_patterns:
      - "*pattern1*"
      - "*pattern2*"
    labels:
      - "label1"
      - "label2"

    # Optional enhancement fields - add only if useful for your project
    notes: |
      Human-readable context about when to run these tests.
      Can include historical context, gotchas, or guidance for developers and AI tools.
    source_coverage:
      - "library/src/file.cpp"
      - "library/src/module.cpp:function_name"
    api_coverage:
      - "apiFunction1"
      - "apiFunction2"
    feature_tags:
      - "performance-critical"
      - "numerical-stability"
    dependencies:
      - "other_category"  # Metadata for related categories (not enforced by parser; for documentation/tooling)

    # Standard fields (from base)
    exclude:
      - "*always_exclude*"
    exclude_windows:
      - "*linux_only*"
    exclude_linux:
      - "*windows_only*"

# Optional: Top-level context for AI/LLM tools
llm_context:
  code_to_test_mapping_guidelines: |
    Guidance for AI tools on how to map code changes to test categories.
    Projects can use this for AI-assisted test selection.

execution_settings:
  default_timeout: 300
  category_timeouts:
    category_name: 600
```

**Optional Field Descriptions:**

| Field | Purpose | Example Use |
|-------|---------|-------------|
| `notes` | Free-form text for context and documentation | "Run when epilogue changes. See bug #8765 for history" |
| `source_coverage` | Source files/functions tested by this category | `["library/src/gemm.cpp:matmul_kernel"]` |
| `api_coverage` | API functions tested by this category | `["hipblasLtMatmul", "hipblasLtMatmulAlgo"]` |
| `feature_tags` | Semantic tags for classification and filtering | `["performance-critical", "mixed-precision"]` |
| `dependencies` | Related test categories (documentation only) | `["auxiliary"]` - advisory metadata for downstream tooling |
| `llm_context` | Top-level guidance for AI-assisted workflows | Instructions for AI tools on test selection logic |

**Key Points:**
- All enhancement fields are **optional** - teams can ignore them entirely ✅
- Projects can adopt incrementally: start with just `notes`, add more later ✅
- **Parser does NOT process these fields** - they are for documentation and downstream tooling only ✅
- Parser gracefully ignores unknown fields - no code changes needed ✅
- Enables richer test documentation and future AI-assisted workflows ✅

### **GPU Exclusion with Hierarchical Matching**

GPU-specific exclusions use hierarchical pattern matching with wildcard 'X':

**Structure:**
- Each `exclude_gpu_gfx*` entry defines patterns to exclude for specific GPU architectures
- Patterns can be shared using YAML anchors (`&name`) and aliases (`*name`)
- Labels include both category labels and `ex_gpu_*` labels for filtering

**Hierarchical Matching:**
- Wildcard 'X' matches any remaining characters (e.g., `gfx11X` matches `gfx1100`, `gfx1150`, `gfx1151`)
- More specific GPUs inherit exclusions from general patterns:
  - `gfx1150` matches both `exclude_gpu_gfx11X` and `exclude_gpu_gfx1150`
  - `gfx1151` matches `exclude_gpu_gfx11X` (inherits from family pattern)

**Generated Tests:**
- For each GPU exclusion, separate tests are generated per applicable category
- Test name format: `{target}-{category}-{gpu_arch}-exclude`
- Uses gtest filter syntax: `{positive_patterns}-{category_excludes}:{gpu_exclusion_patterns}`

**Usage Examples:**
```bash
# On gfx1150 hardware (excludes gfx11X + gfx1150 patterns)
ctest -L quick -L ex_gpu_gfx1150

# On gfx950 hardware (excludes only gfx950 patterns)
ctest -L standard -L ex_gpu_gfx950

# On generic hardware (exclude all GPU-specific tests)
ctest -L quick -LE ex_gpu
```

### **Category-Level Exclusions**

Exclusions are applied using gtest's negative filter syntax (`positive_patterns-negative_patterns`):

**How it works:**
- All test patterns remain in the category definition
- Excluded patterns are added as negative filters after a single `-` separator
- Format: `pattern1:pattern2:pattern3-excluded1:excluded2:excluded3`
- Gtest runs tests matching positive patterns but excludes those matching negative patterns

**Exclusion order:**

1. **Base exclusions** (`exclude`) - Applied to that category
2. **OS-specific exclusions** (`exclude_windows`, `exclude_linux`) - Added based on detected OS at build time
3. **GPU exclusions** (`exclude_gpu`) - Appended for GPU-specific test variants

**Example filters:**
- **Category test**: `*Fusion*:*Conv*-*DeepBench*:*Slow*`
- **GPU exclusion test**: `*Fusion*:*Conv*-*DeepBench*:*Slow*:*gfx942*`

This approach maintains all patterns in the YAML configuration while letting gtest handle the filtering at runtime.

### **Implementation Details**

The `parse_test_categories.py` script builds filter strings using the following approach:

**1. Pattern String Storage**

For each category, the script stores:
```python
# Read timeout settings
base_timeout = timeouts.get(category_name, 300)
timeout = int(base_timeout * timeout_multiplier)  # Apply global multiplier

category_data[category_name] = {
    "positive_string": "pattern1:pattern2:pattern3",  # All test patterns
    "exclude_string": "neg1:neg2:neg3",               # Category + OS exclusions
    "labels": ["quick", "standard"],
    "timeout": timeout  # Final timeout after multiplier applied
}
```

**2. Category Test Generation**
```cmake
# Filter format: positive_string-exclude_string
add_test(
  NAME miopen_gtest_standard_suite
  COMMAND miopen_gtest --gtest_filter="*Fusion*:*Conv*-*DeepBench*:*Slow*"
)
```

**3. GPU Exclusion Test Generation**
```cmake
# Filter format: positive_string-exclude_string:gpu_exclude_string
add_test(
  NAME miopen_gtest_standard_gfx1150_suite
  COMMAND miopen_gtest --gtest_filter="*Fusion*:*Conv*-*DeepBench*:*Slow*:*gfx942*"
)
```

## Integration Guide

This guide walks through integrating the shared YAML-based test categorization into a GTest-based ROCm library project, using MIOpen as the reference implementation.

## Prerequisites

- A monolithic GTest binary (single executable containing all tests)
- Python 3 which is available in the build environment
- The `PyYAML` Python package installed
- `ROCM_LIBRARIES_ROOT` set to the repository root in your CMake

## Step 1: Create test_categories.yaml

Create a `test_categories.yaml` file in your project's test directory.

### Minimal Example

```yaml
test_categories:
  quick:
    description: "Fast sanity checks (target: < 5 min)"
    test_patterns:
      - "Smoke/*"
      - "Unit/*"
    labels:
      - "quick"
      - "pre-commit"

  standard:
    description: "Core functionality (target: < 30 min)"
    test_patterns:
      - "Smoke/*"
      - "Unit/*"
      - "*/GPU_Conv*"
      - "*/GPU_BN*"
    exclude:
      - "*SlowTest*"
    labels:
      - "standard"
      - "pr"

execution_settings:
  timeout_multiplier: 1
  category_timeouts:
    quick: 300
    standard: 1800
```

### Using YAML Anchors to Share Patterns

When multiple categories share the same base patterns, use YAML anchors (`&name`) and aliases (`*name`) to avoid duplication:

```yaml
# Define once at the top level
all_patterns: &all_patterns
  - "*/GPU_Conv*"
  - "*/GPU_BN*"
  - "*/GPU_Reduce*"

test_categories:
  standard:
    test_patterns: *all_patterns
    exclude:
      - "*SlowTest*"
    labels:
      - "standard"
      - "pr"

  comprehensive:
    test_patterns: *all_patterns
    labels:
      - "comprehensive"
      - "nightly"
```

### Adding OS-Specific Exclusions

Exclude tests that only work on a specific OS:

```yaml
test_categories:
  standard:
    test_patterns:
      - "*TestSuite*"
    exclude:
      - "*KnownBroken*"
    exclude_windows:
      - "*LinuxOnlyFeature*"
    exclude_linux:
      - "*WindowsOnlyFeature*"
    labels:
      - "standard"
```

### Adding GPU-Specific Exclusions

Define GPU exclusions with hierarchical matching using a naming convention for `exclude_gpu` keys. In this convention, an `X` suffix in the architecture portion of the key name (for example, `gfx11X`) stands for “all gfx11xx variants”; this is only a key-naming shorthand, not a glob or regex wildcard in `test_patterns`.

```yaml
# Shared patterns using YAML anchors
gpu_common_patterns: &gpu_common_patterns
  - "*KnownGPUFailure*"
  - "*UnsupportedOnGfx11*"

exclude_gpu:
  exclude_gpu_gfx11X:
    # Applies to all gfx11xx GPUs (gfx1100, gfx1150, gfx1151, etc.)
    test_patterns: *gpu_common_patterns
    labels:
      - "quick"        # Must list every category this exclusion applies to
      - "standard"
      - "comprehensive"
      - "full"
      - "ex_gpu_gfx11X"  # Required: ex_gpu_<arch> label for CTest filtering

  exclude_gpu_gfx1150:
    # Additional exclusions specific to gfx1150 (on top of gfx11X)
    test_patterns:
      - "*Gfx1150SpecificFailure*"
    labels:
      - "quick"
      - "ex_gpu_gfx1150"

  exclude_gpu_gfx950:
    test_patterns:
      - "*Gfx950Issue*"
    labels:
      - "quick"
      - "standard"
      - "ex_gpu_gfx950"
```

**How hierarchical matching works for the above snippet:**


| GPU on machine | Exclusion configs applied                    |
| -------------- | -------------------------------------------- |
| gfx1100        | `exclude_gpu_gfx11X`                         |
| gfx1150        | `exclude_gpu_gfx11X` + `exclude_gpu_gfx1150` |
| gfx1151        | `exclude_gpu_gfx11X`                         |
| gfx950         | `exclude_gpu_gfx950`                         |


**Important:** The `labels` list in each `exclude_gpu` entry must include every test category that the exclusion applies to. The parser matches these labels against `category_data` keys to determine which categories receive the GPU-specific test variants.

## Step 2: Integrate in CMakeLists.txt

### Enable CTest

Ensure `enable_testing()` is called somewhere in your CMake hierarchy before `apply_test_category_labels()`. Without it, `add_test()` calls have no effect and CTest won't discover any tests. This is typically done in the project's test-level `CMakeLists.txt`:

```cmake
if(BUILD_TESTING)
    enable_testing()
endif()
```

If your project already calls `enable_testing()` (or uses `include(CTest)`, which calls it internally), no additional change is needed.

### Integration with Install Support

For projects that distribute test binaries (for example, via TheRock), generate an install-time CTestTestfile.cmake, so that it can be installed wherever the final test binaries are present.

```cmake
set(ROCM_LIBRARIES_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
include(${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake)

# Create install-time test file
set(INSTALL_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
file(WRITE "${INSTALL_TEST_FILE}"
[=[
# Generated test definitions for install-time execution.
# Tests use relative paths to the test binary.
]=]
)

# Apply categorization (4th argument enables install file generation)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
    apply_test_category_labels(
        my_project_gtest
        "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
        "${PROJECT_BINARY_DIR}"
        "${INSTALL_TEST_FILE}"
    )
endif()

# Install the generated CTestTestfile.cmake alongside the test binary
install(
    FILES "${INSTALL_TEST_FILE}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}"
    COMPONENT tests
    RENAME "CTestTestfile.cmake"
)
```

### Excluding Category Suites from `make check`

All YAML-generated tests are named with a `_suite` suffix. If your project has a `check` target that runs `ctest`, the newly generated tests can be excluded to avoid extended run times:

```cmake
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --verbose -C ${CMAKE_CFG_INTDIR}
    --exclude-regex ".*_suite"
)
```

## Step 3: Build and Run Tests

```bash
# Build with existing build commands for the project
cmake -DBUILD_TESTING=ON ..
cmake --build .

# List all registered tests
ctest -N

# Run a specific category (no GPU exclusions)
ctest -L quick -LE ex_gpu

# Run a specific category with GPU-specific exclusions
ctest -L quick -L ex_gpu_gfx1150

# Run with verbose output
ctest -L standard -L ex_gpu_gfx950 -V
```

## Generated Test Names

The framework generates tests with predictable naming:


| Type           | Name Format                            | Example                                |
| -------------- | -------------------------------------- | -------------------------------------- |
| Category suite | `{target}_{category}_suite`            | `my_project_gtest_quick_suite`         |
| GPU exclusion  | `{target}_{category}_{gpu_arch}_suite` | `my_project_gtest_quick_gfx1150_suite` |


## Generated Labels

Each test gets labels that enable flexible CTest filtering:


| Test Type      | Labels                                                              |
| -------------- | ------------------------------------------------------------------- |
| Category suite | Category labels from YAML (for example, `quick`, `standard`)               |
| GPU exclusion  | Category labels + `ex_gpu_<arch>` (for example, `quick`, `ex_gpu_gfx1150`) |


## Function Reference

### `apply_test_category_labels(target_name yaml_file working_dir [install_test_file])`


| Parameter           | Required | Description                                                                                         |
| ------------------- | -------- | --------------------------------------------------------------------------------------------------- |
| `target_name`       | Yes      | Name of the GTest executable (must not be empty)                                                   |
| `yaml_file`         | Yes      | Absolute path to `test_categories.yaml` (must exist)                                               |
| `working_dir`       | Yes      | Working directory for test execution (must be a valid directory)                                   |
| `install_test_file` | No       | Path to the generated install-time `CTestTestfile.cmake`. Not required for local builds, but      |
|                     |          | required for install/package workflows (for example, TheRock).                                            |
**Validation:** The function validates all inputs before execution and emits `WARNING` messages if any check fails, returning early without generating tests.

## YAML Reference

### `test_categories` (required)

```yaml
test_categories:
  <category_name>:
    description: "..."                 # Optional: human-readable description
    test_patterns:               # Required: gtest filter patterns (positive match)
      - "..."
    exclude:                     # Optional: patterns to exclude from this category
      - "..."
    exclude_windows:             # Optional: additional exclusions on Windows
      - "..."
    exclude_linux:               # Optional: additional exclusions on Linux
      - "..."
    labels:                      # Required: CTest labels for filtering
      - "..."
```

### `exclude_gpu` (optional)

```yaml
exclude_gpu:
  exclude_gpu_<arch>:
    test_patterns:               # Required: patterns to exclude on this GPU
      - "..."
    labels:                            # Required: must include applicable category names
      - "<category_name>"             #   and an ex_gpu_<arch> label
      - "ex_gpu_<arch>"
```

### `execution_settings` (optional)

```yaml
execution_settings:
  timeout_multiplier: 1               # Global multiplier for all timeouts (default: 1)
  category_timeouts:
    <category_name>: <seconds>         # Per-category timeout (default: 300)
```

## Checklist

- Created `test_categories.yaml` in your project's test directory
- At least one category has `test_patterns` and `labels` defined
- `ROCM_LIBRARIES_ROOT` is set correctly (points to the monorepo root)
- `include(${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake)` added before calling the function
- `apply_test_category_labels()` called with the correct test executable name
- GPU exclusion `labels` lists include every applicable category name
- Each `exclude_gpu_<arch>` entry has an `ex_gpu_<arch>` label
- `ctest -N` shows the expected test suites after building

## Reference Implementation

See the MIOpen integration for a complete working example:

- **YAML config:** [projects/miopen/test/gtest/test_categories.yaml](../../projects/miopen/test/gtest/test_categories.yaml)
- **CMake integration:** [projects/miopen/test/gtest/CMakeLists.txt](../../projects/miopen/test/gtest/CMakeLists.txt)
- **Shared module:** [shared/ctest/TestCategories.cmake](./TestCategories.cmake)
- **Parser script:** [shared/ctest/parse_test_categories.py](./parse_test_categories.py)

### **Install-time CTestTestfile (TheRock / install tree)**

When tests are installed (e.g. into `/opt/rocm/bin/`), the **build-tree** test definitions are not installed. To run CTest from the **installed** location (e.g. on TheRock or any system that only has the install tree), projects can generate an **install-time CTestTestfile** that uses **relative paths** to the test executable.

**How it works:**

1. **Project enables it** by passing an optional **4th argument** to `apply_test_category_labels()`: a path to a file (e.g. `install_CTestTestfile.cmake`) that the parser will create or append to. The parser writes `add_test(...)` and `set_tests_properties(...)` lines into that file using a **relative** command (e.g. `"../rocblas-test"`), so the test binary is found relative to the directory where the file will live after install.

2. **Project installs that file** to a fixed location under the install prefix, typically a **project-specific subdirectory** of the bin dir (e.g. `bin/rocblas/` or `bin/MIOpen/`) and renames it to `CTestTestfile.cmake`. The test executable is installed in the parent `bin/` directory, so from `bin/rocblas/` the path `../rocblas-test` correctly points at the installed binary.

3. **TheRock (or any consumer)** runs CTest **from that installed directory** (e.g. `cd /opt/rocm/bin/rocblas && ctest -L quick`). CTest reads the local `CTestTestfile.cmake`, runs the tests with the same labels and timeouts as in the build tree, and no build tree is required.

**Example (rocBLAS):**

- Build: parser writes tests into `install_CTestTestfile.cmake`; CMake installs it as `CTestTestfile.cmake` to `bin/rocblas/`.
- CMake: `ROCBLAS_ENABLE_CTEST` (in `projects/rocblas/cmake/build-options.cmake`) defaults to **ON** when `${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake` exists; if **ON**, a missing YAML or shared module is a **configure error** (set to **OFF** to skip categorization).
- Install layout: `bin/rocblas-test` (executable) and `bin/rocblas/CTestTestfile.cmake` (test definitions).
- Run from install: `cd /opt/rocm/bin/rocblas && ctest -L quick -N` (list) or `ctest -L quick` (run).

Projects that use this pattern (e.g. MIOpen, rocBLAS) document it in their Integration entries below; the same approach applies to any project that needs install-tree CTest runs.

## Integrations

- **miopen** - [test_categories.yaml](../../projects/miopen/test/gtest/test_categories.yaml) | [CMakeLists.txt](../../projects/miopen/test/gtest/CMakeLists.txt)
- **rocblas** - [test_categories.yaml](../../projects/rocblas/clients/gtest/test_categories.yaml) | [CMakeLists.txt](../../projects/rocblas/clients/gtest/CMakeLists.txt)
