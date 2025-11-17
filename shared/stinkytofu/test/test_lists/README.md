# Test Lists

This directory contains test lists for the stinkytofu test infrastructure.

## File Naming Convention

### Consolidated Format (Recommended)

The preferred format uses a single file per test type with GPU architecture sections:
```
{test_type}.txt
```

Where `test_type` is: `exe-time`, `codegen-time`, or `dbg-verify`

Examples:
- `exe-time.txt` - Execution time tests for all GPU architectures
- `codegen-time.txt` - Code generation time tests for all architectures
- `dbg-verify.txt` - Debug verification tests for all architectures

### Legacy Format (Still Supported)

For backward compatibility, GPU-specific files are still supported:
```
{gpu_arch}_{test_type}.txt
```

Examples:
- `gfx942_exe-time.txt` - Execution time tests for MI300 series only
- `gfx950_codegen-time.txt` - Code generation time tests for MI350 series only

**Note:** If a consolidated file exists (e.g., `exe-time.txt`), it takes precedence over legacy files.

## File Format

### Consolidated Format

Use GPU architecture sections to specify which tests apply to which GPUs:

```
# Comments start with #

# Section for multiple GPUs - tests apply to both gfx942 and gfx950
[gfx942][gfx950]
mfma.yaml
common_test.yaml

# Use glob patterns to match multiple files
sia3/*              # All YAML files in sia3/ subdirectory
benchmarks/*.yaml   # Specific pattern match

# Section for single GPU - tests only for gfx942
[gfx942]
mi300_specific.yaml

# Section for another GPU
[gfx950]
mi350_specific.yaml

# Absolute paths work too
[gfx942][gfx950]
/home/user/shared/stinkytofu/test/yaml/exe-time/shared_test.yaml
```

**Format Rules:**
- GPU sections start with `[gpu_arch]` (e.g., `[gfx942]`)
- Multiple GPUs can be specified on one line: `[gfx942][gfx950][gfx90a]`
- All tests following a section header apply to those GPUs
- Tests must be under a section header (tests before any section are ignored)
- Use relative paths (to `yaml/{test_type}/`) or absolute paths
- **Glob patterns supported**: Use `*`, `?`, `[]` to match multiple files (e.g., `sia3/*` matches all YAML files in sia3/)
- Comments start with `#`
- Empty lines are ignored
- Command-line regex patterns are supported for filtering

### Legacy Format

For GPU-specific files (backward compatible):

```
# This is a comment

# Relative paths (relative to yaml/exe-time/ or yaml/codegen-time/)
mfma.yaml
gemm_test.yaml

# Absolute paths (useful for shared test cases)
/home/user/shared/stinkytofu/test/yaml/exe-time/mfma.yaml

# Relative paths with subdirectories
sia3/mfma.yaml
benchmarks/large_gemm.yaml

# Glob patterns to match multiple files
sia3/*
benchmarks/*.yaml

# Command-line regex patterns (for --pattern flag)
.*_small\.yaml
test_.*\.yaml
```

## Usage

When running tests, the test infrastructure will:
1. Check for consolidated test list (e.g., `exe-time.txt`) first
2. If consolidated file exists, parse GPU sections and use tests for the detected GPU
3. Otherwise, check for legacy GPU-specific file (e.g., `gfx942_exe-time.txt`)
4. If no test list is found, run all YAML files in the corresponding `yaml/{test_type}/` directory
5. Apply any additional pattern filtering specified on the command line

**Priority order:**
1. Consolidated file (`{test_type}.txt`) - **Recommended**
2. Legacy GPU-specific file (`{gpu_arch}_{test_type}.txt`) - For backward compatibility
3. All YAML files in directory - Fallback when no list exists

**Note:** Debug verification tests (`dbg-verify`) use the same YAML files as execution time tests (`exe-time`), but can have their own test lists. This allows you to run a different subset of tests in debug mode if needed.

## GPU Architecture Detection

The GPU architecture is detected automatically using:
```bash
/opt/rocm/bin/rocm_agent_enumerator | head -n2 | tail -n1
```

Common GPU architectures:
- `gfx942`: AMD Instinct MI300 series
- `gfx950`: AMD Instinct MI350 series (future)
- `gfx90a`: AMD Instinct MI250/MI250X
- `gfx908`: AMD Instinct MI100

