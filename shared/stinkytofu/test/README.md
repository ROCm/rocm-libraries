# stinkytofu Test Infrastructure

This directory contains a comprehensive test infrastructure for testing stinkytofu across different GPU models and tracking performance metrics over time.

## Overview

The test infrastructure supports three types of tests:

1. **Test A (exe-time)**: Measures GPU execution time and verifies correctness
2. **Test B (codegen-time)**: Measures tensilelite _getKernelSource code generation time using yappi profiling
3. **Test C (dbg-verify)**: Verifies correctness using debug build

### Key Features

✅ **Unified YAML System**: Single YAML source for all test types, variants created automatically
✅ **Section-Based Test Lists**: Organize tests by architecture with modifier support
✅ **Automatic Variant Generation**: Test default and +sia3 (or other modifiers) from one YAML
✅ **Git-Based Tracking**: Performance database keyed by git commit hash
✅ **Local + Global Results**: Detailed logs per-session, curated database for history
✅ **Comparison Tools**: Compare sessions, export to CSV, track trends

## Directory Structure

```
test/
├── scripts/              # Test execution and analysis scripts
│   ├── main.sh          # Main orchestration script
│   ├── utils.sh         # Utility functions
│   ├── test_a_exe_time.sh       # Test A execution
│   ├── test_b_codegen_time.sh   # Test B execution
│   ├── test_c_dbg_verify.sh     # Test C execution
│   ├── patch_yaml.py            # YAML patching for variants
│   ├── parse_exe_time.py        # Parse Test A results
│   ├── parse_codegen_time.py    # Parse Test B results
│   ├── update_global_database.py # Update performance database
│   ├── compare_logs.py          # Compare two test sessions
│   └── export_database_csv.py   # Export database to CSV
├── yaml/                # Unified test case definitions
│   └── *.yaml           # Single source for all test types
├── test_lists/          # Section-based test lists
│   ├── exe-time.txt     # Execution time tests
│   ├── codegen-time.txt # Code generation tests
│   └── dbg-verify.txt   # Debug verification tests
└── README.md            # This file
```

## Prerequisites

1. Docker container with ROCm and hipblaslt build environment
2. Python 3 with PyYAML (for YAML patching and result parsing)
3. yappi Python package (automatically installed if needed for codegen-time tests)

**Install PyYAML if not present:**
```bash
pip3 install pyyaml
# or inside docker container
docker exec your_container pip3 install pyyaml
```

## Building the Test Binaries

The test infrastructure automatically builds the necessary binaries, but you can also build them manually:

### For Test A and B (Release build):

```bash
docker exec -w /workspace -it ${container_name} bash

cd ${hipblaslt_path}
cmake --preset tensilelite -S . -B tensilelite/rel_build \
      -DTENSILELITE_BUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build tensilelite/rel_build --parallel
```

### For Test C (Debug build):

```bash
docker exec -w /workspace -it ${container_name} bash

cd ${hipblaslt_path}
cmake --preset tensilelite -S . -B tensilelite/dbg_build \
      -DTENSILELITE_BUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build tensilelite/dbg_build --parallel
```

## Running Tests

### Basic Usage

```bash
cd /path/to/your/workspace
/path/to/shared/stinkytofu/test/scripts/main.sh \
    --container-name my_container \
    --docker-path /workspace/hipblaslt \
    --host-path /home/user/rocm-libraries_p/projects/hipblaslt \
    --output-dir /tmp/test_results \
    --test all
```

### Command Line Options

- `--container-name NAME`: Docker container name (required)
- `--docker-path PATH`: Path to hipblaslt inside container (required)
- `--host-path PATH`: Path to hipblaslt on host system (required)
- `--output-dir DIR`: Output directory for logs and results (required)
- `--test TYPE`: Test type to run: `exe_time`, `codegen_time`, `dbg_verify`, or `all` (default: `all`)
- `--gpu-arch ARCH`: GPU architecture (auto-detected if not specified)
- `--pattern REGEX`: Regex pattern to filter test cases (default: `.*`)

### Examples

Run only execution time tests:
```bash
./scripts/main.sh --container-name my_container \
    --docker-path /workspace/hipblaslt \
    --host-path ~/hipblaslt \
    --output-dir /tmp/results \
    --test exe_time
```

Run tests matching a specific pattern:
```bash
./scripts/main.sh --container-name my_container \
    --docker-path /workspace/hipblaslt \
    --host-path ~/hipblaslt \
    --output-dir /tmp/results \
    --test exe_time \
    --pattern "mfma.*"
```

Run all tests for a specific GPU:
```bash
./scripts/main.sh --container-name my_container \
    --docker-path /workspace/hipblaslt \
    --host-path ~/hipblaslt \
    --output-dir /tmp/results \
    --gpu-arch gfx942
```

## Test Results

Results are organized into **database** (persistent) and **logs** (per-session) directories:

```
${output_dir}/${gpu_arch}/
├── database/                   # Persistent performance tracking
│   ├── exe_time-${hostname}_${gpu_arch}.json
│   └── codegen_time-${hostname}_${gpu_arch}.json
└── logs/
    └── ${date}-${hostname}-${git_hash}/  # Per-session logs
        ├── exe_time/
        │   ├── ${test}.yaml         # Patched YAML used for test
        │   ├── ${test}.log          # Full execution log
        │   ├── ${test}/             # Assembly files
        │   ├── local.json           # Summary: {testcase: [times]}
        │   ├── local-sia3cmp.json   # Detailed: full test data
        │   └── local_report.txt     # Human-readable summary
        ├── codegen_time/
        │   ├── ${test}_yappi_results.txt
        │   ├── local.json
        │   └── local_report.txt
        └── dbg_verify/
            ├── ${test}.log
            └── summary.txt
```

### Database Files (Persistent)

**Global Database** (`database/exe_time-${hostname}_${gpu_arch}.json`):
```json
{
  "abc123def": {           // Git commit hash
    "mfma.yaml": 309.648,  // Best time (μs)
    "mfma-sia3.yaml": 315.122
  },
  "def456abc": {
    "mfma.yaml": 308.123
  }
}
```

Keyed by git commit hash, stores only the best scores across all runs.

### Local Results (Per-Session)

**Summary** (`logs/.../exe_time/local.json`):
```json
{
  "mfma.yaml": [309.648, 309.412, 310.000],
  "mfma-sia3.yaml": [315.122]
}
```

**Detailed** (`logs/.../exe_time/local-sia3cmp.json`):
Contains full test metadata, problem configurations, and all individual results.

**Report** (`logs/.../exe_time/local_report.txt`):
Human-readable table showing all runs and best times.

### Workflow

1. **During Test Run**: Each test case saves results to `local-sia3cmp.json` (detailed) and `local.json` (summary)
2. **After All Tests**: Global database is updated with best scores from `local.json`, keyed by git hash
3. **Result**: Database tracks performance over time, logs preserve all run details

This separation allows:
- Local debugging with full details
- Clean database with only best scores
- Historical comparison across git commits

## Comparing Results Across Versions

The infrastructure provides multiple ways to analyze and compare results:

### 1. View Local Reports

Quick summary of a single test session:
```bash
# View test run summary
cat test_results/gfx950/logs/20251203-hostname-abc123/exe_time/local_report.txt

# View detailed results
cat test_results/gfx950/logs/20251203-hostname-abc123/exe_time/local-sia3cmp_report.txt
```

### 2. Compare Two Test Sessions

Use `compare_logs.py` to compare performance between different runs:
```bash
python3 scripts/compare_logs.py \
    --base-folder test_results/gfx950/logs/20251203-hostname-abc123 \
    --compared-folder test_results/gfx950/logs/20251204-hostname-def456 \
    --output-prefix comparison
```

Reads from `local.json` in each folder and generates:
- Performance improvement percentages
- Assembly change detection
- Best time comparisons

### 3. Historical Database Analysis

Query the global database for trends:
```bash
# View database with all historical data
python3 -m json.tool test_results/gfx950/database/exe_time-hostname_gfx950.json

# Export to CSV for spreadsheet analysis
python3 scripts/export_database_csv.py \
    --database-dir test_results/gfx950/database \
    --hostname hostname \
    --gpu-arch gfx950
```

Database structure allows tracking:
- Performance changes across git commits
- Comparison between default and +sia3 variants
- Code generation time trends
- Identification of regressions

## Unified YAML System

The infrastructure uses a **unified YAML system** with automatic variant generation:

### Single Source of Truth
All test types (exe-time, codegen-time, dbg-verify) use the same YAML files from the `yaml/` directory. Test-specific configurations are applied automatically at runtime through YAML patching.

### Test Variants
YAMLs can be tested with different configurations using **modifiers**:

```txt
# test_lists/exe-time.txt
[gfx942][gfx950][+sia3]
mfma.yaml
```

This creates multiple test runs:
- `mfma.yaml` (default: ScheduleIterAlg: [1])
- `mfma-sia3.yaml` (patched: ScheduleIterAlg: [3])

### Available Modifiers
- **[+sia3]**: Use ScheduleIterAlg: [3] instead of default [1]
- **[+sparse]**: Enable sparse matrix operations (Sparse: 1)
- More modifiers can be added in `patch_yaml.py`

### Automatic Patching
- **codegen-time tests**: Automatically patched with profiling settings (PythonProfile: True, NumWarmups: 0, etc.)
- **Modifiers**: Applied on top of base YAML (e.g., +sia3 changes ScheduleIterAlg)
- **Patched YAMLs**: Saved in logs directory for reference

## Adding New Test Cases

1. **Create a YAML file** in the unified directory:
   ```bash
   cp yaml/mfma.yaml yaml/my_new_test.yaml
   # Edit yaml/my_new_test.yaml with your test configuration
   ```

2. **Add to test lists** with section headers:
   ```txt
   # test_lists/exe-time.txt
   [gfx942][gfx950][+sia3]
   mfma.yaml
   my_new_test.yaml  # Add here
   ```

3. **Run tests** - variants are created automatically!
   ```bash
   ./scripts/main.sh --test exe_time --pattern "my_new_test" ...
   ```

## Section-Based Test Lists

Test lists use **section headers** to specify architectures and modifiers:

### Format
```txt
[arch1][arch2]...[+modifier1][+modifier2]...
test_file.yaml
another_test.yaml

[different_arch]
arch_specific_test.yaml
```

### Examples

**Example 1: Multi-architecture with variants**
```txt
[gfx942][gfx950][+sia3]
mfma.yaml        # Creates 4 runs: gfx942 default, gfx942-sia3, gfx950 default, gfx950-sia3
gemm_nn.yaml     # Same - 4 runs total
```

**Example 2: Architecture-specific tests**
```txt
[gfx950]
advanced_feature.yaml  # Only runs on gfx950, default variant

[gfx942]
legacy_test.yaml       # Only runs on gfx942, default variant
```

**Example 3: Multiple modifiers**
```txt
[gfx950][+sia3][+sparse]
sparse_gemm.yaml  # Creates 3 runs: default, sia3, sparse
```

### Test List Files
- `test_lists/exe-time.txt` - Execution time tests
- `test_lists/codegen-time.txt` - Code generation profiling
- `test_lists/dbg-verify.txt` - Debug verification

All test lists follow the same section-based format.

## Troubleshooting

### Container not starting
- Check if the container exists: `docker ps -a | grep ${container_name}`
- Check container logs: `docker logs ${container_name}`

### Build failures
- Ensure the tensilelite folder exists at `${hipblaslt_path}/tensilelite`
- Check CMake configuration: verify the preset exists
- Try a clean build: remove the build directory and rebuild

### Yappi installation fails
- Ensure pip3 is available in the container
- Try manual installation: `docker exec ${container_name} pip3 install yappi`

### No tests found
- Check that YAML files exist in the appropriate directories
- Verify the pattern matches your test files
- Check GPU-specific test lists for syntax errors

### Path mapping issues
- Ensure your working directory is mounted in the Docker container
- Check Docker volume mounts: `docker inspect ${container_name}`
- Try using absolute paths

## Analysis Tools

### 1. YAML Patching Tool (patch_yaml.py)

Create test variants by patching YAML configurations:

```bash
# Create SIA3 variant
python3 scripts/patch_yaml.py \
    --input yaml/mfma.yaml \
    --output /tmp/mfma-sia3.yaml \
    --schedule-iter-alg 3

# Create codegen profiling variant
python3 scripts/patch_yaml.py \
    --input yaml/mfma.yaml \
    --output /tmp/mfma-codegen.yaml \
    --patch-codegen

# Combine multiple patches
python3 scripts/patch_yaml.py \
    --input yaml/mfma.yaml \
    --output /tmp/mfma-sia3-codegen.yaml \
    --schedule-iter-alg 3 \
    --patch-codegen
```

**Note:** Patching happens automatically during test runs. This tool is useful for manual inspection or custom workflows.

### 2. Compare Test Sessions (compare_logs.py)

Compare performance between two test runs:

```bash
python3 scripts/compare_logs.py \
    --base-folder test_results/gfx950/logs/20251203-hostname-abc123 \
    --compared-folder test_results/gfx950/logs/20251204-hostname-def456 \
    --output-prefix comparison
```

**Features:**
- Reads from `local.json` in each session
- Compares execution times and code generation times
- Calculates improvement percentages
- Checks for assembly file changes
- Compares testcase-by-testcase (not SIA-separated)

### 3. Export Database to CSV (export_database_csv.py)

Export historical performance data to CSV for spreadsheet analysis:

```bash
python3 scripts/export_database_csv.py \
    --database-dir test_results/gfx950/database \
    --hostname hostname \
    --gpu-arch gfx950 \
    --output-dir ./csv_exports
```

**Features:**
- Exports entire git history to CSV
- Test cases as rows, git commits as columns
- Compatible with Excel/Google Sheets
- Useful for trend analysis and charting

### 4. Update Global Database (update_global_database.py)

Manually update the global database from local results:

```bash
python3 scripts/update_global_database.py \
    --local-json test_results/gfx950/logs/.../exe_time/local.json \
    --database-file test_results/gfx950/database/exe_time-hostname_gfx950.json \
    --git-hash abc123def
```

**Note:** This happens automatically after test runs. Manual use is for custom workflows or corrections.

## Contributing

When adding new features or tests:

1. Follow the existing naming conventions
2. Update relevant documentation
3. Test with multiple GPU architectures if possible
4. Ensure backward compatibility with existing test results

## License

This test infrastructure is part of the hipblaslt project.

