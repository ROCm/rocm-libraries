# Quick Start Guide

This guide will help you get started with the stinkytofu test infrastructure in just a few minutes.

## Prerequisites

Before you begin, you need:

1. A running Docker container with ROCm and hipblaslt
2. The container name
3. The path to hipblaslt inside the container
4. The path to hipblaslt on your host system

## Step 1: Verify Your Setup

Check that your container is available:

```bash
docker ps -a | grep your_container_name
```

If the container is not running, start it:

```bash
docker start your_container_name
```

## Step 2: Run Your First Test

Navigate to any directory where you want to run the tests (the scripts will work from anywhere):

```bash
cd ~/my_workspace
```

Run the test suite:

```bash
/path/to/shared/stinkytofu/test/scripts/main.sh \
    --container-name your_container_name \
    --docker-path /workspace/hipblaslt \
    --host-path /home/you/rocm-libraries_p/projects/hipblaslt \
    --output-dir ./test_results \
    --test exe_time
```

Replace:
- `your_container_name` with your actual container name
- `/workspace/hipblaslt` with the path inside your container
- `/home/you/rocm-libraries_p/projects/hipblaslt` with the path on your host
- `./test_results` with your desired output directory

## Step 3: Check the Results

After the test completes, check the results:

```bash
# View the summary report
cat test_results/gfx942/exe_time/*_report.txt

# View detailed results
ls -la test_results/gfx942/exe_time/
```

## Example: Full Test Run

Here's a complete example for a typical setup:

```bash
#!/bin/bash

# Configuration
CONTAINER="rocm_dev"
DOCKER_PATH="/workspace/hipblaslt"
HOST_PATH="/home/cycheng2/rocm-libraries_p/projects/hipblaslt"
OUTPUT_DIR="/tmp/hipblaslt_test_$(date +%Y%m%d_%H%M%S)"

# Run all tests
/home/cycheng2/rocm-libraries_p/shared/stinkytofu/test/scripts/main.sh \
    --container-name "$CONTAINER" \
    --docker-path "$DOCKER_PATH" \
    --host-path "$HOST_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --test all

echo "Tests completed! Results are in: $OUTPUT_DIR"
```

## Common Use Cases

### Run Only Quick Verification Test

```bash
./scripts/main.sh \
    --container-name my_container \
    --docker-path /workspace/hipblaslt \
    --host-path ~/hipblaslt \
    --output-dir /tmp/quick_test \
    --test dbg_verify \
    --pattern "mfma"
```

### Run Performance Benchmarks

```bash
./scripts/main.sh \
    --container-name my_container \
    --docker-path /workspace/hipblaslt \
    --host-path ~/hipblaslt \
    --output-dir /tmp/benchmark \
    --test exe_time
```

### Profile Code Generation Time

```bash
./scripts/main.sh \
    --container-name my_container \
    --docker-path /workspace/hipblaslt \
    --host-path ~/hipblaslt \
    --output-dir /tmp/codegen \
    --test codegen_time
```

## Understanding the Output

The output is organized into two main sections:

### Database Directory
Contains JSON files and human-readable reports for tracking performance over time:
```
test_results/
└── gfx950/
    └── database/
        ├── exe_time-hostname_gfx950.json       # Historical performance data
        ├── codegen_time-hostname_gfx950.json   # Code generation metrics
        └── dbg_verify-hostname_gfx950_report.txt
```

### Logs Directory
Contains detailed test logs organized by test session (date-hostname-git_hash):
```
test_results/
└── gfx950/
    └── logs/
        └── 20251203-hostname-abc1234/
            ├── exe_time/
            │   ├── mfma.yaml              # Patched YAML (default)
            │   ├── mfma-sia3.yaml         # Patched YAML (SIA3 variant)
            │   ├── mfma.log               # Test output (default)
            │   ├── mfma-sia3.log          # Test output (SIA3)
            │   ├── mfma/                  # Assembly files (default)
            │   │   ├── 0.s
            │   │   └── 1.s
            │   ├── mfma-sia3/             # Assembly files (SIA3)
            │   │   ├── 0.s
            │   │   └── 1.s
            │   ├── local.json             # Summary results
            │   ├── local-sia3cmp.json     # Detailed results
            │   └── local_report.txt       # Human-readable summary
            ├── codegen_time/
            │   ├── mfma-sia3.yaml
            │   ├── mfma-sia3_yappi_results.txt
            │   ├── mfma-sia3.log
            │   ├── local.json
            │   └── local_report.txt
            └── dbg_verify/
                ├── mfma.yaml
                ├── mfma-sia3.yaml
                ├── mfma.log
                ├── mfma-sia3.log
                └── summary.txt
```

### Key Information in Reports

**Execution Time (Test A)**:
- Problem sizes and configurations
- Execution time (microseconds)
- GFLOPS performance
- SIA version used
- Validation status (PASSED/FAILED)

**Code Generation Time (Test B)**:
- `_getKernelSource` total time in seconds
- Statistical analysis across runs
- Trend tracking

**Debug Verify (Test C)**:
- Total tests run
- Number passed/failed
- Git commit and test metadata

## Troubleshooting

### "Container not found"
```bash
docker ps -a  # List all containers
docker start container_name  # Start the container
```

### "Tensilelite folder not found"
Check that the path is correct:
```bash
docker exec your_container ls -la /workspace/hipblaslt/tensilelite
```

### "Build failed"
Try a clean build:
```bash
docker exec your_container rm -rf /workspace/hipblaslt/tensilelite/rel_build
# Then run the tests again
```

### "No tests found"
Check available test files:
```bash
ls /home/cycheng2/rocm-libraries_p/shared/stinkytofu/test/yaml/
```

## Understanding Test Variants

The infrastructure now uses a **unified YAML system** that automatically creates test variants:

### Single YAML, Multiple Variants
Tests can run with different configurations using modifiers in test lists:

```txt
# test_lists/exe-time.txt
[gfx942][gfx950][+sia3]
mfma.yaml
```

This creates **4 test runs** from one YAML:
- `mfma.yaml` on gfx942 (default: ScheduleIterAlg: [1])
- `mfma-sia3.yaml` on gfx942 (ScheduleIterAlg: [3])
- `mfma.yaml` on gfx950 (default)
- `mfma-sia3.yaml` on gfx950 (SIA3 variant)

### Available Modifiers
- **[+sia3]**: Tests SIA3 scheduling algorithm (ScheduleIterAlg: [3])
- **[+sparse]**: Enables sparse matrix operations
- More modifiers can be added as needed

### Test List Format
Section headers define which architectures and modifiers apply:

```txt
[gfx942][gfx950][+sia3]
mfma.yaml          # Runs on both arches with default + sia3

[gfx950]
advanced_test.yaml  # Runs only on gfx950, default only
```

YAMLs are patched automatically at runtime - no duplication needed!

## Step 4: Analyze Results (Optional)

### Sia3cmp Database (Stinkytofu vs SIA3 Comparison)

When you run tests with both stinkytofu (default) and SIA3 variants, a **sia3cmp database** is automatically created and updated:

```
test_results/
└── gfx950/
    └── database/
        └── sia3cmp-hostname_gfx950.json     # Persistent comparison database
```

**What the sia3cmp database tracks:**
- For each problem and git hash, it maintains:
  - **Fastest result**: The best execution time across all runs
  - **Latest result**: The most recent test run
- Data is stored for both stinkytofu and SIA3 variants
- Database is automatically updated after each test run

### Export All Databases to CSV

Export all databases (exe_time, codegen_time, and sia3cmp) to CSV format:

```bash
python3 scripts/export_database_csv.py \
    --database-dir regression/gfx950/database \
    --hostname myhost \
    --gpu-arch gfx950 \
    --verbose
```

This automatically exports:
- **exe_time databases**: `exe_time_best_*.csv` and `exe_time_latest_*.csv`
- **codegen_time databases**: `codegen_time_best_*.csv` and `codegen_time_latest_*.csv`
- **sia3cmp database** (if exists): `sia3cmp-hostname_gfx950.csv`
  - Spreadsheet format with columns: Problem_Sizes, Git_Hash, Type (Fastest/Latest)
  - Stinkytofu metrics: Time_us, GFLOPS, Validation, Date
  - SIA3 metrics: Time_us, GFLOPS, Validation, Date
  - Percent_Diff, Speedup, DataType, TransposeA, TransposeB, etc.

**Use cases:**
- Export to Excel/Google Sheets for trend analysis
- Track performance across git commits
- Compare fastest vs latest to monitor stability
- Identify regressions in either stinkytofu or SIA3

### Local Test Reports

After each test run, detailed text reports are automatically generated in the log directory:

- **`local-sia3cmp_report.txt`** (exe_time tests): Side-by-side comparison of stinkytofu vs SIA3
  - Shows both LATEST and BEST (fastest) results
  - Table format with problem configuration, validation, time, GFLOPS, and percentage difference

**Example format:**
```
                                                   | Stinkytofu (Latest)              | SIA3 (Latest)                    | % Diff
---------------------------------------------------+----------------------------------+----------------------------------+----------
Problem: (250,250,1,19200)                         |                                  |                                  |
  DataType: H                                      |                                  |                                  |
  TransposeA: 1, TransposeB: 0                     |                                  |                                  |
                                                   |                                  |                                  |
Validation                                         | PASSED                           | PASSED                           |
Time                                               | 310.033 us                       | 315.178 us                       | +1.64%
GFLOPS                                             | 7741.12                          | 7614.75                          |
```

- **`local-sia3cmp_report.txt`** (codegen_time tests): Summary of code generation times
  - Shows latest and best (fastest) codegen times for each test
  - Complete history of all runs

These reports are found in: `regression/gfx950/logs/hostname-githash/exe_time/local-sia3cmp_report.txt`

### Compare Two Test Runs

After running tests multiple times, compare the results:

```bash
python3 scripts/compare_logs.py \
    --base-folder regression/gfx950/logs/20251202-hostname-abc1234 \
    --compared-folder regression/gfx950/logs/20251211-hostname-def5678 \
    --output-prefix my_comparison
```

This generates a comparison report showing:
- Performance improvements or regressions
- Assembly file changes
- Best times from multiple runs

### Export to CSV for Trend Analysis

Export all historical data to CSV:

```bash
python3 scripts/export_database_csv.py \
    --database-dir regression/gfx950/database \
    --hostname myhost \
    --gpu-arch gfx950
```

Open the CSV files in Excel/Google Sheets to:
- Create performance trend charts
- Identify regressions over time
- Compare performance across git commits

📖 **See [ANALYSIS_TOOLS.md](ANALYSIS_TOOLS.md) for detailed documentation.**

## Next Steps

- Read the full [README.md](README.md) for detailed information
- Add your own test cases in `yaml/` (unified folder)
- Update test lists in `test_lists/` with section headers
- Customize the test patterns to focus on specific tests
- Set up automated regression testing with CI/CD
- Explore test variants with modifiers like [+sia3]

## Getting Help

If you encounter issues:

1. Check the logs in the output directory
2. Review the full README.md for detailed documentation
3. Verify your Docker container setup
4. Ensure all paths are correct
5. Check that the hipblaslt build is up to date

## Tips

- Use timestamped output directories to track results over time
- Run tests after each significant code change
- Compare JSON databases to track performance trends
- Use `--pattern` to focus on specific test cases during development
- Keep old results for historical comparison

