# Kernel Count Validator

A tool to detect kernel count regressions in TensileLite by comparing actual solution counts against a known baseline.

## Overview

When TensileLite generates kernels from YAML configs, it produces a set of solutions based on `ForkParameters`. Sometimes code changes can inadvertently cause valid solutions to be rejected, reducing the kernel count. This tool catches such regressions by:

1. Running Tensile on YAML configs with `--cpu-only --build-only`
2. Parsing "Actual Solutions: X / Y" from the output
3. Comparing against baseline counts stored in JSON

## Quick Start

```bash
# Activate environment
source /path/to/venv/bin/activate
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/path/to/venv/lib/python3.10/site-packages/rocisa:$LD_LIBRARY_PATH

# Verify all YAMLs (main CI command)
python3 scripts/kernel_count_validator.py --run-and-verify --yaml-dir tensilelite/Tests/common -j 8

# Add a new YAML to baseline
python3 scripts/kernel_count_validator.py --add-yaml --yaml path/to/new.yaml
```

## Commands

### Verification Commands

| Command | Description |
|---------|-------------|
| `--run-and-verify` | Run Tensile on YAMLs and verify counts against baseline (main command) |
| `--verify` | Verify counts from existing log files against baseline |
| `--validate` | Validate a single Tensile log against baseline |

### Baseline Management Commands

| Command | Description |
|---------|-------------|
| `--add-yaml` | Run Tensile on a single YAML and add/update its baseline entry |
| `--update-from-logs-dir` | Update baseline from all `.log` files in a directory |
| `--update-from-log` | Update baseline from a single log file |
| `--generate-baseline` | Generate baseline from YAML ForkParameters calculations (no Tensile run) |
| `--run-and-update` | Run Tensile on all YAMLs and update baseline with actual counts |

### Utility Commands

| Command | Description |
|---------|-------------|
| `--generate-run-script` | Generate a bash script to run Tensile on all YAMLs |
| `--check-yaml` | Print expected kernel counts for a YAML (calculation only, no Tensile run) |

## Options

### Input Options

| Option | Default | Description |
|--------|---------|-------------|
| `--yaml PATH` | None | Path to a single YAML file |
| `--yaml-dir PATH` | `Tensile/Tests/common` | Directory containing YAML files |
| `--log PATH` | None | Path to a single Tensile log file |
| `--logs-dir PATH` | None | Directory containing `.log` files |

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--arch ARCH` | `gfx950` | Target GPU architecture |
| `--baseline PATH` | `scripts/kernel_count_baseline_gfx950.json` | Path to baseline JSON file |
| `-j N` / `--parallel N` | `1` | Number of parallel jobs |
| `--tensile-path PATH` | Auto-detect | Path to Tensile script |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-script PATH` | `run_all_yamls.sh` | Output path for generated bash script |

## Usage Examples

### Verification

```bash
# Verify all YAMLs with 8 parallel jobs (default arch: gfx950)
python3 scripts/kernel_count_validator.py --run-and-verify \
    --yaml-dir tensilelite/Tests/common -j 8

# Verify a single YAML
python3 scripts/kernel_count_validator.py --run-and-verify \
    --yaml tensilelite/Tests/common/gemm/bf16_tn.yaml

# Verify for a different architecture
python3 scripts/kernel_count_validator.py --run-and-verify \
    --yaml-dir tensilelite/Tests/common -j 8 --arch gfx942

# Verify using a custom baseline file
python3 scripts/kernel_count_validator.py --run-and-verify \
    --yaml-dir tensilelite/Tests/common \
    --baseline scripts/kernel_count_baseline_gfx942.json \
    --arch gfx942

# Verify from existing log files
python3 scripts/kernel_count_validator.py --verify --logs-dir tensile_logs
```

### Adding/Updating Baseline

```bash
# Add a single YAML to baseline
python3 scripts/kernel_count_validator.py --add-yaml \
    --yaml tensilelite/Tests/common/gemm/new_test.yaml

# Add YAML for a different architecture
python3 scripts/kernel_count_validator.py --add-yaml \
    --yaml path/to/test.yaml --arch gfx942

# Update baseline from log files directory
python3 scripts/kernel_count_validator.py --update-from-logs-dir \
    --logs-dir tensile_logs

# Update baseline from a single log file
python3 scripts/kernel_count_validator.py --update-from-log \
    --log tensile_logs/gemm_bf16_tn.log
```

### Generating Run Scripts

```bash
# Generate parallel bash script (8 jobs)
python3 scripts/kernel_count_validator.py --generate-run-script \
    --yaml-dir tensilelite/Tests/common -j 8 \
    --output-script run_all.sh

# Run the generated script
./run_all.sh

# Then verify from the logs
python3 scripts/kernel_count_validator.py --verify --logs-dir tensile_logs
```

### Inspection

```bash
# Check expected counts for a YAML (no Tensile run, just calculation)
python3 scripts/kernel_count_validator.py --check-yaml \
    --yaml tensilelite/Tests/common/gemm/bf16_tn.yaml
```

## Output Format

### Verification Results

```
Running kernel count verification on 170 YAML files...
  Architecture: gfx950
  Parallel jobs: 8

  [1/170] bf16_tn.yaml: [PASS] 2048 solutions
  [2/170] fp32_tn.yaml: [FAIL] problem_0: actual=1024, baseline=2048 (REGRESSION: 1024 fewer solutions)
  [3/170] new_test.yaml: [MISSING] No baseline entry (512 solutions generated)

============================================================
Results: 168 PASS, 1 FAIL, 1 MISSING, 0 SKIP
============================================================

New YAMLs missing from baseline:
  new_test.yaml: [MISSING] No baseline entry (512 solutions generated)

To add them to baseline, run:
  python3 scripts/kernel_count_validator.py --add-yaml --yaml path/to/new_test.yaml

Failed tests:
  fp32_tn.yaml: [FAIL] problem_0: actual=1024, baseline=2048 (REGRESSION: 1024 fewer solutions)

✗ KERNEL COUNT REGRESSION DETECTED
```

### Status Labels

| Label | Meaning |
|-------|---------|
| `[PASS]` | Kernel count matches baseline |
| `[FAIL]` | Kernel count is lower than baseline (regression) |
| `[MISSING]` | Valid YAML but no baseline entry exists |
| `[SKIP]` | YAML skipped (not valid for architecture, ductile backend, etc.) |
| `[WARN]` | Kernel count is higher than baseline (potential baseline update needed) |

## Baseline JSON Format

The baseline file stores expected counts per YAML, per architecture:

```json
{
  "gemm/bf16_tn.yaml": {
    "gfx950": {
      "problem_0_b_TN_Batched_Bias": {
        "actual": 2048,
        "expected": 4096
      }
    },
    "gfx942": {
      "problem_0_b_TN_Batched_Bias": {
        "actual": 1024,
        "expected": 2048
      }
    }
  },
  "gemm/fp32_tn.yaml": {
    "gfx950": {
      "problem_0_s_TN_Batched_Bias": {
        "actual": 2048,
        "expected": 4096
      }
    }
  }
}
```

- `actual`: Number of solutions that passed validation (from Tensile run)
- `expected`: Number of solutions attempted (from ForkParameters expansion)

## YAML Filtering

The tool automatically filters YAMLs based on:

1. **Architecture compatibility** - Checks `GlobalParameters.Architecture`, `GlobalParameters.ISA`, and `TestParameters.marks` for `skip-gfx*` markers
2. **Backend type** - Excludes `ductile` backend YAMLs (evolutionary search, not exhaustive enumeration)
3. **ForkParameters** - Skips YAMLs with empty ForkParameters (e.g., CustomKernels-only configs)

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All checks passed |
| `1` | Regression detected, missing baseline, or error |

## Workflow for Adding New YAMLs

When you add a new YAML test file:

1. Run verification - it will flag the new YAML as `[MISSING]`
2. Add to baseline using the suggested command:
   ```bash
   python3 scripts/kernel_count_validator.py --add-yaml --yaml path/to/new.yaml
   ```
3. Commit the updated baseline JSON along with your YAML

## Workflow for Investigating Regressions

If verification fails with a regression:

1. Check the failing YAML and problem name in the output
2. Compare with the baseline to see expected vs actual counts
3. Investigate recent changes to:
   - `SolutionStructs.py` (solution validation)
   - `KernelWriter.py` (kernel generation)
   - The specific YAML's ForkParameters
4. Fix the regression or update the baseline if the change is intentional

## Environment Setup

```bash
# Activate virtual environment
source /path/to/venv/bin/activate

# Required for rocisa module
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/path/to/venv/lib/python3.10/site-packages/rocisa:$LD_LIBRARY_PATH
```

## Files

| File | Description |
|------|-------------|
| `scripts/kernel_count_validator.py` | Main validator script |
| `scripts/kernel_count_baseline_gfx950.json` | Baseline for gfx950 architecture |
