# Add Ranking Regression Tests for Origami

## Summary

This PR introduces ranking regression tests for the Origami config selection system. These tests ensure that changes to the codebase do not unintentionally alter the ranking behavior of configurations across supported GPU architectures and data types.

## Changes

### New Files

- **`test_ranking_regression.py`**: Main test file containing:
  - `TestRankingRegression` class with `test_ranking_stability` parameterized across architectures (gfx90a, gfx942, gfx950, gfx1100, gfx1201) and data types (f16, bf16, f32)
  - Dynamic matrix instruction discovery via `hardware.get_valid_matrix_instructions()`
  - Config generation using configurable macro tile sizes and depth unroll values
  - Baseline comparison with detailed diff output on failure

- **`test_utils.py`**: Shared test utilities containing:
  - `SUPPORTED_ARCHITECTURES` dictionary with hardware configurations for all supported GPUs
  - `create_hardware()` helper function for consistent hardware object creation

- **`baselines/rankings/*.csv`**: Golden baseline files (13 files) storing top-10 ranked configurations for 200 problem sizes per architecture/dtype combination

- **`baselines/README.md`**: Documentation for baseline generation and maintenance

- **`data/problem_data.csv`**: Test problem definitions (200 GEMM problems with varying M, N, K, batch dimensions)

### Modified Files

- **`conftest.py`**: Added `--generate-baseline` CLI option and `generate_baseline` fixture; updated `hardware` fixture to use shared `create_hardware()`

- **`pyproject.toml`**: Registered `regression` pytest marker

## Usage

### Running Tests
```bash
# Run ranking regression tests
pytest tests/test_ranking_regression.py -v

# Run only regression-marked tests
pytest -m regression -v
```

### Generating/Updating Baselines
```bash
# Generate new baselines (run from develop branch)
pytest tests/test_ranking_regression.py -v --generate-baseline

# Update baseline for specific architecture
pytest tests/test_ranking_regression.py -v --generate-baseline -k gfx942
```

## Test Coverage

| Architecture | f16 | bf16 | f32 |
|-------------|-----|------|-----|
| gfx90a      | Yes | Yes  | Yes |
| gfx942      | Yes | Yes  | Yes |
| gfx950      | Yes | Yes  | Yes |
| gfx1100     | Yes | Yes  | Skip |
| gfx1201     | Yes | Yes  | Skip |

Tests are skipped when the architecture doesn't support the data type (determined dynamically via `get_valid_matrix_instructions`).

## Example Failure Output

When a ranking regression is detected:
```
Failed: Ranking regression detected for gfx942/f16:
36912x62832x4448x1 rank 0: Config mismatch
  Current:  MT=(256, 256, 64), MI=(16, 16, 16), occ=2, wgm=1
  Baseline: MT=(256, 256, 64), MI=(32, 32, 8), occ=2, wgm=1
36912x62832x4448x1 rank 1: Latency diff 5.23% (curr=3.17e+08, base=3.01e+08)
... and 142 more differences
```

## Design Decisions

1. **CSV format for baselines**: Chosen for space efficiency (~108KB per file vs ~635KB for JSON with equivalent data)

2. **Top-10 configs stored**: Captures ranking stability beyond just the best config

3. **Dynamic matrix instruction discovery**: Uses `hardware.get_valid_matrix_instructions()` instead of hardcoded values to automatically adapt to library changes

4. **Separate `test_utils.py`**: Shared hardware configurations avoid duplication while keeping `test_hardware.py` unit tests explicit

## Test Plan

- [x] All 13 ranking stability tests pass against baselines
- [x] All 8 hardware tests pass
- [x] `--generate-baseline` flag correctly creates/updates baseline files
- [x] Skipped tests correctly identified (gfx1100/gfx1201 lack f32 support)
