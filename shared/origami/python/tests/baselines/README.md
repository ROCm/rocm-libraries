# Ranking Regression Baselines

This directory contains golden baseline files for ranking regression tests.

## File Structure

```
baselines/
└── rankings/
    ├── gfx90a.yaml
    ├── gfx942.yaml
    ├── gfx950.yaml
    ├── gfx1100.yaml
    └── gfx1201.yaml
```

Each architecture has its own YAML file containing rankings for all dtype/transpose
combinations.

## Generating Baselines

Baselines should be generated from the `develop` branch to establish the expected
ranking behavior:

```bash
# From the origami python directory
pytest tests/test_ranking_regression.py -v --generate-baseline
```

This creates/updates YAML files containing the top-5 ranked configs for each
architecture, data type, and transpose combination.

## Updating Baselines

If a PR intentionally changes ranking behavior (e.g., fixing a bug or improving
the heuristics), the baselines need to be updated:

1. Ensure the changes are intentional and reviewed
2. Run the baseline generation command above
3. Commit the updated baseline files with the PR

## Baseline File Format

Each YAML file has a hierarchical structure:

```yaml
dtype:
  transpose:
    "MxNxKxBatch":
      - [mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, wgm]  # rank 0
      - [mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, wgm]  # rank 1
      ...
```

Where:
- `dtype`: Data type (f16, bf16, f32)
- `transpose`: Matrix transpose combination (TN, TT, NN, NT)
  - T = Transposed, N = Non-transposed (first char = A, second = B)
- Config tuple: `[mt_m, mt_n, mt_k, mi_m, mi_n, mi_k, occ, wgm]`
- Rank is implicit from list position (0-4 for top-5)

Example (gfx90a.yaml):

```yaml
f16:
  TN:
    36912x62832x4448x1:
      - [256, 256, 64, 16, 16, 16, 2, 1]
      - [256, 256, 64, 16, 16, 16, 2, 4]
      - [256, 256, 64, 16, 16, 16, 2, 8]
      ...
```
