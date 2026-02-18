# Ranking Regression Baselines

This directory contains golden baseline files for the ranking regression tests.

## Directory Structure

```
baselines/
└── rankings/
    ├── gfx90a_f16.json
    ├── gfx90a_bf16.json
    ├── gfx942_f16.json
    ├── gfx942_bf16.json
    ├── gfx950_f16.json
    ├── gfx950_bf16.json
    └── ...
```

## Generating Baselines

Baselines should be generated from the `develop` branch to establish the expected
ranking behavior. Run the following command:

```bash
# From the origami python directory
pytest tests/test_ranking_regression.py -v --generate-baseline
```

This will create JSON files containing the ranked configs for each architecture
and data type combination.

## Updating Baselines

If a PR intentionally changes ranking behavior (e.g., fixing a bug or improving
the heuristics), the baselines need to be updated:

1. Ensure the changes are intentional and reviewed
2. Run the baseline generation command above
3. Commit the updated baseline files with the PR

## Baseline File Format

Each baseline file contains rankings for multiple problem sizes:

```json
{
  "512x512x512": [
    {
      "rank": 0,
      "latency": 0.001234,
      "config": {
        "mt": {"m": 128, "n": 128, "k": 64},
        "mi": {"m": 16, "n": 16, "k": 16},
        "occupancy": 1,
        "workgroup_mapping": 4
      }
    },
    ...
  ],
  "1024x1024x1024": [...],
  ...
}
```
