# Ranking Regression Baselines

This directory contains golden baseline files for the ranking regression tests.

## Directory Structure

```
baselines/
└── rankings/
    ├── gfx90a_f16_TN.csv   # A=Transposed, B=Non-transposed
    ├── gfx90a_f16_TT.csv   # A=Transposed, B=Transposed
    ├── gfx90a_f16_NN.csv   # A=Non-transposed, B=Non-transposed
    ├── gfx90a_f16_NT.csv   # A=Non-transposed, B=Transposed
    ├── gfx90a_bf16_TN.csv
    ├── gfx90a_bf16_TT.csv
    ├── ...
    └── gfx1201_bf16_NT.csv
```

The filename format is `{arch}_{dtype}_{transpose}.csv` where transpose indicates
the combination of A and B matrix transpose types (T=Transposed, N=Non-transposed).

## Generating Baselines

Baselines should be generated from the `develop` branch to establish the expected
ranking behavior. Run the following command:

```bash
# From the origami python directory
pytest tests/test_ranking_regression.py -v --generate-baseline
```

This will create CSV files containing the top-10 ranked configs for each
architecture and data type combination.

## Updating Baselines

If a PR intentionally changes ranking behavior (e.g., fixing a bug or improving
the heuristics), the baselines need to be updated:

1. Ensure the changes are intentional and reviewed
2. Run the baseline generation command above
3. Commit the updated baseline files with the PR

## Baseline File Format

Each baseline file is a CSV with the following columns:

```csv
problem,rank,latency,mt_m,mt_n,mt_k,mi_m,mi_n,mi_k,occ,wgm
```

Where:
- `problem`: Problem dimensions in format `MxNxKxBatch`
- `rank`: Ranking position (0-9 for top-10)
- `latency`: Predicted latency value
- `mt_m`, `mt_n`, `mt_k`: Macro tile dimensions
- `mi_m`, `mi_n`, `mi_k`: Matrix instruction dimensions
- `occ`: Occupancy
- `wgm`: Workgroup mapping

Example:

```csv
problem,rank,latency,mt_m,mt_n,mt_k,mi_m,mi_n,mi_k,occ,wgm
36912x62832x4448x1,0,3.17424e+08,256,256,64,32,32,8,2,1
36912x62832x4448x1,1,3.17424e+08,256,256,64,32,32,8,2,4
36912x62832x4448x1,2,3.17424e+08,256,256,64,32,32,8,2,8
...
```
