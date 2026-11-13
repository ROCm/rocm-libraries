# Param-group OOB runs

Per-parameter-family TensileLite sweeps against BBS TN OOB representatives.

| Folder | Group | Smoke sizes | Full sizes |
| --- | --- | ---: | ---: |
| `01_tdm_loads/` | 1. TDM loads | 20 | 60 |
| `02_workgroup_cluster/` | 2. Workgroup cluster | 20 | 60 |
| `03_lds_segment/` | 3. LDS segment / bank conflict | 20 | 60 |
| `04_memory_hints/` | 4. Memory temporal / volatility hints | 20 | 60 |
| `05_prefetch/` | 5. Prefetch (SWI + GL2) | 20 | 60 |
| `06_persistent_reuse/` | 6. Persistent tile reuse (RAP) | 20 | 60 |
| `07_epilogue/` | 7. Epilogue / store scheduling | 20 | 60 |
| `08_codegen_ra/` | 8. Codegen / RA tooling | 20 | 60 |

## Layout

```
runs/<group>/
  config.yaml          # all 60 Exact sizes, full MI set, group forks
  config_smoke.yaml    # 20 Exact sizes (1/OOB category), MT 256x256 only
  results/             # full run output (Tensile 1_BenchmarkProblems/...)
  results_smoke/       # smoke run output
  summary_*.csv        # flattened numbers after collect_results.py
```

## Run (on a gfx1250 host with ROCm + TensileLite client build)

```bash
cd param_testing/runs
./run_all.sh smoke                 # recommended first pass
./run_all.sh smoke 01_tdm_loads    # one group
./run_all.sh full                  # all 60 sizes (long / memory-heavy)
```

Override Tensile path if needed:

```bash
TENSILE=/path/to/tensilelite/bin/Tensile ./run_all.sh smoke
```

After each group finishes, `collect_results.py` writes `summary_<mode>.csv`
in that group's output dir, then merges `runs/summary_<mode>.csv`.

## Notes

- Full OOB Exact sizes include very large M/N/K; prefer smoke for impact screening.
- Dead knobs (`TDMSplit`, `TDMPlusLdsBuf`) are not forked (see `parameter_combinations.md`).
- This WSL workspace has no ROCm/GPU; run on the gfx1250 machine used for the CSV (e.g. g04-2).
