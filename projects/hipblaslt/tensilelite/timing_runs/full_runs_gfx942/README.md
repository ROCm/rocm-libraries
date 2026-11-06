# gfx942 before/after timing of the TensileLite common tests

| Item | Before | After |
|---|---|---|
| Commit | `aa19af72` (all timing instrumentation, before the first optimization `8ec52f37`) | `c1345541` (branch head + gfx942 script parameters) |
| Client | `tensilelite-client`, Release, `GPU_TARGETS=gfx942`, msgpack only, `/usr/bin/amdclang++` (ROCm 7.2.4) | same, built from its commit |
| Cores / GPU | `taskset -c 0-24`, `ROCR_VISIBLE_DEVICES=2` (NUMA 0) | `taskset -c 48-72`, `ROCR_VISIBLE_DEVICES=6` (NUMA 1) |

Host `sharkmi300x-1`: 2x EPYC 9454, 8x MI300X, shared (40+ users, load average 20-50, of which ~50 is
these two passes). `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=25`, `--timeout=5400`,
`-m "common and not skip-gfx94X"`, `LibraryFormat='msgpack'`. The two sides ran concurrently on
different sockets and GPUs. Both analysed and aggregated with the head's `scripts/analyze_timing.py` /
`aggregate_timing.py`. Scripts: `../full_runs_gfx950/{split,per_config}/` (gfx942 parameters from
`c1345541`).

`SKBUILD_EDITABLE_SKIP` was set to the rocisa build dir: the venv's editable rocisa otherwise re-runs
`cmake --build/--install` under a machine-wide lock on every import (1-9 s per pytest process and per
loky worker). CI installs a wheel. A first attempt without it was discarded.

## Folders

Committed: the four `AGGREGATE_analysis.txt`, the comparison tables, the split `times.txt` and the
config list. Not committed (too large, or local working notes): the raw per-config logs (`*.log.gz`,
~430 MB), the per-config `*_analysis.txt` and `times.txt`, the split logs/XML, the driver and the
hotspot analyses. They are kept on the measurement host next to this file.

- `split_pre/`, `split_post/`: CI-like split (uninstrumented, `-n 4`): `build` (`--build-only
  --gpu-targets gfx942`, caches cleared) then `run` (`--use-cache`, caches cleared). Logs, JUnit XML,
  `times.txt`, the passed run configs (`split_run_passed.txt`) and the shared list (`configs.txt`,
  123 configs, identical on both sides).
- `pre_build/`, `post_build/`: every config built serially, `TimingInstrumentation=True`; caches cleared
  once before the first config, artifacts kept. Per config: raw `*_build.log.gz`, `*_build_analysis.txt`,
  `times.txt` (pytest wall). `AGGREGATE_analysis.txt` over all 123.
- `pre_run/`, `post_run/`: every config run serially (`--use-cache`) on that build's artifacts and with
  its cache (not cleared). Same files.
- `build_comparison.txt`, `run_comparison.txt`: purpose tables side by side (`cmp_agg.py`).
- `opportunity_inventory.md`, `build_hotspots.md`, `run_hotspots.md`: remaining-opportunity analyses.

All 123 configs returned 0 in all four per-config passes. Split: build 145 passed / 1 failed
(`comm/gfx950/fused_a2a.yaml`, known, `CI_SPLIT_TIMING.md` finding 3) on both sides; run 0 failed.

## Results

| Measurement | Before | After | Change |
|---|---:|---:|---:|
| CI-like build phase wall (`-n 4`) | 1,028 s | 634 s | -38% |
| CI-like run phase wall (`-n 4`) | 1,494 s | 742 s | -50% |
| Per-config build TOTAL (instrumented) | 2,134 s | 1,152 s | -46% |
| Per-config build pytest wall sum | 2,898 s | 1,886 s | -35% |
| Per-config run TOTAL (instrumented) | 1,644 s | 829 s | -50% |
| Per-config run pytest wall sum | 2,320 s | 1,484 s | -36% |

No config got slower by more than 2 s in either per-config pass. Single samples (no repeats); the
sides ran on different sockets/GPUs.
