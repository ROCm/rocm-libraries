# Coverage extension of the navi32 HHS-TN re-map — measured inputs, UNVALIDATED output

**Read this first: the catalog in this commit has NOT been performance-validated.** It builds
clean for gfx1101/gfx1102 and is numerically correct, but no A/B benchmark was ever run against
it. The campaign was stopped at 53% of the planned measurement to hand off to real navi32
hardware. Everything below distinguishes what was measured from what was not.

## What changed

`navi32_Cijk_Alik_Bljk_HHS_BH_Bias_HAS_SAV_UserArgs.yaml` — **2 004 of 9 680 rows (20.7%)**
re-pointed at a measured-better kernel, on top of the previously shipped re-map. Grid keys, row
count and kernel pool are untouched; only `element[7]` solution indices change.

| | |
|---|---|
| rows measured cold at 60 CU | **5 201 / 9 680 (53%)** — was 2 139 (22%) |
| rows re-pointed vs shipped | 2 004 (large 166, med 655, skinny_M 593, skinny_N 590) |
| left alone | 4 479 unmeasured, 202 below the 2% gain gate, 1 437 in gated strata |
| gate | `--skip-strata tiny,gemv --min-gain 0.02` |

## Why this should help (measured at the row, not end-to-end)

Headroom of the best *installable* kernel over the shipped pick, non-gated strata:

| rows | median headroom | clears 2% gate |
|---|---|---|
| rows the shipped catalog already measured | **0.0%** | 41% |
| rows it never measured | **~15–19%** (see caveat) | **89%** |

The first line is the internal calibration and the real argument: **shipped already picks the
best available kernel on everything it measured**, so all remaining gain comes from rows never
measured. That is what this extension buys.

**Caveat on the second line.** That median is a *snapshot over a non-representative subsample*
and it moved as the sweep progressed (12.3 → 14.4 → 18.2 → 26.3 → 26.6% across measurement
order). Round-robin ordering guarantees the *final* result is unbiased but says nothing about a
mid-sweep slice. The stable statistic is the **proportion clearing the gate: 87–93% in every
chunk**. Do not quote the median as an expected gain.

## Headroom is an inverted U in problem size

| log10(M·N·K) | 4 | 5 | 6 | **7** | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| median headroom | 9.8% | 10.9% | 16.0% | **36.1%** | 30.6% | 15.8% | 7.1% |

Peaks at mid-size and collapses at both ends. *Interpretation, not established:* tiny shapes are
dispatch-bound so no kernel helps; huge shapes have enough waves to absorb a bad tile; mid-size
shapes tuned for 96 CUs land on awkward wave quantization at 60. This says the gain should come
from `med`, which is 45.6% of the grid.

## What was verified

| check | result |
|---|---|
| grid rows / keys / pool | 9 680 unchanged, keys byte-identical, all indices in-pool |
| build gfx1101 | 97 kernels, 100 solutions, 0 asm errors, 0 overflow, ELF `0x46` |
| build gfx1102 | 97 kernels, 100 solutions, 0 asm errors, 0 overflow, ELF `0x47` |
| numerical correctness | **62 pass / 0 fail** (gfx1100 build, `norm_error` vs `rtol`) |
| matrix index space | wgm8's 298 kernels map identically to the navi31 pool on all 298 |

## What was NOT verified — the honest list

1. **No performance benchmark of this catalog. None.** The 7-arm A/B (shipped / extended /
   extended_ship / nogate / full / lean / A-A, twice over 600 shapes) was staged and never ran.
2. **47% of the grid is still unmeasured** and keeps the shipped pick — which is safe, not wrong.
3. **The tiny/gemv gate is unresolved.** It was adopted because the `argmax` catalog measured
   tiny at 64.80%, but the `full` catalog measures tiny at **99.79%** — a different catalog, so
   the gate was justified against something other than what ships here. Offline, tiny/gemv have
   the *best* kernel-choice transfer of any stratum (99.5%/99.4% vs med's 95.2%). This catalog
   keeps the gate, i.e. the conservative choice, but the gate may be leaving ~3% on the table.
4. **navi33 deliberately untouched.** These measurements are at 60 CU. navi33 is 32 CU, and
   inheriting a CU-count-specific tuning is the exact error this campaign exists to remove.
5. **Measured on gfx1100 emulating navi32** (`--sm_count_target 60` + CU mask), not on real
   gfx1101. Selection is navi32-correct; absolute throughput is not.

## Expected outcome, pre-registered

`PREDICTIONS.md` was written before any benchmark, with the conditions that would falsify it.
Short version: expect **+1 to +3%** on treated queries, *not* the ~19% row-level headroom, because
of a measured ~2% median transfer tax (p10 78–82%) and because only ~38% of treated queries
change kernel at all. **A result near +19% would indicate measurement at grid keys by mistake,
not success.**

## How to continue on real navi32

```bash
# 1. resume the sweep natively (it is resumable and skips measured rows)
python3 matrix_sweep.py --shapes full_grid_shapes.json --out cold_matrix.jsonl \
        --cus 60 --cold --target-us 10000 --timeout 120

# 2. rebuild the catalog from the fuller matrix
python3 remap_grid.py --logic <shipped>.yaml --matrix cold_matrix.jsonl \
        --src-pool <navi31 full pool>.yaml --skip-strata tiny,gemv --min-gain 0.02 --out new.yaml

# 3. the A/B that never ran (also builds the ungated arm to settle the gate question)
bash run_coverage_chain.sh          # waits on row count, builds arms, benchmarks twice, then oracle

# 4. gate before shipping
bash gate_ship.sh <arm>
```

`cold_matrix_summary.json` holds the 5 201 measured rows as (shape → best kernel, best GFlop/s,
current GFlop/s). The full 298-wide matrix was 24 MB and is deliberately not committed: it is a
gfx1100-emulated proxy and should be regenerated natively.

## Two traps that cost real time here

- **A gate that greps for a token the tool never prints fails 100% of the time and looks exactly
  like a real defect.** This gate's first version grepped hipblaslt-bench `-v` output for "PASS"
  and reported **0 pass / 26 fail** on a catalog where every run actually succeeded
  (`norm_error` ~5e-05 vs `rtol` 1e-02). `check_correctness.py` parses the column instead.
- **A control arm must be built on the same base as the comparison baseline.** Re-mapping the
  control on top of `lean` while comparing against `shipped` made held-out rows *revert* shipped's
  re-map rather than hold it constant — 43 of 44 changed queries landed in the control group,
  exactly inverted. Caught by predicting where the effect should appear before measuring.
