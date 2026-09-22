<!-- Copyright (c) Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# rocke.benchmark.perf - performance primitives

Reusable building blocks for measuring rocKE GPU kernels: hardware **counters**
(cycles, cache, waves, instructions, stalls) via `rocprofv3`, static **occupancy**
(VGPR/LDS/...) from a kernel's ELF notes, plus real-world (**wall**) and profiled timing - composed into one
`measurement/v1` record. The primary metric is cycle-based (`busy_cycles`).
Compare repeated runs under comparable clocks, cache state and system load;
cycle counts and the spread-based comparison gate do not eliminate measurement bias.

## Quick start (for testers)

The package runs directly from `platform/python` without an installation step. Set
the path once:

```
cd dnn-providers/hip-kernel-provider/rocke/platform/python
export PYTHONPATH=$PWD
```

1) Sanity-check **without a GPU** - run the unit tests:

```
python -m unittest discover -s rocke/benchmark/perf/tests
```

2) Try it **on a GPU** (needs `rocprofv3`). The sweep example is the easiest
   end-to-end: it builds a couple of GEMM variants, profiles each, stores a record,
   and self-checks. Run it twice - the first sets the baseline, the second compares:

```
python -m rocke.benchmark.perf.examples.profile_gemm_sweep --arch gfx950 --shape 512x512x512
python -m rocke.benchmark.perf.examples.profile_gemm_sweep --arch gfx950 --shape 512x512x512
```

   First run prints `[no baseline]` per variant; the second prints `[within noise]`
   (or an improve/regress verdict if something changed). On CDNA you'll see the full
   counter panel; on RDNA4 (gfx1201) expect clock/wave counters only.

3) Inspect the stored history any time:

```
python -m rocke.benchmark.perf.tool compare --all
```

4) Profile **your own kernel** - hand the tool any launch command:

```
python -m rocke.benchmark.perf.tool profile --arch gfx950 --op gemm \
    --shape '{"M":512,"N":512,"K":512}' --repeats 3 --kernel-name mygemm \
    -- python -m rocke.run_manifest <hsaco> <manifest> --shape 512,512,512 --verify
```

   The command needs no special support: counters and kernel-level timing come from
   the profiler. A launcher that additionally prints a `PerfJSON:` line contributes
   real-world (un-profiled) wall time and TFLOPS/GB/s, which the profiler cannot
   supply - `rocke.run_manifest` does, and any script can with one call:

```python
from rocke.benchmark.perf import perfjson
perfjson.emit(ms=ms, tflops=tflops, gbps=gbps)
```

   Each record's `timing_source` says which was used (`perfjson` or
   `rocprofv3_duration`), so a record never overstates what it measured. With
   neither a profiler nor a `PerfJSON:` line there is nothing to measure and the
   command fails rather than storing an empty record.

Records land in `~/.cache/rocke-perf/` (override with `$ROCKE_PERF_CACHE`). On a
SLURM cluster where the login and compute nodes don't share a home, do the
baseline+compare within a single allocation, or point `$ROCKE_PERF_CACHE` at shared
storage.

---

This directory contains three distinct things. They are separate on purpose; know
which is which:

## 1. The primitives - what ships in rocKE (this package)

Stdlib-only and do not persist records - they produce a record and return it.
Import and compose them; every consumer uses these same pieces.

- `schema` - the measurement-record contract + `validate`. Identity is
  `(arch, op, kernel_name, shape)`; the shape signature is generic (GEMM `M,N,K`;
  conv/attention dims work the same). This is the seam other tools consume.
- `counters` - probe the PMU counters the GPU actually supports
  (`rocprofv3 --list-avail`) and normalize the arch-specific raw names (RDNA
  wave32/`GL2C_*` vs CDNA wave64/`TCC_*`) to stable names.
- `occupancy` - VGPR/AGPR/SGPR/LDS + a coarse occupancy estimate from an HSACO's
  ELF notes. No GPU required. The occupancy model follows the ISA recorded in the
  code object (returned as `target_arch`); the `arch` argument is only the fallback
  when the notes carry no target.
- `perfjson` - emit/parse the `PerfJSON:` launcher line. Optional for the launcher,
  but the only way to contribute wall time and throughput to a record.
- `harness` - profile a kernel-launch command under `rocprofv3` and **return** a
  record: counters + resources + `profiled` timing (the profiled run's own kernel
  duration, or its `PerfJSON:` timing when the launcher emits one) + a separate
  un-profiled `wall` run when the launcher can be timed that way. Options:
  `warmup=N` drops the launcher's cold warmup dispatches from the counter medians
  (and from the duration); `per_dispatch=True` also emits raw per-dispatch counters
  and `duration_ns` (`counter_samples`) for downstream profiling.
  `artifacts_dir=Path(...)` optionally retains the original profiler workspace in
  a new caller-owned directory; existing destinations are refused before launch.
  The primitive still returns its record; the CLI owns JSON bundle persistence.
- `aggregate` - reduce K repeated runs to a median + spread (noise bound).
- `report` - serialize a record, extract the diagnostic panel, and diff two records.

## 2. The local benchmarking tool (`tool/`)

A thin layer that *uses* the primitives so a developer can keep a local history and
see whether a change improved or regressed a workload. It persists history as
**JSON Lines** outside the repo and optionally exports a portable artifact bundle.

- `store` - append/read records in a user cache dir (`~/.cache/rocke-perf`;
  override with `$ROCKE_PERF_CACHE`). Append-only `history.jsonl`.
- `selfcheck` - advisory improve/regress verdict, gated on
  `max(threshold, k*spread)` so run-to-run noise isn't flagged.
- CLI: `python -m rocke.benchmark.perf.tool {profile,occupancy,compare}` (`--json`
  for machine output).

**Scope boundary:** GPU scheduling, fleet orchestration, central storage and
dashboards remain external. Artifact export preserves profiler-owned CSV bytes;
it does not reconstruct CSV from normalized medians or depend on WaveScope.

### Portable artifacts: counter CSVs and measurement JSON

For the WaveScope workflow, use
[`capture_wavescope_pmc.py`](../../../../dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_pmc.py)
and its [usage guide](../../../../dsl_docs/optimization/utilities/tools/wavescope/README.md#capture-pmc-evidence-csv-and-json).
That utility chooses the adjacent perf package, defaults to export-only, and prints
CSV import paths plus the JSON entry point. The generic CLI option below is its
supporting mechanism; it is also available to other perf consumers.

```bash
python3 -m rocke.benchmark.perf.tool profile \
  --arch gfx950 --op gemm --shape '{"M":512,"N":512,"K":512}' \
  --kernel-name my_gemm --match-kernel my_gemm --repeats 3 --warmup 5 \
  --per-dispatch --artifacts-dir /tmp/my-gemm-before \
  -- python3 run_kernel.py
```

Use the actual architecture, operation, shape, dispatched symbol and warmup count.
The destination must not exist. Each repeat gets an isolated profiler workspace:

```text
my-gemm-before/
  manifest.json                  rocke.bench.artifacts/v1
  measurement.json               aggregate rocke.bench.measurement/v1
  comparison.json                existing selfcheck result
  samples/0000/
    measurement.json             individual measurement/v1 + profile_capture
    raw/pmc.txt                  original counter recipe
    raw/prof/.../*counter_collection.csv
  samples/0001/...
```

The manifest is the entry point for consumers. It records lifecycle status,
requested samples, sample indices/run IDs, warmup selection, relative artifact
paths, byte sizes and SHA-256 hashes (hex without a prefix). `files[].kind`
distinguishes `pmc_csv`, `counter_config`, `measurement`, `comparison`, and other
`profiler_output`. Paths remain valid when the bundle is moved. Each sample's
`profile_capture` records the normalized-to-raw counter map, requested replay
groups, selection and profiler status (`complete`, `failed`, or `unavailable`).

`manifest.status=complete` means measurement export finished, not that all
counters were available or performance improved. Check per-sample profiler
status and `captured_counters`. A profiler failure can still yield a complete
wall-only export; its retained CSVs are troubleshooting evidence, not import candidates.
Export failures retain partial files and a failed manifest; interrupted processes
may leave `running`, which is not complete.
Failed exports must not be consumed as finalized baselines. A regression still
exports a complete bundle and exits 1. `--no-store` suppresses history writes,
not an explicitly requested bundle. Without the flag, behavior is unchanged.

**WaveScope CSV import:** pass an existing ATT dispatch folder to the utility with
`--trace-dir` and keep `--output-dir` as its sibling (for example,
`$(dirname "$TRACE_DIR")/pmc_bundle`). The utility copies every replay-pass CSV
from the first successful repeat beside `code.json` using distinct names ending in
`_counter_collection.csv`. WaveScope discovers and merges those top-level files
when it opens the trace folder. Keeping the bundle outside the dispatch directory
prevents browser folder import from counting retained repeats a second time.
Existing sidecars are never overwritten.

Without `--trace-dir`, manually upload a recommended CSV printed by the utility.
Each upload replaces the previous one. Keep repeats separate. Confirm workload,
GPU and binary before correlating separate PMC and ATT captures; the bundle records
their association as UNBOUND.

**JSON readers:** consume `measurement.json` using its existing schema, the sample
records and the manifest. They retain normalized counter medians, sample count,
spread, correctness and separate `wall`/`profiled` timing with `timing_source`.
`--per-dispatch` adds samples with repeat and counter-pass identity. Consumers use
the versioned manifest and measurement schemas to interpret the bundle.
WaveScope's Bottlenecks visualizer does not derive PMC state from this JSON; it
uses the colocated or manually uploaded counter CSVs. The JSON remains the
machine-readable measurement and verdict contract for agents and other consumers.

**Do not equate CSV and JSON totals.** Raw CSVs retain all profiler dispatches,
including warmup and other kernels. JSON counter medians select one target and
drop warmup per pass. Optional `counter_samples` retain warmup for inspection.
Raw PMC import sums and JSON medians therefore need not agree. Ratios use the
same definitions, but a ratio of medians can differ from a ratio of raw sums.
The selected counters include LDS-conflict and CDNA VALU/MFMA diagnostic inputs;
full roofline analysis requires additional counters. Raw profiler files may
contain kernel names, paths or workload-specific information.

## 3. The GEMM sweep integration (`examples/`)

- `examples/profile_gemm_sweep.py` drives the primitives over rocKE's **existing**
  GEMM sweep (`rocke.benchmark.gemm.fp16_rcr_sweep`, reusing its enumeration) to
  produce one record per variant. It ships in the repo as the reference for wiring
  the primitives over a sweep. What it does NOT do is the *system* work - choosing
  which GPUs run the sweep, scheduling, running it at scale, and storing mass
  results - that stays with the external perf framework.

## Running

Needs `PYTHONPATH` pointing at `platform/python`. Live counters need a GPU +
`rocprofv3`; occupancy needs `llvm-readelf`. Without a profiler, profiling degrades
to a wall-only record and warns. A failed kernel command still fails the measurement.

```
# measure a kernel-launch command (a `PerfJSON:` line adds wall metrics; optional)
#   --warmup N     drop the launcher's N warmup dispatches from the counter medians
#   --per-dispatch also emit raw per-dispatch counters (counter_samples)
python -m rocke.benchmark.perf.tool profile --arch gfx950 --op gemm \
    --shape '{"M":512,"N":512,"K":512}' --repeats 3 --warmup 5 -- <kernel launch argv...>

# improve/regress from stored history
python -m rocke.benchmark.perf.tool compare --all

# static occupancy from a compiled HSACO (no GPU)
python -m rocke.benchmark.perf.tool occupancy path/to/kernel.hsaco --arch gfx950

# the sweep example
python -m rocke.benchmark.perf.examples.profile_gemm_sweep --arch gfx950 --shape 512x512x512
```

## Per-arch counter coverage

Counter names differ by family; the harness intersects its normalized selection
with `rocprofv3 --list-avail`. The full selection contains 14 counters on CDNA
and 11 on RDNA. Per-block budgets split CDNA into two replay passes and RDNA into
one. Overlapping ratio constraints form indivisible groups; a group that exceeds
its block budget is rejected. Availability can reduce the selected set.

Derived ratios are `busy_cycles / total_clocks`, `l2_hit / (l2_hit + l2_miss)`,
`lds_bank_conflict / lds_idx_active`, `valu_active_cycles / cu_busy_cycles`, and
`mfma_insts / valu_insts`. Missing inputs or unusable denominators omit the ratio.
Values are ratios, not percentages, and are not universally bounded by one.

On 2026-09-15, gfx90a captures produced nonzero LDS, L2 and VALU diagnostics in
two passes. gfx1201 captures completed in one pass but returned zero for the LDS,
instruction and L2 inputs exercised by the probe; `busy_fraction` remained usable.
These are observations of those machines and profiler installations, not a
guarantee for every GPU in either family. `captured_counters` lists returned
numeric keys, including zeros; it does not certify meaningful hardware support.

Each pass reruns the launcher. Keep inputs, launch geometry and workload state
repeatable. Ratio inputs share a requested pass, but separate replays and repeated
runs can still differ in cache state, scheduling, clocks or data-dependent work.
`lds_insts` is useful for comparing executed LDS work; it is an instruction count,
not bytes transferred or a guarantee of deterministic behavior.

Hardware CSV import was exercised in WaveScope using explicitly synthetic trace
fixtures. This verifies parsing and PMC rule behavior, not real ATT/PMC correlation.
A vector-only fixture and a CSV control with the MFMA counter removed isolate
the counter-based matrix guard from the ATT fallback.

## Tests

Pure and GPU-free:

```
cd platform/python && python -m unittest discover -s rocke/benchmark/perf/tests
```
