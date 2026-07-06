<!-- Copyright (c) Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# rocke.benchmark.perf — performance primitives

Reusable building blocks for measuring rocKE GPU kernels: hardware **counters**
(cycles, cache, waves, instructions, stalls) via `rocprofv3`, static **occupancy**
(VGPR/LDS/…) from a kernel's ELF notes, and **wall** time — composed into one
`measurement/v1` record. The primary metric is cycle-based (`busy_cycles`), which is
clock-invariant and so much less noisy than milliseconds.

## Quick start (for testers)

Everything runs out of `platform/Python` (no build step — rocKE runs Python-only
here). Set the path once:

```
cd dnn-providers/hip-kernel-provider/rocke/platform/Python
export PYTHONPATH=$PWD
```

1) Sanity-check **without a GPU** — run the unit tests:

```
python -m unittest discover -s rocke/benchmark/perf/tests
```

2) Try it **on a GPU** (needs `rocprofv3`). The sweep example is the easiest
   end-to-end: it builds a couple of GEMM variants, profiles each, stores a record,
   and self-checks. Run it twice — the first sets the baseline, the second compares:

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

4) Profile **your own kernel** — hand the tool any launch command that prints a
   `PerfJSON:` line (rocKE's `run_manifest` does):

```
python -m rocke.benchmark.perf.tool profile --arch gfx950 --op gemm \
    --shape '{"M":512,"N":512,"K":512}' --repeats 3 --kernel-name mygemm \
    -- python -m rocke.run_manifest <hsaco> <manifest> --shape 512,512,512
```

Records land in `~/.cache/rocke-perf/` (override with `$ROCKE_PERF_CACHE`). On a
SLURM cluster where the login and compute nodes don't share a home, do the
baseline+compare within a single allocation, or point `$ROCKE_PERF_CACHE` at shared
storage.

---

This directory contains three distinct things. They are separate on purpose; know
which is which:

## 1. The primitives — what ships in rocKE (this package)

Pure, stdlib-only, and **write nothing** — they produce a record and return it.
Import and compose them; every consumer (a developer, an agent, or an external perf
framework) uses these same pieces.

- `schema` — the measurement-record contract + `validate`. Identity is
  `(arch, kernel_name, shape)`; the shape signature is op-agnostic (GEMM `M,N,K`;
  conv/attention dims work the same). This is the seam other tools consume.
- `counters` — probe the PMU counters the GPU actually supports
  (`rocprofv3 --list-avail`) and normalize the arch-specific raw names (RDNA
  wave32/`GL2C_*` vs CDNA wave64/`TCC_*`) to stable names.
- `occupancy` — VGPR/AGPR/SGPR/LDS + a coarse occupancy estimate from an HSACO's
  ELF notes. No GPU required.
- `harness` — profile a kernel-launch command under `rocprofv3` and **return** a
  record (counters + resources + a separate un-profiled wall run).
- `aggregate` — reduce K repeated runs to a median + spread (noise bound).
- `report` — serialize a record, extract the diagnostic panel, and diff two records.

## 2. The benchmarking tool — a dev/agent convenience (`tool/`)

A thin layer that *uses* the primitives so a developer or an agent can keep a local
history and see whether a change improved or regressed a workload. It is the **only**
part that writes, and it writes **outside the repo** (a user cache dir), as **simple
JSON Lines** — nothing more.

- `store` — append/read records in a user cache dir (`~/.cache/rocke-perf`;
  override with `$ROCKE_PERF_CACHE`). Append-only `history.jsonl`.
- `selfcheck` — advisory improve/regress verdict, gated on
  `max(threshold, k·spread)` so run-to-run noise isn't flagged.
- CLI: `python -m rocke.benchmark.perf.tool {profile,occupancy,compare}` (`--json`
  for machine output).

**Scope boundary — the *system* is not here.** Fleet scheduling, central storage,
dashboards, CI orchestration, at-scale analysis, and any **CSV / columnar / export**
storage format are the concern of the external perf framework, which consumes the
same records via the schema. (The harness *reads* `rocprofv3`'s CSV output only as
an input to build a record — this package never produces or stores CSV.)

## 3. The example — a demonstration, not a product surface (`examples/`)

- `examples/profile_gemm_sweep.py` shows how a consumer wires the primitives over
  rocKE's **existing** GEMM sweep (`rocke.benchmark.gemm.fp16_rcr_sweep`, reusing its
  enumeration) to produce one record per variant. Running an at-scale sweep would be
  driven elsewhere; this is only a worked example of the wiring.

## Running

Needs `PYTHONPATH` pointing at `platform/Python`. Live counters need a GPU +
`rocprofv3`; occupancy needs `llvm-readelf`. Without a profiler the primitives
degrade to a wall-only record (and warn), so nothing hard-fails.

```
# measure a kernel-launch command (must print a `PerfJSON:` line for wall metrics)
python -m rocke.benchmark.perf.tool profile --arch gfx950 --op gemm \
    --shape '{"M":512,"N":512,"K":512}' --repeats 3 -- <kernel launch argv...>

# improve/regress from stored history
python -m rocke.benchmark.perf.tool compare --all

# static occupancy from a compiled HSACO (no GPU)
python -m rocke.benchmark.perf.tool occupancy path/to/kernel.hsaco --arch gfx950

# the sweep example
python -m rocke.benchmark.perf.examples.profile_gemm_sweep --arch gfx950 --shape 512x512x512
```

## Per-arch counter coverage

Counter names differ by family, so the harness probes and normalizes them — never
hardcode a counter list. On CDNA (gfx94x/gfx950) the full panel populates. On RDNA4
(gfx1201) the instruction and L2 counters currently read 0 (a `rocprofv3`
limitation), so the panel there is clock/wave-only; the primary cycle metric works
on both. `captured_counters` in each record lists exactly which counters populated,
so a record never overstates coverage.

## Tests

Pure and GPU-free:

```
cd platform/Python && python -m unittest discover -s rocke/benchmark/perf/tests
```
