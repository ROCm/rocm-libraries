# Repeat the StreamK grid study on a new SKU — context + prompt

Paste **§B** to Claude as the task. **§A** is the background it needs; either paste it too
or point at this file. Fill in the `<<< >>>` placeholders first.

---

# §A — Context

## What was done before

A full StreamK launch-grid study was run on gfx1100 (RX 7900 XTX), 69,000 benchmark runs.
Findings: `docs/STREAMK_GRID_FINDINGS.md`. Read it before starting — most of the method
transfers unchanged and several traps will cost you hours if rediscovered.

The short version:

* `TENSILE_STREAMK_DYNAMIC_GRID` defaults to **6** = `k_split_aware`. Not opt-in.
* That selector's **only hardware input is `cu_count` = Origami's `hardware.N_CU`**.
  Everything else is a hardcoded literal with no architecture guard.
* On RDNA, `hardware.cpp` scales `multiProcessorCount` by 2 because HIP reports WGPs, not
  CUs. On CDNA the factor is 1.
* On gfx1100 the shipped ×2 (N_CU = 96) **beat ×1, ×3, ×4 and ×6**. 96 is the first value
  that fills all 48 WGPs at occupancy 2.
* StreamK is **inert on 77.5%** of shapes there — and measurement showed that is the correct
  choice, not a defect.

## The hypothesis to test

The gfx1100 optimum (×2) equals *physical CUs*, which on that part also equals
*WGPs × achievable occupancy* (48 × 2). Those two readings are indistinguishable on gfx1100
because both give 96.

**A SKU that sustains more concurrent workgroups per WGP/CU should therefore prefer a larger
multiplier** — because the grid must be **co-resident** (the fixup spin-wait requires peers
running alongside the owner, not queued behind it), so the useful budget is
"workgroups that fit at once", not "cores".

Be precise about the mechanism, because it is easy to get wrong: raising the multiplier does
**not** make StreamK stream more (inert share was flat at 73–78% across ×1…×6 on gfx1100).
It makes the **launch bigger**. It pays only if those extra workgroups are genuinely
resident and hide latency. So the prediction is:

> optimal multiplier ≈ concurrent workgroups per multiProcessorCount unit

Higher WMMA throughput per CU matters only indirectly — it shifts the compute/memory balance
and can change how much latency there is to hide. **Measure occupancy on the target SKU
before assuming a bigger multiplier will win.** If occupancy is still 2, expect ×2 again.

## Machinery that already exists and should be reused, not rewritten

In `~/hhs_tn_grid_vs_resource_origami_9k/harness/`:

| file | what it does |
|---|---|
| `streamk_env_ab.py` | interleaved A/B where an arm is a **(library, env) pair**; contract-hash group-level resume; killpg on timeout; parses `skGrid`/`skTiles`/`SKItersPerWG` out of `TENSILE_DB=0x40`; `--rotating/--workspace/--limit-file` passthroughs; per-arm `::--flag val` extra args |
| `streamk_contract.py` | run identity — hashes bench sha, **libhipblaslt.so sha**, per-arm library+kernel-object shas, env dicts, arm order, iteration-tier source |
| `run_campaign.py` | declarative phase driver: deadline arbitration (reduces scope, never skips), per-phase hard cap, atomic ledger, telemetry |
| `analyze_campaign.py` | census table, integrity gates, banded paired contrasts with bootstrap CI, null/live partition |
| `validate_grid_model.py` | compares a Python transcription of the selector against observed grids |

---

# §B — The task

## Goal

Determine the optimal Origami `N_CU` multiplier for **`<<<ARCH, e.g. gfx1201>>>`**, and
whether the shipped StreamK grid selector is well-calibrated there. Same method as the
gfx1100 study in `docs/STREAMK_GRID_FINDINGS.md`.

## Environment to fill in

```
ARCH                <<< gfx1201 >>>
GPU                 <<< e.g. RX 9070 XT >>>
hipBLASLt tree      <<< /path/to/checkout — MUST be the tree the bench binary was built
                        from; confirm via build/release/CMakeCache.txt CMAKE_HOME_DIRECTORY >>>
bench binary        <<< .../build/release/clients/hipblaslt-bench >>>
libraries           <<< .../Tensile/library/<ARCH>  — need at least one with StreamK
                        kernels; check the logic YAML for `StreamK: 3` >>>
shape set           <<< reuse state/evaluation_shapes.csv, or generate one for this SKU >>>
```

## Step 0 — establish the ground truth of the part (do this first, it is cheap)

1. Print the device properties: `multiProcessorCount`, `gcnArchName`, `sharedMemPerBlock`,
   `l2CacheSize`. **Do not assume `multiProcessorCount` means CUs** — on RDNA it is WGPs.
2. Find `cus_per_multiProcessorCount` in `shared/origami/src/origami/hardware.cpp` and
   confirm what factor this arch gets (RDNA → 2, CDNA → 1). Compute the resulting `N_CU`.
3. **Measure achievable occupancy** — workgroups resident per multiProcessorCount unit, for
   the macro-tiles the catalog actually uses. This is the quantity the hypothesis predicts
   the optimum tracks. Occupancy is bounded by LDS, VGPRs and the 128 B/instruction limits;
   `CUOccupancy` in the logic YAML is usually `-1` (clamped to 1) and is **not** the answer.
4. Confirm `TENSILE_STREAMK_DYNAMIC_GRID`'s default in `include/Tensile/AMDGPU.hpp` for this
   tree — it has moved before (`0 → 3 → 6`).

## Step 1 — census before timing

Verify `TENSILE_DB=0x40` prints `skGrid`/`skTiles`/`SKItersPerWG` on this build, then run a
census over all shapes × all grid modes at `--iters 1 --cold_iters 0 --rotating 0`. This is
~0.25 s per run and has **no statistical uncertainty**.

Report, per mode: inert share by shape count **and** by kernel time, median grid, and the
distribution of grid vs the WGP/CU count.

Modes: `DYNAMIC_GRID` ∈ {0,1,2,3,4,5,6,7}. Remember **0 bypasses Origami entirely** and
**7 is out of enum range and returns `cu_count`** — both are needed.

Sanity gate: mode 4 (`data_parallel`) must measure **100% inert**. If it does not, the inert
detector is wrong and nothing downstream is trustworthy.

## Step 2 — patch the multiplier to be sweepable

One rebuild instead of one per value. In `shared/origami/src/origami/hardware.cpp`, make
`cus_per_multiProcessorCount` read an env var, **defaulting to the current value so unset
behaviour is byte-identical to stock**:

```cpp
static const char* mult_env = std::getenv("ORIGAMI_RDNA_CU_MULT");
static const long  mult_override = mult_env ? std::atol(mult_env) : 0;
...
    return mult_override > 0 ? static_cast<size_t>(mult_override) : 2;   // 2 = stock for RDNA
```

Keep the original as `hardware.cpp.pre-mult-experiment`. Origami is a **static library**
(`liborigami.a`), so `make hipblaslt` relinks in about a second — do not do a full rebuild.

**Verify before sweeping:** with the variable unset, a known shape must produce exactly the
same `skGrid` as before the patch.

`resolve_num_cus` only accepts *reductions*, so `--sm_count_target` cannot raise the budget —
the patch is the only way up. (And `--sm_count_target` also reaches kernel selection, so it
is not a clean grid-only knob.)

## Step 3 — sweep

Two passes over the same arms, `ORIGAMI_RDNA_CU_MULT` ∈ {1, 2, 3, 4, 6} (extend upward if
occupancy from Step 0 suggests it), all with `DYNAMIC_GRID=6`:

* **census** — `--fixed-iters 2 --rotating 0`, plus `TENSILE_DB=0x40` on every arm
* **throughput** — `--min-iters 200`, interleaved

```bash
cd ~/hhs_tn_grid_vs_resource_origami_9k
SK=<<<library>>> ; B=<<<bench>>>
ARMS=""; for M in 1 2 3 4 6; do
  ARMS="$ARMS mult${M}=$SK:ORIGAMI_RDNA_CU_MULT=$M,TENSILE_STREAMK_DYNAMIC_GRID=6"; done
python harness/streamk_env_ab.py --bench $B --min-ms 0.0 --reps 1 --min-iters 200 \
  --out measurements/campaign/mult_perf.csv --arms $ARMS
```

Also run **`DYNAMIC_GRID=4`** (data-parallel) as an arm — it is the controlled
StreamK-off control with identical kernel binaries.

## Step 4 — analyse

* Reference = the shipped multiplier. Report **banded** geomeans (`<0.1 ms`, `0.1–1 ms`,
  `1–5 ms`, `≥5 ms`) plus ALL, with bootstrap CI. A single number will hide a reversal —
  on gfx1100 ×3/×4 won below 0.1 ms and lost in the 0.1–1 ms band.
* Partition every contrast by the **observed census** into inert vs streaming. The inert
  set must score ~1.000 for any grid-only arm; **its spread is your measured noise floor.**
  Quote it, and judge every other difference against it.
* Report the grid distribution against the co-residency thresholds for this part
  (`< WGP count` = idle, `1× ` = half occupancy, `≥ occupancy × WGP count` = packed).

## Non-negotiable protocol

1. **One binary** for all arms; swap libraries via `HIPBLASLT_TENSILE_LIBPATH`. Different
   binaries confound the selector with the code.
2. **One process per arm** — the `TENSILE_STREAMK_*` vars latch in function-local statics.
3. **Interleave with arm-order rotation.** Sequential A/B drifts past the effect size.
4. **Check the SK-mode mix (`_SK<n>_` in the kernel name) before reading any throughput.**
   A short workspace makes `getSKGrid` silently revert to data-parallel; you can benchmark a
   DP kernel all day and call it StreamK.
5. **Selection agreement between env-only arms must be 100%** — `DYNAMIC_GRID` provably
   cannot reach kernel selection. Less than 100% means nondeterminism or a library mismatch,
   and invalidates the comparison.
6. Never trust an offline transcription of the selector over the census.

## Deliverable

A report in the shape of `docs/STREAMK_GRID_FINDINGS.md`: the census table, the integrity
gates, the banded multiplier comparison with CIs, and a direct answer to —

> **Is the shipped multiplier optimal on this SKU, and if not, what is, and by how much?**

Plus, explicitly: **does the optimum track achievable occupancy?** That is the hypothesis.
If the optimum on this part is again `WGPs × occupancy`, the rule generalises and can be
proposed as a source change. If it is not, say so plainly — a refuted hypothesis stated
clearly is worth more than a hedged one.
