# StreamK on gfx1100 — full runbook

Everything from the 2026-08-18/19 investigation: what was asked, what was found, how it was
measured, what was wrong along the way, and how to pick it up again.

Written so a reader who was not in the conversation can reproduce or continue it.

---

## 1. Where everything lives

| thing | path |
|---|---|
| **The tree that was built and benchmarked** | `~/exp/stock` (and `~/exp/tuned`, identical StreamK source) |
| bench binary | `~/exp/stock/build/release/clients/hipblaslt-bench` |
| SK3 library (192 solutions, **all StreamK:3**) | `~/exp/stock/build/release/Tensile/library/gfx1100` |
| Grid library (298 solutions, **all SK0**) | `~/exp/devlib_stock_grid/library/gfx1100` |
| v6 library (58 solutions, **22 SK3 + 36 SK0**) | `~/exp/devlib_stock_v6/library/gfx1100` |
| harness + ledgers | `~/hhs_tn_grid_vs_resource_origami_9k/` |
| deck (Streamlit + pptx) | `~/streamk_presentation` — `./run_app.sh`, port 8026 |
| docs | `~/streamk_presentation/docs/` |

**`~/rocm-libraries` was never built here.** An early draft cited it and got the CU-count
story backwards as a result. `config.py` now points at `~/exp/stock`, and `verify.py`'s
citation gate pins every line number to that tree.

Ledgers, all in `measurements/campaign/` — **79,500 successful measurements**:

| file | rows | what |
|---|---|---|
| `p0b_census.csv` | 12,000 | 1500 shapes × 8 grid modes, observed launch grid |
| `p1_modes.csv` | 18,000 | 6 grid modes × 2 reps, timed |
| `p2_cumult.csv` | 18,000 | pinned grids {48,96,192,288} + sm_count_target |
| `p3_catalog.csv` | 6,000 | SK3 / SK3-DP / grid / v6 catalogs |
| `p5_mult_census.csv` | 7,500 | `ORIGAMI_RDNA_CU_MULT` ∈ {1,2,3,4,6}, observed grid |
| `p5_mult_perf.csv` | 7,500 | same, timed |
| `p6_smallcap.csv` | 6,000 | small-branch **ceiling** only |
| `p7_thresh.csv` | 4,500 | branch **threshold** only — **INCOMPLETE, stopped at ~1200/1500 shapes** |

---

## 2. The question that started it

A reviewer raised four claims about the gfx1100 StreamK work:

1. "The grid predictor hasn't been validated for performance on this architecture, that
   selects the number of workgroups to launch for a streamk kernel."
2. "Streamk+grid has most of the drawbacks of DP+grid, especially if you're using a grid of
   similar density."
3. "WGM is predicted by streamk and may need to be tweaked for this architecture."
4. "The tile selection (the main thing origami is responsible for) may not be doing well on
   this architecture."

---

## 3. How the grid is actually chosen

Call chain, all in `~/exp/stock`:

| # | file:line | what |
|---|---|---|
| 1 | `tensilelite/include/Tensile/AMDGPU.hpp:289` | reads `TENSILE_STREAMK_DYNAMIC_GRID`, **default 6** |
| 2 | `tensilelite/src/ContractionSolution.cpp:3161` | `solve()` calls `getSKGridImpl` |
| 3 | `…:3934` | `else if (skDynamicGrid > 0)` → into Origami |
| 4 | `…:4026` | `origami::streamk::select_grid_size` |
| 5 | `origami/src/origami/streamk.cpp:443` | mode switch → `:474` `grid_k_split_aware` |
| 6 | **`origami/src/origami/streamk.cpp:335-441`** | **the calculation** |
| 7 | `…ContractionSolution.cpp:1764` | `rv.numWorkGroups.x = sk.grid` — becomes the launch |

`N_CU = 96` from `origami/src/origami/hardware.cpp:139` (`multiProcessorCount × 2` on RDNA).

### The algorithm

```
tiles = ceil(M/MT_M) * ceil(N/MT_N) * batch
sk_grid = tiles                                   // fallback

if tiles > cu_count:            // BIG — shrink the grid to a clean fraction of tiles
    min_even = max(1, tiles / cu_count)
    for frac in {0, ½, ⅛, ⅕, ¼, ⅓}:               // this order, NOT sorted
        g = round(tiles / (min_even + frac))
        skip if MT_M*MT_N*4*g > 128 MB            // hardcoded guard
        skip if the k-iteration FRAGMENT misses a 128 B cache line
        take the first g <= cu_count

elif tiles < cu_count:          // SMALL — split K instead
    for f in {16,12,8,6,4,3,2,1}:
        take first f with  tiles*f <= cu_count  AND  itersPerTile/f >= 8
    sk_grid = tiles * f

else:                           // tiles == cu_count exactly
    sk_grid = tiles                               // neither branch runs

if tiles % sk_grid != 0 and MT_M*MT_N*4*sk_grid > workspace: sk_grid = tiles
```

**The only hardware input is `cu_count`.** `MinItersPerCU = 8`, the fraction lists, 128 MB
and 128 B are all hardcoded with **no architecture guard**.

### One launch serves both phases

There is a single 1-D dispatch of `skGrid` workgroups. In mode 3 each workgroup runs its
data-parallel tiles (round-robin, stride = `skGrid`) *and* its streamed slice inside that
one launch. Measured: 980 tiles, grid 184 → 244 streamed + 736 DP at 4 per workgroup.

### `fragment` — the concept everything turns on

```
iters_per_cta = ceil(tiles * itersPerTile / grid)   // each WG's equal share
fragment      = iters_per_cta % itersPerTile        // the bit that is not a whole tile
```

`fragment == 0` → no workgroup straddles a tile boundary → no tile has two contributors →
**no fixup, no streaming**. The fragment *is* the StreamK-ness of a launch, and it is also
the ragged memory access the cache-line filter measures.

---

## 4. The measurement trick

`TENSILE_DB=0x40` prints the packed kernel arguments:

```
[120..123] itersPerTile:  80 00 00 00 (128)
[132..135] SKItersPerWG:  80 00 00 00 (128)
[136..139] skGrid:        00 04 00 00 (1024)
[140..143] skTiles:       00 04 00 00 (1024)
```

So the grid is **read**, not inferred. At `--iters 1 --cold_iters 0 --rotating 0` that is
~0.25 s per run with **zero statistical uncertainty**. 12,000 census runs took 48 minutes
and told us more than the 18,000 timed runs that followed.

**Do this first on any new architecture.**

---

## 5. Results

### 5.1 The predictor leaves StreamK inert on 77.5% of shapes

Census, 1500 shapes × 8 modes. Inert = `skTiles == skGrid` or `SKItersPerWG == itersPerTile`.

| `DYNAMIC_GRID` | inert (shapes) | inert (kernel time) | median grid |
|---|---|---|---|
| 0 legacy (`computeUnitCount` = 48) | 36.6% | 22.8% | 48 |
| 1 min_resources | 68.9% | 26.2% | 96 |
| 2 energy_aware | 75.7% | 37.4% | 87 |
| 3 reduction_cost_aware | 80.6% | 40.0% | 82 |
| 4 data_parallel | 100.0% | 100.0% | 96 |
| 5 analytical | 79.3% | 97.7% | 96 |
| **6 k_split_aware (shipped)** | **77.5%** | **64.9%** | 96 |
| 7 number_of_cus (= N_CU = 96) | 23.5% | 20.2% | 96 |

Mode 4 at exactly 100% is the gate proving the inert detector is right.

### 5.2 …and nothing beats it

Paired geomean vs the shipped default, 1500 shapes, identical kernels
(selection agreement 100%, as the code requires):

| arm | geomean |
|---|---|
| `m0` legacy grid = 48 | **78.7%** |
| `m1` min_resources | 91.4% |
| `m3` reduction_cost_aware | 92.3% |
| `m4` StreamK off | 97.0% |
| `m7` number_of_cus = 96 | 92.5% |

**The legacy default — still what upstream Tensile ships — is 21% slower.**

### 5.3 The falsifiable test: why both are true

Partition by the **observed** census:

| contrast | inert shapes | streaming shapes |
|---|---|---|
| `m4` (force DP) vs `m6` | **99.11%**, median abs dev **0.48%** | **89.9%** |

Where mode 6 declines to stream, forcing DP changes nothing. Where it streams, turning
StreamK off costs 10%. **Correct in both directions.**

That 0.48% is the campaign's **measured noise floor** — half the ~1 pt we had assumed,
obtained free from shapes identical by construction.

### 5.4 StreamK only pays below ~1 ms

| band | n | StreamK off vs on |
|---|---|---|
| <0.1 ms | 1594 | **93.9%** — StreamK wins |
| 0.1–1 ms | 1102 | 100.2% |
| 1–5 ms | 258 | **101.9%** — StreamK loses |
| ≥5 ms | 36 | **101.7%** |

### 5.5 The CU multiplier — ×2 is the peak

RDNA scales `multiProcessorCount` by 2 (`hardware.cpp:139`). Sweeping the **adaptive**
predictor with different budgets:

| N_CU | meaning | <0.1 ms | 0.1–1 ms | ≥1 ms | ALL | 95% CI |
|---|---|---|---|---|---|---|
| 48 | WGPs (×1) | 95.9 | 96.2 | 97.8 | **96.2** | [95.5, 96.9] |
| **96** | **physical CUs (×2, shipped)** | — | — | — | **100** | — |
| 144 | ×3 | **101.2** | 96.6 | 97.1 | 99.1 | [98.2, 100.0] |
| 192 | SIMD32 lanes (×4) | **100.8** | 96.9 | 98.6 | 99.1 | [98.3, 100.0] |
| 288 | ×6 | 100.3 | 96.8 | 99.1 | 98.9 | [98.1, 99.8] |

Peak at ×2, degrading both ways. **The SIMD count is not the better denominator.**

The threshold that matters is **48, not 96** — the grid counts *workgroups* and a workgroup
occupies a whole WGP:

| grid | consequence | shapes @×2 | shapes @×1 |
|---|---|---|---|
| < 48 | some WGPs idle | 178 | **462** |
| 48–95 | all WGPs busy, half occupancy | 460 | 472 |
| ≥ 96 | 2 WGs per WGP, packed | 862 | 566 |

×1 loses by running the machine at **half occupancy**, not by idling cores. Inert share is
flat across multipliers (73–78%) — the factor changes grid **magnitude**, not streaming.

### 5.6 The real lever is the threshold, not the ceiling

`cu_count` does two jobs: it is the **branch boundary** and the **per-branch grid ceiling**.
Three experiments separate them, measured on small shapes (`tiles < 96`):

| experiment | ×2 | ×3 | ×4 |
|---|---|---|---|
| both raised (§5.5) | — | **100.7%** | **100.4%** |
| **ceiling only** (`ORIGAMI_SMALL_CU_MULT`) | **97.9%** | 97.1% | 96.7% |
| threshold only (`ORIGAMI_THRESHOLD_MULT`) | 100.1% | 99.8% | — |

**Raising the ceiling alone makes small shapes worse, monotonically.** So the gain in §5.5
came from **reclassifying mid-size shapes** (`96 < tiles < 192`) into the K-splitting
branch, not from giving small problems bigger grids.

Threshold-only shows ~no effect on `tiles < 96` — **which is expected**, since those shapes
were already in the small branch. **The band it should move is `96 ≤ tiles < threshold`, and
that analysis was never run** (p7 stopped at ~1200/1500 shapes). *This is the open thread.*

### 5.7 Small problems: StreamK mostly declines

681 shapes have `tiles < 96`:

| split factor `f` | shapes | median itersPerTile |
|---|---|---|
| **1 — no split, pure DP** | **514 (75%)** | **2** |
| 2 | 37 | 64 |
| 8 | 30 | 64 |
| 3–33 | 100 | 128–264 |

**79% of the no-split shapes have `itersPerTile < 16`**, so even `f = 2` violates
`itersPerTile/f ≥ 8`. StreamK **cannot rescue a small shallow GEMM** — there is no K to
spread. Only a smaller macro-tile can, i.e. tile selection.

`MinItersPerCU = 8` decides 514 of 1500 shapes, is un-derived, has no arch guard, and is
not reachable by any env var.

### 5.8 The cache-line filter is a DepthU filter in disguise

```cpp
a_contig = (transA == T) ? fragment * DepthU * bytes_a : 0;   // K contiguous for A
b_contig = (transB == N) ? fragment * DepthU * bytes_b : 0;
reject if either is non-zero and not a multiple of 128
```

**Global** loads, not LDS. Fires on **18.4%** of candidate grids, entirely set by DepthU
(fp16 needs `fragment × DepthU ≡ 0 mod 64`):

| DepthU | vetoed | fragment must be a multiple of |
|---|---|---|
| 16 | 26% | 4 |
| 32 | **43%** | 2 |
| 64 | **0%** | 1 |
| 128 | 0% | 1 |

Dead code at DepthU ≥ 64. At fp8 the threshold doubles and it would fire far more often.

### 5.9 Two workspace checks, one of them fictional

| line | check | against |
|---|---|---|
| `streamk.cpp:387` | candidate filter | **hardcoded 128 MB** |
| `streamk.cpp:438` | final clamp | the actually-allocated workspace |

The 128 MB has no derivation. It is gated on `tiles % grid != 0`, so an *inert* grid can
reserve unlimited workspace unchallenged. On our data it essentially never fires.

---

## 6. The four claims, answered

| # | claim | verdict |
|---|---|---|
| 1 | grid predictor not validated | **ANSWERED — now validated, and it wins.** Residual: the default's provenance is unjustified in-tree (commit `947ed1ef87`, one line, empty body, default moved `3 → 6`). |
| 2 | StreamK+grid inherits DP+grid's drawbacks | **CONFIRMED by our own data.** `grid_sk0` and `v6_stock` agree with the SK3 catalog on **0%** and **24.1%** of kernel choices yet land within 3 points on aggregate throughput. |
| 3 | WGM predicted by StreamK | **RETIRED by a grep.** The gate needs `workGroupMapping == 0`; all 192 shipped SK3 solutions are 1/4/8. Zero GPU time. |
| 4 | tile selection weak | **Partially addressed.** 62 distinct macro-tiles in use; catalog choice moves throughput ~3 points while changing the kernel on 76–100% of shapes — top-1 accuracy is a weak proxy. Clean oracle test un-run. |

---

## 7. Mistakes made, and what caught them

Worth reading — each cost real time.

1. **Cited the wrong source tree.** `~/rocm-libraries` was never built here; the benchmarks
   come from `~/exp/stock`. This produced a completely wrong headline ("our Origami lacks
   the RDNA ×2 fix" — it has it). *Caught by:* checking `CMakeCache.txt`
   `CMAKE_HOME_DIRECTORY` when the user asked "what is my implementation?".
   **Lesson: verify the tree, not just the code.**
2. **Assumed `tiles % skGrid != 0` was sufficient for streaming.** It is necessary but not
   sufficient — a tile with `itersPerTile == 1` cannot be split however you slice it.
   *Caught by:* cross-checking two independently written modules; 39/400 disagreed and every
   one had `ipt == 1`.
3. **Used a nominal `128x128x32` macro-tile for offline analysis.** The real mix spans
   16×16 to 256×128 (62 distinct). Every derived number was mis-specified.
   *Caught by:* the census recording the actual MT per shape.
4. **Trusted the Python transcription of the selector.** It reproduces only ~70% of observed
   grids. Unexplained; the dominant mismatch ratio is exactly 0.5, so it is one missing
   systematic rule, not noise. **The census is ground truth.**
5. **A subagent reported "no library contains StreamK kernels"** — flatly contradicted by a
   smoke test returning 100% `_SK3_`. *Lesson: verify agent claims against direct
   measurement before acting on them.*
6. **Predicted raising the small-branch ceiling would help.** It hurt, monotonically. The
   wrong prediction is what localised the effect to the threshold.

---

## 8. Protocol — non-negotiable

1. **One binary** for all arms; swap libraries via `HIPBLASLT_TENSILE_LIBPATH`.
2. **One process per arm** — `TENSILE_STREAMK_*` latch in function-local statics.
3. **Interleave with arm-order rotation.** Sequential A/B drifts past the effect size.
4. **Check the SK-mode mix (`_SK<n>_`) before reading any throughput** — a short workspace
   makes `getSKGrid` silently revert to DP.
5. **Selection agreement between env-only arms must be 100%.** Less means nondeterminism or
   a library mismatch.
6. **Report in kernel-duration bands.** 92% of the set is sub-1 ms and conclusions reverse.
7. `--min-iters 200`, not `--fixed-iters` — fixes the init-cost artifact without capping
   precision on sub-0.1 ms shapes.

### Traps

- **`DYNAMIC_GRID=0` does not reach Origami** — it gates on `> 0`, so 0 falls through to
  `skGrid = computeUnitCount`. **`DYNAMIC_GRID=7` is out of enum range and hits
  `default: return cu_count`** — the only way to get `number_of_cus`. **8 arms, not 7.**
- **`TENSILE_STREAMK_DATA_PARALLEL=1` is not an off-switch** — its store is overwritten.
  Use `DYNAMIC_GRID=4`.
- **Mode 6 is the only mode that can select parallel reduction** — a confound in m6-vs-m4.
- **`--sm_count_target` reaches kernel selection**, not just the grid (agreement 53.9%).
- **`resolve_num_cus` only accepts reductions**, so the budget cannot be raised from outside.
- **The grid variables cannot affect kernel selection**, so every arm runs kernels chosen
  for the *shipped* grid. Any multiplier result is a **lower bound** — re-selection could
  unlock more.

---

## 9. Source patches currently in `~/exp/stock`

**All three default to stock behaviour**, so an unset environment is byte-identical to the
original. Origami is a static lib — `make hipblaslt` relinks in ~1 second.

| env var | file | default | effect |
|---|---|---|---|
| `ORIGAMI_RDNA_CU_MULT` | `origami/.../hardware.cpp` | 2 | the RDNA `multiProcessorCount` scale factor |
| `ORIGAMI_SMALL_CU_MULT` | `origami/.../streamk.cpp` | 1 | ceiling on the K-split branch only |
| `ORIGAMI_THRESHOLD_MULT` | `origami/.../streamk.cpp` | 1 | branch boundary only, ceilings unchanged |

Original saved as `shared/origami/src/origami/hardware.cpp.pre-mult-experiment`.

**To revert:** copy that back, undo the two `streamk.cpp` blocks (each marked
`EXPERIMENT (2026-08-19)`), `make hipblaslt`.

---

## 10. Tooling

`~/hhs_tn_grid_vs_resource_origami_9k/harness/`:

| file | what |
|---|---|
| `streamk_env_ab.py` | interleaved A/B, arms are **(library, env) pairs**; contract-hash **group-level** resume; killpg on timeout; parses `skGrid`/`skTiles`/`SKItersPerWG`; `--rotating/--workspace/--limit-file`; per-arm `::--flag val` |
| `streamk_contract.py` | run identity — hashes bench sha, **libhipblaslt.so sha**, per-arm library+kernel-object shas, env, arm order, iteration-tier source |
| `run_campaign.py` | declarative driver: deadline arbitration (reduce scope, never skip), per-phase cap, atomic ledger, telemetry |
| `analyze_campaign.py` | census table, integrity gates, banded paired contrasts with bootstrap CI, null/live partition |
| `validate_grid_model.py` | Python transcription vs observed grids |

Deck: `~/streamk_presentation`, `./run_app.sh` → port 8026. `verify.py` runs 7 gates,
including a **citation gate** that re-checks every `path:line:token` against live source —
it caught drift three separate times during this work.

---

## 11. Open threads, in priority order

1. **Finish the threshold experiment.** `p7_thresh.csv` has 4,500 of ~6,000 rows. Resume
   with the identical argv (it resumes at group granularity), then analyse the
   `96 ≤ tiles < threshold` band — that is the band the threshold should move and it was
   never measured. **This is the most likely real win.**
2. **A size-dependent threshold.** The data says the small/big boundary is the tunable, and
   that it should vary with problem size. Concrete, testable, ~2 lines.
3. **Re-selection under a changed budget.** The grid knobs cannot reach kernel selection, so
   every result is "new grid, old kernel choice". Feeding the new `N_CU` into
   `PredictionLibrary` would let the catalog respond.
4. **Measure occupancy directly.** Everything about "2 workgroups per WGP" is *inferred*
   from the ×2 optimum, never measured — `rocprofv3 --kernel-trace` failed under hipBLASLt
   (`Could not load TensileLibrary_lazy_gfx1100.dat`). This is the softest link, and the
   new-SKU hypothesis depends on it.
5. **`MinItersPerCU = 8`** decides 514 shapes and is untested. Needs the same patch trick.
6. **Explain the transcription residual** (~30% of grids, ratio exactly 0.5).
7. **The clean tile-selection oracle** (top-1 vs top-K on a fixed catalog) for claim 4.

For running this on a different SKU, see **`STREAMK_NEW_SKU_PROMPT.md`**.
