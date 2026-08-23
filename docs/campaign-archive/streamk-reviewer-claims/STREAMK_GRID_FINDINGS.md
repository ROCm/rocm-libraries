# StreamK launch-grid selection on gfx1100 — what we found

Measured 2026-08-19 on a Radeon RX 7900 XTX (gfx1100). **69,000 benchmark runs**, one
binary, interleaved with per-shape arm rotation, 1500-shape HHS-TN evaluation set.
Raw report: `~/hhs_tn_grid_vs_resource_origami_9k/reports/STREAMK_CAMPAIGN.md`.

Everything here is against `~/exp/stock` — the tree the binaries were actually built from.

---

## 0. The method that made this tractable

`TENSILE_DB=0x40` makes hipBLASLt print the packed kernel arguments, which include
`skGrid`, `skTiles`, `SKItersPerWG` and `itersPerTile`:

```
[132..135] SKItersPerWG: 80 00 00 00 (128)
[136..139] skGrid:       00 04 00 00 (1024)
[140..143] skTiles:      00 04 00 00 (1024)
```

So the launch grid can be **read**, not inferred. A census pass at
`--iters 1 --cold_iters 0 --rotating 0` costs ~0.25 s per (shape, config) and carries **zero
statistical uncertainty**. Do this first on any new architecture — it answers most questions
before a single timing measurement.

We also transcribed the selector into Python to design the experiment. It only reproduced
~70% of observed grids. **The census is ground truth; a transcription is a hypothesis.**

---

## 1. How the grid is chosen

`TENSILE_STREAMK_DYNAMIC_GRID` defaults to **6** = `origami::grid_selection_t::k_split_aware`.
Not opt-in — it is the path every StreamK launch takes.
(`AMDGPU.hpp:289` → `ContractionSolution.cpp:3934` → `origami/streamk.cpp:334-441`.)

With `cu_count = hardware.N_CU`:

```
tiles = ceil(M/MT_M) * ceil(N/MT_N) * batch
sk_grid = tiles                                   // fallback

if tiles > cu_count:            // BIG — shrink the grid to a clean fraction of tiles
    min_even = max(1, tiles / cu_count)
    for frac in {0, ½, ⅛, ⅕, ¼, ⅓}:               // this order, not sorted
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

The **only hardware input is `cu_count`.** Every other constant — `MinItersPerCU = 8`, the
two fraction lists, 128 MB, 128 B — is a hardcoded literal with **no architecture guard**.

### One grid serves both phases

`ContractionSolution.cpp:1764`: `rv.numWorkGroups.x = sk.grid`, y = z = 1. There is a single
1-D launch. In mode 3 each workgroup runs its data-parallel tiles (round-robin, stride =
`skGrid`) *and* its streamed slice inside that one launch. Measured example: 980 tiles,
grid 184 → 244 streamed + 736 data-parallel at 4 per workgroup.

### `fragment` — the concept everything hinges on

```
iters_per_cta  = ceil(tiles * itersPerTile / grid)   // each WG's equal share
fragment       = iters_per_cta % itersPerTile        // the bit that is not a whole tile
```

`fragment == 0` means no workgroup straddles a tile boundary → no tile has two contributors
→ **no fixup, no streaming**. The fragment *is* the StreamK-ness of a launch. It is also the
ragged memory access the cache-line filter measures.

---

## 2. The shipped predictor leaves StreamK inert on 77.5% of shapes — and it is right to

Census, 1500 shapes × 8 modes. "Inert" = the launch cannot split any tile
(`skTiles == skGrid`, or `SKItersPerWG == itersPerTile`).

| `DYNAMIC_GRID` | inert (shapes) | inert (kernel time) | median grid |
|---|---|---|---|
| 0 legacy fallback (`computeUnitCount` = 48) | 36.6% | 22.8% | 48 |
| 1 min_resources | 68.9% | 26.2% | 96 |
| 2 energy_aware | 75.7% | 37.4% | 87 |
| 3 reduction_cost_aware | 80.6% | 40.0% | 82 |
| 4 data_parallel | 100.0% | 100.0% | 96 |
| 5 analytical | 79.3% | 97.7% | 96 |
| **6 k_split_aware (shipped)** | **77.5%** | **64.9%** | 96 |
| 7 number_of_cus (= N_CU = 96) | 23.5% | 20.2% | 96 |

Mode 4 measuring exactly 100% is the sanity check that the inert detector is correct.

**Then the throughput.** Paired geomean vs the shipped default:

| arm | geomean |
|---|---|
| `m0` legacy grid = 48 | **78.7%** |
| `m1` min_resources | 91.4% |
| `m3` reduction_cost_aware | 92.3% |
| `m4` StreamK off (identical kernels) | 97.0% |
| `m7` number_of_cus = 96 | 92.5% |

**Nothing beats mode 6.** The falsifiable test confirms why: partition shapes by the
*observed* census.

| contrast | on inert shapes | on streaming shapes |
|---|---|---|
| `m4` (force data-parallel) vs `m6` | **99.11%**, median abs dev **0.48%** | **89.9%** |

Where mode 6 declines to stream, forcing data-parallel changes nothing. Where it does
stream, turning StreamK off costs 10%. It is correct in both directions.

That 0.48% is also the campaign's **measured noise floor** — half the ~1 pt we had assumed,
and obtained for free from shapes that are identical by construction.

---

## 3. StreamK only pays below ~1 ms

`m4` (StreamK off, identical kernel binaries, only the grid differs):

| band | n | off vs on |
|---|---|---|
| <0.1 ms | 1594 | **93.9%** — StreamK clearly wins |
| 0.1–1 ms | 1102 | 100.2% — a wash |
| 1–5 ms | 258 | **101.9%** — StreamK loses |
| ≥5 ms | 36 | **101.7%** — StreamK loses |

Exactly what the mechanism predicts: StreamK recovers idle slots, and above ~1 ms there are
none left to recover, so only the fixup cost remains.

---

## 4. The CU multiplier — ×2 is right on this SKU

On RDNA, HIP runs in WGP mode and `multiProcessorCount` reports **WGPs**, so Origami scales
it (`origami/src/origami/hardware.cpp:132`):

```cpp
properties.multiProcessorCount * cus_per_multiProcessorCount(arch_enum)
// gfx1100/1150/1151/1152/1153/1200/1201 -> 2 ;  everything else (CDNA) -> 1
```

gfx1100: 48 WGPs → **N_CU = 96**. Meanwhile TensileLite's own `AMDGPU::computeUnitCount`
stays **48** and drives the launch-side paths. The build carries both numbers, deliberately.

We patched the factor to be settable (`ORIGAMI_RDNA_CU_MULT`, default 2 = stock) and swept
the **adaptive** predictor with different budgets — a different question from `FIXED_GRID`,
which pins the grid and kills adaptivity:

| N_CU | meaning | <0.1 ms | 0.1–1 ms | ≥1 ms | ALL | 95% CI |
|---|---|---|---|---|---|---|
| 48 | WGPs (×1) | 95.9 | 96.2 | 97.8 | **96.2** | [95.5, 96.9] |
| **96** | **physical CUs (×2, shipped)** | — | — | — | **100** | — |
| 144 | ×3 | **101.2** | 96.6 | 97.1 | 99.1 | [98.2, 100.0] |
| 192 | SIMD32 lanes (×4) | **100.8** | 96.9 | 98.6 | 99.1 | [98.3, 100.0] |
| 288 | ×6 | 100.3 | 96.8 | 99.1 | 98.9 | [98.1, 99.8] |

**×2 wins. The SIMD count is not the better denominator.** ×4 reaches parity at the top of
its CI but never beats it.

Why the threshold is 48 and not 96: **the grid counts workgroups, and a workgroup occupies a
whole WGP.**

| grid | consequence | shapes @ ×2 | shapes @ ×1 |
|---|---|---|---|
| < 48 | some WGPs get nothing — idle | 178 | **462** |
| 48–95 | all WGPs busy, 1 WG each — half occupancy | 460 | 472 |
| ≥ 96 | 2 WGs per WGP — packed | 862 | 566 |

×1 loses because it runs the machine at **half occupancy**, not because it idles cores.
That is the real signal: **96 is the first value that fills 48 WGPs at occupancy 2.**

Also: inert share is nearly flat across multipliers (73–78%), so the factor changes grid
**magnitude**, not whether StreamK engages. Raising it gives you a bigger launch, not more
streaming.

**The one live thread:** ×3 and ×4 beat ×2 by ~1% on sub-0.1 ms shapes and lose ~3% in the
0.1–1 ms band. A *size-dependent* multiplier might beat the constant; a larger constant does
not.

---

## 5. Small problems: StreamK mostly declines

681 shapes have `tiles < 96` and take the K-splitting branch. What they actually get:

| split factor `f` | shapes | median itersPerTile |
|---|---|---|
| **1 — no split, pure DP** | **514 (75%)** | **2** |
| 2 | 37 (5%) | 64 |
| 8 | 30 (4%) | 64 |
| 3–33 | 100 (15%) | 128–264 |

**Of the 514 that get no split, 79% have `itersPerTile < 16`** — so even `f = 2` violates
`itersPerTile/f ≥ 8`. Median itersPerTile among them is **2**.

So the mechanism splits cleanly on small-and-deep problems (that is where the win lives) and
**cannot help small-and-shallow ones at all** — there is no K to spread. Those shapes have
the worst wave quantisation of any in the set, and the only remaining lever is a smaller
macro-tile, i.e. tile selection.

`MinItersPerCU = 8` is the single gate deciding 514 shapes. It is un-derived, has no arch
guard, and is not reachable by any environment variable.

---

## 6. The cache-line filter is really a DepthU filter

```cpp
a_contig = (transA == T) ? fragment * DepthU * bytes_a : 0;   // K contiguous for A
b_contig = (transB == N) ? fragment * DepthU * bytes_b : 0;   // K contiguous for B
reject if either is non-zero and not a multiple of 128
```

This is about **global** loads — whether the fragment's contiguous span along K covers whole
128 B cache lines. It only guards the operand whose K is contiguous, so a TN problem checks
both, and NN or TT checks one or neither.

Measured firing rate over candidate grids: **18.4% overall**, entirely determined by DepthU
(fp16: needs `fragment × DepthU ≡ 0 mod 64`):

| DepthU | vetoed | fragment must be a multiple of |
|---|---|---|
| 16 | 26% | 4 |
| 32 | **43%** | 2 |
| 64 | **0%** | 1 |
| 128 | 0% | 1 |

At DepthU ≥ 64 the check is dead code. It is also dtype-dependent: at fp8 the threshold
doubles and it would fire considerably more often.

---

## 7. Two workspace checks, only one of them real

| line | check | against |
|---|---|---|
| `streamk.cpp:387` | candidate filter in the loop | **hardcoded 128 MB** |
| `streamk.cpp:438` | final clamp | the actually-allocated workspace |

The 128 MB has no derivation in source. Where it bites depends on the macro-tile — 256×128
blocks grids above 1024; 16×16 is never blocked. On our data it essentially never fires.
It is also gated on `tiles % grid != 0`, so an *inert* grid can reserve unlimited workspace
unchallenged.

Separately: if the workspace is short, `getSKGrid` does **not** fail — it silently sets
`skGrid = tiles` and reverts to data-parallel. **Any StreamK measurement must confirm the
kernel actually streamed**, which is what the census is for.

---

## 8. Traps worth carrying to any new architecture

1. **`DYNAMIC_GRID=0` does not reach Origami.** It gates on `> 0`, so 0 falls through to
   `skGrid = computeUnitCount` (TensileLite's number). **`DYNAMIC_GRID=7` is out of enum
   range and hits `default: return cu_count`** — the only way to get `number_of_cus`.
   There are 8 distinct arms, not 7.
2. **`TENSILE_STREAMK_DATA_PARALLEL=1` is not an off-switch** — its store is immediately
   overwritten. Use `DYNAMIC_GRID=4`.
3. **Mode 6 is the only mode that can select parallel reduction** (`getSKReduction` returns
   early for all others) — a confound in any m6-vs-m4 contrast.
4. **`--sm_count_target` reaches kernel selection**, not just the grid budget (selection
   agreement was only 53.9%). It is not a clean grid-only knob.
5. **The env vars latch in function-local statics.** One process per arm, always.
6. `resolve_num_cus` only accepts **reductions** (`requested < N_CU`), so you cannot raise
   the budget from outside — that needs the source patch.

---

## 9. What is still open

- **The default's provenance.** `947ed1ef87` moved it `3 → 6` in a one-line commit with an
  empty body and no benchmark data. It happens to be right on gfx1100; nothing records why
  it was expected to be.
- **`MinItersPerCU = 8`** decides 514 of 1500 shapes and has never been tested.
- **A size-dependent CU multiplier** (~×3–4 for sub-0.1 ms, ×2 above) is the one
  configuration our data suggests could beat the shipped default.
- **Our transcription of the selector is not exact** (~70%). The residual is systematic —
  the dominant mismatch ratio is exactly 0.5 — so it is one missing rule, not noise.
