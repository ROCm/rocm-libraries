# StreamK campaign — gfx1100 **TN HHS**

Radeon RX 7900 XTX (gfx1100, RDNA3). Run 2026-08-21, unattended.
Harness: TensileLite's own `tensilelite-client`, built fresh from `exp/stock`.
All numbers below are **TN HHS** (fp16 in/out, fp32 compute, `TransposeA=T, TransposeB=N`).

---

## Headline

**A one-line C++ fix unlocks StreamK modes 4 and 5 on RDNA3, and they compute correct
results. SK4 wins only where prefetch is off — a corner occupied by 8% of shipped solutions.
Turn on the prefetch the catalog actually uses and SK4 loses everywhere, by up to 37%.
Keep SK3, and now we know the mechanism: prefetch needs to know what comes next, and
work-stealing cannot.**

> **Scope.** Swept over **geometry × DepthU × ClusterLocalRead × PrefetchGlobalRead ×
> PrefetchLocalRead** (§6–§9). Three of those axes *invert* the SK4-vs-SK3 ratio, and SK4
> still never tops the **absolute** ranking on any of them. SK4 is faster in **8 of the 35
> cells measured**, and all 8 share one signature: `CLR0 + no prefetch + DepthU 16/32 +
> ≥1 ms` (§9b). Break any one and it loses. Since 88% of shipped solutions prefetch, that
> region is largely outside production. Unswept: WorkGroupMapping and StaggerU.

| result | |
|---|---|
| SK4/SK5 assemble on gfx1100 | **yes, after the fix** — 0 assembler errors, was 6 |
| SK4/SK5 numerically valid | **yes — 48/48 PASSED each**, 24 shapes × 2 geometries |
| SK4 vs SK3, **shipped MT128x128** | **89.8%** at ≥1 ms — SK4 loses |
| SK4 vs SK3, smaller tiles at CLR0 | **114–116%** — SK4 genuinely wins |
| best SK4 **absolute** | **95.0% of best SK3** — recomputed over all 46 runs, validation-filtered (see §8 note) — the ratio win does not survive |
| `ClusterLocalRead` × SK mode | **large, replicated interaction** — see §6 |
| SK5 | **exactly SK3 or SK4**, selected by its runtime mode bit |
| `StreamKFixupTreeReduction` | **null** — 99.5–100.5% at ≥1 ms against a 1.8% floor |
| `StreamKAtomic` | **structurally dead code** — three independent blockers |

**Recommendation: keep SK3.** Not because SK4 is always slower — it isn't — but because
every configuration in which SK4 wins is on a macro-tile that gives up more than SK4 gains.
See §6.

---

## 1. The fix (the shippable artifact)

`th:TH_ATOMIC_RETURN` was **hardcoded unconditionally** in rocisa C++ — so the Python-level
patch this campaign originally planned would have done nothing:

```cpp
// rocisa/rocisa/include/instruction/mem.hpp, GlobalAtomicIncU32Saddr::toString()
kStr += " th:TH_ATOMIC_RETURN";      // BEFORE: every architecture, always
```

That spelling is gfx12-only. RDNA3 has no scalar atomics, so StreamK's work-queue fetch
falls into this vector-atomic path and dies at the assembler. gfx1100 supports the
operation — under the older name `glc`. Now:

```cpp
if(rocIsa::getInstance().getAsmCaps()["HasTHModifier"])
    kStr += " th:TH_ATOMIC_RETURN";  // gfx12+
else
    kStr += " " + getGlcBitName();   // gfx11 -> "glc", CDNA -> "sc0"
```

Both `HasTHModifier` and `getGlcBitName()` already existed. Blast radius is nil: SK4/SK5
ship on no architecture, gfx1100 never had scalar atomics, and GSU's analogous fallback
uses a different instruction (`FlatAtomicDecU32`).

> ⚠ **The tree is left PATCHED and rocisa REBUILT**, deliberately — reverting the source
> without rebuilding would leave the `.so` inconsistent with it. Patch saved to
> `artifacts/rocisa_glc_fix.patch`; backups in `artifacts/`. To revert:
> `git -C ~/exp/stock checkout projects/hipblaslt/tensilelite/rocisa/rocisa/include/instruction/mem.hpp`
> then `make -C ~/exp/stock/build/release/tensilelite/rocisa _rocisa -j$(nproc)`.

## 2. Measurement protocol — the first one was worthless

The first protocol (`NumWarmups 4`, `EnqueuesPerSync 4`, `SyncsPerBenchmark 2` = **8 timed
enqueues**) gave this A/A floor — the same config run twice, where every ratio must be 1.0:

| band | p95 \|dev\| OLD | p95 \|dev\| NEW | |
|---|---|---|---|
| `<0.1ms` | **541.0%** | 19.1% | 28× tighter |
| `0.1-1ms` | 30.5% | 19.4% | |
| `>=1ms` | 10.9% | **1.8%** | 6× |

Fix: `MinFlopsPerSync: 2e9`, making the enqueue count **adaptive** — a 66 kflop shape gets
~30k enqueues, a 1.3 Tflop shape gets 1. Final floor is from **144 pairwise A/A comparisons
over 3 runs**.

**Under the old protocol I would have reported a 12% tree-reduction win that does not
exist.** Two independent instruments caught it: the A/A floor and a negative-control arm.

## 3. Shape selection

24 census shapes where SK3 **actually streams** — `skTiles != skGrid` **and**
`itersPerTile > 1`. Only **337 of 1500** census shapes (22.5%) qualify, the exact
complement of the known "StreamK inert on 77.5%" result. Stratified 8 per duration band.

## 4. SK3 vs SK4 vs SK5 — the first slice (DepthU 32, CLR default)

> **Scope note — read this before the table.** Everything in this section was measured at the DEFAULT
> `ClusterLocalRead: 1`. §5 shows that setting it to 0 — as the shipped MT128x128 solution
> does — reverses the MT128x64 verdict. Read §4 as "SK4 under CLR=1", not as a general claim.

`*` = outside the measured floor for that band.

**MT 128x64x32**

| band | SK3 | SK4 | SK5 | SK4/SK3 | SK5/SK3 |
|---|---:|---:|---:|---:|---:|
| `<0.1ms` | 1 836 | 752 | 760 | **41.0%** * | **41.4%** * |
| `0.1-1ms` | 38 164 | 29 620 | 31 743 | **77.6%** * | 83.2% |
| `>=1ms` | 65 444 | 54 096 | 53 976 | **82.7%** * | **82.5%** * |

**MT 64x64x32**

| band | SK3 | SK4 | SK5 | SK4/SK3 | SK5/SK3 |
|---|---:|---:|---:|---:|---:|
| `<0.1ms` | 2 053 | 1 053 | 1 089 | **51.3%** * | **53.0%** * |
| `0.1-1ms` | 42 228 | 37 497 | 37 806 | 88.8% | 89.5% |
| `>=1ms` | 56 005 | 55 439 | 55 432 | 99.0% | 99.0% |

**SK4 loses, and the deficit shrinks monotonically as kernels lengthen** (41% → 78% → 83%
on MT128x64). That is the signature of a **fixed per-work-item cost**: the `s_atomic_inc`
queue fetch is paid once per item and cannot be amortised over a short tile. Work-stealing
buys load balance that SK3's static, equal-chunk split already has.

**The penalty is geometry-dependent.** At ≥1 ms, MT128x64 loses 17% but MT64x64 loses
nothing (99.0%). Larger tiles mean fewer, longer work items — so fewer queue fetches — yet
MT128x64 is *worse*, which says the cost tracks the number of *tiles per workgroup*, not
tile size alone. §5 shows it is also **ClusterLocalRead-dependent**, which is the stronger
effect and the one worth chasing.

## 5. The shipped geometry — and the result that overturned the headline

**MT128x128 now builds.** Token-diffing my failing kernel name against the shipped
solution isolated one settable difference: **`ClusterLocalRead`** — shipped `CLR0`, default
`CLR1`. CLR0 fits the 256-VGPR budget; CLR1 needs 266. (GlobalReadVectorWidth,
LocalReadVectorWidth and StoreRemapVectorWidth had all been forked earlier and made no
difference — it was never a staging problem.)

**On the shipped geometry, the conclusion holds.** MT128x128, 3 runs, 144/144 PASSED:

| band | SK3 | SK4 | SK5 | SK4/SK3 | SK5/SK3 |
|---|---:|---:|---:|---:|---:|
| `<0.1ms` | 1 527 | 543 | 1 553 | **35.6%** | 101.7% |
| `0.1-1ms` | 40 643 | 28 646 | 46 181 | **70.5%** | 113.6% |
| `>=1ms` | 68 217 | 63 210 | 68 429 | **92.7%** | 100.3% |

**But `ClusterLocalRead` flips the verdict on other geometries.** Same shapes, same
geometry (MT128x64), ≥1 ms band, only CLR differing:

| | CLR=1 | CLR=0 | Δ |
|---|---:|---:|---|
| SK3 | 65 444 | 60 465 | −7.6% |
| SK4 | 54 096 | 67 115 | **+24.1%** |
| **SK4/SK3** | **82.7%** | **111.0%** | **verdict flips** |

CLR0 costs SK3 7.6% and gains SK4 24.1%, turning a 17% SK4 loss into an 11% SK4 **win**.
Both moves are far outside the 1.5% floor.

**So §4's "SK4 is consistently slower" was measured under CLR=1 only and does not
generalise.** The honest statement is narrower: *on the shipped MT128x128 geometry with its
shipped CLR0, SK4 loses in every band* — and `ClusterLocalRead` is a strong interacting
variable that nobody appears to have swept against StreamK mode. That is the clearest
follow-up this campaign produced.

## 6. Mapping it: `ClusterLocalRead` × StreamK mode

16 shapes drawn **entirely from the ≥1 ms band** (where the A/A floor is 2.53% over 576
pairs), 3 runs, every point validated. CLR [0,1] × SK [3,4,5] × 2 geometries, plus the
shipped MT128x128 on the identical shape set.

| geometry | CLR | SK3 | SK4 | SK4/SK3 | |
|---|---|---:|---:|---:|---|
| **MT128x128 (shipped)** | 0 | 71 762 | 64 425 | **89.8%** | SK4 loses |
| MT128x64 | 0 | 57 969 | 66 072 | **114.0%** | **SK4 wins** |
| MT128x64 | 1 | 63 232 | 53 237 | 84.2% | SK4 loses |
| MT64x64 | 0 | 49 189 | 56 990 | **115.9%** | **SK4 wins** |
| MT64x64 | 1 | 55 805 | 54 202 | 97.1% | SK4 loses |

**The interaction is consistent in direction on both geometries.** Going CLR1 → CLR0:

| | MT128x64 | MT64x64 |
|---|---|---|
| SK3 | −8.3% | −11.9% |
| **SK4** | **+24.1%** | **+5.1%** |
| SK5 | −8.1% | −11.4% |

CLR0 hurts the static modes and helps the dynamic one. SK5 tracks SK3 throughout (99.8–100.3%),
consistent with its default mode bit selecting the static path.

### …but the ratio is a trap

Ranking every configuration **absolutely** on the same shapes:

```
1   71 762   100.0%   MT128x128  CLR0  SK3   <-- shipped geometry
2   71 389    99.5%   MT128x128  CLR0  SK5
3   66 072    92.1%   MT128x64   CLR0  SK4   <-- best SK4 anywhere
4   64 425    89.8%   MT128x128  CLR0  SK4
…
15  49 189    68.5%   MT64x64    CLR0  SK3
```

**Best SK4 anywhere is 92.1% of best SK3.** SK4's 114–116% wins are ratios measured against
an SK3 that CLR0 has already crippled, on macro-tiles that are 10–30% slower to begin with.
Beating a weakened baseline on an inferior geometry is not a win.

This is the cleanest lesson of the campaign: **a per-geometry ratio can invert the absolute
ranking.** Had the sweep stopped at ratios it would have recommended SK4.

## 7. …and the CLR preference is DepthU-dependent, which vindicates the catalog

The shipped catalog uses **CLR0 for 128 of its 192 solutions and CLR1 for 64**, with several
geometries appearing under both (MT128x64: 9× CLR0 / 2× CLR1; MT64x64: 7× / 6×). That looked
like it might be leaving ~10% on the table given §6. It is not — the tuner is right and my
claim was too broad.

SK3 only, CLR × DepthU, same 16 shapes, 3 runs, 2.67% floor:

| geometry | DepthU | CLR0 | CLR1 | CLR1/CLR0 | |
|---|---|---:|---:|---:|---|
| MT128x64 | 16 | 39 739 | 39 638 | 99.7% | tie |
| MT128x64 | **32** | 58 424 | 63 488 | **108.7%** | CLR1 better |
| MT128x64 | 64 | **70 727** | — | (CLR1 will not build) | |
| MT64x64 | 16 | 31 707 | 31 788 | 100.3% | tie |
| MT64x64 | **32** | 49 447 | 56 205 | **113.7%** | CLR1 better |
| MT64x64 | **64** | **67 162** | 62 239 | **92.7%** | **CLR0 better — reverses** |

**Every measurement in §4–§6 was taken at DepthU = 32 — which happens to be the
one DepthU where CLR1 wins.** At DepthU 64 the preference inverts, and the fastest SK3
configuration found anywhere in this campaign is `MT128x64 / DepthU 64 / CLR0` at **70 727
GFlop/s** — CLR0, and higher than anything in the DepthU-32 sweep.

So the catalog's mixed CLR usage is not an oversight; the correct CLR is a function of
DepthU (and probably more). **The honest scope of this campaign is: one recipe, DepthU 32.**
The SK3-vs-SK4 rankings should be read as a slice through a space whose axes interact, not
as a general result.

## 8. The mode ranking is DepthU-dependent too — which *strengthens* the conclusion

§7 warned that everything was a DepthU-32 slice. So I swept the mode ranking itself.
SK [3,4,5] × DepthU [16,32,64], CLR0 pinned so DepthU is the only axis moving, same 16
shapes, 3 runs, 2.40% floor:

| geometry | DepthU | SK3 | SK4 | SK5 | SK4/SK3 | |
|---|---|---:|---:|---:|---:|---|
| MT128x64 | 16 | 39 574 | 49 361 | 39 522 | **124.7%** | SK4 wins |
| MT128x64 | 32 | 58 207 | 65 983 | 58 276 | **113.4%** | SK4 wins |
| MT128x64 | **64** | **70 508** | 62 910 | 70 525 | **89.2%** | SK4 loses |
| MT64x64 | 16 | 31 653 | 48 970 | 31 511 | **154.7%** | SK4 wins |
| MT64x64 | 32 | 49 230 | 56 567 | 49 331 | **114.9%** | SK4 wins |
| MT64x64 | **64** | **67 011** | 62 514 | 66 977 | **93.3%** | SK4 loses |

**SK4's advantage falls monotonically as DepthU rises** — 155% → 115% → 93%. That is
mechanistically consistent with the per-work-item atomic story: SK4's work item is a *tile*,
and low DepthU means more k-iterations per tile, i.e. a longer work item over which the
`s_atomic_inc` amortises. Raise DepthU, shorten the work item, and the fixed cost bites. It
is the same effect as the duration-band result in §4, reached by a different axis.

**And the trap reproduces on this axis too:**

```
70 525   MT128x64  DU64  SK5
70 508   MT128x64  DU64  SK3   <- best SK3
67 011   MT64x64   DU64  SK3
66 977   MT64x64   DU64  SK5
65 983   MT128x64  DU32  SK4   <- best SK4 anywhere
```

**Best SK4 = 93.6% of best SK3 _within this section's sweep_.**

> **Four "best SK4" figures appear in this report and they are not interchangeable.**
> 92.1% is §6's CLR sweep; 93.6% is this DepthU sweep; 94.0% in §9 was campaign-wide *as of
> when §9 was written*. The authoritative number is **95.0%**, recomputed 2026-08-22 over
> **all 46 run directories** using `analyze.py`'s own loader so the mandatory validation
> filter is applied — best SK3 92 592.7 and best SK4 87 958.2 GFlop/s, **both from the same
> run (`P15_r1`)**, so it is a like-for-like comparison. 7977 validation records, **0 FAILED**,
> so the filter changes nothing here. §9's 76 079 GFlop/s predates the P15 phase.
> **Quote 95.0%.** The conclusion is identical at every one of these values: SK4 loses.
>
> **A cleaner statistic exists and answers a different question.** 95.0% is *best-vs-best*,
> which may compare two differently-parameterised solutions. Matching solutions whose names
> are **identical except the `_SK<n>_` token** gives the cost of switching mode with
> everything else held fixed — 2288 validated matched pairs across the campaign:
>
> | contrast | median | within 2% |
> |---|---|---|
> | **SK4 / SK3** | **89.38%** | 7.7% |
> | SK5 / SK3 | 99.91% | 74.7% |
> | SK5 / SK4 | 109.33% | 14.7% |
>
> So **SK4 costs ~10.6% against SK3 on like-for-like configurations**, while best-vs-best
> flatters it to 95.0% because each mode is allowed its own best parameters. Both are correct;
> quote 89.4% for "what does choosing SK4 cost me", 95.0% for "how good can SK4 get".
> The SK5 row independently confirms §10 from raw data.
>
> **The same matched-pair method replicates every axis conclusion in this report**, which is
> the strongest evidence here that they are not artefacts of how the sweeps were aggregated:
>
> | axis | matched-pair SK4/SK3 median | this report (best-vs-best) |
> |---|---|---|
> | DepthU 16 / 32 / 64 | 138.9% / 92.7% / 72.1% | 155% / 115% / 93% |
> | `CLR` 0 / 1 | 90.7% / 85.5% | CLR0 favours SK4 (§6) |
> | `PGR` 0 / 1 / 2 | 92.5% / 71.2% / 73.1% | prefetch is what kills SK4 (§9) |
>
> Every direction, ordering and sign agrees: SK4 wins only at DU16, prefers `CLR0`, and
> collapses once prefetch is on. Magnitudes run ~15–20 pt lower throughout — the expected
> signature of best-vs-best letting each mode choose its own parameters. The conclusions are
> robust to aggregation method; only the numbers move.
>
> Reproduce: `python3 matched_pairs.py` (prints all of the above, plus the validation
> census and the best-vs-best table, from the raw run directories).

SK4's 155% ratio is measured at DU16, where *everything*
is slow (31–49k vs 70k at DU64). Every configuration in which SK4 wins is in the slow region
of the space.

### What this does to the conclusion

It makes it stronger, not weaker. Across **2 geometries × 3 DepthU × 2 ClusterLocalRead**,
SK4 never reaches the top of the absolute ranking — its wins are always ratios inside slow
regions. "Keep SK3" is no longer a DepthU-32 slice; it survives every axis swept.

The transferable lesson, now demonstrated twice on independent axes: **a ratio computed
within a configuration can invert the ranking across configurations.** A tuner that
optimised SK4/SK3 per-recipe would pick SK4 and lose 6%.

## 9. Prefetch settles it — SK4's only win is in an 8% corner

The whole campaign used `PrefetchGlobalRead=0, PrefetchLocalRead=0`. That is **16 of 192
shipped solutions — 8%.** The catalog overwhelmingly prefetches (PGR 1: 92×, PGR 2: 78×).
Since prefetch hides exactly the global-read latency SK4's per-work-item atomic exposes,
this was the most load-bearing unswept axis.

MT128x64, SK × PGR × PLR × DepthU, same 16 shapes, 3 runs, 2.58% floor:

| DU | PGR | PLR | SK3 | SK4 | SK4/SK3 | |
|---|---|---|---:|---:|---:|---|
| 32 | 0 | 0 | 58 135 | 66 601 | **114.6%** | SK4 wins ← *my original recipe* |
| 32 | 0 | 1 | 63 904 | 54 379 | 85.1% | SK4 loses |
| 32 | 1 | 0 | 71 121 | 66 472 | 93.5% | SK4 loses ★ |
| 32 | 2 | 1 | 72 370 | 71 497 | 98.8% | tie ★ |
| 64 | 1 | 0 | 69 300 | 47 455 | **68.5%** | SK4 loses ★ |
| 64 | 1 | 1 | **76 079** | 48 025 | **63.1%** | SK4 loses |
| 64 | 2 | 1 | 74 973 | 48 586 | **64.8%** | SK4 loses ★ |

★ = one of the two commonest (PGR,PLR) joints in the shipped catalog.

**SK4 wins in 1 of the 12 cells in this grid — `DU32 / PGR0 / PLR0`, precisely the recipe
the campaign had been using.** (Pooled over the whole campaign it wins 8 of 35; see §9b for
the full map — the extra wins are all at DepthU 16, which this grid does not cover.) Turn on the prefetch that 88% of shipped solutions use and
SK4 loses everywhere, collapsing to 63–69% at DepthU 64.

### Why: prefetch needs to know what comes next, and work-stealing does not

Prefetch hides latency by issuing loads for work you *know* you will do. SK4 decides its
next tile by `s_atomic_inc` on a shared queue — it cannot know, so it cannot prefetch ahead.
SK3's static assignment can.

The codebase already encodes this tension independently of my measurement:

```python
# Solution.py:1726
if state["StreamK"] != 3:
    reject("PrefetchAcrossPersistent is currently supported only with StreamK=3")
```

*(My runs had `PrefetchAcrossPersistent=0`, so PAP is **not** the mechanism behind these
numbers — plain intra-tile PGR is. The restriction is corroborating evidence that the same
tension was already recognised, not an explanation of the measurement.)*

**Best configuration found in the entire campaign: `DU64 / PGR1 / PLR1 / SK3` at 76 079
GFlop/s.** Best SK4 anywhere reaches 94.0% of it.

## 9b. …and it holds in the small bands too, closing the last gap

Everything from §6 onward used the 16-shape ≥1 ms set, so the prefetch result was
established at ≥1 ms only. Since prefetch *reversed* the verdict there, it could plausibly
have reversed the small bands too. It does not — it makes SK4 worse.

MT128x64, 24-shape stratified set (8 per band), 3 runs. Floors: 15.5% / 8.0% / 1.9%.

| DU | PGR | `<0.1ms` | `0.1-1ms` | `>=1ms` |
|---|---|---:|---:|---:|
| 32 | 0 | 43.6% | 94.0% *(tie)* | **112.2%** ← the only SK4 win anywhere |
| 32 | 1 | 43.1% | 79.0% | 90.8% |
| 64 | 0 | 46.8% | 80.1% | 88.7% |
| 64 | 1 | 45.8% | **64.2%** | **70.7%** |

**SK4 loses in all twelve small-band cells, and prefetch widens the gap in every one.**

At `<0.1ms` SK4 sits at **43–47% regardless of prefetch** — flat across all four settings.
That is the fixed per-work-item atomic dominating outright: when the kernel is short there
is not enough work to hide anything behind, so prefetch has nothing to give.

**So "keep SK3" needs no band-qualified caveat.**

### Correction: where SK4 *is* faster — 8 of 35 measured cells, and they form one region

An earlier draft said "exactly one cell". That was scoped to this section's 12-cell grid.
Pooling **all 35 SK4/SK3 ratios measured across the campaign**, SK4 is faster in 8 — and
every one of them has all four of these properties:

```
ClusterLocalRead 0  +  no prefetch (PGR0 PLR0)  +  DepthU 16 or 32  +  >=1 ms
```

| SK4/SK3 | config |
|---|---|
| **154.7%** | MT64x64 · DU16 · CLR0 · PGR0/PLR0 |
| **124.7%** | MT128x64 · DU16 |
| 112.2–115.9% | both geometries · DU32 — 5 independent measurements |

Break any single one and it loses: DepthU 64 → 88.7–93.3%; CLR1 → 84.2/97.1%; prefetch on
→ 63.1–98.8%; 0.1–1 ms → 64.2–94.0%; <0.1 ms → **43.1–46.8%**.

**The spread is +55% to −57%** — SK4 is not marginally different from SK3, it is a different
performance regime depending on where in the space you stand.

**And every win is in a slow region.** Best SK4 anywhere is 71,497; best SK3 is 76,079. The
154.7% is measured at DU16 where SK3 itself reaches only ~31,000. That is the whole reason
the recommendation survives: the ratio is real, the configuration is not competitive.

## 9c. What SK4 actually does — and it is not what the name implies

Read from source and then **confirmed on device** with `TENSILE_DB=0x40`:

```
ItersPerTile: 128    SKGrid: 144        SKSplit: 2
SKTiles:      0      TotalItems: 4224   SKItersPerWI: 64
```

For 4097×8192×4096 at MT128x64, `tiles = 33 × 128 = 4224` — so **`TotalItems == tiles`
exactly and `SKTiles == 0`.**

**Default SK4 hands out whole tiles and never splits K.** It is a *persistent
work-stealing data-parallel* kernel, not a StreamK mode in the split-K sense. The host code
makes this explicit: `skTiles` initialises to 0 and only changes if `TENSILE_STREAMK_TILES`
is set.

Mechanically, per trip round the persistent loop:

1. wave 0 only (`v_readfirstlane(Serial)`, others branch over)
2. `s_atomic_inc` on **one of 8 sharded queues** — `queueIdx = StreamKIdx % 8`, addresses
   strided 256 B apart, commented *"Stride queues to different cache lines"*
3. the fetched index is broadcast to the other waves **through LDS**
4. decode to a tile, branch full/partial, run the MAC loop, repeat

SK3 does none of this — it computed its entire itinerary in `preLoop` before the loop began.

### Turning streaming on changes almost nothing

`TENSILE_STREAMK_TILES` engages correctly (verified: `SKTiles 0 → 144`,
`TotalItems 4224 → 4368 = 4224 − 144 + 144×2`), yet:

| `TENSILE_STREAMK_TILES` | SK4 | vs its own no-stream baseline |
|---|---:|---|
| unset (0 tiles split) | 55 876 | — |
| 144 (3.4% of tiles) | 56 204 | 100.6% |
| 512 (12.1% of tiles) | 55 670 | 99.6% |

**Under 1% either way.** So SK4's ~113% advantage in the favourable region comes entirely
from **dynamic scheduling**, not from streaming. The feature the mode is named for
contributes nothing measurable here; the atomic it costs is what dominates elsewhere.

### Bug: `TENSILE_STREAMK_TILES > tiles` crashes the GPU

SK4's host path is missing the clamp SK3's has (`ContractionSolution.cpp:966`,
`skTiles = std::min(skTiles, tiles)`):

```cpp
uint32_t skTiles = 0;
if (overrideTiles > -1) skTiles = overrideTiles;        // no clamp
uint32_t totalItems = (tiles - skTiles) + skTiles * skSplit;
```

For a 425-tile problem with `skTiles = 2048`, uint32 wraparound yields **2473 work items for
425 real tiles** — items index tiles that do not exist. Reproduced: `TILES=2048` with 8 of
16 shapes below threshold → `unspecified launch failure`. `TILES=144` (0 shapes below) and
`TILES=512` (1 below) both ran and validated.

**Fixed and validated.** One line, mirroring what the SK3 path already does at `:966`:

```cpp
skTiles = std::min(skTiles, static_cast<uint32_t>(tiles));
```

| | before | after |
|---|---|---|
| `TILES=2048` exit | 1 | **0** |
| PASSED | 2 | **96** |
| launch failures | **25** | **0** |

Regression-checked with `TENSILE_DB=0x40` that the clamp does **not** fire when it should
not: unset → `SKTiles 0 / TotalItems 4224`; `=144` → `SKTiles 144 / TotalItems 4368`. Both
unchanged from before the fix. Patch: `artifacts/sk4_clamp_fix.patch`.

## 10. SK5 is mechanically SK3 or SK4, selected at runtime

SK5 emits both paths and picks via bit 30 of `MagicShiftItersPerTile`
(`ContractionSolution.cpp:811`). Sweeping `StreamKHybridMode` separately (2 runs each):

| geometry | band | SK5 mode0 / SK3 | SK5 mode1 / SK4 |
|---|---|---:|---:|
| MT128x64 | `>=1ms` | **100.5%** | **99.9%** |
| MT64x64 | `>=1ms` | **100.5%** | **99.8%** |

Against the 1.8% floor, SK5 mode 0 reproduces SK3 and mode 1 reproduces SK4 **exactly**.
SK5 is therefore not a third algorithm — it is a packaging choice, and it inherits whichever
path it selects. On TN HHS that means it inherits SK4's penalty whenever the dynamic path is
chosen.

## 11. SK3 fixup knobs — a well-supported null

`StreamKFixupTreeReduction: 0 → 1` at ≥1 ms: 99.5% / 100.2% / 100.5% / 99.5% against a
**1.8%** floor. Null.

**The control is what makes this a real null, not an absence of evidence.**
`StreamKXCCMapping: 0 → 2` moved ≥1 ms performance by a consistent **−3%** across all four
cells (97.8 / 98.8 / 96.9 / 96.3%) — outside the floor and one-directional. So the harness
demonstrably *can* see a 3% effect where tree reduction shows 0.5%.

*(XCC mapping is not a pure no-op even on a monolithic part — it permutes `WorkGroup0`,
perturbing L2 locality. A small consistent loss is the expected sign.)*

## 12. `StreamKAtomic` is dead code — three independent blockers

Investigated as an optional side-quest (it cannot apply to true HHS: HHS writes fp16 D,
the emitted instruction is `BufferAtomicAddF32`, and gfx1100 has no
`global_atomic_pk_add_f16`). It is unreachable for **every** architecture and dtype:

1. **`Solution.py:1713`** requires fp32 **input**; StreamK requires a MatrixInstruction
   (`:1686`); gfx1100 has no fp32 MI. Jointly unsatisfiable here.
2. **`_GlobalAccumulation` never becomes `'SingleBuffer'`** when `StreamKAtomic=1`: the
   PartialsBuffer branch is skipped, `GlobalSplitUAlgorithm` defaults to `MultipleBuffer`,
   and the `SingleBuffer` branch needs `computeType != destType` — false for SGEMM, the
   only dtype the gate permits.
3. **`KernelWriterAssembly.py:13927/15478`**: the atomic epilogue requires
   `GlobalSplitU > 1`, which StreamK **rejects outright** ("Cannot enable both Stream-K
   and GSU").

Any one alone makes it unreachable. This fully explains `StreamKAtomic: 0` in **all ~490k**
shipped solutions across 15 architectures: it is not untested, it is **structurally
impossible in the current code**. Making it work needs all three changed plus a correctness
review of beta handling. Patches for (1) and (2) were written and **reverted**; (3) was not
completed.

## 13. What this campaign does NOT tell you: headroom vs the shipped library

Tempting arithmetic: best config here is 76,079 GFlop/s; the census reports the shipped
library at a 47,865 geomean on the same 16 shapes; ratio 1.59×. **That number is worthless
and must not be quoted.** Three independent invalidators, any one sufficient:

1. **Different ProblemType.** Shipped kernels are `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs`
   — bias, scaleAlphaVec and activation capable, carrying that epilogue whether used or not.
   Mine are bare `Cijk_Alik_Bljk_HHS_BH_UserArgs`. My kernels are structurally lighter.
2. **Different tool.** The census used `hipblaslt-bench` — a full library call including
   solution selection and API overhead. Mine used Tensile's client, dispatching one kernel.
3. **Different protocol, and this one is decisive.** The census ran
   `--min-ms 0.0 --reps 1 --fixed-iters 2` — **two iterations, no warmup.** That run was a
   *census*: it existed to read `skGrid`/`skTiles` out of kernel arguments via
   `TENSILE_DB=0x40`, where timing is irrelevant. Its `gflops` column was never a
   performance measurement.

> ⚠ **Do not use `p0b_census.csv`'s `gflops` column as a performance baseline.** It is a
> by-product of a grid-reading pass at 2 fixed iterations. The census is authoritative for
> `sk_grid` / `sk_tiles` / `iters_per_tile` — which is exactly what this campaign used it
> for, in shape selection — and for nothing else.

**A valid headroom comparison would need** the same tool, the same ProblemType (bias/SAV
included), the same timing protocol, and the shipped library's own selector choosing the
kernel. That is a different experiment — essentially the `~/ab1100` A/B harness — and this
campaign does not attempt it.

**So the honest scope: every comparison here is internal.** SK3 vs SK4 vs SK5, CLR0 vs CLR1,
DepthU and prefetch sweeps are all apples-to-apples inside one harness and one ProblemType,
which is what makes them trustworthy. Nothing here supports a claim about the shipped
library's absolute performance.

### Update 2026-08-22 — that experiment has since been run

The four conditions above (same tool, ProblemType with bias/SAV, same protocol, the
library's own selector) were met by `SHIP_TEST.md`: `hipblaslt-bench`, 4 arms, 1500 shapes,
**12000 measurements, zero failures**. Two findings bound this whole report:

1. **A default gfx1100 build ships no StreamK kernels at all.** Across
   `Logic/asm_full/navi31/` there are **2560 `StreamK: 0`** and **22 `StreamK: 3`**, and all
   22 sit in `Experimental/`, which `tasks.py` excludes by default (`experimental=False`).
   `~/exp/stock` — the library every measurement in this report used — is a purpose-built
   SK3 catalog (`82580dfc726 "exp: prune gfx1100 logic to the SK3 Prediction catalog only"`).
   The right choice for studying StreamK, but it means **none of this describes
   default-build behaviour.**
2. **Shipping StreamK would not be a win**: 97.91% of the shipped SK0 library without a size
   gate, 100.25% with one — parity at best. StreamK is +15% on sub-0.1 ms shapes and −9% on
   the largest, and the largest hold 69% of the wall-clock.

Neither invalidates anything above; the internal contrasts stand. But "SK4 loses to SK3" is
a statement about a catalog that does not ship, and should be read that way.

The portable results from that follow-up are the size gate (`GATE_RESULT.md`, +1.4–2.4%
depending on protocol, reproduced across two library builds and two protocols) and the
metric finding: **per-shape geomean and wall-clock disagreed in sign three separate times**,
because sub-0.1 ms shapes are 53% of this suite by count and 5% of it by time.

## 14. Four things this campaign got wrong, and what caught each

Recorded because the checks are more reusable than the results.

| # | The wrong claim I was about to make | What caught it |
|---|---|---|
| 1 | "`StreamKFixupTreeReduction` gives +12%" | **A/A repeat.** The protocol had a **541% p95 noise floor** in the `<0.1 ms` band — 8 timed enqueues. A negative-control arm (`StreamKXCCMapping`) agreed independently. |
| 2 | "SK4 is consistently slower than SK3" | **Closing a caveat.** Getting MT128x128 to build forced `ClusterLocalRead=0`, which *flipped* MT128x64 from 82.7% to 111.0%. The claim had been CLR1-only. |
| 3 | "CLR1 is better than CLR0 for SK3" | **Checking my claim against production.** The shipped catalog uses CLR0 for 128 of 192 solutions. Rather than call it a defect I swept CLR × DepthU — the preference *inverts* at DU64. The catalog was right; I was slicing at DU32. |
| 4 | "There is 1.59× headroom vs the shipped library" | **Validating the comparison before quoting it.** The census `gflops` column comes from `--reps 1 --fixed-iters 2` — a grid-reading pass, not a benchmark. Plus different tool and different ProblemType. |

Two patterns worth carrying forward:

- **Every one of these was a scope error, not an arithmetic error.** The numbers were right; the
  claims attached to them were broader than the evidence. The fix each time was to sweep one
  more axis, and three of the four axes swept turned out to invert a ratio.
- **The checks that caught them were cheap.** An A/A repeat, a negative control, comparing a
  claim against what production actually does, and reading the command line that produced a
  baseline. None took more than minutes; each prevented a confident wrong headline.

## 15. Caveats
- The `<0.1ms` and `0.1-1ms` floors (19.1% / 19.4%) are coarse. Only the ≥1 ms band
  (1.8%) supports fine distinctions; the small-band SK4 result (41%) is far outside its
  floor and is safe, but the SK5-vs-SK3 small-band numbers are not.
- The swept box is 2 geometries × 3 DepthU × 2 CLR, all at ≥1 ms. **PrefetchGlobalRead,
  PrefetchLocalRead, WorkGroupMapping and StaggerU were held fixed** at one recipe's values;
  two of the three axes I *did* sweep turned out to invert a ratio, so assume the unswept
  ones can too.
- 16–24 shapes, 3 geometries: enough to rank modes within a recipe, not to map a space in
  which DepthU, ClusterLocalRead and StreamK mode all interact.
- The CLR interaction was found late and tested on one geometry and one band. It is a solid
  result there (both moves >> the 1.5% floor) but its scope is unmapped.
- The tree is left patched (see §1).

## 15a. Reproduction check — 28 hours later

The campaign ran across ~28 h and ~40 benchmark invocations on one GPU. Re-running the
12-cell small-bands config at the end, against the same config measured near the start:

| band | max \|delta\| | note |
|---|---|---|
| `>=1ms` | **1.4 pt** | +1.4 / +0.1 / −0.4 / +0.2 — floor is 1.9% |
| `0.1-1ms` | 3.4 pt | the loosest band, floor 8.0% |
| `<0.1ms` | 1.9 pt | |

**Verdict flips: 0.** No cell changed which mode wins. Median drift **+1.0 pt** favouring
SK4 — small enough to be noise, but recorded rather than rounded away; it may be a
warm-machine effect. GPU finished at 56 °C edge / 63 °C junction, so no thermal throttling.

This is a stability check the internal A/A floors cannot provide: those are measured within
a session, and would not catch slow drift in driver state, clocks or thermals across a long
campaign. The conclusion is stable on both timescales.

## 15b. The patches are coupled to the deck's citations

Worth knowing before anyone reverts anything. The `sk4_clamp_fix` patch inserts 7 lines at
`ContractionSolution.cpp:795`, which shifts every line below it — including three source
lines the presentation cites by number:

| citation | pre-patch | post-patch |
|---|---|---|
| `computeUnitCount` | 3927 | **3934** |
| `numWorkGroups` | 1764 | **1771** |
| `CeilDivide` | 1731 | **1738** |

The deck's citation gate caught this immediately (10 failing citations across 8 slides) and
the references were updated after confirming against the pre-patch backup that all three
still point at the *same source lines*, not merely at some line containing the token.

**So: applying or reverting the ContractionSolution.cpp patch requires re-running
`verify.py` in `~/streamk_presentation` and re-pointing those three citations.** The gate
will tell you; it fails loudly rather than letting the deck cite the wrong line.

## 16. Artifacts

`~/sk_modes/` — `REPORT.md`, `analyze.py`, `configs/`, `logs/`, `results/`,
`artifacts/{P0_baseline.patch, rocisa_glc_fix.patch, shapes.json, *.BACKUP}`.

### Follow-up campaign, 2026-08-22 (36 000 measurements, zero failures)

Reports, in reading order:

| file | question | answer |
|---|---|---|
| `GATED_POLICY.md` | offline: is StreamK a wall-clock win? | no — 96.95% geomean but **102.17% wall-clock**; the metrics disagree in sign |
| `GATE_RESULT.md` | does a size gate help, measured? | **yes, +1.34%** suite wall-clock (9000 measurements); also the SCOPE section |
| `SHIP_TEST.md` | would shipping StreamK win at all? | **no — parity at best** (12 000 measurements) |

Analysis scripts, each runnable on a partial CSV so a live run can be monitored:
`gated_policy.py`, `gated_robust.py`, `gate_analyze.py`, `plateau_analyze.py`,
`ship_analyze.py`, `matched_pairs.py`. Data: `results/{gate_full,gate_plateau,ship_test}.csv`.
Running log with every intermediate read and every retraction: `logs/gate_interim.md`.

Source change: `ORIGAMI_MN_GATE` in `shared/origami/src/origami/streamk.cpp` — env-gated,
**default 0 is byte-identical to stock**, matching the two env-gated experiments already in
that file. Harness change: `streamk_env_ab.py` gained `--profile` (wired to the existing
`profile.py`; default `None` keeps the hardcoded TN HHS so every earlier run reproduces).

Methods that generalise are in `RUNBOOK.md` §4 and §7 — the inert-partition drift detector,
the `pgrep -f` self-match trap, and why a jackknife belongs in every wall-clock comparison
on this suite.
