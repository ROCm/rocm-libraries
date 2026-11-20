# LoopModel auto-selection (`LoopModelAutoSelect`)

`LoopModelAutoSelect` picks the LoopModel parameters a config leaves free. It chooses the loop order, the operand register rings and the TDM split. The chosen kernel must fit the VGPR budget, build correctly, and be the fastest by the search's performance model.

This README walks through what happens step by step, using the hand-tuned MXF8 kernel `config/maf/mxf8_loop_model.yaml` (gfx1250, MI 16x16x128, DepthU 256, TN) as the running example. It ends with what the model does not know yet.

## Contents

- [Quick start](#quick-start)
- [What is searched](#what-is-searched)
- [Derive mode, step by step](#derive-mode-step-by-step)
- [Expand mode, step by step](#expand-mode-step-by-step)
- [The tiers at a glance](#the-tiers-at-a-glance)
- [Worked example: the tuned MXF8 kernel](#worked-example-the-tuned-mxf8-kernel)
- [Constants and calibration](#constants-and-calibration)
- [Known limitations](#known-limitations)
- [Glossary](#glossary)
- [Module map](#module-map)
- [Tests](#tests)

Notes that go deeper than this README:

- [notes/exposed-ds-latency.md](notes/exposed-ds-latency.md): how the exposed ds_load latency is computed, step by step, on the tuned kernel.
- [notes/score.md](notes/score.md): the tier-0 score, term by term.
- [notes/selection.md](notes/selection.md): which candidates to benchmark, measured against the alternatives, and why the per-order cap stays.

## Quick start

There are three ways to run auto-selection. None needs a GPU until you benchmark.

**Derive mode** picks one kernel per fork during normal solution derivation. Add the parameter to a fork and leave the parameters you want searched at `-1`:

```yaml
ForkParameters:
  - UseLoopModel: [1]
  - LoopModelAutoSelect: [1]
  - WmmaInnerOrder: [-1]
  - PrefetchLocalReadA: [-1]
  # ... any searched parameter left at -1 or not mentioned is free
```

**Benchmark the N best** by setting `LoopModelAutoSelect: [N]` with N > 1. Before solutions are built, `BenchmarkProblems._generateForkedSolutions` replaces each such fork with the search's N best candidates (`expand.expandFork`). Each candidate is stated, with `LoopModelAutoSelect: 0`, so the run builds and times all N and the GPU picks among them. The candidates are ranked as in Derive mode, by tier 0 and checked by tiers 1 and 2, so the first is Derive mode's winner. Each candidate is also emitted, CPU only, as tier 3 does (`expand.emittedLoop`). One that Solution refuses or the writer cannot emit, typically StinkyTofu's allocation going past 1024 VGPRs, is counted as `build` or `emit` and replaced by the next in the ranking; the search ranks 4 candidates per slot asked for (`expand.REFILL`). The candidates are not re-ranked by tier 3. On the MXF8 example N = 8 skips 16 that do not emit and takes about 35 s per fork before the build.

```yaml
ForkParameters:
  - UseLoopModel: [1]
  - LoopModelAutoSelect: [8]
  - WmmaInnerOrder: [-1]
```

**Expand mode** writes a config that benchmarks the search's top candidates on the GPU:

```bash
python -m tensilelite.Utilities.loopmodel_autoselect CONFIG.yaml --out OUT.yaml \
    [--top-k 8] [--tier3 16] [--problem 0] [--size-group 0] [--arch gfx1250] \
    [--no-tier2] [--include '{"WmmaInnerOrder": 2, ...}']
```

The CLI prints the ranking with each tier's cycles. It replaces the free parameters in OUT.yaml with one `Groups:` fork of the top-k, so a normal Tensile run of OUT.yaml measures exactly those candidates. `--include` adds a group of your own, such as a hand-tuned reference. A free parameter the group leaves out is filled with the value Solution derives for it (`resolveInclude`), so a group that leaves out `WmmaFactorK` matches a ranked candidate that states the derived `WmmaFactorK: 1` and is listed once.

## What is searched

`space.SEARCHED_PARAMS` lists the searched parameters:

| Group | Parameters | Values tried |
|---|---|---|
| Loop order | `WmmaInnerOrder`, `WmmaOuterOrder`, `WmmaFactorM/N/K` | inner 1..6, outer 0..3, factors that divide the wave tile (M, N) or `LoopIters` (K), up to 16 |
| Registers, per operand | `PrefetchLocalReadA/B`, `ClusterLocalReadA/B`, `VgprGroupA/B`, `VgprAllocA/B` | VG divides the prefetch unit, up to 8 (`ValidParameters`); PLR 0..8; CLR 0 or 1; with CLR 0, VA from PLR + 1 up to the ring positions × VG (at most 16) |
| TDM split | `TDMSplitA/B` | 0, 1, 2 |

A parameter is **free** when the config sets it to `-1` or does not mention it (`space.freeParams`). A stated value is kept and searched around. To pin a parameter, state it.

The `-1` spellings this needs are valid only alongside `LoopModelAutoSelect`. `WmmaInnerOrder: -1` alone means KMN order. `TDMSplitA/B: -1` alone means 0.

## Derive mode, step by step

`Solution.assignDerivedParameters` calls `derive.autoSelect` when `LoopModelAutoSelect` is 1.

### Step 1: Check the request

`autoSelect` keeps a pristine copy of the state with `LoopModelAutoSelect` cleared. It requires `UseLoopModel`. If no searched parameter is free, the state is derived as usual and nothing is searched.

### Step 2: Derive a starting state

The search needs a derived Solution: the resolved tile, `LocalReadVectorWidth`, `MIInputPerThread`, `LoopIters` and the rest. Solution therefore derives a quiet copy with every free parameter at its usual default. This copy is only the search's starting point and never becomes the kernel.

A free TDM split changes what Solution derives: the LDS layout, a factor it pins, and two SGPRs. So each split pair `(TDMSplitA, TDMSplitB)` gets its own starting derivation and its own search, and the rankings are merged (`derive.rankSplits`). A pair Solution refuses is skipped and counted as `split`.

### Step 3: Tier 0, score every order and register pair

This is `search._tier0`, using `perf.NestWalk`, `perf.sideCost` and `perf.orderCore`. No θ is built yet.

1. **Enumerate loop orders** (`space.orderChoices`). Two parameter sets that produce the same `LoopOrder` word and factors are one order. Nests that differ only by a factor of 1 are also merged (`search._walks`), and a spelling without an extent-1 axis represents the nest when one exists.
2. **Walk the nest.** For one trip of the steady loop, `NestWalk` lists every WMMA in order. For each operand unit it records the first and last WMMA that uses it. A unit is the tiles one prefetch loads, or a VgprGroup sub-unit of them.
3. **Check each ring for correctness** (`NestWalk.distances`). A CLR=0 ring of `VgprAlloc` buffers hands out slots in first-use order. A unit's load wants to issue `PLR` units early, but cannot issue before the last WMMA reading the unit that held its slot. That is the use-to-load (write-after-read) dependency. If the slot is not free by the unit's own first use, the ring would read the wrong tile. The choice is refused, as GIR would refuse it with WRONG SOURCE. Otherwise the walk gives each unit's load-to-use distance in WMMAs.
4. **Count VGPRs.** Each operand's ring needs `unit / VG × VA × (fragment registers + 1 for the MX scale when MX)` VGPRs. Registers are counted without alignment, because StinkyTofu's register allocator packs them. The budget is `1024 − fixed − accumulator`.
5. **Price each operand** (`sideCost`, `NestWalk.exposure`): the exposed ds latency is the time the WMMAs wait on the operand's loads on an in-order clock. A unit waits `max(0, latency − distance × WMMA cycles)`, less the waits that already passed while its load was in flight, so a burst of late refills pays about one latency rather than one each. [notes/exposed-ds-latency.md](notes/exposed-ds-latency.md) works it through on the tuned kernel.
6. **Keep the useful choices per operand** (`_sideOptions`). Only choices that no other choice beats on registers, exposure, units and copies together are kept.
7. **Pair A with B under the budget and score each pair:**

   ```text
   score = max(C_wmma, C_lds) + exposed_A + exposed_B + tdm + tdmExposed + tripOverhead + icache
   ```

   - `C_wmma` is WMMAs per trip × the WMMA's matrix-queue cycles.
   - `C_lds` is ds_loads per trip × the LDS return rate.
   - `tdm` is tensor loads per trip (`perf.tensorLoads`: one per storage region of each descriptor set, so 2 unsplit and 3 or 4 split) × `tdmIssueCycles`. It is the same for every order of one split pair, so it only ranks the split pairs against each other.
   - `tdmExposed` (`perf.tdmExposed`) is the tensor-load latency the trip leaves showing: all of `tdmLatency` for a one-copy steady body, which drains `tensorcnt` every trip, and `tdmLatency` less one trip otherwise. See [notes/score.md](notes/score.md#why-a-one-copy-body-pays-the-whole-tensor-load-latency).
   - `tripOverhead` covers the branch and barrier.
   - `icache` applies only when the unrolled body exceeds the I-cache threshold.

   [notes/score.md](notes/score.md) gives each term's formula and the tuned kernel's 1377.
8. **Rank.** Each order keeps its 16 best pairs (`PER_ORDER`). All are sorted by the `Tier0` tuple: score, then fewer units loaded per trip, then fewer steady-body copies, then fewer VGPRs, then canonical order.

### Step 4: Tier 1, the exact checks

`evaluate.tier1` runs best first, on at most 400 candidates (`MAX_TIER1`). It applies the candidate as **stated** values, exactly as a config naming them would. That matters because stated values turn off the no-load loop (`SuppressNoLoadLoop`). It then runs Solution's and the kernel writer's checks, builds θ, and counts the exact VGPRs, including the MX scale rings. A failure is counted under its stage, for example `budget`.

Tiers 1 and 2 take at most 2 candidates from each order choice (`TIER2_PER_ORDER`; an order choice is the inner and outer order with the WMMA factors) and lower at most 48 (`MAX_LOWERINGS`). The search stops at `topK` survivors, continuing through ties with the last place. `search.search` takes `perOrder`, `tier0PerOrder`, `band` (keep every candidate within that fraction of the best score instead of stopping at `topK`) and `maxLowerings` to widen this; the defaults are the constants. [notes/selection.md](notes/selection.md) measured the cap: it almost never binds inside the tied band.

### Step 5: Tier 2, the correctness gate

The candidate's θ is decoded and lowered to GIR through the production pass pipeline (`perf.lower`). GIR's dataflow checks are the verdict:

- WRONG SOURCE: a WMMA reads a register holding a different tile.
- OVERWRITE-BEFORE-USE: a refill takes a slot back before its value is read.
- RING-DOES-NOT-ROTATE: an unrolled copy does not advance a ring.

A refused candidate is counted as `gir`. A candidate whose steady body would need more than 6 copies is not lowered and is counted as `copies`.

The lowered body is also simulated (`perf.simulate`). Those cycles are **reported, not ranked on**, because they ordered measured kernels worse than tier 0 did. The simulation charges each tensor load `tdmIssueCycles`, makes a `ds_load` of an LDS generation wait for the tensor load filling it (`tdmLatency` after issue), and makes that load wait for the last read of the buffer's previous generation. A generation is the shared ref's `gdelta` plus the steady copies before it.

Lowering is about 0.6 s per candidate, nearly all of the search time. Derive mode therefore asks for the winner alone (`topK=1`) and stops at the first candidate that lowers. It continues only through candidates tied with that one on score, units and copies, because the ranking breaks those ties on the exact VGPR count, which only tier 1 knows. The result is the same winner as a full ranking.

### Step 6: Derive the winner for real

The pristine state is restored and the winner's values are written in as if the config had stated them. Solution then derives it normally. The final Solution carries no trace of the search: `LoopModelAutoSelect` is 0, and a config naming the same values builds the same kernel.

If Solution refuses the winner, the full 16-candidate search runs and up to 4 untried finalists are derived in turn (`MAX_FINALISTS`). If nothing is buildable, the Solution is rejected with the most common refusal and an example of it (`AutoResult.reason`).

## Expand mode, step by step

`Utilities/loopmodel_autoselect.py`:

1. **Read the config.** Its forks are enumerated with the same `BenchmarkProcess` code a Tensile run uses (`--problem`, `--size-group`).
2. **Search each fork.** Steps 2–5 above run for every fork that leaves a parameter free, keeping the top `--top-k` (default 8) instead of the winner alone.
3. **Tier 3: time the emitted loop.** With `--tier3 N` (default 16; 0 turns it off), each of the first N candidates is built and emitted, CPU only. `asmsim.loopCycles` then times the body between `label_LoopBeginL:` and its back edge, in order:
   - each `v_wmma*` holds the matrix queue;
   - `ds_load*` returns after `asmDsLatency`, no sooner than `dsCyclesPerLoad` after the previous return;
   - `s_wait_dscnt N` and `s_wait_tensorcnt N` block until at most N loads are outstanding;
   - each `tensor_load_to_lds` lands after `asmTdmLatency`;
   - barriers cost their signal-to-wait latency.

   The body runs several times and the last copy is timed. Timed candidates are re-ranked by these cycles. A candidate Solution refuses or the writer cannot emit, for example because StinkyTofu's allocation needs more than 1024 VGPRs, is rejected and counted as `emit`. Candidates past the first N keep their tier-0 place after the timed ones.
4. **Write OUT.yaml.** The free single-key forks and `LoopModelAutoSelect` are removed, and one `Groups:` fork lists the ranked candidates, with `--include` groups first unless the search already ranks the same kernel.
5. **Benchmark** OUT.yaml with Tensile as usual. This is the only GPU step.

## The tiers at a glance

| Tier | Where | Runs on | Decides | Cost on the MXF8 example |
|---|---|---|---|---|
| 0 | `search._tier0`, `perf.NestWalk` / `sideCost` / `orderCore` | every order × register pair (662) | ranking, ring correctness, approximate VGPR budget | well under 1 s total |
| 1 | `evaluate.tier1` | best first, ≤ 400 | Solution validity, exact VGPRs | a few ms each |
| 2 | `perf.lower`, `perf.simulate` | tier-1 survivors | correctness (GIR); cycles reported only | ~0.6 s each |
| 3 | `asmsim`, Expand CLI; the emit alone in `LoopModelAutoSelect: N` | top N | ranking by the scheduled loop | ~5 s each |

Typical wall time on the MXF8 example:

- Derive mode: about 1.3 s of search; the whole derivation takes about 3 s.
- `LoopModelAutoSelect: 8`: about 35 s per fork, mostly the emits that skip unbuildable candidates.
- Expand mode with defaults: about 80–150 s, mostly the 16 tier-3 emits.

Derive mode cannot run tier 3, because emitting a kernel inside Solution derivation would be circular. `LoopModelAutoSelect: N` emits before the benchmark's solutions are built, so it can.

## Worked example: the tuned MXF8 kernel

The tuned kernel is defined as follows:

| Parameter | Value |
|---|---|
| MI | `[16,16,128,1,1,8,10,2,2]` |
| MT | 256×320 |
| DepthU | 256 |
| Loop order | WIO 2 / WOO 3 / WFM 2 / WFN 2, i.e. `KKNMNM`, axes `[Ki2, No2, Mo2, Ni5, Mi4]`, 160 WMMAs per trip |
| A ring | PLR 2, CLR 0, VG 1, VA 3 |
| B ring | PLR 1, CLR 0, VG 1, VA 2 |
| TDM | TDMSplit 0, TDMFuse 1 |
| `StinkyTofuRegisterAllocation` | 3 |

**Nest walk.** A's four units are used at WMMAs (0–59), (20–79), (80–139) and (100–159). Each A unit comes back for the second N-outer step. B's units are used once each: (0–39), (40–79), (80–119), (120–159).

**Ring check.** A's ring of 3 with PLR 2 gives distances [80, 40, 80, 40]. B's ring of 2 with PLR 1 gives 40 for each unit. A ring of 1 buffer for A is refused, because a slot is refilled before its tile's second use. So is VG 4 with PLR 4 and VA 5. GIR refuses exactly the same rings.

**Budget.** The rings take A 192, B 160, MX scales 12 + 10, the accumulator 640 and fixed registers 10, for exactly 1024. The emitted kernel uses `.amdhsa_next_free_vgpr` 1023. One more A buffer (VA 4) is over budget. WFN 1 needs 1184 VGPRs (1193 in the real kernel) and is refused.

**Score.** At 8 cycles per WMMA, `C_wmma` is 160 × 8 = 1280; `C_lds` is smaller. Every distance × 8 is at least the 300-cycle effective latency, so nothing is exposed, and the two tensor loads cost 2 × 40. The body has 3 copies, so one 1280-cycle trip covers the 800-cycle tensor-load latency: 1280 + 0 + 0 + 80 + 0 + 17 = **1377**. That is the floor, and with the register parameters free the tuned kernel is alone at it; the one-copy choices that also expose nothing (A CLR 1) pay the full 800.

**Tier 3 against the GPU.** 27 variants were measured on one gfx1250 card (4096×5120×65536, GFLOPS ×1e6):

| Kernel | Tier 0 | Tier 3 cycles | Measured |
|---|---|---|---|
| KKMNMN WFN5, PLRA1/VAA2, PLRB5/VAB6 | 1377 | 1919 | **9.355** |
| KKNMNM VGA2, PLRA4/VAA5 | 1809 | 1938 | 9.172 |
| KKNMNM WFM4, PLRA4/VAA5 | 1377 | 1960 | 8.897 |
| tuned | 1377 | 1961 | 9.330 |
| WOO0 (KNM) | 1849 | 2137 | 8.886 |
| VGA4 KKMNMN | 1377 | 2517 | 7.414 |
| WOO0 WFN1, VGB5 | 1849 | 7105 | 5.941 |

Tier 3 ranks the 27 at Spearman +0.90 against GFLOPS. Its three fastest predictions are the three fastest measured. Tier 0 ranks them at +0.44: it scores VGA4 at the floor, yet the kernel is 20% slower.

## Constants and calibration

`hw.Gfx1250Costs` holds every cost. `hw.calibration` records the data each fitted value came from.

| Constant | Value | Source |
|---|---|---|
| `wmmaCycles` | 8 for 16×16×128 | copied from StinkyTofu `Gfx1250Formats.def` |
| `dsLatency` | 56 for `ds_load_b128` | copied from StinkyTofu `Gfx1250Instructions.def` |
| `dsCyclesPerLoad` | 6 | copied from StinkyTofu, the DAG scheduler's `DsReadCyclesPerLoad` |
| `dsQueueDepth` | 16 | copied from StinkyTofu `HWModel` `lds.readQueueDepth` |
| `tripOverhead` | 17 | copied from StinkyTofu: barrier signal-to-wait 11 + jump 6 |
| `dsEffectiveLatency` | 300 | **fitted** to 17 measured kernels |
| `asmDsLatency` / `asmTdmLatency` | 250 / 2500 | **fitted** to 27 measured kernels |
| `tdmIssueCycles` | 40 | **measured**: median slowdown per extra tensor load over 13 split variants |
| `tdmLatency` | 800 | **fitted** for tier 2 to 40 measured kernels; a latency from GIR's load position, not the hardware's. Tier 0's one-copy term uses it too: 600 and up rank 6 data-type / tile variants within 0.02 of the best |
| `fixedVgpr`, `fixedSgpr` | 10, 86 | fitted to emitted kernels' register counts |

`perf.drainLatency` reimplements StinkyTofu's `computeDynamicDrainLatency`. It is used as a floor under `dsEffectiveLatency`.

The fitted latencies come from one MI family and may not carry over to other data types or tile sizes.

## Known limitations

**Tier 0's 300-cycle latency is not ds latency.** With StinkyTofu's own LDS numbers instead, tier 0 exposes nothing on any of the 27 measured kernels: about 158 loads × 6 cycles is less than 1280 WMMA cycles, and even the shortest distance, 20 WMMAs, covers 56 cycles. Every kernel then scores the 1377 floor. The tie-break would pick PLRA1/VAA2 (measured 8.67) over the tuned kernel (9.33).

**The missing dependency is the tensor load (TDM).** A `ds_load` can only read a K-slice after its `tensor_load_to_lds` has landed in LDS and the barrier has passed. The next tensor load cannot overwrite that LDS buffer until the `ds_load`s reading it complete. Tier 2 models both steps from GIR's dependences, and tiers 0 and 2 charge each tensor load's issue. Neither sees the wait depth that decides the cost (below); tier 0 approximates it from the steady-body copies only (`tdmExposed`).

Removing the waits from tier 3's simulation shows the effect, at StinkyTofu's ds latency of 56:

| Waits kept | Spearman |
|---|---|
| all | +0.82 |
| all but `s_wait_dscnt` | +0.79 |
| all but `s_wait_tensorcnt` | +0.585 |

The fast kernels keep two tensor loads in flight (`s_wait_tensorcnt 2`). The slowest drain them every other copy (`s_wait_tensorcnt 1`, then `0`).

**The wait depth is set after scheduling.** On 40 measured kernels (27 unsplit, 13 split), the emitted slack, the WMMAs from a tensor load to the `s_wait_tensorcnt` that retires it, ranks GFLOPS at +0.57. It is about 190 on the tuned kernel, 126 when the wait comes just before the trip's loads (`tensorcnt 0`), and 34 to 80 on the slowest kernels. GIR's slack does not predict it (Spearman −0.21), even with every load hoisted to its buffer's last read. StinkyTofu moves the loads (GIR places the tuned kernel's at WMMA 137, the emitted loop at 76 and 96), and `StinkyWaitCntInsertionPass` counts the waits on the scheduled loop, conservatively: a one-copy steady body waits for `tensorcnt 0` every trip, where 2 would do.

So tier 2's `tdmLatency` is fitted to GIR's positions. From 600 to 1000 cycles only the two slowest kernels wait (WOO0 WFN1, whose tensor load GIR places 31 WMMAs before its first read), and Spearman rises from −0.07 to +0.05. From 1100 cycles most kernels wait, and it falls to −0.30 at 2000. Tier 0 moves from +0.54 to +0.52 overall, and it now orders 75% of the 22 same-configuration split pairs correctly (all tied before). Tier 3 ranks the same 40 at +0.82.

**StinkyTofu has no TDM latency.** `EstimateAsmCyclesPass` charges `tensor_load_to_lds` and `s_wait_tensorcnt` only their issue cycles; the code has a TODO. Its published loop estimate (`EstimateAsmCyclesPass.totalCycles`) also cannot be read from Python today: rocisa and the `stinkytofu` package are separate nanobind modules. `asmTdmLatency` should come from a `tensor_load_to_lds` microbenchmark rather than a fit.

**Tier 0 undercounts θ's registers on degenerate units.** A WMMA factor equal to an axis's full extent leaves the inner part with extent 1. For example, B in `KKMNMN` with factors M1, N10, K1 walks `Ki`2, `No`10, `Mi`8, `Ni`1. The factor axis `No` becomes a region and is stripped from the prefetch unit, which leaves the unit as `Ni`1, one tile. θ calls this a degenerate unit (`register_unit_modes` in `LoopModel/traversal.py`). θ widens it back to `original_prefetch` (`No`10 × `Ni`1, ten tiles per slot) when all of these hold:

1. The operand has factor axes, and its unit has extent 1.
2. The un-widened ring has more positions per K chunk than VA. Positions are the unit-sized pieces one chunk walks, here `Ki`2 × `No`10 = 20.
3. The positions divide by VA, so every trip starts on the same ring phase.

θ's reason, from its comments, is that a region moves the load address but not the register index. Two region values could therefore share a register on every trip. At VA 5 the widening takes B from 160 to 800 VGPRs (5 × 10 × 16). At VA 3 nothing changes, because 20 does not divide by 3. Tier 0 keeps the one-tile unit, so this is the source of the largest tier 0 vs tier 1 mismatches: 43 of about 25,600 register choices for the MXF8 config.

The widening has not been shown to be needed. With it turned off, the same kernel at VA 5 and 160 VGPRs passes GIR's register-dataflow, ring-rotation and region-coverage checks. The ring keeps the `No` values apart in time. Only a GPU result check of that kernel would settle it. Until then the rule makes θ, and through it tier 1, reject or overcharge these candidates.

Tier 0 represents each walk by a clean spelling when one exists. For 17 of the config's 150 walks the first spelling in canonical order is degenerate though a clean one exists: `MMKNKN` (M1, N10, K1) walks the same tiles as `MKN` (M1, N1, K1), and `MKN` is the one searched. The other 16 degenerate walks can only be spelled degenerately. `LoopIters` is 2, so any word with two K letters splits K into 2 × 1. The tuned kernel's `KKNMNM` (M2, N2, K1) is not degenerate.

The tier 0 register count also lacks θ's K-step floor: an operand always holds at least one K-step of its tiles (`free_tiles` × fragment VGPRs) for the tail loop. Adding the floor makes tier 0's operand data count match θ on 99.8% of choices. The remainder are the degenerate units above.

**Other gaps:**

- **Register refill timing.** Tier 0 frees a ring slot one WMMA position after its last reader. It does not add cycle-level operand-read time or hazard wait states, and a stalled WMMA does not delay the slots it frees.
- **StinkyTofu reschedules.** The real load-to-use distances differ from the walk's.
- **Small wave tiles.** With a 4×4 wave tile a trip (32 WMMAs) is far shorter than the tensor-load latency. `tdmExposed` still charges a one-copy body 256 more than a multi-copy one (800 vs 544 on BF16 4×4), and that is backwards there: the one-copy kernels with both operands resident (CLR 1) run at 0.97 PF against 0.79 for every multi-copy kernel tier 0 ties at its best. They score 20% above the best and so are never benchmarked; tier 2 ranks them first (408–422 cycles against 832). See [notes/selection.md](notes/selection.md). BF16 4×4 is ranked backwards by every tier (tier 0 −0.63, tier 3 −0.18): its slowest kernels prefetch all of the next trip's A and B past v255 and switch `s_set_vgpr_msb` inside the WMMA chain, which no tier models.
- **TDM split ranking.** Tiers 0 and 2 rank a split below its unsplit kernel by the extra loads' issue only. One of the 13 measured splits is faster than its unsplit kernel (WFM1 PLR4/1 with B split, +11%), because splitting changed its emitted waits from `tensorcnt 0, 0` to `3, 1`. Only tier 3 sees that.
- **Calibration coverage.** The constants have been checked on one MI family only.

## Glossary

### Kernel parameters

- **`LoopOrder`**: the WMMA loop-nest word, such as `KKNMNM`. Each letter is one loop. A doubled letter is a dimension split into an outer and an inner part. `WmmaInnerOrder` and `WmmaOuterOrder` choose the word.
- **`WmmaFactorM/N/K` (WFM, WFN, WFK)**: how a split dimension divides between its outer and inner parts. The outer part becomes a region axis.
- **`MIWaveTile`**: WMMA tiles per wave in M and N; 8×10 for the tuned kernel.
- **`LoopIters`**: K steps per loop trip, `DepthU` / MI K. It is 2 for DepthU 256 with MI K 128.
- **PLR (`PrefetchLocalRead`)**: how many prefetch units ahead of their use the `ds_load`s issue.
- **VA (`VgprAlloc`)**: the ring depth, the number of buffers of one prefetch unit.
- **VG (`VgprGroup`)**: the factor the prefetch unit is divided by, giving more, smaller slots.
- **CLR (`ClusterLocalRead`)**: 1 keeps the operand's whole K-step resident, instead of rotating a ring.
- **`TDMSplitA/B`**: how an operand's tensor load is split into regions.
- **`StinkyTofuRegisterAllocation`**: turns on StinkyTofu's register allocator; 3 in the tuned kernel.

### Loop nest

- **Operand**: one input of the GEMM: A, B, or the MX scales MXSA and MXSB.
- **Tile**: one WMMA input fragment of an operand. Its size in VGPRs is the fragment size (16 for mxf8).
- **Axis and extent**: one loop of the nest (`Mo`, `Mi`, `No`, `Ni`, `Ko`, `Ki`) and its number of steps.
- **Extent-1 axis**: a loop with one step, left behind when a factor takes a dimension's whole extent.
- **Degenerate spelling**: an order or factor choice whose nest has an extent-1 axis.
- **Region axis (factor axis)**: the outer part a WMMA factor creates. It moves the load address, but is not part of the register index.
- **Nest or walk**: the WMMA sequence of one trip, outermost loop first. Tier 0 scores the walk (`NestWalk`).
- **Spelling**: one parameter set that produces a walk. Several spellings can give one walk; tier 0 keeps the first without an extent-1 axis, or the first when every spelling has one.
- **Trip or chunk**: one iteration of the steady loop, which covers DepthU of K.
- **Copies**: how many trips the steady loop is unrolled into, so that every ring returns to its starting slot.

### Register rings

- **Prefetch unit**: the tiles one ring slot receives together.
- **Slot**: one buffer of the ring, holding one prefetch unit.
- **Ring**: the VA slots an operand rotates through. A slot is refilled once its previous unit has been read.
- **Position**: one unit-sized piece of data the walk visits in a chunk.
- **Ring period**: the number of positions in one chunk (`_ring_period`). Slot = running position mod VA.
- **Ring axes**: the axes that move the ring offset, the varying axes not in the unit.
- **Read-ahead and refill**: a refill is the load that puts a new unit into a slot. With read-ahead, the load issues PLR units before the unit's first use.
- **Write-after-read (ring liveness)**: a refill cannot issue before the last WMMA reading the slot's previous unit. Tier 0's `NestWalk.distances` checks this.
- **K-step floor**: an operand holds at least one full K-step of its tiles for the tail loop, `free_tiles` × fragment VGPRs.
- **Degenerate unit**: a unit left as one tile after the region axis is stripped. θ widens it in some cases; see [Known limitations](#known-limitations).
- **Read merge, fold and carrier**: one `ds_load` may land several adjacent tiles, called a fold. The carrier is the register run that receives it. A slot never holds less than one fold.
- **Generation**: one value of a rotating buffer: which fill of the ring the data belongs to.

### θ and lowering

- **θ (`Theta`, `LoopModel/theta.py`)**: the full LoopModel description of one kernel. It holds the nest (`ord`) and, for each operand, an `OperandGeometry` and a `Trajectory`. `Components/LoopModel/Theta.py` builds it from the kernel parameters.
- **`Trajectory` and `Movement`**: an operand's path, global memory to LDS to registers, one hop per movement.
- **`Fragment`**: an operand's register placement: groups, ring depths (VA), VG and read-ahead.
- **LoopIR**: the loop program `decode_theta` builds from θ. Refills are placed among the WMMAs here (`_schedule_refill_run`).
- **GIR**: the GEMM dataflow IR (`Lowering/gir`). Its nodes include `Move` (one hop down the memory hierarchy), `Mma` (compute) and `Mark` (a fact such as a wait or a ring swap, with no data movement). Each value names the operation that produced it. `run_pipeline` checks the dataflow. Tier 2 is lowering to GIR.
- **Block and phase**: GIR's control flow. Each software-pipeline phase is a block: prologue, steady loop and drain.
- **`PendingMark` and `Region`**: a mark not yet placed, and the window where it may legally go. `PlacementPass` picks a spot in that window.
- **Frame**: the ring phase a block executes in. The loop repeats after a finite number of frames, which bounds StinkyTofu's analyses.
- **GIR tag**: a `<GIR: ...>` comment on an emitted instruction. It carries GIR's producer and consumer facts to StinkyTofu.

### Synchronization

- **TDM and `tensor_load_to_lds`**: the tensor data mover copies a global tile straight into LDS, counted by `tensorcnt`.
- **`ds_load`**: an LDS-to-VGPR read, counted by `dscnt`.
- **Hazards**: RAW, read after write: a `ds_load` must not read LDS before the tensor load has landed. WAR, write after read: the next tensor load must not overwrite an LDS buffer that other waves are still reading. WAW, write after write: two writes to one buffer must stay in order.
- **Fence**: a workgroup barrier (`s_barrier`) placed before an instruction to settle these hazards across waves. Every wave must pass it before the guarded instruction runs. `GirFencePlacementPass` decides where fences go, within each one's legal window.
- **Wait (`s_wait_tensorcnt N`, `s_wait_dscnt N`)**: stalls the wave until at most N loads of that kind are still in flight. Waits are inserted after scheduling. A fence needs a wait before it so that the data has really landed. The fast mxf8 kernels keep `s_wait_tensorcnt 2`.

### StinkyTofu passes

These run in this order for gfx1250 (`shared/stinkytofu/src/pipeline/backend/Gfx1250Backend.cpp`):

1. **`GirFrameContractImportPass`**: reads the GIR tags.
2. **`GirFencePlacementPass`**: places the fences. It takes about 3.2 s on the tuned kernel, most of the time it takes to generate the kernel.
3. **`StinkyDAGSchedulerPass`**: the instruction scheduler. It reorders each region from a dependency DAG, without moving anything across a GIR fence.
4. **`StinkyWaitCntInsertionPass` and `GirWaitCntInsertionPass`**: insert the waits.
5. **`EstimateAsmCyclesPass`**: StinkyTofu's own cycle estimate. It charges tensor loads only their issue cycles.

### Auto-selection

- **Derive and Expand modes**: Derive mode runs inside Solution derivation and picks one kernel. `LoopModelAutoSelect: N` benchmarks the N best in the same Tensile run. Expand mode is the CLI and lists the top candidates, re-ranked by tier 3, for benchmarking.
- **Free parameter**: a searched parameter left at -1. The search picks it.
- **Tiers 0 to 3**: the four tiers, from fastest to most accurate:
  - Tier 0 scores the nest walk.
  - Tier 1 runs the exact Solution checks and the VGPR count.
  - Tier 2 is the GIR correctness gate.
  - Tier 3 times the emitted loop (`asmsim`).
- **Score**: tier 0's cycles per trip, `max(cWmma, cLds) + exposedA + exposedB + tdm + tdmExposed + tripOverhead + icache`.
- **Exposure**: the load latency a unit's load-to-use distance does not hide. It is added per operand.
- **Side options (`_sideOptions`)**: one operand's register choices that no other choice beats on registers, exposure, units and copies together, a Pareto filter.
- **Finalist**: a top-ranked candidate Derive mode tries to build for real.
- **`fixedVgpr`**: a constant 10 VGPRs added for everything outside the operand rings and the accumulator.
- **`dsEffectiveLatency`, `asmDsLatency`, `asmTdmLatency`**: fitted latencies for tier 0 and tier 3; see [Constants and calibration](#constants-and-calibration).

## Module map

| Module | Role |
|---|---|
| `space.py` | the searched parameters, their value domains, `freeParams`, `registerShapes` |
| `perf.py` | tier 0 (`NestWalk`, `sideCost`, `orderCore`), tier 1's analytic estimate, tier 2's lowering and timeline |
| `evaluate.py` | applying an order; tier 1 (`tier1`): stated derivation, θ, exact registers |
| `resources.py` | VGPR and SGPR counts without alignment |
| `search.py` | the order and register search: tiers 0–2, limits, ranking |
| `asmsim.py` | tier 3: emitted-loop extraction and in-order timing |
| `derive.py` | Derive mode: split pairs, the winner-alone search, the final derivation and fallbacks |
| `expand.py` | `LoopModelAutoSelect: N`: a fork's N best candidates that emit, stated, for the benchmark; `emittedLoop`, the emit tier 3 times |
| `report.py` | `Ranked`, `AutoResult`: what was chosen, why, and the refusal counts |
| `hw.py` | gfx1250 costs and their calibration notes |

Outside this package:

- `SolutionStructs/Solution.py` hooks Derive mode into `assignDerivedParameters`.
- `BenchmarkProblems._generateForkedSolutions` expands `LoopModelAutoSelect: N` forks before building them.
- `Components/LoopModel/Order.py` gives `WmmaInnerOrder: -1` its default meaning.
- `Utilities/loopmodel_autoselect.py` is the Expand CLI and tier 3.

## Tests

`Tests/unit/test_loopmodel_auto.py` runs on the CPU, about 45 s, against the tuned kernel in `Tests/unit/characterization/_codegen/data/test_data/_designed/gfx1250/loopmodel_mxf8_autoselect.yaml`. It pins:

- the nest walk;
- ring correctness against GIR's verdicts;
- that a free CLR 0 ring is offered only with VA > PLR;
- that each nest is searched in a spelling without an extent-1 axis when one exists;
- that tier 3 rejects a candidate it cannot emit;
- the tensor loads per trip of each split pair (2, 3, 3, 4), that tier 0 charges each, and that tier 2 waits for a late one;
- the VGPR budget: exactly 1024, VA 4 over budget, WFN1 at 1184;
- that the free search picks the tuned kernel;
- that a winner-alone search gives the same winner;
- tier 3's wait handling;
- that tier 3 times the tuned kernel ahead of a shorter A ring (9.33 vs 8.67 measured);
- determinism;
- that every lowered candidate passes GIR;
- that Derive mode states its choice in the final Solution;
- that `LoopModelAutoSelect: N` yields N distinct stated kernels that emit, the first of them Derive mode's winner.
