# Scheduler tuning parameters (gfx1250)

This page lists every tuning value of the gfx1250 (CDNA5) DAG scheduler: what it
controls, its default, how an unset value is resolved, and where to set it.
For the placement mechanisms behind the prefetch and filler knobs, see
[DAG wait and prefetch placement](../developer/dag-wait-and-prefetch-placement.md).

## Where to set a value

| Entry point | How |
|---|---|
| TensileLite | `GlobalParameters: StinkyTofuModuleOptions: {WmmaQueueDepth: 8, WmmaQueueCoverCycles: 32, DsReadPerCap: 8}` in the yaml (any module option, applied last), or add the key to `stinky_module_options` in `Tensile/KernelWriter.py` |
| `stinkytofu-opt` | the `--flag=N` listed per parameter |
| C++ pass pipeline | `PassFeatureConfig::dagFeatures.<field>` |

A few values have no module option and are reachable only from
`stinkytofu-opt` or `dagFeatures` (marked *CLI only* below).

Defaults below are the production (TensileLite → `Gfx1250Backend`) defaults.
`stinkytofu-opt` and a hand-built pipeline start from the `dagFeatures`
defaults instead, where `EvenSpreadFillers`, `DsSlotFirst`, the prefetch lead,
the wait-alu hold, the hide-budget prescan (`--enable-wmma-hide-budget-prescan`)
and the knob heuristic are all off.

### How an unset value is resolved

Each parameter resolves independently, first match wins:

1. An explicit value you set.
2. For `DsReadPerCap`, `DsReadThrottleLatency` and
   `ClusterBarrierRule3SignalLeadCycles`: the knob heuristic, computed from the
   main loop (`loopWithPrefetch`) when it has both WMMAs and ds_loads.
3. The per-arch scheduling default (`CDNA5Config`).
4. The hardware fact (`HWModel`).

## Concepts

- **WMMA window**: the `L` cycles after a WMMA issues (`L` = its
  `latencyCycles`, `I` = its `issueCycles`; `{I, L} = {1, 8}` for most gfx1250
  WMMAs). VALU may co-issue only in the window's `coIssueWindow` slots, and
  nothing may issue in a scale WMMA's blocked (LD_SCALE) slot.
- **WMMA queue**: the matrix pipe buffers about 8 WMMAs (measured on gfx1250 with
  `DISABLE_XDL_ARB_STALL` set), so a WMMA can issue while earlier ones are still
  outstanding. Queued WMMAs run one after another in the pipe, each taking its full
  `L`, and the pipe window is their segments end to end. With depth 1 a WMMA waits for
  the previous one to finish: the single-window model.
- **WMMA clock**: knobs measured "in WMMAs" count issued WMMAs, so they do not
  change meaning with the queue.

## WMMA queue

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `WmmaQueueDepth` | 1 | `--wmma-queue-depth=N` | Max WMMAs outstanding in the pipe. A WMMA is appended whenever fewer than N are outstanding; it waits only when the queue is full. 1 = one WMMA at a time (the original schedule). |
| `WmmaQueueCoverCycles` | 0 (off) | `--wmma-queue-cover-cycles=N` | Cycles of queued WMMA work that must remain before a ds_load, filler or `tensor_load` may issue. Below N, and with room in the queue, the next ready WMMA goes first, so the pipe never runs dry. Forced picks (a promoted barrier) are not held. Useful range is up to about 64 (the pipe buffers ~8 WMMAs of 8 cycles). |

The queue model runs only with **both** `WmmaQueueDepth > 1` and `WmmaQueueCoverCycles > 0`. With
either one off, the schedule is the original single-window one, exactly.

```
cover = cycles until the pipe has run everything queued
if cover >= N, or the queue is full:   issue the next ds_load / filler / tensor_load
elif a WMMA is ready:                  issue the WMMA          // refill the queue first
else:                                  issue the other pick    // never idle or deadlock
while outstanding >= depth: wait for the oldest WMMA to finish // only when a WMMA is issued
```

A larger N keeps the pipe busier but delays ds_loads; a smaller one issues ds_loads sooner
but lets the queue drain. N should cover the longest stall you want hidden: a ds_load stall is
a few cycles, a stage barrier or `tensor_load` stall is 65 to 75 cycles. When the queue model
is on, the wait pass also merges the waits of back-to-back WMMAs onto the first one, so the
run stays back-to-back. The knob heuristic does **not** scale with the queue: when you raise
the depth, set `DsReadPerCap` explicitly.

## ds_load issue

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `DsReadPerCap` | -1 → heuristic | `--ds-read-per-cap=N` | Ceiling A: at most N ds_loads per `DsIssueCapSpanCycles` cycles (how the span expires is set by `DsIssueCapMode`). It is a wait, not a veto. Must be > 0 when set. Alias: `DsReadPerWmma` (deprecated). |
| `DsIssueCapSpanCycles` | 0 → one WMMA window | `--ds-issue-cap-span-cycles=N` | The span X `DsReadPerCap` applies over. The pair is the cap: neither means anything alone. TensileLite: global parameter `StinkyTofuDsIssueCapSpanCycles`. |
| `DsIssueCapMode` | 0 (`Sliding`) | `--ds-issue-cap-mode=sliding\|periodic` | How the cap expires. `0` / `sliding`: each ds_load frees its slot X cycles after its own issue (keeps the LDS return queue from running busy). `1` / `periodic`: a period opens at its first ds_load and all slots free X cycles later, so the cap is exactly A per X-cycle period. Tight back-to-back bursts behave the same in both modes. TensileLite: global parameter `StinkyTofuDsIssueCapMode`. |
| `DsReadQueueDepth` | 0 → HW 16 | `--ds-read-queue-depth=N` | In-flight ds_load credits modeled for the LDS return queue. |
| `DsReadThrottleLatency` | -1 → heuristic | `--ds-read-throttle-latency=N` | Lifetime of one credit. A saturated queue issues one ds_load per `DsReadThrottleLatency / DsReadQueueDepth` cycles. |
| `DsReadThrottleTransitionFactor` | 1.0 | `--ds-read-throttle-transition-factor=F` | Fraction of the full throttle interval used for the first `TransitionEntries` loads past the queue depth. Clamped to [0, 1]; 1.0 = full throttle. |
| `DsReadThrottleTransitionEntries` | 0 | `--ds-read-throttle-transition-entries=N` | Loads past the queue depth that use the transition factor. 0 = no transition; negative = one queue depth. |
| `DsReadDrainLatency` | 0 → dynamic | `--ds-read-drain-latency=N` | Barrier timing only: cycles a barrier waits for its ds_loads to return. 0 derives it from the matching loads. Does not affect pacing. |
| `DsReadOrder` | -1 → `Ascending` (1) | `--ds-read-order=Name` | ds_load priority: 0 `ProgramOrder` (AABB), 1 `Ascending` (A0 B0 A1 B1), 2 `AscendingCache` (A0 B0 B1 A1). |
| `LockDsReadOrder` | true | – | ds_loads on the same memory token issue strictly in `DsReadOrder` priority. false = a ready lower-priority load may go first. |
| `DsSlotFirst` | true | `--ds-slot-first` | With 2+ ds_loads per WMMA window, a ds_load that still fits the window goes before fillers and prefetches. |

**Heuristic values** (when unset and the main loop has WMMAs and ds_loads):

```
perWmma           = min(3, ceil(dsLoads / wmmas))
DsReadPerCap      = 3 if wmmas <= 128 else perWmma
DsReadThrottleLatency = max(72, (L / perWmma) * 16)        // L = first main-loop WMMA latency
```
Without a usable main loop: `DsReadPerCap = 3`, `DsReadThrottleLatency = 72`.

**Shared ds issue pipe:** when `HWModel::lds.wavesPerDsIssuePipe > 1`,
the issue cost of each ds_load is multiplied by `min(NumWaves, wavesPerDsIssuePipe)`
(the cap itself is unchanged). On gfx1250 it is currently 1, so this is inert.

## Global reads and tensor loads

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `GlobalReadQueueDepth` | 0 (off) | `--global-read-queue-depth=N` | In-flight `tensor_load_to_lds` credits. 0 disables the throttle. |
| `GlobalReadDrainLatency` | 0 | `--global-read-drain-latency=N` | Cycles until one tensor_load credit frees. |
| `TensorLoadWmmaSpace` | 0 (off) | `--tensor-load-wmma-space=N` | Extra WMMAs between exclusive after/before barrier groups: after-thresholds move N/2 WMMAs earlier, before-thresholds ceil(N/2) later. |
| `TensorLoadDsLoadGapCycles` | 64 | `--tensor-load-ds-load-gap-cycles=N` | Extra gap, in cycles, between an after-barrier and the before-side ds_loads on the gap-placement path. Rounded up to whole WMMAs. 0 = off. |

Per WMMA window, at most `globalReadPerWmma` (arch default 1) tensor_loads issue while other work is ready; not a module option.

## Fillers and prefetches

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `EvenSpreadFillers` | true | – | Each WMMA is owed `ceil(fillers / WMMAs)` SALU/VALU fillers; a window closes once its quota is met. |
| `PrefetchLeadWmmas` | 25 (KernelWriter sets 4 without HalfPLR; 25 sub-byte A, else 40) | `--prefetch-lead-wmmas=N` | A global prefetch is held until N WMMAs before its tensor_load. Below 8 = single-stage grouping; 0 = off. |
| `PrefetchLeadMinStageWmmas` | 64 | `--prefetch-lead-min-stage-wmmas=N` | Blocks whose stages are shorter than N WMMAs run with no prefetch lead. |
| `WaitAluHoldStrictCount` | 2 (only with `EnableESM2`) | `--wait-alu-hold-strict-count=N` | A filler that would get an `s_wait_alu` of count ≤ N is held until the two windows before the next `s_barrier_wait`. < 0 = off. |

## Hazards and barriers

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `WarGateWmmas` | -1 → derived `L / I` | `--war-gate-wmmas=N` | WMMAs a ds_load waits before overwriting a vgpr a WMMA read. Active only with `EnableESM2` + `EnableESM2TrackValuVsrc`. |
| `ClusterBarrierRule3SignalLeadCycles` | -1 → heuristic | – | Cluster-barrier Rule 3 signal lead: 200 if the main loop's WMMA latency sum > 500, else 100. See [Insert cluster barrier pass](../developer/cluster-barrier.md). |
| *CLI only* `mergeBarrierThreshold` | 0 → 11 | `--merge-barrier-threshold=N` | Max cycle distance for `StinkyMergeBarrierPass` to merge adjacent barrier groups. |

## Arch defaults and hardware facts

| Source | Value (gfx1250) |
|---|---|
| `CDNA5Config` | `dsReadPerCap = 3`, `globalReadPerWmma = 1`, `tensorLoadWmmaSpace = 0`, `dsIssueCapSpanCycles = 8` (fallback WMMA latency where a region has none), `warGateWmmas = 0` (derive) |
| `HWModel::lds` | `readQueueDepth = 16`, `readThrottleLatency = 72`, `readDrainLatency = 0` (dynamic), `wavesPerDsIssuePipe = 1` |

gfx1250v0 uses the same scheduling defaults.

## Pattern recipes

Every pattern still respects data readiness, WAR on in-flight WMMA sources,
barriers and the ds order lock, so the output follows the pattern wherever
dependences allow.

| Pattern | Options |
|---|---|
| Default interleave `W ds ds ds W …` | none |
| WMMA queue kept covered, ds_loads capped at 8 per 32-cycle period | `WmmaQueueDepth=8, WmmaQueueCoverCycles=32, DsIssueCapMode=1, DsIssueCapSpanCycles=32, DsReadPerCap=8` |
| ds at a fixed rate with the LDS queue model off (measured 0.7 to 3.7% slower: bursts stall the WMMA stream) | add `DsReadThrottleLatency=1` |
| Fillers packed early instead of spread | `EvenSpreadFillers=false` |
| Free ds order | `LockDsReadOrder=false` |
| A ds_loads per fixed X-cycle period (e.g. 12 per 32) | `DsReadPerCap=12, DsIssueCapSpanCycles=32, DsIssueCapMode=1, DsReadQueueDepth=16, DsReadThrottleLatency=1` |
