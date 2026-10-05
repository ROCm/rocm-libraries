# Scheduler tuning parameters (gfx1250)

This page lists every tuning value of the gfx1250 (CDNA5) DAG scheduler: what it
controls, its default, how an unset value is resolved, and where to set it.
For the placement mechanisms behind the prefetch and filler knobs, see
[DAG wait and prefetch placement](../developer/dag-wait-and-prefetch-placement.md).

## Where to set a value

| Entry point | How |
|---|---|
| TensileLite | add the key to `stinky_module_options` in `Tensile/KernelWriter.py` (passed to `rocisa.toStinkyTofuModule(..., options=...)`) |
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
- **Batch window**: with `WmmaBatchSize = N`, up to N independent WMMAs issue
  back-to-back and open one window of `L + (N-1)*(L-I)` cycles (8, 15, 22, 29,
  36 for N = 1..5). With N = 1 a batch window is a WMMA window.
- **WMMA clock**: knobs measured "in WMMAs" count issued WMMAs, so they do not
  change meaning with batching.

## WMMA batch

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `WmmaBatchSize` | 0 → arch default 1 | `--wmma-batch-size=N` | Max independent, data-ready WMMAs issued back-to-back as one batch. A WMMA reading a batch member's D cannot join; any non-WMMA pick closes the batch. 1 = no batching. |

With N > 1, window-based mechanisms follow the batch window: the ds_load cap
span, the filler quota and co-issue slots, the global-read allowance, the
hide budget, and the barrier thresholds (cycles convert to WMMAs at
`batch window / N`). The knob heuristic does **not** scale with N: when you
batch, set `DsReadPerCap` explicitly.

## ds_load issue

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `DsReadPerCap` | -1 → heuristic | `--ds-read-per-cap=N` | Ceiling: at most N ds_loads in any `DsIssueCapSpanCycles` cycles (a sliding window on the real timeline). It is a wait, not a veto. Must be > 0 when set. Alias: `DsReadPerWmma` (deprecated). |
| *CLI only* `dsIssueCapSpanCycles` | 0 → one batch window | `--ds-issue-cap-span-cycles=N` | The span `DsReadPerCap` applies over. The pair is the cap: neither means anything alone. |
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

**Shared ds issue pipe:** when `HWModel::lds.wavesPerDsIssuePipe > 1`, the cap
is clamped to `span / min(NumWaves, wavesPerDsIssuePipe)`. On gfx1250 it is
currently 1 (disabled), so the clamp is inert.

## Global reads and tensor loads

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `GlobalReadQueueDepth` | 0 (off) | `--global-read-queue-depth=N` | In-flight `tensor_load_to_lds` credits. 0 disables the throttle. |
| `GlobalReadDrainLatency` | 0 | `--global-read-drain-latency=N` | Cycles until one tensor_load credit frees. |
| `TensorLoadWmmaSpace` | 0 (off) | `--tensor-load-wmma-space=N` | Extra WMMAs between exclusive after/before barrier groups: after-thresholds move N/2 WMMAs earlier, before-thresholds ceil(N/2) later. |
| `TensorLoadDsLoadGapCycles` | 64 | `--tensor-load-ds-load-gap-cycles=N` | Extra gap, in cycles, between an after-barrier and the before-side ds_loads on the gap-placement path. Rounded up to whole WMMAs. 0 = off. |

Per batch window, at most `globalReadPerWmma` (arch default 1) × the WMMAs in
the window tensor_loads issue while other work is ready; not a module option.

## Fillers and prefetches

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `EvenSpreadFillers` | true | – | Each WMMA is owed `ceil(fillers / WMMAs)` SALU/VALU fillers; a window closes once its quota (× WMMAs in the batch) is met. |
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
| `CDNA5Config` | `dsReadPerCap = 3`, `globalReadPerWmma = 1`, `tensorLoadWmmaSpace = 0`, `dsIssueCapSpanCycles = 8` (fallback WMMA latency where a region has none), `warGateWmmas = 0` (derive), `wmmaBatchSize = 1` |
| `HWModel::lds` | `readQueueDepth = 16`, `readThrottleLatency = 72`, `readDrainLatency = 0` (dynamic), `wavesPerDsIssuePipe = 1` |

gfx1250v0 uses the same scheduling defaults.

## Pattern recipes

Every pattern still respects data readiness, WAR on in-flight WMMA sources,
barriers and the ds order lock, so the output follows the pattern wherever
dependences allow.

| Pattern | Options |
|---|---|
| Default interleave `W ds ds ds W …` | none |
| Fixed batch `W×5 ds×12 fillers` | `WmmaBatchSize=5, DsReadPerCap=12, DsReadQueueDepth=16, DsReadThrottleLatency=1` |
| Batch, ds paced by the LDS queue model | `WmmaBatchSize=5, DsReadPerCap=12` |
| Batch, uncapped ds bursts | `WmmaBatchSize=5, DsReadPerCap=1000, DsReadThrottleLatency=1` |
| Fillers packed early instead of spread | `EvenSpreadFillers=false` |
| Free ds order | `LockDsReadOrder=false` |
