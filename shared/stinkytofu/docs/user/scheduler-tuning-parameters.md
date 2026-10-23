# Scheduler tuning parameters (gfx1250)

This page lists every tuning value of the gfx1250 (CDNA5) DAG scheduler: what it
controls, its default, how an unset value is resolved, and where to set it.
For the placement mechanisms behind the prefetch and filler knobs, see
[DAG wait and prefetch placement](../developer/dag-wait-and-prefetch-placement.md).

## Where to set a value

| Entry point | How |
|---|---|
| TensileLite | `GlobalParameters: StinkyTofuModuleOptions: {WmmaQueueDepth: 8, WmmaQueueCoverCycles: 32, WmmaBatchSize: 2}` in the yaml (any module option, applied last), or add the key to `stinky_module_options` in `Tensile/KernelWriter.py` |
| `stinkytofu-opt` | the `--flag=N` listed per parameter |
| C++ pass pipeline | `PassFeatureConfig::dagFeatures.<field>` |

A few values have no module option and are reachable only from
`stinkytofu-opt` or `dagFeatures` (marked *CLI only* below).

Defaults below are the production (TensileLite → `Gfx1250Backend`) defaults.
`stinkytofu-opt` and a hand-built pipeline start from the `dagFeatures`
defaults instead, where `EvenSpreadFillers`, the prefetch lead,
the wait-alu hold and the knob heuristic are off, `DsSpread` is on, and the WMMA queue model is
off (`WmmaQueueDepth` 1, `WmmaQueueCoverCycles` 0, `DsReadCyclesPerLoad` 0). The production defaults
turn the queue on (8, 56, 6), so `stinkytofu-opt` output matches TensileLite only with those three
set.

### How an unset value is resolved

Each parameter resolves independently, first match wins:

1. An explicit value you set.
2. For `ClusterBarrierRule3SignalLeadCycles`: the knob heuristic, computed from the
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
| `WmmaQueueDepth` | 8 (`stinkytofu-opt`: 1) | `--wmma-queue-depth=N` | Max WMMAs outstanding in the pipe. A WMMA is appended whenever fewer than N are outstanding; it waits only when the queue is full. 1 = one WMMA at a time (queue model off). |
| `WmmaQueueCoverCycles` | 56 (`stinkytofu-opt`: 0 = off) | `--wmma-queue-cover-cycles=N` | Cycles of queued WMMA work that must remain before a ds_load, filler or `tensor_load` may issue. Below N, and with room in the queue, the next ready WMMA goes first, so the pipe never runs dry. Forced picks (a promoted barrier) are not held. Useful range is up to about 64 (the pipe buffers ~8 WMMAs of 8 cycles). |

The queue model runs only with **both** `WmmaQueueDepth > 1` and `WmmaQueueCoverCycles > 0`, which
is the production default. With either one off, the schedule is the original single-window one.

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
run stays back-to-back.

### WMMA refill (batches)

The queue is refilled in batches. A refill starts when the queued cover drops below the refill
line (`WmmaQueueCoverCycles`) and the queue has room. It then issues WMMAs ahead of other work
until it is done. A full WMMA queue only makes the next WMMA wait, which costs no efficiency, so
the queue is kept busy and covers the small stalls of the ds_loads and barriers between batches.

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `WmmaRefillMode` | 0 (`Fixed`) | `--wmma-refill-mode=fixed\|dynamic` | `0` / `fixed`: each refill issues `WmmaBatchSize` WMMAs. `1` / `dynamic`: each refill issues WMMAs until the queued cover reaches `WmmaRefillToCycles`; the count follows the live timeline and the WMMA latency. |
| `WmmaBatchSize` | 1 | `--wmma-batch-size=N` | Fixed mode: WMMAs per refill. 1 is the one-WMMA-at-a-time refill. |
| `WmmaRefillToCycles` | 0 → full queue | `--wmma-refill-to-cycles=N` | Dynamic mode: the queued cover (cycles) a refill fills up to. 0 = `WmmaQueueDepth` × the WMMA latency. |

A refill also ends when the queue is full. A dependent WMMA that needs coexec padding is held as
before.

## ds_load pacing

The ds_loads of a region are paced by the ds credit queue and, with `DsSpread`, spread evenly over
the region's WMMAs. This replaces the old cap / throttle / slot-first / hide-budget pacing.

| Module option | Default | CLI | Meaning |
|---|---|---|---|
| `DsSpread` | true | `--ds-spread` / `--no-ds-spread` | Pace the region's ds_loads by WMMA batch (`DsBatchWmmas` WMMAs). Each batch takes the region's even share `ceil(ds_loads / usable batches)`, where the last `ceil(LDS return time / WMMA window) + kDrainSlackWmmas` WMMAs of the region are kept free for the last load to return. A sparse region therefore spreads its loads, a dense one gets a share above what the LDS rate fits (the credits throttle it), and a batch that began behind the even line takes `DsCatchUpPace` more. Nothing is held once the region's last WMMA has issued; a held ds_load never holds a WMMA. false = a ds_load issues as soon as it is ready and a credit is free (the credit queue is the only pacing). |
| `DsBatchWmmas` | 4 | `--ds-batch-wmmas=N` | WMMAs per ds_load pacing batch (independent of the queue refill batch `WmmaBatchSize`). |
| `DsReadCyclesPerLoad` | 6 (`stinkytofu-opt`: 0 = off) | `--ds-read-cycles-per-load=N` | LDS return rate: cycles per ds_load. Capped at the region's even spread (`WMMAs x WMMA latency / ds_loads`). Decides how long the stage's ds_loads take to issue: `ceil(ds_loads x N / WMMA window)` WMMA windows from the region start. The barrier is placed after the last of them; the gap from the last ds_load to the barrier is the drain, set by the barrier cover. Replaces the old throttle latency T (one ds_load per `T / DsReadQueueDepth` cycles). Used only with `DsSpread`. |
| `WmmaBarrierCoverCycles` | -1 (automatic) | `--wmma-barrier-cover-cycles=N` | Queued WMMA cover a ready barrier waits for before it issues. -1 = a tail of 4 WMMAs (needs `DsSpread`); 0 = off; always capped at a full queue. Needs the queue model on. |
| `DsCatchUpPace` | 2 | `--ds-catch-up-pace=N` | ds_loads a batch may take above the even share when it began behind the even line. 0 = no catching up. |
| `DsReadQueueDepth` | 0 → HW 16 | `--ds-read-queue-depth=N` | In-flight ds_load credits modeled for the LDS return queue. A credit lives `HWModel::lds.readCreditLifetime` cycles. |
| `DsReadDrainLatency` | 0 → dynamic | `--ds-read-drain-latency=N` | Barrier timing only: cycles a barrier waits for its ds_loads to return. 0 derives it from the matching loads. |
| `DsReadOrder` | -1 → `Ascending` (1) | `--ds-read-order=Name` | ds_load priority: 0 `ProgramOrder` (AABB), 1 `Ascending` (A0 B0 A1 B1), 2 `AscendingCache` (A0 B0 B1 A1). |
| `LockDsReadOrder` | true | – | ds_loads on the same memory token issue strictly in `DsReadOrder` priority. false = a ready lower-priority load may go first. |

A held ds_load is released early only when no WMMA is ready to issue in its place.

**Barrier sizing:** the ds_loads in front of a barrier are given `WMMAs / ds_loads` windows each,
the same density the spread plans with, so the barrier thresholds and the pacer agree.

**Removed options and their replacements.** The old keys are still accepted by TensileLite
(`StinkyTofuParameters`, `StinkyTofuModuleOptions`) but ignored, with a warning, so a tuned yaml
loses its tuning unless it is translated:

| Old option | What it did | Now |
|---|---|---|
| `DsReadThrottleLatency` T | lifetime of one LDS queue credit: a saturated queue gives one ds_load per `T / DsReadQueueDepth` cycles | `DsReadCyclesPerLoad = T / DsReadQueueDepth` (16 on gfx1250): 96 -> 6, 112 -> 7 |
| `DsReadPerCap` A, `DsIssueCapSpanCycles` X, `DsIssueCapMode` | burst cap: at most A ds_loads per span X (default one WMMA window) | no equivalent: the per-batch share (`DsBatchWmmas`, `DsCatchUpPace`) replaces it |
| `DsReadPerWmma`, `DsReadThrottleTransitionFactor`, `DsReadThrottleTransitionEntries`, `DsSlotFirst` | per-WMMA ds count, loads past the queue depth, ds before fillers | removed |
| `BarrierHalfSlack` | WMMA windows kept inside a barrier signal/wait pair (0 = together) | removed: the separation slack is the constant 1 |
| `DsReadDrainLatency` | barrier timing only | unchanged |

The matching `stinkytofu-opt` flags and `--enable-wmma-hide-budget-prescan` are gone too.
TensileLite's `StinkyTofuDsIssueCapMode` and `StinkyTofuDsIssueCapSpanCycles` global parameters are
gone.

**Shared ds issue pipe:** when `HWModel::lds.wavesPerDsIssuePipe > 1`,
the issue cost of each ds_load is multiplied by `min(NumWaves, wavesPerDsIssuePipe)`.
On gfx1250 it is currently 1, so this is inert.

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
| `PrefetchLeadMinStageWmmas` | 64 | `--prefetch-lead-min-stage-wmmas=N` | Blocks whose stages are shorter than N WMMAs run with no prefetch lead of 8 or more. A single-stage lead (below 8) is kept at any stage length. |
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
| `CDNA5Config` | `globalReadPerWmma = 1`, `tensorLoadWmmaSpace = 0`, `wmmaLatencyFallbackCycles = 8` (WMMA latency where a region has none), `warGateWmmas = 0` (derive) |
| `HWModel::lds` | `readQueueDepth = 16`, `readCreditLifetime = 72`, `readDrainLatency = 0` (dynamic), `wavesPerDsIssuePipe = 1` |

gfx1250v0 uses the same scheduling defaults.

## Pattern recipes

Every pattern still respects data readiness, WAR on in-flight WMMA sources,
barriers and the ds order lock, so the output follows the pattern wherever
dependences allow.

| Pattern | Options |
|---|---|
| Default: ds_loads spread evenly over the WMMAs | none |
| WMMA queue kept covered, refilled in batches of 4 | `WmmaQueueDepth=8, WmmaQueueCoverCycles=32, WmmaBatchSize=4` |
| ds_loads as soon as they are ready and a credit is free | `DsSpread=false` |
| Fillers packed early instead of spread | `EvenSpreadFillers=false` |
| Free ds order | `LockDsReadOrder=false` |
