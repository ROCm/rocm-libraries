# Insert Cluster Barrier Pass

`createInsertClusterBarrierPass` inserts cluster-barrier instructions at six
well-defined rules covering the prologue, the main summation loop, and the tail
loop. This document reflects the current implementation in
`src/transforms/asm/InsertClusterBarrierPass.cpp` and the gfx1250 pipeline in
`src/pipeline/backend/Gfx1250Backend.cpp`.

```cpp
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass(
    int pgrValue = 1);
```

## Pipeline placement

On gfx1250 the pass runs **once at kernel scope**, **after the first
kernel-wide `CFGBuilderPass`** and **before `RegionClonePass` /
`InsertVgprMsbPass`**:

```
region scheduling + splice-back
  -> CFGBuilder            (flat IR -> real CFG)
  -> InsertClusterBarrierPass
  -> RegionClone
  -> InsertVgprMsb
  -> CFGBuilder            (re-materialize new branches/labels)
  -> ...
```

Implications:

- Anchor lookup uses **real basic blocks and CFG edges**, not inline
  label/branch segment simulation.
- Rules 3 and 6 walk **CFG predecessors/successors** when publication points
  span multiple BBs.
- Rule 4 scans each BB only up to the `/* Tail Loop */` marker and finds anchor
  waits with an **in-BB backward scan** (`findPrecedingWorkgroupBarrierWaitInBB`).
- Emission uses an `AsmIRBuilder` rooted in the BB that owns each anchor (Rules
  3 and 6 split 5a/5b across BBs when needed).

## Overview

Rules fire in **kernel execution order**:

| Rule | Where | What it emits |
|------|-------|---------------|
| 1 | after `label_GSU_1:` | priming cluster signal |
| 2 | before the kernel's first `tensor_load_to_lds` | bare cluster wait (whole-kernel IR only; **suppressed when `pgrValue == 0`**) |
| 3 | LDS publication point before `label_openLoopL:` | priming cluster signal (**suppressed when `pgrValue == 0`**) |
| 4 | after each loop load's workgroup wait | per-iteration cluster handshake |
| 5 | tail loop | tail cluster handshake (5a signal + 5b wait) |
| 6 | loop-exit convergence label | trailing cluster wait (LCL gate off, wait-before-signal only) |

Barriers use two scopes:

- **Workgroup**: `s_barrier_signal -1` / `s_barrier_wait -1`
- **Cluster**: `s_barrier_signal -3` / `s_barrier_wait -3`

### Offset ping-pong

Each per-iteration cluster `wait -3` consumes the **previous** iteration's
`signal -3`. Rules 1 and 3 emit **priming** signals consumed by the first loop
wait; Rule 6 (or, in the gated scheme, an extra in-loop wait) drains the loop's
last signal.

Only wave 0 signals (`s_cmp_eq_u32 s[sgprWaveIdx], 0`); every wave executes
cluster waits.

### Skip-label generation numbering

`<N>` in emitted labels is the cluster-barrier **generation** number, assigned
as the pass walks the kernel top-to-bottom (Rule 1 -> 3 -> 4 -> 5 -> 6):

| Prefix | Role | Numbering |
|--------|------|-----------|
| `label_skipCBPreSignal_<N>` | inner WaveIdx gate | signal generation `N` |
| `label_skipCBSignal_LCL_<N>` | outer LCL gate on a **signal** | same `N` as paired signal |
| `label_skipCBWait_LCL_<N>` | outer LCL gate on a **wait** | generation the wait **drains** (previous signal) |

Example (gated default, typical kernel): Rule 1 opens gen 0; Rule 3 opens gen
1; Rule 4 pairs `Wait_LCL_1` (drains gen 1) with signal gen 2; Rule 5a opens
gen 3. Bare waits (Rule 2 / Rule 5b) carry no skip label.

### Idempotency

Each rule self-disables when its handshake is already present (per-anchor checks
for Rules 1/3/4/5; function-wide first-load check for Rule 2). Re-running the
pass is a no-op on already-instrumented IR.

## Compile-time switches

Two `constexpr bool` switches in the `.cpp` file. The default is the **gated**
scheme (`kClusterBarrierDrainGateEnabled = true`).

### `kClusterBarrierDrainGateEnabled` (default `true`)

- **`true` (default):** asymmetric Rule 4 drain gates (`wait` at `LCL <= pgr`,
  `signal` at `LCL <= pgr+1`), Rule 1 gated on `LCL != 0`, Rule 3 gated on
  `LCL <= pgr`. Rule 6 is **disabled** (the asymmetric gate leaves one extra
  in-loop wait). When **`pgrValue == 0`**, Rules 2 and 3 are suppressed and
  Rule 4 emits a **bare wait** plus an **LCL-gated signal only** (see
  [PGR=0 exception](#pgr0-exception) below).
- **`false`:** ungated priming signals. With wait-before-signal ordering (default
  when the second switch is off), Rule 4 emits bare wait + WaveIdx-gated signal
  and Rule 6 plants a loop-exit trailing wait. With signal-before-wait ordering
  (second switch on), Rules 3 and 6 are also disabled.

### `kRule4SignalBeforeWaitEnabled` (default `false`)

Only affects the **ungated** scheme (ignored while the drain gate is on).
Selects signal-before-wait vs wait-before-signal ordering for Rule 4; when
true, Rules 3 and 6 are also disabled.

---

## Rule 1 -- Post-`GSU==1` priming signal

After each `label_GSU_1:` (survives region extraction).

**Gated (default):** outer `LCL != 0` gate, workgroup sync inside the skip region,
inner WaveIdx gate, then `s_barrier_signal -3`. Signal-only (no cluster wait).

**Ungated:** plain WaveIdx-gated signal only.

---

## Rule 2 -- First kernel load wait (whole-kernel IR only)

One bare `s_barrier_wait -3` immediately before the function's first
`tensor_load_to_lds`. Requires whole-kernel IR (Gfx1250: before RegionClonePass; do not run inside ScopeAdaptor). Skipped when the
load is already preceded by a cluster wait.

**PGR=0 (`pgrValue == 0`):** suppressed. With no prefetch iterations there is
no orphaned prologue signal for this leading wait to consume; Rule 1's priming
signal is paired by the first Rule 4 bare wait inside the main loop instead.

---

## Rule 3 -- LDS-publication priming signal

Priming cluster signal at the LDS publication point before `label_openLoopL:`.
Disabled in ungated signal-before-wait legacy mode (Rules 3 and 6 both off).
Also disabled when **`pgrValue == 0`** (no prefetch publication point to prime).

### Anchor scan (`scanRule3PublicationPoint`)

Walk **backward** from `label_openLoopL:` through its BB and **CFG
predecessors**, stopping at the first `tensor_load_to_lds` (prefetch boundary).

- **Existing publication wait:** nearest preceding `s_barrier_wait -1` found
  -> anchor after that wait (may live in a **predecessor BB** after CFGBuilder).
- **Synthesized publication sync:** no wait before the boundary -> anchor at
  `label_openLoopL:` and synthesize workgroup `signal -1` / `wait -1` immediately
  before the cluster signal.

Section idempotency: if the scan already sees cluster `-3` signal/wait in the
publication section, Rule 3 disables. Anchor-level idempotency: skip if the
existing wait is already followed by a cluster handshake, or if Rule 4 has
queued the same wait as a trigger.

Emission uses `AsmIRBuilder` rooted in the anchor's owning BB. The
synthesized-publication path is selected purely from IR shape (no preceding
workgroup wait before the prefetch boundary), not from Tensile
`PrefetchLocalRead`.

---

## Rule 4 -- Per-iteration cluster handshake

For each `tensor_load_to_lds` **before** the `/* Tail Loop */` marker in each
BB:

1. Forward-scan loads in `[bb.begin(), marker)`.
2. Backward-scan **within the same BB only**
   (`findPrecedingWorkgroupBarrierWaitInBB`) for the anchor `s_barrier_wait -1`.
3. Dedup triggers by wait identity (shared wait -> one handshake).

A workgroup wait in a **different BB** from its load is **not** a Rule 4 anchor
(CFG boundary). Rule 2 may still gate the function's first load.

### Wait placement (random-hang fix)

In wait-before-signal schemes the cluster `wait -3` is planted immediately
**before** the paired workgroup `signal -1`; the cluster `signal -3` stays
**after** the workgroup `wait -1`, with the workgroup pair **between** cluster
wait and cluster signal.

### Handshake layouts (selected by compile-time switches)

| Layout | Condition | Rule 4 shape |
|--------|-----------|--------------|
| LCL-gated wait-before-signal | `kClusterBarrierDrainGateEnabled == true` (default), `pgrValue >= 1` | Asymmetric LCL gates on wait (`LCL <= pgr`) and signal (`LCL <= pgr+1`); when the schedule hoists `s_sub LCL` above the anchor, both immediates subtract `lclPreDecrement` (e.g. pgr 1 with one hoisted decrement -> wait threshold 0); workgroup pair between cluster wait and signal |
| PGR=0 wait-before-signal | drain gate on, `pgrValue == 0` | Bare cluster wait (no LCL gate) before the workgroup signal; LCL-gated signal only at `LCL <= pgr+1 - lclPreDecrement` (typically `LCL <= 1` before the hoisted sub, `LCL <= 0` after) so the drain iteration suppresses the signal but still executes the wait |
| Ungated wait-before-signal | drain gate off, `kRule4SignalBeforeWaitEnabled == false` | Bare cluster wait then WaveIdx-gated signal (same split placement); Rule 6 supplies loop-exit drain |
| Legacy signal-before-wait | drain gate off, `kRule4SignalBeforeWaitEnabled == true` | Signal, optional SCC restore, then bare wait (all at one anchor); Rules 3 and 6 off |

In the default gated layout, optional SCC restore clone lands after the block
when `findLiveSccCmpUpstream` finds a live loop-exit compare above the anchor.

---

## Rule 5 -- Tail-loop cluster handshake

Anchored on the `/* Tail Loop */` TEXTBLOCK. Region scope never sees the marker
(TEXTBLOCK erased), so Rule 5 self-disables there.

### Function-wide scan (`scanRule5TailPoint`)

1. Find the tail TEXTBLOCK marker.
2. **Forward BFS** from the marker to the first `tensor_load_to_lds` (tail load).
3. **Backward BFS** from the tail load toward the marker for the nearest
   workgroup `s_barrier_wait -1`.

### 5a -- tail priming signal

WaveIdx-gated signal-only block (no LCL gate):

- **Real wait found:** signal immediately after the wait (possibly in another BB).
- **Fallback:** synthesize workgroup sync after the marker, then signal.

Defers to Rule 4 if the same wait is already a Rule 4 trigger.

### 5b -- tail load wait

Bare `s_barrier_wait -3` immediately before the tail load (possibly in another
BB). Skipped when the load is already preceded by a cluster wait.

---

## Rule 6 -- Loop-exit trailing wait

Bare `s_barrier_wait -3` at the loop-exit convergence label. **Disabled** when
the LCL drain gate is on (Rule 4's asymmetric gate already drains the last
signal). Enabled when the LCL gate is off and Rule 4 uses wait-before-signal
ordering (`kRule4SignalBeforeWaitEnabled == false`).

---

## PGR=0 exception

When `pgrValue == 0` (`PrefetchGlobalRead = 0`), the default asymmetric drain
gate collapses badly: the wait-side threshold `LCL <= pgr` becomes `LCL <= 0`
(which never skips after a hoisted `s_sub LCL`), while the signal-side threshold
`LCL <= pgr+1` still skips on the drain iteration. That leaves an extra bare wait
with no matching signal and can hang.

The pass therefore treats PGR=0 as a special case while keeping the gated scheme
enabled:

| Rule | PGR=0 behavior |
|------|----------------|
| 1 | unchanged (`LCL != 0` gate) |
| 2 | **suppressed** |
| 3 | **suppressed** |
| 4 | **bare wait** + **LCL-gated signal only** (wait-side drain gate omitted) |
| 5 | unchanged |
| 6 | unchanged (still disabled under the gated scheme) |

Rule 1's priming signal is consumed by the first Rule 4 bare wait. Each main-loop
iteration then pairs one bare wait with one gated signal except on the drain
iteration, where the signal is skipped but the wait still runs -- keeping the
offset ping-pong balanced for all `LoopCounterL` values.

---

## Parameters

| Parameter | Default | Used by |
|-----------|---------|---------|
| `pgrValue` | `1` | Gated Rule 3/4 drain thresholds (`PrefetchGlobalRead`); triggers the [PGR=0 exception](#pgr0-exception) when zero |

---

## Unit tests

`tests/unit/asm/InsertClusterBarrierPassTest.cpp` exercises the default gated
scheme (`kClusterBarrierDrainGateEnabled = true`). Tests construct multi-BB CFGs
directly (no CFGBuilder pass in the fixture) to mirror post-CFGBuilder IR.

| Test | Rule / behavior |
|------|-----------------|
| `Rule1AndRule2_PrologueHandshake` | Rules 1 + 2 on a prologue-shaped kernel |
| `Pgr0_SkipsRule2AndRule3` | PGR=0 suppresses Rules 2 and 3; Rule 1 only |
| `Pgr0_Rule4_BareWaitGatedSignal` | PGR=0 Rule 4 bare wait + LCL-gated signal |
| `Rule3_CrossBB_FindsWaitInPredecessor` | Rule 3 existing-publication-wait path across BB boundary |
| `Rule3_CrossBB_SynthesizesWgSyncAtLabel` | Rule 3 synthesized-publication path |
| `Rule4_WaitMovesBeforeWorkgroupSignal` | Rule 4 wait-move fix |
| `Rule4_DistinctWaits` | Multiple anchors |
| `Rule4_SharedWait` | Trigger dedup |
| `Rule4_RestoresLclCmp` | SCC restore clone |
| `Rule4_StopsAtTailMarker` | Rule 4 / Rule 5 partition |
| `Rule4_PerBasicBlock` | Per-BB Rule 4 |
| `Rule5_EmitsSignalAndWait` | Rule 5a + 5b in one BB |
| `Rule5a_FallbackSynthesizesSync` | Rule 5a fallback |
| `Rule5_CrossBB_FindsWaitAndLoad` | Rule 5 across BBs |
| `CombinedLoadDrivenRules_NoLoadIsNoOp` | Negative: no load -> no-op |

Run:

```bash
./unit_tests --gtest_filter=InsertClusterBarrierPassTest.*
```

### Integration verification (half.yaml)

With `ClusterBarrier: true` on gfx1250, regenerating `half-gen.yaml` solutions
after the CFGBuilder-first + cross-BB Rules 3/5 fixes produces cluster barrier
counts matching the pre-move baseline (`signal=144`, `wait=112` across 40
kernels). The reference sample kernel matches `half-out-before` byte-for-byte
for cluster-barrier insertion.

---

## Analysis invalidation

The pass mutates the CFG (new branches and labels). `run()` returns
`PreservedAnalyses::none()`.

---

## See also

- `src/transforms/asm/InsertClusterBarrierPass.cpp` -- implementation
- `include/stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp` -- public API
- `src/pipeline/backend/Gfx1250Backend.cpp` -- pipeline ordering
- [Architecture Overview](architecture.md)
