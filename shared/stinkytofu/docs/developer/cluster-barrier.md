# Insert Cluster Barrier Pass

`createInsertClusterBarrierPass` inserts cluster-barrier instructions at six
well-defined rules covering the prologue, the main summation loop, and the tail
loop. This document reflects the current implementation in
`src/transforms/asm/InsertClusterBarrierPass.cpp` and the gfx1250 pipeline in
`src/pipeline/backend/Gfx1250Backend.cpp`.

```cpp
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass(
    int pgrValue = 1,
    int plrValue = 1);
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
  3 and 6 split 6a/6b across BBs when needed).

## Overview

Rules fire in **kernel execution order**:

| Rule | Where | What it emits |
|------|-------|---------------|
| 1 | after `label_GSU_1:` | priming cluster signal |
| 2 | before the kernel's first `tensor_load_to_lds` | bare cluster wait (whole-kernel IR only) |
| 3 | LDS publication point before `label_openLoopL:` | priming cluster signal |
| 4 | after each loop load's workgroup wait | per-iteration cluster handshake |
| 5 | loop-exit convergence label | trailing cluster wait (ungated scheme only) |
| 6 | tail loop | tail cluster handshake (6a signal + 6b wait) |

Barriers use two scopes:

- **Workgroup**: `s_barrier_signal -1` / `s_barrier_wait -1`
- **Cluster**: `s_barrier_signal -3` / `s_barrier_wait -3`

### Offset ping-pong

Each per-iteration cluster `wait -3` consumes the **previous** iteration's
`signal -3`. Rules 1 and 3 emit **priming** signals consumed by the first loop
wait; Rule 5 (or, in the gated scheme, an extra in-loop wait) drains the loop's
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
1; Rule 4 pairs `Wait_LCL_1` (drains gen 1) with signal gen 2; Rule 6a opens
gen 3. Bare waits (Rule 2 / Rule 6b) carry no skip label.

### Idempotency

Each rule self-disables when its handshake is already present (per-anchor checks
for Rules 1/3/4/6; function-wide first-load check for Rule 2). Re-running the
pass is a no-op on already-instrumented IR.

## Compile-time switches

Two `constexpr bool` switches in the `.cpp` file. The default is the **gated**
scheme (`kClusterBarrierDrainGateEnabled = true`).

### `kClusterBarrierDrainGateEnabled` (default `true`)

- **`true` (default):** asymmetric Rule 4 drain gates (`wait` at `LCL <= pgr`,
  `signal` at `LCL <= pgr+1`), Rule 1 gated on `LCL != 0`, Rule 3 gated on
  `LCL <= pgr`. Rule 5 is **disabled** (the asymmetric gate leaves one extra
  in-loop wait).
- **`false`:** ungated priming signals; Rule 4 bare wait + WaveIdx-gated signal;
  Rule 5 plants a loop-exit trailing wait.

### `kRule4SignalBeforeWaitEnabled` (default `false`)

Only affects the **ungated** scheme (ignored while the drain gate is on).
Selects signal-before-wait vs wait-before-signal ordering for Rule 4; when
true, Rules 3 and 5 are also disabled.

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

---

## Rule 3 -- LDS-publication priming signal

Priming cluster signal at the LDS publication point before `label_openLoopL:`.
Disabled in ungated signal-before-wait legacy mode (Rules 3 and 5 both off).

### Anchor scan (`scanRule3PublicationPoint`)

Walk **backward** from `label_openLoopL:` through its BB and **CFG
predecessors**, stopping at the first `tensor_load_to_lds` (prefetch boundary).

- **Mode (a):** nearest preceding `s_barrier_wait -1` found -> anchor after that
  wait (may live in a **predecessor BB** after CFGBuilder).
- **Mode (b):** no wait before the boundary -> anchor at `label_openLoopL:` and
  synthesize workgroup `signal -1` / `wait -1` immediately before the cluster
  signal.

Section idempotency: if the scan already sees cluster `-3` signal/wait in the
publication section, Rule 3 disables. Anchor-level idempotency: skip if the
existing wait is already followed by a cluster handshake, or if Rule 4 has
queued the same wait as a trigger.

Emission uses `AsmIRBuilder` rooted in the anchor's owning BB.

**Note:** `plrValue` is passed through the API for Tensile parity but is **not
consulted** by the current Rule 3 scan; mode (b) is driven purely by IR shape.

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

### Gated emission (default)

Asymmetric drain gates on wait and signal; optional SCC restore clone after the
block when `findLiveSccCmpUpstream` finds a live loop-exit compare above the
anchor.

---

## Rule 5 -- Loop-exit trailing wait

Bare `s_barrier_wait -3` at the loop-exit convergence label. **Disabled** in the
default gated scheme. Enabled only in ungated wait-before-signal mode.

---

## Rule 6 -- Tail-loop cluster handshake

Anchored on the `/* Tail Loop */` TEXTBLOCK. Region scope never sees the marker
(TEXTBLOCK erased), so Rule 6 self-disables there.

### Function-wide scan (`scanRule6TailPoint`)

1. Find the tail TEXTBLOCK marker.
2. **Forward BFS** from the marker to the first `tensor_load_to_lds` (tail load).
3. **Backward BFS** from the tail load toward the marker for the nearest
   workgroup `s_barrier_wait -1`.

### 6a -- tail priming signal

WaveIdx-gated signal-only block (no LCL gate):

- **Real wait found:** signal immediately after the wait (possibly in another BB).
- **Fallback:** synthesize workgroup sync after the marker, then signal.

Defers to Rule 4 if the same wait is already a Rule 4 trigger.

### 6b -- tail load wait

Bare `s_barrier_wait -3` immediately before the tail load (possibly in another
BB). Skipped when the load is already preceded by a cluster wait.

---

## Parameters

| Parameter | Default | Used by |
|-----------|---------|---------|
| `pgrValue` | `1` | Gated Rule 3/4 drain thresholds (`PrefetchGlobalRead`) |
| `plrValue` | `1` | Reserved / API parity (`PrefetchLocalRead`); not read by current Rule 3 logic |

---

## Unit tests

`tests/unit/asm/InsertClusterBarrierPassTest.cpp` exercises the default gated
scheme (`kClusterBarrierDrainGateEnabled = true`). Tests construct multi-BB CFGs
directly (no CFGBuilder pass in the fixture) to mirror post-CFGBuilder IR.

| Test | Rule / behavior |
|------|-----------------|
| `Rule1_EmitsGatedPrimingSignal` | Rule 1 gated shape and emission order |
| `Rule3_CrossBB_FindsWaitInPredecessor` | Rule 3 mode (a) across BB boundary |
| `Rule3_CrossBB_SynthesizesWgSyncAtLabel` | Rule 3 mode (b) synthesis |
| `Rule3_SkipsWhenSectionAlreadyHasClusterBarrier` | Rule 3 section idempotency |
| `Rule2_InsertsWaitBeforeFirstLoad` | Rule 2 placement |
| `Rule2_SkipsExistingWait` | Rule 2 idempotency |
| `Rule4_EmitsHandshake` | Rule 4 gated handshake + Rule 2 interaction |
| `Rule4_WaitMovesBeforeWorkgroupSignal` | Rule 4 wait-move fix |
| `Rule4_DistinctWaits` | Multiple anchors |
| `Rule4_SharedWait` | Trigger dedup |
| `Rule4_BranchBreaksAnchor` | No cross-BB Rule 4 anchor |
| `Rule4_RestoresLclCmp` | SCC restore clone |
| `Rule4_StopsAtTailMarker` | Rule 4 / Rule 6 partition |
| `Rule4_PerBasicBlock` | Per-BB Rule 4 |
| `Rule5_DisabledInGatedDefault` | Rule 5 off in default scheme |
| `Rule6_EmitsSignalAndWait` | Rule 6a + 6b in one BB |
| `Rule6a_FallbackSynthesizesSync` | Rule 6a fallback |
| `Rule6_CrossBB_FindsWaitAndLoad` | Rule 6 across BBs |
| `Rule6b_SkipsExistingWait` | Rule 6b idempotency |
| `CombinedLoadDrivenRules_NoLoadIsNoOp` | Negative: no load -> no-op |

Run:

```bash
./unit_tests --gtest_filter=InsertClusterBarrierPassTest.*
```

### Integration verification (half.yaml)

With `ClusterBarrier: true` on gfx1250, regenerating `half-gen.yaml` solutions
after the CFGBuilder-first + cross-BB Rules 3/6 fixes produces cluster barrier
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
