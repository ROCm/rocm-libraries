# Insert Cluster Barrier Pass

`createInsertClusterBarrierPass` builds a pass that inserts cluster-barrier
instructions at five well-defined rules covering the main and tail loops. This
document describes each rule, the emitted code shapes, and the pass parameters.

The pass is created via:

```cpp
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass(
    bool isKernelScope = true,
    int  pgrValue      = 1,
    int  plrValue      = 1);
```

## Overview

Rules are numbered in **kernel-execution order** -- the rule with the lowest
number is the first to fire when the kernel runs.

The cluster handshake uses signal/wait pairs at two scopes:

- **Workgroup scope**: `s_barrier_signal -1` / `s_barrier_wait -1`
- **Cluster scope**: `s_barrier_signal -3` / `s_barrier_wait -3`

`<HASH>` in the emitted labels is a fresh 16-character alphanumeric identifier
generated per insertion. Only the first wave (`WaveIdx == 0`) executes the
signal; the other waves fall through to the label.

**Idempotency:** each rule has its own skip check, so re-running the pass is a
no-op when the handshake is already present.

---

## Rule 1 -- Post-GSU==1 signal-only

Signal-only (no leading cluster wait), emitted immediately **after** each
`label_GSU_1:` label (Tensile's post-`GSU==1`-guard label), wrapped in an outer
`LoopCounterL != 0` gate so the cluster-barrier signal only fires on non-zero
iterations.

A workgroup-scope `s_barrier_signal -1` / `s_barrier_wait -1` pair sits **inside**
the outer LCL skip region (and **before** the inner `WaveIdx` gate) so every wave
in the workgroup has reached the post-`GSU==1` join before any wave issues the
cluster signal:

```asm
    s_cmp_eq_u32 s[sgprLoopCounterL], 0
    s_cbranch_scc1 label_skipCBPreSignal_LCL_<HASH_OUTER>
    s_barrier_signal -1                                          // workgroup signal
    s_barrier_wait -1                                            // workgroup wait
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<HASH_INNER>
    s_barrier_signal -3
  label_skipCBPreSignal_<HASH_INNER>:
  label_skipCBPreSignal_LCL_<HASH_OUTER>:
```

---

## Rule 2 -- First kernel load wait (kernel scope only)

A single `s_barrier_wait -3` immediately before the first `tensor_load_to_lds`
of the whole kernel.

---

## Rule 3 -- LDS-publication handshake (currently disabled)

> Rule 3 is currently disabled via the `kRule3Enabled` master switch in the
> `.cpp`. The description below is retained for when it is re-enabled.

A LoopCounterL-gated signal-only handshake at the LDS publication point that
precedes `label_openLoopL:`. It has the same shape as Rule 1 but with the outer
gate set to `s_cmp_le_u32 s[sgprLoopCounterL], pgrValue` (the `cbranch` skips the
signal when there are too few iterations left, so the producer is not needed).

The gate mirrors Tensile's own `s_cmp_le_u32 LCL, pgrValue` /
`s_cbranch_scc1 LoopEndL` loop-entry guard, so the cluster signal is suppressed
on the exact same control-flow paths where the corresponding `s_barrier_wait -3`
inside the unrolled loop body is skipped -- keeping `signal -3` / `wait -3`
paired everywhere.

### Two anchor modes (backward scan from `label_openLoopL:`)

**(a) Publication point already exists** (typical for PrefetchLocalRead > 0 schedules)

An `s_barrier_wait -1` is already present. Anchor at the successor of that wait;
no new workgroup sync is synthesized. The scan stops as soon as a
`tensor_load_to_lds` is reached: that instruction marks the prefetch section,
before any workgroup sync could sit, so an earlier workgroup wait would be
unrelated. Defers to Rule 4 if the same wait would also be a Rule-4 trigger.

**(b) No publication point** (typical for PrefetchLocalRead == 0 schedules)

No `s_barrier_wait -1` between the prefetch tail and `label_openLoopL:` (the
prologue has no local-read preamble barrier). Only active when `plrValue == 0`;
anchor at the label and synthesize an `s_barrier_signal -1` / `s_barrier_wait -1`
pair **inside** the LCL skip region (between the outer LCL skip-branch and the
inner `WaveIdx` gate) so the workgroup sync sits on the same control-flow path as
the cluster signal -- both are bypassed together on the `LCL <= pgrValue` skip
path (matching Tensile's loop-entry guard: when the unrolled loop body is
skipped, the LDS reads inside it are skipped too, so no LDS publication is
needed). Emitted shape immediately **before** the `label_openLoopL:` label:

```asm
    s_cmp_le_u32 s[sgprLoopCounterL], <pgrValue>          // outer LCL gate
    s_cbranch_scc1 label_skipCBPreSignal_LCL_<HASH_OUTER> // skip when LCL <= pgr
    s_barrier_signal -1                                  // workgroup signal
    s_barrier_wait -1                                    // workgroup sync
    s_cmp_eq_u32 s[sgprWaveIdx], 0                        // inner wave gate
    s_cbranch_scc0 label_skipCBPreSignal_<HASH_INNER>
    s_barrier_signal -3
  label_skipCBPreSignal_<HASH_INNER>:
  label_skipCBPreSignal_LCL_<HASH_OUTER>:
```

Internal control-flow labels inside the prefetch prologue (e.g.
`label_skipPGR2_*`) do not match `label_openLoopL` by exact-name comparison and
are walked through.

**Section-level idempotency:** the backward scan also flags whether a
cluster-scope signal/wait already sits in the section; if so (e.g. a prior pass
run already emitted Rule 3), Rule 3 self-disables. Unlike a `TEXTBLOCK` anchor,
the label/instruction-based scan survives `ScopeAdaptor::moveIRToBlock`, so Rule
3 keeps working whenever `Gfx1250Backend::buildGfx1250Pipeline` runs this pass at
kernel scope when `moduleOptions.ClusterBarrier == true`.

---

## Rule 4 -- Cluster handshake before loop loads

A cluster handshake around the workgroup barrier that precedes a
`tensor_load_to_lds`. For every load in a label-/branch-delimited segment, the
pass walks backward to the nearest preceding workgroup `s_barrier_signal -1` --
this signal is the **trigger**. Triggers are deduplicated by identity so multiple
loads sharing the same barrier yield exactly one handshake. If no workgroup
signal is found in the segment, Rule 4 does not apply to that load (there is no
barrier pair to anchor the cluster wait before).

The handshake is split into two independently-anchored halves:

- **Rule 4(a)** -- the SIGNAL half (`insertRule4ClusterBarrierSignal`).
- **Rule 4(b)** -- the WAIT half (`insertRule4ClusterBarrierWait`).

They are passed as separate `signalAnchor` / `waitAnchor` arguments to
`insertClusterBarrierHandshakeBefore`, resolved in this order:

1. The **WAIT anchor** (Rule 4(b)) is the trigger workgroup `s_barrier_signal
   -1` itself. The cluster `s_barrier_wait -3` is inserted immediately **before**
   it, so it precedes the whole workgroup barrier pair -- and hence the tensor
   load.
2. The **SIGNAL anchor** (Rule 4(a)) is derived by walking backward **from the
   WAIT anchor** by a configurable cycle lead (see below). Measuring the lead
   from the wait guarantees the signal lands at or before it.

Rule 3/5 still resolve their own anchors via the workgroup-WAIT finders; their
conflict guards bridge each wait to its paired signal
(`findPrecedingWorkgroupBarrierSignalInSegment`) so the identity comparison
against Rule 4's signal triggers still detects overlap.

Resulting shape around the trigger:

```asm
    s_barrier_wait -3        // Rule 4(b) cluster wait (before the workgroup pair)
    s_barrier_signal -1      // trigger workgroup signal (WAIT anchor)
    s_barrier_wait -1        // workgroup wait
    ...
    tensor_load_to_lds
```

### Signal cycle-lead placement (`kRule4SignalLeadCycles`)

By default the SIGNAL (Rule 4(a)) is planted `kRule4SignalLeadCycles` (default
**500**) *estimated cycles* earlier than its paired WAIT, so the cluster
producer starts as early as possible. `findRule4SignalAnchorByCycleLead` walks
backward **from the WAIT anchor**, consulting a per-instruction cycle map, until
the estimated cumulative cycle drops at least that far below the wait anchor's
cycle, and anchors the signal there. The walk is **bounded by the segment**
(`segBegin`, i.e. the first instruction after the previous label/branch/call): it
never crosses a control-flow boundary, so the SIGNAL stays on the same path as
its WAIT and the `signal -3` / `wait -3` pairing stays balanced. If the segment is
shorter than the requested lead, the signal clamps to the segment start.

The cycle map comes from `computeEstimatedCyclesPerInstruction` (a
non-annotating query form of `EstimateAsmCyclesPass`). It is only populated for
the modeled `label_LoopBeginL` unrolled-loop region on Gfx1250; when a trigger
has no estimate (other arch, outside the loop region, or `kRule4SignalLeadCycles
<= 0`) Rule 4(a) **falls back** to co-locating the signal with the wait.

The emission is selected by the `kRule4ForceUngatedSignalMode` master switch in
the `.cpp` (default **on** = mode (c)):

**(c) always-ungated mode** (active) -- for every trigger emit a `WaveIdx`-gated
`s_barrier_signal -3` (Rule 4(a)) ahead of a bare `s_barrier_wait -3` (Rule
4(b)); the cluster signal is **never** wrapped in an LCL skip branch.
The SCC restore is decided at scan time and **always rides with Rule 4(a)**,
right after the signal block, so the SCC clobbered by its `WaveIdx s_cmp_eq_u32`
is rebuilt locally before any downstream consumer (even when the signal leads the
wait by many cycles). The decision (`anySccWriterInRange` +
`findLiveRestorableSccCmpUpstream`):

- If **any** instruction in `[signalAnchor, waitAnchor)` already writes SCC, that
  write masks the clobber before a consumer sees it -- **no restore**.
- Otherwise scan backward **from the SIGNAL anchor** for a live, safely-clonable
  SCC producer whose SCC a downstream `s_cbranch_scc*` consumes; if found, a clone
  of it is re-emitted after the signal block to restore SCC. A producer qualifies
  only when it (1) writes SCC, (2) is a pure comparator -- **no allocatable (GPR)
  destination**. Note an SOPC compare (`s_cmp_*`) models its SCC result as an
  explicit SCC-typed destination, so the test is "no GPR dest", **not** "no dest";
  arithmetic/logic SCC writers like `s_sub_u32` additionally write a real GPR and
  are excluded, and
  (3) has **no source register overwritten** between it and the signal anchor
  (checked via physical-register overlap), so the clone recomputes the identical
  SCC. The loop-exit `s_cmp_eq LCL, imm` remains the dominant match.

Rule 4(b) always emits a bare, SCC-neutral wait.

When the switch is **off**, two fallback modes are selected by
`findLiveRestorableSccCmpUpstream`:

**(a) inherited-SCC mode** -- when SIA (typically `ScheduleIterAlg=4`) hoists the
loop-exit `s_cmp_eq_{u32,i32} LCL, imm` above the anchor, a downstream `cbranch`
still consumes its SCC. Emit an ungated leading `s_barrier_wait -3` **first**,
then the inherited-SCC signal block (the signal is single-iter skipped via the
inherited SCC; a clone of the upstream cmp is re-emitted between the inner and
outer skip labels to rebuild SCC for that downstream cbranch).

**(b) drain-gated mode** -- when no such upstream cmp is live, gate the handshake
with ASYMMETRIC LCL thresholds: skip the WAIT (Rule 4(b)) at `LCL <= pgrValue`
and the SIGNAL (Rule 4(a)) one stage earlier at `LCL <= pgrValue+1` (both lowered
by any hoisted LCL pre-decrement). Mode (b) naturally maps its WAIT gate to
`waitAnchor` and its SIGNAL gate to `signalAnchor`. The drain iterations -- where
the paired `tensor_load_to_lds` is disabled -- drop the handshake while keeping
`signal -3` / `wait -3` balanced.

Shape (mode (c), **co-located** anchors, i.e. the fallback when no cycle lead is
applied; the `<clone of upstream LCL cmp>` line is emitted only when a live
upstream cmp exists):

```asm
    s_cmp_eq_u32 s[sgprWaveIdx], 0                               // inner wave gate
    s_cbranch_scc0 label_skipCBPreSignal_<HASH_INNER>
    s_barrier_signal -3
  label_skipCBPreSignal_<HASH_INNER>:
    s_barrier_wait -3                                            // bare cluster wait
    <clone of upstream LCL cmp>                                  // restore SCC (if any)
```

Shape (mode (c), **separated** anchors, the default when `kRule4SignalLeadCycles`
places the signal ~N cycles ahead of the wait):

```asm
    s_cmp_eq_u32 s[sgprWaveIdx], 0                               // inner wave gate  (Rule 4(a))
    s_cbranch_scc0 label_skipCBPreSignal_<HASH_INNER>
    s_barrier_signal -3
  label_skipCBPreSignal_<HASH_INNER>:
    <clone of upstream LCL cmp>                                  // restore SCC (if any)
    ...                                                          // ~N cycles of intervening code
    s_barrier_wait -3                                            // bare cluster wait (Rule 4(b))
```

---

## Rule 5 -- Tail-loop cluster handshake (kernel scope only, paired)

Anchors on the first `tensor_load_to_lds` that follows the `/* Tail Loop */`
`TEXTBLOCK` marker. The wait and signal are emitted at two distinct sites because
the load and the preceding workgroup wait sit in different label/branch-delimited
segments (the tail TDM-reset block between them is not synchronization-critical,
so collapsing both into a single site would unnecessarily serialize the cluster).

- **5a** -- signal-only handshake (no LoopCounterL gate) immediately **after** the
  workgroup-scope wait (`s_barrier_wait -1`) that precedes the tail load (searched
  backward from the load, bounded by the `/* Tail Loop */` marker). Defers to Rule
  4 if that wait is already a Rule-4 trigger.
- **5b** -- a single `s_barrier_wait -3` immediately **before** the tail
  `tensor_load` itself.

Each half has its own idempotency check so re-runs are no-ops.

---

## Parameters

### `isKernelScope` (default `true`)

Must be `true` for the GFX1250 backend pipeline (whole-kernel insertion). When
`false`, the pass is intended for region-scoped invocation via
`createKernelToRegionsPassAdaptor` (not used by the backend today). Rule 2 only
fires when this is `true` because the "first `tensor_load` of the whole kernel"
anchor is meaningful only at kernel scope.

### `pgrValue` (default `1`, i.e. PrefetchGlobalRead=1)

Tensile's `PrefetchGlobalRead` setting. It is consulted only by Rule 4's
drain-gated fallback mode (b) (the `LCL <= pgrValue` / `LCL <= pgrValue+1`
thresholds), i.e. when `kRule4ForceUngatedSignalMode` is off. The active mode (c)
does not use it, and Rule 3 (whose `LoopCounterL <= pgrValue` gate would also use
it) is disabled via `kRule3Enabled`. It is retained in the signature both for
mode (b)'s drain gate and so the Rule 3 gate can be reinstated without an API
change.

### `plrValue` (default `1`)

Tensile's `PrefetchLocalRead` setting. It only takes effect while Rule 3 is
enabled (see `kRule3Enabled`), where it selects Rule 3's anchor mode (b): when
`plrValue == 0` and the backward scan from `label_openLoopL:` finds no
`s_barrier_wait -1` before reaching the prefetch boundary (`tensor_load_to_lds`),
the rule synthesizes the missing publication point (workgroup
`s_barrier_signal -1` / `s_barrier_wait -1`) followed by the same
`LCL <= pgrValue` gated cluster signal as anchor mode (a). Any non-zero value
disables mode (b) (default).

---

## Analysis invalidation

This pass mutates the CFG (new branches and a new label), so dependent CFG /
dominance analyses are invalidated.

---

## See Also

- [Architecture Overview](architecture.md) -- system architecture and pass pipeline
- `src/transforms/asm/InsertClusterBarrierPass.cpp` -- implementation
- `include/stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp` -- public API
