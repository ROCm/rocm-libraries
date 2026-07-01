# Insert Cluster Barrier Pass

`createInsertClusterBarrierPass` builds a pass that inserts cluster-barrier
instructions at six well-defined rules covering the prologue, the main
summation loop, and the tail loop. This document describes each rule, the
emitted code shapes, the two compile-time switches that select the handshake
scheme and ordering, and the pass parameters. It reflects the current
implementation in `src/transforms/asm/InsertClusterBarrierPass.cpp`.

The pass is created via:

```cpp
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass(
    bool isKernelScope = true,
    int  pgrValue      = 1,
    int  plrValue      = 1);
```

## Overview

Rules are numbered in **kernel-execution order** -- the rule with the lowest
number is the first to fire when the kernel runs:

| Rule | Where | What it emits |
|------|-------|---------------|
| 1 | after `label_GSU_1:` | priming cluster signal |
| 2 | before the kernel's first `tensor_load_to_lds` | bare cluster wait (kernel scope only) |
| 3 | LDS publication point before `label_openLoopL:` | priming cluster signal (disabled in legacy signal-before-wait mode) |
| 4 | after each loop load's workgroup wait | per-iteration cluster handshake |
| 5 | loop-exit convergence label | trailing cluster wait (ungated scheme only; disabled in the default gated scheme) |
| 6 | tail loop | tail cluster handshake (6a signal + 6b wait) |

The cluster handshake uses signal/wait pairs at two scopes:

- **Workgroup scope**: `s_barrier_signal -1` / `s_barrier_wait -1`
- **Cluster scope**: `s_barrier_signal -3` / `s_barrier_wait -3`

`<N>` in the emitted labels is the cluster-barrier *generation* number, assigned
directly at insertion time as the pass walks each kernel top-to-bottom
(Rule 1 -> 3 -> 4 -> 5 -> 6). It counts from 0 and reflects which generation of
the offset ping-pong the guarded signal/wait belongs to. There are three skip
label prefixes:

- `label_skipCBPreSignal_<N>` -- inner WaveIdx gate (only wave 0 signals);
  numbered with the guarded signal's generation.
- `label_skipCBSignal_LCL_<N>` -- outer `LoopCounterL` gate guarding a cluster
  **signal**; numbered with that signal's generation.
- `label_skipCBWait_LCL_<N>` -- `LoopCounterL` gate guarding a cluster **wait**;
  numbered with the generation the wait drains (the previous signal's).

So each signal opens a new generation `N` (0, 1, 2, ...) and the wait that
consumes it carries that same `N`: e.g. Rule 1's priming signal is gen 0,
Rule 3's is gen 1, and Rule 4's per-iteration handshake pairs a WAIT gate `1`
(draining gen 1) with a SIGNAL gate `2` (opening gen 2). Bare waits (Rule 2 /
Rule 6b) carry no skip label. The counter is function-scoped, so every kernel
numbers from 0.

The cluster signal is guarded by an inner `s_cmp_eq_u32 s[sgprWaveIdx], 0` check
so that only the first wave (`WaveIdx == 0`) of each workgroup signals; the other
waves fall through to the skip label. Every wave executes the cluster `wait -3`.

The per-iteration cluster handshake is an *offset ping-pong*: each per-iteration
`wait -3` consumes the **previous** iteration's `signal -3`. Rules 1 and 3 emit
the *priming* signal that the loop's first wait consumes; Rule 5 (or, in the
gated scheme, an extra in-loop wait) drains the loop's last signal.

**Idempotency:** each rule has its own skip check, so re-running the pass is a
no-op when the handshake is already present.

## Compile-time switches

Two `constexpr bool` switches in the `.cpp` control emission.
`kClusterBarrierDrainGateEnabled` defaults to `true` (the gated scheme);
`kRule4SignalBeforeWaitEnabled` defaults to `false` and is ignored while the
drain gate is on.

### `kClusterBarrierDrainGateEnabled` (default `true`)

The master switch selecting between two mutually exclusive, internally balanced
handshake schemes across Rules 1, 3, 4, and 5 (Rule 6 is unaffected). The
schemes must move together to keep `signal -3` / `wait -3` balanced.

- **`true` -- gated (default).** Every priming / per-iteration signal is wrapped in a
  `LoopCounterL` drain gate so it is suppressed on exactly the drain iterations
  whose paired counterpart is also skipped:
  - Rule 4 uses **asymmetric** gates -- the WAIT is skipped at `LCL <= pgrValue`
    and the SIGNAL one stage earlier at `LCL <= pgrValue + 1` (both lowered by
    any hoisted `LoopCounterL` pre-decrement). This leaves the loop with exactly
    one extra in-loop WAIT that consumes the last SIGNAL, so **Rule 5 is
    disabled** (a trailing wait would be unpaired).
  - Rule 1's priming signal is gated on `LCL != 0` (with a leading workgroup
    sync).
  - Rule 3's publication-point signal is gated on `LCL <= pgrValue`.
- **`false` -- ungated.** Priming signals (Rules 1/3) are plain `WaveIdx`-gated
  signal-only blocks with no `LoopCounterL` gate. Rule 4 emits a bare `wait -3`
  followed by the `WaveIdx`-gated `signal -3` (ordering subject to
  `kRule4SignalBeforeWaitEnabled`, below). Because the loop emits an equal number
  of waits and signals, the loop's last signal has no in-loop wait to consume it,
  so **Rule 5** plants one trailing `wait -3` at the loop-exit convergence label.

### `kRule4SignalBeforeWaitEnabled` (default `false`, ignored while the drain gate is on)

Selects Rule 4's handshake **ordering**, and applies only to the **ungated**
scheme (it is ignored when `kClusterBarrierDrainGateEnabled == true`, which is
the default and owns its own ordering).

- **`false` -- wait-before-signal.** Rule 4 emits the bare
  `s_barrier_wait -3` FIRST and then the `WaveIdx`-gated `s_barrier_signal -3`.
  This is the offset ping-pong: each wait consumes the *previous* iteration's
  signal, and the loop's last signal is drained by Rule 5's loop-exit wait. The
  cluster `s_barrier_wait -3` is planted immediately before the anchor's paired
  workgroup `s_barrier_signal -1` (so the workgroup barrier fences the cluster
  wait from the next-generation cluster signal -- random-hang fix, see Rule 4),
  while the signal block and any live-SCC restore clone stay AFTER the workgroup
  wait.
- **`true` -- signal-before-wait (legacy).** Reverts to the pre-rewrite layout:
  the `WaveIdx`-gated `s_barrier_signal -3` FIRST, then the bare
  `s_barrier_wait -3` LAST, so the wait stays the load's immediate predecessor
  and Rule 2's `isImmediatelyPrecededByClusterBarrierWait` idempotency guard
  keeps working. Any live-SCC restore clone is placed immediately BEFORE the
  trailing wait (not after the whole block). Because each iteration self-pairs
  its own signal with its own trailing wait, the loop is already balanced
  without the offset ping-pong's boundary pieces, so **both Rule 3 (the pre-loop
  priming signal) and Rule 5 (the loop-exit drain wait) are disabled** in this
  mode (they would otherwise be unpaired). Ignored when the drain gate is on.

---

## Rule 1 -- Post-`GSU==1` priming signal

Emitted immediately **after** each `label_GSU_1:` label (Tensile's
post-`GSU==1`-guard label), which survives region extraction, so idempotency
handles re-entry across scopes.

**Ungated** -- a plain `WaveIdx`-gated cluster signal; the signal is
just the priming credit for the paired wait, so it fires on every control-flow
path:

```asm
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<N>
    s_barrier_signal -3
  label_skipCBPreSignal_<N>:
```

**Gated (default)** -- wrapped in an outer `LoopCounterL != 0` gate, with a workgroup-scope
`s_barrier_signal -1` / `s_barrier_wait -1` pair **inside** the skip region and
**before** the inner `WaveIdx` gate so every wave reaches the post-`GSU==1` join
before any wave signals:

```asm
    s_cmp_eq_u32 s[sgprLoopCounterL], 0
    s_cbranch_scc1 label_skipCBSignal_LCL_<N>
    s_barrier_signal -1                                   // workgroup signal
    s_barrier_wait -1                                     // sync workgroup before cluster signal
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<N>
    s_barrier_signal -3
  label_skipCBPreSignal_<N>:
  label_skipCBSignal_LCL_<N>:
```

---

## Rule 2 -- First kernel load wait (kernel scope only)

A single `s_barrier_wait -3` immediately before the first `tensor_load_to_lds`
of the whole kernel. Only fires when `isKernelScope == true` because the
"first `tensor_load` of the whole kernel" anchor is meaningful only at kernel
scope. Idempotency: skipped when the load is already preceded by a cluster-scope
wait.

---

## Rule 3 -- LDS-publication priming signal

A priming cluster signal at the LDS publication point that precedes
`label_openLoopL:`. **Enabled in every scheme except the ungated
signal-before-wait legacy mode** (`kClusterBarrierDrainGateEnabled == false &&
kRule4SignalBeforeWaitEnabled == true`), where the loop self-pairs each
iteration and needs no priming. It primes the loop's offset ping-pong: the
signal is consumed by the first per-iteration cluster WAIT (Rule 4) inside the
loop body.

### Anchor modes (backward scan from `label_openLoopL:`)

**(a) Publication point already exists** (typical for `PrefetchLocalRead > 0`).
An `s_barrier_wait -1` is already present between the prefetch tail and
`label_openLoopL:`. Anchor at the successor of that wait; no new workgroup sync
is synthesized. The scan stops as soon as a `tensor_load_to_lds` is reached
(that instruction marks the prefetch section, before any workgroup sync could
sit). Defers to Rule 4 if the same wait would also be a Rule-4 trigger.

**(b) No publication point** (typical for `PrefetchLocalRead == 0`). No
`s_barrier_wait -1` before `label_openLoopL:`. Only active when `plrValue == 0`;
anchor at the label and synthesize an `s_barrier_signal -1` / `s_barrier_wait -1`
pair so the workgroup has published its LDS writes before any wave signals.

**Ungated** -- emit only the `WaveIdx`-gated cluster signal (preceded
by the synthesized workgroup sync pair in mode (b)):

```asm
    s_barrier_signal -1                                   // workgroup signal (mode (b) only)
    s_barrier_wait -1                                     // workgroup sync    (mode (b) only)
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<N>
    s_barrier_signal -3
  label_skipCBPreSignal_<N>:
```

**Gated (default)** -- wrap the signal in an outer `s_cmp_le_u32 s[sgprLoopCounterL],
pgrValue` gate (mirroring Tensile's own loop-entry guard), with the workgroup
sync pair inside the skip region in mode (b):

```asm
    s_cmp_le_u32 s[sgprLoopCounterL], <pgrValue>          // outer LCL gate
    s_cbranch_scc1 label_skipCBSignal_LCL_<N>
    s_barrier_signal -1                                   // workgroup signal (mode (b) only)
    s_barrier_wait -1                                     // workgroup sync    (mode (b) only)
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<N>
    s_barrier_signal -3
  label_skipCBPreSignal_<N>:
  label_skipCBSignal_LCL_<N>:
```

**Idempotency.** The backward scan also flags whether a cluster-scope
signal/wait already sits in the section; if so, Rule 3 self-disables. It also
defers to Rule 4 when the existing workgroup wait is already a Rule-4 trigger or
is already followed by a cluster handshake. The label/instruction-based anchor
survives `ScopeAdaptor::moveIRToBlock`, so Rule 3 keeps working at kernel scope.

---

## Rule 4 -- Per-iteration cluster handshake before loop loads

A cluster handshake around each workgroup-scope barrier that precedes a
`tensor_load_to_lds`. For every load in a label-/branch-delimited segment, the
pass walks backward to the nearest preceding `s_barrier_wait -1` (the LDS
publication point); triggers are deduplicated by identity, so multiple loads
sharing the same anchor wait yield exactly one handshake. The backward scan
stays within the load's segment so it never crosses a control-flow boundary,
giving per-iteration coverage for `ExpandPointerSwap` unrolled loops.

Rule 4 owns **only the main-loop region**: the forward scan is bounded at the
`/* Tail Loop */` marker so any tail load belongs exclusively to Rule 6 and the
two rules never share an anchor wait. When no marker exists (e.g. region scope,
where it is erased) the scan sweeps the whole block.

**Wait placement (random-hang fix).** In the wait-before-signal schemes (the
default gated + ungated) the handshake is **split across the workgroup barrier**: the
cluster `s_barrier_wait -3` (plus its `LCL` drain gate, if any) is planted
immediately **before** the anchor wait's paired `s_barrier_signal -1` (the
workgroup signal), while the `WaveIdx`-gated `s_barrier_signal -3` (plus its
drain gate) and any SCC restore stay **after** the workgroup `s_barrier_wait -1`.
This puts the workgroup `signal -1` / `wait -1` **between** the cluster
`wait -3` and `signal -3`.

Rationale: the per-iteration handshake is an offset ping-pong -- each `wait -3`
consumes the *previous* iteration's `signal -3`, and only wave 0 issues the
*next-generation* `signal -3`. If the workgroup barrier sat before the cluster
wait (old layout `signal -1`/`wait -1`/`wait -3`/`signal -3`), nothing fenced
wave 0 between finishing this iteration's `wait -3` and emitting the next
iteration's `signal -3`; a fast wave 0 could race ahead and signal the next
generation while sibling waves were still draining the current-generation
`wait -3`, landing an extra signal in the wrong generation, desynchronizing the
barrier's signal/wait count and occasionally deadlocking (the random hang).
Fencing the workgroup between the cluster wait and signal guarantees every wave
completes the current-generation `wait -3` before wave 0 can emit the
next-generation `signal -3`. The legacy signal-before-wait layout self-pairs
each iteration's own signal/wait and is left untouched.

**Ungated, wait-before-signal
(`kClusterBarrierDrainGateEnabled == false && kRule4SignalBeforeWaitEnabled == false`)**
-- the bare `s_barrier_wait -3` moves
above the workgroup signal; the `WaveIdx`-gated `s_barrier_signal -3` and any SCC
restore stay after the workgroup wait:

```asm
    s_barrier_wait -3                                     // cluster barrier wait (moved above workgroup signal)
    s_barrier_signal -1                                   // workgroup signal
    s_barrier_wait -1                                     // workgroup sync
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<N>
    s_barrier_signal -3
  label_skipCBPreSignal_<N>:
    <clone of live upstream s_cmp_* LCL>                  // SCC restore (if any)
```

**Ungated, signal-before-wait (legacy:
`kRule4SignalBeforeWaitEnabled == true`)** -- unchanged: the `WaveIdx`-gated
`s_barrier_signal -3` first, then any SCC restore, then the bare
`s_barrier_wait -3` last (so the wait remains the load's immediate predecessor,
after the workgroup wait):

```asm
    s_barrier_signal -1                                   // workgroup signal
    s_barrier_wait -1                                     // workgroup sync
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<N>
    s_barrier_signal -3
  label_skipCBPreSignal_<N>:
    <clone of live upstream s_cmp_* LCL>                  // SCC restore (if any)
    s_barrier_wait -3                                     // cluster barrier wait
```

**Gated (default)** -- two asymmetric `LoopCounterL` drain gates. The WAIT (no inner
`WaveIdx` gate; every wave skips/executes in lockstep) is skipped at
`LCL <= pgrValue - preDec`; the SIGNAL is skipped one stage earlier at
`LCL <= pgrValue + 1 - preDec`, where `preDec` is the sum of any
`s_sub s[sgprLoopCounterL], ..., imm` decrements the schedule hoisted above the
anchor. As above, the gated WAIT block moves before the workgroup signal while
the gated SIGNAL block and SCC restore stay after the workgroup wait:

```asm
    s_cmp_le_i32 s[sgprLoopCounterL], <pgr - preDec>      // WAIT drain gate
    s_cbranch_scc1 label_skipCBWait_LCL_<N>
    s_barrier_wait -3                                     // cluster barrier wait (moved above workgroup signal)
  label_skipCBWait_LCL_<N>:
    s_barrier_signal -1                                   // workgroup signal
    s_barrier_wait -1                                     // workgroup sync
    s_cmp_le_i32 s[sgprLoopCounterL], <pgr + 1 - preDec>  // SIGNAL drain gate
    s_cbranch_scc1 label_skipCBSignal_LCL_<N>
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<N>
    s_barrier_signal -3
  label_skipCBPreSignal_<N>:
  label_skipCBSignal_LCL_<N>:
    <clone of live upstream s_cmp_* LCL>                  // SCC restore (if any)
```

### SCC restore

In all schemes the pass consults `findLiveSccCmpUpstream`: if scheduling (e.g.
`ScheduleIterAlg=4`) hoisted a live loop-exit `s_cmp_* s[sgprLoopCounterL], imm`
above the anchor whose SCC a downstream `s_cbranch_scc{0,1}` still consumes, a
verbatim clone of it is re-emitted to rebuild the SCC that the WaveIdx /
drain-gate compares clobber. Placement depends on the ordering:

- wait-before-signal (ungated) and gated (default): **after** the entire inserted block
  (past every skip label, so it runs on the gated and fall-through paths alike);
- signal-before-wait (legacy): immediately **before** the trailing wait.

Only an `s_cmp_*` value compare (`isCompareClass`) is accepted, because it writes
SCC and nothing else. Because a live compare is accepted only when it is the
first SCC writer above the anchor (and `s_sub LCL` also writes SCC),
`LoopCounterL` is guaranteed unchanged between the compare and the restore point,
so re-running it reproduces the original SCC.

---

## Rule 5 -- Loop-exit trailing wait (ungated scheme only; disabled by default)

A single bare `s_barrier_wait -3` at the loop-exit convergence label, consuming
the loop's last otherwise-orphaned cluster signal. **Only enabled in the ungated
wait-before-signal scheme** (`kClusterBarrierDrainGateEnabled == false &&
kRule4SignalBeforeWaitEnabled == false`). It is disabled in the other two modes:

- gated scheme -- the asymmetric Rule 4 gate already leaves one extra in-loop
  WAIT that consumes the last signal;
- ungated signal-before-wait (legacy) -- each iteration self-pairs its own
  signal/wait, so a loop-exit wait would be unpaired.

Anchor selection: scan backward from the kernel's first `tensor_load_to_lds` for
the `s[sgprLoopCounterL] == 0` guard and resolve which label the `LCL == 0` path
actually lands on (`resolveLoopCounterLZeroTargetLabel`) -- the convergence point
of every "loop bypassed" path:

- short-branch encoding -> `label_LoopEndL` (both the normal loop exit and the
  `LCL == 0` skip land here);
- long-branch encoding -> `label_PrefetchGlobalLastIterEnd` (the `LCL == 0` long
  branch and the `LCL == 1` drain both fall through here).

Falls back to `label_LoopEndL` when no guard is found. Emitted after the resolved
label:

```asm
  <resolved label>:
    s_barrier_wait -3                                     // cluster barrier wait (loop end)
```

Idempotency: skipped when the label is already followed by a cluster-scope wait.

---

## Rule 6 -- Tail-loop cluster handshake

A paired handshake around the tail loop, anchored on the `/* Tail Loop */`
`TEXTBLOCK` marker. Emitted at two distinct sites because the workgroup wait and
the tail load sit in different label-/branch-delimited segments (the tail
TDM-reset block between them is not synchronization-critical, so collapsing both
into one site would unnecessarily serialize the cluster). **Independent of both
switches.** Region-scope invocations never observe the marker
(`ScopeAdaptor::moveIRToBlock` erases `TEXTBLOCK` directives), so the rule
self-disables there.

### 6a -- tail priming signal

A `WaveIdx`-gated signal-only block (no `LoopCounterL` gate). Two anchor forms:

- **workgroup wait exists** -- emitted immediately **after** the nearest
  preceding `s_barrier_wait -1` of the tail load (searched backward from the
  load, bounded by the marker). Defers to Rule 4 if that wait is already a
  Rule-4 trigger; idempotency skip when already followed by a cluster handshake.
- **no-wait fallback** (`PrefetchLocalRead=0` style) -- when a tail load exists
  but no `s_barrier_wait -1` sits between the marker and it, synthesize the LDS
  publication point immediately **after** the marker: an
  `s_barrier_signal -1` / `s_barrier_wait -1` pair (the wait carrying comment
  `"tail workgroup sync"`) followed by the `WaveIdx`-gated cluster signal. This
  mirrors Rule 3's mode (b) synthesis but ungated. On a re-run the synthesized
  `s_barrier_wait -1` is detected as a real wait, so the fallback is not taken
  again (idempotent).

```asm
    s_barrier_signal -1                                   // workgroup signal (fallback only)
    s_barrier_wait -1                                     // tail workgroup sync (fallback only)
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<N>
    s_barrier_signal -3
  label_skipCBPreSignal_<N>:
```

### 6b -- tail load wait

A single `s_barrier_wait -3` immediately **before** the first
`tensor_load_to_lds` that follows the marker. Idempotency skip when the load is
already preceded by a cluster wait.

---

## Parameters

### `isKernelScope` (default `true`)

Must be `true` for the GFX1250 backend pipeline (whole-kernel insertion). When
`false`, the pass is intended for region-scoped invocation via
`createKernelToRegionsPassAdaptor` (not used by the backend today). Rule 2 only
fires when this is `true` because the "first `tensor_load` of the whole kernel"
anchor is meaningful only at kernel scope.

### `pgrValue` (default `1`, i.e. `PrefetchGlobalRead=1`)

Tensile's `PrefetchGlobalRead` setting. Consulted only by the **gated** scheme:
Rule 4's asymmetric drain-gate thresholds (`LCL <= pgrValue` /
`LCL <= pgrValue + 1`) and Rule 3's `LCL <= pgrValue` gate, which is the default.
Unused in the ungated scheme.

### `plrValue` (default `1`, i.e. `PrefetchLocalRead=1`)

Tensile's `PrefetchLocalRead` setting. Selects Rule 3's anchor mode (b): when
`plrValue == 0` and the backward scan from `label_openLoopL:` finds no
`s_barrier_wait -1` before reaching the prefetch boundary, the rule synthesizes
the missing workgroup publication point. Any non-zero value disables mode (b).
(Rule 6a's no-wait fallback is driven by the tail-loop IR shape, not by
`plrValue`.)

---

## Analysis invalidation

This pass mutates the CFG (new branches and new labels), so dependent CFG /
dominance analyses are invalidated (`run` returns `PreservedAnalyses::none()`).

---

## See Also

- [Architecture Overview](architecture.md) -- system architecture and pass pipeline
- `src/transforms/asm/InsertClusterBarrierPass.cpp` -- implementation
- `include/stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp` -- public API
