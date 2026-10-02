# Insert Cluster Barrier Pass

`createInsertClusterBarrierPass` inserts cluster-barrier handshakes at four rules
covering the main and tail loops.

The pass is created via:

```cpp
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass(
    bool streamKMulticast = false, int pgrValue = 1, int rule3SignalLeadCycles = 100,
    int rule3Mode = 0, int producerDrain = -1, int loopPeriod = 1, int driftCanaryCycles = 0,
    int skewCanaryCycles = 0);
```

Gfx1250Backend fills these from the module options `StreamKMulticast`,
`PrefetchGlobalRead`, `ClusterBarrierRule3SignalLeadCycles` (resolved through
SchedulingKnobHeuristics), `ClusterBarrierRule3Mode`, `ClusterProducerDrain`,
`ClusterBarrierLoopPeriod`, `ClusterDriftCanaryCycles` and `ClusterSkewCanaryCycles`.
stinkytofu-opt takes them as `--InsertClusterBarrierPass=streamKMulticast,pgr=2,lead=200,rule3Mode=2,producerDrain=0,loopPeriod=3,driftCanary=256,skewCanary=512`.
See [Placement modes](#placement-modes), [Producer drain](#producer-drain),
[Loop period](#loop-period) and [Cluster canaries](#cluster-canaries).

## Overview

Rules are numbered in **kernel-execution order** -- the rule with the lowest
number is the first to fire when the kernel runs.

The cluster handshake uses signal/wait pairs at two scopes:

- **Workgroup scope**: `s_barrier_signal -1` / `s_barrier_wait -1`
- **Cluster scope**: `s_barrier_signal -3` / `s_barrier_wait -3`

`<HASH>` in the emitted labels is a fresh 16-character alphanumeric identifier
generated per insertion. Only the first wave (`WaveIdx == 0`) executes the
cluster signal; the other waves fall through to the label.

**Idempotency:** each rule has its own skip check, so re-running the pass is a
no-op when the handshake is already present.

---

## Rule 1 -- Post-GSU==1 signal-only

Signal-only (no leading cluster wait), emitted immediately **after** each
`label_GSU_1:` label, wrapped in an outer `LoopCounterL != 0` gate so the
cluster-barrier signal only fires on non-zero iterations.

A workgroup-scope `s_barrier_signal -1` / `s_barrier_wait -1` pair sits **inside**
the outer LCL skip region (and **before** the inner `WaveIdx` gate) so every wave
in the workgroup has reached the post-`GSU==1` join before any wave issues the
cluster signal:

```asm
    s_cmp_eq_u32 s[sgprLoopCounterL], 0
    s_cbranch_scc1 label_skipCBPreSignal_LCL_<HASH_OUTER>
    s_barrier_signal -1
    s_barrier_wait -1
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<HASH_INNER>
    s_barrier_signal -3
  label_skipCBPreSignal_<HASH_INNER>:
  label_skipCBPreSignal_LCL_<HASH_OUTER>:
```

---

## Rule 2 -- First kernel load wait

A single `s_barrier_wait -3` immediately before the first `tensor_load_to_lds`
of the whole kernel, above any wait-cnt drains that precede it (see
[Drain hoisting](#drain-hoisting)).

---

## Rule 3 -- Loop-body cluster handshake

Rule 3 applies only to **`tensor_load_to_lds` inside a loop**. For each qualifying
load whose segment contains a preceding workgroup `s_barrier_signal -1` (the
**trigger**), the pass emits a full handshake independent of Rules 1 and 2:

- **Rule 3(a)** -- WaveIdx-gated `s_barrier_signal -3` at the signal anchor.
- **Rule 3(b)** -- bare `s_barrier_wait -3` immediately before the trigger.

Multiple loads sharing the same workgroup signal receive one handshake.

### Anchor resolution

1. The **wait anchor** (Rule 3(b)) is the workgroup `s_barrier_signal -1`.
2. The **signal anchor** (Rule 3(a)) is found by walking backward from the wait
   anchor until it stands `ClusterBarrierRule3SignalLeadCycles` estimated cycles
   ahead of it (default 100; for 0 and negative values see below), somewhere the
   handshake may legally go. The lead is a target, not a cap: a spot inside a
   live SCC range is not one the walk may take, so it keeps climbing, and
   `kRule3SignalMaxLeadCycles` is what bounds the answer. Past that ceiling it
   turns around and sinks back towards the wait instead.

   It stops outright at a preceding handshake or at any `s_barrier_wait -3`, so
   cluster phases never overlap, and at a call or an unconditional branch. With
   `kRule3CrossLoop` false every label and branch stops it too, which confines it
   to the wait's own segment.

With no lead to buy -- a lead of 0, or no cycle estimate at the wait -- the signal
co-locates with the wait, but only where SCC is dead there. Where SCC is live at
the wait, the walk runs as if the lead were already met at the wait and settles
on the nearest spot above it that SCC leaves free (see [SCC](#scc)). A walk with
a lead that comes back to the wait's own spot -- a boundary or a hard stop ends
it before it buys any -- has bought none, and is placed the same way.

A negative lead is unset, like the module option's -1. Gfx1250Backend resolves
that through SchedulingKnobHeuristics before building the pass; the pass resolves
one it is handed directly, such as stinkytofu-opt's `lead=-1`, the same way. A
pass sees no module and so no main-loop statistics, which leaves the static
default, 100.

Cross-segment hoisting and loop-carried compensation are gated by the compile-time
switch `cluster_barrier::kRule3CrossLoop` in
`InsertClusterBarrierPass.hpp` (default **false**). See
[`kRule3CrossLoop`](#krule3crossloop) below.

### SCC

The signal block opens with `s_cmp_eq_u32 s[sgprWaveIdx], 0`, so wherever it
lands it destroys the SCC value standing there. Nothing puts that value back;
the anchor search is what keeps the block out of a live range in the first
place, and it may give up lead to do so — climbing to a spot in front of the
def, or sinking below the range towards the wait. A range the loop closes
across its back edge counts too, which is why the climb carries a liveness flag
of its own alongside the forward scan the placement check runs.

#### Liveness without a CFG — `isSccLiveIn`

`isSccLiveIn(at)` answers "is SCC live at the point in front of `at`". Liveness
is a property of the code that *follows* a point, so the walk runs forward while
the answer describes where it started — the name says which point it is about,
not which way the walk goes. Both the backward anchor climb and the downward
correction ask it about the spot they are standing on, which is where the
handshake's `s_cmp_eq_u32` would land.

This pass runs before `CFGBuilderPass`, on one flat block holding the whole
kernel with its labels and branches still inline, so reading the block top to
bottom describes the fall-through and nothing else. That is not good enough for
SCC, whose consumer is typically a branch — the reader is frequently *not* on
the fall-through, and the code below an unconditional branch may not be
reachable at all. The flat layout is also the way out: every branch target is a
label in the same block, so the walk resolves each branch and follows both of
its edges. It is a worklist over instruction positions, with a `walked` set that
both merges joins and terminates back edges, and a label index built lazily on
the first branch so the common case (an SCC access within a few instructions)
pays nothing for it.

Only a `false` grants permission to clobber SCC, so anything the walk cannot see
through reads as live. Two things are left:

| Encountered | Answer |
|---|---|
| Reads SCC | live |
| Rewrites SCC | dead — nothing past it wants the old value |
| Call | **live** — the callee may use SCC as scratch |
| Branch with no matching label in the block | **live** — target unknown |
| Branch with a matching label | follow both edges (fall-through only if conditional) |
| `s_endpgm`, or the end of the block | dead — no successor block to run |

The last row is why the end of the block is not an unknown: with branches
followed, running out of instructions is a genuine path end rather than an
artifact of having strolled past a branch. `s_endpgm` is stopped at explicitly
for the same reason — it ends that path, and the code textually below it belongs
to some other one.

#### Placement stops vs. liveness stops

`findSccDeadAnchorBelow` and `isSccLiveIn` stop at different things, on purpose,
because they answer different questions. The first is a *placement* search
("where may this signal go?"); the second is a *dataflow* question ("does
anything read SCC from here?").

| | `findSccDeadAnchorBelow` | `isSccLiveIn` |
|---|---|---|
| The wait being led (`limit`) | stops — a signal below its own wait is not a handshake | reads through — a value consumed below the wait is live above it |
| `s_barrier_wait -3` | stops — the signal would lose its token there | reads through — a barrier neither reads nor writes SCC |
| Segment boundary | stops, except the SCC-reading exit branch that holds the range open | reads through, following the branch |

Each stop rule is load-bearing in its own walk and would be wrong in the other.
A range that reaches the wait is the shape that leaves the placement walk with
nothing to return; the caller then aborts rather than clobber (`Rule 3 signal
anchor: SCC live at the wait`).

What `findSccDeadAnchorBelow` returns is an *anchor* — the instruction the
handshake is planted in front of, as everywhere else in this pass — not the
instruction that clobbers SCC, though the two often coincide. A range is held
open by its last reader, so the first dead point is directly below that reader:
the clobber when the clobber is what directly follows, and just the next
instruction along otherwise.

### Drain hoisting

`StinkyWaitCntInsertionPass` runs before this pass and anchors its counter
drains on the same instructions the cluster waits target, so the slot right
before an anchor is usually already occupied by an `s_wait_tensorcnt` (or
another `s_wait_*cnt`). Every cluster wait -- Rules 2, 3(b) and 4(b) -- is
therefore emitted **above** that run of drains:

```asm
    s_barrier_wait -3
    s_wait_tensorcnt N
    s_barrier_signal -1
```

Both orders are correct; the inverted one measured materially slower, and that
measurement is the entire justification. **The mechanism is not established.**
Both instructions block on independent conditions -- a per-wave local counter,
and peer arrival at the barrier -- and two such waits commute, so the obvious
"the drain overlaps the barrier latency" argument does not actually hold.

Wait-cnt instructions never write SCC, so hoisting past them cannot move a
cluster wait into or out of a live SCC range.

### Emitted shape (separated anchors)

```asm
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<HASH>
    s_barrier_signal -3
  label_skipCBPreSignal_<HASH>:
    ...
    s_barrier_wait -3
    <wait-cnt drains hoisted below the cluster wait>
    s_barrier_signal -1
    s_barrier_wait -1
    tensor_load_to_lds ...
```

### Placement modes

`ClusterBarrierRule3Mode` moves the handshake around its trigger, the workgroup
barrier that protects the slot the refill load writes. Every mode keeps the
signal and the wait in the trigger's segment and above that load.

| Mode | Signal | Wait |
|---|---|---|
| 0 (default) | the signal anchor above, at the lead | above the trigger's drains |
| 1 | above the trigger's drains, where mode 0 waits | right after the trigger's `s_barrier_wait -1` |
| 2 | right after the trigger's `s_barrier_wait -1`, sunk below any live SCC range | right after the signal |

Mode 1 overlaps the cluster barrier with the workgroup barrier. Mode 2 puts every
workgroup's last read of the slot before any workgroup's refill of it, whichever
workgroups a multicast load writes, and pays the whole cluster barrier latency on
every trip.

A block takes mode 1 or 2 for all of its triggers or for none. When a trigger's
`s_barrier_wait -1` does not come before its load, or SCC is live where the signal
block would go (mode 1: above the drains; mode 2: anywhere from that wait down to
the load), the whole block keeps mode 0 and the pass emits a `Rule3ModeFallback`
remark that names the cause. The decision is per block because the signal climb stops at the other
handshakes' waits by their triggers, which is where those waits are only when the
whole block is placed by mode 0.

Modes 1 and 2 leave no cluster wait right above the trigger, so a trigger with a
cluster barrier between it and its load counts as already handled. This keeps a
re-run a no-op.

### Producer drain

With `streamKMulticast` and `pgrValue >= 2`, the pass plants `s_wait_tensorcnt 0`
right after each Rule 3 tensor-load group (the whole run of adjacent loads), so the
cooperative load retires before the back edge. `ClusterProducerDrain` overrides the
condition: -1 keeps it, 0 never drains, 1 drains after every Rule 3 group.
TensileLite no longer passes the `StreamKMulticast` module option (#12817,
e69ccbe3), so on TensileLite kernels -1 (auto) never drains and 1 forces the drain.

### Loop period

`ClusterBarrierLoopPeriod` (k) numbers each loop's triggers in program order from
the loop head and keeps the handshake of the ones whose number is a multiple of k:
1 (default) keeps every trigger, 0 keeps none, so the loop body runs without cluster
barriers and Rules 1, 2 and 4 are all that is left. The same triggers are kept on
every trip, which spaces the kept handshakes evenly only when k divides the
triggers per trip (one per loop copy: 3 with HalfPLR, 2 with ExpandPointerSwap
alone, 1 otherwise). A loop where it does not keeps every handshake, and the pass
emits a `LoopPeriodFallback` remark for it.

No branch is involved: every kept handshake is a whole pair in its own segment,
placed exactly where period 1 places it, and every workgroup of a cluster runs the
same trips, so the phases stay balanced. A dropped trigger keeps its producer
drain, and still counts as Rule 3's when Rule 4 checks for a conflict. The
numbering includes triggers a previous run already handled, so a re-run drops the
same ones again. With no handshake to mark it handled, a dropped trigger is collected
again on a re-run, which finds the producer drain already planted behind its group
(by its comment, before the next tensor load) and plants no second one.

### Cluster canaries

`ClusterDriftCanaryCycles` and `ClusterSkewCanaryCycles` are diagnostics, off at 0.
Each plants, at every Rule 3 site, a sleep of about that many cycles behind
`s_bitcmp1_b32` / `s_cbranch_scc0` on bit 0 of `sgprWorkGroup0`, then the same on
bit 0 of `sgprWorkGroup1`. Neighbouring workgroups of a cluster differ in one of
those bits, so in every multicast group some workgroups sleep that much longer
than others. `s_sleep` counts units of 64 clocks.

| Canary | Where | What it shows |
|---|---|---|
| drift | right after the Rule 3 signal; where the loop period dropped the handshake, where its wait would have gone | with period 0 the delay piles up trip after trip; with handshakes it lands between a workgroup's signal and its last reads of the slot |
| skew | right above the Rule 3 wait | a fixed spread in when the workgroups issue the refill load, reset by every handshake |

The canaries move nothing else. They need the `.set` of both workgroup-id symbols
(else the pass emits `ClusterCanaryNoWorkGroupId` and plants none), and a spot
where SCC is dead (else that site is skipped and counted in a `ClusterCanarySkipped`
remark). A re-run does not plant them twice.

They stay out of a Rule 3 site that comes after the latch of a loop nested in its
own loop: the site past the main loop where a PAP kernel prefetches the next tile.
By then the persistent loop may have moved `sgprWorkGroup0` and `sgprWorkGroup1` on
to that tile, so the canary there would sort the cluster by another tile's ids. The
site keeps its handshake; its canaries are counted in a `ClusterCanarySkipped` remark
of their own ("past an inner loop").

---

## Rule 4 -- Tail-loop cluster handshake (paired)

Two emission sites because the workgroup wait and the tail load sit in different
label/branch-delimited segments:

- **Rule 4(a)** -- signal-only handshake immediately **after** the nearest
  preceding `s_barrier_wait -1` of the tail load.
- **Rule 4(b)** -- bare `s_barrier_wait -3` immediately **before** the first
  `tensor_load_to_lds` after the `/* Tail Loop */` TEXTBLOCK marker.

Rule 4(a) is skipped when Rule 3 already targets the same workgroup signal.
Region-scope invocations never observe the TEXTBLOCK marker, so Rule 4
self-disables there.

---

## kRule3CrossLoop

Compile-time switch in `cluster_barrier::kRule3CrossLoop`
(`include/stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp`). Rebuild
rocisa after changing it. It gates **Rule 3 cross-loop hoisting** and the
scheduler **live-out SCC lead ceiling** in `StinkyDAGSchedulerPass`.

### Shared behavior (true and false)

With cluster barrier enabled:

- Rules 1, 2, and 4 are unchanged.
- Rule 3 still inserts signal/wait handshakes for each qualifying load segment.
- Rule 3 still walks backward from the wait for cycle lead and SCC clearance.
- `StinkyDAGSchedulerPass::applyClusterBarrierSccRule` still **pins** live-out
  SCC defs (loop counter compares) **after** every cluster barrier wait in the
  region — the compare may not issue before the last wait completes.

### `kRule3CrossLoop == false` (default)

**InsertClusterBarrierPass**

- `maxSegmentHops = 0`: the Rule 3 climb cannot leave the wait's segment.
- Signals stay **in-segment**, paired near their waits.
- No **loop-wrap** at the latch: the climb stops at the loop head rather than
  following the back edge.
- No **preheader signal**, **loop-exit drain wait**, or **`skipCBWait`**
  bypass labels (`emitLoopCarriedCompensation` is never called).

**StinkyDAGSchedulerPass**

- Live-out SCC defs get barrier-pin edges only.
- `DAGNode::earliestClock` stays `INT_MIN`.
- Once the pin frees the compare, the scheduler may issue it immediately — often
  far from the latch branch that reads SCC.

**Typical assembly:** signal and wait remain close in the loop body; loop exit
has no `drain loop-carried cluster signal` wait.

### `kRule3CrossLoop == true` (opt-in)

**InsertClusterBarrierPass**

- `maxSegmentHops = kMaxSegmentHops` (1): Rule 3 may cross **one** segment
  boundary per climb.
- When the climb reaches the loop head short of the lead target, it **follows the
  latch** (`findLatchBranchFor`) rather than leaving the loop textually into the
  preheader, and keeps accumulating from there — so the lead is the sum of the
  two parts (wait→head and latch→anchor). It only crosses while the preheader
  has a spot for the compensating signal
  (`preheaderCanTakeCompensatingSignal`); otherwise it comes to rest below the
  head as if the hops had run out.
- When `found.hops > 0`, **loop-carried compensation** runs via
  `emitLoopCarriedCompensation`:
  - Optional **preheader signal** when the climb crossed the back edge.
    `findPreLoopSignalAnchor` climbs from the loop head and comes to rest below the
    nearest `s_barrier_wait -1`, `s_barrier_wait -3`, kernel `label_*`, or
    `tensor_load_to_lds` (pass `skipCB*` labels are skipped). A cluster wait stops
    it for a second reason: that wait drinks a token, so a signal planted above one
    never reaches the first trip. Signal-only after a workgroup wait — the one stop
    that already proves the group has gathered — and workgroup barrier + signal
    after anything else. The anchor slides down to the first SCC-dead spot when SCC
    is live where it landed, and a preheader offering no spot at all is why the
    climb declines to cross the back edge in the first place.
  - **Drain wait** at loop exit: `s_barrier_wait -3` with comment
    `drain loop-carried cluster signal`.
  - **`label_*_skipCBWait`** on paths that leave the loop with no token in
    flight (zero-trip guards, exits below a wait, etc.).

**StinkyDAGSchedulerPass**

- In addition to barrier pins, live-out SCC defs get
  `earliestClock = regionCycles - kLiveOutSccDefLeadCycles` (50 cycles).
- The compare is held near the latch branch instead of issuing immediately after
  the last barrier wait.

**Typical assembly:** signals appear earlier in the loop body (hoisted from
waits); loop latch may compare `counterL==0` before the back-edge branch; loop
exit has drain wait + skip path.

### Code map

| Location | false | true |
|----------|-------|------|
| `maxSegmentHops` in Rule 3 | 0 | 1 |
| Segment-boundary climb | stops at boundary | may cross one hop |
| Latch follow at the loop head | not reached | wraps when the climb is short of the lead |
| `if (found.hops > 0)` block | skipped | preheader + hoistedLoops |
| `emitLoopCarriedCompensation` | not called | drain / skipCBWait |
| Live-out SCC `earliestClock` | not set | ≤50 cycles from region end |

### Tests

- `InsertClusterBarrierPassTest` / `DAGSchedulerPassTest`: cross-loop cases use
  `IF_RULE3_CROSS_LOOP`, so they are omitted from the binary while
  `STINKY_KRULE3_CROSS_LOOP` is 0. Cases that need the switch off stay in the binary
  and `GTEST_SKIP` themselves when it is 1. Covering the pass therefore means
  building and running both settings.

---

## Source files

- `src/transforms/asm/InsertClusterBarrierPass.cpp` -- implementation
- `include/stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp` -- public API
