# Insert Cluster Barrier Pass

`createInsertClusterBarrierPass` inserts cluster-barrier handshakes at four rules
covering the main and tail loops, or -- under StreamK cluster multicast --
producer-side tensor drains instead.

The pass is created via:

```cpp
enum StreamKMulticastMode : int {
    kStreamKMulticastOff = 0,
    kStreamKMulticastOn = 1,
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass(
    StreamKMulticastMode streamKMulticast = kStreamKMulticastOff);
```

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

**Rules 1--4 are not unconditional.** All four fire only in
`kStreamKMulticastOff`; see [Operating modes](#operating-modes) below.

---

## Operating modes

The `streamKMulticast` argument selects between two disjoint behaviours:

| Mode | Rules 1--4 | Producer tensor drains |
|------|------------|------------------------|
| `kStreamKMulticastOff` (default) | emitted | none |
| `kStreamKMulticastOn` | **suppressed** | emitted -- see [Producer-side tensor drains](#producer-side-tensor-drains-multicast-on) |

The value is plumbed in from TensileLite: `streamKMulticast(kernel)`
(`Tensile/Common/Utilities.py`) becomes the `StreamKMulticast` module option
(`Tensile/KernelWriter.py:6848`), crosses the binding as
`ModuleOptions::StreamKMulticast`
(`include/stinkytofu/bindings/python/Module.hpp:91` -- an `int`, so the enum can
gain modes without another ABI change) and is cast back to
`StreamKMulticastMode` in `buildGfx1250Pipeline`
(`src/pipeline/backend/Gfx1250Backend.cpp:221-223`, under `ClusterBarrier`).

### Why multicast suppresses the rules

Every rule below plants a `-3` signal or wait inside the kernel's loop or tail
structure. That is sound only when every workgroup in the cluster reaches it the
same number of times.

Under StreamK with `StreamKForceDPOnly=0` it does not. Peers in one cluster are
assigned different iteration counts by construction, so `sgprLoopCounterL`
differs across the cluster; a per-iteration compiler-inserted `-3` then deadlocks
on unbalanced arrive/wait counts, one peer signalling N times while another waits
N+1 times. Rule 1's `LoopCounterL != 0` gate only excludes the zero-iteration
case -- it does not equalise the nonzero ones.

In this mode TensileLite owns the handshakes and emits them only where it can
predicate them explicitly, so every peer arrives the same number of times:
`_clusterElectArriveSignal`, `streamKMulticastPrologueSignal`,
`streamKMulticastProloguePrefetchHandshake`,
`streamKMulticastZeroIterClusterWait` and `streamKClusterPadEarlyExit` in
`Tensile/Components/StreamK.py`. What the pass still contributes is the drain
placement, which depends on the post-scheduling instruction order TensileLite
cannot see.

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

## Rule 3 -- Cluster handshake before loop loads

For each `tensor_load_to_lds` whose segment contains a preceding workgroup
`s_barrier_signal -1`, the pass emits a cluster handshake:

- **Rule 3(a)** -- WaveIdx-gated `s_barrier_signal -3` at the signal anchor.
- **Rule 3(b)** -- bare `s_barrier_wait -3` immediately before the workgroup
  `s_barrier_signal -1`.

Multiple loads sharing the same workgroup signal receive one handshake.

### Anchor resolution

1. The **wait anchor** (Rule 3(b)) is the workgroup `s_barrier_signal -1`.
2. The **signal anchor** (Rule 3(a)) is found by walking backward from the wait
   anchor by up to `kRule3SignalLeadCycles` estimated cycles, bounded by the
   current segment. The walk stops at a preceding handshake so cluster phases
   never overlap.

When cycle estimates are unavailable, the signal co-locates with the wait.

### SCC restore

If SIA hoisted a live loop-exit `s_cmp_eq LCL, imm` whose SCC a downstream
`cbranch` consumes, and no instruction between the signal and wait anchors
redefines SCC, a clone of that compare is re-emitted after the signal block.

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

Wait-cnt instructions never write SCC, so hoisting past them does not disturb
the SCC restore below.

### Emitted shape (separated anchors)

```asm
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<HASH>
    s_barrier_signal -3
  label_skipCBPreSignal_<HASH>:
    <optional SCC restore cmp>
    ...
    s_barrier_wait -3
    <wait-cnt drains hoisted below the cluster wait>
    s_barrier_signal -1
    s_barrier_wait -1
    tensor_load_to_lds ...
```

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

## Producer-side tensor drains (multicast On)

Two `s_wait_tensorcnt 0` drains, both emitted by
`insertProducerTensorDrainBefore` and both gated on
`streamKMulticast >= kStreamKMulticastOn`. Neither consults
`PrefetchGlobalRead`: the loop drain's earlier `PGR >= 2` restriction is gone, so
it now fires at every prefetch depth (which is how the PGR=1 grouping bug below
became reachable).

### Loop drain -- after each cooperative `tensor_load_to_lds` group

Emitted at the instruction following each cooperative load group. Groups are
discovered by the same scan Rule 3 uses (a `tensor_load_to_lds` with a preceding
workgroup `s_barrier_signal -1` in its segment, one drain per distinct trigger),
so the drains land exactly where Rule 3's handshake would have.

The cooperative load is asynchronous and lands in a **peer** workgroup's LDS, so
the consumer's own tensor counter cannot order it. Retiring it before the back
edge makes the broadcast coherent, and the next loop-head workgroup barrier then
publishes it for the consuming waves.

Emitted comment: `retire cooperative tensor_load_to_lds before back-edge`.

#### Group termination

`afterCooperativeTensorLoadGroup` returns the instruction after the **last**
tensor load of the group, not after the first. A cooperative group is the operand
load plus its MX-scale load (A + MXSA), and `ScheduleIterAlg=4` may schedule TDM
descriptor increments or `ds_load`s *between* the two.

The original walk advanced only over immediately-adjacent `tensor_load`s, so at
PGR=1 with SIA=4 it stopped at the first interleaved instruction and planted the
drain between A and MXSA -- a confirmed hang, which did not reproduce at SIA=0.

The walk therefore skips non-terminator instructions and stops only at an
`isCoopLoadGroupTerminator`: a segment boundary (label / branch / call), an
`s_xor_b32`, an `s_wait_tensorcnt`, or a barrier.

### Prologue drain -- before the LDS ping-pong `s_xor_b32`

The prologue TDM descriptor SGPRs are ping-ponged to the next LDS buffer by an
`s_xor_b32`. Hardware keeps reading a `tensor_load_to_lds` descriptor **after**
the instruction issues, so that XOR must not execute while the load is in flight.

The pass takes the function's first `tensor_load_to_lds`, finds the first
`s_xor_b32` after it in the same segment (`findXorInSegmentAfter`) and inserts a
drain before it -- unless one already sits between the two
(`hasWaitTensorCntBetween`) or immediately above the XOR
(`isImmediatelyPrecededByWaitTensorCnt`).

Emitted comment: `retire TDM before LDS ping-pong XOR`.

The same hazard is modelled generally, for every in-flight descriptor rather than
just the prologue one, by
[the tensor-descriptor WAR scan](../user/stinky-waitcnt-insertion-pass.md#tensor-descriptor-war)
in `StinkyWaitCntInsertionPass`. That pass is region-scoped to `loopWithPrefetch`
and `noLoadLoopBody`, so it does not see the prologue; this rule covers it. Where
the two do overlap, the guards above suppress the duplicate.

---

## Source files

- `src/transforms/asm/InsertClusterBarrierPass.cpp` -- implementation
- `include/stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp` -- public API
- `src/pipeline/backend/Gfx1250Backend.cpp` -- pipeline placement (kernel scope,
  after the region adaptor that runs `StinkyWaitCntInsertionPass`)
- `tests/unit/asm/InsertClusterBarrierPassTest.cpp` -- unit tests, including the
  multicast-mode suppression and both producer drains
