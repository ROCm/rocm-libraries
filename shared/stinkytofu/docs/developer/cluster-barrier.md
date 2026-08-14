# Insert Cluster Barrier Pass

`createInsertClusterBarrierPass` inserts cluster-barrier handshakes at four rules
covering the main and tail loops.

The pass is created via:

```cpp
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass();
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
of the whole kernel.

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

### Emitted shape (separated anchors)

```asm
    s_cmp_eq_u32 s[sgprWaveIdx], 0
    s_cbranch_scc0 label_skipCBPreSignal_<HASH>
    s_barrier_signal -3
  label_skipCBPreSignal_<HASH>:
    <optional SCC restore cmp>
    ...
    s_barrier_wait -3
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

## Source files

- `src/transforms/asm/InsertClusterBarrierPass.cpp` -- implementation
- `include/stinkytofu/transforms/asm/InsertClusterBarrierPass.hpp` -- public API
