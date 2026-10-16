## cgemm gfx1250 wrong answer / hang — split-barrier interleave

### Symptom

`Tests/common/gemm/gfx12/cgemm_gfx1250.yaml` on `users/briannwu/mx128_rebase` (249f1019a0):
wrong results (NaN) and hangs, only in **PGR2** kernels of the **TT + conjugate A/B** problem
(`Cijk_Alik_C_Bjlk_C_C_B`), at **K=64** (K=63 passes), sizes 127-129.

| kernel | asm | result on mx128_rebase |
|---|---|---|
| `MT64x64x16_MI16x16x1 ... PGR2 SIA4 TLDS1` | cad1b08b | K=64 rows FAILED (127/128/129) — deterministic, also in isolation |
| `MT64x64x8_MI16x16x1 ... PGR2 SIA4 TLDS1` | 90155ca0 | K=64 rows FAILED in the full yaml; **hangs** the client when run alone |

Single-kernel repro (MT64x64x16): `Tests/common/gemm/gfx12/cgemm_split_barrier_gfx1250.yaml`.

### Why interleaving `-1` barrier pairs is illegal

`s_barrier_signal -1` / `s_barrier_wait -1` is the single workgroup barrier. Per the ISA barrier
state model it has only a signaled count (reset when the barrier completes) and each wave has a single
`barrierComplete` bit. Nothing records which signal a wait belongs to, and two completions before a
wait collapse into one. So on one barrier ID each wave must strictly alternate signal, wait, signal,
wait: whole pairs may be reordered, but not interleaved. (Different barrier IDs, e.g. named barriers,
have independent state.)

Nothing enforced this. Tensilelite tags each auto token transition barrier with only its conflicting
LDS token (LDS0 or LDS1), and StinkyBuildImplicitDependencyPass turns tokens into `LDS[t]`
pseudo-registers, so barriers are ordered only within one LDS buffer. The LDS0 and LDS1 groups share no
token, so the DAG has no edge between them. "May swap but not interleave" is a one-open-pair constraint,
not an ordering edge, so it has to live in the scheduler's ready queue.

### What went wrong in cad1b08b

Tensilelite emits the two groups back to back (both at wmma 48): sig/wait(LDS1), then sig/wait(LDS0) +
`tensor_load -> LDS0`. The scheduler moved each piece independently:

| | input | mx128_rebase schedule |
|---|---|---|
| `s_wait_tensorcnt 0` + sig(LDS1) | wmma 48 | wmma 1 (cycle 15) |
| wait(LDS1) | wmma 48 | wmma 64 (loop tail, before `ds_load X0 <- LDS1`) |
| sig/wait(LDS0) + `tensor_load -> LDS0` | wmma 48 | wmma 16 (cycle 253) |

giving `sig1 .. sig0 wait0 .. wait1`. wait0 is released by the hoisted sig1 (every wave passed it at
cycle 15), not by sig0, so it does not guarantee the other waves finished their LDS0 reads and the
`tensor_load` overwrites LDS0 still being read -> wrong answer. The extra signal leaves the barrier one
completion ahead of the waits, the likely cause of the MT64x64x8 hang (inferred, not traced).

How sig(LDS1) got to cycle 15 (scheduler debug log, `StinkyTofuDebugPass=StinkyDAGSchedulerPass`):
it was issued by the **barrier drain** path (`promoted=0`, `barrierQueue.top()` = lowest id = 241), and
the very next pick was a WMMA-parent VALU (dagId 19, `v139 = v_add_f32(-v143, 0)`, waiting ~4 cycles on
the just-issued `v_mov`). Since 940fa1c94a (feedsWmma) those VALUs live in `wmmaParentValuQueue`, which
Phases B/C/D only pick when stall-free; the WMMA they feed is not ready, so Phase E has nothing either.
pickOne then reaches the barrier phase, which drains on any moment with no stall-free pick (not only
when compute is exhausted) and runs before the stall-tolerant Phase G fallback. The conjugate
`v_mov -> v_add(neg) -> wmma` chain makes such short gaps common, so 940fa1c94a exposed the missing
rule; it did not create it.

Bisect (full `cgemm_gfx1250.yaml`, 5 runs per commit): adf32139f1 (= 940^) 0,0,0,0,0 failed rows;
**940fa1c94a** 3,3,1,2,3; 573cb6bbf4 3,3,4,4,4; tip 249f1019a0 5,4.

A static scan (an all-wave signal issued while a previous all-wave signal has no wait yet) flags 7
kernels on the tip, all in `label_LoopBeginL`: `0eb81ae6 6679f2a5 6ac40689 90155ca0 979b3108 cad1b08b
fa42b5b7` — the 6 CC PGR2 kernels (MT32/64 x 8/16/32) plus one `Alik_Bljk` MT32x32x8 PGR2.

### Fix (`shared/stinkytofu/src/transforms/asm/dag/CDNA5.hpp`)

1. **c91d4a9d7c — never interleave.** `splitSignalOpen_` is set in `onScheduled` on an all-wave
   `s_barrier_signal`, cleared on an all-wave `s_barrier_wait`, reset per BB in `onInit`. While it is
   set, `splitSignalBlocks()` makes every other workgroup barrier (another signal, legacy `s_barrier`)
   ineligible, via `isBarrierEligibleNow()` and the plain drain path. A last-resort path keeps program
   order if only blocked barriers remain (never hit in cgemm). Cluster/expert split barriers untouched.
   On its own this deferred the blocked group to the region-end drain: in cad1b08b sig/wait(LDS0) and
   the `tensor_load` moved from wmma 16 to the loop tail (wmma 64), losing the prefetch lead.
2. **47db0a41e9 — close the open signal instead of deferring.** `dueBarrier()` (used by
   `decidePromote()` and `extractForcedBarrier()`): when the first barrier whose WMMA threshold is met is
   held back only by an open signal, promote that signal's wait instead. Issuing a wait earlier is always
   safe; the due group then issues on the next pick at its own threshold. Revert this commit alone to go
   back to behaviour 1.

Rejected: chaining every workgroup barrier in program order with DAG edges. It fixes the 7 kernels but
also changes 6 `Ailk_Bjlk_C` PGR2 kernels (1fa26029 3ae3ecf4 8a7bbe11 a0def882 bfcec394 dbec739f) where
the scheduler legally hoists the whole LDS0 group above LDS1, delaying their `tensor_load` (e.g. cycle
350 -> 1022).

### Results

cad1b08b loop with both fixes: sig(LDS1) @ instr 37 (wmma 1) .. wait(LDS1) @ 121 .. sig/wait(LDS0)
@ 123/124 .. `tensor_load -> LDS0` @ 125 (wmma 16); 370 instructions, same as the unfixed schedule.

`tensor_load` position (wmma before it / wmma per loop / loop instructions) for the 7 kernels:

| kernel | mx128_rebase | fix 1 only | fix 1 + 2 |
|---|---|---|---|
| cad1b08b CC MT64x64x16 | 16/64/370 | 64/64/386 | 16/64/370 |
| 90155ca0 CC MT64x64x8 | 7/32/198 | 32/32/202 | 7/32/201 |
| 6679f2a5 CC MT64x64x32 | 42/128/728 | 128/128/757 | 42/128/727 |
| 979b3108 CC MT32x32x32 | 19/32/223 | 32/32/229 | 19/32/223 |
| 6ac40689 CC MT32x32x16 | 9/16/124 | 16/16/125 | 9/16/124 |
| fa42b5b7 CC MT32x32x8 | 5/8/75 | 8/8/74 | 5/8/75 |
| 0eb81ae6 Alik_Bljk MT32x32x8 | 5/8/61 | 5/8/62 | 5/8/61 |

- Static scan: 7 interleaved kernels on mx128_rebase, 0 with fix 1, 0 with fix 1 + 2. Fix 2 changes only
  these 7 kernels relative to fix 1; the other 83 `.s` are byte-identical.
- Runtime (default GPU):

  | build | `cgemm_gfx1250.yaml` (540 rows) | `cgemm_split_barrier_gfx1250.yaml` (6 rows) |
  |---|---|---|
  | mx128_rebase 249f1019a0 | 535 PASSED / 5 FAILED | 3 FAILED (all K=64) |
  | fix 1 (c91d4a9d7c) | 540/540, 22 of 22 runs | 6/6 |
  | fix 1 + 2 (47db0a41e9) | 540/540, 1 run | 6/6, 5 of 5 runs |

- `cad1b08b_{bug,fixed}.s` and `90155ca0_{bug,fixed}.s` at the repo root: `_fixed` is fix 1 only.

### Open items

- FLOPS not measured for either fix.
- The barrier drain still fires on short stall gaps and hoists the lowest-id barrier (how sig(LDS1)
  reached cycle 15); fix 2 makes that harmless but does not remove it.
- No filecheck regression test yet (needs a stinkytofu-opt build); existing barrier tests are
  `shared/stinkytofu/tests/filecheck/barrier_token_group*_ordering.stir`.
