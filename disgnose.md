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

### Trigger (bisect)

Full `cgemm_gfx1250.yaml`, GPU execute+validate, 5 runs per commit:

| commit | what | failed rows per run |
|---|---|---|
| adf32139f1 (= 940^) | before feedsWmma | 0,0,0,0,0 |
| **940fa1c94a** | feedsWmma: schedule WMMA-parent VALU with WMMA in CDNA5 Phase B | 3,3,1,2,3 |
| 573cb6bbf4 | tip minus 7573cc64 (unbound ds_loads ruled out) | 3,3,4,4,4 |
| tip 249f1019a0 | — | 5,4 |

940fa1c94a only changes pick timing in the scheduler; the bug it exposes is older (below).

### Root cause

`s_barrier_signal -1` / `s_barrier_wait -1` is the single workgroup barrier: hardware counts
arrivals, it has no notion of which LDS buffer a signal is for. The scheduler orders barriers only
through LDS MemTokenData tokens (StinkyBuildImplicitDependencyPass). The LDS0 and LDS1 sync groups
share no token, so the DAG has no edge between them: the ready queue may swap them as whole groups
(legal) **or interleave them** (illegal).

cad1b08b main loop, after 940fa1c94a:

```
  ds_load X1 (part) <- LDS0
  s_wait_tensorcnt 0
  s_barrier_signal -1        <- sig(LDS1) hoisted (cycle 15)
  ds_load X1 rest, X2, X3 <- LDS0
  s_wait_dscnt 0
  s_barrier_signal -1        <- second signal with no wait in between
  s_barrier_wait   -1        <- "wait(LDS0)"
  tensor_load -> LDS0
  ...wmma...
  s_barrier_wait   -1        <- wait for sig(LDS1)
  ds_load X0 <- LDS1
```

Each wave signals twice before its first wait. That wait is released once every wave reaches the
hoisted sig(LDS1), i.e. before slow waves finish their LDS0 reads, so the following `tensor_load`
overwrites LDS0 while it is still being read -> wrong answer. The extra signal also leaves the barrier
phase one ahead of the waits, the likely cause of the MT64x64x8 hang (inferred, not verified).

A static scan of the generated asm (an all-wave signal issued while a previous all-wave signal has no
wait yet) flags 7 kernels on the tip, all in `label_LoopBeginL`:
`0eb81ae6 6679f2a5 6ac40689 90155ca0 979b3108 cad1b08b fa42b5b7` — the 6 CC PGR2 kernels
(MT32/64 x 8/16/32) plus one `Alik_Bljk` MT32x32x8 PGR2 that has not been seen failing.

### Fix (c91d4a9d7c, `shared/stinkytofu/src/transforms/asm/dag/CDNA5.hpp`)

Invariant: whole-group swaps are allowed, interleaving is not — all-wave signal and wait must
strictly alternate in the **scheduled** order (not program order).

- `splitSignalOpen_`: set in `onScheduled` on an all-wave `s_barrier_signal`, cleared on an all-wave
  `s_barrier_wait`, reset per BB in `onInit`.
- `splitSignalBlocks(node)`: while a signal is open, any workgroup barrier other than an all-wave wait
  (another signal, legacy `s_barrier`) is ineligible. Folded into `isBarrierEligibleNow()`, so it
  covers decidePromote, `extractForcedBarrier` and the clusterBarrier drain path.
- Plain drain path (`barrierQueue.top()`, which ignored eligibility with clusterBarrier off) now picks
  the lowest-id barrier not blocked by an open signal.
- Last resort before the "all buckets empty" assert: if only blocked barriers remain (input already
  interleaved), issue `top()` as before with a PASS_DEBUG note. Never hit in cgemm.
- Cluster/expert (non-all-wave) split barriers are untouched. Early split signals stay legal: other
  instructions may still sit between a signal and its wait.

Rejected alternative: chain every workgroup barrier in program order with DAG edges. It also fixes
the 7 kernels but changes 6 more (`Ailk_Bjlk_C` PGR2: 1fa26029 3ae3ecf4 8a7bbe11 a0def882 bfcec394
dbec739f), where Tensile emits the LDS1 group first and the scheduler legally hoists the whole LDS0
group + tensor_load above it; program order delays that tensor_load (e.g. cycle 350 -> 1022) and
loses prefetch lead.

### Results

- Asm: the fix changes exactly the 7 flagged kernels (scan 7 -> 0); the other 83 `.s` are
  byte-identical, including the 6 legal swaps. Modelled loop length and wmma count (272) unchanged.
  Fixed cad1b08b keeps the early sig(LDS1); its wait comes before sig/wait(LDS0) + `tensor_load -> LDS0`.
- Runtime, `cgemm_gfx1250.yaml` (540 rows), default GPU:

  | branch | runs | result |
  |---|---|---|
  | mx128_rebase 249f1019a0 | 1 | 535 PASSED / 5 FAILED |
  | fix c91d4a9d7c | 22 | 540/540 PASSED every run |

- Runtime, `cgemm_split_barrier_gfx1250.yaml` (6 rows): mx128_rebase 3 FAILED (all K=64),
  fix 6/6 PASSED.
- Broad regression `-k 'gfx1250 or stinky_sia4'` with the fix: 42 yamls run (stopped early), 0 FAIL.

### Open items

- Perf: fixed cad1b08b moves the LDS0 group + tensor_load to the loop end, whereas 940^ issued
  sig/wait(LDS0) + tensor_load early and sig/wait(LDS1) late. Needs a FLOPS comparison.
- No filecheck regression test yet (needs a stinkytofu-opt build); existing barrier tests are
  `shared/stinkytofu/tests/filecheck/barrier_token_group*_ordering.stir`.
- Which pick path chose the early sig(LDS1) on the tip was not PASS_DEBUG-verified; the fix guards
  all paths.
