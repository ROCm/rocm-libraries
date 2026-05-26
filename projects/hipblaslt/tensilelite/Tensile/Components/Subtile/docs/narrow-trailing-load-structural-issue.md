# Narrow trailing-load: structural issue with align-DOWN + DWORD-aligned bound check

## TL;DR

The Option A structural fix (align-DOWN tail SRD + lane-targeted narrow
`buffer_load_ushort … lds`) does NOT close the bf16/fp16 odd-K
page-fault concern for the bf16-anyk-odd gauntlet, because align-DOWN's
per-DWORD buffer-OOB clip drops **more than one element** when
`(K_remain * bpe) % bpr != 0`. The single-BF16 narrow load fixes only
one element per A/B/wave; the remaining clipped elements stay zero.

This is the root cause of the K=1 / K=3 / K=97 BSS failures on
`Cijk_Alik_Bljk_BSS_BH_UserArgs_MT128x128x64` in the
`subtile_bf16_anyk_odd.yaml` gauntlet on branch
`users/bnemanich/subtile-narrow-trailing-load` at tip `f36c34754f9`.

## Mechanism (worked example: MT 128×128 BF16 ASEM=1 K=3)

### Per-row data layout

For BF16 with K=3 (stride=K=3, bpe=2, row stride in bytes = 6):
- Row m's K=0..2 bytes occupy global bytes `[m*6, m*6+6)`.
- Row m's data is NOT DWORD-aligned: row 127 K=0 starts at byte 762
  (which is not a multiple of bpr=4).

### Align-DOWN tighten

For K_remain=3 K_aligned=0:
- `alignedBytes_dn = roundDown(K_remain*bpe, bpr) = roundDown(6, 4) = 4`
- `delta = DepthU*bpe - alignedBytes = 128 - 4 = 124`
- `NumRecords_after = NumRecords_before - delta = 890 - 124 = 766`

### Per-DWORD buffer-OOB clip on lane 62 of wave 3 (the "boundary" lane)

Lane 62 of wave 3 sId0=3 GR-reads bytes `[762, 778)` from global
(row 127's K=0..7 data, with K=3..7 past the matrix).

DWORD-aligned bound checks (DWORD-end > NumRecords ⇒ DWORD zeroed):

| DWORD range | Lane bytes | Global K-element content | Bound check (end ≤ NumRecords=766) | Result |
|---|---|---|---|---|
| `[760, 764)` | bytes 0..1 | row 127 K=0 (bytes 762..763) | 764 ≤ 766 ✓ | **LOADED** |
| `[764, 768)` | bytes 2..5 | row 127 K=1 (764..765) + row 127 K=2 (766..767) | 768 > 766 ✗ | **ZEROED** |
| `[768, 772)` | bytes 6..9 | past-K (row 128 in stride-3 = matrix-OOB) | OOR | zeroed |
| ... | ... | ... | ... | zeroed |

So **K=1 AND K=2 of row 127 are clipped** (BOTH in the same OOR
DWORD), even though K=1 is a fully valid in-matrix element.

The narrow load only fixes ONE BF16 element (= K=K_remain-1 = K=2 of
row 127). K=1 of row 127 stays zero.

### The actual gauntlet failure pattern matches

For K=3, the gauntlet shows ~102/16384 incorrect values, all in
**column N=127** of D, with diffs of magnitude consistent with one
A[m, K]·B[127, K] term missing. The narrow load fixed B[127, K=2]
(the K=K_remain-1 element), but B[127, K=1] is still zero → D[m, 127]
is off by A[m, 1] · B[127, 1] for each m.

## Generalization to other K_remain values

For BF16, the number of K-elements per row that align-DOWN drops
depends on the relationship between `K_remain*bpe`, `bpr=4`, and the
row stride `stride*bpe`:

| K_remain | K_remain*bpe % bpr | stride*bpe % bpr | Dropped per row | Narrow load fixes |
|---|---|---|---|---|
| 1 (e.g. K=1, 65, 129) | 2 | (stride odd) ≠ 0 | K=0 of row M-1 **AND row M-2** (cross-row DWORD) | only row M-1 K=0 |
| 1 (e.g. K=2, K_remain=0) — N/A | — | — | (no tail loop) | N/A |
| 3 (K=3, 67, 131) | 2 | depends | K=1 AND K=2 of row M-1 (2 elts, same DWORD) | only K=2 |
| 5 (K=5, 69, 133) | 2 | depends | K=3 AND K=4 of row M-1 | only K=4 |
| 7 (K=7, 71, 135) | 2 | depends | K=5 AND K=6 of row M-1 | only K=6 |
| 9, 11, 13, ... (K_remain odd > 7) | 2 | depends | depends on stride+lane voff | only K=K_remain-1 |

For ANY K_remain where `K_remain*bpe % bpr != 0` (= all odd K_remain
for bf16), align-DOWN drops AT LEAST the K=K_remain-2 AND K=K_remain-1
elements of row M-1 (which fall in the same OOR DWORD), and
potentially MORE rows if `stride*bpe` is not bpr-aligned.

## Why the cross-check harness missed this

`computeNarrowLoadDescriptor` derives a single `m0_target` per
operand for ONE byte (the K=K_remain-1 BF16). The harness asserts
`GR.m0_target == LR.lds_byte_target` for that single byte. The
narrow load writes to that byte. But the harness DOES NOT model the
per-DWORD bound check that drops additional bytes in the same
DWORD. It also doesn't model the multi-row drop that happens when
`stride*bpe < bpr` (K_remain=1 cases).

The harness as designed is a "formula equality" oracle, not a
simulation oracle. The bug is in the design's assumption that "only
one BF16 per row is clipped" — that assumption holds only when
`stride*bpe % bpr == 0`, which is FALSE for all bf16 odd-K cases
(the entire bf16-anyk-odd gauntlet target).

## Implications for Option A

Option A as designed (align-DOWN + single-element narrow load) cannot
close the bf16 odd-K page-fault concern. The mechanism is structurally
mismatched to the per-DWORD bound-check semantics of `buffer_load_*`
on gfx9/gfx950.

### Possible variants

1. **A1.b — 4-byte narrow load**: Use `buffer_load_dword` (or
   `BufferLoadB32`) instead of `buffer_load_ushort`. Fixes the 2-BF16
   case (K_remain=3, 5, 7, ..., where the dropped DWORD contains
   K_remain-2 AND K_remain-1 of row M-1). Does NOT fix the K_remain=1
   case (where multiple ROWS' K=0 are dropped). Closes maybe 80% of
   the failing shapes.

2. **A1.c — per-row narrow loop**: Emit one narrow load per affected
   row × affected K-element. For K_remain=1, that's potentially
   `(DepthU - K_remain_floor) / stride` rows (could be many). Way out
   of the original "single narrow load" budget.

3. **A1.d — exempt K_remain*bpe < bpr from align-DOWN**: Static gate
   align-DOWN on `ASEM*bpe % bpr == 0 OR ASEM*bpe >= bpr`. For
   ASEM=1 bf16 (= the failing gauntlet), this disables align-DOWN
   entirely → page-fault concern remains for those shapes. Effectively
   reverts to Option B from the prior worker's analysis.

4. **Abandon Option A**: Keep align-UP everywhere. Page-fault concern
   stays as-is (matches the pre-PR behavior). The reviewer's audit
   note can be replied with the ASEM K-padding contract argument.

## Branch state

- Tip: `f36c34754f9` (3 commits above `ac9884c2575` post-cherry-pick).
- Nothing pushed.
- Unit suite: 346 passed + 42 skipped + 1 xfailed (cross-check
  harness pins the descriptor against a formula-equality oracle,
  which doesn't catch this structural mismatch).
- YAML gauntlet: `subtile_bf16_anyk_odd.yaml` K=1 / K=3 / K=97 BSS
  shapes FAIL with align-DOWN + single-element narrow load. K=1: 214
  incorrect / 16384. K=3: 102 incorrect / 16384.

## Phase A1.e — two narrow loads + runtime K_remain=1 fallback

### What changed

1. `computeNarrowLoadDescriptorsForBoundary(kernel, ti, K_remain, tc)`
   returns the list of narrow-load descriptors needed to repair the
   align-DOWN-clipped DWORD: two entries for K_remain odd >= 2 (one
   for K_remain-2, one for K_remain-1), zero entries for K_remain=1
   (multi-row clip, unsafe).
2. `_emitNarrowLoadForOperand` now emits TWO `buffer_load_ushort …
   lds` per operand under one per-wave gate + EXEC=1 region.
3. `emitTailSrdTightenSubtile` adds a runtime `K_remain*bpe >= bpr`
   check (via `s_cmp_eq + s_cselect`): when the align-DOWN aligned
   bytes is 0 (= K_remain=1 for bf16), bump to `bpr` so the SRD
   behaves like the legacy align-UP. The narrow-load emit gates the
   same way (= skips when K_remain*bpe < bpr).
4. Cross-check harness extended (`computeLRReadersForBoundary` +
   list-aware `_cross_check_one`) to validate BOTH narrow-load slots
   per operand against the LR-reader oracle.

### Phase A1.e gauntlet results — `subtile_bf16_anyk_odd.yaml`

Tested with two-narrow-loads + K_remain=1 align-UP fallback:

| MT          | K (= K_remain) | Status | Errors / 16384 |
|---|---|---|---|
| 64×64       | 1, 3, 7, 17, 31 | PASS | 0 |
| 128×32      | 1, 3, 7, 17, 31 | PASS | 0 |
| 128×128     | 1               | PASS | 0 (align-UP fallback) |
| 128×128     | 3               | FAIL | 203 |
| 128×128     | 7               | FAIL | ~190 |
| 128×128     | 17              | FAIL | ~150 |
| 128×128     | 31              | FAIL | ~100 |
| 128×128     | 33 (BSS)        | FAIL | 213 (BSS) |
| 128×128     | 97 (BSS)        | FAIL | 122 (BSS) |

A/B isolation (single K_remain-1 narrow load only, no K_remain-2):

- MT 128×128 K=3: 111 errors (vs 203 with both loads). The K_remain-2
  narrow load ADDS ~92 errors instead of fixing any.

No narrow load at all (align-DOWN baseline):

- MT 128×128 K=3: 183 errors. Adding K_remain-1 narrow load: 111
  (= -72). Adding K_remain-2 narrow load: 203 (= +92 over K_remain-1).

### What's wrong (open question)

The K_remain-2 narrow load is structurally correct on paper:

- `computeNarrowLoadDescriptorsForBoundary` returns descriptors whose
  `m0_target` matches the LR-reader oracle's `lds_byte_target` for
  K_local=K_remain-2 — the harness asserts this for all
  `subtile_bf16_anyk_odd.yaml` shapes.
- The asm emits the load with the right `vaddr=0 / soffset =
  m_last*stride*bpe + K_local*bpe / m0 = LDS byte for lane_target's
  chunk slot for K=K_remain-2`.
- For MT 64×64 and MT 128×32 the two-load repair works (= the
  pre-Option-A page-fault gauntlet matches).

But MT 128×128 with both narrow loads gives MORE errors than with
the single narrow load. The K_remain-2 narrow load is overwriting
a byte that — for MT 128×128 specifically — already has the right
value (or the LR consumer reads from a different byte than the
oracle predicts).

Hypotheses still to investigate:
1. The LR oracle's `consumer_wave` / `wave_partition` derivation is
   off for MT 128×128 wg=(2,2) when sId0_LR has a non-trivial value
   (3 in this case). The harness's formula matches the GR-side
   m0 by construction, but neither side may match the actual ASM's
   `_lraTileAssignment_legacy` for sId0=3.
2. The wide DTL for MT 128×128's larger LDS layout might be writing
   the K_remain-2 byte from a different lane (= not lane 57/62 with
   colId_post=0 as the harness assumes) due to a subtile-row-walk
   offset the harness doesn't model.
3. The narrow load's `m0` computation in asm shares an SGPR scratch
   with the K_remain-1 load below it; an aliasing / dependence bug
   could be writing the wrong K_local value.

### Next steps

- Either: dump the actual LDS bytes via a CPU-readable scratch
  buffer at the end of the tail loop (= add a temporary debug
  store) to confirm WHICH byte is wrong.
- Or: hard-code a known constant into the K_remain-2 narrow load
  (= write 0xDEAD to the target byte) and inspect D's error
  pattern to confirm the byte IS the one the LR consumer reads.
- Or: revert the K_remain-2 emit and ship only Phase 2
  (= K_remain=1 align-UP fallback + single K_remain-1 narrow load
  for the K_remain odd >= 3 partial repair). This restores
  MT 128×128 K=1 (the dominant failing shape) and accepts the
  remaining MT 128×128 K=3 etc. failures as documented limitation.
