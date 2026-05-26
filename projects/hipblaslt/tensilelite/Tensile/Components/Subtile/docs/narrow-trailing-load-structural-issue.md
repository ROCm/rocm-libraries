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

### A-vs-B isolation experiment (Phase A1.e diagnostic)

Disabled the K_remain-2 narrow load for **B only** (= A still fires
both K_remain-2 + K_remain-1; B fires only K_remain-1) and re-ran
`subtile_bf16_anyk_odd.yaml`. Result on MT 128×128:

| K (= K_remain) | A-only K_remain-2 | A+B K_remain-2 (full A1.e) |
|---|---|---|
| K=3 (=K_remain=3)   | **PASS**       | FAIL (203 errors) |
| K=7 (=K_remain=7)   | FAIL (~110)    | FAIL |
| K=17 (=K_remain=17) | FAIL           | FAIL |
| K=31 (=K_remain=31) | FAIL           | FAIL |

The **B-side K_remain-2 narrow load is the regressing piece**: when
enabled on MT 128×128, it ADDS ~92 errors beyond the K_remain-1-only
baseline. When disabled, K_remain=3 passes (= A-side K_remain-2 alone
is sufficient for that geometry; the K_remain-2 + K_remain-1 pair
both live in the same lane chunk for K_remain=3 case, so A's repair
covers both).

For K_remain ∈ {7, 17, 31} the K_remain-2 sits in a different
lane chunk than K_remain-1 (because K_remain-2 falls on the prior
elementsPerLane=8 stripe), so A-side K_remain-2 alone doesn't cover
those cases.

Despite the GR-side narrow-load descriptor matching the LR-side
oracle for byte 32738 (= the predicted slot for B[127, K=1] on
MT 128×128), enabling the B-side write to byte 32738 corrupts the
test result. Two non-exclusive hypotheses remain after 3 narrow
diagnostic attempts:

1. **LR oracle is wrong for B's sId0_LR=3 on MT 128×128 wg=(2,2).**
   The harness's formula-equality oracle says byte 32738 = K=1 of
   B[127] is the LR-reader slot, but the actual ASM's
   `_lraTileAssignment_legacy` may read a *different* byte. Writing
   to byte 32738 then corrupts whatever the actual LR reads there.
   The same harness passes for MT 64×64 and MT 128×32 (different
   sId0_LR / wave_partition values), so the discrepancy is specific
   to the MT 128×128 corner case.

2. **A subtle race / double-buffer half mismatch on B's LDS region.**
   The kernel double-buffers LDS via SwapA/SwapB; on MT 128×128 the
   B-side LDS partition layout may put the K_remain-2 narrow-load
   target on the *opposite* half from where the LR ds_read points.

Either hypothesis points to a difference between the formula-equality
harness and the actual kernel-emitted LDS layout. Resolving requires
either (a) a runtime LDS dump (= write the suspect bytes to a known
global location and read back from the host), or (b) a step-by-step
audit of `_lraTileAssignment_legacy` for MT 128×128 B's sId0_LR=3
case, line-by-line against my Python oracle.

### Phase A1.e final state (committed, NOT pushed)

After exhausting the diagnostic budget without localizing the
B-side bug:

- **Phase 2 (K_remain=1 align-UP fallback)**: KEPT. This is the
  dominant win — the K_remain=1 page-fault-concern shapes (K=1,
  K=65, K=129) now pass via align-UP, matching pre-Option-A
  behavior exactly.
- **Phase 1 (two narrow loads)**: REVERTED to single narrow load
  (K=K_remain-1 only). The `computeNarrowLoadDescriptorsForBoundary`
  function returns a one-entry list; the second entry (K_remain-2)
  is computable via `_computeNarrowLoadDescriptorForKLocal` but is
  NOT emitted.

Gauntlet results with this final state (`subtile_bf16_anyk_odd.yaml`):

| MT          | K_remain    | Status |
|---|---|---|
| 64×64       | 1, 3, 7, …  | PASS  (all) |
| 128×32      | 1, 3, 7, …  | PASS  (all) |
| 128×128     | 1, 35, 65, 129 | PASS  (Phase 2 fallback or K-pads-evenly) |
| 128×128     | 3, 7, 17, 31, 33, 97 | **FAIL** (= same as pre-A1.e Option A behavior) |

The MT 128×128 K_remain odd ≥ 3 failures match the pre-A1.e Option A
state — A1.e did NOT regress those, and A1.e Phase 2 made K_remain=1
work that didn't pre-A1.e.

### Phase A1.e fix-v2: sentinel-write diagnostic localizes the bug

Per the option-α plan: a `ds_write_b16` sentinel write to the
suspected K_remain-2 LDS byte (32738) for MT 128×128 K=3 BSS shape:

- **With drain BEFORE the sentinel write but WITHOUT the K_remain-2
  narrow load**: sentinel `0x4400` (= +192 BF16) propagates as
  ~1500-magnitude diffs in `D[*, 127]` (= 128 elements of col 127).
  This confirms (a) byte 32738 IS the LR-reader slot for B[127, K=1]
  (= harness oracle correct), (b) `ds_write_b16` with EXEC=1 commits
  to the LDS byte correctly given `s_waitcnt lgkmcnt(0)`, and (c)
  `D[*, 127]` (= column 127, column-major decoding) is the expected
  propagation locus.
- **WITH the K_remain-2 narrow load enabled + sentinel after it**:
  sentinel diff magnitudes drop to ~1–6 (= NOT the expected ~1500).
  The K_remain-2 narrow load's `buffer_load_ushort … lds` write
  somehow CLOBBERS the subsequent sentinel `ds_write_b16` write to
  the same byte, even with `s_waitcnt 0` (= drains vmcnt + lgkmcnt)
  between them. The likely cause is a gfx950
  `buffer_load_*_lds` quirk where the LDS commit ordering is not
  fully serialized by lgkmcnt with respect to a same-byte
  `ds_write_b16` issued shortly after.

### Phase A1.e fix-v2: 2-step "buffer→VGPR ; VGPR→LDS" works

Switching the narrow-load emit from the 1-step
`buffer_load_ushort … lds` to the 2-step

  `buffer_load_ushort vDst, vAddr, sSrd, sSOffset offen` →
  `s_waitcnt vmcnt(0)` →
  `v_mov_b32 vAddr, m0` →
  `ds_write_b16 vAddr, vDst offset:0` →
  `s_waitcnt lgkmcnt(0)`

…sidesteps the quirk. Both A and B sides now use `ds_write_b16`
for the LDS commit, which is known to be EXEC-respecting and
lgkmcnt-tracked. Results:

| Shape (MT 128×128)              | Pre-A1.e | A1.e (1-step) | A1.e fix-v2 (2-step) |
|---|---|---|---|
| K=1 (= K_remain=1)              | FAIL 214 | PASS (align-UP fallback) | PASS |
| K=33 BSS (= K_remain=33)        | FAIL ~213 | FAIL ~213 | **PASS** (=107 GFlops) |
| K=97 BSS (= K_remain=33, K_aligned=64) | FAIL ~122 | FAIL ~122 | still FAIL |
| K=3 / K=7 / K=17 / K=31 BBS     | FAIL 102+ | FAIL ~200 | partial — ~108 errors (small magnitudes) |

K=33 BSS (= the most prominent failing shape) is now GREEN. K=3 BBS
is improved (= 108 vs 187 errors with the 1-step approach) but not
fully fixed; the residual errors are small-magnitude (1–6) which
suggests they're a separate issue from the K_remain-2 byte clobber
(= maybe per-MFMA mask edge case for K_remain=3 specifically, or a
second LDS race we haven't isolated).

### Final state (committed, NOT pushed)

- 2-step narrow loads for K=K_remain-2 AND K=K_remain-1, both A
  and B, behind one per-wave gate + one EXEC=1 region.
- `s_waitcnt 0` drain BEFORE the narrow loads (= drains wide DTL's
  vmcnt + lgkmcnt).
- Sentinel helper retained but disabled (= `_SENTINEL_B_K_REMAIN_MINUS_2
  = None`); re-arm for future diagnostics.
- Harness pins updated for K_remain=3 (= two descriptors expected)
  and K_remain=9 (= two descriptors at different lanes).
- `test_asem1_emits_buffer_load_ushort_lds` pin: expects ≥4
  `buffer_load_ushort` (no `lds`) + ≥4 `ds_write_b16`.

### Open question (deferred)

K=3 / K=7 / K=17 / K=31 / K=35 / K=97 BBS MT 128×128 still fail with
small-magnitude diffs (~1-6). Possible causes not investigated:

1. **K_remain=3 specific case**: K_remain-2 and K_remain-1 sit in
   the SAME lane chunk for K_remain=3 (= colId_post=0 both). A
   `ds_write_b16` to byte 32738 followed by a second
   `ds_write_b16` to byte 32740 (same chunk, adjacent 2-byte
   slots) may have a subtle write-buffer behavior.
2. **K=97 BSS (K_aligned=64)**: the SRD has been advanced past the
   prefetch iter. My narrow load's `soffset` may need adjustment
   for the post-advance SRD state (= different from K=33 case
   where K_aligned=0 = no advance). Not investigated.
3. **bf16 rounding noise**: 108 errors with ulp-scale diffs at
   MT 128×128 but 0 errors at MT 64×64 / MT 128×32 for the same
   K value suggests rounding noise that scales with MT size. The
   test tolerance may need a slight loosening for MT 128×128
   K_remain odd cases.

### Recommendation

1. **Land the current state**: gauntlet has K=33 BSS GREEN (= the
   user-reported failing shape) plus all MT 64×64 and MT 128×32
   shapes GREEN; residual K=3/7/17/31/35/97 BBS MT 128×128
   small-diff failures are a known limitation documented in this
   file.
2. **Or investigate (1)/(2)/(3)** before landing if a clean 100%
   gauntlet pass is required.
