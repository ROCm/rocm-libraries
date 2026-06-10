# ds_read vs pack ORDERING TABLE — cms=0 / cms=1 / SHADOW

Kernel: **BPG#11 TF32 4x4 TN canonical** (`CANONICAL_KERNEL_CONFIG`,
`Cijk_Alik_Bljk_S_MX_B ... MT128x128x32_MI16x16x1 ...`).

Subject of interest: the **A-side `ValuA_X0_I0+12..15`** rotating-buffer fragment —
the `ds_read` that loads it and the `v_mfma_f32_4x4x4_16b_bf16` (+ surrounding
`v_cvt_pk_bf16_f32`) that finalizes/packs it, **within each loop body**.

All artifacts live under
`.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite/`:
- `hxcx_artifacts/cms0_kernel.s` — real cms=0 default-codegen assembly (raw only, no capture).
- `hxcx_artifacts/cms0_bodies.txt` — cms=0 body label markers.
- `Tensile/Components/h7lo_uvrl_artifacts/kernel.s` — cms=1 emitted assembly (MAINLOOP macro form).
- `Tensile/Components/h7lo_uvrl_artifacts/cms_capture_listing.txt` — cms=1 CAPTURE (body-segmented).
- `Tensile/Components/h7lo_uvrl_artifacts/shadow_capture_listing.txt` — SHADOW capture (only view).

> Capture column format: `<file_line> | body | body_idx | mfma_index | seq | render`.
> "body_idx" is the position **within that body** — the comparison axis for ordering.

---

## THE DEFINITIVE 3×3 TABLE

Rows = schedule; Cols = body. Cell = order of the **A-side X0_I0+12 ds_read** vs its
**pack-mfma**, *within that body*, with citations.

| schedule | ML (main unrolled loop) | NGL (no-global-load loop) | NLL (no-load / tail) |
|----------|-------------------------|---------------------------|----------------------|
| **cms=0** | **N/A — T0→X0 scheme** (X0+12 is a *pack destination*, not a ds_read target; T0+8/12 ds_reads `cms0_kernel.s:1900,1906` & `:2093,2096` precede cvt-X0+12 `:2163`) | **N/A — T0→X0 scheme** (T0+8/12 ds_reads `cms0_kernel.s:2228,2234` & `:2407,2410` precede cvt-X0+12 `:2480`; 0 global loads in 2205-2511) | **ds_read BEFORE pack** (in-place X0 scheme; ds_read `cms0_kernel.s:3302` < pack-mfma `:3643`, pack-cvt `:3640`) |
| **cms=1** | **ds_read BEFORE pack** (capture `cms_capture_listing.txt:402`, body_idx **155** < pack-mfma `:424`, body_idx **177**; raw `kernel.s:2175 < :2208`) | **ds_read BEFORE pack** (capture `:584`, body_idx **141** < pack-mfma `:604`, body_idx **161**; raw `kernel.s:2175 < :2208`, usePLR=1 path) | **N/A — no in-body ds_read** (no offset:192 read; pack-mfma `:752`, body_idx 133, consumes X0+12 loaded in prior NGL body) |
| **SHADOW** | **ds_read BEFORE pack** (capture `shadow_capture_listing.txt:388`, body_idx **143** < pack-mfma `:410`, body_idx **165**) | **ds_read AFTER pack** ⚠️ (capture `:610`, body_idx **171** > pack-mfma `:477`, body_idx **38**) | **N/A — no in-body ds_read** (no offset:192 read; pack-mfma `:651`, body_idx 32, consumes X0+12 loaded in prior NGL body) |

**The one flipped cell: SHADOW / NGL.** Everywhere else cms=1 and SHADOW agree.

---

## §1 Method + the scheme caveat

### What "the pack" and "the ds_read" mean per scheme

There are **two distinct pack schemes** in play, and the before/after frame only
makes sense under the in-place scheme:

1. **In-place `X0_I0` scheme** (cms=1 ML/NGL, SHADOW ML/NGL, cms=0 **tail only**):
   `doFullPackCodePrefetch` OFF. The `ds_read_b128 ... v[vgprValuA_X0_I0+12:+3]
   ... offset:192` loads **directly into** the same X0+12 register that the
   `v_mfma_f32_4x4x4_16b_bf16 v[X0_I0+12:+3], v[74:75], v[X0_I0+10:+1],
   v[X0_I0+12:+3]` then reads and overwrites (in place). So "ds_read of X0+12 vs
   the pack of X0+12" is a well-posed question: same register, two ops, one order.

2. **`T0_I0 → X0_I0` scheme** (cms=0 ML/NGL): `doFullPackCodePrefetch` ON. The
   `ds_read` loads `ValuA_T0_I0+8..15`. A `v_cvt_pk_bf16_f32 v[X0_I0+12],
   v[T0_I0+8], v[T0_I0+9]` then **writes X0+12 from T0**. Here X0+12 is *never* a
   ds_read destination in ML/NGL — it is a pack *output*. The load-target (T0) and
   the pack-dest (X0) are **different registers**, so "is the ds_read of X0+12
   before or after the pack" is the **wrong frame** — answer: **N/A** (see §4 Q4).

### Body segmentation

- **cms=0** (raw `.s` only): segmented by `cms0_bodies.txt` label markers.
  ML = `label_LoopBeginL`(1872)..`label_LoopEndL`(2200). The **true NGL** (No-Global-Load
  loop, "Ord. NoGlobalLoadLoop_1") begins right after `label_LoopEndL` (~2205) and runs
  to `label_toPGR1`(2511) — verified **0** `buffer_load`/`global_load` in 2205-2511.
  NLL/tail = `label_TailLoopBeginL`(3296)..`label_TailLoopEndL`(3766), also 0 global loads.
  (The `label_toPGR1end_OrdNLL` region 3122-3296 still has 8 global loads — it is a PGR1
  ordinary path, **not** the NGL.)
- **cms=1** & **SHADOW**: segmented by the capture listings' `### body PRO/ML_PREV/ML/NGL/NLL`
  markers. ML_PREV(iter0) and ML(iter1) are two unrolled iterations of the main loop and
  are identical in ordering for both; this table reports the ML column from the ML body
  (and notes ML_PREV agrees).

---

## §2 Per-cell evidence (9 cells)

### cms=0 / ML — N/A (T0→X0)
- T0+8 ds_read `cms0_kernel.s:1900` (rel 29 of 1872), T0+12 ds_read `:1906` (rel 35);
  second-half-iter T0+8 `:2093`, T0+12 `:2096`.
- pack-cvt that writes X0+12 from T0+8: `cms0_kernel.s:2163` (= `cms0_bodies.txt:36`).
- No `ds_read ... v[vgprValuA_X0_I0+12:` anywhere in ML. Whole-kernel grep finds the
  **only** X0+12 ds_read at line **3302** (the tail). → frame N/A.

### cms=0 / NGL — N/A (T0→X0)
- Region 2205-2511, **0 global loads** (confirmed by count).
- T0+8 ds_read `cms0_kernel.s:2228` & `:2407`; T0+12 ds_read `:2234` & `:2410`.
- pack-cvt writes X0+12 from T0+8 at `:2480` (= `cms0_bodies.txt:40`).
- No X0+12 ds_read in region. → frame N/A.

### cms=0 / NLL (TailLoop) — ds_read BEFORE pack
- ds_read X0+12 (offset:192): `cms0_kernel.s:3302` (= `cms0_bodies.txt:41`).
- pack-cvt staging X0+10/11 from X0+12..15: `:3640`,`:3641`.
- pack-mfma X0+12 (in-place): `:3643` (= `cms0_bodies.txt:44`).
- Order: **3302 < 3643 → BEFORE.** This is the in-place scheme (tail loop only).

### cms=1 / ML — ds_read BEFORE pack
- Capture: ds_read `cms_capture_listing.txt:402`, body_idx **155**;
  pack-mfma `:424`, body_idx **177**. 155 < 177 → BEFORE.
  (ML_PREV agrees: ds_read `:206` idx 155, pack-mfma `:228` idx 177.)
- Raw kernel.s (MAINLOOP macro body): ds_read `kernel.s:2175` (guarded `.if usePLR==1`),
  pack-mfma (unguarded) `kernel.s:2208`. 2175 < 2208 → BEFORE.
  ML instance = `MAINLOOP 0` (`kernel.s:2239`), defaults useGR=1/usePLR=1.

### cms=1 / NGL — ds_read BEFORE pack
- Capture: ds_read `cms_capture_listing.txt:584`, body_idx **141**;
  pack-mfma `:604`, body_idx **161**. 141 < 161 → BEFORE.
- Raw kernel.s: NGL instance = `MAINLOOP 0, 0, 1, 1, 0` (`kernel.s:2253`, comment
  `useGR=0, usePLR=1`). Since usePLR=1, the same macro X0+12 ds_read at `:2175` is
  emitted, still before pack-mfma `:2208` → BEFORE.

### cms=1 / NLL — N/A (no in-body ds_read)
- NLL instance = `MAINLOOP 0, 0, 0, 0, 0` (`kernel.s:2285`/`:2484`, comment `usePLR=0`).
  usePLR=0 ⇒ the `.if usePLR==1` X0+12 ds_read at `:2175` is **omitted**.
- Capture NLL: **no** `ds_read ... offset:192`. The 8 NLL ds_reads load T0+8/12 and
  X0+20/28 (offsets 256/320/384/448), never X0+12. pack-mfma X0+12 at
  `cms_capture_listing.txt:752` (body_idx 133) consumes the X0+12 loaded in the
  prior (NGL) body. → frame N/A (consumes an earlier-body load).

### SHADOW / ML — ds_read BEFORE pack
- Capture: ds_read `shadow_capture_listing.txt:388`, body_idx **143**;
  pack-mfma `:410`, body_idx **165**. 143 < 165 → BEFORE.
  (ML_PREV agrees: ds_read `:194` idx 143, pack-mfma `:216` idx 165.)

### SHADOW / NGL — ds_read AFTER pack ⚠️ (THE FLIP)
- Capture: ds_read X0+12 (offset:192) `shadow_capture_listing.txt:610`, body_idx **171**;
  pack-mfma X0+12 `:477`, body_idx **38**. 171 > 38 → **AFTER**.
- The pack-mfma is near the *top* of the NGL body (idx 38) while the local-read of the
  fragment it should feed is pushed to the *bottom* (idx 171). The schedule lifted the
  pack early and deferred the ds_read.

### SHADOW / NLL — N/A (no in-body ds_read)
- Capture NLL: **no** `ds_read ... offset:192`. The 8 NLL ds_reads load
  T0+8/12 and X0+20/28 (offsets 256/320/384/448). pack-mfma X0+12 at
  `shadow_capture_listing.txt:651` (body_idx 32) consumes a prior-body load. → frame N/A.

---

## §3 cms=1 capture-vs-kernel.s faithfulness cross-check

| body | capture order | raw kernel.s order | agree? |
|------|---------------|--------------------|--------|
| ML   | ds_read idx 155 < pack-mfma idx 177 → BEFORE | ds_read `:2175` < pack-mfma `:2208` (`MAINLOOP 0`, usePLR=1) → BEFORE | ✅ |
| NGL  | ds_read idx 141 < pack-mfma idx 161 → BEFORE | ds_read `:2175` < pack-mfma `:2208` (`MAINLOOP 0,0,1,1,0`, usePLR=1) → BEFORE | ✅ |
| NLL  | no offset:192 ds_read; pack-mfma idx 133 | `MAINLOOP 0,0,0,0,0`, usePLR=0 ⇒ ds_read `:2175` omitted | ✅ |

**The cms=1 capture is faithful to the cms=1 kernel.s in every body.** The macro
mechanism is explicit: the X0+12 ds_read is gated by `.if \usePLR == 1`
(`kernel.s:2174-2176`); the pack-mfma is unconditional (`kernel.s:2208`). ML and NGL
both instantiate with usePLR=1 (ds_read present, before pack); NLL instantiates with
usePLR=0 (ds_read absent). Capture body_idx ordering matches this exactly.

---

## §4 Answers to the four key questions

**Q1. Which (schedule, body) cells are ds_read BEFORE vs AFTER?**
- **BEFORE:** cms=1/ML, cms=1/NGL, SHADOW/ML, cms=0/NLL(tail). (cms=1 ML_PREV also BEFORE.)
- **AFTER:** **SHADOW/NGL only.**
- **N/A:** cms=0/ML, cms=0/NGL (T0→X0 scheme); cms=1/NLL, SHADOW/NLL (no in-body load).

**Q2. Is the cms=1 capture's per-body ordering consistent with cms=1 kernel.s?**
**Yes — fully faithful** (see §3). Both say ds_read BEFORE pack in ML and NGL via the
usePLR=1 path, and ds_read omitted in NLL via usePLR=0.

**Q3. Does SHADOW's ordering differ from cms=1's in any body?**
**Yes — and ONLY in NGL.** cms=1 NGL = BEFORE (idx 141<161); SHADOW NGL = AFTER
(idx 171>38). ML/ML_PREV are identical (both BEFORE) and NLL is identical (both no
in-body ds_read). They share the same in-place instruction set; the *scheduler* placed
the NGL ds_read late in SHADOW. So **the divergence is precisely one body: NGL.**

**Q4. Does cms=0's T0→X0 scheme even map onto this question?**
**No — for ML and NGL it is the wrong frame.** Under `doFullPackCodePrefetch` ON, the
ds_read targets `T0_I0` and the cvt *writes* `X0_I0+12` from T0; X0+12 is a pack
**destination**, never a ds_read load destination, in cms=0 ML/NGL. There is no "ds_read
of X0+12" to order against the pack. The only cms=0 body where X0+12 is itself ds_read'd
(in-place) is the **TailLoop**, where the frame applies and the order is BEFORE
(`:3302 < :3643`). So cms=0's ML/NGL cells are genuinely N/A, not "BEFORE-by-default".

---

## §5 What this says about the 16 failures / uvrl

The 16 `compare_graphs` failures (`compare_graphs_failures.txt`) are all
**EdgeRoutedDifferently** of the form:

> Subject's consumer `PackA3[..]`/`PackB3[..]` reads from subject's producer
> `PackA3[..]`/`PackB3[..]` (same body), but **reference routes through
> `LRA3[..]`/`LRB3[..]` of the next iteration**.

`LRA3`/`LRB3` = the **local-read (ds_read)** of the rotating fragment; the byte_keys
are exactly the X0+12..15 A-side family (`('v',11..14)`, plus 3/4/6 = X0+10/11 staging)
and the B-side equivalents (`('v',35..46)`). This is the same X0+12 fragment this table
tracks.

The ordering table explains the routing difference directly:
- In the **reference (SHADOW)**, the NGL ds_read (`LRA3`/`LRB3`) is placed **AFTER** the
  pack (body_idx 171 vs 38). So the reference's correct producer for the consumer is the
  **next iteration's** late ds_read — exactly what the failure text says ("routes through
  LRA3[..] of next iteration").
- In the **subject (cms=1)**, the NGL ds_read is **BEFORE** the pack (body_idx 141 vs
  161). So the consumer's nearest producer is the **same-body pack** — exactly the
  "reads from subject's producer PackA3[..]" the failure flags as different.

**Yes, the table supports "SHADOW is the odd one out," and it is because SHADOW is the
synthetic hybrid.** SHADOW = the **default scheduler ordering applied to the cms=1
in-place instruction set**. cms=1's own scheduler keeps the NGL ds_read early (BEFORE);
the default scheduler, fed the same in-place instructions, lifts the pack-mfma to the top
of the NGL body and defers the X0+12 local-read to the bottom (AFTER). That single NGL
reordering is what makes the reference's dependency edges route through the next
iteration's ds_read, producing all 16 EdgeRoutedDifferently failures. ML/ML_PREV/NLL do
not flip, which is consistent with the failures being concentrated on the rotating
fragment whose load the NGL scheduler moved.

Note also the scheme difference is *not* the cause here: cms=1 and SHADOW share the
**same in-place X0 instruction set** (both `doFullPackCodePrefetch` OFF). The failures are
purely an **ordering/scheduling** artifact in NGL, not a pack-scheme artifact. The cms=0
T0→X0 scheme is a separate codegen world (it would not even present these X0+12 ds_read
edges in ML/NGL), which is why it cannot be the reference for this comparison.

---

## §6 Bead filed

`rocm-libraries-d5mw` (bug, p1, labels validator/cms/ngl/tensilelite):
"SHADOW NGL flips ds_read AFTER pack-mfma; sole divergence from cms=1 driving 16
EdgeRoutedDifferently failures." Captures the table, the NGL flip citations, the
faithfulness cross-check, and the link to the 16 failures.
