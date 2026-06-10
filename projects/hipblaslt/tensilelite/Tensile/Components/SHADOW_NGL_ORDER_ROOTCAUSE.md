# SHADOW NGL pack-before-ds_read — ROOT CAUSE

Config: BPG#11 TF32 4x4 TN canonical kernel (`CANONICAL_KERNEL_CONFIG` in
`Tensile/Tests/unit/_dump_hxcx_assembly.py`). `LoopIters=4`, `numItersPLR=1`,
`isBarrier=3`, `numMfmaPerIter=12`, SIA3.

All capture-listing line numbers cite
`Tensile/Components/kernel_comparison/shadow_capture_listing.txt` (and the
freshly regenerated `hxcx_artifacts/shadow_capture_listing.txt`, byte-equivalent
in its data rows). KernelWriter / ScheduleCapture citations are against this
worktree.

---

## SUMMARY — verdict: **CAUSE-SCHED** (the capture is FAITHFUL, not buggy)

The SHADOW pack-before-ds_read ordering in NGL is produced by the **default SIA3
scheduler itself**, not by the capture infrastructure. The decisive
pre-capture-vs-post-capture test proves it:

- **Pre-capture (raw scheduler output).** NGL is scheduled across 4 separate
  `_makeSubIterSchedule` calls (subiter 0,1,2,3). In **subiter 0**'s raw
  `iterCode.flatitems()` the head pack `v_cvt_pk_bf16_f32 ValuA_X0_I0+0 <-
  ValuA_T0_I0+0,+1` is emitted at emission-index 1 (with the TF32 finalizer
  `v_mfma_f32_4x4x4_16b_bf16 ValuA_T0_I0+0..3` at emit 5). The ds_read that
  FEEDS it — `ds_read_b128 ValuA_T0_I0+0..3 ... offset:0` — is not emitted until
  **subiter 3**, at emission-index 6. So the *scheduler's own emission order*
  already has the pack before its feeding ds_read.
- **Post-capture (listing).** That head pack lands at `:440` body_idx 1
  (mfma_index -1); the feeding ds_read lands at `:600` body_idx 161
  (mfma_index 38). This is the SAME ordering as the raw scheduler output — the
  capture reorders nothing.

The capture's per-body listing order is `sorted(key=(slot.mfma_index,
slot.sequence))` (`ScheduleCapture.py:784-788`), and those slot indices are
assigned by walking `iterCode.flatitems()` **in scheduler emission order**
(`_captureSubIterToBuilder`, `KernelWriter.py:2786-2905`). The sort therefore
re-expresses, never overrides, the scheduler's slot decision. Conclusion: the
flip is a genuine product of the default scheduler applied to the cms=1
instruction set. The fix target is the validator model (uvrl) and/or the choice
to use the default scheduler as the SHADOW reference for an in-place pack
instruction set — **NOT** the capture infrastructure.

---

## §1 The anomaly (restated)

For the BPG#11 TF32 4x4 TN canonical kernel, in the SHADOW capture's NGL body the
`pack` group appears BEFORE its feeding `ds_read` group:

- pack `v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+0], v[vgprValuA_T0_I0+0],
  v[vgprValuA_T0_I0+1]` at `shadow_capture_listing.txt:440`, body_idx 1,
  mfma_index -1.
- feeding `ds_read_b128 v[vgprValuA_T0_I0+0:vgprValuA_T0_I0+0+3],
  v[vgprLocalReadAddrA+0] offset:0` at `:600`, body_idx 161, mfma_index 38.

Neither the real cms=0 kernel nor the real cms=1 kernel does this — both do
ds_read-before-pack in every body (see `DSREAD_PACK_CLASS_ORDERING.md` §2). ONLY
the SHADOW synthetic capture flips, and only in NGL.

The two candidate causes:
- **CAUSE-SCHED**: the default scheduler genuinely computes this order.
- **CAUSE-CAPTURE**: the order is an artifact of how the SHADOW schedule is
  captured (reordering, drops, mis-segmented body boundaries, mis-attributed
  leftover pack).

---

## §2 The SHADOW capture path (producer side, code reading)

The SHADOW (default) capture for NGL/NLL runs through the **same** code path the
CMS capture uses — `noLoadLoopBody` → `_makeSubIterSchedule` →
`_captureSubIterToBuilder`. There is NO bespoke SHADOW-only reordering route.

1. **Entry / flag.** `_captureDefaultSchedule` is set (`KernelWriter.py:512`,
   `:5618`) and consumed at the no-load-loop emitter and `_loopBody`. The flag
   gates *whether a capture object is threaded through* and whether
   `structural_clone` (not `deepcopy`) is used; it does NOT change scheduling.

2. **NGL emission.** `_noLoadLoopBodyDefault` drives the per-subiter interleave
   via `_makeSubIterSchedule` with `capture_body_label = "n_gl" if isNGLL else
   "n_ll"` (`KernelWriter.py:4063-4080`). Each unroll sub-iteration `u` is a
   separate `_makeSubIterSchedule` call; each call's returned `iterCode` Module
   is captured immediately.

3. **`_makeSubIterSchedule` capture-vs-noncapture difference is NON-reordering.**
   At `KernelWriter.py:992-1001`, `capture is not None` only swaps
   `structural_clone(self.codes.perIterGlobalRead[iteration])` for
   `deepcopy(...)` — identity preservation for GR-inc categorization. The
   scheduling logic below (the `isBarrier` this-loop/next-loop split at
   `:1004,1364-1366`; the next-loop-late block at ~`:2009-2015`; the pack-pre
   carry) is byte-identical with or without capture. So the capture path cannot
   produce an order the non-capture (real assembly) path wouldn't.

4. **`_captureSubIterToBuilder` is an in-order walk.** It iterates
   `for item in iterCode.flatitems(): ... capture.append(inst=item, ...)`
   (`KernelWriter.py:2786,2901-2905`), assigning `mfma_index` derived purely
   from how many MFMAs have been seen so far in this subiter
   (`:2874-2891`): instructions before this subiter's first MFMA get
   `mfma_index = -1` (subiter 0) or `subiter*numMfmaPerIter - 1` (subiter > 0);
   instructions after get `subiter*numMfmaPerIter + local_mfma_idx`. The walk
   appends in `flatitems()` (= scheduler emission) order; the mfma_index is a
   TAG, not a reordering.

5. **Listing order = stable re-sort of the scheduler's own slots.** The final
   per-body listing order is `sorted(instructions, key=lambda ti:
   (ti.slot.mfma_index, ti.slot.sequence))`
   (`ScheduleCapture.py:784-788`, `assign_stream_indices_for_body`; same key in
   `assign_emission_ordinals` `:813-816`). Because `slot.mfma_index` /
   `slot.sequence` are assigned from the scheduler's emission walk (step 4), the
   sort re-expresses the scheduler's slot decision. It does not invent a new
   order.

Net: the capture is a faithful transcription of the scheduler's per-subiter
emission order, re-keyed onto a `(mfma_index, sequence)` timeline.

---

## §3 The decisive test: pre-capture Module order vs post-capture listing

**Method.** TEMPORARY debug logging was inserted in `_makeSubIterSchedule`
immediately before the `_captureSubIterToBuilder` call
(`KernelWriter.py` ~2694), gated on `capture_body_label == "n_gl"` and an env
var. It dumped the raw `iterCode.flatitems()` emission order per NGL subiter,
classifying each leaf as DSREAD / PACK. The kernel was rebuilt via
`python Tensile/Tests/unit/_dump_hxcx_assembly.py`. The logging was then
**REVERTED** (§7).

**Pre-capture raw scheduler order (NGL, A-side `T0_I0+0..3` fragment):**

```
=== n_gl subiter=0 RAW iterCode.flatitems() order ===
  emit=1  PACK   v_cvt_pk_bf16_f32 ValuA_X0_I0+0 <- ValuA_T0_I0+0, ValuA_T0_I0+1
  ...
  emit=5  PACK   v_mfma_f32_4x4x4_16b_bf16 ValuA_T0_I0+0..3 (TF32 finalizer)
  emit=15 DSREAD ds_read_b128 ValuA_T0_I0+8..11 offset:256      (this-loop read)
  ...
=== n_gl subiter=3 RAW iterCode.flatitems() order ===
  emit=6  DSREAD ds_read_b128 ValuA_T0_I0+0..3   offset:0       (NEXT-loop read)
  emit=8  DSREAD ds_read_b128 ValuA_X0_I0+4..7   offset:64
  emit=10 DSREAD ds_read_b128 ValuB_T0_I0+0..3   offset:0
  ...
```

The pack of `ValuA_T0_I0+0..3` is emitted by the scheduler in **subiter 0**
(emit 1-5). Its feeding ds_read `ValuA_T0_I0+0..3 offset:0` is emitted by the
scheduler in **subiter 3** (emit 6). Since subiter 0's iterCode is appended to
the body before subiter 3's, the scheduler's own linear emission order already
has **pack before its feeding ds_read**.

**Post-capture listing order (same instructions):**

```
:440  NGL | body_idx 1   | mfma_index -1 | v_cvt_pk_bf16_f32 ValuA_X0_I0+0 <- T0_I0+0,+1
:600  NGL | body_idx 161 | mfma_index 38 | ds_read_b128 ValuA_T0_I0+0..3 offset:0
```

The post-capture order is identical in direction to the pre-capture order: pack
first (body_idx 1), feeding ds_read last (body_idx 161). The mfma_index -1 vs 38
exactly reflects that the pack came before subiter-0's first MFMA while the
ds_read came in subiter 3 (3*12 + 2 ≈ slot 38).

**Conclusion.** The pre-capture scheduler Module ALREADY has pack-before-ds_read.
Therefore **CAUSE-SCHED** — the scheduler produced it; the capture is faithful.
(If the pre-capture Module had been ds_read-first and only the listing flipped,
that would have been CAUSE-CAPTURE. It was not.)

---

## §4 Leftover-pack / body-attribution analysis

**Is the NGL-head pack NGL's own, or a carry from ML mis-attributed by the
capture's body-segmentation?**

It is NGL's OWN pack. The raw dump proves the head packs are emitted inside
**NGL's subiter=0 `_makeSubIterSchedule` call** — the one invoked with
`capture_body_label == "n_gl"` (`KernelWriter.py:4079`). They come out of NGL's
own `iterCode`, not ML's. The capture's body-segmentation did not reach into the
ML body and pull them in; there is no cross-body splice in
`_captureSubIterToBuilder` (it only ever walks the single `iterCode` passed to
it). So this is NOT the body-boundary-attribution flavor of CAUSE-CAPTURE.

**What the head pack IS:** the default scheduler's cross-body pack-pre / leftover
carry. The `isBarrier = LoopIters - numItersPLR = 3` split
(`KernelWriter.py:1004,1364-1366`) classifies the last sub-iteration's local
reads as *next-loop* reads, scheduled LATE (post-barrier block ~`:2009-2015`);
their packs are deferred and flushed at the HEAD of the following body. NGL,
being the first no-load body after the main loop, opens by flushing the prior
(ML) body's deferred pack work. So the head pack's *source values* trace to ML's
ds_read, but the pack *instruction* is genuinely scheduled into NGL's own
subiter-0 iterCode by the default scheduler. This is a legitimate cross-body
software-pipelining carry that the capture faithfully records — CAUSE-SCHED, not
a body-attribution bug.

(Register-agnostic feeding-inversion counts confirming this is unique to SHADOW
NGL are in `DSREAD_PACK_CLASS_ORDERING.md` §3: SHADOW NGL = 64 feeding
inversions; every other (schedule, body) cell = 0.)

---

## §5 Why cms=1 doesn't flip but SHADOW does

The cms=1 capture and the SHADOW capture use the **identical** capture code path:
`noLoadLoopBody` → `_makeSubIterSchedule(..., capture=...)` →
`_captureSubIterToBuilder` (`KernelWriter.py:4063-4080`). There is no
SHADOW-specific reordering branch in the capture machinery — the only
capture-conditional code in `_makeSubIterSchedule` is the
`structural_clone`-vs-`deepcopy` of globalReadCode and the category-map plumbing
(`:992-1071`), neither of which reorders instructions.

Therefore, if the capture infrastructure were the cause, cms=1 would flip too.
It does not (`cms1_capture_listing.txt`, `DSREAD_PACK_CLASS_ORDERING.md` §2/§3:
cms=1 NGL is ds_read-first, 0 feeding inversions). The difference is entirely in
the **scheduler input**, not the capture path:

- **cms=1** injects a custom main-loop schedule (CustomSchedule slot arrays) that
  keeps each local read and its pack inside the same body, so the captured order
  is ds_read-first uniformly.
- **SHADOW** applies the **default SIA3 scheduler** to the same cms=1 in-place
  instruction set (`doFullPackCodePrefetch=False`; see bead
  `rocm-libraries-svds`). SIA3's `isBarrier` split scatters the last
  sub-iteration's reads to the body tail while their packs carry to the next
  body's head — producing the NGL flip.

Same capture infra + same instruction set + DIFFERENT scheduler = the flip
appears only on the default-scheduler side. That isolates the cause to the
scheduler.

---

## §6 Verdict and fix target

**CAUSE-SCHED.** The default SIA3 scheduler genuinely emits pack-before-feeding-
ds_read in NGL for this config; the SHADOW capture transcribes that order
faithfully. The capture infrastructure is NOT buggy: it neither reorders, drops,
nor mis-attributes instructions across body boundaries. The decisive evidence is
the pre-capture raw `iterCode.flatitems()` order (§3), which already has the flip
before `_captureSubIterToBuilder` runs.

**Fix target.** Because the SHADOW reference legitimately reflects the default
scheduler, the corrective action is NOT in the capture infrastructure. It is:
1. the validator's unrolled latest-writer model (uvrl) — it over-reports an
   ordering that is benign cross-body software pipelining (the head pack reads the
   prior body's already-packed, live output; see `SHADOW_NGL_SCHEDULER_TRACE.md`
   §4/§6); and/or
2. the upstream decision to use the default scheduler as the SHADOW reference for
   a CMS (in-place pack) instruction set whose real default codegen
   (`doFullPackCodePrefetch=True`, cms=0) uses a different pack scheme entirely
   (bead `rocm-libraries-svds`).

No capture-infrastructure fix is warranted; the 16 failures are NOT capture
artifacts in the sense of a recording bug — they are a real scheduler/validator
modeling mismatch.

---

## §7 Debug-logging revert confirmation (net-zero source change)

The only source edit was a TEMPORARY debug block in `_makeSubIterSchedule`
(`KernelWriter.py`, immediately before the `_captureSubIterToBuilder` call),
gated on env var `SHADOW_NGL_RAWDUMP` and `capture_body_label == "n_gl"`,
dumping raw `iterCode.flatitems()` order. It has been **REVERTED**.

Verification: `git diff --stat` is EMPTY (no tracked changes); `git status
--short` shows only untracked `hxcx_artifacts/` and `oplb_artifacts/` build-output
dirs plus this new findings file under `Tensile/Components/`. Net source change
is zero.

---

## §8 Bead filed

No new P0 capture-bug bead — the task's "P0 if CAUSE-CAPTURE" condition does NOT
apply, since the verdict is CAUSE-SCHED (capture faithful).

The decisive pre-capture-vs-post-capture finding (CAUSE-SCHED) was recorded as a
comment on the existing, on-point bead **`rocm-libraries-d5mw`** ("SHADOW NGL
flips ds_read AFTER pack-mfma; sole divergence from cms=1 driving 16
EdgeRoutedDifferently failures", open, P1). Related context beads:
`rocm-libraries-svds` (SHADOW = default scheduler over cms=1 instruction set) and
the uvrl validator-modeling fix already tracked.
