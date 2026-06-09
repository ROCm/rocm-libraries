# NGL capture faithfulness investigation (rocm-libraries-uvrl)

Read-only investigation. No source changed. Every ordering claim cites file + line.

All `kernel.s` / `*_capture_listing.txt` line numbers below resolve into the
committed frozen snapshot under
`Tensile/Components/h7lo_uvrl_artifacts/` (the gitignored `hxcx_artifacts/`
copies are byte-identical regen output).

---

## §1 The hypothesis under test

The 16 `EdgeRoutedDifferentlyFailure`s attributed to `rocm-libraries-uvrl` are
explained today (in `Tensile/Components/H7LO_UVRL_NORMATIVE_ORDERING.md`) as a
**real** CMS-vs-default schedule divergence in the NGL body: CMS orders
`ds_read -> pack_chain`, default orders `pack_chain -> ds_read`, and the per-byte
latest-writer model cannot see the two as value-equivalent.

The user's hypothesis: that explanation assumes both `FourPartCapture`s
faithfully reflect the emitted assembly. Maybe the NGL body capture is
**unfaithful** — analogous to the hxcx prologue bug (`5267acc76bed`), where
`build_prologue_capture` concatenated PRE-interleave per-side snapshots and
dropped the real `_interleavePackAB` output + SNop pads. If the NGL capture has
the same defect, the "divergence" is a capture artifact, not a real schedule
difference.

Possible verdicts: **H1** CMS NGL capture unfaithful (subject side); **H2**
SHADOW NGL capture unfaithful (reference side); **H3** both faithful, uvrl
framing stands; **H4** something else.

---

## §2 CMS: captured NGL order vs emitted `kernel.s` NGL order

### 2.1 The captured "NGL" body is the OptNLL `MAINLOOP` macro

The validator's NGL body is the `Opt. NoLoadLoop` instantiation. In `kernel.s`,
OptNLL begins at `kernel.s:2260` (`/* Opt. NoLoadLoop - Begin */`) and is driven
by `MAINLOOP 0, 0, 1, 1, 0` at `kernel.s:2253` (args `useGR=0, usePLR=1,
useGRInc=1, useLoop=0`). The `MAINLOOP` body is **defined once** as a GAS macro
at `kernel.s:1889` (`.macro MAINLOOP ...`) and runs to `.endm` at
`kernel.s:2237`. With `useGRInc=1` every `.if \useGRInc == 1` block in that macro
body is active — so the macro text at `kernel.s:1889-2237` IS the literal NGL
emission.

> Correction to the prior memo: `H7LO_UVRL_NORMATIVE_ORDERING.md` §"one concrete
> byte" cites `kernel.s:2665 / 3000 / 3008` as the NGL writers. Those lines are
> in the **tail loop** — `label_TailLoopBeginL` is at `kernel.s:2659`
> (`ds_read` for `ValuA_X0_I0+12` at `kernel.s:2665` is the tail-loop local read,
> not NGL). The prior memo cross-checked CMS faithfulness against the wrong
> region. The correct NGL writers are inside the `MAINLOOP` macro body (below).
> The prior memo's *conclusion* (capture faithful, order is ds_read→pack) is
> nonetheless correct; only its `kernel.s` citations were wrong.

### 2.2 Direct comparison for `ValuA_X0_I0+12 / +15` (the failure-3 byte)

Inside the `MAINLOOP` macro body (`kernel.s:1889-2237` = the NGL emission):

| `kernel.s` line | instruction | role |
|---|---|---|
| `kernel.s:2175` | `ds_read_b128 v[vgprValuA_X0_I0+12:+3], ... offset:192` | ds_read (LRA3) — FIRST |
| `kernel.s:2208` | `v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:+3], ...` | pack-mfma — middle |
| `kernel.s:2225` | `v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[...+14], v[...+15]` | pack-cvt — LAST |

Emitted order: **ds_read → pack-mfma → pack-cvt**.

CMS capture (`cms_capture_listing.txt`), same three instructions:

| listing line | mfma_index | instruction | role |
|---|---|---|---|
| `cms_capture_listing.txt:584` | 37 | `ds_read_b128 v[vgprValuA_X0_I0+12:+3] offset:192` | FIRST |
| `cms_capture_listing.txt:604` | 43 | `v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:+3]` | middle |
| `cms_capture_listing.txt:612` | 46 | `v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15]` | LAST |

Captured order: **ds_read → pack-mfma → pack-cvt**. Body head also matches: macro
first instr `v_mfma acc[0:3]...A_X0_I0+0` (`kernel.s:1891`) = capture
`cms_capture_listing.txt:443` (mfma_index 0); macro first ds_read
`...A_T0_I0+8 offset:256` (`kernel.s:1895`) = capture
`cms_capture_listing.txt:445`.

**The CMS capture is faithful to the emitted macro body.** Capture and emission
AGREE on the ds_read-vs-pack ordering. → **H1 is refuted.**

---

## §3 SHADOW: NGL capture plausibility / internal consistency

The SHADOW NGL has no emitted `.s` (default schedule is never written as text on
a CMS build), so §2's direct cross-check is impossible. Two independent checks
instead:

### 3.1 The suspicious head block is NOT a side-snapshot artifact

`shadow_capture_listing.txt:439-452` is a block of 14 instructions all carrying
`mfma_index = -1`, BEFORE the first real mfma at idx 0 (`:453`):
`s_waitcnt lgkmcnt(0)`, 4 A pack-cvts, 2 A pack-mfmas, 4 B pack-cvts, 2 B
pack-mfmas, `s_nop 0`. Superficially this resembles the hxcx "all packs grouped
at the head" concatenation symptom.

But it is NOT grouped-by-snapshot: after the head block the body is properly
**interleaved** — `v_mfma_f32_16x16x32_bf16` (mfma_index 0) at `:453`, then
`ds_read` at `:454`, then pack-cvts at `:458-465`, then `s_nop`, then mfma_index
1 at `:467`, ds_read `:468`, packs `:472-483`, etc. A side-snapshot bug (hxcx
shape) would produce ALL packs then ALL ds_reads with NO mfma interleave. The
SHADOW capture instead has packs and ds_reads woven between mfmas at every
mfma_index. The `mfma_index = -1` head block is the genuine **pre-first-mfma
leftover-pack flush** the default SIA3 scheduler emits for PLR (the buffer
prefetched by the previous body is packed before the first NGL mfma). It is real
emission, not concatenation.

### 3.2 ds_read lands LAST, consistent with a real default schedule

The A-side iter-3 `ds_read`s (`ValuA_X0_I0+12`, the failure byte) appear at
`shadow_capture_listing.txt:610` (mfma_index 43), with the whole A/B ds_read
sweep clustered at `:600-616` (mfma_index 38-45) — at the END of the body, after
the pack chains (`:458-571`). This is exactly what default codegen does: it
schedules the next-tile local reads late, behind the compute. Internally
consistent and plausible; no grouping or duplication anomaly.

---

## §4 How the NGL body is built (producer-side code reading)

Both NGL bodies are produced by the SAME scheduler routine, and the capture is a
**post-interleave walk of the emitted module**, not a pre-interleave snapshot.

- SHADOW (default / reference) NGL: `KernelWriter.py:4185-4205`. A
  `LoopBodyCaptureBuilder` (`shadow_capture`) drives
  `self._noLoadLoopBodyDefault(... capture=shadow_capture)` — the **same**
  function the real non-CMS build calls (`KernelWriter.py:4226` for non-CMS
  capture, `:4240` `noLoadLoopBody`). The result is finalized into
  `ctx.default_n_gl` (`KernelWriter.py:4205`).
- Inside `_noLoadLoopBodyDefault`, the per-subiter stream is built by
  `_makeSubIterSchedule(... capture=capture, capture_body_label="n_gl")`
  (`KernelWriter.py:4041-4057`), and the SAME object is emitted:
  `module.add(subIterCode)` (`KernelWriter.py:4059`).
- In `_makeSubIterSchedule` (`KernelWriter.py:983`), the SIA3 path assembles
  `iterCode` by popping `packItems` and interleaving them with `macIterItems` at
  scheduled mfma slots (`KernelWriter.py:1222-1271`). The capture is taken
  **after** that interleave by a post-hoc walk: `_captureSubIterToBuilder(
  iterCode=iterCode, ...)` at `KernelWriter.py:2694-2704`, documented at
  `:2688-2693` as a "post-hoc walk of iterCode", and the same `iterCode` is
  `return`ed at `KernelWriter.py:2706`. `_captureSubIterToBuilder` is defined at
  `ScheduleCapture.py:2708` and "Walk[s] iterCode.flatitems() and append[s]
  TaggedInstructions to capture" (`ScheduleCapture.py:2713`).

This is the **structural inverse** of the hxcx prologue bug. hxcx snapshotted
PRE-interleave per-side pack lists; here the capture observes the already-emitted
`iterCode` Module after `_interleave`/slot-pop has run. Capture order == emission
order by construction, for both CMS and SHADOW (the CMS macro path,
`_emitNoLoadLoopBodyCMSMacro` at `KernelWriter.py:3401`, feeds the same
`_makeSubIterSchedule`/post-hoc-walk machinery; cf. `build_cms_four_part_capture`
at `ScheduleCapture.py:2763`).

No `_interleave`-style step exists whose OUTPUT is emitted but whose INPUT is
captured. → faithful single-stream on both sides.

---

## §5 ds_read-vs-pack ordering: schedule intent vs emitted vs captured, per side

### CMS (subject)
- Schedule intent (`_128x128x32_TF32.py`, `('TN',False,1)` branch): `lra3` =
  `[36,36,37,37]` (`_128x128x32_TF32.py:87`) places the A ds_read at mfma 36-37,
  **before** `pack_a3` = `[39,39,...,46,46,47,47]` (`_128x128x32_TF32.py:90`).
  Intent: ds_read FIRST.
- Emitted (`kernel.s`): ds_read `:2175` < pack-mfma `:2208` < pack-cvt `:2225`.
  Matches intent.
- Captured (`cms_capture_listing.txt`): ds_read `:584` (mfma 37) < pack-mfma
  `:604` (43) < pack-cvt `:612` (46). Matches emitted.
- **All three agree: ds_read → pack.**

### SHADOW (reference / default)
- No schedule-array source (default codegen does not use the CMS slot arrays).
- No emitted `.s`.
- Captured (`shadow_capture_listing.txt`): pack chain `:458-571` < A ds_read
  `:610` (mfma 43). ds_read LAST. Internally consistent (§3) and faithful to the
  `_noLoadLoopBodyDefault` emission by construction (§4).
- **pack → ds_read**, the genuine default-scheduler decision.

The two sides genuinely order these oppositely, and **both captures reflect their
respective real emissions.**

---

## §6 Verdict

**H3 — both captures are faithful.** The divergence is a real CMS-vs-default
schedule ordering difference; the per-byte latest-writer model limitation (the
current uvrl framing) stands.

Concrete evidence:
1. CMS capture order == emitted macro-body order, byte-for-byte on the failure
   byte: `cms_capture_listing.txt:584/604/612` == `kernel.s:2175/2208/2225`
   (§2.2). H1 refuted.
2. SHADOW capture is a post-interleave walk of the real `_noLoadLoopBodyDefault`
   emission (`KernelWriter.py:4196-4205`, `:4041-4059`, `:2694-2706`;
   `ScheduleCapture.py:2708-2713`), structurally the inverse of the hxcx
   pre-interleave-snapshot bug (§4). Its head `mfma_index=-1` block is an
   interleaved real leftover-pack flush, not a grouped concatenation artifact
   (§3.1). H2 refuted.
3. CMS schedule intent (`_128x128x32_TF32.py:87` `lra3` slots 36-37 ahead of
   `:90` `pack_a3` slot 39+) matches the emitted+captured ds_read-first order;
   default's pack-first order is the real alternative (§5).

The user's hypothesis (capture-pipeline unfaithfulness on either side) is not
borne out for the NGL body. The hxcx defect class does not recur here because the
NGL capture path observes the post-schedule `iterCode`, whereas the prologue path
that hxcx fixed snapshotted pre-interleave side lists.

---

## §7 Since H3: uvrl framing confirmed

No capture bug to fix. `uvrl` remains a **validator-modeling** question (case B1
in `H7LO_UVRL_NORMATIVE_ORDERING.md`): teach the per-byte latest-writer model
that, for the rotating pack buffer `ValuA/B_X0_I0+k`, the `ds_read` that loads
byte K and the pack-cvt that re-derives byte K are value-equivalent producers.
No kernel/schedule change; consistent with the "differ ONLY in scheduling"
contract.

One documentation defect found and filed (see §8): the prior memo's `kernel.s`
NGL line citations point at the tail loop, not the NGL macro body. This is a
stale-citation bug in `H7LO_UVRL_NORMATIVE_ORDERING.md`, not a capture bug.

---

## §8 Confidence + residual uncertainty

**Confidence: high** on H3 for the NGL body.
- The CMS side is proven by exact text+order match against the macro that IS the
  emission (§2.2) — not inference.
- The producer-side reading (§4) shows the capture is a post-hoc walk of the
  emitted `iterCode` on both paths; the hxcx failure mode is architecturally
  absent.

**Residual uncertainty:**
- SHADOW has no emitted `.s`, so its faithfulness rests on the producer-side
  argument (§4) plus internal-consistency (§3), not a direct text diff. What
  would fully close it: a one-off build with `_captureNonCmsBuild` on a non-CMS
  kernel (`KernelWriter.py:4218-4237`) that DOES emit the default NGL as `.s`,
  then diff that `.s` against `shadow_capture_listing.txt`. Out of scope here
  (read-only), but it would convert the §4 structural argument into a §2-style
  direct proof.

### Filed
- bead for the stale `kernel.s` NGL citations in
  `H7LO_UVRL_NORMATIVE_ORDERING.md` (tail-loop `:2665/3000/3008` should be
  macro-body `:2175/2208/2225`).
