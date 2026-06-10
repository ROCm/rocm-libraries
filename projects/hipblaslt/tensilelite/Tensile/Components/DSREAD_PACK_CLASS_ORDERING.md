# ds_read vs pack — register-agnostic CLASS ordering per loop body

**Config:** BPG#11 TF32 4x4 TN canonical kernel (CANONICAL_KERNEL_CONFIG),
`LoopIters=4`, `numItersPLR=1`, `isBarrier=3`, `numMfmaPerIter=12`, SIA3.

Comparison kernels are preserved for manual inspection at
`Tensile/Components/kernel_comparison/`:
- `cms0_kernel.s` — real cms=0 (UseCustomMainLoopSchedule=0) raw assembly.
- `cms1_kernel.s` — real cms=1 (UseCustomMainLoopSchedule=1) raw assembly.
- `cms1_capture_listing.txt` — cms=1 capture (body-segmented).
- `shadow_capture_listing.txt` — SHADOW (default synthetic reference) capture.

All line numbers below cite those committed files (capture listings use their own
`file_line` column; `.s` files use raw line numbers). KernelWriter citations are
against `Tensile/KernelWriter.py` in this worktree.

---

## SUMMARY — hypothesis CONFIRMED

Treating `ds_read*` as the opaque class **DSREAD** and both `v_cvt_pk_bf16_f32`
and the TF32-emulation finalizer `v_mfma_f32_4x4x4_16b_bf16` as the opaque class
**PACK**, register-agnostically:

- **cms=0** does DSREAD before PACK in EVERY loop body (ML, NGL, OptNLL, OrdNLL,
  Tail). No pack ever precedes its feeding ds_read. First event of every loop
  body is DSREAD.
- **cms=1** does DSREAD before PACK in EVERY loop body (ML_PREV, ML, NGL, NLL).
  No pack ever precedes its feeding ds_read. First event of every loop body is
  DSREAD.
- **SHADOW** does NOT. In the **NGL** body the PACK group is hoisted to the body
  head and runs **before** the DSREAD that feeds it — a genuine pack-before-its-
  feeding-ds_read inversion that no real kernel (cms=0 or cms=1) produces.

So the hypothesis holds. The precise, sharpest localization: **SHADOW NGL is the
unique (schedule, body) cell where a pack precedes the in-body ds_read that loads
its source registers.** (SHADOW NLL also *opens* with packs, but those packs are
fed by the previous body's NGL ds_reads — the terminal last loop emits no
next-loop ds_read — so it is not a feeding inversion; see §3 note.)

### The table (register-agnostic class ordering)

Two columns per cell:
- **first event** = the raw class of the first DSREAD/PACK instruction in the body.
- **feeding test** = does any PACK precede the *in-body* DSREAD that loads its
  source? `clean` = no; `INVERTED` = yes (pack before its feeding ds_read).

| schedule | ML_PREV | ML | NGL | NLL | (cms0 extra) |
|---|---|---|---|---|---|
| **cms=0** | — | DSREAD / clean | DSREAD / clean | OptNLL DSREAD / clean ; OrdNLL DSREAD / clean | Tail DSREAD / clean |
| **cms=1** | DSREAD / clean | DSREAD / clean | DSREAD / clean | DSREAD / clean | — |
| **SHADOW** | DSREAD / clean | DSREAD / clean | **PACK / INVERTED** | PACK / clean* | — |

\* SHADOW NLL first event is PACK, but feeding-clean: its head packs consume
registers loaded by NGL ds_reads, not by a later in-NLL ds_read (terminal loop).
cms=1 NLL by contrast leads with DSREAD. The decisive divergence is **NGL**.

(cms=0's ML body internally maps two unroll sub-iterations; cms=1's ML_PREV is a
clone of ML. PRO is a prologue, not a loop body: both SHADOW and cms=1 open PRO
with PACK consuming prologue-loaded data — identical, not a divergence — so PRO is
omitted from the loop-body table.)

---

## §1 Method

**Class definition (register-agnostic).** An instruction is **DSREAD** iff its
mnemonic begins `ds_read` (here always `ds_read_b128`). It is **PACK** iff its
mnemonic is `v_cvt_pk_bf16_f32` (the half/bf16 pack) or
`v_mfma_f32_4x4x4_16b_bf16` (the TF32-emulation pass that finalizes packing). All
other instructions are ignored. The physical/symbolic register a given DSREAD or
PACK touches (T0 vs X0, +0 vs +12, …) is irrelevant to the *class* ordering and is
NOT used to decide before/after. (Registers ARE used, separately, only for the
feeding sub-test in §3, to answer the stronger question "did a pack run before the
ds_read that produced its inputs.")

**Body segmentation.**
- *cms=1* and *SHADOW*: the capture listings are already body-segmented by the
  `body` column; body headers `### body PRO/ML_PREV/ML/NGL/NLL`. The capture
  `body_idx` column is the in-body scheduled position (the timeline order within
  that body), which is what "before/after" means here.
  - SHADOW bodies: PRO `:6`, ML_PREV `:50`, ML `:244`, NGL `:438`, NLL `:618`.
  - cms=1 bodies: PRO `:6`, ML_PREV `:50`, ML `:246`, NGL `:442`, NLL `:618`.
- *cms=0*: no capture exists; the raw `.s` was segmented by labels:
  - ML  = `label_LoopBeginL:1872` .. `label_LoopEndL:2200`.
  - NGL = `Ord. NoGlobalLoadLoop_1 - Begin :2205` .. `:2511`.
  - OptNLL no-load = `Opt. NoLoadLoop - Begin :2517` .. `:2733`.
  - OrdNLL no-load = `Ord. NoLoadLoop - Begin :2928` .. `:3122`.
  - Tail = `label_TailLoopBeginL:3296` .. `label_TailLoopEndL:3766`.
  Within each region the raw `.s` line order is the timeline order.

The A-side fragment (`ValuA*`/`LocalReadAddrA`) was used as the primary test; the
B-side (`ValuB*`/`LocalReadAddrB`) was confirmed to mirror it (§3).

---

## §2 Per-(schedule, body) ordered class events (evidence)

Only the leading window of each body is shown — enough to establish first-event
class and the head ordering. "[DS]" = DSREAD, "[P]" = PACK.

### cms=0 (raw `cms0_kernel.s`)

**ML A-side** — DSREAD group first, then PACK group consuming those regs:
```
L1886 [DS] ds_read T0+0..3        L1894 [DS] T0+4..7   L1900 [DS] T0+8..11  L1906 [DS] T0+12..15
L1938 [P]  cvt X0+16 <- T0+0,+1   ... L1946 [P] mfma-finalize T0+0..3
L2081 [DS] T0+0..3 (next wave)    L2120 [P] cvt X0+0 <- T0+0,+1
```
First event = DSREAD (L1886). Every cvt-pack's source T0 reg is loaded by an
earlier in-body ds_read. `clean`.

**NGL A-side** (Ord. NoGlobalLoadLoop):
```
L2214 [DS] T0+0..3  L2222 [DS] T0+4..7  L2228 [DS] T0+8..11  L2234 [DS] T0+12..15
L2266 [P]  cvt X0+16 <- T0+0,+1   ... L2274 [P] mfma-finalize T0+0..3
L2395 [DS] T0+0..3 (next wave)    L2437 [P] cvt X0+0 <- T0+0,+1
```
First event = DSREAD (L2214). `clean`.

**OptNLL / OrdNLL / Tail:** first event DSREAD (L2548 / L2937 / L3299
respectively), feeding-clean (§3 counts).

### cms=1 (capture `cms1_capture_listing.txt`)

**ML A-side** — body OPENS with DSREAD:
```
:249 body_idx 2  [DS] T0+8..11    :250 body_idx 3 [DS] X0+20..23
:253 body_idx 6  [DS] T0+12..15   :254 body_idx 7 [DS] X0+28..31
:263 body_idx 16 [P]  cvt X0+16 <- T0+8,+9   (pack AFTER its feeding ds_read)
```
First event = DSREAD (body_idx 2). `clean`.

**NGL A-side** — body OPENS with DSREAD:
```
:445 body_idx 2  [DS] T0+8..11    :446 body_idx 3 [DS] X0+20..23
:449 body_idx 6  [DS] T0+12..15   :450 body_idx 7 [DS] X0+28..31
:459 body_idx 16 [P]  cvt X0+16 <- T0+8,+9
... later, the OTHER half:
:578 body_idx 135 [DS] T0+0..3    :579 body_idx 136 [DS] X0+4..7
:588 body_idx 145 [P]  cvt X0+0 <- T0+0,+1   (still pack AFTER its feeding ds_read)
```
First event = DSREAD (body_idx 2). `clean`. Every cvt-pack source reg has an
earlier in-body ds_read.

**NLL A-side** — body OPENS with DSREAD:
```
:620 body_idx 1 [DS] T0+8..11   :621 body_idx 2 [DS] X0+20..23
:628 body_idx 9 [P] cvt X0+16 <- T0+8,+9
```
First event = DSREAD (body_idx 1). `clean`.

### SHADOW (capture `shadow_capture_listing.txt`)

**ML A-side** — body OPENS with DSREAD (matches cms=1):
```
:247 body_idx 2  [DS] T0+8..11    :252 body_idx 7 [DS] X0+20..23
:257 body_idx 12 [DS] T0+12..15   :262 body_idx 17 [DS] X0+28..31
:287 body_idx 42 [P] cvt X0+16 <- T0+8,+9
```
First event = DSREAD (body_idx 2). `clean`.

**NGL A-side — THE FLIP. Body OPENS with PACK; feeding ds_read is at the END:**
```
:440 body_idx 1 [P]  cvt X0+0 <- T0+0,+1     <-- FIRST EVENT IS PACK
:441 body_idx 2 [P]  cvt X0+1 <- T0+2,+3
:442 body_idx 3 [P]  cvt X0+2 <- X0+4,+5
:443 body_idx 4 [P]  cvt X0+3 <- X0+6,+7
:444 body_idx 5 [P]  mfma-finalize T0+0..3
:454 body_idx 15 [DS] T0+8..11   <-- first ds_read, AFTER the head packs
...
:600 body_idx 161 [DS] T0+0..3   <-- the ds_read that FEEDS the body_idx-1 pack
:602 body_idx 163 [DS] X0+4..7   <-- feeds the body_idx-3 pack
```
First event = PACK (body_idx 1, `:440`). The head pack at `:440` reads
`ValuA_T0_I0+0,+1`; the ds_read that loads `ValuA_T0_I0+0..3` is `:600`
(body_idx 161) — at the END of the same body. **PACK precedes its feeding DSREAD
by 160 in-body positions. INVERTED.**

**NLL A-side** — body OPENS with PACK, but feeding-clean:
```
:620 body_idx 1 [P]  cvt X0+0 <- T0+0,+1     <-- first event PACK
:624 body_idx 5 [P]  mfma-finalize T0+0..3
:634 body_idx 15 [DS] T0+8..11
```
First event = PACK. But the head packs (`:620` reading T0+0..1) are fed by the
*previous body's* (NGL) ds_reads; NLL is the terminal loop (`hasLiveLdsData=False`)
and emits no in-NLL ds_read that loads T0+0..3, so there is no in-body feeding
inversion. `clean` under the feeding test.

---

## §3 The decisive feeding test (A and B side)

For each `v_cvt_pk_bf16_f32` we resolved its source `ValuX_*` registers and looked
up the in-body ds_read that first loads each. A **violation** = the feeding ds_read
is scheduled LATER (greater body_idx / line) than the pack. Counts:

| (schedule, body) | first event | cvt-packs fed by earlier in-body ds_read (OK) | VIOLATIONS (pack before feeding ds_read) | pack srcs prefetched from a PRIOR body |
|---|---|---|---|---|
| SHADOW PRO       | PACK   | 0   | 0   | 64 |
| SHADOW ML_PREV   | DSREAD | 128 | 0   | 0  |
| SHADOW ML        | DSREAD | 128 | 0   | 0  |
| **SHADOW NGL**   | **PACK** | 64 | **64** | 0  |
| SHADOW NLL       | PACK   | 64  | 0   | 64 |
| cms=1 PRO        | PACK   | 0   | 0   | 64 |
| cms=1 ML_PREV    | DSREAD | 128 | 0   | 0  |
| cms=1 ML         | DSREAD | 128 | 0   | 0  |
| cms=1 NGL        | DSREAD | 128 | 0   | 0  |
| cms=1 NLL        | DSREAD | 64  | 0   | 64 |
| cms=0 ML         | DSREAD | 128 | 0   | 0  |
| cms=0 NGL        | DSREAD | 128 | 0   | 0  |
| cms=0 OptNLL     | DSREAD | 64  | 0   | 0  |
| cms=0 OrdNLL     | DSREAD | 64  | 0   | 0  |
| cms=0 Tail       | DSREAD | 96  | 0   | 0  |

**SHADOW NGL is the ONLY cell with violations (64).** First six violations
(A-side):
```
pack@body_idx 1 (:440) reads A_T0+0 ; feeding ds_read@body_idx 161 (:600) is LATER
pack@body_idx 1 (:440) reads A_T0+1 ; feeding ds_read@body_idx 161 (:600) is LATER
pack@body_idx 2 (:441) reads A_T0+2 ; feeding ds_read@body_idx 161 (:600) is LATER
pack@body_idx 2 (:441) reads A_T0+3 ; feeding ds_read@body_idx 161 (:600) is LATER
pack@body_idx 3 (:442) reads A_X0+4 ; feeding ds_read@body_idx 163 (:602) is LATER
pack@body_idx 4 (:443) reads A_X0+6 ; feeding ds_read@body_idx 163 (:602) is LATER
```

**B-side mirrors A-side.** Per-side breakdown of NGL:
```
SHADOW NGL A-side : first=PACK(body_idx 1, :440)  OK=32  VIOL=32
SHADOW NGL B-side : first=PACK(body_idx 7, :446)  OK=32  VIOL=32
cms=1  NGL A-side : first=DSREAD(body_idx 2, :445) OK=64 VIOL=0
cms=1  NGL B-side : first=DSREAD(body_idx 29,:472) OK=64 VIOL=0
```

---

## §4 Verdict on the hypothesis — CONFIRMED

- **cms=0**: DSREAD-before-PACK in ALL bodies. The prior analysis that marked
  cms=0 ML/NGL as "N/A" was wrong *as a class question*: once register identity is
  ignored, cms=0 ML and NGL both unambiguously emit the DSREAD group first and the
  PACK group after, with zero feeding inversions (§2, §3). CONFIRMED.
- **cms=1**: DSREAD-before-PACK in ALL bodies (ML_PREV, ML, NGL, NLL). Zero
  feeding inversions. First event of every loop body is DSREAD. CONFIRMED.
- **SHADOW**: NOT DSREAD-before-PACK everywhere. In **NGL** the PACK group is
  hoisted to the body head and runs before the feeding DSREAD — 64 feeding
  inversions, first event PACK, on both A and B sides. CONFIRMED (and it is NGL,
  exactly as predicted).

Precise statement: **The SHADOW synthetic reference reorders the pack group ahead
of its feeding ds_read in NGL — the single (schedule, body) cell no real kernel
(cms=0 or cms=1) produces.** That is the exact defect in the SHADOW reference for
this config.

---

## §5 Where the SHADOW reordering originates (scheduler code)

The reorder is produced by the default SIA3 scheduler, `_makeSubIterSchedule`
(`KernelWriter.py:983`), specifically the this-loop / next-loop local-read split
plus the cross-body pack carry:

1. **The split** (`KernelWriter.py:1364-1366`):
   ```python
   localReadItems          = localReadCodeAB.flatitems()
   localReadItemsThisLoop  = localReadItems if iteration <  isBarrier else []
   localReadItemsNextLoop  = localReadItems if iteration >= isBarrier else []
   ```
   with `isBarrier = LoopIters - numItersPLR` (`:1004`) = `4 - 1 = 3`. The last
   sub-iteration's reads (iteration `3 >= isBarrier`) are classified **next-loop**.

2. **Next-loop reads are scheduled LATE** — only after the barrier, in the block
   headed "scheduled local read for next loop / localReads for next loop should
   after barrier" (`KernelWriter.py:2009-2043`), gated
   `if self.states.numItersPLR and iteration >= isBarrier` (`:2015`). This is why
   the feeding ds_read for the head packs lands at the body TAIL (SHADOW NGL
   `:600`, body_idx 161) — at mfma slot 43, late within the body.

3. **The pack of a next-loop read is deferred to the FOLLOWING body and flushed at
   its head.** Within a single iteration's slot the emit order is correct —
   next-loop localReads (`:2009-2043`) precede pack-pre (`:2089-2147`) precede pack
   (`:2149+`). But the *pack-pre carry* (`doPackPreSchedulingNextLoop`, e.g.
   `:2028-2029`, `:2082-2084`; `startPrePackIndex` computed near `:1663`) hoists
   the previous body's deferred pack into the head of the next body. In NGL
   (`_noLoadLoopBodyDefault`, `KernelWriter.py:3454`, per-iteration interleave via
   `_makeSubIterSchedule`) the body head therefore *opens* by flushing ML's
   deferred pack work (SHADOW `:440-445`), while NGL's own next-loop ds_read still
   schedules late at `:600`. Result: pack-before-(its-feeding)-ds_read in NGL.

In **ML** (`_loopBody`) the analogous deferred-pack flush lands in the trailing
`mfma_index=-1` block at the END of the body, *after* the ds_read — so ML reads
DSREAD-first. The asymmetry is purely *which body owns the head-flush*: a steady
main-loop iteration trails it; the NGL body (first no-load body after the main
loop) inherits the prior body's flush at its head.

**cms=1 avoids this** because its custom schedule (CustomSchedule slot arrays,
e.g. `CustomSchedule/gfx950/_128x128x32_TF32.py`) keeps each read and its pack
inside the same body, so DSREAD precedes PACK uniformly. **cms=0** keeps the
read+pack pairing within the body for this config (the next-loop reads in cms=0's
ML/NGL are still emitted ahead of the packs that consume them — §2), so it too is
DSREAD-first everywhere.

This matches the prior trace `Tensile/Components/SHADOW_NGL_SCHEDULER_TRACE.md`
(§3, §5), which independently localized the same `isBarrier` split
(`:1364-1366`, `:2009-2015`) plus cross-body pack-pre carry. The new contribution
here is the strictly register-agnostic class table across all three schedules,
including cms=0 (which that trace did not cover), and the feeding sub-test that
isolates SHADOW NGL as the unique inversion cell.

---

## §6 What this means for the 16 failures / uvrl

The SHADOW NGL pack-before-ds_read inversion is a property of the **synthetic
reference schedule only** — it is never emitted as real assembly. Per
`SHADOW_NGL_SCHEDULER_TRACE.md §4/§6`, the inversion is hardware-correct software
pipelining: the head packs read the previous body's already-packed output (live,
un-clobbered data), and the late ds_read feeds the *next* body's consumer. The
inversion does NOT by itself produce a validator failure at the NGL level — the
NGL pack-mfma edge is byte-key-identical between SHADOW and cms and cancels in
set-diff.

The 16 `EdgeRoutedDifferentlyFailure`s have their consumer in **NLL**, where both
SHADOW and cms root at the same 3rd (NGL) ds_read = the same physical LDS
fragment. They are false positives of the flat-register latest-writer model, which
treats the NGL pack-cvt and the NGL ds_read as distinct producers of the rotating
register buffer. That is the `uvrl` (validator-modeling) fix — NOT a scheduler
fix. This class-ordering result reinforces that conclusion: the *only* place the
SHADOW reference behaves differently from a real kernel is the NGL head-flush,
which is benign pipelining; nothing here argues for changing the default
scheduler. The corrective action lives in the validator's unrolled latest-writer
model (uvrl), which must treat a register-buffer pack-output and its feeding
ds_read as the same logical fragment when they target the same physical LDS slot
across the cross-body prefetch boundary.

---

## §7 Bead

No new bead filed. The scheduler behavior is correct (benign cross-body
pipelining); the actionable item is the already-tracked `uvrl` validator-modeling
fix, and the mechanism is already documented in
`Tensile/Components/SHADOW_NGL_SCHEDULER_TRACE.md`. This document adds the
register-agnostic class table (incl. cms=0) and corrects the prior "N/A" framing
for cms=0 ML/NGL; it does not surface new work requiring a blocker. (Task #409
tracks this investigation.)
