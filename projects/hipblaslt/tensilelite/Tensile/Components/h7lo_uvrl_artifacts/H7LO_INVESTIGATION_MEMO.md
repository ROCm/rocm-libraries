# H7LO investigation memo

Bead `rocm-libraries-h7lo`: 16 `EdgeRoutedDifferentlyFailure`s from
`compare_graphs` on the BPG#11 TF32 4x4 TN canonical fixture that persist after
the hxcx capture fix.

Kernel: BPG#11 TF32 4x4 TN canonical, gfx950, `UseCustomMainLoopSchedule=1`,
`UsePLRPack=True`, `UseMFMAF32XEmulation=True`, PGR=2 (= `CANONICAL_KERNEL_CONFIG`).

All findings below are produced by `hxcx_artifacts/_h7lo_probe{,2..6}.py` (left in
the artifacts dir; re-runnable from the worktree).

---

## §1 The 16 failures, grouped + analyzed

Failure list is in `compare_graphs_failures.txt`. Grouping by side, consumer, and
the cited `('v', K)` byte_key:

### A-side (consumer category `PackA3`, NGL)

| consumer | byte_keys (`v`) | subj producer ordinals | ref target |
|---|---|---|---|
| `PackA3[11]` | 14, 13, 11, 12 | `PackA3[16,17,19,18]` | `PackA0[9,3,1,2]` @ idx=-1 (PRO) |
| `PackA3[9]`  | 5, 4, 6, 3      | `PackA3[13,14,12,15]` | **no prior writer** |

### B-side (consumer category `PackB3`, NGL) — symmetric

| consumer | byte_keys (`v`) | subj producer ordinals | ref target |
|---|---|---|---|
| `PackB3[11]` | 46, 43, 44, 45 | `PackB3[16,19,18,17]` | `PackB0[9,1,2,3]` @ idx=-1 (PRO) |
| `PackB3[9]`  | 37, 36, 38, 35 | `PackB3[14,12,15,…]` | **no prior writer** |

**Register range.** The A-side bytes are `v3..v6` and `v11..v14`; the B-side
bytes are `v35..v38` and `v43..v46`. In this kernel `vgprValuA_X0_I0` resolves so
that `X0_I0+15 → ('v',14)`, `X0_I0+12 → ('v',11)`, etc. — i.e. **the high four
bytes of each 8-wide pack-buffer half** (`X0_I0+12..15` ⇒ `v11..v14`). These are
exactly the bytes the "pack-final" `v_cvt_pk_bf16_f32` CVTs write (the second of
the two `__TF32` halves). The 8 "routes through PRO" cases are the `Pack*[11]`
consumers; the 8 "no prior writer" cases are the `Pack*[9]` consumers. The split
is structural, not random.

**Same root cause for all 16.** Every failure is the same shape: a subj edge
whose producer is a `Pack{A,B}3` CVT in NGL and whose consumer is a
`Pack{A,B}3` MFMA, on a single high-byte of the rotating pack buffer.

---

## §2 Concrete walk of Failure 0

Failure 0 reads: *subj consumer `PackA3[11]` @ idx=43 reads from subj producer
`PackA3[16]` @ idx=46 at `('v',14)`, but reference routes through `PackA0[9]`
@ idx=-1 (PRO body).*

### The actual subj edge (`_h7lo_probe4/5`)

The edge `compare_graphs` flagged is **not** intra-NGL. It is cross-iteration:

```
subj producer: body=NGL  cat=PackA3  unrolled_pos=599
  v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]
subj consumer: body=NLL  cat=PackA3  unrolled_pos=737   (iter_delta = 1)
  v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:+3], v[74:75],
                            v[vgprValuA_X0_I0+10:+1], v[vgprValuA_X0_I0+12:+3]
subj edge_key:
  (None, 0, (('v',14),), (('v',14),), 'raw_intrawave', (3,), src=0, sink=2)
```

The producer at pos=599 is genuinely the closest-prior writer of `('v',14)`
before the NLL consumer at pos=737 in the subj timeline (verified in `probe6`).
So the **subj side is internally consistent** — this is a legitimate per-byte
RAW edge in the CMS schedule.

### Why ref reports "routes through PackA0[9] @ idx=-1"

`diagnose_extra_edge` (CMSValidator.py:4304) resolves the ref-side consumer **by
identity**:

```python
ref_cons_node = next((n for n in ref_graph.nodes if n.identity == cons.identity), None)
```

`GraphNode.identity` is `(canonical_render, source_module_id, emission_ordinal)`
and is **body-blind by construction** (CMSValidator.py:919). For this consumer
the tuple is `(<the X0_I0+12 mfma render>, None, 0)`.

That identity tuple is **NOT unique** — the same canonical MFMA appears once in
*every* body. `probe3` confirms the identity matches **5 nodes** in each graph:

```
ref nodes with this identity (node-iteration order):
  body=PRO   pos=26     <-- next() returns THIS one
  body=ML-1  pos=58
  body=ML    pos=250
  body=NGL   pos=464
  body=NLL   pos=636    <-- the instance that actually corresponds to subj's NLL consumer
```

`next(...)` returns the **PRO instance at pos=26** (lowest position, first in node
order). `diagnose_extra_edge` then sets `cons_pos = 26` and searches ref's
`byte_key_writers[('v',14)]` for the closest writer strictly before pos=26:

```
ref writers of v14 before pos=26:  pos=5 (PRO PackA0 mfma), pos=16 (PRO PackA0 cvt)
closest-prior = pos=16  body=PRO  cat=PackA0   ← this is "PackA0[9]" in the failure text
```

Identity of that prologue writer (`<X0_I0+4 cvt>, None, 0`) ≠ subj producer
identity (`<X0_I0+15 cvt>, None, 0`), so the `all_match` test fails and an
`EdgeRoutedDifferentlyFailure` is emitted naming `PackA0[9] @ idx=-1`.

The "no prior writer" variant (Failures 2,3,4,8,9,10,12,15) is the identical
mechanism where the wrongly-resolved PRO consumer has *no* v14-class writer
before its low position — so `priors` is empty and `ref_producer=None`.

So the literal text of the failure ("routes through PackA0[9] @ idx=-1") is an
**artifact of resolving the wrong body-instance of the consumer**. But that is
only the *messaging* artifact. The substantive question (does the edge belong in
the set-diff at all?) is answered in §4.

---

## §3 SHADOW vs CMS prologue content comparison

**The prologue is the SAME object on both sides.** `probe1`:

```
cms.prologue is default.prologue ?  True
```

`build_cms_four_part_capture` assigns `prologue=default_capture.prologue`
verbatim (ScheduleCapture.py:2859); the prologue is emitted once in `kernelBody`
between `setupNewTile` and `openLoop` (KernelWriter.py:5947-6025) from the
post-`_interleavePackAB` stream and shared.

Per-body node counts are **identical** across the two graphs:

```
subj body counts:  {ML-1:184, ML:184, NGL:164, NLL:136, PRO:40}
ref  body counts:  {ML-1:184, ML:184, NGL:164, NLL:136, PRO:40}
PRO PackA0 = 20 on both sides; PRO PackB0 = 20 on both sides.
```

**This rules out hypothesis (a) outright.** There are no "ref-side phantom
prologue writes." Both captures observe the identical 40-leaf prologue with the
identical 20 `PackA0` + 20 `PackB0` writers. The ref side's PRO `PackA0[9]` exists
on the subj side too (subj `byte_key_writers[('v',14)]` lists `pos=5` and `pos=16`
PRO PackA0 entries, same as ref). The prologue is not where the divergence lives.

The one place the two captures *do* tag differently: the NGL pack chain is tagged
`PackA3`/`PackB3` on the subj (CMS) side but `PackA0`/`PackB0` on the ref
(SHADOW) side (`probe1`: `subj NGL → {PackA0:20, PackA3:20}` vs
`ref NGL → {PackA0:20, PackA1:20}`). That category-label difference is cosmetic
for byte-key routing (byte_keys are physical regs) but it is a *symptom* of the
real difference in §4.

---

## §4 byte_key_writers list per side for `('v',14)`

Full v14 writer timelines (`probe6`), annotated with the NLL consumer position:

### subj (CMS) — NLL consumer at pos=737

```
pos=  5 PRO  PackA0  (mfma X0_I0+4)
pos= 16 PRO  PackA0  (cvt X0_I0+4)
pos=197 ML-1 LRA3    ds_read X0_I0+12:+15
pos=219 ML-1 PackA3  (mfma X0_I0+12)
pos=229 ML-1 PackA3  (cvt X0_I0+15)
pos=391 ML   LRA3    ds_read X0_I0+12:+15
pos=413 ML   PackA3  (mfma X0_I0+12)
pos=423 ML   PackA3  (cvt X0_I0+15)
pos=571 NGL  LRA3    ds_read X0_I0+12:+15        <-- ds_read FIRST
pos=591 NGL  PackA3  (mfma X0_I0+12)
pos=599 NGL  PackA3  (cvt X0_I0+15)              <-- pack cvt AFTER ds_read; closest-prior of v14 before 737
pos=737 NLL  PackA3  consumer (mfma X0_I0+12)
pos=745 NLL  PackA3  (cvt X0_I0+15)
=> subj closest-prior writer of v14 before 737: pos=599 (NGL PackA3 cvt)
```

### ref (SHADOW default) — NLL consumer at pos=636

```
pos=  5 PRO  PackA0  (mfma X0_I0+4)
pos= 16 PRO  PackA0  (cvt X0_I0+4)
pos= 58 ML-1 PackA3  (mfma X0_I0+12)
pos= 59 ML-1 PackA3  (cvt X0_I0+15)
pos=225 ML-1 LRA3    ds_read X0_I0+12:+15
pos=250 ML   PackA3  (mfma X0_I0+12)
pos=251 ML   PackA3  (cvt X0_I0+15)
pos=417 ML   LRA3    ds_read X0_I0+12:+15
pos=464 NGL  PackA0  (mfma X0_I0+12)             <-- pack chain FIRST
pos=477 NGL  PackA0  (cvt X0_I0+15)
pos=597 NGL  LRA3    ds_read X0_I0+12:+15        <-- ds_read AFTER pack chain; clobbers v14
pos=636 NLL  PackA0  consumer (mfma X0_I0+12)
pos=646 NLL  PackA0  (cvt X0_I0+15)
=> ref closest-prior writer of v14 before 636: pos=597 (NGL LRA3 ds_read)
```

The decisive contrast — **the two schedulers order the NGL ds_read vs the NGL
pack chain oppositely**:

* **subj (CMS):** NGL does `ds_read(571) → pack mfma(591) → pack cvt(599)`.
  The pack cvt is the last v14 writer before the NLL consumer ⇒ subj forms a
  single-byte edge `PackA3 cvt → consumer` `(('v',14),)`.
* **ref (default):** NGL does `pack mfma(464) → pack cvt(477) → ds_read(597)`.
  The `ds_read_b128` is the last v14 writer before the NLL consumer ⇒ ref forms
  a 4-byte edge `LRA3 ds_read → consumer` `(('v',11),('v',12),('v',13),('v',14))`.

`probe5` confirms the subj edge `(None,0,(('v',14),),(('v',14),),'raw_intrawave',
(3,),0,2)` is **absent** from `ref.edge_keys()`, while subj's three other v14
edges to the *intra-body* consumers (ML-1/ML/NGL) all match ref. Only the
NGL-pack→NLL-consumer byte is routed differently.

This is a **genuine per-byte latest-writer divergence between the two
schedules**, not a byte-key computation asymmetry — both sides use the same
shared `name_to_idx` (KernelWriter.py:6228-6246), so `('v',14)` means the same
physical register on both sides. **Hypothesis (c) is ruled out.**

---

## §5 Verdict

**Fourth option — and it is twofold.**

1. **Messaging artifact (the literal failure text).** `diagnose_extra_edge`'s
   `next((n for n in ref_graph.nodes if n.identity == cons.identity))` resolves
   the ref consumer to the **wrong body-instance** (the PRO copy at pos=26)
   because `GraphNode.identity` is body-blind and the canonical MFMA recurs in
   all 5 bodies. That is why the message says "routes through PackA0 @ idx=-1
   (PRO)" / "no prior writer." The correct ref instance is the NLL copy at
   pos=636, whose real v14 source is the NGL `ds_read` at pos=597. The current
   `next()` lookup re-introduces exactly the body sensitivity the design
   (DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3, CMSValidator.py:3771-3774) declared
   "intentionally absent." So the diagnostic is **misattributed**.

2. **Real routing divergence (the substance).** Even after fixing the
   consumer-instance resolution, the edge still does not cancel: subj routes the
   single high-byte `v14` of the NLL consumer through the **NGL pack CVT**, while
   ref routes the whole 4-byte read through the **NGL `ds_read`**, because the two
   schedulers place the NGL pack chain on opposite sides of the NGL `ds_read`.
   This is the same family as hypothesis **(b)** — a real ordering difference in
   the schedules — but it is rooted in a known structural weakness, not a
   miscompiled kernel: the validator's **per-byte single-timeline latest-writer
   model treats the rotating pack buffer (`X0_I0`) as one physical register
   reused across iterations**, so the cross-iteration NGL→NLL read picks up
   whichever of {ds_read, pack-cvt} the scheduler happened to emit last. The CMS
   schedule emitting the pack chain after the ds_read is a *legitimate scheduling
   choice* (the pack cvt re-derives the same logical value the ds_read loaded);
   the default scheduler simply emits them in the other order. Neither is wrong
   on hardware — both produce the correct value into the buffer before the
   consumer — but the byte-level latest-writer abstraction cannot see that they
   are *equivalent producers* and so reports a spurious topology difference.

The net: **this is not a CMS kernel bug and not a phantom-write capture bug.** It
is (1) a consumer-identity-resolution defect in `diagnose_extra_edge` that
mislabels the failure, layered on (2) a real-but-benign schedule ordering
difference that the per-byte latest-writer model over-reports because it cannot
recognize the rotating-buffer pack-CVT and the ds_read as interchangeable
producers of the same logical value.

---

## §6 Confidence + what would resolve uncertainty

**Confidence: HIGH** for the mechanism (§2, §4 are direct dumps of the live
graphs: identity-collision across 5 bodies, the exact subj edge_key, its absence
from ref, and the opposite NGL ds_read/pack ordering on each side — all
reproduced by `_h7lo_probe{2..6}.py`).

**MEDIUM** confidence on the *normative* call in §5 part 2 — i.e. whether the
default scheduler's "pack chain before ds_read" or CMS's "pack chain after
ds_read" is the intended canonical order for NGL under `UsePLRPack` rotation.
Resolving that requires confirming the architectural intent of the NGL pack
schedule (does NGL deliberately defer the pack chain to overlap the prefetched
ds_read of the next tile, or is the SHADOW default order authoritative?). The
person who owns `customMainLoopSchedule`'s NGL emission ordering can confirm in
one read. Either way, the conclusion that the *byte-level* abstraction
over-reports the equivalence stands.

---

## §7 What needs fixing and where

This is option (1)+(2), so both a validator-correctness fix and a validator-
modeling decision — **no kernel-writer change**.

### Fix A (correctness, required): consumer-instance resolution in `diagnose_extra_edge`

`CMSValidator.py:4362-4366` must not resolve the ref consumer with a body-blind
`next(... identity ...)`. It needs the ref instance that **corresponds to the
subj consumer's body/iteration**, not the first identity match (which is always
the lowest-position PRO copy when the instruction recurs across bodies). Options,
in order of principle:

* Match on `(identity, body_label, iter_index)` — the tuple that uniquely picks
  the NLL instance. This requires the comparison to admit a body-aware key for
  *consumer resolution* even though edge_keys stay body-blind; that is
  consistent, because the question "which ref node is this subj consumer" is
  inherently positional, unlike "do these two edges describe the same physical
  byte flow."
* Equivalently, carry the consumer's body/iter through the extra-edge into
  `diagnose_extra_edge` and select the ref node by it.

Without Fix A every cross-body recurring-instruction extra-edge will be
misattributed to PRO / "no prior writer", masking what the real divergence is —
the same defect would mislabel genuine bugs too.

### Fix B (modeling decision, the substance): rotating-buffer equivalent producers

After Fix A the edge still won't cancel because subj and ref legitimately route
the byte through different (but value-equivalent) producers (NGL pack-CVT vs NGL
ds_read). The principled resolution is to make the comparison recognize that,
for the rotating pack buffer, **a pack-CVT that re-derives byte K and the
ds_read that loaded byte K are interchangeable last-writers for a
cross-iteration consumer** — i.e. the validator's per-byte timeline must model
the pack rotation, not treat `X0_I0+15` as a single flat physical register whose
"latest writer" flips with scheduler order. This is the load-bearing design
question (it is the same single-timeline assumption flagged generally in
DEFAULT_SCHEDULER_REFERENCE_DESIGN.md). If instead the NGL ordering itself is
considered normative, the fix is to align CMS's NGL pack/ds_read order with the
default — but that is a kernel-writer/schedule change and would need the
ownership confirmation noted in §6, and it contradicts the "differ ONLY in
scheduling" premise (the schedules are *allowed* to differ in ordering; the
validator must tolerate value-equivalent reorderings of the rotating buffer).

Per the standing "no tactical fixes" rule: the correct sequence is Fix A first
(stop misattributing), then Fix B as the principled modeling change (teach the
per-byte resolver about pack-buffer rotation equivalence) rather than
special-casing PRO-routed pack edges out of the set-diff.
