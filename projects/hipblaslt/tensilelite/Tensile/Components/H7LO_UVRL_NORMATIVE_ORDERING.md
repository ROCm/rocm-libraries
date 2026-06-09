# NGL ds_read / pack ordering — the normative question behind rocm-libraries-uvrl

This doc captures the one design question that gates `rocm-libraries-uvrl`
(the load-bearing half of the original h7lo investigation). It is written so a
reviewer who has never seen the C-chain can reproduce the issue, read the exact
diverging instructions in two committed listings, and answer the question in one
sitting.

Everything below was **re-derived from scratch against the current code state**
(post-Fix-A, post-hxcx). Where it corrects the prior memo, that is called out in
the "What changed since the prior memo" section at the end.

Related beads:
- `rocm-libraries-h7lo` (CLOSED, committed `e2d85a76b0d8`) — **Fix A**, the
  *messaging* half. DONE. See "Fix A" below.
- `rocm-libraries-uvrl` (OPEN, P0) — **Fix B**, the *substantive* routing
  divergence described here. Blocks `rocm-libraries-r62g` (Phase 3 go/no-go gate).
- `rocm-libraries-hxcx` (resolved earlier, committed `5267acc76bed`) — fixed the
  24 `TimingTooCloseFailure`s (a prologue capture-ordering bug). Separate and DONE;
  not re-litigated here. Its only relevance: `validator_failures.txt` now has 0
  entries, so the only remaining failures are the 16 below.

---

## Fix A (rocm-libraries-h7lo) — DONE and committed (`e2d85a76b0d8`)

Fix A was the **messaging-correctness** half. `diagnose_extra_edge` and
`diagnose_missing_edge` resolved reference/subject nodes via bare-identity
`next()` lookups. Bare identity is body-blind (`emission_ordinal` is assigned
per-body, so a canonical instruction recurring across all five bodies gets
ordinal 0 in *each*), so the lookups returned the lowest-position PRO-body copy
instead of the correct NGL/NLL instance. That made the 16 messages cite a
PRO-body `PackA0[...] @ idx=-1` instead of the real NGL/NLL reference instruction.

Three sites now match on `(identity, body_label, iter_index)`. Verified in the
current tree:

| Site | File / line | What it resolves |
|---|---|---|
| `diagnose_missing_edge` consumer | `Tensile/Components/CMSValidator.py:3950` | the subj consumer node |
| `diagnose_missing_edge` SCC orig_producer | `Tensile/Components/CMSValidator.py:4142` | the ref-side original SCC producer |
| `diagnose_extra_edge` ref-consumer | `Tensile/Components/CMSValidator.py:4387` | the ref consumer's position |

All three use the three-field predicate
(`n.identity == ... and n.body_label == ... and n.iter_index == ...`). The
closest-prior-writer logic (byte-key-writers walk + `unrolled_position`
comparison) is unchanged.

Observable consequence in the current artifacts: every one of the 16 messages now
cites an **NGL/NLL-era reference producer** (`LRA3[...]` / `LRB3[...]` "of next
iteration"), not the old PRO-body `PackA0`. See
`Tensile/Components/h7lo_uvrl_artifacts/compare_graphs_failures.txt`.

Fix A did **not** make the 16 failures cancel. That is Fix B (`uvrl`), below.

---

## Committed artifacts (stable; every line number in this doc resolves into these)

A frozen snapshot is committed alongside this doc so the citations never go stale:

- `Tensile/Components/h7lo_uvrl_artifacts/kernel.s` — the CMS-emitted assembly
  (~7,200 lines). **SUBJECT (CMS) schedule only.** The SHADOW schedule is not
  emitted as assembly text.
- `Tensile/Components/h7lo_uvrl_artifacts/cms_capture_listing.txt` — the CMS
  (subject) capture as a line-numbered, body-ordered instruction listing.
- `Tensile/Components/h7lo_uvrl_artifacts/shadow_capture_listing.txt` — the
  SHADOW (reference / default-codegen) capture, same format. **This is how you
  read the reference schedule** — it has no `.s` file.
- `Tensile/Components/h7lo_uvrl_artifacts/compare_graphs_failures.txt` — the 16
  `EdgeRoutedDifferentlyFailure`s.
- `Tensile/Components/h7lo_uvrl_artifacts/H7LO_INVESTIGATION_MEMO.md` — prior
  investigation memo. **Treat as stale context** — its messages and some line
  numbers predate Fix A; this doc supersedes it.

Listing columns: `<file_line> | body | body_idx | mfma_index | seq | render`.
Body order is the unrolled timeline: `PRO -> ML_PREV(iter0) -> ML(iter1) -> NGL -> NLL`.

---

## How to reproduce (exact commands)

From the worktree, with rocisa freshly built (a cached `build_tmp` binary
predates the C-chain and gives stale results):

```bash
cd /home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite
pip install -e ./rocisa
python Tensile/Tests/unit/_dump_hxcx_assembly.py
```

That regenerates `hxcx_artifacts/` (gitignored build output — the committed copy
above is the permanent record):
- `hxcx_artifacts/kernel.s` — full CMS-emitted assembly.
- `hxcx_artifacts/cms_capture_listing.txt`, `hxcx_artifacts/shadow_capture_listing.txt`.
- `hxcx_artifacts/compare_graphs_failures.txt` — the 16 failures.
- `hxcx_artifacts/validator_failures.txt` — 0 entries (hxcx confirmed clean).

```bash
cat hxcx_artifacts/compare_graphs_failures.txt   # 16 EdgeRoutedDifferentlyFailures
```

Two probes back the per-byte timeline used below:
- `hxcx_artifacts/_h7lo_probe6.py` — **authoritative**; builds the actual
  validator dataflow graph and dumps `byte_key_writers[('v',14)]` with
  unrolled positions. This is the source of truth for the byte-key mapping
  (`('v',14)` = `ValuA_X0_I0+15`) and the latest-writer pairing.
- `hxcx_artifacts/_uvrl_byte_probe.py` — a fast listing-derived cross-check (pure
  text, no kernel build); its `('v', …)` label is naive (ignore it), but its
  unrolled positions match `_h7lo_probe6.py` exactly.

---

## The fixture

`CANONICAL_KERNEL_CONFIG` in `_dump_hxcx_assembly.py`: BPG#11 TF32 4x4 TN,
gfx950, MI=[16,16,32,1,1,4,4,2,2], MacroTile 128x128, DepthU 32,
`PrefetchGlobalRead=2`, `PrefetchLocalRead=1`, `UseCustomMainLoopSchedule=1`,
`UsePLRPack=True`, `UseMFMAF32XEmulation=True`.

CMS schedule dispatched:
`Tensile/Components/CustomSchedule/gfx950/_128x128x32_TF32.py`,
`_get_schedule_128x128x32_TF32`, the `('TN', False, 1)` branch
(`isTN(kernel) and not useLDSTr and TLDS==1`, the `if` at line 56).

The SHADOW capture is `writer._last_default_capture` — the **default-codegen**
schedule, which does **not** use the CMS slot arrays. So this is genuinely a
CMS-vs-default comparison.

---

## The 16 failures, grouped (from `compare_graphs_failures.txt`)

All 16 are `EdgeRoutedDifferentlyFailure`. They split cleanly into two mirror
groups:

**Group A — A-side (8 failures: 2,3,4,5,9,10,12,13).** Subject consumer
`PackA3[9]` or `PackA3[11] @ idx=43`; subject producer `PackA3[12..19] @ idx=44..47`;
reference routes through `LRA3[1] @ idx=39` or `LRA3[3] @ idx=43` "(of next iteration)".
Byte keys `('v', 3..6)` and `('v', 11..14)`.

**Group B — B-side (8 failures: 0,1,6,7,8,11,14,15).** Subject consumer
`PackB3[9]` or `PackB3[11] @ idx=32`; subject producer `PackB3[12..19] @ idx=33..36`;
reference routes through `LRB3[1] @ idx=41` or `LRB3[3] @ idx=45` "(of next iteration)".
Byte keys `('v', 35..38)` and `('v', 43..46)`.

The shape is identical in both groups: **subject says the producer is a `PackA3`/
`PackB3` pack instruction; reference says the producer is the `LRA3`/`LRB3`
`ds_read`.** That is the divergence.

> Byte-key namespace: `('v', N)` is the **numeric byte-key** the validator assigns
> via `_byte_keys_for_resource` (`ScheduleCapture.py:1598`), which resolves
> `ValuA_X0_I0` through the body-local `name_to_idx` base. On this kernel it maps
> `('v', 14)` → `ValuA_X0_I0+15`, `('v', 13)` → `+14`, `('v', 12)` → `+13`,
> `('v', 11)` → `+12` (i.e. `('v', N)` = `X0_I0+(N+1)`). The mapping was confirmed
> by building the actual validator graph (probe `_h7lo_probe6.py`): the
> `('v', 14)` writer is the cvt of `v[vgprValuA_X0_I0+15]` with
> `intra_operand_byte_offset = 3`. Do **not** read `('v', N)` as `X0_I0+N`.

---

## The one concrete byte: `('v', 14)` = `ValuA_X0_I0+15` (failure 3)

Register `ValuA_X0_I0+15` is a **rotating scratch register**: within every
loop-body it is written by three different kinds of instruction:
1. the `ds_read` that loads the next tile's A fragment (the `LRA3` local read;
   writes the `+12:+15` span, covers `+15`),
2. the `v_mfma_f32_4x4x4_16b_bf16` pack-mfma (writes the `+12:+15` span, covers `+15`),
3. the `v_cvt_pk_bf16_f32` pack-cvt (writes `+15` directly).

All three deposit the same logical value into the rotating buffer. The schedules
order them oppositely.

The failure-3 edge the validator actually flags is **cross-iteration**
(verified by `_h7lo_probe6.py` against the live graph): subject producer is the
NGL pack-cvt at unrolled position 599; consumer is the NLL pack-mfma at position
737 (`iter_delta = 1`). Reference's producer for the same consumer-byte is the NGL
`ds_read` (LRA3) at position 597. The relevant in-body ordering that creates this
is the NGL body, shown below.

### Subject (CMS) order — `cms_capture_listing.txt`, NGL body

| listing line | body / mfma_index | instruction (render) | role |
|---|---|---|---|
| `cms_capture_listing.txt:584` | NGL / mfma 37 | `ds_read_b128 v[vgprValuA_X0_I0+12:+3], ...` | **ds_read (LRA3) — FIRST** |
| `cms_capture_listing.txt:604` | NGL / mfma 43 | `v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:+3], ...` | pack-mfma (consumer `PackA3[11]@idx43`) |
| `cms_capture_listing.txt:612` | NGL / mfma 46 | `v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], ...` | pack-cvt (the `('v',14)` producer `PackA3@idx46`) |

Observe: the `ds_read` comes **before** the pack chain. Confirmed in the real
emitted assembly. The NGL body is emitted as the OptNLL `MAINLOOP` macro (macro
body defined `kernel.s:1889-2237`, instantiated `MAINLOOP 0,0,1,1,0` at
`kernel.s:2253`). The relevant A-fragment instructions inside that macro body:

| `kernel.s` line | instruction |
|---|---|
| `kernel.s:2175` | `ds_read_b128 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], v[vgprLocalReadAddrA+0] offset:192` |
| `kernel.s:2208` | `v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:vgprValuA_X0_I0+12+3], ...` |
| `kernel.s:2225` | `v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], v[vgprValuA_X0_I0+14], v[vgprValuA_X0_I0+15]` |

ds_read (`:2175`) precedes pack-mfma (`:2208`) precedes pack-cvt (`:2225`) — the
captured order (`cms_capture_listing.txt:584 -> 604 -> 612`) matches the emitted
order exactly. The CMS NGL capture is **faithful to the emitted assembly**
(verified directly; this was the explicit test of the "are we capturing NGL
properly?" hypothesis — answer: yes).

Note: do NOT confuse these NGL macro-body lines with the visually identical
ds_read/mfma/cvt sequence at `kernel.s:2665/3000/3008` — those are under
`label_TailLoopBeginL` (the tail loop, starts `kernel.s:2659`), a different body
that is not where the 16 failures live.

### Reference (default) order — `shadow_capture_listing.txt`, NGL body

| listing line | body / mfma_index | instruction (render) | role |
|---|---|---|---|
| `shadow_capture_listing.txt:477` | NGL / mfma 1 | `v_mfma_f32_4x4x4_16b_bf16 v[vgprValuA_X0_I0+12:+3], ...` | pack-mfma — FIRST |
| `shadow_capture_listing.txt:490` | NGL / mfma 2 | `v_cvt_pk_bf16_f32 v[vgprValuA_X0_I0+15], ...` | pack-cvt |
| `shadow_capture_listing.txt:610` | NGL / mfma 43 | `ds_read_b128 v[vgprValuA_X0_I0+12:+3], ...` | **ds_read (LRA3) — LAST** |

Observe: the pack chain comes **first** and the `ds_read` lands **last**.

**The two schedules order `ds_read` vs the pack chain exactly oppositely.** This
holds in every body, not just NGL — see the full per-body trace below.

### Side-by-side (per-body order of the three writers of `ValuA_X0_I0+15`)

From the live validator graph (`_h7lo_probe6.py`, `byte_key_writers[('v',14)]`),
plus the committed-listing line numbers for the same instructions:

```
CMS (subject):                                 SHADOW (reference / default):
  PRO     pack-mfma  pos=5                        PRO     pack-mfma  pos=5
          pack-cvt   pos=16                               pack-cvt   pos=16
  ML(-1)  ds_read    pos~                          ML-1    pack-mfma  pos=58
          pack-mfma  pos~                                  pack-cvt   pos=59
          pack-cvt   pos~                                  ds_read    pos=225
  ML      ds_read    pos=391                       ML      pack-mfma  pos=250
          pack-mfma  pos=413                               pack-cvt   pos=251
          pack-cvt   pos=423                               ds_read    pos=417
  NGL     ds_read    pos=571 (l.584) <-- FIRST     NGL     pack-mfma  pos=464 (l.477) <-- FIRST
          pack-mfma  pos=591 (l.604)                       pack-cvt   pos=477 (l.490)
          pack-cvt   pos=599 (l.612)                       ds_read    pos=597 (l.610) <-- LAST
  NLL     pack-mfma  pos=737 (consumer)            NLL     pack-mfma  pos=636 (consumer)
          pack-cvt   pos=745                               pack-cvt   pos=646
```

(Subject's ML-1 copies are PackA0-categorized in the prologue and omitted for
clarity; the structural point — CMS puts `ds_read` before the pack chain, default
puts it after — is identical in every steady-state body.) The divergence drives
the failures in the NGL/NLL region.

### The latest-writer consequence (why the 16 failures fire)

The validator builds a per-byte **latest-writer** dataflow graph and set-diffs
the edges. For byte `('v', 14)` at the NLL pack-mfma consumer (the failure-3 edge,
from `_h7lo_probe6.py`):

- **CMS:** consumer at unrolled position 737 (NLL). Closest-prior writer of
  `('v', 14)` is the **NGL pack-cvt at position 599** (`cms_capture_listing.txt:612`)
  — because CMS's in-body `ds_read` (`:584`, pos 571) and pack-mfma (`:604`,
  pos 591) both precede the pack-cvt, the pack-cvt is the latest writer. Subject
  edge: producer = `PackA3` cvt, `read_bk=(('v',14),)`, `intra=(3,)`.
- **Reference:** same logical consumer at position 636 (NLL). Closest-prior writer
  of `('v', 14)` is the **NGL `ds_read` (LRA3) at position 597**
  (`shadow_capture_listing.txt:610`) — because default puts the `ds_read` *after*
  the pack chain, the `ds_read` is the latest writer. Reference edge (verified):
  producer = `LRA3` ds_read, `read=(('v',11),('v',12),('v',13),('v',14))`,
  `intra=(0,1,2,3)`.

Subject's latest writer of the byte is a **pack instruction**; reference's is the
**`ds_read`**. Same physical register, same logical value, different producer node
→ one `EdgeRoutedDifferentlyFailure` per affected byte. The model treats
`ValuA_X0_I0+15` as one flat register and cannot see the `ds_read` and the
pack-cvt as **value-equivalent producers** of the rotating buffer.

---

## The normative-ordering question (what a reviewer must decide)

**Precisely:** in the NGL body, is the CMS ordering
(`ds_read` (LRA3/LRB3) → pack chain) the *intended* one, or is the default's
(pack chain → `ds_read`) authoritative?

### Where the CMS ordering is controlled (schedule source)

`Tensile/Components/CustomSchedule/gfx950/_128x128x32_TF32.py`,
`_get_schedule_128x128x32_TF32`, the `('TN', False, 1)` branch. These arrays are
**mfma-slot placements** — each integer is the `mfma_index` (0..47, the same
`mfma_index` column in the listings) at which that op is scheduled in the body:

| schedule array | file / line | slots | what it places |
|---|---|---|---|
| `lra3` | `_128x128x32_TF32.py:87` | `[36,36,37,37]` | A-side iter-3 `ds_read` (LRA3) at mfma 36–37 |
| `pack_a3` | `_128x128x32_TF32.py:90` | `[39,39,40,40, 43,43, 44,44,45,45, 41,41,42,42, 43,43, 46,46,47,47]` | A-side pack chain (incl. the `43` pack-mfma and `46` pack-cvt the failures cite) |
| `lrb3` | `_128x128x32_TF32.py:82` | `[24,24,25,25]` | B-side iter-3 `ds_read` (LRB3) at mfma 24–25 |
| `pack_b3` | `_128x128x32_TF32.py:85` | `[28,28,29,29, 32,32, 33,33,34,34, 30,30,31,31, 32,32, 35,35,36,36]` | B-side pack chain |

These arrays put `lra3` (slots 36–37) and `lrb3` (slots 24–25) **before** their
respective pack chains (`pack_a3` starts at 39, `pack_b3` at 28). That is exactly
the CMS NGL order observed above: `ds_read` first (mfma 36/37 for A), pack-mfma at
43, pack-cvt at 46. The CMS schedule **deliberately hoists the next-iteration
local read early** so its latency hides behind the pack/mfma work — a legitimate
software-pipelining decision encoded directly in `lra3`/`lrb3`. The
default-codegen scheduler does not use these arrays and places the local read
last.

The reviewer compares this schedule-side artifact (the `lra3`/`pack_a3`/`lrb3`/
`pack_b3` slot arrays) against the emitted order in `kernel.s` and the two
listings, and decides:

- **(B1) Both orderings legitimate (expected).** Both deposit the same logical
  value into the rotating buffer before the consumer. Then the validator's
  per-byte model is wrong to flag this, and `uvrl` is a **validator-modeling
  change**: teach the latest-writer model that, for the rotating pack buffer, a
  pack-cvt that re-derives byte K and the `ds_read` that loads byte K are
  interchangeable producers. No kernel change. Consistent with the "differ ONLY
  in scheduling" contract (`DEFAULT_SCHEDULER_REFERENCE_DESIGN.md` §3).

- **(B2) One ordering is wrong.** Then it is a real scheduler bug and the fix is
  in the schedule (the `lra3`/`pack_a3` slot arrays), not the validator. This
  contradicts the "differ only in scheduling" premise, so it is the less likely
  answer — but it must be ruled out by whoever owns the NGL pack-emission ordering.

`uvrl` cannot be planned until B1 vs B2 is answered, because the two answers fix
different layers. Whoever owns `_get_schedule_128x128x32_TF32`'s NGL ordering can
confirm which case applies in one read of the `lra3`/`pack_a3` arrays above.

---

## Verdict (Fix B diagnosis, against current code)

**Confirmed.** The prior conclusion stands against the current tree:

1. **Fix A (messaging)** — DONE, committed `e2d85a76b0d8`. The three sites resolve
   nodes by `(identity, body_label, iter_index)`; the 16 messages now cite
   NGL/NLL-era `LRA3`/`LRB3` reference producers.
2. **Fix B (`uvrl`)** — open. The per-byte latest-writer model treats the rotating
   pack buffer (`ValuA/B_X0_I0+k`) as one flat register and cannot see the
   `ds_read` and the pack-cvt as value-equivalent producers. CMS orders
   `ds_read` → pack (`lra3` hoisted to mfma 36–37 ahead of `pack_a3`); default
   orders pack → `ds_read`. Legitimate value-equivalent reorders therefore get
   flagged as routing divergences — the 16 `EdgeRoutedDifferentlyFailure`s.

Fix A and hxcx did **not** change this picture; they removed the messaging
misattribution and the timing failures respectively, leaving the substantive
divergence cleanly isolated.

### Ruled out: "is the NGL captured incorrectly?" (the hxcx-style hypothesis)

Because the hxcx bug WAS an unfaithful capture (prologue built from
pre-interleave side-snapshots), the natural next suspicion is that the NGL body
has the same defect — that the "divergence" is a capture artifact, not a real
schedule difference. This was investigated explicitly (see
`hxcx_artifacts/NGL_CAPTURE_INVESTIGATION_MEMO.md`) and **ruled out**:

- **CMS NGL capture is faithful to emission.** The captured order
  (`cms_capture_listing.txt:584 ds_read -> 604 mfma -> 612 cvt`) matches the
  emitted OptNLL `MAINLOOP` macro body exactly (`kernel.s:2175 -> 2208 -> 2225`,
  macro instantiated at `:2253`). Direct text comparison; they agree.
- **SHADOW NGL capture is faithful by construction.** It is produced by the real
  default emitter `_noLoadLoopBodyDefault(...)` (`KernelWriter.py:4196-4205`) and
  captured by a **post-interleave** walk of the already-assembled `iterCode`
  (`_captureSubIterToBuilder`, `KernelWriter.py:2694-2706`; defined
  `ScheduleCapture.py:2708-2713`) — the structural *inverse* of the hxcx bug,
  which snapshotted *pre*-interleave side lists. The `mfma_index=-1` head block at
  `shadow_capture_listing.txt:439-452` is genuine pre-first-mfma PLR
  leftover-pack flush, not a concatenation artifact (the body below it is
  properly mfma-interleaved).

So both NGL captures reflect what the schedules actually emit. The divergence is
real, and Fix B is a validator-modeling change (case B1), not a capture fix.

---

## Why this is benign today

The 16 failures fire only inside the inline xj16 validation assertion on
`UseCustomMainLoopSchedule=1` builds. The emitted kernel is correct on hardware.
The decisive proof is no longer the unsupported "the `ds_read` reloads the same
logical value the pack-cvt re-derives" assertion — it is the full register-lifetime
trace below ("Register-lifetime trace: does the SHADOW pack consume the same
data?"). That trace shows **both schedules issue byte-for-byte identical `ds_read`s
(same `offset:192`, same `vgprLocalReadAddrA`, same per-body positions) and differ
only in WHEN, within each body, the `ds_read` is placed relative to the pack chain
that consumes it.** CMS consumes the data its same-body `ds_read` just loaded;
SHADOW consumes the prior-body `ds_read` and prefetches for the next. Same logical
k-fragment, different pipeline depth. These are validator false-positives; `uvrl`
is a validator-modeling fix (case B1), proven below.

---

## Register-lifetime trace: does the SHADOW pack consume the same data? (B1 vs B2)

This is the proof that replaces the formerly-unproven "value-equivalent" assertion.
Concrete register range traced: `ValuA_X0_I0+12..15` (the A-side; failure-3 byte
`('v',14)` = `X0_I0+15` lives in this span). Every read/write below is parsed
directly from the two committed listings by `_uvrl_lifetime_trace.py` (a pure-text
latest-writer walk over the unrolled timeline; the listings are already
body-ordered `PRO -> ML_PREV -> ML -> NGL -> NLL`, so file order == unrolled order).

### The TF32-emulation pack chain (what each instruction reads/writes)

From the emitted macro body (`kernel.s:2175-2233`) and identically in both
captures. Per body the `+12..15` span is touched by, in dependency order:

| instr (render) | reads `+12..15` | writes `+12..15` | role |
|---|---|---|---|
| `ds_read_b128 v[+12:+3], v[LocalReadAddrA] offset:192` (`kernel.s:2175`) | — | +12,+13,+14,+15 | loads fp32 high fragment |
| `v_cvt_pk_bf16_f32 v[+10], v[+12], v[+13]` (`kernel.s:2201`) | +12,+13 | — | derive packed lo |
| `v_cvt_pk_bf16_f32 v[+11], v[+14], v[+15]` (`kernel.s:2202`) | +14,+15 | — | derive packed lo |
| `v_mfma_f32_4x4x4_16b_bf16 v[+12:+3], v[74:75], v[+10:+11], v[+12:+3]` (`kernel.s:2208`) | +12,+13,+14,+15 (acc) | +12,+13,+14,+15 | **pack-mfma** (the failure consumer's producer) |
| `v_cvt_pk_bf16_f32 v[+15], v[+14], v[+15]` (`kernel.s:2225`) | +14,+15 | +15 | **pack-cvt** (the `('v',14)` producer) |
| `v_cvt_pk_bf16_f32 v[+14], v[+12], v[+13]` (`kernel.s:2226`) | +12,+13 | +14 | pack-cvt |
| `v_cvt_pk_bf16_f32 v[+13], v[T0+6], v[T0+7]` (`kernel.s:2232`) | — | +13 | pack-cvt (overwrites from T0) |
| `v_cvt_pk_bf16_f32 v[+12], v[T0+4], v[T0+5]` (`kernel.s:2233`) | — | +12 | pack-cvt (overwrites from T0) |

The chain is **destructive and self-contained**: the pack-mfma reads the fp32 the
`ds_read` just deposited (plus `+10/+11`, which the two preceding cvts derived from
that same fp32) and overwrites `+12..15` in place. So a pack-mfma operating on
`+12..15` **always** consumes whatever the most-recent prior writer of `+12..15`
left there — by construction that is either a `ds_read` (fresh fp32) or a prior
body's pack-cvt output (the rotated, already-packed value). The user's premise —
"a pack chain on `+12..15` must consume data a ds_read loaded" — is therefore the
right question; the trace answers *which* ds_read.

### Full ordered read/write trace of `ValuA_X0_I0+12..15`

Generated by `_uvrl_lifetime_trace.py`; latest-writer column is the closest prior
writer of each read byte in the unrolled timeline. Steady-state bodies (ML, NGL,
NLL) shown; PRO/ML_PREV are warmup and identical in shape.

**CMS (subject) — each body's `ds_read` feeds its OWN body's pack chain:**

| listing line | body / pos | instr | reads `+12..15` from |
|---|---|---|---|
| `cms_capture_listing.txt:402` | ML / mfma37 | `ds_read` | (writes +12..15) |
| `cms_capture_listing.txt:418-424` | ML / mfma42-43 | cvts + **pack-mfma** | **ML ds_read L402** |
| `cms_capture_listing.txt:434-439` | ML / mfma46-47 | pack-cvts | ML pack-mfma L424 |
| `cms_capture_listing.txt:584` | NGL / mfma37 | `ds_read` | (writes +12..15) |
| `cms_capture_listing.txt:598-604` | NGL / mfma42-43 | cvts + **pack-mfma** | **NGL ds_read L584** |
| `cms_capture_listing.txt:612-616` | NGL / mfma46-47 | pack-cvts | NGL pack-mfma L604 |
| `cms_capture_listing.txt:752` | NLL / mfma43 | **pack-mfma (failure consumer)** | **NGL pack-cvts L612-616** |

**SHADOW (reference) — each body's pack chain reads the PRIOR body's pack output; the `ds_read` lands at body end and feeds the NEXT body:**

| listing line | body / pos | instr | reads `+12..15` from |
|---|---|---|---|
| `shadow_capture_listing.txt:388` | ML / mfma43 | `ds_read` | (writes +12..15) |
| `shadow_capture_listing.txt:407-410` | ML / mfma-1 | cvts + **pack-mfma** | **ML ds_read L388** |
| `shadow_capture_listing.txt:411-414` | ML / mfma-1 | pack-cvts | ML pack-mfma L410 |
| `shadow_capture_listing.txt:474-477` | NGL / mfma1 | cvts + **pack-mfma** | **ML pack-cvts L411-414** |
| `shadow_capture_listing.txt:490-493` | NGL / mfma2 | pack-cvts | NGL pack-mfma L477 |
| `shadow_capture_listing.txt:610` | NGL / mfma43 | `ds_read` | (writes +12..15) |
| `shadow_capture_listing.txt:648-651` | NLL / mfma1 | cvts + **pack-mfma (failure consumer)** | **NGL ds_read L610** |

The two schedules' *NGL pack-mfmas* consume different-labelled producers
(CMS: NGL ds_read L584; SHADOW: ML pack output), and indeed different k-fragments
(CMS NGL pack = NGL fragment; SHADOW NGL pack = ML fragment) — this is pure
pipeline-depth shift, NOT the failure. The validator's failure consumer is the
**NLL pack-mfma**. Trace it to the originating `ds_read` on both sides:

- **CMS NLL pack-mfma `:752`** ← NGL pack-cvts `:612-616` ← NGL pack-mfma `:604`
  ← **NGL `ds_read` `:584`**.
- **SHADOW NLL pack-mfma `:651`** ← **NGL `ds_read` `:610`** (directly).

Both NLL consumers root at the **NGL `ds_read`**. The divergence is only that CMS
interposes its NGL pack chain between that ds_read and the NLL consumer, so CMS's
*latest byte-writer* is a pack-cvt node, while SHADOW's is the ds_read node.

### ds_read offset / fragment comparison (the decisive artifact)

Every `ds_read` of this span on both schedules is **byte-identical**:
`ds_read_b128 v[vgprValuA_X0_I0+12:+3], v[vgprLocalReadAddrA+0] offset:192`
(`cms_capture_listing.txt:206/402/584`; `shadow_capture_listing.txt:194/388/610`).
Same dest, same `offset:192`, same `vgprLocalReadAddrA+0`.

The LDS read address is double-buffered: `v_xor_b32 v[vgprLocalReadAddrA],
0x10000, ...` (`kernel.s:2003`, guarded `usePLR==1`) toggles the red/black bit
**exactly once per body** on both schedules
(`cms_capture_listing.txt:117/313/509`; `shadow_capture_listing.txt:89/283/521`).
So with constant `offset:192`, the *physical* k-fragment a `ds_read` targets is
fixed by the swap count preceding it. In the unrolled timeline both schedules
interleave swap→ds_read identically:

```
swap(ML_PREV) -> ds_read(ML_PREV) -> swap(ML) -> ds_read(ML) -> swap(NGL) -> ds_read(NGL)
CMS:    L117      L206                 L313       L402            L509        L584
SHADOW: L89       L194                 L283       L388            L521        L610
```

The NGL `ds_read` is the **3rd** `ds_read` = after **3** swaps on BOTH schedules
(CMS L584 after L117/L313/L509; SHADOW L610 after L89/L283/L521). Therefore CMS's
NGL `ds_read` and SHADOW's NGL `ds_read` load the **same physical LDS k-fragment**.
Since both NLL consumers root at their NGL `ds_read` (above), **both NLL pack-mfmas
consume the identical logical k-fragment.** This is the concrete B1 proof: same
data, different producer-node identity.

### Clobber check (CMS side)

Does CMS's NGL `ds_read` (`:584`) overwrite `+12..15` data that a later reader
still needed in its pre-ds_read (ML-pack-output) form? No. The NGL body spans
`cms_capture_listing.txt:442-617`; the **first** instruction in that body to touch
`+12..15` is the `ds_read` at `:584` itself (verified: no `ValuA_X0_I0+12..15`
read appears between `:442` and `:584`). The ML pack output is fully dead — already
consumed inside ML — before NGL's `ds_read` overwrites it. Every reader after
`:584` (`:598, :599, :604`, then NLL `:746+`) consumes the newly-loaded value.
**No clobber.** (B-side mirrors A-side: `cms_capture_listing.txt` swap `:514` /
`ds_read` `:546` for `ValuB`, identical structure.)

### VERDICT: B1 (benign — same logical data; `uvrl` is a validator-modeling fix)

Proof that both packs consume the same logical k-fragment:
1. All `ds_read`s of `+12..15` are byte-identical (`offset:192`,
   `vgprLocalReadAddrA+0`).
2. The LDS double-buffer swap occurs exactly once per body on both schedules, and
   swap→ds_read interleave identically, so the Nth `ds_read` targets the same
   physical fragment on both sides; the NGL `ds_read` is the 3rd on both.
3. The validator's failure consumer (NLL pack-mfma) roots at the NGL `ds_read` on
   **both** schedules (CMS via its NGL pack chain `:584->604->612->752`; SHADOW
   directly `:610->651`). Same fragment, different producer-node identity.
4. CMS's NGL `ds_read` clobbers nothing live — ML's pack output is dead before it
   runs.

The schedules differ ONLY in pipeline depth / WHEN the `ds_read` is placed
relative to the pack chain, exactly the "differ only in scheduling" contract. The
per-byte latest-writer model flags the producer-node difference (pack-cvt vs
ds_read) as a routing divergence even though both encode the same k-fragment. The
16 `EdgeRoutedDifferentlyFailure`s are validator false-positives. **`uvrl` is a
validator-modeling fix (case B1), not a scheduler bug.**

(The trace was produced by `_uvrl_lifetime_trace.py`, committed alongside the
listings; rerun `python3 Tensile/Components/h7lo_uvrl_artifacts/_uvrl_lifetime_trace.py
<listing>` to reproduce the read/write+latest-writer dump cited above.)

---

## What changed since the prior memo

The prior memo / prior version of this doc was written **before Fix A** and
before the two new capture listings existed. Re-deriving against current
artifacts changed these concrete things (the substantive conclusion is unchanged):

1. **Byte-register mapping reconfirmed (prior memo was right; an intermediate
   assumption was wrong).** `('v', 14)` = `ValuA_X0_I0+15` (mapping `('v', N)` =
   `X0_I0+(N+1)`), verified directly from the live validator graph via
   `_h7lo_probe6.py` (the `('v',14)` writer is the cvt of `v[vgprValuA_X0_I0+15]`,
   `intra_operand_byte_offset = 3`). A naive read of the listing as `('v',N)` =
   `X0_I0+N` is off by one — the `name_to_idx` base in `_byte_keys_for_resource`
   (`ScheduleCapture.py:1598`) shifts it.
2. **`kernel.s` line citations corrected (twice — see rocm-libraries-bxqw).**
   The original memo cited `kernel.s:1814 / 1845 / 1856` (a prologue/first-ML
   region). An intermediate revision then cited `kernel.s:2665 / 3000 / 3008` —
   but those are under `label_TailLoopBeginL` (the **tail loop**, starts
   `kernel.s:2659`), NOT the NGL body. The correct NGL writers are inside the
   OptNLL `MAINLOOP` macro body: `:2175` (ds_read), `:2208` (pack-mfma), `:2225`
   (pack-cvt `X0_I0+15`), instantiated at `kernel.s:2253`. The
   captured-vs-emitted NGL order match was verified explicitly against these
   lines (the "are we capturing NGL properly?" check — capture is faithful).
3. **Reference order is now first-class.** The prior doc had to reconstruct the
   default ordering from a graph-dump memo "because the SHADOW schedule is not
   emitted as assembly." It is now read directly from
   `shadow_capture_listing.txt` (`:477` / `:490` / `:610`) with line numbers,
   cross-checked against the live graph.
4. **Failure messages now NGL/NLL-era.** Post-Fix-A the 16 messages cite
   `LRA3`/`LRB3` "(of next iteration)" reference producers, not the old PRO-body
   `PackA0[...] @ idx=-1`.

The substantive conclusion — opposite `ds_read`/pack ordering, flat-register
latest-writer model, B1/B2 normative question — is unchanged.
