# rocm-libraries-e293: GRInc + OverriddenInputFailure(SCC) NGL residual

Bead: `rocm-libraries-e293` (P0 question, sub-bead of `rocm-libraries-3ija`).
Investigator: e293-investigator (Claude Opus 4.7, May 2026).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/`.
Validation tip: `bc72aec71a` (post-aixt). **Investigation only — no production code changes.**

---

## Verdict (TL;DR)

**(b) Validator false positive.** All four affected fixture configurations
(`_192x256x32_TF32 NN` × 2 LDSTr variants, `_256x160x64_16bit TN`)
exhibit the **same identity-collision mechanism** documented in
`EXAMPLE_YAML_DEFECT_INVESTIGATION.md` (z012). The CMS schedules
emit byte-correct, two-instruction SCC carry chains in NGL with no
intervening foreign SCC writers. The validator's
`OverriddenInputFailure(SCC)` detection at
`Tensile/Components/CMSValidator.py:3822-3848` fires because the
cross-build edge-key matcher operates on
`(canonical_render, emission_ordinal)` identities and the same
canonical render-text — `s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]`
— is emitted **once per side** (A and B), with the ordinal assignment
order (which side gets ordinal=0 vs ordinal=1) flipping between
default and CMS captures because their stream-positions of the two
emissions differ.

z012's gating helper
`_producer_has_same_render_peer_with_distinct_category` was the
correct fix; it disappeared from the validator at some point during
the post-z012 refactor cascade (4up4 / hdem / hdu1) — likely when
`class_tag` was dropped from the identity tuple and ordinal-based
disambiguation was assumed to cover both within-build and cross-build
cases. **It does not cover cross-build.** That is the root cause.

No hardware impact. No schedule defect. Recommendation: re-instate
the z012-style gate (or implement an equivalent ordinal-aware
cross-build identity-collision detector) and remove the four false
positives from the 3ija residual surface.

---

## Q1 — GRInc carry chains in source

### Anatomy of one GRInc (default-side, kernel writer)

`KernelWriterAssembly.globalReadIncrement` (called per tensor side
from `globalReadIncrementAB` at `KernelWriterAssembly.py:9173/9181`)
emits this exact 9-instruction sequence per A or B side, in this
order:

```
1. s_cmp_eq_u32  s[sgprLoopCounterL], s[sgprStaggerUIter]   # writes SCC (wrap-iter check)
2. s_cselect_b32 sN,   s[sgprWrapU{A,B}+0], s[sgprGlobalReadIncs{A,B}+0]  # reads SCC -> incLower
3. s_cselect_b32 sN+1, s[sgprWrapU{A,B}+1], 0                              # reads SCC -> incUpper
4. s_add_u32     s[sgprSrd{A,B}+0], s[sgprSrd{A,B}+0], sN     # writes SCC (carry-out)
5. s_addc_u32    s[sgprSrd{A,B}+1], s[sgprSrd{A,B}+1], sN+1   # reads SCC (carry-in)  -> SRD chain
6. s_sub_u32     s[sgprShadowLimit{A,B}+0], ..., sN           # writes SCC (carry-out)
7. s_subb_u32    s[sgprShadowLimit{A,B}+1], ..., sN+1         # reads SCC (carry-in)  -> ShadowLimit chain
8. s_cmp_eq_u32  s[sgprShadowLimit{A,B}+1], 0                 # writes SCC
9. s_cselect_b32 s[sgprSrd{A,B}+2], s[sgprShadowLimit{A,B}+0], BufferLimit  # reads SCC
```

Three SCC carry-chain pairs per side:

| # | Producer (writes SCC) | Consumer (reads SCC) | Span |
|---|---|---|---|
| chain-1 | (1) `s_cmp_eq_u32 LoopCounterL, StaggerUIter` | (2)+(3) `s_cselect_b32 WrapUx_lo/hi` | adjacent (1 hop) |
| chain-2 | (4) `s_add_u32  Srdx+0, ..., sN`             | (5)   `s_addc_u32 Srdx+1, ..., sN+1`   | adjacent (1 hop) |
| chain-3 | (6) `s_sub_u32  ShadowLimitx+0, ..., sN`     | (7)   `s_subb_u32 ShadowLimitx+1, ..., sN+1` | adjacent (1 hop) |
| chain-4 | (8) `s_cmp_eq_u32 ShadowLimitx+1, 0`         | (9)   `s_cselect_b32 Srdx+2, ShadowLimitx+0, BufferLimit` | adjacent (1 hop) |

Each chain is **structurally adjacent** in the per-side emission. The
A-side group and B-side group are emitted as two independent 9-instruction
blocks. CMS schedules slot the 18 sub-instructions (9 A + 9 B) at distinct
mfma-index slots per the per-tile schedule definition — but each carry
chain remains intra-side; CMS is **not interleaving carry-chain
producer and consumer of the same chain across sides**.

### Per-fixture slot placement

#### `_192x256x32_TF32 NN` (Tensile/Components/CustomSchedule/gfx950/_192x256x32_TF32.py:181-357)

```
'GRIncB' : grIncB = create_range(min_val=max(lrb0)+1, num=3, step=1, repeat=3)
'GRIncA' : grIncA = create_range(min_val=max(lrb0)+1, num=3, step=1, repeat=3)
```

GRIncB and GRIncA each occupy 9 slots. They are clustered, but at
distinct mfma-index ranges (GRIncB starts after LRB0; GRIncA starts
after GRIncB).

#### `_256x160x64_16bit TN` (Tensile/Components/CustomSchedule/gfx950/_256x160x64_16bit.py:100-101)

```
'GRIncA' : [[29,30,31,32,33,34,35,36,37]],   # 9 sub-instructions at mfma-index 29..37
'GRIncB' : [[0,1,2,3,4,5,6,7,8]],            # 9 sub-instructions at mfma-index 0..8
```

GRIncA and GRIncB are placed in **fully disjoint mfma-index ranges**.
There is zero cross-side interleaving in this fixture's CMS layout —
B at 0-8, A at 29-37, with everything else in between.

---

## Q2 — Actual NGL asm captured from CMS Build #1

Captured via a one-off dump script
(`Tensile/Tests/unit/_e293_dump.py`, removed after capture; output is
included verbatim below). For each affected fixture, all 4 GRInc
NGL failures emitted by `compare_graphs(ref_graph, subj_graph)` were
serialized along with: producer/consumer/intervening_writer asm,
their identity tuples, and the list of every SCC writer between the
producer and consumer in BOTH the reference and subject graphs.

### Fixture 1: `_192x256x32_TF32 NN, LDSTr=True, TLDS=1`

5 total failures; 4 are GRInc OrderInverted/OverriddenInput in NGL.

Selected residual (Failure[1], `OverriddenInputFailure` on SCC):

```
producer  label = GRIncB[0] @ idx=6 body=NGL
   asm   = s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]
   identity = ('s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]', 0)
consumer  label = GRIncA[2] @ idx=11 body=NGL
   asm   = s_cselect_b32 s69, s[sgprWrapUA+1], 0
   identity = ('s_cselect_b32 s69, s[sgprWrapUA+1], 0', 0)
intervening_writer  label = GRIncA[0] @ idx=11 body=NGL
   asm   = s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]
   identity = ('s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]', 1)
   resource = SCC

REF  scc writers between producer and consumer (excl): 0
SUBJ scc writers between producer and consumer (excl): 4
   stream_index=17  GRIncB s_add_u32  SrdB+0, SrdB+0, s68
   stream_index=19  GRIncB s_sub_u32  ShadowLimitB+0, ..., s68
   stream_index=22  GRIncB s_cmp_eq_u32  ShadowLimitB+1, 0
   stream_index=29  GRIncA s_cmp_eq_u32  LoopCounterL, StaggerUIter   <-- the "intervening writer"
```

Identical shapes appear for Failures 2/3 (same producer, different
consumers — the two `s_cselect_b32 ... WrapUA ...` instructions of
the GRIncA chain that read the wrap-iter SCC) and the 2 OrderInverteds
(GRIncA[0] producer → GRIncB[1]/[2] consumer).

### Fixture 2: `_192x256x32_TF32 NN, LDSTr=False, TLDS=1`

Identical residual shape to Fixture 1. Same 4 GRInc failures, same
producer/consumer/intervening_writer asm, same identity tuples.

### Fixture 3: `_256x160x64_16bit TN, LDSTr=False, TLDS=1`

4 total failures, all GRInc NGL. Selected residual (Failure[1]):

```
producer  label = GRIncB[0] @ idx=0  body=NGL
   asm   = s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]
   identity = ('s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]', 0)
consumer  label = GRIncA[1] @ idx=30 body=NGL
   asm   = s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0]
   identity = ('s_cselect_b32 s68, s[sgprWrapUA+0], s[sgprGlobalReadIncsA+0]', 0)
intervening_writer  label = GRIncA[0] @ idx=29 body=NGL
   asm   = s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]
   identity = ('s_cmp_eq_u32 s[sgprLoopCounterL], s[sgprStaggerUIter]', 1)
   resource = SCC

REF  scc writers between producer and consumer (excl): 0
SUBJ scc writers between producer and consumer (excl): 4
   stream_index=10  GRIncB s_add_u32  SrdB+0, SrdB+0, s68
   stream_index=17  GRIncB s_sub_u32  ShadowLimitB+0, ..., s68
   stream_index=22  GRIncB s_cmp_eq_u32  ShadowLimitB+1, 0
   stream_index=55  GRIncA s_cmp_eq_u32  LoopCounterL, StaggerUIter   <-- the "intervening writer"
```

Same identity-collision shape. The "producer" and the
"intervening_writer" share **byte-identical canonical render** and
differ ONLY by `emission_ordinal`.

---

## Q3 — Foreign SCC writers between actual producer and consumer

The validator's framing — "foreign writer to SCC between producer
and consumer of the SAME carry chain" — has a precondition: the
producer and the consumer of the SCC edge belong to the same
physical chain. In every captured failure, **that precondition
fails**:

| Fixture | "Producer" identity ord | "Consumer" identity ord | Same physical chain? | REF foreign writers between | Real defect? |
|---|---|---|---|---|---|
| `_192x256x32_TF32 NN, LDSTr=True`  | s_cmp ord=0 | s_cselect WrapUA ord=0 | NO (cmp→cselect-WrapUA only on same side) | 0 | NO |
| `_192x256x32_TF32 NN, LDSTr=False` | s_cmp ord=0 | s_cselect WrapUA ord=0 | NO | 0 | NO |
| `_256x160x64_16bit TN, LDSTr=False` | s_cmp ord=0 | s_cselect WrapUA ord=0 | NO | 0 | NO |

All four failures per fixture trace to the same identity-collision
pair: a `s_cmp_eq_u32 LoopCounterL, StaggerUIter` (chain-1
producer) emitted once on the A side and once on the B side, both
rendering byte-identically. In the **default schedule** the two
A-side and B-side groups are emitted contiguously (A's chain-1
producer is immediately followed by A's chain-1 consumers; B's
similarly), so REF has 0 SCC writers between any matched
producer/consumer — chain-1 is intact on both sides in BOTH builds.

In the **CMS schedule**, the 18 sub-instructions are scheduled at
distinct mfma-index slots per the schedule's `'GRIncA'` /
`'GRIncB'` slot lists. Because both sides' chain-1 producers
collapse to the same canonical render, `assign_emission_ordinals`
assigns them ordinal=0 and ordinal=1 in stream-position order. CMS's
stream-position order swaps which side wins ordinal=0 relative to
default. The validator's edge-key matcher then sees the
default-side "ord=0 cmp → ord=0 cselect WrapUA" edge as missing in
subj because in subj the ord=0 cmp belongs to the B-side group while
the ord=0 cselect WrapUA belongs to the A-side group.

The "foreign SCC writers between" the validator counts in subj are
all **legitimately part of the B-side carry chain** (chain-2 add,
chain-3 sub, chain-3 cmp) plus the second emission of chain-1's cmp
(the "intervening_writer" — also legitimate, just on the OTHER side).

**On real hardware** each carry chain is an adjacent 2-instruction
pair within its own side's group. The CMS schedule preserves that
adjacency. SCC is not clobbered.

---

## Q4 — Validator's `OverriddenInputFailure` detection logic

Source: `Tensile/Components/CMSValidator.py:3807-3851`
(`diagnose_missing_edge`, SCC branch).

```python
ref_resource = ref_edge.resource
if getattr(ref_resource, "regType", None) == "scc":
    intervening_writer = None
    for e in subj_graph.edges:
        if (e.consumer.identity == c_id
                and getattr(e.resource, "regType", None) == "scc"
                and e.producer.identity != p_id):
            intervening_writer = e.producer
            break
    if intervening_writer is not None:
        return [OverriddenInputFailure(...)]
```

**The detection is purely identity-based, not stream-position-based.**
It walks every edge in `subj_graph.edges`, looking for any SCC edge
whose consumer-identity matches the missing edge's consumer-identity
and whose producer-identity differs from the missing edge's producer.
Any matching producer is reported as the "intervening writer" — even
if that producer is not actually positioned between the original
producer and consumer in stream order.

**Two false-positive surfaces:**

1. **Identity collision on canonical render-text.** Documented at
   length in `EXAMPLE_YAML_DEFECT_INVESTIGATION.md` (z012). When the
   same canonical render-text is emitted twice in one body
   (different physical instances of the same scalar op for two
   tensor sides), `assign_emission_ordinals` assigns ordinals 0 and
   1 in stream-position order. If default and CMS captures order the
   two emissions differently, the same physical instruction maps to
   different identities across captures — the matcher sees a
   "missing edge" and the SCC branch then finds the alias as an
   "intervening writer." The producer's asm and the
   intervening_writer's asm are **byte-identical** in this scenario,
   which is the diagnostic fingerprint. **All four captured e293
   failures show this fingerprint.**

2. **No stream-position adjacency check.** Even if the consumer
   genuinely had two distinct SCC producers in subj, the validator
   would name the FIRST one found in `subj_graph.edges` iteration
   order — which is not necessarily "the SCC writer that physically
   sits between producer and consumer." The current code is
   diagnostic-only correct (it tells the user "some other SCC writer
   exists"), not localization-correct. For e293 this manifests as
   the `intervening_writer` being reported at a stream position
   that is NOT between the producer and the consumer (e.g. in
   `_256x160x64_16bit TN`, the intervening_writer's stream position
   is `idx=29` while the consumer is at `idx=30` — the writer is
   the consumer's own chain-1 producer, not an interloper).

The z012 fix introduced a gate
(`_producer_has_same_render_peer_with_distinct_category`) that
detected case (1) by scanning the producer's body for any other
instruction sharing the producer's canonical render but having a
different category, and skipping Phase 1's order check (which would
otherwise emit `OrderInvertedFailure`) when the ref edge's
`regType` is `scc` or `m`. **That gate is no longer present in the
current `CMSValidator.py`** — `grep -n
"_producer_has_same_render_peer_with_distinct_category\|is_collapsed_state_edge\|_state_reg_kind"
CMSValidator.py` returns no matches.

The gate appears to have been deleted at some point during the post-
z012 refactor cascade (4up4: replaced identity tuple with
`(loop_index, canonical_render, emission_ordinal)`; hdem: dropped
`loop_index` to `(canonical_render, emission_ordinal)`; hdu1:
removed `class_tag`). The 4up4 design assumed `emission_ordinal`
would disambiguate physically-distinct same-render emissions and
make the z012 gate unnecessary — and within a single build it does.
But cross-build matching breaks because ordinals are assigned in
stream-position order, and stream-position order legitimately
differs between default and CMS. The 4up4 commit message
(`83c5d507b4`) does not address the cross-build case.

Note: the SCC branch's docstring at `CMSValidator.py:3807-3812`
acknowledges that "the consumer simply lost its SCC edge to the
producer for an unrelated reason" can reach this branch and falls
through to the ALU early-return. But the precondition checked is
only "no other SCC writer with a different identity targets the same
consumer in subj" — which the identity-collision case fails (a
peer-with-different-ordinal IS another producer with a different
identity). So the fall-through never fires for collision cases.

---

## Q5 — Per-fixture classification

| Fixture | Verdict | Mechanism |
|---|---|---|
| `_192x256x32_TF32 NN, LDSTr=True, TLDS=1`  | **(b)** validator FP | identity collision on `s_cmp_eq_u32 LoopCounterL, StaggerUIter` |
| `_192x256x32_TF32 NN, LDSTr=False, TLDS=1` | **(b)** validator FP | identical to above |
| `_256x160x64_16bit TN, LDSTr=False, TLDS=1` | **(b)** validator FP | identical to above |

All three classify cleanly as (b). The producer asm and the
intervening_writer asm in every one of the 12 SCC failures (4 per
fixture × 3 fixtures) are byte-identical, satisfying the
identity-collision diagnostic fingerprint.

The OrderInvertedFailures on the same producer/consumer pair in the
opposite direction (`GRIncA[0]` producer → `GRIncB[1/2]` consumer)
are the **same identity-collision mechanism** firing on Phase 1's
order check (`CMSValidator.py:3792-3798`) instead of the SCC branch.
They are not a separate bug; they are the same edge-mismatch
re-classified by Phase 1's branch precedence.

---

## Q6 — Hardware-impact assessment

**Not applicable** — verdict is (b) across the board. No real
defect, so no hardware risk.

For completeness: the actual emitted CMS asm for each fixture's
NGL body preserves the carry chains correctly (each chain's
producer is immediately adjacent to its consumer within the same
side's group). On real hardware the SRD increment, ShadowLimit
update, and wrap-iter cselect all compute correct values. The 3ija
triage's `_3ija_residual_triage_runner.py` confirms zero
wait-coverage residuals across every fixture, supporting the
"emitted asm is correct" reading from a different angle.

---

## Recommended next-step bead(s)

(SCOPE — do not file; user dispatches.)

### Bead R1 (P0, validator fix): re-instate cross-build identity-collision gate

Detect when a missing SCC (or m0) edge involves a producer whose
canonical render has another emission within the same body capture
(in either default or subj). When detected, suppress the
OrderInverted / OverriddenInputFailure branch for that edge and
emit nothing (the edge is a cross-build identity-aliasing artifact,
not a real divergence).

Two principled implementation paths:

1. **Re-introduce the z012 gate verbatim, scoped to SCC + m0 edges.**
   The original `_producer_has_same_render_peer_with_distinct_category`
   helper already encoded this; the post-z012 refactors deleted it
   in the assumption that emission_ordinal would solve the same
   problem. It does for within-build identity disambiguation; it
   does NOT for cross-build matching when stream-position order
   differs.

2. **Promote `class_tag` (or a side-discriminating render-text
   suffix) back into the SCC carry-chain identity surface.** The
   underlying issue is that `s_cmp_eq_u32 LoopCounterL, StaggerUIter`
   on the A side is semantically distinct from the same render on
   the B side: they touch independent SCC carry chains. The kernel
   writer side's `class_tag` ("GRIncA" vs "GRIncB") encodes this
   distinction; dropping it from the identity tuple at hdu1 made the
   two emissions cross-build-aliasable. A scoped fix (per-resource:
   keep class_tag in identity for SCC/m0 edges, drop it for
   register-typed edges) would address the precise mechanism without
   regressing the body-blindness benefits 4up4/hdem brought to
   register dataflow.

Both options are non-trivial design decisions; pick the principled
one in design review. The "minimal-LOC" option (option 1) is also
the more surgical principled option since it targets the exact
phenomenon — but option 2 closes the door on related cross-build
identity collisions that aren't currently surfacing (m0 is the
known second case; LCC's `s_sub_u32 LoopCounterL, ...` is a
candidate third).

### Bead R2 (P2, validator hygiene): make `OverriddenInputFailure` SCC-detection stream-position-aware

Even with R1, the SCC branch's "find any other SCC producer" search
will misreport the intervening writer if the false-positive surface
is ever extended (e.g. when more than two same-render emissions
coexist). Add an explicit stream-position check: only consider an
SCC writer "intervening" if its position satisfies
`producer.position < writer.position < consumer.position` in subj.
This is hygiene; not a correctness fix on its own.

### Bead R3 (P2, runner): suppress GRInc-NGL false positives in 3ija residual surface

Once R1 lands, re-run `_3ija_residual_triage_runner.py` and update
`3IJA_RESIDUAL_TRIAGE.md` §3.B / §6 Q1 to reflect the resolution.
The 6 false positives counted in the 3ija aggregate should drop to
0; the residual surface for the strict-XFAIL pin in
`test_approach_a_non_cms_reference.py` should narrow accordingly.

---

## Cross-references

- `EXAMPLE_YAML_DEFECT_INVESTIGATION.md` (z012) — original
  identity-collision investigation; same mechanism, different
  fixture (`_160x128x64_TF32` triggered via `example.yaml`).
- `Tensile/Components/CMSValidator.py:3807-3851` — current
  `OverriddenInputFailure` SCC detection (post-deletion of z012's
  gate).
- `Tensile/Components/CMSValidator.py:1300-1303` — `edge_keys`
  identity-based matching (body-blind under hdem Approach A).
- `Tensile/Components/ScheduleCapture.py:729-760` —
  `assign_emission_ordinals` stream-position-keyed counter.
- `Tensile/KernelWriterAssembly.py:9149-9181` —
  `globalReadIncrementAB` per-side emission of the 9-instruction
  GRInc sequence.
- `Tensile/Components/CustomSchedule/gfx950/_192x256x32_TF32.py:181-357`
  (NN branch) and `_256x160x64_16bit.py:83-131` (TN branch) —
  affected schedule definitions.
- `3IJA_RESIDUAL_TRIAGE.md §3.B` — the 3ija triage row that
  surfaced this question.
- `4up4` commit `83c5d507b4` — emission_ordinal introduction; did
  not address cross-build ordinal-flip case.
- `hdu1` commit `b288591d2b` — class_tag deletion; closed the door
  on the side-discriminator that would otherwise have prevented the
  cross-build collision.
