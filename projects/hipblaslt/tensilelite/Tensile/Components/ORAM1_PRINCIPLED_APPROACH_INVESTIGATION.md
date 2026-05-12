# oram.1 — principled-approach investigation (post 4up4 + xqj3 framing)

Investigator: oram1-principled-approach (Claude Opus 4.7, 2026-05-12).
Worktree: `/home/alvasile/rocm-libraries/.worktrees/oram1-principled-approach/`.
Branch: `oram1-principled-approach`, forked off vlt at `0f4a522dfdb` (4up4 design memo
already in tree).

Bead: `rocm-libraries-oram.1` (P0; under `rocm-libraries-71hw` Approach-A meta and
`rocm-libraries-xqj3` umbrella).

Cite-policy: every line:column reference verified at the worktree tip. Where a memo
already in the tree cites a different line because it was written against an older fork,
that memo's number is reproduced verbatim and the verified-as-of-this-memo number is
shown alongside it.

---

## §1 — Problem restatement under 4up4 + xqj3

### §1.1 The blocker (recap)

`compare_graphs` reports false positives when a producer-consumer dataflow that exists
identically in two real builds has its endpoints land in different bodies on the two
sides. The motivating case is `UsePLRPack` pipelining: when CMS sets `UsePLRPack=True`
the per-subiter Pack code is appended to `module` from the prologue
(`KernelWriter.py:5009, :5048, :5088` — `packPrePrefetchA/B.add(...)` then
`module.addItems(...)`); when default emits with `UsePLRPack=False` the same Pack code
lands in `pack[plrIdx]` and is consumed during steady-state mainloop iterations
(`KernelWriter.py:5011, :5050`). The dataflow is identical (`PRELOOP_CAPTURE_PHASE1.md
§1`); only the body in which the producer is emitted differs.

Two structural carriers of body-sensitivity propagate the divergence into
`compare_graphs`:

- **Identity construction** at `ScheduleCapture.py:534`:
  `loop_idx = BODY_LABEL_TO_LOOP_INDEX[body_label]` is the second slot of the identity
  tuple `(class_tag, loop_index, canonical_render)`.
- **Edge-key construction** at `CMSValidator.py:1238-1241`:
  `(producer.identity[0], producer.position, ..., consumer.identity[0],
  consumer.position, ...)` — `producer.position` is a `SchedulePosition(loop_index,
  stream_index)` (`ScheduleCapture.py:713-741`), so the edge key carries `loop_index`
  twice (once via identity[0]'s body coupling — through 4up4 this slot becomes
  `_role`, see §1.3 — and once explicitly via `position`).

The legitimate-CMS-reorder branch at `CMSValidator.py:3417-3424` requires identity
equality on both endpoints; with `loop_index` in identity (today) or with
`position.loop_index` in the edge-key tuple (today AND under 4up4), cross-body
pipelined producers cannot match.

### §1.2 4up4's identity tuple — what changes, what doesn't

4up4 (`EMISSION_ORDINAL_DESIGN.md §2.1`) replaces the identity tuple with
`(loop_index, canonical_render, emission_ordinal)`. Two structural facts matter for
oram.1:

1. **`loop_index` survives.** 4up4 drops `class_tag`, not body. `loop_index` is still
   the lead element of the identity tuple. A pack instruction with the same canonical
   render and same per-body emission ordinal in PRO body (`loop_index=-1`) and ML-1
   body (`loop_index=0`) would derive different identities under 4up4 just as it
   does today.
2. **The edge-key tuple still encodes body via `position`.** 4up4 does not touch
   `DataflowGraph.edge_keys`'s `producer.position` / `consumer.position` slots
   (`CMSValidator.py:1238-1241`); 4up4 only proposes migrating `identity[0]` to a
   rocisa-derived `_role` (`EMISSION_ORDINAL_DESIGN.md §4.2`). So even after 4up4,
   the edge key carries `position.loop_index` as a discriminator on both endpoints.

**Consequence.** 4up4 does not address oram.1's blocker. oram.1's design must compose
on top of 4up4's tuple; the live design surface is "what to do about the body element
in identity and/or in edge-key positions, given 4up4's tuple shape."

### §1.3 xqj3's principle — rocisa-derived, not CMS-shaped

The xqj3 umbrella (`rocm-libraries-xqj3`) requires every layer of identity / edge-key
machinery to be grounded in rocisa-derivable properties: the rocisa instruction class
and its `_CLASS_NAME_TO_CATEGORY` registry (`InstructionCategory.py:111-227`), the
register byte-key (`_byte_keys_for_resource` at `ScheduleCapture.py:1203-1253`), the
canonical render-text (`WrappedInstruction.canonical_str` at
`ScheduleCapture.py:303-333`), and stream order. CMS-derived buckets (the `LRA*`,
`PackA*`, `LWS*` category strings produced by `build_idmap` at `:908-947`) are
disallowed in identity-layer mechanics. Approaches B (body-name buckets), C (classifier
patch keyed on the `category` strings in `diagnose_missing_edge`), and H (CMS-emit
forced as the comparison reference) were ruled out by the user on 2026-05-12 for
violating this principle (`PRELOOP_CAPTURE_PHASE1.md §4`).

### §1.4 What "body" means in rocisa terms

A rocisa instruction has no intrinsic body. Body is a property the validator assigns
based on which capture-builder appended the instruction (the `LoopBodyCaptureBuilder`
attached to `BODY_LABEL_PROLOGUE` / `BODY_LABEL_ML_PREV` / `BODY_LABEL_ML` /
`BODY_LABEL_NGL` / `BODY_LABEL_NLL`). It is therefore a *capture-time* attribute, not
an instruction-derivable one. This is the rocisa-shape root of oram.1's blocker:
identity construction promotes a capture-time attribute (which builder appended this
instruction) into a comparison-time discriminator.

The xqj3-aligned framing: identity should be derived from properties of the
instruction (rocisa class, register byte-key, canonical render-text) and from what
the instruction *does* in the dataflow (which producer it sources, which consumer
sinks it). Body is a stream-order locator — it belongs on `position`, where the
within-graph builder uses it to define cross-body order, but it does not belong in
identity, where it forces two structurally-identical instructions in different
bodies to compare unequal.

### §1.5 The false-positive / false-negative budget

A principled fix minimizes both:

- **False positive**: `compare_graphs` reports a failure when the two builds' dataflow
  is in fact equivalent. Source: identity / edge-key carries a discriminator that the
  underlying dataflow does not require.
- **False negative**: `compare_graphs` does not report a failure when the two builds'
  dataflow does diverge. Source: identity / edge-key drops a discriminator that the
  underlying dataflow does require.

Body-label-in-identity (today) is a pure false-positive generator for cross-body
pipelining. Dropping body entirely from identity AND from edge-key positions would
exchange that for a false-negative risk: a real producer in PRO that should have been
in ML could pass undetected if the (canonical_render, ordinal, write-byte-key,
read-byte-key) signature happens to coincide with another producer in ML. The
question is whether such coincidence is possible under the rocisa-derivable signal
set; §6 enumerates concretely.

---

## §2 — Approach A re-derived under 4up4

### §2.1 Mechanism

Drop the body element from identity. Under 4up4, today's tuple
`(loop_index, canonical_render, emission_ordinal)` becomes
`(canonical_render, emission_ordinal)` — a 2-tuple with no body slot.

`SchedulePosition` (`ScheduleCapture.py:713-741`) is unchanged. `GraphNode.position`
still carries `loop_index`; cross-body order is preserved on the `position` axis;
within-graph stream-order checks (e.g. the `default_p_before_c` test at
`CMSValidator.py:3434-3445`) read `position`, not identity, and continue to work.

### §2.2 What `compare_graphs` compares

- **Identity-coverage gate** (`CMSValidator.py:3303-3336`). `_data_flow_ids(graph)`
  returns the set of identities for nodes in `_DATA_FLOW_KINDS`. Under 4up4 the
  filter migrates to rocisa-derived predicates (`EMISSION_ORDINAL_DESIGN.md §4.1`);
  the set comparison itself is unchanged. With Approach A, two structurally-identical
  data-flow nodes in different bodies collapse to the same identity. The gate
  therefore becomes weaker: it sees "one shared identity" where today it sees "two
  bodies' identities differing." See §6 for false-negative analysis.

- **Edge-key tuple** (`CMSValidator.py:1238-1241`). Approach A does NOT touch the
  edge-key tuple. `producer.position` and `consumer.position` still encode
  `loop_index`. So an edge `P → M` with P in PRO body on the CMS side and P in ML
  body on the default side STILL produces different edge keys, because the
  `producer.position` slot differs. **Approach A alone does not fix the blocker.**
  It removes the identity-gate sensitivity but leaves the edge-key sensitivity.

- **Diagnose missing edge** (`CMSValidator.py:3417-3424`). The legitimate-reorder
  branch matches on identity. Under Approach A, the cross-body-but-canonically-same
  pair would now share identity, so the branch would consider them legitimate. But
  it never reaches this branch for the motivating case, because the missing-edge
  classifier runs only on the set `ref_keys - subj_keys`, and the keys still differ
  via position. The classifier would still emit `OrderInvertedFailure`s.

### §2.3 Rocisa-derived signal A relies on

`canonical_render` (rocisa-derived; `ScheduleCapture.py:303-333`) and
`emission_ordinal` (computed from canonical_render and capture-builder ordering,
both rocisa-grounded under 4up4). No CMS-string consultation. Body is dropped, not
re-bucketed via CMS-shaped names.

### §2.4 Verdict for A standalone

A is a NECESSARY but INSUFFICIENT step for the motivating case. It cleans the identity
layer of body coupling — which xqj3 demands as a baseline — but the structural blocker
(the edge-key carrying body via `position`) survives. A unblocks D and E from having
to wrestle with body-in-identity downstream; A on its own leaves
`UsePLRPack`-pipelined edges firing false positives.

---

## §3 — Approach D re-derived under 4up4

### §3.1 Mechanism

Bake register-rotation pipelining into the canonical render-text. iteration N+1's
pack writing `vgprValuA_X0_I0+16` and iteration N's pack writing `vgprValuA_X0_I0+0`
would, after canonicalization, render identically.

Today's `WrappedInstruction.canonical_str` (`ScheduleCapture.py:303-333`) renders
the rocisa instruction with operand text as-emitted. The proposed change normalizes
operand offsets within each rotation cycle: instead of rendering
`vgprValuA_X0_I0+offset`, render `vgprValuA_X0_I0+(offset mod cycle_size)`, where
`cycle_size` is derived from the per-vgpr buffer width.

### §3.2 Identity tuple under D + 4up4

Same as 4up4 baseline: `(loop_index, canonical_render_normalized, emission_ordinal)`.
Body is preserved; canonical render is normalized; ordinal is per-(body,
normalized-render).

### §3.3 What `compare_graphs` compares

- **Identity-coverage gate**: Pack producers in PRO body (CMS side) and Pack
  producers in ML-1 body (default side) STILL get different identities (different
  `loop_index`). D does not address the cross-body case; D addresses only same-body
  cross-iteration rotation (the iter-N-vs-iter-N+1 case where both producers are in
  the same body but different iterations).

- **Edge-key tuple**: similar. The two endpoints' `producer.position` and
  `consumer.position` differ by body; D doesn't touch positions.

### §3.4 Rocisa-derivable signal D relies on

Canonical render + a "rotation cycle size" derived from the symbolic-register
allocator's buffer count (`numVgprBuffer`, `numPackBuffer`,
`KernelWriter.py:3289-3301`). The cycle size is a rocisa-runtime property of the
allocator, not a CMS-scheduler concept. xqj3-clean.

### §3.5 D's actual scope

D's mechanism solves a DIFFERENT problem from oram.1's motivating case. D is the
right tool when iter N's pack and iter N+1's pack land in the same body but at
different offsets — `compare_graphs` would today treat them as different identities
because canonical render differs, and the dataflow comparison would see one extra
producer on each side (not "one missing edge" but "two distinct identities that
should be one"). D folds them into one identity by normalizing the offset.

But oram.1's motivating case is CROSS-BODY, not same-body cross-iteration. D's
normalization, no matter how well-tuned, cannot make `loop_index=-1` equal
`loop_index=0`. D is orthogonal to the body-label issue.

### §3.6 Risks

- **Iteration-context threading.** D needs to know which iteration the instruction
  belongs to in order to compute `(offset mod cycle_size)` consistently across
  builds. iteration context is not on the bare rocisa instruction; it's on the
  capture-builder. Threading it through `WrappedInstruction.canonical_str` requires
  API surface changes that touch every consumer of canonical_str.
- **Coupled to d0xd-style symbolic→numeric resolution.** If a build emits the
  rotated operand symbolically (`vgprValuA_X0_I0+16`) and another emits it
  numerically (`v[92:95]`), the modulo-normalization needs to operate on the
  numeric form. Today's symbolic-name resolver (`_byte_keys_for_resource`,
  `name_to_idx`) is per-body and runs at edge-formation time, NOT at canonical_str
  time. Aligning these is an implementation surface comparable in scope to E.
- **Over-collapse risk (false negative).** If a future bug truly emits
  iter-N+1's pack at the wrong offset (say `+24` instead of `+16` due to a real
  rotation-bookkeeping bug), D's normalization would absorb the bug as long as the
  modulo collapses to a "valid" rotation slot. The normalization deletes the
  signal that distinguishes "iter-N+1 wrote the right place" from "iter-N+1 wrote
  somewhere wrong but inside the rotation window."

### §3.7 Verdict for D standalone

D is the wrong tool for oram.1's motivating case. D solves a same-body cross-iteration
problem; oram.1's blocker is cross-body. D may be needed alongside an oram.1 fix to
handle a separate class of false positives (cross-iteration rotation within one
body) but it is NOT a candidate primary solution for oram.1.

---

## §4 — Approach E re-derived under 4up4

### §4.1 Mechanism

Replace `edge_keys`'s producer/consumer-identity slot AND position slots with
resource-flow keys: `(producer_write_byte_key, consumer_read_byte_key, edge_kind,
intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`. Body falls out of
the edge-key tuple by construction; the comparison becomes "the same physical byte
flowed from one writer to one reader" rather than "the same identified producer-node
flowed to the same identified consumer-node."

The byte-key is rocisa-derived (`_byte_keys_for_resource`,
`ScheduleCapture.py:1203-1253`):
- numeric register: `(regType, regIdx + i)`;
- symbolic register: `(regType, name, base + i)` — or, when `name_to_idx` resolves
  the name, `(regType, resolved + base + i)` (collapsing symbolic to numeric).
- memory: `("mem", space, buffer_id, offset_byte)`.

### §4.2 Identity tuple under E + 4up4

Identity tuple is unchanged from 4up4: `(loop_index, canonical_render,
emission_ordinal)`. E does not touch identity; E rewrites the EDGE-KEY layer only.
The identity-coverage gate at `:3303` continues to use identity, which retains body
sensitivity. So the gate would still fire on cross-body identity drift for data-flow
nodes (LR/LW/GR/MFMA) — for the motivating case (PACK), the gate is exempt
(`_DATA_FLOW_KINDS` excludes PACK), so E by itself sidesteps the gate-firing
problem.

But for any future case where a cross-body LR or LW drift mattered, E alone would
NOT cover it (the gate would still fire on the identity-set divergence). Pairing E
with A — drop body from identity AND switch edge-key matching to byte-key — covers
both layers; see §5.3.

### §4.3 What `compare_graphs` compares

- **Identity-coverage gate**: unchanged behavior (data-flow only; PACK exempt).
- **Edge-key matching**: now by `(producer_write_byte_key, consumer_read_byte_key,
  edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`. Two
  edges with the same physical byte flowing from a writer of class X to a reader of
  class Y match, regardless of which body either endpoint sits in.
- **Diagnose missing edge** (`CMSValidator.py:3417-3424`): the legitimate-reorder
  branch's identity-equality match would NOT be hit by edges that match under the
  new byte-key scheme (they wouldn't be in `ref_keys - subj_keys`). For edges that
  legitimately differ (different producer at the byte-key level, e.g. one build
  added a real new write), the classifier still runs and returns appropriate
  failure shapes.

### §4.4 Rocisa-derivable signal E relies on

`_byte_keys_for_resource` is fully rocisa-derived. The byte-key derivation reads
`RegisterContainer.regType`, `regIdx`, `regNum`, and `regName` (all rocisa
attributes). Symbolic-name resolution via `name_to_idx` is populated from the
writer's `RegSet` directives at capture time (`LoopBodyCapture.name_to_idx`), which
is a property of the rocisa-emit stream. No CMS string consultation.

### §4.5 Risks

- **Symbolic-vs-numeric naming asymmetry.** If one build emits a byte-key
  symbolically (`("v", "ValuA_X0_I0", 0)`) and another emits it numerically
  (`("v", 76)`), the keys don't match by direct equality. The `name_to_idx`
  resolution at `_byte_keys_for_resource:1248-1252` already maps symbolic names to
  numeric byte-keys when both forms refer to the same physical register. **As long
  as both builds populate `name_to_idx` with the same RegSet directives**
  (which they do: both builds run inside the same `kernelBody` invocation and see
  the same `vgprPool` snapshot), this works.
- **MemoryRegion equivalence.** `("mem", space, buffer_id, offset_byte)` requires
  that two captures' `buffer_id` values agree for the same physical buffer. This
  is the d0xd surface: the buffer-id model is per-resolution, not per-rocisa.
  Today the existing `compare_graphs` consumers depend on this anyway (see
  `_byte_keys_for_resource:1232-1236`); Approach E inherits whatever guarantees
  hold today and extends them to the edge-key layer.
- **Same byte-key from different producers in legitimately-different ways.**
  Concrete: in iter N, vgpr+0 is written by Pack0 and consumed by MFMA0; in iter
  N+1, vgpr+0 is written by Pack1 (after register rotation completes a cycle) and
  consumed by MFMA1. Under E, both edges have the same producer_write_byte_key
  and the same consumer_read_byte_key. They differ in `src_operand_slot` /
  `sink_operand_slot` (probably the same — both are operand-0 producers and
  operand-0 consumers), and they differ in `intra_operand_byte_offset` if the
  writes are partial. If they have identical `(write_key, read_key, edge_kind,
  intra_offset, src_slot, sink_slot)`, they collapse to one edge in the set.
  **Is this a false negative?** Both edges represent the same physical dataflow
  (vgpr+0 written then read); both builds will have both edges; both will
  collapse on both sides; the comparison sees both empty residuals. NOT a false
  negative — the residual is shared on both sides.
- **The "single edge collapses two physical events" concern is bounded.** In
  cross-body comparison the question is "did the same bytes flow the same way";
  if two builds emit the same flow in two different bodies, the byte-key
  comparison correctly accepts them as equivalent.
- **`SchedulePosition` is no longer in the comparison.** Loss: within-graph
  stream-order discrimination among edges with identical (write_key, read_key)
  collapses. The within-graph order check at `:3434` now needs to operate on a
  per-(write_key, read_key) bucket — for each such bucket on each side, compare
  the residual order. This is a real classifier rewrite, not just a tuple
  rewrite. See §4.7.

### §4.6 Migration scope

E is a real refactor of `compare_graphs` and the missing-edge classifier. Concrete
files touched:
- `CMSValidator.py:1238-1241`: `edge_keys()` body — rewrite to byte-key tuple.
- `CMSValidator.py:3349-3354`: `ref_edges_by_key` lookup dict — rewrite to byte-key
  tuple.
- `CMSValidator.py:3434-3457`: order check in `diagnose_missing_edge` Phase 1 —
  rewrite the "default_p_before_c" comparison to operate on per-byte-key buckets.

That last item is the load-bearing one. The current order check assumes one edge
key maps to one (producer, consumer) pair on each side; under E, one byte-key pair
may map to multiple physical edges (if the same byte was reused across iterations).
The classifier needs to consider whether the relative order of all such edges is
preserved between sides, not just whether one edge is in the right place.

### §4.7 Verdict for E standalone

E is the most principled single-axis fix for the motivating case: it removes body
from edge-key matching by replacing instruction-identity with resource-flow as the
matching key. It is xqj3-aligned (byte-key is rocisa-derived) and 4up4-orthogonal
(does not touch identity, so 4up4 lands independently). Standalone E leaves the
identity-coverage gate body-sensitive, but the gate excludes PACK, and the
motivating case is PACK-driven. For non-PACK cross-body cases (hypothetical future
LR/LW pipelining), E alone is insufficient.

---

## §5 — Hybrid candidates

### §5.1 A + D

Drop body from identity AND normalize register rotation in canonical render.

**Identity tuple**: `(canonical_render_normalized, emission_ordinal)`.

**Edge-key tuple**: unchanged (still `(identity[0]_replacement_role,
producer.position, ...)`). `position.loop_index` still discriminates by body.

**Coverage of motivating case**: NO. Same as A standalone for the cross-body case;
D adds same-body cross-iteration coverage but doesn't address cross-body.

**Verdict**: not sufficient for oram.1's blocker.

### §5.2 A + E

Drop body from identity AND switch edge-key matching to resource byte-key.

**Identity tuple**: `(canonical_render, emission_ordinal)` (no body).

**Edge-key tuple**: `(producer_write_byte_key, consumer_read_byte_key, edge_kind,
intra_operand_byte_offset, src_operand_slot, sink_operand_slot)` (no position, no
identity, no body).

**Coverage of motivating case**: YES. Both layers (identity gate AND edge-key
matching) are body-blind. PACK identity in PRO body and PACK identity in ML-1 body
collapse to one identity; the edge from prologue Pack to mainloop MFMA uses
byte-keys that match the default-side edge (same physical vgpr written, same
physical vgpr read).

**Coverage of non-motivating cross-body cases**: YES. Cross-body LR drift would
collapse identities (so the gate doesn't fire) and edges would still match by
byte-key.

**Symmetry**: A and E hit the two structural carriers of body sensitivity; together
they remove body from both. The combination is the smallest set of independent
changes that closes the blocker without leaving residual carriers.

**Cost**: A is ~5 LoC in `ScheduleCapture.py:534`. E is ~150-300 LoC across
`CMSValidator.py:1238`, `:3349`, `:3434-3457`, plus test updates. Most of the
budget is in E; A is a one-liner that complements E by ensuring the gate doesn't
re-introduce body sensitivity for non-PACK kinds.

**Verdict**: principled; this is the recommendation. See §7.

### §5.3 D + E

Normalize register rotation in canonical render AND switch edge-key matching to
byte-key.

**Identity tuple**: `(loop_index, canonical_render_normalized, emission_ordinal)`.
Body stays in identity.

**Edge-key tuple**: byte-key based (no position, no identity, no body).

**Coverage of motivating case**: PARTIAL. Edge-key matching covers the cross-body
PACK→MFMA case via byte-keys. But the identity-coverage gate STILL fires for
non-PACK data-flow kinds (LR/LW/GR/MFMA) when their identities differ across
bodies — the gate consults the full identity tuple, which still has `loop_index`.
This means D+E covers the motivating case (PACK is exempt from the gate) but
leaves the gate body-sensitive for any future cross-body LR/LW/GR/MFMA case.

**Cost**: D is ~50-100 LoC plus iteration-context threading; E is ~150-300 LoC.
Combined cost dominates A+E without expanding coverage usefully.

**Verdict**: dominated by A+E. D adds a separate-axis canonicalizer that doesn't
carry its weight here; the cross-iteration rotation case D solves is independently
solved by E's byte-key matching (because rotated registers point at different
physical bytes — exactly what E discriminates on). D's only unique contribution is
collapsing identities, but the rotation case doesn't need identity collapse if the
edge-key matching is byte-based.

### §5.4 A + D + E

Same identity tuple as A+E (no body). Same edge-key as E. Add D's rotation
normalization to canonical_render in identity.

**Coverage**: same as A+E.

**Cost**: A+E plus D's full migration cost.

**Verdict**: D adds nothing over A+E for this blocker. A+E already passes the
rotation case at the edge-key layer (byte-keys discriminate exactly the way
rotation requires). Adding D normalizes the identity-side rendering of rotated
registers, which would only matter if some downstream consumer needs identity
collapse for two physically-distinct rotated emissions. No such consumer exists in
the validator today; identity is consulted by the gate (which under A+E doesn't
need rotation collapse) and by missing-edge diagnosis (which under E doesn't
require identity equality for body-different endpoints). D's contribution is
unused.

If a future requirement emerged that DID need identity collapse for rotated
emissions (e.g., a per-identity diagnostic counter that should treat rotated
emissions as one bucket), D could be added incrementally on top of A+E without
disturbing the comparison logic. There is no design reason to land D upfront.

---

## §6 — False-positive / false-negative table per approach

Scenarios enumerated:

- **S1: UsePLRPack pipelining** — CMS+UsePLRPack=True emits Pack in PRO; default
  emits dataflow-equivalent producer in ML-1 / ML.
- **S2: Cross-iteration register rotation, same body** — iter N pack writes
  vgpr+0; iter N+1 pack writes vgpr+16; both in body ML.
- **S3: PGR variants** — PGR=2 emits an extra prefetch-LR pair in the prologue
  that PGR=1 doesn't have. Identity divergence: extra producer on one side.
- **S4: Tail-loop differences** — NLL emits one extra MFMA on the CMS side because
  of a CMS carve-out. Identity divergence: extra producer on one side.
- **S5: Cross-body legitimate-write divergence** — one build emits a write to
  vgpr+4 in PRO; the other build emits no such write. Real bug, must surface.
- **S6: Same canonical render, different bodies, different purposes** — e.g., a
  `s_cmp_eq_u32 LoopCounterL, StaggerUIter` emitted by GRIncA in body ML and a
  bug-introduced same-text emission in PRO with a different downstream consumer.
- **S7: Resource byte-key coincidence** — two unrelated dataflows happen to write
  the same physical vgpr in different bodies (e.g. pre-mainloop GR writes G2L+0
  in PRO; mainloop GR writes G2L+0 in ML).
- **S8: Missing-instruction bug** — one build forgets a pack write; the other has
  it.

| Scenario | A | D | E | A+D | A+E | D+E | A+D+E |
|---|---|---|---|---|---|---|---|
| S1 (UsePLRPack pipelining) — FP risk | unchanged: gate fires on PACK absence (PACK is exempt from gate today, per `:3301-3304`); edge-key still differs via position | unchanged: cross-body case unaddressed | RESOLVED: byte-keys match across bodies | unchanged | RESOLVED | RESOLVED | RESOLVED |
| S2 (cross-iteration rotation, same body) — FP risk | unchanged: identities differ by canonical_render | RESOLVED: canonical render normalized | RESOLVED via byte-key rotation: rotated registers point to physically-different bytes, so different byte-keys; both builds emit both — neither "missing" | RESOLVED (D part) | RESOLVED (E part) | RESOLVED | RESOLVED |
| S3 (PGR=2 extra prefetch) — FN risk | gate would normally fire on extra LR identity in PRO; A drops body from identity, so PRO and ML LRs collapse — gate may MISS the extra LR if its render+ordinal coincide with an ML LR | unchanged: gate fires correctly | unchanged: gate fires correctly | RISK present from A part | gate FN risk inherited from A; but E's byte-key edge layer would notice the extra dataflow edge from the new producer | gate fires correctly | RISK from A part |
| S4 (tail-loop extra MFMA) — FN risk | same as S3: identity collapse may absorb the extra MFMA | unchanged | unchanged | RISK present from A | RISK present; mitigated by edge-layer detection | unchanged | RISK |
| S5 (cross-body legitimate-write divergence) — FN risk | depends on whether the lone write's render+ordinal collides with another body's identity | unchanged | RESOLVED at edge-key: missing dataflow edge surfaces directly | RISK present | gate FN possible but edge-layer catches the missing edge from the bytes-being-read side | unchanged | RISK at gate / OK at edge |
| S6 (same render, different bodies, different consumers) — FN risk | the two PACKs collapse to one identity, but their consumers differ; the dataflow edges differ; gate doesn't catch instruction-side divergence; edge-layer comparison catches it | unchanged | RESOLVED via differing read-byte-keys at the consumer side | RISK at gate / OK at edge | OK at edge | OK at edge | OK at edge |
| S7 (resource byte-key coincidence cross-body) — FP risk | unchanged | unchanged | LOW: byte-key coincidence is rare AND the comparison only matches if both builds have the SAME coincidence pattern — if they don't, the residual is non-empty and the diagnosis fires | unchanged | LOW | LOW | LOW |
| S8 (missing pack write) — FN risk | gate may miss if the missing render+ordinal coincides with another body's identity | gate fires correctly | RESOLVED at edge-key: the missing producer's downstream MFMA has no edge | RISK at gate | OK at edge | OK at gate | RISK at gate |

### §6.1 Reading the table

The S3/S4 false-negative risk introduced by Approach A is the most subtle and
deserves expansion. A drops `loop_index` from identity, so two emissions of the
same `(canonical_render, emission_ordinal)` pair in different bodies collapse to
one identity. The risk is: a real bug that adds an instruction in body B whose
`(canonical_render, emission_ordinal)` happens to coincide with an existing
emission in body B' would be absorbed by the identity collapse.

How likely is that coincidence? `emission_ordinal` is per-(body, canonical_render)
under 4up4 (`EMISSION_ORDINAL_DESIGN.md §2.2`). With body removed from the key,
the ordinal becomes per-canonical_render. Two bodies emitting the same
canonical_render would each contribute to the same per-canonical_render counter
sequence. If body PRO emits `v_cvt_pk_bf16_f32 v[a], v[b], v[c]` (ordinal 0) and
body ML emits the same render-text (ordinal 1), a real-bug variant where body PRO
emits the same render-text TWICE (ordinal 0 and ordinal 1) would shadow the ML
ordinal-1 emission's identity. The gate would not flag it.

Mitigation under A+E: even though the gate misses the gate-level divergence, the
edge-layer comparison would surface a missing-or-extra edge whose endpoints have
the wrong byte-key sequence. The bug surfaces, just one layer down.

Mitigation more robustly: keep the ordinal counter PER-(body, canonical_render) as
4up4 specifies, but remove body from identity. Two emissions in different bodies
both get ordinal 0 (per their own bodies); identity is `(canonical_render, 0)`
for both; they collide. This is the desired collapse for the motivating case
(cross-body pipelining) AND the risky collapse for the false-negative case
(cross-body extra emission). The two cannot be separated by ordinal-keying alone.

The clean resolution: rely on E's byte-key edge layer to catch what A's
identity-collapse hides. This is the structural argument for A+E: A is a
necessary-but-incomplete identity-layer fix; E catches the residual at the
edge-layer; the combination is jointly complete.

### §6.2 Counts for the recommendation (A+E)

- **False positive risk count**: 1 (S7 — resource byte-key coincidence
  cross-body, LOW likelihood and triggered only when both builds happen to
  coincide identically).
- **False negative risk count**: 0 enumerated. All S3/S4/S5/S6/S8 scenarios that
  are at-risk under A standalone are caught by E at the edge layer.

For comparison with the standalone candidates:
- A: 0 false-positive risks closed for the motivating case (S1 still fires);
  3-4 false-negative risks added (S3, S4, S5, S8 at the gate). Worst overall.
- D: 0 false-positives closed for the motivating case; 0 false-negatives added.
  Wrong tool for this blocker.
- E: motivating case closed (S1); other cross-body cases (S5, S6, S8) closed at
  edge layer. 1 LOW false-positive risk from S7. **Standalone E closes the
  motivating case but leaves the gate body-sensitive for non-PACK kinds (a
  hypothetical future LR cross-body case would re-surface).**
- A+E: motivating case closed; all enumerated cross-body cases closed; 1 LOW
  false-positive risk from S7; 0 false-negatives. **Best.**

---

## §7 — Recommendation + rationale

**Land Approach A + Approach E together as a single design unit.**

### §7.1 Why A+E and not E alone

E alone closes the motivating case (S1) at the edge layer. But the
identity-coverage gate (`CMSValidator.py:3303-3304`) would still consult identity,
and identity still carries `loop_index` post-E. For the PACK motivating case the
gate is exempt by `_DATA_FLOW_KINDS` (`("LR", "LW", "GR", "MFMA")`). For any
future cross-body case that involves an LR or LW (e.g., a future schedule that
moves an LR pair from prologue to mainloop), the gate would fire false positives
again — exactly the same blocker, just on a different category.

A removes the identity-layer carrier of body sensitivity once and for all. With
A+E, no future cross-body movement of any rocisa instruction class can re-surface
the blocker.

The cost difference is negligible: A is a one-line change to `identity_for`
(`ScheduleCapture.py:534`). The complementarity is high: A makes E future-proof
across categories.

### §7.2 Why not A alone

A alone leaves `position.loop_index` in the edge-key tuple. The motivating case
(S1) continues to fire false positives because the edge from PRO-body Pack to
ML-body MFMA still has a different `producer.position` from the edge from ML-body
Pack to ML-body MFMA. A is a necessary precursor to E, not a replacement.

### §7.3 Why not D in any form

D solves a separate-axis problem (same-body cross-iteration register rotation).
The motivating case is cross-body, not cross-iteration. E's byte-key matching
handles cross-iteration rotation correctly anyway: rotated registers point to
physically-different bytes, so the edges are correctly distinguished. D would
add ~50-100 LoC plus an iteration-context plumbing surface that yields zero
unique coverage on top of E. D is not principally objectionable; it is simply
unnecessary.

If a future requirement appears for identity-side rotation collapse (e.g., a
diagnostic counter that should bucket rotated emissions together), D can be added
incrementally without disturbing A+E's comparison logic.

### §7.4 4up4 composition

A+E composes with 4up4 cleanly. 4up4 lands first (or alongside). 4up4 changes
identity from `(class_tag, loop_index, canonical_render)` to
`(loop_index, canonical_render, emission_ordinal)`. A then drops `loop_index`,
yielding `(canonical_render, emission_ordinal)`. E rewrites the edge-key tuple to
byte-keys; this is independent of identity and does not interact with 4up4's
identity refactor.

Under 4up4 + A, downstream identity-layer consumers that 4up4 already migrates to
rocisa-derived predicates (`EMISSION_ORDINAL_DESIGN.md §4`) remain unchanged: the
`_DATA_FLOW_KINDS` filter, the edge-key role producers, and the diagnostic
labellers all consult `node.rocisa_inst` directly under 4up4's design — none of
them need `loop_index` in identity.

Under E, the diagnostic labellers (`cms_node_label`, `CmsLabelRenderer`) continue
to consult `node.category` (the raw string) for human-facing failure rendering;
edge-key matching uses byte-keys; the matching layer becomes purely structural
while the rendering layer keeps its CMS-string vocabulary. xqj3-aligned: no CMS
string in identity or in edge-key matching; CMS strings remain only in
human-facing diagnostic text where they are clearer than rocisa class names.

### §7.5 xqj3 composition

A removes the only remaining CMS-coupled concept from identity (body, encoded as
`loop_index`, is a capture-time attribute that was promoted into a comparison
discriminator). E removes the only remaining body coupling from edge-key
matching. After A+E, identity and edge-key matching depend exclusively on:
- canonical render-text (rocisa-derived);
- emission ordinal (rocisa-derived under 4up4);
- byte-key (rocisa-derived);
- operand-slot indices (rocisa-derived).

This is the xqj3 endgame for the comparison layer. svb1 (idmap-factory
unification) and 4up4 (drop class_tag) are upstream contributions to the same
endgame; A+E is the contribution from oram.1.

### §7.6 What A+E does NOT do

- **Does not change diagnostic rendering.** Failure messages still surface as
  `PackA0[3]`-style labels via `cms_node_label` and `CmsLabelRenderer`
  (`cms_to_timeline.py:81-143`). The CMS-string vocabulary survives where it is
  user-facing.
- **Does not address d0xd's symbolic→numeric resolution.** A+E inherits whatever
  guarantees today's `name_to_idx` provides; if symbolic-vs-numeric mismatches
  surface in some build, that is a d0xd concern independent of A+E. The existing
  `_byte_keys_for_resource:1248-1252` resolver path is the same surface today's
  edge-formation already depends on.
- **Does not absorb real bugs.** S3/S4/S5/S6/S8 scenarios that should surface as
  failures continue to surface under A+E — at the edge layer rather than the
  gate layer when relevant. Diagnostic precision shifts; correctness does not.

---

## §8 — Migration sequencing

### §8.1 Relative to 4up4

A+E lands AFTER 4up4. Reason: A's one-line change rewrites the slot that 4up4
re-defines; landing A first would touch a tuple that 4up4 then replaces wholesale,
producing rebase friction. Landing 4up4 first establishes the
`(loop_index, canonical_render, emission_ordinal)` shape; A then drops
`loop_index`; clean diff.

E is independent of 4up4 (E touches edge_keys, not identity). E can land in the
same PR as A or in a sibling PR; the only coupling is that the test suite needs
to pass under the combined change set, which is easier to validate as one PR.
**Recommended: one PR for A+E, after 4up4 lands.**

### §8.2 Relative to svb1

svb1 (idmap-factory unification) operates on the categorization layer
(`build_idmap` / `build_id_to_category_per_iter`). A+E operates on the identity
and edge-key layers. They do not interact directly. svb1 may land before, after,
or alongside A+E.

### §8.3 Relative to on0t

on0t (intra-capture identity collisions for state registers) is subsumed by 4up4
in the per-emission ordinal design. on0t's resolution is a no-op once 4up4 lands.
A+E does not interact with on0t.

### §8.4 Relative to 71hw (Approach A meta)

A+E is a critical-path prerequisite for the Approach-A real-build comparator under
`rocm-libraries-71hw`. Without A+E, real-CMS-vs-real-default comparison would fire
the false positives this memo enumerates as S1 (and any future variant of cross-body
movement). Sequencing: A+E lands; 71hw's real-build PR consumes the new
comparator semantics.

### §8.5 Recommended overall sequence

1. **4up4 lands.** Identity shape becomes `(loop_index, canonical_render,
   emission_ordinal)`. Existing test suite passes under the new shape.
2. **oram.1 fix lands as A+E in one PR.** Identity becomes
   `(canonical_render, emission_ordinal)` (drop `loop_index`); edge_keys becomes
   byte-key based. Test suite extended with the cross-body PACK pipelining
   regression (verified to fail before this PR, pass after).
3. **svb1 lands** (independently). Categorization layer refactor.
4. **71hw lands** (Approach A real-build comparator). Consumes the cleaned
   comparator semantics from steps 2-3. The shadow-shared-prologue trick at
   `Tests/unit/test_prologue_capture.py:342` (`assert cap_with_cms.prologue is
   cap_with_default.prologue`) is removed; the test compares two real captures
   with independently-derived prologues; A+E lets it pass.

---

## §9 — Test fixtures the recommendation requires

### §9.1 Pre-fix regression (must FAIL before A+E lands)

A unit test that builds two captures of the same problem with `UsePLRPack` flipped
and runs `compare_graphs` directly. Construct via the existing
`build_prologue_capture` test infrastructure (`test_prologue_capture.py`); remove
the `cap_with_cms.prologue is cap_with_default.prologue` assertion at `:342` so
each side derives its own prologue. Without A+E, the comparison emits
`OrderInvertedFailure` or unexplained-missing-edge raises (per `PRELOOP_CAPTURE_PHASE1.md
§2.4`). With A+E, the comparison passes.

### §9.2 Cross-iteration rotation regression (validates §3.5 / §6 S2 claim)

A unit test that builds a capture where iter N's pack writes vgpr+0 and iter N+1's
pack writes vgpr+16 (both in body ML). Run `compare_graphs` against a synthesized
counterpart that has the same dataflow. Both before and after A+E, the comparison
passes — the test pins that A+E does not break this case (and that D is not
required).

### §9.3 Cross-body extra-write false-negative pin (validates §6.1 mitigation)

A unit test that builds two captures where one has an extra Pack producer in PRO
that the other doesn't. Run `compare_graphs`. Without E's byte-key edge layer, the
gate might miss it (under A standalone — to demonstrate the failure mode). With
A+E, the missing dataflow edge surfaces as an unmatched edge-key in `ref_keys -
subj_keys`, and the diagnosis fires.

### §9.4 Full unit suite under
`pytest --ignore=tensilelite/Tensile/Tests/unit/test_MatrixInstructionConversion.py`
must remain green. Tests that pin identity tuple shape (e.g.,
`test_dataflow_graph_lcc.py:128`, `test_dataflow_graph_builder.py:590`) get
rewritten per `EMISSION_ORDINAL_DESIGN.md §4.5` to consult
`node.category` or `node.rocisa_inst` rather than `identity[0]`.

---

## §10 — Discipline check — what this design does NOT contain

- **No CMS-derived concept in identity.** Body (`loop_index`) is the last
  capture-time-derived attribute in identity; A drops it. The remaining slots are
  rocisa-derived: `canonical_render` from `WrappedInstruction.canonical_str`;
  `emission_ordinal` from per-canonical_render append-time counter under 4up4.
- **No CMS-derived concept in edge-key matching.** Byte-keys are
  rocisa-derived through `_byte_keys_for_resource`. `src_operand_slot` /
  `sink_operand_slot` are positional small integers from the rocisa
  `getDstParams` / read-slot conventions. `edge_kind` and
  `intra_operand_byte_offset` are derived from the resolver's overlap logic —
  not from CMS strings.
- **No tactical patch.** A is one line; E is a 150-300 LoC refactor of the
  comparator core. The PR's surface is a comparator-core rewrite, not a
  classifier patch (which is what C would have been). The user's "no cheap, no
  pragmatic, no LoC framing" rule is honored: this memo recommends the
  RIGHT fix, not the SMALLEST one.
- **No false-negative-introducing collapse without an edge-layer mitigation.**
  A's identity collapse exposes a residual false-negative risk for cross-body
  extra emissions; E's byte-key edge layer catches the residual. The pair is
  jointly complete; neither is recommended standalone.
- **No fallback that depends on CMS category strings.** The Approach-C-style
  classifier patch keyed on `category.startswith(...)` is excluded by xqj3's
  ruling and is not present in any branch of the recommended design.

---

## §11 — Citations (verified by reading file:line this investigation)

Worktree: `/home/alvasile/rocm-libraries/.worktrees/oram1-principled-approach/`
at HEAD (vlt fork point `0f4a522dfdb`).

- `Tensile/Components/ScheduleCapture.py:303-333` —
  `WrappedInstruction.canonical_str` (the rocisa-derived render-text input to
  identity).
- `Tensile/Components/ScheduleCapture.py:374-426` —
  `WrappedInstruction.class_tag_for_category` (CMS-shaped collapse table; dropped
  from identity by 4up4, irrelevant to A+E).
- `Tensile/Components/ScheduleCapture.py:507-536` — `identity_for`, today's
  `(class_tag, loop_index, canonical_render)` tuple. Line 534 sets
  `loop_idx = BODY_LABEL_TO_LOOP_INDEX[body_label]`. Approach A drops this slot.
- `Tensile/Components/ScheduleCapture.py:702-708` —
  `BODY_LABEL_TO_LOOP_INDEX` (PRO=-1, ML-1=0, ML=1, NGL=2, NLL=3).
- `Tensile/Components/ScheduleCapture.py:713-741` — `SchedulePosition` shape
  `(loop_index, stream_index)` after the `5v4u` collapse.
- `Tensile/Components/ScheduleCapture.py:744-760` —
  `assign_stream_indices_for_body` lex-sort by `(slot.mfma_index,
  slot.sequence)`.
- `Tensile/Components/ScheduleCapture.py:1203-1253` —
  `_byte_keys_for_resource`, the rocisa-derived byte-key producer that
  Approach E threads into edge-key matching. Lines 1248-1252 are the
  `name_to_idx` symbolic→numeric resolver.
- `Tensile/Components/ScheduleCapture.py:1256-1340` — `_resolve_producers`,
  the per-byte writer-lookup that already operates on byte-keys.
- `Tensile/Components/CMSValidator.py:892-921` — `GraphNode` dataclass
  (note: docstring at `:913` says `(rocisa_class_name, loop_index,
  signature_tuple)` — stale; actual shape is
  `(class_tag, loop_index, canonical_render)` today).
- `Tensile/Components/CMSValidator.py:1203-1241` —
  `DataflowGraph.edge_keys`. Line 1238 is the tuple Approach E rewrites to
  byte-key form.
- `Tensile/Components/CMSValidator.py:1789` — last-writer-wins identity
  collapse `nodes_by_identity[node.identity] = node` at graph-build time.
- `Tensile/Components/CMSValidator.py:1828-1913` — Phase 2 edge formation,
  consuming `_byte_keys_for_resource` and `_resolve_producers`. The
  byte-key infrastructure E extends to the edge-key tuple is already in
  use here.
- `Tensile/Components/CMSValidator.py:3295-3358` — `compare_graphs`
  control flow: identity-coverage gate (`:3301-3336`) and edge-key
  comparison (`:3338-3358`). Approach A weakens the gate (collapses
  body-keyed identities); Approach E rewrites the edge-key set
  construction at `:3338-3354`.
- `Tensile/Components/CMSValidator.py:3361-3424` —
  `diagnose_missing_edge`. The legitimate-CMS-reorder branch at
  `:3417-3424` requires identity equality on both endpoints; under A+E
  this branch is rarely entered (because the byte-key matching usually
  resolves the cross-body case before the classifier runs) but remains
  correct when entered (identity equality on cross-body endpoints holds
  under A).
- `Tensile/Components/CMSValidator.py:3434-3457` — Phase 1 order check
  (same-body only); rewrite under E to operate on per-byte-key buckets.
- `Tensile/Components/EMISSION_ORDINAL_DESIGN.md §2-§5` — 4up4's
  identity refactor. A composes on top by dropping `loop_index` from
  the new tuple shape.
- `Tensile/Components/PRELOOP_CAPTURE_PHASE1.md §1-§4` — verified
  pipelining model and the §4 approach catalog (with B/C/H ruled out).
- `Tensile/KernelWriter.py:4961` (current) / `:4910` (original §Q1
  cite) — `usePLRPack` gate.
- `Tensile/KernelWriter.py:5009, :5048, :5088` (current) — prologue
  Pack-chain emission route.
- `Tensile/KernelWriter.py:5011, :5050` (current) — mainloop
  Pack-chain route.
- `Tensile/KernelWriter.py:3289-3301` — producer-consumer rotation in
  `_loopBody` (D's domain).
- `Tensile/Tests/unit/test_prologue_capture.py:165-274` —
  `test_preloop_divergence_catches_useplrpack_change`.
- `Tensile/Tests/unit/test_prologue_capture.py:277-350` —
  `test_whole_kernel_useplrpack_cms_matches_both_defaults`. The
  `cap_with_cms.prologue is cap_with_default.prologue` assertion at
  `:342` is the shadow-shared-prologue trick that A+E enables removing.

---

## §12 — Reporting summary

- **Recommended approach**: A + E. Drop `loop_index` from identity AND
  switch edge-key matching from instruction-identity to resource
  byte-key. xqj3-clean (rocisa-derived signal at every layer);
  4up4-orthogonal (composes on top of 4up4's tuple).
- **False-positive risk count for A+E**: 1 (S7 — resource byte-key
  coincidence cross-body, LOW likelihood).
- **False-negative risk count for A+E**: 0 enumerated (A's gate-layer
  collapses are caught by E's edge-layer).
- **Migration sequencing**: After 4up4. Single PR landing A and E
  together. Independent of svb1; precondition for 71hw's real-build
  comparator.
- **Framing corrections to the dispatching brief**:
  - The brief's claim that `producer.position` carries `loop_index`
    "directly" is correct (`SchedulePosition.loop_index` per
    `ScheduleCapture.py:730`), but the brief did not name that
    `position` is THE second carrier of body sensitivity (the first
    being identity). Approach A alone does not address this carrier;
    only Approach E (or rewriting `position.loop_index` to be omitted
    from the edge-key tuple — equivalent to E's identity-removal
    flavor) does. The recommendation honors this two-carrier reality.
  - The §6 false-positive / false-negative table reveals that
    Approach A standalone is the WORST candidate (closes nothing
    motivating, opens 4 false-negative scenarios), not a partial-
    fix candidate. A is only useful in combination with E.
  - The brief enumerates "false positive in PGR variants" as a
    discovery target. Verified: PGR=2 emits an extra prefetch-LR pair
    in PRO; this is a real cross-build divergence (when comparing
    PGR=1 vs PGR=2 builds), and under A+E it correctly surfaces at
    the edge layer (the extra producer's downstream MFMA edge has no
    counterpart byte-key on the PGR=1 side). Not a false positive;
    a true positive that A+E preserves.
