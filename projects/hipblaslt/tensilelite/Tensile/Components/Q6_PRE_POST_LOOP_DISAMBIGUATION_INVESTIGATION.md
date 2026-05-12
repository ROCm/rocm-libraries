# Q6: PRE_LOOP vs POST_LOOP @-1 disambiguation in node.name / dump tool

Investigation memo (sub-bead under `rocm-libraries-71hw`). Read-only on
production code. Filed 2026-05-11 by Opus 4.7 from
`2LZD_INVESTIGATION.md §5 Q6` follow-up candidate (also re-surfaced by
the dispatching agent on 2026-05-12).

## §1 — Problem statement (verified)

Two distinct kernel-writer code paths emit pack instructions stamped
with `mfma_index=-1` but **different `slot_kind`** values. The dump
tool (and `node.name` itself) collapses them visually under the same
`@-1.X` notation.

### Verified code paths

- **PRE_LOOP path (in-iter, before first MFMA).**
  `KernelWriter.py:2666-2683` — `_captureSubIterToBuilder`. For
  `subiter==0`, any pre-MFMA instruction in the iter stream gets
  `slot_kind=SLOT_KIND_PRE_LOOP, mfma_index=-1`. Lands in the SAME
  `LoopBodyCapture` as the iter mainloop (ML-1 / ML / NGL / NLL).

- **PRE_LOOP path (prologue body).**
  `KernelWriter.py:5138-5144` calls `build_prologue_capture(…)`
  (`ScheduleCapture.py:1979-2016`), which for every leaf in
  `prefetch_pack_a / prefetch_pack_b` calls
  `builder.append(inst=leaf, category=f"PackA{plr_idx}", subiter=0,
  slot_kind=SLOT_KIND_PRE_LOOP, mfma_index=-1)`. Lands in
  `BODY_LABEL_PROLOGUE` ("PRO"), a separate body — no collision with
  iter mainloop here.

- **POST_LOOP path (leftover-pack capture).**
  `KernelWriter.py:4624-4628` — leftover-pack capture at the end of
  `_loopBody`'s shadow-emit branch. Stamps leaves IN THE SAME
  `LoopBodyCapture` as the iter mainloop with
  `slot_kind=SLOT_KIND_POST_LOOP, mfma_index=-1, subiter=kernel["LoopIters"]`.
  Lands in ML-1 / ML alongside the PRE_LOOP-tagged iter prologue
  entries.

### Verified counter behavior

`LoopBodyCaptureBuilder.append` (`ScheduleCapture.py:817-835`) keys its
sequence counter on `(slot_kind, mfma_index)`:

```
self._seq_counter = {}  # (slot_kind, mfma_index) -> next sequence
```

So PRE_LOOP@-1 and POST_LOOP@-1 are **separate sequence series**,
both starting from 0. PRE_LOOP@-1.X and POST_LOOP@-1.X with the same
`X` are different events — but render identically in `node.name`.

### Verified lex sort

`assign_stream_indices_for_body` (`ScheduleCapture.py:744-760`):

```
sorted_tis = sorted(instructions, key=lambda ti: (ti.slot.mfma_index, ti.slot.sequence))
```

`slot_kind` is NOT in the sort key. PRE_LOOP@-1.5 and POST_LOOP@-1.5
collide on the lex tuple; sort stability decides the tie-break (which
in production matches insertion order: PRE_LOOP entries first because
they're appended during per-iter capture, POST_LOOP appended later by
the leftover-pack pass).

### Verified `_make_node` rendering

`CMSValidator.py:1587-1617` — `_make_node`:

```
name = f"{tagged_inst.category}@{slot.mfma_index}.{slot.sequence}"
```

No `slot_kind` in the rendered string. `node.name` collides for
(category, mfma_index, sequence)-tuple-equal entries with different
`slot_kind`.

### Verified dump renderer

`Tests/unit/_dump_carveout_assembly.py:133`:

```
f"// pos={n.position} cat={n.category} name={n.name}{graph_tag}\n"
```

Uses `n.name` directly. The `pos.stream_index` IS distinct (because
sort stability gives PRE/POST entries with same sequence different
stream_indices via lex-tuple ordering across the body), so a careful
reader could disambiguate via `stream_index` — but that's not what
the bead is about. The eye-confusing collision is on the `name`
field.

## §2 — Quantified collision in existing artifacts

Re-using `tensilelite/Tensile/Tests/unit/default.s` from the
`validator_long_term_plans` worktree (the latest TF32 4×4 TN canonical
dump available on disk; PGR=0 / no PRO body). Per-body @-1 counts:

| Body | @-1 entries (verified) |
|------|------------------------|
| ML-1 | 74 |
| ML   | 74 |
| NGL  | 14 |
| NLL  | 14 |

PRE/POST split for ML-1 (back-derived by walking the per-category
sequence numbers in the dump):

| Series | Count | Categories | Sequence range |
|--------|-------|------------|----------------|
| PRE_LOOP@-1 | 14 | SYNC, PackA0, PackB0, SNOP | seq 0..13 |
| POST_LOOP@-1 | 60 | PackA3, PackB3 | seq 0..59 |

Concrete observed collisions (lines from `default.s`, both render as
`@-1.13` and a human reader cannot tell PRE_LOOP from POST_LOOP):

```
// pos=...stream_index=26 cat=SNOP name=SNOP@-1.13     [NOT-IN-GRAPH]   ← PRE_LOOP
// pos=...stream_index=27 cat=PackA3 name=PackA3@-1.13                  ← POST_LOOP
```

(They have different stream_indices, so they ARE distinct nodes in
the graph; only the rendered `name` collides.)

**Per-body collision incidence.** A name collision between
PRE_LOOP@-1.X and POST_LOOP@-1.X happens whenever both series have an
entry at the same sequence number AND same category. Since the
PRE_LOOP series uses categories `{SYNC, PackA0, PackB0, SNOP}` and
the POST_LOOP series uses `{PackA3, PackB3}` for this kernel, **no
exact (category, name)-pair collision occurs in default.s** — only
`@-1.SEQ` suffix collisions. But the categories are NOT
fundamentally disjoint:

- PRE_LOOP's `SLOT_KIND_PRE_LOOP, mfma_index=-1` tagging at
  `KernelWriter.py:2672` applies to ANY pre-MFMA leaf, classified via
  `id_to_category` from the iter's `build_id_to_category_per_iter`
  (`PackA{u}` for u in 0..LoopIters-1).
- POST_LOOP's leftover-pack capture (`KernelWriter.py:4604-4612`)
  uses `leftover_idmap` from the same `build_idmap` factory PLUS the
  `prefetch_pack_a/b` safety-net (`f"PackA{_plrIdx}"`).

Both can produce categories `PackA0..PackA{LoopIters-1}`. A kernel
configuration where the same `PackA{u}` category appears in both
series WOULD produce indistinguishable `node.name`. The current
TF32 4×4 TN dump just doesn't trigger this case (PRE side sees
`PackA0`/`PackB0` only, POST side sees `PackA3`/`PackB3` only).

## §3 — Downstream consumers of `node.name`

Grep results (`grep -rn 'node\.name\|n\.name'`):

- `Tensile/Tests/unit/_dump_carveout_assembly.py:86` — dump renderer
  prints `name={n.name!r}`.
- `Tensile/Tests/unit/_dump_carveout_assembly.py:133` — dump renderer
  prints `name={n.name}` per node line.
- `Tensile/Components/CMSValidatorVisualization.py:76` — matplotlib
  visualization label: `name = node.name or node.category`.

That's it. The Failure renderers (`FailureNodeLabel`, `cms_node_label`,
`CmsLabelRenderer.render` / `render_position`) do NOT consume
`node.name` — they construct their own `category[N] @ idx={mfma_index}`
strings via `cms_to_timeline.py:112-120`.

So failure messages produce e.g. `PackA3[5] @ idx=-1` — and PRE_LOOP /
POST_LOOP STILL render identically there too, since `[N]` is the
per-category-stream `name_idx` resolved against the body's
`instructions` list and `idx=-1` is the shared `mfma_index`. The
failure-message collision is a SEPARATE manifestation of the same
underlying issue and not resolved by changes to `node.name` alone.

## §4 — Identity / Z012 impact

`TaggedInstruction.identity_for` (`ScheduleCapture.py:507-536`):

```
return (cls_tag, loop_idx, WrappedInstruction.canonical_str(inst))
```

`slot_kind` is NOT in the identity tuple. Adding it to `node.name` or
to the lex sort key does NOT change identity, so:

- `compare_graphs` cross-graph identity matching is unaffected.
- The Z012 family's `identity_for` keys remain stable.

`canonical_str(inst)` is the rocisa render text — also slot_kind-
independent. Safe under any of the candidate approaches below.

## §5 — Test-pinning footprint

Searched `Tensile/Tests/` for test assertions matching `name=='Pack…@-…'`
or `node.name == "..."` patterns:

- No tests pin exact `@-1.SEQ`-shape strings.
- Only test_dataflow_graph_register_gaps.py:3509 constructs a
  synthetic `name="SSETPRIO@0.0"` for a GraphNode — but only asserts
  on `label.primary` (the `cms_node_label`-derived string), not on
  `node.name`. Safe.
- 53 test mentions of `slot_kind` exist; only 1 mentions
  `SLOT_KIND_POST_LOOP` (`test_ScheduleCapture.py:42` import + line
  925's PRE_LOOP slot_kind assertion). Approach D's lex-sort change
  would need to verify that `assign_stream_indices_for_body`-derived
  `stream_index` values remain in tests' expected order — but tests
  generally exercise `slot_kind` at the SlotKey level, not stream
  ordering.

## §6 — Candidate approaches

### A. Renderer-only (dump tool + visualization)

**Mechanism.** Modify `_dump_carveout_assembly.py` and
`CMSValidatorVisualization.py:76` to read
`node.tagged_inst.slot.slot_kind` and render
`@PRE-1.X` vs `@POST-1.X` (or `@-1.X[PRE]` / `@-1.X[POST]`).

**What changes.**
- 1 line in `_dump_carveout_assembly.py:133` (the `@` rendering).
- 1 line in `_dump_carveout_assembly.py:86` (FIRST/LAST debug print).
- 1 line in `CMSValidatorVisualization.py:76`.
- No production code changes; `node.name` field stays as-is.

**Test footprint.** Zero existing tests pin renderer output. Could
add a regression test that exercises the disambiguation explicitly.

**Risks.** None for graph identity / failure rendering. Doesn't fix
the failure-message collision (FailureNodeLabel renders
`PackA3[5] @ idx=-1` without slot_kind too, §3 above).

**Open questions.** Should the visualization be updated atomically?
Should `_dump_carveout_assembly.py:86` (the FIRST/LAST debug print
used in `_print_failed_kernel_dataflow`) also be updated for
symmetry?

### B. Add slot_kind to `node.name`

**Mechanism.** Modify `_make_node` at `CMSValidator.py:1601` to
include slot_kind in the name format:

```
name = f"{tagged_inst.category}@{slot.slot_kind[:3].upper()}-{slot.mfma_index}.{slot.sequence}"
```

(Or any encoding that disambiguates.)

**What changes.** 1 line in `CMSValidator.py`. Affects EVERY
consumer of `node.name` (the three sites listed in §3) atomically.
Names go from `PackA3@-1.13` to e.g. `PackA3@POST-1.13` /
`PackA3@PRE-1.13`. MFMA-slot names go from `MFMA@5.0` to
`MFMA@MFM5.0` (ugly).

**Test footprint.** Zero pinning tests; visual diffs only.

**Risks.**
- Renderer-side strings change everywhere, including in the
  matplotlib visualization (existing visual snapshots if any would
  drift).
- The MFMA-slot rendering becomes uglier. Could mitigate by only
  prefixing for `mfma_index=-1`.

**Open questions.** Should it be `@PRE-1.X` (only when slot_kind
matters) or always include slot_kind for symmetry? What encoding is
least visually noisy?

### C. New `slot_kind` field on GraphNode

**Mechanism.** Add `slot_kind: str` to `GraphNode` dataclass
(`CMSValidator.py:892-920`), populated by `_make_node` from
`tagged_inst.slot.slot_kind`. Renderers (dump tool, visualization)
read the new field directly when they want to disambiguate.

**What changes.**
- Add `slot_kind: str = SLOT_KIND_MFMA` field on `GraphNode`.
- Populate in `_make_node` from `tagged_inst.slot.slot_kind`.
- Update renderers to use the new field.

**Test footprint.** Tests that construct synthetic `GraphNode`
instances may need to pass `slot_kind=` (only
`test_dataflow_graph_register_gaps.py:3502-3511` does this; default
value covers it).

**Risks.** Low. Adds redundancy (slot_kind already accessible via
`tagged_inst.slot.slot_kind`), so introduces no new info; arguably
cosmetic. Doesn't fix the failure-message collision (§3) unless
the FailureNodeLabel renderers also start consuming the field.

**Open questions.** Worth the redundant field if `tagged_inst.slot`
is always available? (Probably no — Approach A is simpler.)

### D. Promote slot_kind into the lex-sort key

**Mechanism.** Change
`assign_stream_indices_for_body` (`ScheduleCapture.py:744-760`) to:

```
key=lambda ti: (ti.slot.mfma_index, ti.slot.slot_kind, ti.slot.sequence)
```

So PRE_LOOP@-1 and POST_LOOP@-1 entries get distinct
`stream_index` ranges automatically (PRE_LOOP first by Python
string ordering: `"mfma" < "post_loop" < "pre_loop"` — wait, that
puts mfma first which is fine, then `post_loop` before `pre_loop`).

Actually the natural Python string order on the SLOT_KIND_*
constants is `"mfma" < "post_loop" < "pre_loop"` — that REVERSES
the desired PRE-then-POST order. So Approach D requires either
renaming constants to give the right lex order OR adding an
explicit ordering map.

**What changes.**
- One line in `assign_stream_indices_for_body`.
- BUT: tests that depend on stream_index order (production code via
  `SchedulePosition` and the bridge that emits `(loop_index,
  stream_index)` tuples) need careful review. Production currently
  relies on insertion-order-equals-lex-order at line
  `assign_stream_indices_for_body`'s docstring claim:

  > "By construction LoopBodyCaptureBuilder.append issues
  > slot.sequence per (slot_kind, mfma_index) bucket starting at 0,
  > with the sequence counter shared across subiters — so the
  > natural list-append order matches the lex order in production."

  Adding slot_kind as a sort component CHANGES the stream_index
  assignment for current production builds (the tied @-1.X entries
  reorder). That changes `SchedulePosition.stream_index` values
  visible to `compare_graphs`, every queue-state simulator, and
  every test that inspects `stream_index` directly.

**Test footprint.** Larger blast radius. At minimum: every test
that exercises `LoopBodyCaptureBuilder.append` + indirect
stream_index ordering.

**Risks.** Highest. Breaks "stream_index reflects emission order"
intuition. Production scheduling-correctness logic relies on
stream_index ordering.

### E. Renderer-only WITH per-body legend (refinement of A)

**Mechanism.** Like A, but additionally emit a one-line legend at
the start of each body-section in the dump:
`// SLOT_KINDS PRE=N POST=M MFMA=K`. Lets the reader cross-check
counts visually.

Trivial extension; doesn't change the disambiguation mechanism, just
augments it.

## §7 — Recommendation

**Approach A (renderer-only).** Smallest blast radius, zero
production-code risk, fully solves the dump-tool collision the bead
surfaces.

**Cost estimate.** ~3 LoC across two files
(`_dump_carveout_assembly.py` lines 86 and 133;
`CMSValidatorVisualization.py:76`); 1 small regression test that
constructs a body containing both PRE_LOOP and POST_LOOP @-1
entries and asserts on the rendered output.

**Risk assessment.** Low.
- No identity / canonical_str / stream_index changes.
- No FailureNodeLabel changes.
- No `node.name` field changes (existing in-process consumers stay
  byte-stable).
- Visual diff in dump output only — no pinning tests affected.

**What it does NOT solve.**
- The Failure-message rendering collision (`PackA3[5] @ idx=-1` is
  ambiguous between PRE and POST). If/when a real failure case
  produces a confusing message, file a separate bead to update
  `CmsLabelRenderer.render_position()` (`cms_to_timeline.py:118-120`)
  to include slot_kind for `mfma_index=-1` cases.
- The matplotlib visualization can be left for a follow-up if the
  dump-tool fix is the immediate need.

**Interaction with `rocm-libraries-71hw` (meta).** Independent. The
meta-bead concerns Approach A for the validator (real-build
reference). This Q6 follow-up is a UX issue in the dump tool used
WHILE diagnosing validator problems. Fixing it makes future
investigations under the meta easier (less eye-strain reading
`@-1.X` entries) but is not on the critical path for the meta's
correctness goals. Worth wiring as `meta blocked-by Q6` only if a
specific Approach-A debugging session expects to lean on the dump
tool heavily; otherwise the dependency is informational.

## §8 — Cross-references

- `2LZD_INVESTIGATION.md §5 Q6` — original follow-up candidate.
- `2LZD_INVESTIGATION.md §1 (D1)` — per-body @-1 head replication
  analysis; cited counts (74/74/14/14) match this memo's §2.
- `KernelWriter.py:2666-2683` — PRE_LOOP iter-stream tagging.
- `KernelWriter.py:4624-4628` — POST_LOOP leftover-pack tagging.
- `KernelWriter.py:5138-5144` — PRE_LOOP prologue-body capture entry.
- `Components/ScheduleCapture.py:117-119` — slot_kind constants.
- `Components/ScheduleCapture.py:744-760` — lex sort.
- `Components/ScheduleCapture.py:817-835` — sequence counter keying.
- `Components/ScheduleCapture.py:1979-2016` — `build_prologue_capture`.
- `Components/CMSValidator.py:1587-1617` — `_make_node` name format.
- `Components/CMSValidator.py:892-920` — `GraphNode` dataclass.
- `Components/CMSValidatorVisualization.py:73-77` — visualization
  consumer of `node.name`.
- `Components/cms_to_timeline.py:112-120` — `CmsLabelRenderer`
  (failure-rendering path; SEPARATE collision out of scope here).
- `Tests/unit/_dump_carveout_assembly.py:86, 133` — dump renderer.
