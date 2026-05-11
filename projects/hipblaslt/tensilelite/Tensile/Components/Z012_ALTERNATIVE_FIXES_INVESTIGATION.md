# z012 alternative-fix investigation: identity-collision artifact at construction

Bead: rocm-libraries-z012 (follow-up to investigation memo
`EXAMPLE_YAML_DEFECT_INVESTIGATION.md`).
Investigator: z012-alt-approaches (Claude Opus 4.7, May 2026).
Working tree: `/home/alvasile/rocm-libraries/.worktrees/z012-alt-approaches/`.
Base branch: `users/alvasile/validator_long_term_plans` @ `702c88be689`.

This memo characterizes alternative principled fixes for the
`OrderInvertedFailure` artifact described in z012's primary memo
(`EXAMPLE_YAML_DEFECT_INVESTIGATION.md`). The user has explicitly rejected
two earlier candidates:

1. **Diagnosis-layer carve-out** (gate Phase-1's `OrderInvertedFailure`
   on a state-register-identity-collision detector at
   `diagnose_missing_edge`). Rejected as a band-aid.
2. **Original principled fix** (extend `identity_for` with `category` as a
   4th tuple element). Rejected because it surfaces a separate
   default-vs-CMS Pack-tagging-strategy divergence (z012 §6) that would
   need to be fixed first.

The brief asks for approaches that **DON'T require category labels** in
the identity tuple, and ranks them.

---

## 1. Summary

**Top recommendation: do not pursue any of the construction-layer
alternatives in the catalog. The problem is mis-framed at the layer the
brief targets.**

Probes confirmed that every attempt to discriminate the colliding
emissions WITHOUT a stable cross-capture side label either (a) fails to
disambiguate (occurrence-index, position-based, topology-based) or (b)
adds non-determinism to identity (next-instruction context, slot fields)
that breaks legitimate CMS-reorder cases the existing tests already
cover. The two structurally clean fixes — full no-collapse, or a
selective no-collapse — also break the comparator because the entire
comparison machinery (compare_graphs, diagnose_missing_edge, edge_keys)
is built on 1:1 identity → node and on edge_keys keyed by
`producer.position`.

**Backup recommendation: refactor `EXAMPLE_YAML_DEFECT_INVESTIGATION.md`'s
§6 finding into a real bead and fix the Pack-tagging-strategy divergence
properly. Then revisit whether the original principled fix becomes
viable.** The Pack-tagging fix is bounded in scope (one module —
`assign_id_to_category_for_default` lines 1041-1074 vs CMS-side
`expand_cms_macro` line 2147) and SHOULD be done before any identity
restructuring. But probing also surfaced a similar divergence on
LRSwap (LRSA/LRSB), which suggests the divergence is a class of bug
not a single instance.

If the user wants to pursue a fix THIS cycle and is unwilling to do the
Pack-tagging refactor first, the **least-bad option in this catalog is
Approach 8** (filter spurious state-register edges at construction). It
works on the synthetic repro, is scope-limited to state-register edges,
and aligns with the SCC cross-body precedent (theq) that "fix at the
right layer rather than suppress at the diagnosis layer". But it DOES
silently drop edges that downstream code would have classified, so it
trades a false-positive failure for a silent edge-loss — which is itself
a project anti-pattern. Honest framing: every viable alternative is some
flavor of suppression; the question is which layer the suppression lives
at.

---

## 2. Background

The defect is described in full in
`EXAMPLE_YAML_DEFECT_INVESTIGATION.md` (z012 memo). Briefly:

- `build_dataflow_graph` collapses physically-distinct instructions
  sharing the same `canonical_str` into a single `GraphNode` via
  `nodes_by_identity[node.identity] = node` (last-writer-wins),
  `CMSValidator.py:1266` (line is `:1146` in the memo's pre-merge tree;
  `:1266` in the working tree).
- Kernel writers emit the same canonical scalar instruction multiple
  times for state registers (e.g.
  `s_cmp_eq_u32 LoopCounterL, StaggerUIter` once per A-side / B-side GR
  lowering; `s_add_u32 m0, m0, 4224` once per DTL group).
- The collapsed survivor's stream-position differs across captures whose
  scheduling differs. Because `edge_keys` (`CMSValidator.py:733-736`)
  embeds `producer.position` in the edge key, edges referencing the
  collapsed-survivor get different keys in default vs CMS, surfacing in
  `compare_graphs` as missing edges and routed through Phase 1's order
  check, which fires `OrderInvertedFailure`.

**Off-limits per the brief:**

- The diagnosis-layer carve-out (gating Phase 1 in
  `diagnose_missing_edge`).
- The original principled fix (`category` as 4th tuple element).

**On-limits:** any construction-layer change to `identity_for`,
`build_dataflow_graph`, or edge formation that doesn't require a stable
cross-capture category label.

**Note on baseline state:** the z012 memo (in `Tensile/Components/`)
documents a fix that lives in the unmerged `validator_long_term_plans`
branch's HEAD as a memo, but the carve-out CODE (`is_collapsed_state_edge`
gate, `_producer_has_same_render_peer_with_distinct_category` helper)
DOES NOT exist in the actual `CMSValidator.py` on this branch
(`grep -rn _producer_has_same_render_peer` finds matches only in the
memo). So `example.yaml` against the current `_get_schedule_160x128x64_TF32`
WOULD raise the 8 `OrderInvertedFailure`s on a fresh build. This memo is
exploring fixes for a real, present defect — not a hypothetical refactor
of an existing fix.

---

## 3. Approach catalog

Each approach is presented with: **shape**, **probe outcome**,
**verdict**.

### Approach 1 — Position-based identity tuple

**Shape.** Extend `identity_for` to include `(loop_index, stream_index)`
or the kernel-writer slot fields `(slot.mfma_index, slot.sequence)` as
identity discriminators.

**Probe.** `Tests/scratch/probe_z012_approach1_position.py` extends the
identity tuple with `(slot.mfma_index, slot.sequence)`. On the synthetic
repro the cmp identities became distinct per-emission; default and CMS
both gained per-emission identities; but the consumer (cselect)
identities ALSO became distinct because slot.sequence differs across
captures for the same logical instruction (default sequence 1.2 vs CMS
sequence 0.2 for "WrapUB1"). Result: `compare_graphs` raises
`CaptureConsistencyError` because identity sets diverge.

Looked at the post-bridge `stream_index` instead: same problem, worse —
stream_index is only defined AFTER `assign_stream_indices_for_body` runs
during `build_dataflow_graph`, and by construction differs between
default and CMS for the same logical instruction whenever CMS reorders.

**Verdict.** Doesn't work. Position is exactly the thing that varies
across captures; embedding it in identity makes the identity set diverge,
which is the failure mode `compare_graphs` is checked against at entry.
The brief's open question — "are positions stable enough across CMS-vs-
default for compare_graphs to be meaningful?" — answers itself: by
construction they are NOT stable, that's the whole reason CMS exists.

### Approach 2 — No-collapse at construction

**Shape.** Replace `nodes_by_identity[node.identity] = node` with
`nodes_by_identity.setdefault(node.identity, []).append(node)`. Each
emission becomes its own node.

**Probe.** Skipped as not viable. The downstream comparison machinery is
built around 1:1 identity → node:

- `DataflowGraph.nodes` is a dict `{identity: GraphNode}` (singular)
  consumed at `subj_graph.nodes.get(p_id)` in `diagnose_missing_edge`,
  `compare_graphs`'s identity-coverage check (which would have to
  compare LISTS not nodes), and many test fixtures.
- Edge formation walks `sorted_nodes` and uses each node as both edge
  source and `latest_writer` value; a list-per-identity model requires
  rewriting the edge formation phase.
- `compare_graphs` enters via `_data_flow_ids` (set difference), which
  works on hashable identities but no longer on (identity, instance)
  pairs.

The blast radius is the entire comparator + every test fixture that
synthesizes a graph by `nodes={ident: node}` (50+ test sites surveyed
in `test_dataflow_graph_*.py`).

**Verdict.** Structurally invalid. The 1:1 identity → node assumption is
load-bearing across the comparator. Any fix that abandons it requires
re-architecting the comparison layer, which is much larger than the
problem.

### Approach 3 — Distinguish by occurrence-index within body

**Shape.** Identity = `(cls_tag, loop_idx, occurrence_index, canonical_str)`,
where `occurrence_index` is the Nth occurrence of
`(cls_tag, canonical_str)` walked in the body's stream order.

**Probe.** `Tests/scratch/probe_z012_approach3_occurrence.py` patches
`_make_node` to assign per-body occurrence indices. On the synthetic
repro:

- Default body order: cmpA (occ=0), cmpB (occ=1).
- CMS body order:    cmpB (occ=0), cmpA (occ=1).
- Both sides have IDENTICAL identity sets (each side has one (cmp, occ=0)
  and one (cmp, occ=1)) — `compare_graphs` identity-coverage check
  passes. But:
- The cmp at (occ=0) in default IS the GRIncA cmp; in CMS it's the GRIncB
  cmp. Same identity, different physical side.
- The cselect_b for WrapUA pairs with cmpA (occ=0) in default but with
  cmp(occ=1) in CMS — different SCC producer identity in the consumer's
  edge.
- Result: 2 `OrderInvertedFailure`s, exactly the same shape as the
  baseline. Approach 3 does NOT solve the problem.

**Verdict.** Doesn't work. Occurrence-index is itself stream-order-
dependent in exactly the way that breaks cross-capture matching.
Confirms the brief's open question: "across the CMS schedule corpus,
how often does the emission order of a given canonical_str differ
between default-side and CMS-side captures?" — for our synthetic case,
always; production exhibits it for at least the GRIncA/B and GRA/GRB
cases the z012 memo documents.

### Approach 4 — Topology-based identity

**Shape.** Build identity from the instruction's reads/writes
fingerprint — the set of (regType, regIdx) pairs the instruction
touches.

**Probe.** Skipped — the brief's framing is correct on the merits. For
the GRIncA/B `s_cmp_eq_u32 LoopCounterL, StaggerUIter` case, both
emissions read `(s, sgprLoopCounterL)` and `(s, sgprStaggerUIter)` and
write `("scc", 0)`. The fingerprints are identical. Topology is too
coarse a discriminator for instructions whose entire register footprint
is shared across lowerings.

A finer variant — including byte-offsets within the operand or the
intra-operand slot tuple — gains nothing for these instructions, since
the operands carry no side-bearing index.

**Verdict.** Doesn't work for the GRIncA/B SCC case (acknowledged in the
brief). Doesn't help the m0 case either (both writes are
`s_add_u32 m0, m0, 4224` with identical reads/writes).

### Approach 5 — Tighten canonical_str via lowering-context window

**Shape.** Embed surrounding-context info (e.g., the next instruction's
canonical_str) in the identity tuple for state-register-writing
instructions.

**Probe.** `Tests/scratch/probe_z012_approach5_context.py` patches
`_make_node` to look up the successor instruction's `canonical_str` and
appends it to identity for SCmp* and m0-writing SAdd* instructions.

- On the synthetic repro: 0 failures. Both default and CMS attribute
  the same context-augmented identity to the same logical-side cmp;
  edges connect correctly.
- On the existing test suite (`probe_z012_approach5_pytest.py`): the
  test `test_clobber_yields_scc_conflict_failure` raises
  `CaptureConsistencyError`. The producer identity in the reference
  graph carries successor `s_cselect_b32 s100, s50, s51`; in subject
  graph the producer is the LAST instruction in the body, so its
  successor is `""`. Two different identities for the same physical
  cmp emission across captures. Approach 5 BREAKS existing tests.

**Verdict.** Half-solves: it disambiguates per-side WHEN the next
instruction is structurally distinct cross-side (e.g., cselect_a vs
cselect_b). But it FAILS whenever the surrounding context is itself
asymmetric across captures (e.g., the cmp is the last instruction in one
body but not in the other; or the next instruction is itself a
collision-prone state-register write). Adds non-determinism to identity
that breaks the "render-string identity is stable across captures"
contract.

### Approach 6 — Drop identity collapse for state-register writers

**Shape.** For instructions whose write-set contains SCC or m0,
identity is augmented with a per-emission unique tag (e.g.,
`id(tagged_inst)`) so collapse cannot fold them.

**Probe.** `Tests/scratch/probe_z012_approach6_no_collapse_state.py`
patches `_make_node` to append `id(tagged_inst)` to identity for
state-register writes. On the synthetic repro: each cmp emission becomes
its own node (no collapse). But `id(tagged_inst)` is per-OBJECT, and
default and CMS construct different `TaggedInstruction` objects, so
identity sets diverge. `compare_graphs` raises
`CaptureConsistencyError`.

The cross-capture-stable variant of this approach would need a
"per-emission discriminator that's identical across default and CMS for
the same logical emission" — which is exactly the lowering-source/side
tag (i.e., the rejected category-based approach).

**Verdict.** No cross-capture-stable per-emission discriminator exists
without revisiting the category-tagging story. Approach 6 reduces to
"either Pack-tagging gets fixed or this doesn't work," same as the
original principled fix.

### Approach 7 — Move the validation question

**Shape.** Build a separate state-register dataflow validator that runs
alongside `compare_graphs` and reports state-register-specific failures
with state-register semantics ("is there an intervening clobber?" rather
than "is the order preserved?"). `compare_graphs` could then drop
state-register edges from its diff entirely.

**Probe.** Skipped after analysis. Two structural concerns:

1. Project discipline §"no parallel APIs" forbids two validators that
   each handle a subset of dataflow. The same comparison machinery
   (`compare_graphs`, `diagnose_missing_edge`, `validate_edge_wait_coverage`)
   already discriminates per-resource-type internally; a parallel pass
   would either duplicate that machinery or have to be plumbed back into
   the existing pass — which collapses back to "add a state-register
   discriminator to the existing pass."
2. The within-pass discrimination already exists: the SCC clobber
   detector at `CMSValidator.py:2877` runs on SCC-typed edges as a
   separate branch within `diagnose_missing_edge`. The artifact at z012
   fires from Phase 1's `OrderInvertedFailure` BEFORE the SCC branch
   runs (Phase 1 returns early on order inversion). A clean version of
   Approach 7 would route SCC/m0 edges directly to the clobber branch
   and skip Phase 1 entirely — which is functionally identical to z012's
   diagnosis-layer carve-out (already rejected).

**Verdict.** Reduces to either the rejected diagnosis-layer carve-out or
to a parallel-API anti-pattern. Doesn't provide a third option.

### Approach 8 — Filter spurious state-register edges at construction

**Shape.** Keep collapsed identity. At edge formation, when a state-
register edge's producer is "collision-prone" (i.e., another instruction
in the same body has the same canonical_str AND is also a state-register
writer), drop the edge.

**Probe.** `Tests/scratch/probe_z012_approach8_edge_filter.py` patches
`build_dataflow_graph` to compute the set of `(body_label, canonical_str)`
keys with multiple state-register-writing emissions, and drops every
state-typed edge whose producer matches such a key.

- On the synthetic repro: 0 failures. The 3 SCC edges (default) and 1
  SCC edge (CMS) are all dropped at construction; `compare_graphs` sees
  no state-register edges in either graph, so no diff.
- This is the cleanest construction-layer outcome of any catalog
  approach, but it has a hard cost: real semantic SCC dependencies
  between two genuinely-distinct lowerings would also be dropped. The
  current SCC clobber detector relies on `subj_graph.edges` being
  populated for SCC-typed edges so that `OverriddenInputFailure` can
  surface intervening writers.

**Verdict.** Works mechanically, but trades an `OrderInvertedFailure`
for a SILENT edge-loss. Project discipline (the validator's
no-silent-ignore contract per §"Discipline") explicitly rejects this
pattern. The SCC cross-body precedent (`SCC_CROSS_BODY_INVESTIGATION.md`,
theq) deleted suppression at the diagnosis layer in favor of NOT
EMITTING the artifact-edges in the first place — a stronger version of
this approach. But theq's case was qualitatively different: the
cross-body SCC edges are NEVER semantic dataflow (SCC is single-bit and
not preserved across loop iterations). The same-body collision-prone
edges in z012's case CAN be semantic dataflow (the cmp's SCC IS read by
the immediately following cselect — that IS the dataflow). Filtering
them at construction loses real edges; theq's filter dropped only
artifacts.

### Approach 9 — Collapse PACK + use category for non-PACK (PROPOSED, NOT IN BRIEF)

**Shape.** Identity = `(cls_tag, loop_idx, canonical_str, category_proj)`,
where `category_proj` is the category string EXCEPT every Pack* category
maps to a single `'PACK'` bucket. The §6 Pack-tagging divergence is
neutralized by intentional projection; non-Pack categories keep their
side-bearing labels and the GRInc cmp / m0 add collisions are
discriminated.

**Probe.** `Tests/scratch/probe_z012_approach10_pack_excluded.py`
implements the projection (the file is named "10" for sequencing —
treat as Approach 9 here).

- On the synthetic repro: 0 failures. Confirmed.
- On the unit suite under `test_dataflow_graph_*.py` (168 tests): all
  pass.
- On `test_ScheduleCapture.py` (real-kernel build path): **10 failures**.
  Sample failure: identity
  `('LRS', 0, 'v_xor_b32 v[vgprLocalReadAddrA], 0x10000, v[vgprLocalReadAddrA]', 'LRS')`
  is in reference but not subject. The DEFAULT side tags this LRSwap
  emission with category `'LRS'` (or some other non-LRSA value); the
  CMS side tags it `'LRSA'`. The §6 Pack-tagging divergence is NOT the
  only category-tagging divergence — LRSwap (LRSA/LRSB) exhibits the
  same pattern.

**Verdict.** Doesn't work in production. Reveals a SECOND
category-tagging divergence beyond §6's Pack* finding. The general
problem is that the default-side capture path
(`build_id_to_category_per_iter`, `_captureSubIterToBuilder`) and the
CMS-side capture path (`expand_cms_macro`, `tag_by_origin_id`) tag the
same physical instructions with potentially different category strings
across MULTIPLE category families, not just Pack*. Any identity
augmentation that uses category text directly is fragile until the
tagging strategies are reconciled.

The z012 memo's §6 finding is INCOMPLETE — it should be expanded to
"category-tagging-strategy divergence (multiple families)" rather than
"Pack-tagging-strategy divergence."

---

## 4. Comparative analysis

| Approach | Solves SCC GRIncA/B | Solves m0 GR A/B | Independent of category labels | Probe outcome | Blast radius (LoC) | Risk of masking real bugs | Aligns with discipline |
|---|---|---|---|---|---|---|---|
| 1 (position) | No | No | Yes | Breaks: identity-set diverges | small (~5) | None (raises instead) | Yes (raises loud) but unusable |
| 2 (no-collapse) | (would) | (would) | Yes | Skipped: comparator can't accept multi-node identities | very large (>500) | Low | No (re-architects core) |
| 3 (occurrence) | No | No | Yes | Breaks: 2 OrderInverteds remain | small (~10) | Low (no new edges) | Yes |
| 4 (topology) | No | No | Yes | Skipped (analytic): fingerprints identical cross-side | small | High (collapses real distinct ops) | No |
| 5 (context) | Partially | Partially | Yes | Breaks 1+ existing test | medium (~30) | High (non-deterministic identity) | No |
| 6 (no-collapse for state) | No (id() differs cross-process — well, cross-capture) | No | Yes | Breaks: identity diverges | small (~10) | None | Yes but unusable |
| 7 (separate validator) | (would) | (would) | Yes (no identity change) | Skipped (analytic): reduces to band-aid or parallel API | medium-large | Low | No (parallel API) |
| 8 (edge filter) | Yes | Yes | Yes | Works: 0 failures on synthetic | small (~30) | Medium (silent edge-loss) | No (silent suppression) |
| 9 (PACK-collapse + cat for rest) | Yes (synthetic) | Yes (synthetic) | NO — uses category | Breaks 10 real-kernel tests due to additional LRSwap divergence | small (~10) | None for caught failures | No (unmasks 2nd tagging bug) |

The "Independent of category labels" column is the brief's REQUIRED
short-list criterion. Approaches 1-8 satisfy it; approach 9 does not (it
projects categories but still uses category text in identity).

The "Probe outcome" column shows that **NO approach in the catalog
both solves the artifact AND preserves the existing test suite** without
SILENTLY suppressing edges (Approach 8) or REINTRODUCING the §6
divergence problem in expanded form (Approach 9).

---

## 5. Recommendation

**Primary recommendation: do not pursue any catalog approach. Address
the underlying category-tagging-strategy divergence as the prerequisite
to ANY identity restructuring.**

The reason every catalog approach fails or backslides is that THE
SAME-CANONICAL-STR COLLISION IS EXACTLY THE INFORMATION CARRIED BY THE
LOWERING-SOURCE TAG (i.e., the category). Every alternative discriminator
this investigation tested either (a) reduces to the lowering-source tag
(Approaches 5, 6, 9), (b) is too coarse (Approaches 1, 3, 4), or (c)
suppresses information rather than disambiguating it (Approaches 7, 8).

The lowering-source tag is the right discriminator. The only reason it's
"forbidden" is that it ISN'T STABLE CROSS-CAPTURE in production — that's
the actual bug. The z012 memo's §6 (Pack-tagging divergence) and this
investigation's discovery of an analogous LRSwap (LRSA/LRSB) divergence
together show this is a CLASS of bug, not a single instance.

The right sequence of work is:

1. **Audit and fix the category-tagging-strategy divergence across all
   category families.** Files to compare:
   `assign_id_to_category_for_default` / `build_id_to_category_per_iter`
   (default side, `ScheduleCapture.py:974-1074`) and `expand_cms_macro`
   + `build_idmap` / `invert_idmap_to_id_to_category` (CMS side,
   `ScheduleCapture.py:865-933`, `:2069-2180`). Both walk the same
   rocisa modules in different orders; the tagging output should agree
   per-instruction. A test fixture that runs both paths against the same
   FourPartCapture shape and asserts `default_id_to_category[id(inst)] ==
   cms_id_to_category[id(inst)]` per inst would catch both Pack* and
   LRS* divergences and any future ones.

2. **After the tagging is reconciled, the original principled fix
   (extend `identity_for` with category as 4th element) becomes viable.**
   This investigation confirms it works on the synthetic repro
   (Approach 9 / probe10 showed 0 failures), and it would pass all
   existing tests once the LRS divergence is resolved alongside Pack.

3. **`example.yaml` returns to green** without any carve-out at any
   layer. The validator stays principled.

**Backup recommendation: if the user decisively will not do step 1 this
cycle, the least-bad construction-layer alternative is Approach 8 (edge
filter).** It has clear scope (state-register edges in same-body
collision-prone bodies), works on the synthetic repro, doesn't touch
identity semantics, and has a clean single-responsibility commit. The
honest cost: it silently drops edges that downstream code would have
classified, and represents a project-discipline violation in the same
genus as the rejected diagnosis-layer carve-out (suppression, not fix).
A second-best option pinned for this purpose is Approach 8. I would not
recommend implementing it; I would file a follow-up bead for the
tagging audit instead.

**Anti-recommendation: do NOT implement Approaches 1, 3, 5, 6, or 9.**
Each fails one of the following: (a) breaks the identity-coverage check
at `compare_graphs` entry by making identity unstable cross-capture, (b)
fails to disambiguate the actual collision, or (c) reveals adjacent
divergences that will appear in production tests.

---

## 6. Open questions for the user

1. **Is the category-tagging-strategy divergence audit (per §5 step 1)
   acceptable as the prerequisite for any identity restructuring?**
   This is the load-bearing question. If yes, the original principled
   fix (extend `identity_for` with category) becomes viable after the
   audit and this entire alternative-fixes investigation is moot. If
   no, the user is constrained to Approach 8 (edge filter) or to
   continuing to absorb `example.yaml`'s 8 failures via the
   already-rejected diagnosis-layer carve-out.

2. **Is silently dropping construction-layer edges (Approach 8) more
   acceptable than gating diagnosis-layer failures (the rejected z012
   carve-out)?** Both are suppressions; both lose information. The
   difference is layer (early vs late) and what's lost (an edge vs a
   failure surfacing). If the user considers Approach 8's
   construction-layer edge-loss more principled than the diagnosis-layer
   gating because it matches the SCC cross-body precedent (theq)
   structurally, then Approach 8 is a coherent forward path. If not,
   only step 1 (tagging audit) remains as a real fix.

3. **Was the §6 finding's framing as "Pack-tagging-strategy divergence"
   too narrow?** This investigation found at least one additional
   divergence (LRSA/LRSB on LRSwap) in the default-vs-CMS tagging.
   Should §6 be re-scoped to "category-tagging-strategy divergence
   (multiple families)" before any follow-up bead is filed? Re-scoping
   matters because (a) the audit work is ~2x the size if there are
   really two families to fix, and (b) any future "we'll only fix
   Pack*" patch would be incomplete.

---

## 7. Things that surprised me

- **All eight catalog approaches fail or backslide.** The brief
  presumed at least one would be viable; my investigation shows the
  problem is genuinely under-determined without a stable cross-capture
  side discriminator. The user's intuition that the principled fix
  needs to wait on Pack-tagging reconciliation is RIGHT — and stronger
  than the brief acknowledged: it needs to wait on a class of tagging
  fixes, not just Pack.
- **Approach 5 (context window) initially seemed to work** on the
  synthetic repro and surprised me by failing on existing SCC clobber
  tests when the producer is the last instruction in its body. The
  asymmetry of "successor depends on what comes next, which itself can
  vary cross-capture" is fatal, and not obvious until the test suite
  runs.
- **Approach 9 (PACK-bucketed category) revealed the LRS divergence**
  that hadn't been called out in z012's §6. Without running the real-
  kernel test suite, I'd have signed off on Approach 9 as the
  recommended fix. Lesson: the §6 finding is a strict subset of the
  divergence, and the audit work in step 1 would surface more.
- **Approach 8 works mechanically but feels wrong** in a way the SCC
  cross-body precedent (theq) doesn't. The difference: theq's filter
  removes edges that are pure resolver artifacts (cross-body SCC has no
  semantic meaning); Approach 8's filter removes edges that ARE
  semantic dataflow within the body. The "right layer" framing from
  theq doesn't transfer cleanly because the artifacts here are
  intermingled with real dataflow.

---

## 8. Files referenced

- `Tensile/Components/CMSValidator.py:1266` — collision site
  (`nodes_by_identity[node.identity] = node`).
- `Tensile/Components/CMSValidator.py:733-736` — `edge_keys` (uses
  `producer.position`).
- `Tensile/Components/CMSValidator.py:2671-2754` — `compare_graphs` and
  identity-coverage check at entry (`_data_flow_ids`).
- `Tensile/Components/CMSValidator.py:2757-3047` — `diagnose_missing_edge`
  (Phase 0 missing-node raise; Phase 1 OrderInverted; SCC clobber branch).
- `Tensile/Components/ScheduleCapture.py:507-536` — `identity_for`.
- `Tensile/Components/ScheduleCapture.py:974-1074` — default-side
  `build_id_to_category_per_iter`.
- `Tensile/Components/ScheduleCapture.py:865-933` — `build_idmap` /
  `invert_idmap_to_id_to_category`.
- `Tensile/Components/ScheduleCapture.py:2069-2180` — CMS-side
  `expand_cms_macro` (where `tag_by_origin_id` is consumed).
- `Tensile/Components/EXAMPLE_YAML_DEFECT_INVESTIGATION.md` (z012,
  primary memo).
- `Tensile/Components/CROSS_SUBITER_ALU_FP_INVESTIGATION.md` (bwfr).
- `Tensile/Components/SCC_CROSS_BODY_INVESTIGATION.md` (theq).
- Probes (uncommitted, in `Tests/scratch/`):
  - `probe_z012_baseline.py` — synthetic repro (2
    `OrderInvertedFailure`s on collision).
  - `probe_z012_approach1_position.py` — slot-based identity → identity
    set diverges.
  - `probe_z012_approach3_occurrence.py` — occurrence-index → 2
    failures persist.
  - `probe_z012_approach5_context.py` — successor-render in identity
    → 0 failures on synthetic.
  - `probe_z012_approach5_pytest.py` — Approach 5 vs unit suite → 1
    failure.
  - `probe_z012_approach6_no_collapse_state.py` — id()-tagged identity
    → identity set diverges cross-capture.
  - `probe_z012_approach8_edge_filter.py` — edge-filter at construction
    → 0 failures on synthetic.
  - `probe_z012_approach10_pack_excluded.py` — category projection
    excluding Pack → 0 failures on synthetic.
  - `probe_z012_approach10_pytest.py` — Approach 9 vs real-kernel suite
    → 10 failures (LRS divergence).
