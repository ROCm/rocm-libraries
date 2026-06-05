# Plan — Transition to Unrolled-Timeline Validation

**Status:** revised 2026-06-02 incorporating critic findings + NLL structure investigation + user directives on hack-deletion, identity, and primary keying.

**Supersedes:** earlier feature-flag-first revision (rejected: backwards-compat bloat).

**Companion artifacts:**
- `6QIB_DESIGN.md` — design + decision doc, including the §0.7 / §0.8 / §0.9 framing
- `6QIB_ALGORITHMIC_REVIEW.md` — algorithmic alternatives (SSA value-numbering framing)
- `n7og_PROBE_REPORT.md` — empirical mechanism findings + exemption-classification probe (N3=192/192)
- `NLL_STRUCTURE_INVESTIGATION.md` — NLL is single-invocation; `kernel["NoLoadLoopIter"]` does not exist
- `UNROLLED_VALIDATION_PLAN_CRITIQUE.md` — critic's review of the prior plan revision (Q1-Q10)

---

## 0. TL;DR

Replace the validator's identity-tuple-keyed body-local-walk model with an unrolled-timeline + byte-key-keyed model. Delete the cross-subiter ALU-producer exemption FIRST, then land the principled fix across the following commits. Validator is RED for the duration of the work, by design — "validator-honest at every commit" replaces "validator-green at every commit."

All twelve §9 design questions are now RESOLVED with concrete answers (see `UNROLLED_VALIDATION_ANSWERS.md` for empirical evidence + `Q4_SPURIOUS_EDGE_CLASSIFIER_DESIGN.md` for the Q4 classifier deep-dive). Key resolved choices baked into the plan:
- `ML_MAT_COUNT = 2` hardcoded constant + `assert kernel["PrefetchGlobalRead"] == 2` at the validator entry. Anything else fails loud and forces re-investigation rather than silently running with an unverified count
- Identity stays iter-blind 3-tuple; only its dict-keying role migrates
- Eager byte-key reverse index built alongside `latest_writer` (one-line addition in Phase 2)
- Q4 classifier: Option C hybrid (byte-key-driven multi-phase) with three terminal outcomes — `CaptureConsistencyError`, `EdgeRoutedDifferentlyFailure` (renamed from `SpuriousEdgeFailure`), `UnexplainedExtraEdgeError`

Two beads filed as P0 blockers on `r62g`:
- `rocm-libraries-tne8` (rescoped) — ML iter-materialization count for graph coverage. NLL is not materialized (NLL is single-invocation per `NLL_STRUCTURE_INVESTIGATION.md`).
- `rocm-libraries-67us` (rescoped) — `compare_graphs` `subj − ref` symmetric direction + `EdgeRoutedDifferentlyFailure` classifier.

A larger architectural shift sits beneath the visible commits: identity-tuple-as-primary-key migrates to byte-key-as-primary-key. Identity becomes a diagnostic annotation, not a lookup basis. This is required by the combination of: (a) iter-blind identity (pipelining must cancel); (b) the `subj_graph.nodes` dict not collapsing iter copies; (c) cross-graph comparison resolving dataflow-equivalent edges to the same key regardless of which iter they ended up in.

---

## 1. Constraints (binding for plan and implementation)

These constraints together define what counts as an acceptable implementation. The implementing agent must reject any approach that violates any of them.

| Constraint | Source | Operational implication |
|---|---|---|
| Validate the schedule as one singular timeline of instructions | User directive (`feedback_validate_as_single_timeline`) | Body labels are codegen organization; the dataflow model walks the unrolled program as a linear chain |
| Delete hacks immediately, no safety net | User directive (`feedback_delete_hacks_immediately`) | The cross-subiter ALU-producer exemption at `CMSValidator.py:3831-3843` is deleted in commit 1 |
| No backwards-compatibility bloat | Standing rule (`feedback_no_tactical_fixes`) | No feature flag, no kwarg preserving old behavior, no transition-period two-path code |
| No deferred discoveries | Standing rule (`feedback_no_deferred_discoveries`) | Real defects surfaced during implementation get filed as P0 with `br dep add r62g`; not "TODO for later" |
| Identity stays iter-blind | User directive (2026-06-02) | Pipelined instructions that move between iters must produce identical identity tuples so set-diff cancels them; `(canonical_render, source_module_id, emission_ordinal)` is the existing shape and stays as-is |
| Don't quote LoC numbers | User directive (`feedback_no_loc_numbers`) | Plans / reports / commit messages / bead descriptions use semantic scope descriptions, not line counts |
| NLL is single-invocation; do NOT materialize copies | `NLL_STRUCTURE_INVESTIGATION.md` (2026-06-02) | `FourPartCapture.n_ll = {0: body}` is intentional and correct; the NLL body already contains the full subiter chain via the internal `for uIdx in range(0, kernel["LoopIters"])` at `KernelWriter.py:3508` |
| ML materialization is for graph coverage, not faithful unrolling | Combined directive | Materialize ML some small number of times (likely 2 or 3) so the cross-iter producer-consumer dependencies become visible in the unrolled stream; do NOT replicate `kernel["LoopIters"]` faithfully (the validator cares about dependency-structure shape, not production iter count) |

---

## 2. Data-model design

### 2.1 The unrolled-timeline representation

Replace `DataflowGraph.nodes: Dict[identity_tuple, GraphNode]` with an ordered sequence indexed by position in the unrolled stream. The exact container is an implementation detail; the contract is:

- Iteration order matches stream execution order (PRO first, then ML's first iter copy, then ML's next iter copy, then NGL, then NLL, then POST)
- Multiple GraphNodes with the same `identity` (different iter copies of the same `TaggedInstruction`) coexist; no squashing
- Each GraphNode carries its `unrolled_position`, the body-label annotation it came from, and (for ML iter copies) the iter index

Identity stays `(canonical_render, source_module_id, emission_ordinal)` — iter-blind, exactly as today.

### 2.2 Byte-key as primary keying basis

Identity-tuple-as-primary-key gets replaced with byte-key-as-primary-key across the validator's lookup surface:

**Today:**
- `subj_graph.nodes.get(p_id)` finds producer by identity. Used in `diagnose_missing_edge` Phase 0.
- `DataflowGraph.edge_keys()` returns tuples containing `(producer.identity, consumer.identity, ...)`.

**After:**
- A byte-key reverse index alongside the node sequence: `Dict[byte_key, List[(GraphNode, unrolled_position)]]`. Built during `build_dataflow_graph` Phase 2 walk (same data `latest_writer` already maintains).
- `edge_keys()` returns tuples containing `(producer_write_byte_keys, consumer_read_byte_keys, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`. Allocation-invariant; pipelined edges that move between iters produce identical keys on both sides.
- `diagnose_missing_edge` lookups switch to byte-key reverse-index queries. "Find a producer in subj_graph that emits this byte-key write at a position consistent with the ref_edge's expected ordering."
- Identity becomes a label for diagnostics: `cms_node_label` and human-readable failure formatters consume `node.canonical_render + node.unrolled_position` (or `node.iter_index` for ML iter copies).

This is the algorithmic-review's SSA value-numbering framing: the value-number is the byte-key flow; canonical_render is the operand name (annotation); emission_ordinal is the position (annotation).

### 2.3 What the unrolled stream contains

Concatenation, in execution order:

```
PRO → ML_iter[0] → ML_iter[1] → ... → ML_iter[ML_MAT_COUNT-1] → NGL → NLL → POST
```

Where:
- `PRO`, `NGL`, `NLL`, `POST` appear once each (NLL's internal subiter loop is already physicalized inside the single body)
- `ML_iter[k]` reuses the underlying `TaggedInstruction` objects from the captured ML body; each iter copy stamps a distinct `unrolled_position` (and `iter_index = k`); identity stays the same across copies
- `ML_MAT_COUNT = 2` — hardcoded module constant in `CMSValidator.py`. Guarded by `assert kernel["PrefetchGlobalRead"] == 2` at the validator entry point (the inline xj16 site at `KernelWriter.py:6292`). If a kernel arrives with a different prefetch depth, the assertion fires — the validator has only been empirically verified for `PrefetchGlobalRead=2`, and silently running with a different prefetch would be unverified-behavior. Re-investigation required to extend coverage. No formula, no parameterization — fail-loud-and-revisit. See §9 Q1.

### 2.4 Body labels survive as annotations

Every existing diagnostic consumer of `body_label` (`cms_node_label`, `render_node_label`, the dump tool, the matplotlib visualization) continues to work. Each GraphNode carries `body_label` derived from which capture body it materialized from. Body labels never gate dataflow resolution; the unrolled walk treats them as descriptive metadata only.

---

## 3. `build_dataflow_graph` rewrite

### 3.1 Phase 1 — node construction over the unrolled stream

`build_dataflow_graph` calls `UnrolledCapture.from_four_part_capture(fpc)` to get the unrolled sequence (see §2.3). For each `TaggedInstruction` in the sequence, instantiate one `GraphNode` (each iter copy gets its own Python object — distinct `id(node)`, same `node.identity` for iter copies of the same source instruction).

`DataflowGraph.nodes` becomes the position-indexed sequence; no identity-keyed dict.

### 3.2 Phase 2 — single `latest_writer` walk over the unrolled stream

One linear pass. `latest_writer` initialized empty before PRO; updated by every write; queried by every read. No per-body or per-iter resets except the existing SCC-boundary clearing (barriers reset SCC's `latest_writer` entry — that behavior is preserved).

Build the byte-key reverse index alongside: every write into `latest_writer` also appends `(node, unrolled_position)` to `byte_key_writers[bk]`.

`_resolve_producers`'s grouping key `(id(writer_node), id(write_resource), write_slot)` stays — it correctly produces one edge per writer group regardless of which iter the writer is in (different iter copies have different `id(writer_node)`, so they form distinct groups; same iter's bytes from the same writer coalesce into one wide edge).

### 3.3 Cross-body / cross-iter live-ins resolve naturally

Worked example from BPG#11 (the case that motivated the rewrite):

- SHADOW unrolled stream: `... ML_iter[K] writes via PackB-linear-emission ... ML_iter[K+1] writes via PackB-linear-emission ... NLL stream_index 7-10 writes via PackB0 ... NLL stream_index 14 MFMA reads ...`
- CMS unrolled stream: `... ML_iter[K] writes via PackB3-pipelined ... ML_iter[K+1] writes via PackB3-pipelined ... NLL stream_index 0 MFMA reads (latest_writer was populated by ML_iter[ML_MAT_COUNT-1] PackB3 writes — cross-body, cross-iter live-in resolved correctly) ... NLL stream_index 84-88 writes via PackB3-pipelined`

Both sides emit edges from their respective most-recent producers. Under iter-blind identity + byte-key edge_keys, those edges cancel in set-diff because:
- Producer byte_key is `('v', 31..34)` on both sides
- Consumer byte_key is `('v', 31..34)` on both sides
- edge_kind, intra_offset, slots are all identical
- The edge_keys themselves are byte-equal → set-diff cancels them

The 192 BPG#11 NLL extras and 16 NGL missing edges both resolve to 0.

### 3.4 Edge attribution

`DataflowEdge` gains diagnostic annotations: `producer_unrolled_position`, `consumer_unrolled_position`, `producer_body_label`, `consumer_body_label`, `producer_iter_index`, `consumer_iter_index`. These are for human display and Phase 1 order-check arithmetic only; they do NOT appear in `edge_keys()`.

---

## 4. `compare_graphs` + `diagnose_missing_edge` changes

### 4.1 Symmetric direction + `diagnose_extra_edge` classifier

Currently `compare_graphs` (`CMSValidator.py:3711-3713`) computes `missing_keys = ref_keys - subj_keys` only. The plan adds `extra_keys = subj_keys - ref_keys` processing. Tracks bead `rocm-libraries-67us`.

`diagnose_extra_edge` design — **Option C hybrid** (byte-key-driven multi-phase) per `Q4_SPURIOUS_EDGE_CLASSIFIER_DESIGN.md`. Phase structure mirrors `diagnose_missing_edge` for reviewer familiarity; every branch decision is byte-key reverse-index-driven (no class lists, no subiter arithmetic, no pattern matching — explicitly avoids the m7o5 anti-pattern).

**Three terminal outcomes:**

| Outcome | When | Disposition |
|---|---|---|
| `CaptureConsistencyError` (raised) | subj writes byte_keys ref has no writer for | Capture-pipeline bug — violates `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3` same-instruction-set contract |
| `EdgeRoutedDifferentlyFailure` (new typed Failure) | ref has writers, but ref's closest-prior writer for subj_edge's byte_keys differs in identity from subj's producer | Structurally a clobber; symmetric counterpart of `OverriddenInputFailure`; real CMS defect; **hard-fail** |
| `UnexplainedExtraEdgeError` (raised) | fall-through OR ref's closest-prior matches subj's identity exactly | Validator bug (the edge should have cancelled in set-diff but didn't) |

**Naming note:** earlier drafts called the typed Failure `SpuriousEdgeFailure`. Renamed to `EdgeRoutedDifferentlyFailure` because "spurious" presupposes the edge is wrong; per §3 the actual defect for surviving extras is the ROUTING, not the edge's existence.

**Failure rendering:** `EdgeRoutedDifferentlyFailure.format()` cites `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3` (the contract the failure violates) and shows one representative byte's routing divergence + a total-affected-bytes count, rather than enumerating every byte.

**SCC handling subsumed:** the existing `diagnose_missing_edge` SCC clobber branch (`CMSValidator.py:~3859-3870`) becomes generic under the byte-key model — a CMS-introduced SCC clobber surfaces as `EdgeRoutedDifferentlyFailure` with byte_key `('s', 'scc')`. No special SCC branch in the new classifier.

### 4.2 Phase 1 order check (over unrolled positions)

Today's Phase 1 (`CMSValidator.py:3826-3858`) gates on `if p_node.body_label == c_node.body_label` and uses `p_node.position < c_node.position` within that body.

After: the gate disappears (the unrolled stream is a single position space). The check becomes `if p_node.unrolled_position < c_node.unrolled_position` etc. Cross-body and cross-iter inversions surface naturally.

### 4.3 Phase 0 lookup migrates to byte-key reverse index

Today's Phase 0 (`CMSValidator.py:3776-3783`) does `subj_graph.nodes.get(p_id)`. After: `subj_graph.byte_key_writers[ref_edge.producer_write_byte_keys[0]]` returns candidate producers; the classifier picks the one consistent with the missing edge's expected position relationship.

If no candidate matches, the edge is genuinely absent — the failure surfaces honestly.

**Concrete migration surface** (per `UNROLLED_VALIDATION_ANSWERS.md §Q3`): exactly 3 `.nodes.get(...)` call sites in `CMSValidator.py`:

| Site | Current use | Migration |
|---|---|---|
| `:3693` | Iteration over identity-keyed dict | Iterate the ordered sequence directly |
| `:3773` | Phase 0 producer lookup | Byte-key reverse-index query |
| `:3774` | Phase 0 consumer lookup | Byte-key reverse-index query |

Plus 4 write-side `nodes_by_identity` insertion sites in `_make_node` and surroundings — replaced by append into the ordered sequence + append into the byte-key reverse index.

### 4.4 Cross-subiter ALU-producer exemption — DELETED

Commit 1 deletes `CMSValidator.py:3831-3843` outright. No safety net, no preservation.

After deletion, the 192 BPG#11 SHADOW-extras surface as `OrderInvertedFailure` from Phase 1. These failures are HONEST — they expose the body-local walk's blind spot the exemption was hiding. Per the standing rule, the validator runs RED until the unrolled walk lands and resolves them.

### 4.5 Downstream consumers of `body_label` migrate

Per the Q2 + Q11 investigation in `UNROLLED_VALIDATION_ANSWERS.md`:

- `cumulative_issue_cycles` (`CMSValidator.py:2495-2604`) — migrate the body-discovery scaffolding at `:2493-2533`, `:2543-2569`, and `:2600` to walk the unrolled stream. The simulator's core arithmetic (MFMA contention, type-switch +1) is body-agnostic and unchanged. No existing `test_cumulative_issue_cycles_*.py` files exist; the relevant tests live in `test_dataflow_graph_register_gaps.py::TestCumulativeIssueCycles` and use synthetic captures that don't exercise cross-iter ML→ML — so zero re-pinning required.
- `_BODY_BUILD_ORDER` and `body_for` helpers — repurpose as diagnostic annotations only.
- Other `body_label`-touching control flow: 75 total grep hits across `CMSValidator.py`; ~30 control-flow sites need migration to unrolled-stream iteration, ~45 are diagnostic-annotation-only (no migration). Concrete enumeration in `UNROLLED_VALIDATION_ANSWERS.md §Q11`.

---

## 5. Sequencing

Delete the hack first. Land the principled fix second. No safety net, no feature flag, no transition period.

### Commit 1 — Delete the cross-subiter ALU-producer exemption + introduce `ML_MAT_COUNT` + PrefetchGlobalRead assertion

Delete `CMSValidator.py:3831-3843` (the entire exemption block including the `_is_alu_producer` check and the `subiter` lookup).

Add `ML_MAT_COUNT = 2` as a module constant in `CMSValidator.py` (not consumed yet — that lands in C3a).

Add `assert kernel["PrefetchGlobalRead"] == 2` at the validator entry (the inline xj16 site at `KernelWriter.py:6292`, or whichever site has `kernel` in scope at the start of validation). The assertion message cites `UNROLLED_VALIDATION_PLAN.md §9 Q1` so anyone tripping it knows where to look.

Add a regression assertion: `grep "cross-subiter ALU-producer exemption" Tensile/Components/CMSValidator.py` returns 0 matches.

Run the full unit suite. Classify every surfaced failure as:
- **(a)** Representational gap the unrolled walk will resolve → expected; tracked under the broader work; tests stay RED until the unrolled-walk commits land
- **(b)** Test that was structurally pinning the exemption's silencing behavior → re-fixture (deferred to commit 4)
- **(c)** Real bug the exemption was wrongly silencing → file P0 bead with `br dep add r62g <new-bead>`; do NOT re-fixture to make it pass

**Validator state after this commit:** RED. The 192 BPG#11 SHADOW-extras and equivalent failures on other fixtures surface honestly. This is correct per `feedback_delete_hacks_immediately` — these tests were always failing; the hack was hiding it.

### Commit 2 — REMOVED (collapsed into C1 / not needed)

Earlier drafts of this plan threaded prefetch parameters through `FourPartCapture` to drive an `ML_MAT_COUNT` derivation formula. The user-corrected approach (per §2.3) is: hardcode `ML_MAT_COUNT = 2` + assert `kernel["PrefetchGlobalRead"] == 2` at the validator entry. No formula, no `FourPartCapture` threading, no separate commit. The constant + assertion fold into Commit 1 alongside the exemption deletion (the assertion guards the RED-state validator from accidentally running on unverified prefetch values).

### Commit 3 — DECOMPOSED into 3a-3h

Originally framed as a single monolithic commit bundling the full rewrite. Decomposed into a chain of smaller commits so each is independently reviewable and bisectable. Each sub-commit has its own bead; the dependency graph is enforced via `br dep add`.

**Commit 3a — `UnrolledCapture` materializer + tests (no validator wiring)**

Add `UnrolledCapture` and `UnrolledIterRecord` classes in `ScheduleCapture.py`. The materializer takes a `FourPartCapture` and produces the unrolled sequence per §2.3 using `ML_MAT_COUNT = 2`. No validator code consumes it yet — purely additive infrastructure.

Add unit tests verifying materialization: PRO/NGL/NLL/POST appear once each; ML appears `ML_MAT_COUNT` times; verify `unrolled_position` monotonicity; verify each iter copy of an ML node shares `identity` with the other copies (iter-blind identity contract).

Validator state: unchanged (still RED from C1).

**Commit 3b — `DataflowGraph.nodes` refactor to ordered sequence + byte-key reverse index**

Replace `DataflowGraph.nodes: Dict[identity_tuple, GraphNode]` with an ordered sequence indexed by `unrolled_position`. Add a parallel `byte_key_writers: Dict[byte_key, List[(GraphNode, unrolled_position)]]` built alongside the existing `latest_writer` walk (eager construction).

Migrate all current `nodes.get(...)` / `nodes.values()` / `nodes[...]` call sites in `CMSValidator.py` per the table in §4.3.

Validator state: still RED from C1 + C3a; the rewrite path is partly in place but `build_dataflow_graph` hasn't switched over to consume `UnrolledCapture` yet.

**Commit 3c — `build_dataflow_graph` rewrite consuming `UnrolledCapture`**

Switch `build_dataflow_graph` to call `UnrolledCapture.from_four_part_capture(fpc)` and walk the result. Phase 1 walks the unrolled iter sequence producing one `GraphNode` per stream position. Phase 2 single `latest_writer` walk + populates the byte-key reverse index from C3b.

`DataflowEdge` gains diagnostic annotations: `producer_iter_index`, `consumer_iter_index`, `producer_body_label`, `consumer_body_label`.

Validator state: still partly RED. The walk is correct but `edge_keys()` and `compare_graphs` haven't switched to byte-key basis yet, so cross-graph comparison still uses identity-tuple keying and the RED state from C1 persists.

**Commit 3d — `edge_keys()` byte-key basis migration**

`DataflowGraph.edge_keys()` returns tuples `(producer_write_byte_keys, consumer_read_byte_keys, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`. Allocation-invariant. Iter-blind. Pipelined edges that move between iters produce identical keys on both sides.

Validator state: probable transition to mostly GREEN here — the n7og probe fixtures should now resolve to 0 mismatches in the `ref − subj` direction. n7og xfail removal is deferred to C3h so any regression here surfaces as a test failure rather than being masked by xfail.

**Commit 3e — `compare_graphs` symmetric direction + `EdgeRoutedDifferentlyFailure` classifier**

Tracks bead `rocm-libraries-67us`.

Add `extra_keys = subj_keys - ref_keys` processing. Add `EdgeRoutedDifferentlyFailure` typed Failure class + `UnexplainedExtraEdgeError` exception. Implement `diagnose_extra_edge` as Option C hybrid byte-key-driven multi-phase classifier (per §4.1 + `Q4_SPURIOUS_EDGE_CLASSIFIER_DESIGN.md`) with the three terminal outcomes (`CaptureConsistencyError` raised, `EdgeRoutedDifferentlyFailure` produced, `UnexplainedExtraEdgeError` raised).

Validator state: fully covers both `ref − subj` and `subj − ref` directions. On current fixtures the classifier rarely fires (per Q10).

**Commit 3f — `diagnose_missing_edge` Phase 0/1 migration**

Phase 0 lookup migrates to byte-key reverse-index queries (replaces `subj_graph.nodes.get(p_id)` per §4.3 + the migration table). Phase 1 order check migrates to unrolled-position comparisons (the `if p_node.body_label == c_node.body_label` gate is removed; the unrolled stream is a single position space).

Validator state: missing-edge classifier is fully consistent with the new graph structure.

**Commit 3g — `cumulative_issue_cycles` + remaining `body_label` control-flow migration**

Migrate the 3 body-discovery scaffolding sites in `cumulative_issue_cycles` (`CMSValidator.py:2493-2533`, `:2543-2569`, `:2600`) to walk the unrolled stream. Migrate the ~30 control-flow `body_label` consumers per §4.5 + `UNROLLED_VALIDATION_ANSWERS.md §Q11`. The ~45 diagnostic-annotation-only sites are unchanged.

Validator state: unchanged from C3f.

**Commit 3h — n7og xfail removal + new cross-iter / cross-body unit tests**

Remove the xfail markers from `test_n7og_edge_keys_multifixture.py` for BPG#11 and oplb-style fixtures (they should now resolve to 0 mismatches in both directions). Add new unit tests:
- Cross-iter live-in: ML iter 0 writes → ML iter 1 reads, edge with correct iter annotations
- Cross-body live-in: PRO writes → ML iter 0 reads, edge with correct body annotations
- `EdgeRoutedDifferentlyFailure` classifier outcomes

Validator state: GREEN for principled reasons. n7og fixtures pass without xfail. Inline xj16 validation passes on TF32+UsePLRPack production builds (with the `PrefetchGlobalRead=2` assertion satisfied).

### Commit 4 — Re-fixture the (b)-class tests

Re-fixture every test that was structurally pinning the exemption's silencing behavior (enumerated during commit-1 classification). Each test either:
- Gets its assertion updated to expect the unrolled-walk's correct behavior, OR
- Gets deleted if its semantics are obsolete under the unrolled model

Any test that turns out to be (c)-class (real bug, not pinning the silencing) gets filed as a P0 bead per the standing rule — DO NOT re-fixture to make it pass.

**Validator state after this commit:** all tests pass cleanly. No remaining red-flag patterns (no `setdefault`, no defensive classifications, no test skips, no exemptions).

---

## 6. Test surface impact

### 6.1 Existing tests on the break list

The 11 tests on `6QIB_DESIGN.md §0.8 / §2.1`'s break list:

| Test | Classification | Disposition |
|---|---|---|
| `test_validate_pack_graph.py::test_pack_before_swap_orderinverted` | (a) | passes unchanged — pure within-body reorder unaffected by unrolled walk |
| `test_ValidateSCCoverlap.py::*` (5 tests) | (a) | passes unchanged — SCC-clobber semantics intact |
| `test_validate_gr_not_too_early_graph.py::TestGRNotTooEarlyDtlPlusLdsBufGraph::test_negative_one_prev_iter_lr0_not_drained` | (a) | passes unchanged |
| `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_validates_clean_with_carveout_engaged` | (b) | re-fixture: assert clean from unrolled walk, no exemption needed |
| `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures` | (b) | re-fixture or delete |
| `test_cross_subiter_pack_artifact.py::test_carveout_suppresses_artifact_and_neutralization_surfaces_it` | (b) | re-fixture |

Companion tests in `test_cross_subiter_pack_artifact.py` (`test_artifact_present_in_default_graph`, `test_correct_edge_present_in_cms_graph`) are borderline (b)/(c). They assert the artifactual `PackA1 → MFMA` edge in the default graph. Under the unrolled walk this assertion may flip (the more-recent prior writer in the unrolled stream becomes `PackA0`, not `PackA1`). Re-evaluate against actual unrolled-walk output during commit 4. If the new behavior IS the correct dataflow, re-fixture. If the old behavior was actually correct and the unrolled walk is wrong, file as a bug.

### 6.2 Tests not on the break list

Tests that don't trigger the exemption don't break when commit 1 deletes it, by construction. Tests that don't exercise the rewritten code paths in commits 2-3 don't change behavior. No pre-audit needed; pytest's assertions catch behavior changes that matter. If a test fails unexpectedly during the work, investigate at that point.

(Previous draft of this section recommended a baseline-snapshot diff strategy — removed; see §9 Q5 for the rationale.)

### 6.3 New tests required

- `UnrolledCapture` materialization: PRO/NGL/NLL/POST appear once each; ML appears ML_MAT_COUNT times; verify `unrolled_position` monotonicity; verify each iter copy of an ML node shares `identity` with the other copies
- Cross-iter live-in: ML iter 0 writes → ML iter 1 reads, edge formed with correct `producer_iter_index` / `consumer_iter_index` annotations
- Cross-body live-in: PRO writes → ML iter 0 reads, edge formed with correct `producer_body_label="PRO"` / `consumer_body_label="ML"` annotations
- `diagnose_extra_edge` classifier: subject emits an edge reference doesn't have; verify each terminal outcome — `CaptureConsistencyError` raised when ref has no writer for the byte_keys; `EdgeRoutedDifferentlyFailure` produced when ref has writers but identity differs (the clobber case); `UnexplainedExtraEdgeError` raised on fall-through
- BPG#11 / oplb-style fixtures: edge-keys symmetric set-diff is empty in both directions (replaces the current xfailed probe assertion)

### 6.4 n7og xfail removal

The n7og probe's xfail markers come off in commit 3 (when the unrolled walk lands). `strict=True` is set, so if the unrolled walk is incomplete the markers stay (the suite reports failures honestly).

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ML materialization count is wrong (see §9 Q1) | Low — derivation formula parameterized on prefetch depth | Default value 2 matches current TF32+UsePLRPack fixtures (`PrefetchGlobalRead=2`); higher prefetch values automatically increase the count via the derivation formula. If a fixture surfaces an unexpected `EdgeRoutedDifferentlyFailure` whose root cause is cross-iter span beyond `ML_MAT_COUNT`, refine the formula |
| `subj_graph.nodes` migration misses a call site | Resolved (Q3 enumeration: 3 read sites + 4 write sites in `CMSValidator.py`) | Migration table in §4.3; verify with grep again before commit 3 lands |
| `cumulative_issue_cycles` migration produces wrong cycle counts under unrolled walk | Low (Q2 / Q12: no current tests pin cycle counts that would change) | New tests pin behavior; existing tests use synthetic captures unaffected by ML iter copies |
| `EdgeRoutedDifferentlyFailure` classifier (Q4-resolved Option C hybrid) misses an extra-edge subcategory (legitimate CMS-introduced barriers, SCC-clobber paths, etc.) | Low | Three terminal outcomes (CaptureConsistencyError / EdgeRoutedDifferentlyFailure / UnexplainedExtraEdgeError) form a closed partition; SCC-clobber paths subsume under byte_key `('s', 'scc')`. Any case the classifier can't categorize raises `UnexplainedExtraEdgeError` (validator-bug class), not silent suppression |
| (c)-class real bugs surface during commit 4 re-fixturing | Medium | Standing rule: file P0 bead with `br dep add r62g`, do NOT make the test pass by re-fixturing |
| The byte-key reverse index has hidden complexity (sparse mem keys, SCC sentinel, symbolic-unresolved names) | Medium | Build incrementally; test each byte-key type independently |
| The unrolled walk produces unexpected new edges (legitimate, but never seen before) that we're not sure how to classify | Low | Per the standing rules, these are HONEST surfaced behavior; classify each and either re-fixture tests or file bugs |
| Per-body diagnostics (matplotlib visualization, dump tool) break under the unrolled model | Low | The annotations are preserved; only consumers that assumed per-body structure need updates |
| Memory cost of materializing ML some count of times | Low | Validator runs once per kernel; sub-second cost compared to actual codegen+compile |

---

## 8. Acceptance criteria

The work is done when:

- `CMSValidator.py:3831-3843` (the exemption block) is deleted; `grep` returns 0 matches.
- `compare_graphs` checks both `ref − subj` AND `subj − ref` directions; `EdgeRoutedDifferentlyFailure` is a typed Failure produced by `diagnose_extra_edge`; `CaptureConsistencyError` and `UnexplainedExtraEdgeError` raise per §4.1's three-terminal-outcome contract.
- `DataflowGraph.nodes` is an ordered sequence indexed by unrolled position; no identity-keyed dict remains in the validator.
- `edge_keys()` returns byte-key-based tuples; no identity-tuple references remain in the keying basis.
- `build_dataflow_graph` walks one unrolled stream; no per-body dispatch in the walk logic.
- The n7og probe tests pass without xfail markers on BPG#11, oplb-style, AND bf16.
- Inline xj16 validation passes on every TF32+UsePLRPack production build.
- No new red-flag patterns: no `setdefault`, no defensive classifications, no test skips, no exemptions.
- The (b)-class tests have been re-fixtured to assert the unrolled-walk's correct behavior.
- Any (c)-class real bug surfaced during the work is filed as a P0 bead, not silently re-fixtured.

---

## 9. Question status (all resolved)

Originally framed as "Unanswered Questions" — the investigation agents (`aa2a4df8f0de156ff` for Q1-Q12 broad survey; `aa2bc0f0f71aff123` for the Q4 deep dive) have answered all twelve. Full evidence in `UNROLLED_VALIDATION_ANSWERS.md` and `Q4_SPURIOUS_EDGE_CLASSIFIER_DESIGN.md`. Summaries below.

### Q1 — RESOLVED

`ML_MAT_COUNT = 2`, hardcoded module constant in `CMSValidator.py`. Guarded by an `assert kernel["PrefetchGlobalRead"] == 2` at the validator entry (the inline xj16 site at `KernelWriter.py:6292`).

**Rationale:**
- The cross-iter dependency span is determined by the kernel's prefetch depth. For `PrefetchGlobalRead=2`, the chain wraps within 2 iter copies.
- Hardcoding the value + asserting the precondition fails loud if a kernel arrives with a different prefetch depth. We have only verified the unrolled walk empirically for PGR=2; running on a different PGR without re-investigation is unverified-behavior.
- No formula, no parameterization, no `FourPartCapture` threading. When a higher-PGR kernel needs to be supported, the assertion fires, the validator is re-investigated for the new prefetch depth, and `ML_MAT_COUNT` + assertion are updated as a deliberate change rather than silently flexing.

The constant + assertion land in Commit 1 alongside the exemption deletion. The constant is consumed by `UnrolledCapture` in Commit 3a.

### Q2 — RESOLVED

`cumulative_issue_cycles` migration is mechanical: 3 body-discovery scaffolding sites at `CMSValidator.py:2493-2533`, `:2543-2569`, `:2600`. Core arithmetic (MFMA contention, type-switch +1) is body-agnostic and unchanged. No standalone `test_cumulative_issue_cycles_*.py` files exist; relevant tests are in `test_dataflow_graph_register_gaps.py::TestCumulativeIssueCycles` using synthetic captures that don't exercise cross-iter ML→ML — **zero re-pinning required**.

### Q3 — RESOLVED

Exactly 3 `.nodes.get(...)` sites in `CMSValidator.py` (`:3693` iteration; `:3773` / `:3774` Phase 0 lookups) + 4 `nodes_by_identity` write sites in `_make_node` and surroundings. Concrete migration table in §4.3 above.

### Q4 — RESOLVED

Option C hybrid (byte-key-driven multi-phase). Three terminal outcomes:
- `CaptureConsistencyError` raised (capture-pipeline bug)
- `EdgeRoutedDifferentlyFailure` typed Failure (renamed from `SpuriousEdgeFailure` for semantic precision; real CMS defect; hard-fail)
- `UnexplainedExtraEdgeError` raised (validator bug)

Full design + worked example in `Q4_SPURIOUS_EDGE_CLASSIFIER_DESIGN.md`. Summary in §4.1 above. SCC clobber paths and NOP-introduced waits all subsume into `EdgeRoutedDifferentlyFailure`'s byte-key model — no special branches.

### Q5 — REMOVED (concern dissolves under scrutiny)

Originally framed as "how do we detect silent behavior changes in class (d) tests?" This was a misreading of the critic's Q6, which was actually about explicit enumeration of class (d) tests for reviewer triage — not silent-regression detection.

The actual concern dissolves: pytest tests have assertions. If a test's assertions are sufficient to detect the rewrite's behavior change, the test fails — not silently. If a test's assertions are too weak to detect any behavior change, the test wasn't meaningful for that behavior anyway. Specific to the exemption deletion in commit 1: tests that don't trigger the exemption can't break when it's removed, by construction.

No baseline-snapshot strategy is needed. No class-(d) enumeration is needed. Trust pytest; if something breaks unexpectedly, investigate then.

### Q6 — RESOLVED (recommended choice accepted 2026-06-02)

When multiple iter copies of an ML instruction exist in the unrolled stream and a diagnostic message references "the producer," `cms_node_label` displays **the iter copy whose `unrolled_position` is closest-PRIOR-to the consumer**, selected upstream by the byte-key reverse-index lookup. The label string includes the iter index when ambiguous (e.g., `PackB0@ML_iter2.6`).

Rationale: this is the iter copy whose write actually populated `latest_writer[bk]` at the consumer's resolution moment, so it's the producer that semantically fed the consumer. Both alternatives (all iter copies joined, or the iter copy that emitted the byte-key footprint) are either noisier or equivalent in the common case.

### Q7 — RESOLVED

`node.identity` stays as a 3-tuple attribute. Only its dict-keying role migrates to the ordered sequence + byte-key reverse index. Diagnostic uses (SCC clobber lookup at `CMSValidator.py:3885-3887`, `FailureNodeLabel` rendering) keep consuming the identity tuple directly.

### Q8 — RESOLVED

Eager build, confirmed. The byte-key reverse index is a one-line addition inside the existing Phase 2 `latest_writer` walk — zero overhead beyond the `defaultdict.append` per write. Lazy build buys nothing.

### Q9 — RESOLVED

`LoopIters` is always statically known. Computed at Solution-build time in `Tensile/SolutionStructs/Solution.py:4258-4263` from compile-time integers. No dynamic case exists; the plan can assume static throughout.

### Q10 — RESOLVED

All 16 NGL "missing in SHADOW" edges on BPG#11 are body-local-walk artifacts (same mechanism as the 192 NLL extras, opposite direction). Under the unrolled walk they cancel in set-diff because byte-key edge_keys become byte-equal on both sides. `diagnose_extra_edge` is NOT invoked for any of them. The Q4 classifier exists to handle future CMS schedules that may surface real CMS-introduced clobbers, not the current corpus.

### Q11 — RESOLVED

75 total grep hits on `body_label` / `body_for` / `_BODY_BUILD_ORDER` / `for body in` in `CMSValidator.py`. Classified: ~30 are control-flow sites that need migration to unrolled-stream iteration (touched in commit 3); ~45 are diagnostic-annotation-only (no migration needed). Full enumeration table in `UNROLLED_VALIDATION_ANSWERS.md §Q11`.

### Q12 — RESOLVED

Zero existing tests need re-pinning. There are no standalone `test_cumulative_issue_cycles_*.py` files; relevant tests live in `test_dataflow_graph_register_gaps.py::TestCumulativeIssueCycles` and use synthetic captures that don't exercise cross-iter ML→ML. New tests required per §6.3.

---

## 10. Beads (filed, all P0, chained to block `r62g`)

| Step | Bead | Title | Depends on |
|---|---|---|---|
| C1   | `rocm-libraries-5tf9` | Delete exemption + ML_MAT_COUNT constant + PrefetchGlobalRead assertion | — |
| C3a  | `rocm-libraries-abgv` | `UnrolledCapture` materializer + tests (no validator wiring) | 5tf9 |
| C3b  | `rocm-libraries-wg77` | `DataflowGraph.nodes` refactor → ordered sequence + byte-key reverse index | abgv |
| C3c  | `rocm-libraries-1rsy` | `build_dataflow_graph` rewrite consuming `UnrolledCapture` | abgv + wg77 |
| C3d  | `rocm-libraries-xxj4` | `edge_keys()` byte-key basis migration | 1rsy |
| C3e  | `rocm-libraries-67us` | `compare_graphs` symmetric direction + `EdgeRoutedDifferentlyFailure` classifier | xxj4 |
| C3f  | `rocm-libraries-i190` | `diagnose_missing_edge` Phase 0/1 migration | 1rsy + wg77 |
| C3g  | `rocm-libraries-ktwt` | `cumulative_issue_cycles` + remaining `body_label` consumers migration | 1rsy |
| C3h  | `rocm-libraries-si5f` | n7og xfail removal + new cross-iter/cross-body unit tests | xxj4 + 67us + i190 |
| C4   | `rocm-libraries-5ryl` | Re-fixture (b)-class tests | si5f |

C4 (`5ryl`) → blocks `r62g`. `br dep cycles` returns 0.

**Closed during planning:** `rocm-libraries-tne8` — its original scope (thread iteration counts through `FourPartCapture`) was dissolved by the user-corrected hardcoded-constant-with-assertion design. The constant + assertion fold into C1.

Additional beads will be filed during implementation per the no-deferred-discoveries rule if any (c)-class real bugs or unrelated defects surface.
