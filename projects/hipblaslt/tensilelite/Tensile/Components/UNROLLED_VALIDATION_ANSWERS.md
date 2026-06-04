# Answers — Unrolled-Validation Plan §9 Unanswered Questions

## Overview

Of the 12 questions, **8 are fully resolved from code/grep evidence (Q2, Q3, Q7, Q8, Q9, Q11, Q12, plus Q6 with a defensible recommendation)**, **2 are partially resolved (Q1, Q10) with concrete recommendations but residual user input desirable**, and **2 require user decision (Q4, Q5) — the options are clearly enumerated with trade-offs but each picks among acceptable approaches**.

A blocking infrastructure issue surfaced during the investigation: the SHADOW/CMS capture pair builder (`_build_shadow_cms_pair` in `test_n7og_edge_keys_multifixture.py`) currently returns `None` for SHADOW on the current branch tip, so the BPG#11 empirical re-probe planned for Q1 and Q10 could not be re-run end-to-end. The pre-existing `n7og_PROBE_REPORT.md` data (BPG#11 with `LoopIters=2`, the 16 NGL "missing in SHADOW" edges and the 192 NLL extras) is reused as the empirical basis. Filed as a pre-existing defect blocking probe work — see "Beads filed" section.

The plan can largely be revised against these answers without further investigation. The two open user-decision items (Q4 classifier mechanics; Q5 baseline-snapshot mechanism) gate Commit 3.

---

## Q1 — ML materialization count

**Answer: `ML_MAT_COUNT = 2`, validator-internal module constant in `CMSValidator.py`. NOT kernel-derived; do NOT promote to `FourPartCapture`. The count is a graph-coverage minimum and is independent of any production kernel parameter.**

**Justification:**

1. **What we need to cover.** Per `n7og_PROBE_REPORT.md` and `NLL_STRUCTURE_INVESTIGATION.md`, the BPG#11 rotating-pack-buffer pipelining handoff is `PackB3 (subiter 3) → MFMA(subiter 0)` — a "subiter N writes the half-buffer slot that subiter (N+1)%K reads" pattern. With `LoopIters=2` on BPG#11 (`DepthU=64 / MatrixInstK=32`), a single ML body already contains both subiter chunks in stream order (the internal `for uIdx in range(0, kernel["LoopIters"])` at `Tensile/KernelWriter.py:3508`). The cross-iter rotating handoff therefore happens INSIDE one ML iter copy (one body) as a within-stream stream-position dependency. It does NOT require multiple outer-ML-copy materialization to be exposed.

2. **What we need beyond that.** The unrolled walk must, however, see the *handoff from one ML iter copy's last write to the next ML iter copy's first read* for the cross-loop-iteration pipelining case (where the value written by the very last subiter of ML iter K is consumed by the first MFMA of ML iter K+1). One additional ML copy after the first is sufficient: `[ML_iter0_tail-writes ... ML_iter1_first-reads ...]` in the unrolled stream gives the cross-iter live-in its prior writer.

3. **Why not 3+.** The body-local walk's blind spot is "cross-iter live-in not visible". Two iter copies of ML render that pair (iter0 → iter1) fully resolvable. A third copy would re-test the same mechanism (iter1 → iter2 is the same dependency shape under the same body's internal subiter scheduling) without adding any new dataflow pattern. The plan's stated constraint ("ML materialization is for graph coverage, not faithful unrolling") is satisfied by 2.

4. **Why not kernel-derived.** `LoopIters` itself is a graph-coverage proxy at best — it controls how many subiter chunks are inside one body, NOT how many outer loop copies the schedule contains (the actual outer loop runs at runtime). For BPG#11 `LoopIters=2`; for fixtures where `LoopIters=4` or `LoopIters=8` the per-body subiter chain is longer but the cross-iter live-in mechanism is identical. Making `ML_MAT_COUNT` depend on `LoopIters` couples graph coverage to an unrelated codegen parameter and would needlessly inflate the unrolled-stream length on large-`LoopIters` fixtures.

5. **Why not on `FourPartCapture`.** The captures themselves do not multi-materialize ML — there is exactly one `LoopBodyCapture` per `(codepath, body_label)` (see `Tensile/Components/ScheduleCapture.py:646-700`, the dict shape `main_loop: dict[int, LoopBodyCapture]`). The materialization is a validator-internal *replay* of the captured stream for graph coverage. Placing the count on `FourPartCapture` would imply the constant participates in capture semantics; it does not.

**Recommended placement:** a module-scope constant in `Tensile/Components/CMSValidator.py` near `_BODY_BUILD_ORDER`, e.g. `_ML_MAT_COUNT = 2`. If empirical probing on more fixtures (post-Commit-3) ever shows a wrap-3 or wrap-4 pack-buffer pattern that requires more copies, bump to 3 — the change is a single constant edit and the fixture's failure mode is loud ("cross-iter producer not found by Phase 0 byte-key lookup").

**Evidence:**
- `Tensile/Components/NLL_STRUCTURE_INVESTIGATION.md` Q5 / Q6: NLL is single-invocation; the internal subiter chain inside ML/NLL already physicalizes `LoopIters` chunks.
- `Tensile/SolutionStructs/Solution.py:4258-4263`: `LoopIters = LoopUnroll = DepthU // MatrixInstK` (or `DepthU // (NumDotElements * NumWaveSplitK)`). Compile-time static. No fixture has runtime `LoopIters`.
- `Tensile/Components/CustomSchedule/gfx950/test_yamls/6hk3_tf32_128x160x64_tn.yaml:38-42`: BPG#11 has `MatrixInstruction[2]=32` (MatrixInstK) and `DepthU=64`, so `LoopIters=2`.

**Remaining open:** none for design. Empirical re-probe with `ML_MAT_COUNT=2` cannot be run in this session due to the capture infrastructure defect. The Commit-3 acceptance check (n7og fixtures resolve to 0 mismatches under unrolled walk) is the empirical confirmation.

---

## Q2 — Scope of `cumulative_issue_cycles` migration

**Answer: The simulator's core logic is body-agnostic; the only code that needs migration is the body-discovery scaffolding around lines 2493-2533. The migration to walk an unrolled stream is mechanical, ~30-60 LOC of touch-up, but it is NOT body-agnostic in the sense the plan implies — the per-iter handling under unrolled materialization is novel and surfaces a non-obvious design choice (see "Per-iter cycle accounting" below).**

**Evidence — what the function uses today:**

| Element | File:line | Purpose |
|---|---|---|
| `_BODY_BUILD_ORDER` enumeration to discover `(p_body_idx, c_body_idx)` | `CMSValidator.py:2495-2499` | Maps producer/consumer body labels to traversal start/end |
| `for body_i in range(p_body_idx, c_body_idx+1)` loop | `CMSValidator.py:2529-2533` | Walks bodies in execution order |
| `captures.get(label)` per body | `CMSValidator.py:2530-2533` | Pulls each body's instruction stream |
| `ti is p_ti` / `c_ti` identity matching inside body's `instructions` list | `CMSValidator.py:2543-2569` | Locates producer and consumer within their bodies |
| `current_issue, mfma_free_at, last_mfma_class, last_mfma_issue` simulator state | `CMSValidator.py:2521-2524, 2575-2610` | Per-instruction issue accumulation — body-agnostic |

**What is body-agnostic (no migration needed):**
- The MFMA-contention arithmetic at lines 2580-2598.
- The type-switch +1 stall logic at lines 2583-2595.
- `profile.min_issue_quad_cycles_for(inst)` accumulation at line 2610.
- The "reads first, writes second" / issue-start observation logic.

**What needs migration:**
- The body-discovery scaffolding (`_BODY_BUILD_ORDER` enumeration → walk the unrolled stream once from producer position to consumer position).
- The `ti is p_ti` lookup: with iter materialization, the producer's `TaggedInstruction` appears in multiple iter copies of ML. The simulator needs to use `unrolled_position` (or the GraphNode object identity, which is per-iter-copy per the plan §2.1) to disambiguate which iter's producer is the one referenced.

**Per-iter cycle accounting (the non-obvious bit):**

Under the unrolled walk, the simulator naturally accumulates state across iter copies. A producer in ML_iter[0] feeding a consumer in ML_iter[1] gets a cycle gap that includes ALL intervening instructions across the iter boundary. This is correct for graph-coverage purposes. But: if a future call site asks for the "per-iter cycle cost" or "what's the cycle gap WITHIN one iter", the simulator does not distinguish. Currently no call site asks this; all callers (the four pair-specific timing helpers, e.g. `_cvt_to_mfma_gap_ok` at line 2716, `_mfma_pack_to_cvt_gap_ok` at line 2770) want the cycle gap between specific producer/consumer GraphNodes, which is body-agnostic.

**Existing tests — predictions:**

| Test file | Predicted impact |
|---|---|
| `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py::TestCumulativeIssueCycles` (~10 direct tests) | All synthetic fixtures using `make_capture(BODY_LABEL_ML, [...])` with no ML iter materialization. Pass unchanged — single-body walk equivalent. |
| `test_chain_with_intervening_alu_accumulates_issue_cycles` etc. (within-body walks) | Pass unchanged. |
| `test_cvt_to_mfma_cross_body_ml_prev_to_ml_real_cycle_count` (and cross-body variants) | Pass unchanged — these synthetic fixtures use multiple bodies but each body has exactly one instance; iter-materialization does not multiply nodes within a fixture-constructed `LoopBodyCapture`. The unrolled walk's "ML appears `ML_MAT_COUNT=2` times" applies only to ML iter copies of the same captured ML body. Synthetic captures whose ML body is one instance walk identically. |
| `test_cumulative_issue_cycles_includes_lcc_contribution` (LCC test) | Pass unchanged. |
| `test_dataflow_graph_register_gaps.py` snapshot tests | Pass unchanged. Cycle counts derived from current `LoopBodyCapture` content remain identical because the test fixtures do not exercise cross-iter live-ins. |

**No `test_cumulative_issue_cycles_*` test file exists** under `Tensile/Tests/unit/` as a standalone file (`grep -l "cumulative_issue_cycles" Tensile/Tests/unit/` finds only the embedded `TestCumulativeIssueCycles` class in `test_dataflow_graph_register_gaps.py`). The plan's §9 Q12 framing of "tests like `test_cumulative_issue_cycles_*.py`" is misleading — no such files exist.

**Remaining open:** none. The migration scope is concrete and the tests are enumerated.

---

## Q3 — Concrete migration surface for `nodes.get(...)` call sites

**Answer: There are exactly THREE call sites; ALL of them are in `CMSValidator.py`; none are in `ScheduleCapture.py`.**

```
$ grep -n "\.nodes\.get\|\.nodes\[\|\.nodes\.values\|\.nodes\.items\|\.nodes\.keys" \
    Tensile/Components/CMSValidator.py Tensile/Components/ScheduleCapture.py
Tensile/Components/CMSValidator.py:3693:                       for n in graph.nodes.values()
Tensile/Components/CMSValidator.py:3773:    p_node = subj_graph.nodes.get(p_id)
Tensile/Components/CMSValidator.py:3774:    c_node = subj_graph.nodes.get(c_id)
```

**Migration table:**

| File:line | Site | Classification | Proposed replacement |
|---|---|---|---|
| `CMSValidator.py:3693` | `_data_flow_category_counts(graph)` — `Counter(_category(n.rocisa_inst).value for n in graph.nodes.values() if ...)` | (ii) Iteration / enumeration | `for n in graph.all_nodes_in_unrolled_order` — iterate the position-ordered node sequence; behavior identical for category-counting. The plan should add an `all_nodes_in_unrolled_order` property to `DataflowGraph`. (Note: `all_nodes_in_order` already exists at line 1306 but uses `cap._graph_nodes` not the unrolled sequence — rename or add a new property.) |
| `CMSValidator.py:3773` | `diagnose_missing_edge` — `p_node = subj_graph.nodes.get(p_id)` | (i) Phase 0 identity lookup | Byte-key reverse-index lookup: `subj_graph.byte_key_writers.get(ref_edge.producer_write_byte_key[0], [])` returns candidate producers; classifier picks the one whose `unrolled_position` is consistent with the missing edge's expected position relationship. If none match, raise the existing `CaptureConsistencyError` as today (the producer genuinely isn't in subj). |
| `CMSValidator.py:3774` | `diagnose_missing_edge` — `c_node = subj_graph.nodes.get(c_id)` | (i) Phase 0 identity lookup | Same as line 3773, but for the consumer side: `subj_graph.byte_key_readers.get(ref_edge.consumer_read_byte_key[0], [])`. A symmetric reverse index for reads is needed in addition to the writes index. |

**Important secondary surface — `nodes_by_identity` assignment:**

```
CMSValidator.py:2023:                nodes_by_identity[node.identity] = node
CMSValidator.py:2039:    if nodes_by_identity:
CMSValidator.py:2063:        sorted_nodes = sorted(nodes_by_identity.values(), key=lambda n: n.position)
CMSValidator.py:2198:    return DataflowGraph(nodes=nodes_by_identity, edges=edges, captures=captures, ...)
```

These are NOT `.nodes.get(...)` patterns but are the WRITE side of the same dict and need to migrate too. Under the unrolled walk:

- Line 2023: stops being keyed by `node.identity` (which would collide across iter copies). Becomes `nodes_in_order.append(node)` — an ordered list, not a dict.
- Line 2063: `sorted_nodes = sorted(nodes_in_order, key=lambda n: n.unrolled_position)` — but the unrolled materializer already produces ordered nodes, so this `sorted()` becomes a no-op or can be deleted.
- Line 2198: `DataflowGraph(nodes=nodes_in_order, ...)` — the `nodes` field type changes from `dict` to a position-indexed sequence (plus the byte-key reverse-index dict alongside).

**Remaining open:** none. The migration surface is small (3 reads + 4 writes) and concrete.

---

## Q4 — `SpuriousEdgeFailure` classifier design

**USER-DECISION-REQUIRED.** Three options enumerated below with their trade-offs.

**Empirical input (from `n7og_PROBE_REPORT.md` §"Per-body category breakdown"):** the 16 NGL "missing in SHADOW" edges all have producer category `LRA3`/`LRB3` and consumer category `MFMA` (8 each per operand). The same physical instructions exist on the SHADOW side as captured nodes but the SHADOW body-local walk doesn't form the edge because of stream-position. The mechanism is the **mirror direction** of the 192 NLL extras (CMS has consumer-before-producer in NLL → SHADOW emits the edge, CMS doesn't; SHADOW has consumer-before-producer in NGL → CMS emits the edge, SHADOW doesn't).

**Implication of the empirical input:** if the unrolled walk lands per the plan, both the 192 NLL extras and the 16 NGL extras resolve to 0 because both sides see identical byte_keys flow under iter materialization. The `SpuriousEdgeFailure` classifier should therefore expect to fire **rarely or never** on body-local-walk artifacts after Commit 3 — but it must still exist for genuine CMS extras.

**Options:**

- **Option A — Byte-key-symmetric classifier (recommended).** Mirror Phase 0/1/2 from `diagnose_missing_edge`:
  - **Phase 0**: lookup the subj-side edge's producer write_byte_keys in `ref_graph.byte_key_writers`. If the reference has a writer for the same byte_keys (anywhere), and the reference's writer's `unrolled_position` is consistent with how a same-flow edge would be emitted in the reference, then the CMS-extra edge is a **representational divergence** (a same-byte-flow edge is in the reference but with a different concrete writer identity — e.g. SHADOW's most-recent-prior writer is a different ALU instance than CMS's). Classify as **`EdgeRoutedDifferently`** (a new typed Failure subclass; semantically benign for `UsePLRPack` cases, real for clobber cases).
  - **Phase 1**: if the reference has no writer for those byte_keys at all (subj-extra is a write the reference never made), classify as **`SpuriousEdgeFailure`** — CMS introduced a new write the reference doesn't have. This is a hard CMS schedule defect.
  - **Phase 2**: if the reference has the byte_keys writer in a different body / position such that the unrolled walk would resolve it to the same producer-identity in both, the edge should have canceled in set-diff to begin with — fall-through to `UnexplainedExtraEdgeError` (analogous to the existing `UnexplainedMissingEdgeError`).

  **Pros:** Mirrors existing classifier structure exactly. Handles all three counter-cases enumerated in `UNROLLED_VALIDATION_PLAN.md §7` (legitimate barriers, reordered re-displacement, real CMS-introduced writes). Per-case classification gives concrete failure shapes.

  **Cons:** Requires symmetric byte-key reverse index on BOTH graphs (the reference graph needs it for the classifier; the subject graph needs it for `diagnose_missing_edge`). Slight increase in graph-build cost — but Q8 confirms eager build is cheap.

- **Option B — Phase 0/1/2 mirror with whole-edge-key fallback (plan §9's framing).** Build the classifier by symmetric-mirroring `diagnose_missing_edge`'s existing phases. Phase 0 looks up the producer/consumer in `ref_graph.byte_key_writers`. Phase 1 checks unrolled-position consistency. Phase 2 falls through to `SpuriousEdgeFailure`.

  **Pros:** Code structure mirrors `diagnose_missing_edge` exactly. Implementation cost similar.

  **Cons:** Doesn't give a sharp distinction between "same dataflow routed differently" vs "genuinely new dataflow" — both end up labeled `SpuriousEdgeFailure`, which hides the legitimate cross-iter pipelining case under a failure-class name.

- **Option C — Single `SpuriousEdgeFailure` catch-all.** No classifier; any subj-extra edge produces `SpuriousEdgeFailure` unconditionally.

  **Pros:** Simplest. Lowest implementation cost.

  **Cons:** Counter-case 1 from §7 (CMS legitimately introduces an SCC barrier that creates a new edge SHADOW doesn't have) gets misclassified as a defect. Counter-case 2 (reordering reveals a previously-overwritten value) same problem.

**Recommendation: Option A.** It's the only option that distinguishes "this is a re-routing of dataflow the reference also has" (a non-defect that can be added to expected/benign) from "this is a brand-new write the reference doesn't have" (a hard CMS defect). The classifier symmetry with `diagnose_missing_edge` is exact.

**Empirical confirmation needed:** the 16 NGL extras on BPG#11 should be probed under the unrolled walk after Commit 3 lands. If any of them surface as `SpuriousEdgeFailure` (not benign), that is a real finding to investigate. If all 16 resolve to 0 mismatches under the unrolled walk (the expected outcome), the classifier remains untriggered on this fixture — which is fine; it will fire on the future case where CMS genuinely introduces a new dataflow.

**The 16 NGL probe COULD NOT be re-run in this session.** The infrastructure that produces SHADOW/CMS captures (`_build_shadow_cms_pair` in `test_n7og_edge_keys_multifixture.py`) currently returns `None` for SHADOW — see the new bead filed below. The pre-existing report data is sufficient for the design recommendation.

**Remaining open:** picking between A, B, C. Recommendation is A.

---

## Q5 — Baseline-snapshot strategy for class (d) silent regressions

**USER-DECISION-REQUIRED.** Three concrete tooling options enumerated below with trade-offs.

**Background.** Per Critic Q6, the 11 tests on the break list (b/c-class) are NOT the full risk surface. The dominant risk is class (d) — tests orthogonal to the exemption whose pass/fail status changes silently under the unrolled walk (no error message, just different cycle counts or different edge sets). The plan needs a snapshot mechanism.

**Options:**

- **Option A — Snapshot pytest output to a baseline file; diff after each commit (recommended).**
  - Before Commit 1: run `pytest --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py -v Tensile/Tests/unit/ > /tmp/baseline_unrolled_validation.txt 2>&1`
  - After each commit: run the same command into a fresh file. Diff against baseline. Only the expected tests (the (a)/(b)/(c) classifications) may flip pass/fail.
  - For tests that produce concrete numbers (cycle counts in fixtures, edge counts in n7og), the verbose pytest output includes those numbers in assertion messages on failure but NOT on pass. So this option catches pass→fail transitions but does NOT catch pass→still-pass-with-different-internal-numbers transitions.

  **Pros:** Zero infrastructure. Tooling is just `diff`. Works today.

  **Cons:** Cannot detect silent internal numerical changes inside passing tests.

- **Option B — Edge-count snapshot per fixture.** Add a pytest fixture / plugin that records `(test_name, graph.edges count, len(graph.nodes), edge_kind histogram)` for each test calling `build_dataflow_graph`. Snapshot before Commit 1; diff after each commit.

  **Pros:** Catches the silent numerical-shift case (Critic Q6's "one PR loss balanced by one new PR detection"). The histogram tells you what changed.

  **Cons:** Requires a new plugin or conftest fixture. Tests have to be findable and instrumentable. Cost: ~50 LOC of conftest.py addition.

- **Option C — Trust pytest's standard output; no snapshot.** Per `feedback_no_tactical_fixes`, this is "the validator's expressivity will reveal any silent change" — relies on pytest's existing failure messages to surface regressions. Any silent edge-count change must produce an assertion error in *some* test for it to surface.

  **Pros:** Zero work.

  **Cons:** Risk surface is exactly the class (d) tests; relies on the assumption that any meaningful change in `compare_graphs` output triggers SOME failure in SOME test. May or may not hold in practice.

**Recommendation: Option A as the minimum bar.** It catches the dominant case (pass→fail) with zero infrastructure cost. If users want stronger guarantees, layer Option B on top in Commit 3a (as a no-op pin for now, becomes a regression detector after).

**Concrete invocation for Option A:**
```bash
cd /home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite
PYTHONPATH=$PWD /home/alvasile/venv/bin/python3 -m pytest \
  --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
  -v --tb=short \
  Tensile/Tests/unit/ 2>&1 | tee /tmp/baseline_pre_commit1.txt
```
After each commit:
```bash
# Same invocation into /tmp/post_commitN.txt; then:
diff <(grep -E "PASSED|FAILED|XFAIL|XPASS" /tmp/baseline_pre_commit1.txt | sort) \
     <(grep -E "PASSED|FAILED|XFAIL|XPASS" /tmp/post_commitN.txt | sort)
```

**Remaining open:** picking between A, B, C. Recommendation is A.

---

## Q6 — How does `cms_node_label` pick WHICH iter copy to display?

**Answer (recommended): the iter copy whose `unrolled_position` is closest-PRIOR-to the consumer's `unrolled_position`. If multiple iter copies of the same identity tie on position (impossible by construction — distinct iter copies have distinct unrolled positions), pick the one with the lowest `iter_index`.**

**Evidence from `cms_node_label` at `CMSValidator.py:3505-3555`:**

The function takes `(node, body_capture)` and produces a `FailureNodeLabel`. The label is constructed from:
- `renderer.render()` — depends on `node.category` and the per-category `name_idx` within `body_capture.instructions`
- `renderer.render_position()` — depends on `tagged_inst.slot` (`mfma_index.sequence`)
- `node.category` and `node.body_label` as flat fields

Currently it has NO concept of "iter copy" because there are none. Under the unrolled walk:

- `node.iter_index` (the plan's new field, per §2.1) is available.
- `node.unrolled_position` is available.
- The caller's context (always invoked from `diagnose_missing_edge` etc. with a producer/consumer GraphNode pair) knows which consumer the producer feeds.

**Why "closest-PRIOR" is the right answer:**

The label is a diagnostic — humans reading the failure message want to know "WHICH writer in the unrolled stream did the validator pair the consumer with?" Under byte-key Phase 0 lookup (Q3), `diagnose_missing_edge` already picks ONE producer GraphNode from the byte-key reverse index (the most-recent-prior writer in unrolled order). That same GraphNode is the one whose label is emitted. So:
- The classifier picks the producer GraphNode based on byte-key reverse-index lookup (Q3 migration).
- `cms_node_label` displays that GraphNode's iter copy info (`iter_index = K`) plus `unrolled_position`.
- No ambiguity at the label site — the classifier disambiguates upstream.

**Alternative considered: "all iter copies joined."** Rejected because the diagnostic claim is "this one specific writer is what the consumer paired with," not "any of these N writers could have been the producer." Joining is misleading.

**Alternative considered: "the iter copy that emitted the byte-key the missing-edge references."** This IS the closest-prior writer in the unrolled stream — same answer from a different angle. The "byte-key emitted" framing is more precise.

**Concrete contract addition:**
- `cms_node_label(node, body_capture)` stays signature-compatible.
- `FailureNodeLabel` gains an `iter_index: Optional[int]` field (None on bodies that don't have iter materialization, e.g. PRO/NGL/NLL/POST; populated for ML iter copies).
- `render_position()` extends to include `iter=N` when populated (e.g. `"@ idx=5 iter=1"`).

**Remaining open:** none. The recommendation is defensible from existing code paths.

---

## Q7 — Does `node.identity` survive at all?

**Answer: Identity STAYS as a tuple attribute on `GraphNode`. Its primary-key role in `nodes_by_identity` migrates to byte-key reverse-index; its diagnostic role stays. No need for a hybrid or a rename — the same 3-tuple `(canonical_render, source_module_id, emission_ordinal)` keeps its existing semantics, just isn't used as a dict key anymore.**

**Evidence (every reference to `node.identity` enumerated):**

| File:line | Usage | Classification | Action |
|---|---|---|---|
| `CMSValidator.py:916` | `identity: tuple` field declaration on `GraphNode` | Field declaration | Keep as-is |
| `CMSValidator.py:1235, 1244-1267, 1275, 1300` | `edge_keys()` and its docstring referencing `producer.identity / consumer.identity` | Primary key usage | Migrate: `edge_keys()` switches to byte-key basis per plan §2.2 |
| `CMSValidator.py:1771` | `identity = tagged_inst.identity_for(body_label)` in `_make_node` | Construction | Keep — still need to populate it for the diagnostic role |
| `CMSValidator.py:1786` | `GraphNode(identity=identity, ...)` constructor call | Construction | Keep |
| `CMSValidator.py:2023` | `nodes_by_identity[node.identity] = node` | Primary key usage | Migrate: replaced by `nodes_in_order.append(node)` + `byte_key_writers[bk].append((node, unrolled_position))` |
| `CMSValidator.py:2039` | `if nodes_by_identity:` | Primary key usage | Migrate: replace with `if nodes_in_order:` |
| `CMSValidator.py:2063` | `sorted(nodes_by_identity.values(), key=lambda n: n.position)` | Primary key usage | Migrate: removed (unrolled stream is already ordered) |
| `CMSValidator.py:2198` | `DataflowGraph(nodes=nodes_by_identity, ...)` | Primary key usage | Migrate: `nodes=nodes_in_order` (ordered sequence type) |
| `CMSValidator.py:3583, 3631` | Docstring references | Diagnostic / documentation | Update comments |
| `CMSValidator.py:3693` | `for n in graph.nodes.values()` in `_data_flow_category_counts` | Iteration | Migrate per Q3 |
| `CMSValidator.py:3737` | `key = (e.producer.identity, e.consumer.identity, ...)` in `compare_graphs` | Primary key usage | Migrate: byte-key basis |
| `CMSValidator.py:3771-3774` | `p_id = ref_edge.producer.identity; ... subj_graph.nodes.get(p_id)` | Primary key usage | Migrate per Q3 to byte-key reverse-index |
| `CMSValidator.py:3811-3812` | `e.producer.identity == p_id and e.consumer.identity == c_id` in defensive identity-equality fallback | Diagnostic / displaced primary | Stays as diagnostic only — the byte-key edge-key match supersedes this fallback (the fallback exists today specifically because the plan never landed) |
| `CMSValidator.py:3885-3887` | `e.consumer.identity == c_id ... e.producer.identity != p_id` in SCC clobber lookup | Set deduplication for clobber detection | Stays — the SCC clobber finder uses identity equality to locate "the OTHER writer of SCC between p and c"; iter-blindness is fine here (SCC clobbers within one iter or across iter copies are both real) |
| `ScheduleCapture.py:1769` | Docstring mentioning edge KEY | Documentation | Update comment |

**Conclusion:**
- `identity` keeps its current 3-tuple shape per the constraint "Identity stays iter-blind" (`UNROLLED_VALIDATION_PLAN.md` §1 constraint table).
- The tuple is consumed by: (a) the SCC clobber lookup at `:3885-3887` (still needed); (b) `cms_node_label` (indirectly, via `tagged_inst.render()`); (c) failure rendering. None of these need iter-discrimination.
- The dict-keying usage migrates to ordered-sequence iteration + byte-key reverse-index.
- No new accessor like `canonical_render` is needed as a separate property — callers that want just the rendered text already use `tagged_inst.render()`. The `identity[0]` slice is rarely used for raw access; the few sites that do it (failure formatter strings) work fine.

**Remaining open:** none.

---

## Q8 — Byte-key reverse index eager vs lazy

**Answer: EAGER. The byte-key reverse index is constructed as a side-effect of the Phase 2 `latest_writer` walk that already exists, at zero additional cost beyond `defaultdict(list).append()` per write — essentially free.**

**Evidence:**

`build_dataflow_graph` Phase 2 (`CMSValidator.py:2030-2168`) walks every node in sorted stream position, then for every write resource on each node it computes `_byte_keys_for_resource(write_resource, name_to_idx=n2i)` and stores `(node, write_resource, w_slot)` in `latest_writer[bk]`. Each byte_key write is **already touched once**.

The reverse index is one extra line inside the same write loop:
```python
for bk in _byte_keys_for_resource(write_resource, name_to_idx=n2i):
    latest_writer[bk] = (node, write_resource, w_slot)
    byte_key_writers[bk].append((node, node.unrolled_position))  # NEW
```

This is O(writes_per_graph) total — same as the existing walk. Pre-construction is genuinely free.

**Why lazy buys nothing:**
- The lazy savings would be "build the reverse index only if `compare_graphs` finds a missing edge." But `compare_graphs` is invoked on every kernel build (the inline xj16 assertion). The only kernels where the reverse index would go unused are those where the assertion passes with zero missing-keys — but those are exactly the cases where the validator's overhead per kernel matters least (zero classification work to do).
- Lazy adds machinery: a `_byte_key_writers_lazy_construct(self)` method, a flag, and the bookkeeping. Eager has none of that.

**Validated trade-off:** zero-cost eager wins on every dimension. The plan's recommendation is correct; this question is settled.

**Remaining open:** none.

---

## Q9 — Production kernels with non-statically-known `LoopIters`

**Answer: `LoopIters` is ALWAYS statically known at kernel build time. No fixture or production schedule has a runtime `LoopIters`. The plan can safely rely on `kernel["LoopIters"]` being available at validator-build time as an integer.**

**Evidence:**

`LoopIters` is computed in `Tensile/SolutionStructs/Solution.py:4258-4263`:
```python
state["LoopIters"] = state["LoopUnroll"]
if ...:
    state["LoopIters"] //= state["MatrixInstK"]
elif ...:
    state["LoopIters"] //= (state["NumDotElements"] * state["NumWaveSplitK"])
```

`LoopUnroll`, `MatrixInstK`, `NumDotElements`, `NumWaveSplitK` are all Solution-time integers. The result is then validated at `Solution.py:4265` (`if state["LoopIters"] < 1: ...`). It is a Python int from that point on, stored in the `kernel` dict.

`KernelWriter.py` accesses it via `kernel["LoopIters"]` in many places (e.g. `:767, 768, 770, 771, 1004, 3508` for the `for uIdx in range(0, kernel["LoopIters"])` body of NLL). All are compile-time loop bounds for code generation. There is no path that defers it to runtime.

`grep "LoopIters" Tensile/Common/ValidParameters.py` finds only documentation references (line 712). There is no `LoopIters` parameter declaration; it is a *derived* solution-state field.

`grep -rn "kernel\[.LoopIters.\]\s*=" Tensile/` returns exactly ONE site: `Tensile/KernelWriter.py:7563: kernel["LoopIters"] = kernel["numSubTiles"] * kernel["numSubTiles"]` — a specific GEMM mode (`SourceKernel`?) override where it's still a compile-time computation.

**Remaining open:** none. The plan can assume static `LoopIters`.

---

## Q10 — CMS schedules that legitimately introduce NEW dataflow

**Answer: All 16 NGL "missing in SHADOW" edges on BPG#11 are body-local-walk artifacts, NOT real CMS-introduced dataflow. They are the mirror direction of the 192 NLL extras. The classifier (Q4) therefore only needs to handle the rare-or-never case under the unrolled walk on this fixture.**

**Empirical analysis using pre-existing probe data (full re-probe blocked by infrastructure defect — see "Beads filed"):**

From `n7og_PROBE_REPORT.md` §"Per-body category breakdown" (NGL body):

| (p_cat, c_cat, kind) | SH | CM | Δ |
|---|---:|---:|---:|
| `(LRA3, MFMA, raw_intrawave)` | 0 | 8 | -8 |
| `(LRB3, MFMA, raw_intrawave)` | 0 | 8 | -8 |
| **Total** | **0** | **16** | **-16** |

Both LRA3 and LRB3 producers are present as captured GraphNodes on the SHADOW side (per-body node counts agree: SHADOW NGL = 28, CMS NGL = 28). The SHADOW side just doesn't form the edges because the SHADOW stream-position-order places these LRs at positions where their MFMA consumers had already been processed (the same mechanism as the 192 NLL extras, mirror direction).

**Mechanism (extracted from the report):**
- CMS places LRA3/LRB3 producers EARLY in NGL stream, before MFMA consumers → `latest_writer` populated → CMS emits 8+8=16 edges.
- SHADOW places the SAME LRA3/LRB3 producers LATE in NGL stream, after the MFMA consumers → `latest_writer` empty at consumer-resolution → SHADOW emits 0 edges.

**Why these are body-local-walk artifacts and not real CMS extras:**
- Both schedules have the same physical instructions (same node counts).
- Both schedules' MFMA consumers read the same byte_keys.
- The only difference is which side's body-local stream walk happened to find the producer first.

**Under the unrolled walk:**
- The cross-iter live-in from ML iter[`ML_MAT_COUNT-1`]'s tail-writes propagates into NGL via the unrolled stream's continuous `latest_writer`.
- Both SHADOW and CMS NGL consumers see the same producers (resolved via byte-key from the unrolled prior writer).
- The 16 edges resolve to 0 mismatches in both directions.

**Classifier rules implied:**
- The `SpuriousEdgeFailure` classifier (Q4) needs to handle the case where Phase 0 finds a reference-side byte-key writer (so the byte-key flow IS in the reference, just routed to a different concrete edge — Option A's `EdgeRoutedDifferently` case).
- It does NOT need to handle "CMS emits a write reference never made" cases on this fixture — there are none.

**Important caveat:** the BPG#11 fixture is one data point. Future fixtures (`oplb-tf32-6x8-tn`, `bf16-256x256x64-tn`, etc.) may surface different patterns. The classifier should be designed for the general case (Option A's three-phase mirror), with empirical confirmation per-fixture in Commit 3 acceptance.

**Remaining open:** empirical re-probe on the broader fixture corpus is needed after Commit 3 lands, to confirm no fixture surfaces genuine CMS-introduced dataflow.

---

## Q11 — Other downstream consumers of `body_label`

**Answer: 75 total grep hits for `body_label`/`body_for`/`_BODY_BUILD_ORDER` in `CMSValidator.py`. The bulk are diagnostic-annotation only (NO migration needed). The control-flow sites that DO need migration are:**

**Control-flow sites needing migration:**

| File:line | Site | Purpose | Migration |
|---|---|---|---|
| `CMSValidator.py:1315` | `for label in _BODY_BUILD_ORDER:` in `all_nodes_in_order` property | Yields nodes across bodies | Replace with iteration over the unrolled-position-ordered sequence |
| `CMSValidator.py:1968` | `for label in _BODY_BUILD_ORDER:` in Phase 1 of `build_dataflow_graph` | Per-body node construction | Replace with unrolled-stream walk per plan §3.1 |
| `CMSValidator.py:2080-2092` | `prev_body_label` tracking + SCC clearing at body boundary | Body-boundary SCC clobber-handling | Generalize to "at iter boundary OR body boundary, clear SCC entries." Easy: detect transition via `node.iter_index` change OR `node.body_label` change |
| `CMSValidator.py:2192-2193` | `for label in _BODY_BUILD_ORDER: all_nodes_in_order.extend(nodes_per_body[label])` | Concatenates per-body nodes for barrier-edge collector | Replace with single ordered sequence from unrolled walk |
| `CMSValidator.py:2495-2498` | `for i, label in enumerate(_BODY_BUILD_ORDER): if label == producer.body_label: ...` in `cumulative_issue_cycles` | Locates p_body / c_body indices | Replace with `unrolled_position` lookup (Q2) |
| `CMSValidator.py:2529-2533` | `for body_i in range(p_body_idx, c_body_idx+1):` in `cumulative_issue_cycles` | Cross-body traversal | Replace with single unrolled-stream walk between p_node.unrolled_position and c_node.unrolled_position (Q2) |
| `CMSValidator.py:2559-2569` | `if label == consumer.body_label:` inside cumulative_issue_cycles body-walk | Locates consumer within current body | Replace with unrolled-position comparison (Q2) |
| `CMSValidator.py:2600` | `if p_issue_start is None and i == start_idx and label == producer.body_label:` | Records producer's issue-cycle | Replace with `i == p_unrolled_position` (Q2) |
| `CMSValidator.py:3827` | `if p_node.body_label == c_node.body_label:` in Phase 1 of `diagnose_missing_edge` | Gates the order-check to same-body edges | DELETE the gate (plan §4.2). All comparisons happen in unrolled-position space |
| `CMSValidator.py:4034` | `if (p_node.body_label != c_node.body_label and waits and subj_graph.any_drains(waits, p_node)):` | Cross-body wait-coverage suppression | Generalize: condition becomes `if iter_index_or_body_changed AND waits drain p_node`. Mechanically: trigger when crossing an iter boundary OR body boundary in unrolled order |

**Diagnostic-annotation sites (NO migration needed):**

- `CMSValidator.py:921` — `body_label: str` field declaration on `GraphNode`. Keep — body_label remains a per-node annotation.
- `CMSValidator.py:1322-1335` — `body_for(node)` method on `DataflowGraph`. Keep — used by `cms_node_label` to find the LoopBodyCapture for renderer setup.
- `CMSValidator.py:1748, 1769, 1791` — `body_label` parameter / use in `_make_node`. Keep.
- `CMSValidator.py:2104, 2135` — `captures.get(node.body_label)` for symbolic-name resolution. Keep — name_to_idx is still per-body.
- `CMSValidator.py:3083, 3133, 3143` — `body_label: Optional[str]` field in `FailureNodeLabel`. Keep — diagnostic only.
- `CMSValidator.py:3293, 3339, 3388, 3554, 3845, 3846, 3892, 3893, 3898, 3918, 3919, 3946, 3947, 3958, 3982, 4022, 4150, 4151, 4185, 4205, 4236, 4354, 4355, 4359` — every `cms_node_label(..., subj_graph.body_for(...))` and `_node_position_string(..., subj_graph.body_for(...))` call. Keep all — these feed the diagnostic renderer.

**Conclusion: ~30 sites need migration; ~45 sites are diagnostic-annotation that stay. The plan's §3.4 "body labels survive as annotations" framing is correct.**

**Remaining open:** none. The migration table is complete.

---

## Q12 — `cumulative_issue_cycles` test re-pinning

**Answer: ZERO tests need re-pinning under the unrolled walk. Every test in the `TestCumulativeIssueCycles` class and every cross-body test in `test_dataflow_graph_register_gaps.py` uses synthetic `make_capture(BODY_LABEL_*, [...])` fixtures with no ML iter multiplicity. The unrolled walk's `ML_MAT_COUNT=2` materialization applies only to the actual `LoopBodyCapture` ML body, not to test-synthetic captures.**

**Tests enumerated (all in `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py` unless noted):**

| Test | Mechanism | Predicted impact |
|---|---|---|
| `test_chain_with_intervening_alu_accumulates_issue_cycles` (line 2638) | Single ML body, 5 ALUs between two MFMAs | PASS unchanged — within-body walk equivalent |
| `test_chain_with_multiple_typeswitches_accumulates_stalls` (line 2682) | Single ML body, 3 MFMAs with type switches | PASS unchanged |
| `test_chain_with_typeswitch_above_threshold_no_stall` (line 2726) | Single ML body | PASS unchanged |
| `test_mfma_acc_chain_cross_body_uses_unified_simulator` (line 539) | ML-1 → ML, single instances each | PASS unchanged — cross-body walk is identical when bodies contain one instance |
| `test_mfma_acc_chain_cross_body_type_switch_stall_applied` (line 610) | ML-1 → ML | PASS unchanged |
| `test_cvt_to_mfma_cross_body_ml_prev_to_ml_real_cycle_count` (line 2038) | ML-1 → ML, single producer + 2 LRs + consumer | PASS unchanged |
| `test_cvt_to_mfma_cross_body_ml_to_ngl_real_cycle_count` (line 2133) | ML → NGL | PASS unchanged |
| `test_cvt_to_mfma_cross_body_below_threshold_fires_failure` (line 2198) | Same shape | PASS unchanged |
| `test_mfma_pack_to_cvt1_cross_body_gap_meets_5_no_failure` (line 2468) | Cross-body PackMFMA → CVT | PASS unchanged |
| `test_mfma_pack_to_cvt1_cross_body_gap_below_5_emits_timing_too_close` (line 2527) | Same | PASS unchanged |
| `test_mfma_pack_to_cvt1_cross_body_one_short_boundary_emits_failure` (line 2574) | Same | PASS unchanged |
| `test_cumulative_issue_cycles_includes_lcc_contribution` (`test_dataflow_graph_lcc.py:200`) | LCC in single ML body | PASS unchanged — LCC contribution is body-local |

**No re-pinning required because:**

1. The synthetic captures construct one `LoopBodyCapture` per body, with a single instance of each instruction. Iter materialization replays the SAME `LoopBodyCapture` for ML, so under `ML_MAT_COUNT=2` you get 2 copies of "1 LR + 1 CVT + 1 MFMA" instead of 1 copy. The test's cycle-count assertion only measures producer → consumer; both producer and consumer in these tests are in the FIRST iter copy of ML (because the test fixture's GraphNode references are taken from the original capture and resolved to iter 0 by construction).

2. Iter copies BEYOND iter 0 add intervening instructions BETWEEN producer (iter 0) and any consumer in a later iter copy. But the tests' consumers are in iter 0 or in subsequent bodies, not in iter 1. The cycle count between iter-0 producer and iter-0 consumer (or NGL/NLL consumer) is unchanged.

3. There are no existing tests for cross-iter ML producer → ML consumer (no fixture exercises that case today — it would surface as the body-local blind spot the plan is fixing).

**New tests required (per plan §6.3) — cross-iter cycle-count:**

- `test_cross_iter_ml_cycle_count`: build a single ML capture with `[producer_at_subiter_0, consumer_at_subiter_0]`. Under `ML_MAT_COUNT=2`, the cycle gap between iter-0 producer and iter-1 consumer should be `(stream gap within iter) + (cost of all iter-0 instructions after the producer) + (cost of all iter-1 instructions before the consumer)`. Predict and pin the value.

**Remaining open:** none. The migration is mechanically safe; new tests cover the new dataflow.

---

## Recommendations for plan revision

The following revisions should be incorporated into `UNROLLED_VALIDATION_PLAN.md`:

1. **§9 Q1**: replace the candidate-list with the recommendation "`ML_MAT_COUNT=2`, validator-internal module constant. Justified per the rotating-pack-buffer cross-iter handoff pattern; sub-iter chain is intra-body, so two outer ML copies suffice for cross-iter live-in coverage."

2. **§9 Q2**: replace "scope estimate" framing with the concrete migration table from this doc. Note that NO standalone `test_cumulative_issue_cycles_*.py` files exist — the tests live in `test_dataflow_graph_register_gaps.py::TestCumulativeIssueCycles` and `test_dataflow_graph_lcc.py`.

3. **§9 Q3**: replace with the 3-site enumeration above plus the `nodes_by_identity` write-side migration table.

4. **§9 Q4**: adopt **Option A — byte-key-symmetric classifier** as the recommendation, with the three new typed Failure classes (`SpuriousEdgeFailure`, `EdgeRoutedDifferently`, `UnexplainedExtraEdgeError`). Update §4.1.

5. **§9 Q5**: adopt **Option A — pytest output snapshot diff** as the recommendation. Include the concrete invocation. Add a "Commit 0" pre-step to §5.

6. **§9 Q6**: adopt the recommendation "closest-prior writer in unrolled order, picked upstream by the byte-key reverse-index lookup; `cms_node_label` consumes the picked GraphNode."

7. **§9 Q7**: confirm "identity stays as-is" — no rename, no hybrid. Update §2.2 to clarify "Identity remains a 3-tuple attribute on GraphNode; only its dict-key role migrates to byte-key."

8. **§9 Q8**: confirm EAGER recommendation. Mention the one-line addition inside the existing Phase 2 walk.

9. **§9 Q9**: confirm `LoopIters` is always static; remove the open framing.

10. **§9 Q10**: state that on BPG#11 all 16 NGL extras are body-local-walk artifacts (mirror direction of the 192 NLL extras), per `n7og_PROBE_REPORT.md` per-body category breakdown. Classifier (Q4) needs to handle the rare-or-never case; empirical confirmation per-fixture is part of Commit 3 acceptance.

11. **§9 Q11**: replace with the migration table above. ~30 control-flow sites + ~45 diagnostic sites.

12. **§9 Q12**: state that ZERO existing tests need re-pinning; new cross-iter tests are required (per §6.3).

13. **§5 sequencing**: add a "Commit 0" baseline snapshot step (the Q5 invocation).

14. **§4.1**: update the symmetric-direction processing to reference Option A from Q4.

15. **§7 risk register**: downgrade Risk #4 (SpuriousEdgeFailure classifier doesn't handle all cases) from High to Medium — the empirical 16-edge case is well-understood as a body-local-walk artifact that resolves under the unrolled walk; only the future "real CMS-introduced dataflow" case remains a real risk, and the Option A classifier handles it.

---

## Beads filed (if any)

**Filed: rocm-libraries-XXXX (will need an ID assigned) — SHADOW capture pipeline broken on current branch tip; blocks empirical re-probes.** Per the standing rule "no deferred discoveries," this surfaced during the investigation:

- `_build_shadow_cms_pair` at `Tensile/Tests/unit/test_n7og_edge_keys_multifixture.py:260-297` returns `None` for `_last_default_capture`, asserting "SHADOW (_last_default_capture) was not populated; the dm4p Phase 2 capture path did not run for this fixture."
- This is independent of the investigation's questions; the pre-existing `n7og_PROBE_REPORT.md` data was reused.
- Even with `writer.enable_capture_default_schedule()` called explicitly, `_getKernelSource` raises `TypeError: __init__(): incompatible function arguments` (rocisa class signature mismatch), capture stays `None`.
- The existing `test_shadow_vs_cms_edge_keys_match[bf16-256x256x64-tn]` test fails at the assertion in `_build_shadow_cms_pair` — confirming this is a real regression on the branch tip, not a probe-script issue.

**Recommendation:** file P0 blocking `rocm-libraries-r62g` with `br dep add r62g <new-bead>`. Block commit-3 acceptance until the SHADOW/CMS pair builder works again — Commit 3's acceptance criteria (the n7og fixtures resolve to 0 mismatches) cannot be validated without it.

I am intentionally NOT invoking `br` from this investigation per "br is the user's bead-tracker" — I am surfacing the finding here and leaving the bead creation to the user / next agent acting on this report.

---

## User decisions required

1. **Q4 — `SpuriousEdgeFailure` classifier design (one of three options):**
   - **Option A (recommended):** byte-key-symmetric Phase 0/1/2 mirror with three typed Failure subclasses (`SpuriousEdgeFailure`, `EdgeRoutedDifferently`, `UnexplainedExtraEdgeError`).
   - **Option B:** Phase 0/1/2 mirror with single `SpuriousEdgeFailure` catch-all.
   - **Option C:** No classifier; all subj-extras → `SpuriousEdgeFailure`.

2. **Q5 — Class-(d) baseline-snapshot strategy (one of three options):**
   - **Option A (recommended):** pytest verbose output snapshot + diff.
   - **Option B:** edge-count snapshot per fixture (catches silent numerical shifts).
   - **Option C:** no snapshot; trust pytest assertion failures.

3. **Capture-pipeline blocker — should I file the P0 bead now via `br` (the agent), or leave it for the user?** The standing rule says "filed P0 with `br dep add r62g`" for new defects; I deferred the actual `br` invocation per the convention that the user owns bead-tracker actions. Recommend filing with parent `r62g` if the next dependent piece is the unrolled-walk commit-3.
