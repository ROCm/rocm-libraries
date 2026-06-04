# Critique — Unrolled-Validation Plan

## Verdict
**PLAN-NEEDS-MAJOR-REVISION.** The framing of "validate as one singular unrolled timeline" is sound, the bead-filing posture is right, and the 4-commit sequencing avoids feature-flag bloat. But the plan rests on three structural assumptions that are either contradictory or factually wrong, and at least one of them (Q3 — identity-tuple collapse) is fatal to the materializer design as written. A second (Q5 — `kernel["NoLoadLoopIter"]` does not exist) makes Commit 1 unbuildable as specified. A third (Q1's "provably reachable-nothing by end of Commit 3") is not actually proven and may not even be true. Three of the ten critique questions also find real scope gaps (Q3, Q7, Q9) that put the ~500 LOC budget under serious doubt.

The principled direction is the right direction. The plan needs another spike before it is ready to execute.

---

## Per-question critique

### Q1 — Exemption "reachable-nothing" claim

**Status:** UNPROVEN. The plan asserts that once the unrolled walk lands, the exemption at `CMSValidator.py:3831-3843` receives zero hits on the TF32+UsePLRPack fixtures. Walking the logic:

1. The exemption fires when `default_p_before_c AND NOT subj_p_before_c` (subj side has producer AFTER consumer) AND producer is ALU AND subj subiter labels differ (`CMSValidator.py:3827-3843`).
2. Under the unrolled walk, the plan argues `subj_p_before_c` becomes TRUE because the producer is materialized in an earlier iter than the consumer, so the inversion gate fails.

The hole: the exemption's `default_p_before_c` and `subj_p_before_c` are computed on `ref_p.position` / `p_node.position` of the GraphNodes that landed in the comparison after `compare_graphs` ran set-difference on `edge_keys()`. The position field is `SchedulePosition(loop_index, stream_index)` (`CMSValidator.py:1772` via `make_position`). The plan never explicitly says how `position` will be extended to encode unrolled-stream index. If `position` stays at `(loop_index, stream_index)`, then iter-0 and iter-1 copies of the same TaggedInstruction have IDENTICAL `position` values — so `p_node.position < c_node.position` answers the same question it answered before. The plan implies (§2.4) "the unrolled coordinates collapse the question," but it never says `SchedulePosition` itself is being rebased on unrolled_index. If it is, that is a much larger change than declared (every consumer of `position.__lt__` — `cumulative_issue_cycles:2480`, the position comparisons in Phase 1, the LR sort in `_collect_barrier_edges`, etc. — has to be re-validated).

Additionally, the n7og probe addendum (`n7og_PROBE_REPORT.md:339-410`) measures `N3=192` on BPG#11 — a single fixture. The plan generalizes to "all TF32+UsePLRPack fixtures" without re-running on `oplb-tf32-6x8-tn`. The exemption probe should be re-run on every fixture in the corpus before Commit 3's assertion is wired in.

**Verdict:** the reachable-nothing claim is plausible-on-BPG#11 only and requires an actual probe to substantiate on the broader fixture set. The acceptance gate for Commit 3 (§4) must be tightened.

### Q2 — Commit 3 atomic scope

**Status:** TOO LARGE; SPLIT REQUIRED. Commit 3 packs into one commit: Phase 1 rewrite (~300 LOC), Phase 2 unrolled walk, `subj − ref` direction + `SpuriousEdgeFailure` (~50 LOC), Phase 1 gate rewrite in `diagnose_missing_edge`, xfail flip on `test_n7og_edge_keys_multifixture.py`, cross-iter unit tests (~120 LOC), and an instrumentation-counter assertion at the exemption call site. The 11 break-list tests must all stay green throughout, AND `compare_graphs` symmetric direction lands here (introducing a new failure surface).

Concrete failure modes baked into this packaging:

1. The `subj − ref` direction is being added at the SAME time as the unrolled walk. If the unrolled walk attributes the 192 NLL extras correctly but mis-attributes anything in the 16 NGL missing-in-SHADOW set, those become `SpuriousEdgeFailure`s — which surface as test failures in the same commit that's introducing both code paths. There's no way to bisect.
2. The xfail flip on `test_n7og_edge_keys_multifixture.py` is gated on `strict=True`. If the unrolled walk produces 1 residual edge on either fixture, the test fails as XFAIL-still-failing (which masks the XPASS signal the plan relies on for "remove the marker").
3. The break-list re-fixturing happens in Commit 4, but during Commit 3 those tests are pinned by the still-present exemption. Any silent change in the exemption's hit-pattern (e.g., the unrolled walk shifts an edge from "exempted" to "no longer reaches Phase 1 at all") is invisible until Commit 4 — at which point it surfaces as one of MANY simultaneous regressions.

**Recommended split (no flag, just smaller commits):**

- **Commit 3a:** add `subj_keys − ref_keys` direction + `SpuriousEdgeFailure` class + `diagnose_extra_edge`. Under the CURRENT body-local walk, this surfaces the 16 NGL missing-in-SHADOW edges as 16 new `SpuriousEdgeFailure`s on BPG#11. That's intentional — it makes the 16-edge surface visible BEFORE the unrolled walk is supposed to absorb it. The implementer can then verify (4 commits later) that the count drops to 0 — a clean before/after observation.
- **Commit 3b:** add `UnrolledCapture`-driven Phase 2 walk and Phase 1 rewrite. By construction this is "the unrolled walk lands." Tests in the break list still pass (still-present exemption + careful pinning). XFAIL flips correctly on n7og — that becomes the success signal of THIS commit specifically.
- **Commit 3c:** add exemption-hit-counter assertion (the instrumentation), verify count == 0 across the fixture corpus. If non-zero, file a P0 sub-bead and STOP — do not proceed to Commit 4.

The plan's monolithic Commit 3 conflates the "did the unrolled walk land correctly" question with "did the new failure direction land correctly", which makes Commit 3 unbisectable.

### Q3 — `UnrolledCapture` identity-tuple collapse

**Status:** CRITICAL DEFECT IN PLAN. The plan is internally contradictory on whether iter-duplicated nodes share Python identity:

- §1.1 says: "`original_node: GraphNode` is the SAME object across copies of an iter-duplicated ML body — there is exactly one `GraphNode` per captured `TaggedInstruction`. Only `unrolled_index` and `iter_index` differ across copies."
- §2.1 says: "Phase 1 walks the unrolled iter sequence; each captured `TaggedInstruction` produces ONE `GraphNode` per iter occurrence (multiple iters of the same physical rocisa instance get distinct `GraphNode` objects, each with a different `unrolled_index` and `(body_label, iter_index)` annotation pair)."

These cannot both be true. The correct answer is forced by:

1. `nodes_by_identity` is a dict keyed by `node.identity` (`CMSValidator.py:1962, 2023, 2198`). Identity tuple = `(canonical_render, source_module_id, emission_ordinal)` per `ScheduleCapture.py:537-551`. None of these three components depends on iter_index. So iter-0's PackA0 and iter-1's PackA0 have the SAME `identity`. If they are two different `GraphNode` objects (§2.1), they collide on `nodes_by_identity[node.identity] = node` — the second assignment overwrites the first, silently losing iter-0's node. If they are the same `GraphNode` (§1.1), then `position` is single-valued and the linear walk cannot distinguish iter-0's write of `('v',31)` from iter-1's write at a different unrolled_index.
2. `edge_keys()` (`CMSValidator.py:1229-1303, 1235`) tuple-keys on `(producer.identity, consumer.identity, ...)`. Identity is iter-blind, so 4 iter copies of "PackA0 → MFMA(iter K's consumer)" produce ONE entry in `edge_keys()`. The set-difference at `compare_graphs:3713` literally cannot distinguish iter-K's edge from iter-K+1's edge.
3. `subj_graph.nodes.get(p_id)` at `CMSValidator.py:3773-3774` retrieves a single GraphNode by identity. There is no iter discriminator on the lookup path.

**To fix this the plan must:**

- Extend `identity` to include iter_index (a fourth tuple element), OR
- Keep one GraphNode per TaggedInstruction but extend `position` to carry unrolled_index AND extend `edge_keys()` to encode the iter_index of producer/consumer endpoints, OR
- Reject the wrapper approach and accept that iter copies require physical GraphNode duplication with extended identity, AND change the dict key for `nodes_by_identity` to `(identity, iter_index)`.

Each of these has cascading consequences:

- Extending identity: `TaggedInstruction.identity_for(body_label)` no longer composes the identity tuple — it would need iter context. The "body_label is no longer consulted" simplification (§3.4) is reversed; iter_index is the new disambiguator. Every test that asserts on the 3-tuple identity (and there are at least the `test_dataflow_graph_emission_ordinal.py` set per `CMSValidator.py:1985`) needs re-fixturing. This is the OPPOSITE of the plan's "identity stays unchanged" framing.
- Extending position: `cumulative_issue_cycles` at `CMSValidator.py:2419+` uses position comparisons throughout, and the comparison semantics get baked into the per-arch issue simulator. The simulator's body-walk (`for body_i in range(p_body_idx, c_body_idx + 1):` at line 2529) cannot resolve "this consumer is in iter K+1 of ML, this producer is in iter K of ML" because it walks a SINGLE body's instructions list. The simulator needs a full redesign — not a "~40 LOC touch-up".
- Extending the dict key: `subj_graph.nodes.get(p_id)` in `diagnose_missing_edge` has to thread iter_index. But `diagnose_missing_edge` is invoked from `compare_graphs:3743`, which only has the edge key (the 6-tuple) — no iter_index dimension because edge_keys collapsed iters at the set level. The lookup has no information to thread.

**This is the biggest unaddressed cost in the plan.** Until the plan picks ONE of these three approaches and traces the implications through ALL downstream consumers, the LOC budget is meaningless. Plausible real cost: 600-1000 LOC across `identity_for`, `edge_keys`, `diagnose_missing_edge`, `cumulative_issue_cycles`, `body_for`, plus identity-asserting tests across the entire suite. The plan's "~500 LOC" estimate is likely under by 2-3×.

### Q4 — `SpuriousEdgeFailure` classification

**Status:** UNDER-SPECIFIED. The plan claims (§3.1) that "CMS emits a dataflow edge SHADOW does not have, meaning CMS introduced a write or read SHADOW did not. This is structurally a CMS schedule defect (the CMS scheduler should not introduce new dataflow facts that the default scheduler didn't produce from the same instructions)." This is not necessarily true under the unrolled walk. Counter-examples:

1. **Re-displacement**: CMS reorders such that a register written and immediately re-overwritten in SHADOW now has a window where its first write is the latest_writer for some reader. That's a NEW edge in CMS, not in SHADOW — and it is correct dataflow per CMS's schedule, not a defect.
2. **NLL→NLL within-NoLoadLoop pipelining**: the n7og report (`n7og_PROBE_REPORT.md` §"Per-body category breakdown") shows 24 CMS-only `PackA3 → PackA3` edges and 24 `PackB3 → PackB3` edges. These already exist today on the subj side. Under the unrolled walk they'd still exist, AND they're real dataflow (Pack3 in subiter N writes the rotating-buffer half that the same Pack3 in subiter N+1's chain reads via the half-buffer rotation). They're not CMS bugs.
3. **Cross-iter clobber by an intervening writer that SHADOW puts elsewhere**: if SHADOW's schedule places an unrelated `v_mov` that clobbers `('v', 31)` between an ML iter K writer and an ML iter K+1 consumer, SHADOW emits the v_mov→consumer edge (with no edge from the iter K writer). CMS's schedule that doesn't insert the v_mov emits the iter K writer→consumer edge. CMS has an edge SHADOW does not — not a defect, just a scheduling choice.

The plan needs an actual classifier design for `diagnose_extra_edge`, not the assertion "CMS-extra is structurally a defect." Per §7 open question 2, the plan itself acknowledges this is unsettled — but Commit 3 is supposed to land the class anyway. That's the wrong order: design first, land after.

### Q5 — NGL multi-iter handling AND `kernel["NoLoadLoopIter"]` does not exist

**Status:** CRITICAL CORRECTNESS DEFECT IN PLAN. The plan §1.2 says "`kernel["NoLoadLoopIter"]` — the tail-loop iteration count for `NLL`" and §4 Commit 1 says "Wire `kernel["LoopIters"]` and `kernel["NoLoadLoopIter"]` through both SHADOW and CMS capture sites."

Grepping `kernel\["NoLoadLoopIter"\]` across the entire Tensile codebase returns ZERO matches. The kernel parameters that exist are:

- `SuppressNoLoadLoop` (`ValidParameters.py:292`) — boolean.
- `OptNoLoadLoop` (`ValidParameters.py:346`) — integer {0, 1, 2}.

NLL itself is invoked at most once per kernel via `noLoadLoopBody` (`KernelWriter.py:3440`); the kernel runs through ONE `_emitNoLoadLoopBodyCMSMacro` invocation per kernelBody. The "iter count for NLL" the plan references DOES NOT EXIST as a kernel parameter — and arguably should not, because NLL runs once.

The 192 NLL extras' actual structure (per `n7og_PROBE_REPORT.md` `Stream-position trace` and the addendum table at line 367-371): the producer's `subj_subiter == 3` and consumer's `subj_subiter ∈ {0, 1, 2}`, both within `(NLL, NLL)`. The handoff is WITHIN a single NLL invocation across subiter chunks — NOT across multiple NLL iterations. The unrolled-stream model must handle within-body cross-subiter handoffs in the SAME NLL invocation, not duplicate NLL N times.

The plan's framing of "NLL iter (0..NoLoadLoopIter-1)" copies (§1.2) is built on a parameter that doesn't exist, and even if it did exist, would not address the actual data flow pattern in the n7og fixture.

**This is independently filable as a P0 sub-bead.** The plan must rewrite §1.2, §1.3, §4 Commit 1, §5.2 #1, §6 risk 4, and §7 open question 1 to remove all references to `kernel["NoLoadLoopIter"]`. NLL is iter-count-1; the within-NLL cross-subiter handoff is the actual mechanism that needs the unrolled walk.

Also: NGL is invoked once and has `iter_count=1`. The 16 NGL missing-in-SHADOW edges come from CMS-placed producers visible to consumers within NGL — same single-invocation, cross-position story. The plan doesn't need NGL multi-iter either.

The REAL story under the unrolled walk is:

- PRO: 1 copy.
- ML-1: 1 copy (when present).
- ML: `LoopIters` copies (cross-iter handoff is the rotating pack-buffer).
- NGL: 1 copy (cross-iter handoff happens via the LAST ML iter's tail-state into NGL's reads — not via NGL self-duplication).
- NLL: 1 copy (rotating-buffer handoff happens within NLL across mfma_index/subiter chunks — same single-stream walk, no iter dimension).

This is a substantially simpler model than what the plan describes. It also means the n7og's NLL 192 figure is RESOLVED by:
(a) walking the NLL stream once linearly (the body-local walk ALREADY does this);
(b) seeding `latest_writer` from the prior ML iter's tail state, OR
(c) acknowledging that the within-NLL Pack3 → MFMA handoff is across subiter chunks and the body-local walk is processing them in the wrong order (CMS placed consumer at stream_idx 0, producer at stream_idx 84-88).

If (c) is the real mechanism, the unrolled walk solution may actually have to do a DIFFERENT thing: it has to walk a body's instructions in their LOGICAL execution order (subiter ascending, then within-subiter sequence), not in the CMS-scheduled stream-index order. The plan does not engage with this at all.

### Q6 — Class (d) tests (orthogonal to exemption)

**Status:** MISSING FROM PLAN. The plan §3.3 enumerates 11 tests as (a)/(b)/(c). But the n7og exemption probe (`n7og_PROBE_REPORT.md:347-356`) shows the exemption fires 192/192 on BPG#11 and 0 times on every other branch — meaning every OTHER test in the test suite is class (d): orthogonal to the exemption. The plan doesn't enumerate the (d) class explicitly, leaving the reviewer to assume "everything else is fine." This is a bad mental model:

- The (d)-class tests are the ones the unrolled walk MUST not regress. They are by far the larger surface (~hundreds of tests vs 11 break-list).
- The plan §4 Commit 3 acceptance criterion ("all existing tests pass") is the proxy for class (d). But that bar is too loose: any silent edge-set change that happens to land at zero-net-failures (e.g., one PR loss perfectly balanced by one new PR detection) is invisible.
- A baseline edge-count audit BEFORE Commit 3 (per-test edge-count count) and AFTER would catch this. The plan does not call for one.

**Recommended addition**: pre-Commit-3 baseline `compare_graphs` edge-count snapshot across all unit tests; post-Commit-3 diff. Anything > 0 net change is a class-(d) regression that must be explained.

### Q7 — `_resolve_producers` under the unrolled walk

**Status:** PARTIALLY ADDRESSED, BUT FRAGILE. The plan §2.2 acknowledges that `_resolve_producers`'s grouping key `(id(writer_node), id(write_res), write_slot)` (`ScheduleCapture.py:1529`) needs `unrolled_iter_index` added. Good. But:

1. The plan §1.1 says "the SAME [GraphNode] object across copies", which means `id(writer_node)` is the SAME across copies. So the existing key would NOT distinguish writes from different iters — they'd collapse to one group, exactly what the plan §2.2 implicitly admits when it adds the fourth element.
2. If the writer_node is duplicated (one GraphNode per iter copy per Q3 option), then `id(writer_node)` already distinguishes iters — the fourth key element is redundant.
3. The `latest_writer` map's value tuple has to grow from `(writer_node, write_resource, write_slot)` to `(writer_node, write_resource, write_slot, writer_unrolled_iter)`. This is a small change, but it changes the contract that `_resolve_producers` consumes — and `_resolve_producers` is in `ScheduleCapture.py`, not `CMSValidator.py`. The dependency direction is correct (Validator → Capture), but the plan attributes the LOC to one of these without saying which.

The bigger risk per the n7og worked example: when consumer is in ML iter K+1 and `latest_writer[('v',31)]` returns ML-iter-K's writer (the most-recent prior in unrolled order), `_resolve_producers` correctly fans out per-byte. But: what about iter K-1's writer of `('v', 31)`? It was overwritten by iter K's writer, so it's no longer in `latest_writer`. Correct. What about iter K+1's writer of `('v', 31)` arriving AFTER the consumer's read? The plan says "reads emit edges FIRST, writes update latest_writer SECOND" within a single node (per `CMSValidator.py:2107-2167` Phase 2a/2b). Good — but under the unrolled walk's record-iteration, this "reads first" semantic is per-RECORD, not per-node-original. If two iter copies share a GraphNode reference, the SAME node's reads in iter K+1 might race the SAME node's writes in iter K. The plan does not engage with this race.

### Q8 — (c)-class real bug handling

**Status:** TOO VAGUE. The plan §3.3 row 5 lists `test_cross_subiter_pack_artifact.py::test_artifact_present_in_default_graph` as a (c)-class — "the BUG the exemption was masking." The plan §6 risk 3 says "if any test turns out to be (c) (real bug surfaced), file a P0 bead immediately — do NOT defer or silence it." But:

- The plan doesn't specify the bead's parent. Per the standing rule and the user's memory file (`feedback_no_deferred_discoveries.md`), it must be filed P0 blocking `r62g`.
- The plan doesn't specify whether Commit 4 should halt mid-implementation or proceed. The implementing agent would be in the middle of deleting `CMSValidator.py:3831-3843` AND re-fixturing 3 tests AND adding the regression-grep assertion. If they hit a (c) mid-Commit-4, do they stop? Revert? Continue?
- "Almost certainly an anti-pattern" (§9 last bullet) hand-waves the response. Per the standing rule "no tactical fixes", a (c)-class surfacing must lead to either a principled fix in branch or a documented P0 blocker — not a "we'll see" deferral.

The plan needs an explicit STOP-and-file-P0 rule for Commit 4, with the bead's parent fixed to `r62g`.

### Q9 — Downstream `body_label` consumers

**Status:** UNDER-COUNTED. Grepping `body_label`, `_BODY_BUILD_ORDER`, `body_for` in `CMSValidator.py` returns 95 hits. The plan §3.4 enumerates 6 categories. The actually-affected sites (after filtering out diagnostic-only `body_for`):

- `cumulative_issue_cycles` (`CMSValidator.py:2419-2617`) — walks bodies via `_BODY_BUILD_ORDER`, looks up producer/consumer position via `tagged_inst is ti` (`:2544, :2562`). Under the unrolled walk, the simulator must walk the unrolled stream. The plan says "~40 LOC of touch-up" — this is wrong. The simulator's lookup uses `ti is p_ti` to find the producer; under iter-duplication, the same `p_ti` appears in `instructions` list multiple times (or once with multiple unrolled positions, depending on Q3's resolution). The function would need a complete rewrite around the unrolled-record stream. Estimated cost: 80-150 LOC plus its 6+ unit test files in `Tests/unit/test_cumulative_issue_cycles_*`.
- Phase 1 order check (`CMSValidator.py:3827`): the `if p_node.body_label == c_node.body_label` gate. The plan says this becomes "is producer.unrolled_position < consumer.unrolled_position in subj graph?" But position-vs-unrolled-position is a SchedulePosition change (per Q1). Whoever lands this must rebase SchedulePosition itself OR maintain a parallel "unrolled_position" dimension on GraphNode.
- Wait-coverage suppression (`CMSValidator.py:4034`): the cross-body-handoff exemption. Plan says "generalize to cross-iter or cross-body." Doable, but the gate is currently "different body AND drains" — under the unrolled walk the iter discrimination has to be threaded.
- `body_for` (`CMSValidator.py:1322`, ~30 call sites): diagnostic mostly, but the call site at `_node_position_string` (lines 3293, 3339, 3388, etc.) feeds the OrderInvertedFailure's rendered output. Under iter-duplication, "this node lives in body X" can be ambiguous (X iter K vs X iter K+1).
- SCC clearing at body boundary (`CMSValidator.py:2083-2091`): the plan §2.1 says "clear at body OR iter boundary." This needs the unrolled walker to know what "iter boundary" means in terms of the unrolled record stream. Definition: any transition where `iter_index` changes OR `body_label` changes. Easy. ~5 LOC.
- `compare_graphs` per-category node counts (`CMSValidator.py:3700-3704`) — counts nodes by category for parity check. Under iter-duplication, ML iter K and ML iter K+1's PackA0 nodes are both PackA0 → the count doubles on the SHADOW side too, so parity may hold by construction. Plan doesn't address.

The plan's §3.4 framing of "no structural blockers" is too optimistic. The cumulative_issue_cycles rewrite alone is bigger than the plan budgets. Also worth noting: `body_for(node)` is called at least 15 times in `CMSValidator.py:3845-4185`; if a node has multiple iter-copy positions, the "which body does this node belong to" question has multiple answers. The plan must specify which one `body_for` returns.

### Q10 — End-of-Commit-3 assertion lifecycle

**Status:** UNSPECIFIED. The plan §4 Commit 3 says:

> An instrumentation assertion at `diagnose_missing_edge` Phase 1's exemption call site records exemption-hit-count = 0 across all CMS test fixtures.

But the plan never says:

1. **Is the assertion `assert` or a Python `warnings.warn`?** Under `python -O` the assert is removed. The plan §3 says `assert` is "the validator's no-silent-ignore contract" — so it's `assert`. OK.
2. **Is the assertion preserved after Commit 3 or removed before Commit 4?** Plan §4 Commit 4 says "Delete `CMSValidator.py:3831-3843` (the entire exemption block including the `nmps`/`_is_alu_producer` lookup)." That includes the instrumentation if it lives in the same block. So implicitly the instrumentation is removed in Commit 4. But that loses the regression guard.
3. **Where exactly does the assertion live?** Inside `diagnose_missing_edge` (the per-edge path)? Or at a global "after all classifications" call site? If per-edge, every test that flows through `diagnose_missing_edge` will fire the counter — but the assertion is "across all CMS test fixtures", which is a process-wide aggregate. The aggregator is undefined.
4. **What if the assertion DOES find non-zero?** §9's last bullet says "file as a P0 sub-bead under the unrolled-walk work and DO NOT proceed to Commit 4 until resolved." Good — but the gate is "no fixture should hit the exemption", and "all fixtures" is undefined (every parametrize entry? every test method? every kernel under matmul_gtest.yaml?).

**Recommended:**

- The instrumentation is a process-global counter living in `CMSValidator.py` module scope, incremented at `:3843` before `return []`.
- A conftest.py fixture asserts the counter is 0 at session end.
- The counter stays in tree across Commit 4 — it's "always 0 by construction" once the exemption is gone, so the test acts as a regression pin.

The plan must specify these mechanics.

---

## Hidden costs / scope creep risks

1. **Identity-tuple expansion forced by iter discrimination** (Q3) is the dominant hidden cost. The plan budgets ~500 LOC; realistic estimate is 800-1200 LOC factoring identity threading.
2. **`cumulative_issue_cycles` rewrite is bigger than ~40 LOC** (Q9) — closer to 80-150 LOC plus existing test updates.
3. **`kernel["NoLoadLoopIter"]` parameter does not exist** (Q5) — Commit 1 as written is unbuildable. The plan must rewrite Commit 1's threading goal to thread only `LoopIters` (real parameter) plus a derived/inferred NLL iter count (which is always 1).
4. **`SpuriousEdgeFailure` classifier is undesigned** (Q4) — the plan defers this to "later" while landing the class. Real risk: the new failure surface catches benign CMS-extra edges and breaks tests; without a classifier to triage them, every such test fails LOUD with no way to distinguish "real defect" from "scheduler reordering edge."
5. **Position field semantics under iter-duplication** (Q1, Q9) — unclear whether `SchedulePosition` rebases to unrolled coordinates or a parallel field is added. Either choice has cascading implications on `cumulative_issue_cycles`, the Phase 1 gate, the LR sort in `_collect_barrier_edges`, and any test that asserts on `position` values.
6. **`subj − ref` direction interacts with the unrolled walk in untracked ways** (Q2, Q4) — Commit 3 introduces both simultaneously; bisection is impossible if a regression surfaces.
7. **No baseline edge-count snapshot is required** (Q6) — class-(d) silent regressions are invisible.

## Risks the plan understates

1. **The 11-test break list may be incomplete.** Per Q6, the exemption probe was run only on BPG#11. Other fixtures in the corpus (per `test_n7og_edge_keys_multifixture.py:223-252`, only 3 fixtures are pinned, but many more exist across the test suite) have not been audited for exemption sensitivity.
2. **The `position < position` gate in Phase 1 (CMSValidator.py:3829) silently extends to unrolled positions if `position` is rebased.** If anything in the codebase relies on `position.loop_index` being a fixed enum value, that breaks. The plan does not audit this.
3. **`edge_keys()` collapsing iter copies at the set level may mask real cross-iter bugs.** Under the unrolled walk, "PackA0 iter 0 → MFMA iter 1" and "PackA0 iter 1 → MFMA iter 2" produce DIFFERENT physical dataflow but may collide into one `edge_keys` entry (because identity is iter-blind). If both sides emit both entries, the set has one element; if one side drops iter 1's edge, the set diff still has one element. The test misses it.
4. **The plan's framing of "POST does not exist today" (§1.2)** is right — but PR `udqg` (or similar) may land POST in parallel; if it does, the unrolled-walk concatenation order needs revision mid-implementation.
5. **xfail flip semantics under `strict=True`**: if Commit 3 produces 0 residuals on BPG#11 but 1 residual on `oplb-tf32-6x8-tn`, the XPASS for BPG#11 wins (the test name is parametrized) but oplb's xfail remains valid (still fails). The xfail markers are independent per parametrize entry — the plan should explicitly handle the partial-success case.
6. **The exemption probe at `/tmp/exemption_probe.py` is in `/tmp/`.** It is not version-controlled. By the time the implementing agent gets to Commit 3, the probe may have evaporated.

## New beads filed

None. The defects surfaced in this critique (Q3 identity-tuple collapse, Q5 `NoLoadLoopIter` non-existence, Q9 `cumulative_issue_cycles` rewrite scope) are all design defects in the unrolled-validation plan itself — they should be addressed by revising the plan rather than filed as new beads. The existing beads (`tne8`, `67us`) cover the original surface; these critique points are PLAN defects, not code defects.

If the user wants any of these tracked as beads anyway (e.g., for follow-up enforcement), the candidates are:

- "`kernel["NoLoadLoopIter"]` does not exist; plan §1.2 / §4 Commit 1 must be revised" — would block `tne8`.
- "Identity tuple must encode iter discrimination under unrolled walk; affects `edge_keys`, `nodes_by_identity`, `diagnose_missing_edge.subj_graph.nodes.get`" — would block the rewrite.

These are plan-level revisions; not filing per the standing "file P0 if you discover a genuinely new DEFECT" interpretation (these are plan defects, not code defects). User may instruct otherwise.

## Specific revisions recommended

1. **Resolve Q3 contradiction in §1.1 vs §2.1.** Pick ONE: GraphNodes are shared across iter copies (then position/identity must encode iter externally), OR GraphNodes are duplicated per iter copy (then identity must include iter_index AND every `nodes_by_identity` consumer must thread the iter dimension). The plan must explicitly state which.

2. **Delete all references to `kernel["NoLoadLoopIter"]` in §1.2, §1.3, §4 Commit 1, §5.2 #1, §6 risk 4, §7 question 1.** Replace with: NLL is iter-count-1 (single tail invocation). The 192 NLL extras are an INTRA-NLL cross-subiter handoff, not a cross-iter one. Restructure §2.3's worked example accordingly.

3. **Specify `SchedulePosition` semantics under the unrolled walk.** Add a §1.4 explaining whether `position` is rebased on unrolled_index, or whether a parallel `unrolled_position` is added to GraphNode. Document the impact on `cumulative_issue_cycles`, the Phase 1 gate, `_collect_barrier_edges`, and any test asserting on position values.

4. **Specify the `SpuriousEdgeFailure` classifier explicitly.** Add a §3.5 detailing the diagnose_extra_edge classifier. At minimum: cross-iter intra-body pipelining handoffs in CMS that SHADOW didn't model — those are not defects, they're absorbed. Real CMS bugs are: producer/consumer pair that has no equivalent dataflow in SHADOW under either iter coordinate.

5. **Split Commit 3 into 3a/3b/3c per Q2's recommended split.** Each is independently bisectable.

6. **Add a Commit 0 (or pre-Commit-1) baseline-snapshot step:** run the entire unit-test suite with edge-count diagnostic counters; persist as a baseline. Compare after Commit 3. Anything > 0 net change is investigated.

7. **Specify the exemption-hit-counter mechanics per Q10**: process-global counter, conftest-level session-end assertion, preserved across Commit 4 as a regression pin.

8. **Pin the exemption probe** (currently `/tmp/exemption_probe.py`) into `Tensile/Tests/unit/` or `Tensile/Components/` so it is version-controlled and re-runnable by future implementers.

9. **Audit the 11-test break list against ALL fixtures in the corpus, not just BPG#11.** Re-run the exemption probe with parametrized fixtures. Update §3.3 if the (a)/(b)/(c) classifications change.

10. **Add a STOP-and-file-P0 rule for Commit 4** (Q8): if a (c)-class bug surfaces during re-fixturing, halt Commit 4, file P0 blocking `r62g`, decide whether to proceed or revert. Make explicit so the implementer doesn't silently fix or silently defer.

11. **Audit downstream consumers of `body_label`** per Q9 with grep; update §3.4 with the actual count and per-site cost. Increase the LOC budget for §9 accordingly.

12. **Document the LOC budget revision.** The plan's "~500 LOC implementation" is unrealistic given Q3/Q5/Q9. Revise to a defensible range (800-1200 LOC) and re-validate Commit 3 as the bottleneck.

13. **Add a "post-implementation correctness assertion" to §8**: the SHADOW and CMS edge_keys sets must be IDENTICAL on every fixture in the corpus (not just bf16 and the two TF32+UsePLRPack), measured under the new `edge_keys()` definition. Anything else is a regression of class (d).

---

**End of critique.**
