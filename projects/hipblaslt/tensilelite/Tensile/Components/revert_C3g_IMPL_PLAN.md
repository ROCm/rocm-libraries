# Revert C3g (rocm-libraries-ktwt) Implementation Plan

**Commit to revert:** `07800a7d7622`
**Branch:** users/alvasile/mxfp4_fast_ref_min (worktree: validator_long_term_plans)
**Blocking bead:** `rocm-libraries-4q1k` (24 TimingTooCloseFailure on rebuilt rocisa)
**Verification status:** Verified 2026-06-09. Root-cause analysis confirmed correct; one framing
correction in §2C (the test error comes from `compare_graphs`/`diagnose_missing_edge`, not
`validate_edge_wait_coverage`); R2/R3 verified safe by code inspection; hand-revert decision
confirmed correct. Ready to implement.

---

## §1 Scope

**Revert:**
- `cumulative_issue_cycles` global-position migration (CMSValidator.py, roughly the 2561-2660 region)

**Decide (see §2 B):**
- `diagnose_missing_edge` wait-gate simplification (~line 4237) — keep (unrelated to regression)

**Keep:**
- `test_cross_iter_ml_cycle_count` in `test_dataflow_graph_register_gaps.py` — keep with modification (see §3)
- `C3g_ktwt_IMPL_PLAN.md` — keep (history document)

---

## §2 Investigation Findings

### A — What C3g changed (functional inventory)

Two independent functional changes in `CMSValidator.py`:

1. **`cumulative_issue_cycles` (primary change):** Replaced body-discovery scaffolding
   (`p_body_idx`, `c_body_idx`, identity search `ti is p_ti`, slot-key fallback `_slot_key`) with a
   `global_pos` counter matched against `producer.unrolled_position` /
   `consumer.unrolled_position`. The outer `_BODY_BUILD_ORDER` loop is preserved. The MFMA
   contention simulator logic is structurally identical.

2. **`diagnose_missing_edge` wait-gate (secondary change):** Removed `p_node.body_label !=
   c_node.body_label` from the cross-body suppression predicate. New form: `if waits and
   subj_graph.any_drains(waits, p_node)`.

3. **Test added:** `test_cross_iter_ml_cycle_count` in `test_dataflow_graph_register_gaps.py`
   exercises the new unrolled-position walk with a cross-body (ML_PREV → ML) edge.

4. **`C3g_ktwt_IMPL_PLAN.md`** added (history document, no functional role).

### B — Root cause of the 4q1k regression

**VERIFIED.** The root-cause analysis is correct.

Phase 1 of `build_dataflow_graph` stamps `unrolled_position` by walking `record.instructions`
from each `UnrolledIterRecord`. Those records are built by `UnrolledCapture.from_four_part_capture`
(in `ScheduleCapture.py`), which contains a local `_sorted_instructions(body)` function that sorts
by `(ti.slot.mfma_index, ti.slot.sequence)` before storing the result in the record. So
`record.instructions` is slot-lex sorted.

C3g's `cumulative_issue_cycles` (CMSValidator.py ~line 2605) walks
`captures.get(label).instructions` — the raw `LoopBodyCapture.instructions` list in its original
insertion order. On real kernels with SWait/SBarrier nodes interleaved in insertion order, raw
order diverges from slot-lex order. The `global_pos` counter then never matches
`producer.unrolled_position`, `p_issue_start` stays `None`, the function returns 0, and the
timing check fires `TimingTooCloseFailure` per affected edge — 24 total on the TF32 production
kernel.

The `_sorted_instructions` name referenced in the plan is not a module-level function: it is a
local closure inside `UnrolledCapture.from_four_part_capture`. Phase 1 reaches it via
`record.instructions`, which already holds the sorted result. This does not affect the analysis.

The task prompt's secondary framing ("SWaits advance global_pos while unrolled_position is
assigned only over dataflow-participating instructions") is also real but subordinate: even if
SWait nodes were absent, a raw vs slot-sort ordering difference could still cause a mismatch.

**Conclusion:** the 4q1k failure is entirely caused by the `cumulative_issue_cycles` migration.

### C — Does the wait-gate change cause any regression?

**VERIFIED with one framing correction.**

The plan's §2C claim that "the 4q1k failures are from `validate_edge_wait_coverage`, not
`diagnose_missing_edge`" is an oversimplification. Both paths call `cumulative_issue_cycles` and
both are broken by C3g. Specifically, `test_real_kernel_validates_clean_with_carveout_engaged`
calls `compare_graphs` (which invokes `diagnose_missing_edge`), not `validate_edge_wait_coverage`.
The 24 `TimingTooCloseFailure` entries are emitted from the `compare_graphs` / `diagnose_missing_edge`
path in that test, not from `validate_edge_wait_coverage`.

The wait-gate keep decision remains correct: the wait-gate simplification in `diagnose_missing_edge`
changes the cross-body suppression predicate (a classification gate for missing edges), not the
`cumulative_issue_cycles` walk. The two are independent. Under Resolution 1, ML iter copies already
have distinct `body_labels` (ML_PREV vs ML), so the removed `body_label !=` predicate was always
satisfied for cross-iter ML edges — the gate was redundant by proof.

**Decision: keep the wait-gate simplification.** Correct and unrelated to the regression.

### D — `test_cross_iter_ml_cycle_count` under reverted `cumulative_issue_cycles`

**VERIFIED.** The test fixture has:
- ML_PREV body: one LR at slot=0 (no SWaits)
- ML body: one MFMA at slot=(0,0) (no SWaits)

The OLD code (body-discovery + identity search) would:
1. Locate producer body (`BODY_LABEL_ML_PREV`) via `_BODY_BUILD_ORDER` scan.
2. Find producer TI via `ti is p_ti` identity search.
3. Locate consumer in `BODY_LABEL_ML`.
4. Run the simulator from producer to consumer.
5. Produce gap = 0 (same arithmetic as C3g for this simple fixture).

The test also asserts `producer.iter_index == 0` and `consumer.iter_index == 1` (stamped in Phase
1 of `build_dataflow_graph`, unchanged by the revert) and `producer.unrolled_position <
consumer.unrolled_position` (also Phase 1, unchanged). These assertions are independent of
`cumulative_issue_cycles`.

The test does call `cumulative_issue_cycles(g, producer, consumer)` directly (confirmed by
reading the test). Under the OLD code, the gap == 0 result comes from the correct arithmetic
path (not a fallback): the OLD code finds the producer via `ti is p_ti` identity (the tagged_inst
object is directly accessible on the GraphNode, and with one LR and no SWaits there is no
ordering divergence to cause a miss). The docstring comment about "the wrong reason (fallback 0)"
is the plan's acknowledgment that this distinction is undetectable at gap==0 — the iter_index
and unrolled_position assertions distinguish Phase 1 correctness from cumulative_issue_cycles
correctness.

**Decision: keep the test.** It remains a meaningful regression guard.

The test's docstring references "the new unrolled-position-based walk." After the revert, update
the docstring to note that the gap assertion passes under both the OLD and new implementations
(the test guards Phase 1 stamping correctness, not the specific walk strategy).

### E — Does the revert affect bead 37d3?

Bead `rocm-libraries-37d3` covers the `_alu_cross_subiter_passthrough` GapRule in
`_classify_edge_coverage` (within-graph timing path). C3g touched neither `_classify_edge_coverage`
nor any GapRule. Reverting C3g has zero effect on 37d3's scope or blocking relationship.
**37d3 is unaffected.**

### F — Failure-count prediction after revert

- Before revert (current, with fresh rocisa): 19 FAILED + 1 ERROR (the ERROR is
  `test_real_kernel_validates_clean_with_carveout_engaged` from 24 TimingTooCloseFailure due to
  4q1k).
- After revert: predicted 19 FAILED + 0 ERROR. The OLD body-discovery + identity-based
  `cumulative_issue_cycles` code is immune to the slot-sort/raw-order divergence because it
  matches via `ti is p_ti` identity (object identity), not position counters. The 24
  TimingTooCloseFailure entries resolve; the test returns to the C3f passing state.
- The 19 remaining FAILs are pre-existing beads (u6nn, nyb5) and C-chain residuals outside
  C3g scope — unchanged.

---

## §3 Per-file Revert Plan

### `Tensile/Components/CMSValidator.py`

**Revert:** The `cumulative_issue_cycles` function body (from the guard at ~line 2569 through the
function end at ~line 2659). Restore the old code:
- `if not (producer.position < consumer.position): return 0` guard
- `p_body_idx`, `c_body_idx` discovery loop over `_BODY_BUILD_ORDER`
- `p_ti = getattr(producer, "tagged_inst", None)` and `c_ti = ...`
- `_slot_key` helper and `p_key`, `c_key`
- `found_producer = False` flag
- Per-body walk `for body_i in range(p_body_idx, c_body_idx + 1)` with `start_idx`,
  `end_idx`, `consumer_idx_in_body` logic and the identity/slot-key search within each body

**Keep:** The `diagnose_missing_edge` wait-gate change at ~line 4237. No change to this line.

**Keep:** All `all_nodes_in_order` sites, Phase 1 stamping, and everything else.

### `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py`

**Keep `test_cross_iter_ml_cycle_count`** with docstring update.

Update the docstring to:
1. Remove the claim that the test specifically validates "the new unrolled-position-based walk."
2. State that the gap == 0 arithmetic holds under both the OLD body-discovery and any future
   unrolled-position walk implementations.
3. Retain the iter_index and unrolled_position ordering assertions as Phase 1 regression guards.

### `Tensile/Components/C3g_ktwt_IMPL_PLAN.md`

No change — historical record.

---

## §4 Step-by-step Implementation Order

**Do not use `git revert 07800a7d7622`** — C3g contains two independent changes in CMSValidator.py
(cumulative_issue_cycles and wait-gate) and we are keeping the wait-gate. A full `git revert`
would revert both plus the test; we then need to re-apply the wait-gate and the test. A
hand-revert of the cumulative_issue_cycles section is cleaner and easier to verify.

1. **Read** `CMSValidator.py` around `cumulative_issue_cycles` (current C3g code, ~lines 2566-2659).
2. **Read** the C3g diff (`git show 07800a7d7622`) to extract the exact OLD code for the body-discovery
   scaffolding — it is fully preserved in the diff's `-` lines.
3. **Edit** `CMSValidator.py`: restore the OLD `cumulative_issue_cycles` body (lines 2566-2659)
   from the diff's removed lines. The key sections to restore:
   - Guard: `if not (producer.position < consumer.position): return 0`
   - `p_body_idx`/`c_body_idx` loop
   - `p_ti`, `c_ti`, `_slot_key`, `p_key`, `c_key`
   - `found_producer = False`
   - Outer `for body_i in range(p_body_idx, c_body_idx + 1)` with `label = _BODY_BUILD_ORDER[body_i]`
   - Inner `start_idx` / identity-search / `end_idx` / `consumer_idx_in_body` logic
   - Inner `for i in range(start_idx, end_idx + 1)` walk with the simulator
4. **Verify** the wait-gate at ~line 4237 is unchanged (`if waits and subj_graph.any_drains(waits, p_node)`).
5. **Edit** `test_dataflow_graph_register_gaps.py`: update `test_cross_iter_ml_cycle_count`
   docstring to remove the unrolled-walk-specific claim and note the implementation-agnostic gap
   arithmetic. No assertion changes.
6. **Run** `pytest Tensile/Tests/unit/test_dataflow_graph_register_gaps.py -k test_cross_iter_ml_cycle_count`
   to confirm the test still passes.
7. **Run** the full unit suite (excluding the slow file per standing rule) to confirm 19 FAILED + 0 ERROR.

---

## §5 Validation — Expected Failure Delta

| Metric | Before revert | After revert |
|--------|---------------|--------------|
| ERROR count | 1 (`test_real_kernel_validates_clean_with_carveout_engaged`) | 0 |
| FAILED count | 19 | 19 (unchanged pre-existing) |
| TimingTooCloseFailure in 4q1k test | 24 | 0 |
| `test_cross_iter_ml_cycle_count` | PASS | PASS |

The 1 ERROR resolves because the OLD `cumulative_issue_cycles` uses `ti is p_ti` identity search
which is immune to ordering divergence between raw and slot-sorted instruction lists.

---

## §6 Bead Actions

| Bead | Action | Reason |
|------|--------|--------|
| `rocm-libraries-4q1k` | Close after revert + validation pass | Root cause (C3g global_pos/ordering mismatch) is eliminated by revert |
| `rocm-libraries-ktwt` | Reopen | C3g is being reverted; ktwt is not done |
| `rocm-libraries-37d3` | No action | Unaffected by this revert |

Note: `rocm-libraries-ktwt` should be reopened with an updated description noting the regression
root cause: `global_pos` walks `body.instructions` (raw order) while `unrolled_position` was
stamped from `_sorted_instructions` (slot-lex order). A correct re-implementation of the
unrolled-position walk must use the same slot-lex-sorted instruction order in both Phase 1 and
`cumulative_issue_cycles`, or use identity-based matching (`ti is p_ti`) inside a sorted walk.

---

## §7 Risks

**R1 — The reverted OLD code may have its own bugs on real kernels.** The pre-C3g code worked on
the May 26 build_tmp binary (per 4q1k description: "the cached binary did NOT produce these
failures"). This is strong evidence the OLD code is correct for the current real-kernel fixture.

**R2 — `producer.position` may be missing or None after C3-series changes.** The OLD guard is
`if not (producer.position < consumer.position)`. **VERIFIED SAFE:** `GraphNode.position:
SchedulePosition` is still a required (non-default) field on `GraphNode` (confirmed at line 920
of CMSValidator.py). It is populated by `_make_node` from the tagged_inst slot. The `__lt__`
comparison on `SchedulePosition` is untouched by C3a–C3h.

**R3 — Identity search `ti is p_ti` where `p_ti` comes from `getattr(producer, "tagged_inst", None)`
may return None if `tagged_inst` was removed.** **VERIFIED SAFE:** `GraphNode.tagged_inst:
"TaggedInstruction"` is still a required field (confirmed at line 923 of CMSValidator.py). The
slot-key fallback (`_slot_key`) is the secondary path when `tagged_inst` is None, but that path
is not reached in the normal case.

Both risks are structural — the test suite run (step 7 in §4) remains the final validation gate,
but pre-flight shows both fields are present.

**R4 — The test `test_cross_iter_ml_cycle_count` passes under the OLD code for the wrong reason
(fallback 0 rather than computed 0).** The test's design already accounts for this: the
iter_index and unrolled_position assertions would have caught a Phase 1 stamping regression;
the gap == 0 from fallback is arithmetically indistinguishable from correct in this fixture. This
is a known limitation documented in the test's comment block. Acceptable — the test guards Phase
1 invariants and acts as a future regression guard for any re-implementation.

---

## §8 New Beads to File

**rocm-libraries-ktwt (reopened):** Updated with root cause: slot-sort vs raw-order mismatch
between Phase 1 stamping and `cumulative_issue_cycles` walk. Future re-implementation must use
consistent instruction ordering across both sites.

No new beads are required. The 4q1k bead closes as resolved after validation.
