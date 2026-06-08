# C3e implementation plan — `compare_graphs` symmetric direction + `EdgeRoutedDifferentlyFailure` classifier

**Bead:** `rocm-libraries-67us`
**Depends on:** `rocm-libraries-56e3` (closed), `rocm-libraries-xxj4` (closed)
**Blocks:** `rocm-libraries-si5f` (C3h), `rocm-libraries-r62g`

---

## §1 Scope

This commit adds the `subj_keys - ref_keys` direction to `compare_graphs`, which today only processes `ref_keys - subj_keys`. It adds two new exception/failure classes — `EdgeRoutedDifferentlyFailure` (a typed `Failure` subclass, hard-fail) and `UnexplainedExtraEdgeError` (a raised exception, validator-bug class) — and implements `diagnose_extra_edge`, the byte-key-driven multi-phase classifier that routes each CMS-extra edge to one of three terminal outcomes. No existing `diagnose_missing_edge` logic is touched. The KernelWriter inline assertion at line 6302 is unchanged by this commit (it asserts `not graph_failures`, which remains correct: `EdgeRoutedDifferentlyFailure` instances go into `graph_failures` and cause the same hard-fail as existing failure types). The n7og probe test already does its own symmetric set-diff; this commit makes the production validator do the same, so they are consistent.

---

## §2 Investigation findings (A–I)

### A. Current `compare_graphs` shape

`compare_graphs` (`CMSValidator.py:3664`) currently:
1. Runs the per-category data-flow node count gate (`_data_flow_category_counts`), raising `CaptureConsistencyError` if counts differ.
2. Computes `ref_keys = reference.edge_keys()` and `subj_keys = subject.edge_keys()` (both return sets of 8-field tuples: `(source_module_id, emission_ordinal, producer_write_byte_key, consumer_read_byte_key, edge_kind, intra_operand_byte_offset, src_operand_slot, sink_operand_slot)`).
3. Computes only `missing_keys = ref_keys - subj_keys` (one direction).
4. Builds `ref_edges_by_key` dict (keyed on the same 8-tuple) from `reference.edges`, taking the first edge per key.
5. For each key in `missing_keys`, calls `diagnose_missing_edge(ref_edge, subject)` and extends `failures`.
6. Returns `failures`.

The `subj_keys - ref_keys` direction is computed nowhere in production code. The KernelWriter inline assertion at line 6302 (`assert not graph_failures`) only covers the failures list returned by `compare_graphs`; CMS-extra edges are silently invisible to it.

### B. Failure class taxonomy — current state

- `CaptureConsistencyError`: defined in `ScheduleCapture.py:91`, imported into `CMSValidator.py` at line 53. Already exists.
- `UnexplainedMissingEdgeError`: defined in `ScheduleCapture.py:113`, imported at line 56. Already exists.
- `EdgeRoutedDifferentlyFailure`: does **not** exist anywhere in `CMSValidator.py` or `ScheduleCapture.py`. Confirmed by grep returning 0 matches.
- `UnexplainedExtraEdgeError`: does **not** exist. Confirmed by grep returning 0 matches.

Both new classes are introduced by this commit.

### C. Option C hybrid classifier design

Per `Q4_SPURIOUS_EDGE_CLASSIFIER_DESIGN.md`, the classifier is a byte-key-driven multi-phase function with these phases:

- **Phase 0 (capture-consistency gating):** for each byte_key in `subj_edge.producer_write_byte_key`, query `ref_graph.byte_key_writers.get(bk)`. If any byte_key has NO writers in ref, raise `CaptureConsistencyError` — the §3 same-instruction-set contract is violated.
- **Phase 1 (closest-prior-writer comparison):** for each byte_key, find ref's most-recent writer whose `unrolled_position < cons_pos`. Compare identity against `subj_edge.producer.identity`. If any byte_key's closest-prior ref writer has a DIFFERENT identity from subj's producer, emit `EdgeRoutedDifferentlyFailure`. If all match, raise `UnexplainedExtraEdgeError` (the edge should have canceled in set-diff — validator bug). If ref has writers but none pre-consumer, treat as identity-mismatch (ref_writer=None) and emit `EdgeRoutedDifferentlyFailure`.
- **Phase 2 (fall-through):** unreachable under correct implementation; raises `UnexplainedExtraEdgeError`.

No instruction-class lists, no category-prefix matches, no subiter arithmetic. Decisions are purely byte-key-driven.

### D. `subj_keys - ref_keys` integration

The new direction integrates as a second processing block inside `compare_graphs`, parallel to the existing `missing_keys` block. Both blocks append to the same `failures` list. The function returns one combined list. The caller (KernelWriter `assert not graph_failures`) sees failures from both directions without any change to the assertion itself.

`subj_edges_by_key` is built from `subject.edges` using the same 8-field key construction as `ref_edges_by_key`. For each key in `extra_keys = subj_keys - ref_keys`, the classifier is called as `diagnose_extra_edge(subj_edge, reference)`.

### E. `diagnose_extra_edge` signature

```python
def diagnose_extra_edge(
    subj_edge: DataflowEdge,
    subj_graph: DataflowGraph,
    ref_graph: DataflowGraph,
) -> List["Failure"]:
```

Consumes: the extra subj edge (has `.producer`, `.consumer`, `.producer_write_byte_key`, `.consumer.unrolled_position`), the subject graph (needed for `subj_graph.body_for(node)` to construct `FailureNodeLabel`s via `cms_node_label`), and the reference graph (has `.byte_key_writers`). Produces: a list of `Failure` objects (only `EdgeRoutedDifferentlyFailure` can appear), or raises (`CaptureConsistencyError`, `UnexplainedExtraEdgeError`).

**Correction vs. original plan:** the original signature omitted `subj_graph`. `cms_node_label(node, body_capture)` requires a `LoopBodyCapture` obtained via `subj_graph.body_for(node)`. Without `subj_graph` in scope, `EdgeRoutedDifferentlyFailure`'s `subj_producer` and `subj_consumer` fields cannot be populated. The call site in `compare_graphs` (§4) must pass `subject` as the second positional argument: `diagnose_extra_edge(subj_edge, subject, reference)`. The Q4 design's pseudocode used `subj_graph_unused_here=None` as placeholder ellipsis — this was already a signal the signature was incomplete; the fix is to make `subj_graph` an explicit required parameter.

### F. KernelWriter.py inline assertion update

The assertion at KernelWriter line 6302 (`assert not graph_failures`) is **unchanged**. After C3e, `graph_failures` will include `EdgeRoutedDifferentlyFailure` instances if CMS introduced clobbering-style routing divergences. The assertion already treats any non-empty list as a hard-fail with formatted output — this is the correct behavior. No change needed to the assertion site.

### G. Expected test impact — n7og probe

The n7og probe (`test_n7og_edge_keys_multifixture.py`) manually computes `missing_in_shadow = cms_edges - default_edges` and `extra_in_shadow = default_edges - cms_edges` at lines 306–307, asserting `total_mismatches == 0` (with xfail markers for the TF32+UsePLRPack fixtures where this assertion currently fails). The probe's test function uses the raw `edge_keys()` sets from `build_dataflow_graph` directly, **not** `compare_graphs`. Adding the symmetric direction to `compare_graphs` does not affect this probe at all — the probe remains exactly as-is.

Regarding the 16 NGL extras on BPG#11: per `Q4_SPURIOUS_EDGE_CLASSIFIER_DESIGN.md §Worked example` and `UNROLLED_VALIDATION_PLAN.md §Q10`, all 16 are sub-case 1 (pipelining re-routing) — under the unrolled walk they cancel in set-diff because the byte-key edge_keys are byte-equal on both sides. After the C3d byte-key migration that is already landed (`rocm-libraries-xxj4`, closed), these 16 NGL extras must already be absent from `subj_keys - ref_keys` (they appear in the n7og probe's `missing_in_shadow = cms_edges - default_edges` which is the `subj − ref` direction, but only under the pre-C3d identity-keyed edge_keys). **Prediction:** with the post-C3d byte-key edge_keys now in production, `extra_keys = subj_keys - ref_keys` will be 0 on all current fixtures. `diagnose_extra_edge` will not be invoked for any production fixture in this commit.

The xfail markers in n7og remain. They are removed in C3h (`rocm-libraries-si5f`).

### H. Re-fixture work

Tests that assert `compare_graphs(...) == []` (an empty list): `test_dataflow_graph_comparison.py` at lines 103, 121, 151, 227, and `test_dataflow_graph_register_gaps.py` at line 3116. These construct synthetic graphs where ref and subj are either equal or differ in the `ref − subj` direction only. None of these tests construct a scenario where subj has extra edges that ref doesn't. Adding the symmetric direction does not change their behavior — `extra_keys` will be empty for all of them.

Tests that assert `compare_graphs(...) is truthy` (non-empty): `test_dataflow_graph_register_gaps.py` at lines 245, 389, 2960, 2999, 3196, 3238. These construct scenarios where ref has edges subj lacks. The symmetric direction adds nothing to these — `extra_keys` will still be empty.

**Conclusion: zero re-fixture work is needed for existing tests.** No test currently constructs a CMS-extra scenario; all existing compare_graphs tests use synthetic graphs with the ref-extra direction only.

New tests ARE required (see §6).

### I. Related beads

**`rocm-libraries-zvzu`** (P0, open): classify `LR → v_cvt_pk_bf16_f32` (CVT) raw_intrawave edges in `diagnose_missing_edge`. This is a `ref − subj` (missing-edge) classifier gap, entirely on the `diagnose_missing_edge` side. C3e touches `compare_graphs` and adds `diagnose_extra_edge` (`subj − ref` direction). These are independent; zvzu does not need to be resolved before or concurrent with C3e. No interaction.

**`rocm-libraries-i190`** (C3f, after 67us): `diagnose_missing_edge` Phase 0/1 migration to byte-key + unrolled positions. C3e does not preempt this — `diagnose_missing_edge` is entirely untouched by C3e.

---

## §3 Design — new Failure classes and `diagnose_extra_edge`

### `UnexplainedExtraEdgeError` (new exception class, mirrors `UnexplainedMissingEdgeError`)

```python
class UnexplainedExtraEdgeError(Exception):
    """diagnose_extra_edge couldn't classify an extra edge — classifier or pipeline bug."""
```

Defined alongside `UnexplainedMissingEdgeError` in `ScheduleCapture.py`. Exported from `ScheduleCapture.py` and imported in `CMSValidator.py` next to `UnexplainedMissingEdgeError`.

### `EdgeRoutedDifferentlyFailure` (new `@dataclass Failure` subclass)

```python
@dataclass
class EdgeRoutedDifferentlyFailure(Failure):
    subj_producer: FailureNodeLabel = None
    subj_consumer: FailureNodeLabel = None
    ref_producer: Optional[FailureNodeLabel] = None  # None if ref has no prior writer
    byte_keys: tuple = ()
    # Per-byte routing: bk -> (ref_writer_identity, subj_writer_identity). Verbose; complete.
    byte_key_routing: dict = field(default_factory=dict)

    def _format_canonical(self) -> str:
        ref_part = (
            f"reference routes through {self.ref_producer.primary} "
            f"{self.ref_producer.position}"
            if self.ref_producer is not None
            else "reference has no prior writer at this consumer position"
        )
        return (
            f"Subject's consumer {self.subj_consumer.primary} "
            f"{self.subj_consumer.position} reads from subject's producer "
            f"{self.subj_producer.primary} {self.subj_producer.position} "
            f"at byte_keys {self.byte_keys}, but {ref_part}"
            f"{self._iter_suffix()}. The subject schedule inserted or moved "
            f"an intervening writer between the reference's producer and the "
            f"consumer. (DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3)"
        )
```

Placed in the Failure class section after `OverriddenInputFailure` (its semantic mirror from the missing-edge side). The `_iter_suffix()` uses `iter_delta` inherited from `Failure` base class.

### `diagnose_extra_edge` — Option C hybrid, three terminal outcomes

Full Phase 0 + Phase 1 structure per `Q4_SPURIOUS_EDGE_CLASSIFIER_DESIGN.md §Recommended design`. Uses `ref_graph.byte_key_writers` (already built by C3b/C3c, confirmed present in `DataflowGraph` at `ScheduleCapture.py:1227`).

---

## §4 `compare_graphs` integration

The new direction integrates inside `compare_graphs` after the existing `missing_keys` loop. No new function-level parameters. No new return-value shape. The section header comment is updated to note both directions.

```python
# --- symmetric: extra keys (subj has, ref does not) ---
extra_keys = subj_keys - ref_keys
subj_edges_by_key = {}
for e in subject.edges:
    key = (e.producer.identity[1], e.producer.identity[2],
           e.producer_write_byte_key, e.consumer_read_byte_key,
           e.edge_kind, e.intra_operand_byte_offset,
           e.src_operand_slot, e.sink_operand_slot)
    subj_edges_by_key.setdefault(key, e)
for key in extra_keys:
    subj_edge = subj_edges_by_key[key]
    failures.extend(diagnose_extra_edge(subj_edge, subject, reference))
```

The `subj_edges_by_key` building pattern mirrors `ref_edges_by_key` exactly (same 8-field key, same `setdefault` dedup taking the first-seen edge per key, same rationale: edges sharing a key have uniform producer/consumer/kind attributes).

The block header comment explains the symmetric direction, cites `rocm-libraries-67us`, and notes `diagnose_extra_edge`'s three terminal outcomes.

---

## §5 KernelWriter.py:6302 inline assertion update

No change. The assertion:

```python
assert not graph_failures, (
    f"Dataflow graph comparison failed for kernel {kernel_label}: "
    ...
)
```

already formats every `f.format()` in the list, making `EdgeRoutedDifferentlyFailure` self-reporting. If this failure ever fires on a production build, the formatted output will include the byte_key routing information via `_format_canonical()`. The assertion itself is correct and complete.

---

## §6 Test impact + re-fixture work

### 6.1 No re-fixture for existing tests (confirmed)

All existing `compare_graphs(...) == []` and `compare_graphs(...) is truthy` tests use synthetic graphs with the `ref − subj` direction only. Adding `subj − ref` processing produces empty `extra_keys` for all of them. Zero re-fixture required.

### 6.2 New tests required

A new test file (or section in `test_dataflow_graph_comparison.py`) covering:

**Test 1 — `diagnose_extra_edge` Phase 0 path: `CaptureConsistencyError`**
Build a subj graph with a writer for byte_key `('v', 42)` and a matching consumer. Build a ref graph that has zero writers for `('v', 42)` anywhere. Call `diagnose_extra_edge(subj_edge, ref_graph)`. Assert it raises `CaptureConsistencyError` with "same-instruction-set contract" in the message.

**Test 2 — `diagnose_extra_edge` Phase 1 path: `EdgeRoutedDifferentlyFailure`**
Build a subj graph with producer P_subj (some identity) → consumer C at byte_key `('v', 31)`. Build a ref graph with a different producer P_ref (different identity) writing `('v', 31)` at an unrolled position before C's position (so ref's closest-prior writer differs from subj's). Call `diagnose_extra_edge(subj_edge, ref_graph)`. Assert one `EdgeRoutedDifferentlyFailure` is returned with the correct subj_producer/ref_producer labels.

**Test 3 — `diagnose_extra_edge` Phase 1 path: `UnexplainedExtraEdgeError` (identity match case)**
Build both graphs so ref's closest-prior writer for the byte_key has the SAME identity as subj's producer. Call `diagnose_extra_edge`. Assert it raises `UnexplainedExtraEdgeError` (the edge should have canceled).

**Test 4 — `compare_graphs` symmetric direction: extra_keys processing**
Build a minimal pair where subj has one extra edge (present in subj, absent from ref, ref has a writer with a different identity). Call `compare_graphs`. Assert one `EdgeRoutedDifferentlyFailure` in the result.

**Test 5 — `compare_graphs` symmetric direction: extra_keys empty on symmetric graph**
Call `compare_graphs(g, g)` (ref == subj). Assert `[]` returned. This already passes (existing test in `test_dataflow_graph_comparison.py:103`) but confirm it remains green.

**Test 6 — `EdgeRoutedDifferentlyFailure.format()` output**
Instantiate one with all fields set and call `.format()`. Assert the output contains the subj_producer primary, the ref_producer primary, "byte_keys", and "DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3".

---

## §7 Step-by-step implementation order

1. **Add `UnexplainedExtraEdgeError` to `ScheduleCapture.py`** next to `UnexplainedMissingEdgeError` (identical shape). Export it.

2. **Update `CMSValidator.py` imports** — add `UnexplainedExtraEdgeError` to the import tuple from `ScheduleCapture` at line 56.

3. **Add `EdgeRoutedDifferentlyFailure` to `CMSValidator.py`** in the Failure class section, after `OverriddenInputFailure` (§3 above). It uses `Optional` from the existing imports and `field` from `dataclasses`.

4. **Implement `diagnose_extra_edge`** in `CMSValidator.py` immediately below `compare_graphs` / `diagnose_missing_edge` (around line 4115, before the `validate_edge_wait_coverage` section). Full Phase 0 + Phase 1 per §3's pseudocode.

5. **Update `compare_graphs`** — add the `extra_keys` block after the `missing_keys` loop (§4 above). Add `diagnose_extra_edge` to the `compare_graphs` docstring's "routed through..." sentence.

6. **Update the section header comment** at CMSValidator.py:3652 — extend the comment to describe both directions (`ref − subj` via `diagnose_missing_edge`, `subj − ref` via `diagnose_extra_edge`).

7. **Write new tests** per §6.2. File location: new file `Tensile/Tests/unit/test_diagnose_extra_edge.py`, or append to `test_dataflow_graph_comparison.py`.

8. **Run the unit suite** (skipping `test_MatrixInstructionConversion.py`). Assert all existing tests remain green. Assert new tests pass.

---

## §8 Validation — expected failure delta vs end-of-56e3

After C3d (xxj4) landed, the 16 NGL extras (CMS-has, SHADOW-lacks, BPG#11) already cancel in set-diff under byte-key edge_keys — the `subj − ref` direction is empty for all current fixtures. This means:

- `extra_keys` will be empty on all production builds today.
- `diagnose_extra_edge` will not be invoked for any real kernel currently.
- The production `graph_failures` list is unchanged from the end-of-56e3 state (currently 0 failures on the BPG#11 TF32 fixture, per the passing validator assertion).
- **Failure delta = 0 new failures on current fixtures.** The classifier exists for future correctness defense, not to change the current pass/fail status.

What must stay green: all existing unit tests in the suite. Specifically:
- `test_dataflow_graph_comparison.py` (all `compare_graphs` tests)
- `test_dataflow_graph_register_gaps.py` (all compare_graphs + validate_edge_wait_coverage tests)
- `test_n7og_edge_keys_multifixture.py` (unchanged; xfail markers remain)
- `test_ValidateGRsCompleteBeforeLr1s.py`
- `test_validate_gr_not_too_early_graph.py`
- `graph_native_validation_base.py`-based tests

---

## §9 Risks / open questions

**Risk 1 — byte_key_writers population**: `DataflowGraph.byte_key_writers` is confirmed present (ScheduleCapture.py:1227) and populated by C3c (the write is at ScheduleCapture.py:2253). But confirm it is populated for WRITE-side nodes only (producers), not for read-side nodes, because `diagnose_extra_edge` Phase 0 queries it for "does ref have any writer for this byte_key" — the answer should only count writers, not readers. The code at line 2253 appends `(node, unrolled_position)` inside the write-operand processing arm of the Phase 2 walk. This is correct.

**Risk 2 — `subj_edges_by_key` key construction**: the key uses `e.producer.identity[1]` and `e.producer.identity[2]` (source_module_id and emission_ordinal). This mirrors the `ref_edges_by_key` construction at line 3812. Must confirm the identity tuple shape is `(canonical_render, source_module_id, emission_ordinal)` at this branch tip (it is, per the compare_graphs docstring at line 3677 citing `dfd8 / 56e3`).

**Risk 3 — `cons_pos` in `diagnose_extra_edge`**: the classifier uses `subj_edge.consumer.unrolled_position` to find ref's closest-prior writer. This is the CMS consumer's unrolled position. The ref graph's writers are at ref-side unrolled positions. Under the unrolled walk, both pipelines materialize the same physical instructions at (mostly) the same unrolled positions — but the critical contract is that both pipelines' unrolled streams have the same length (same total nodes). Per `UNROLLED_VALIDATION_PLAN.md §3.3`, the consumer in CMS and SHADOW have the same `identity`, so their unrolled positions should be close enough to find the "pre-consumer" ref writers correctly. The Phase 1 closest-prior lookup uses `< cons_pos` where `cons_pos` is from the subj side; a small position discrepancy between CMS and SHADOW for the same consumer is acceptable because we want the closest-prior ref writer relative to "approximately where the consumer is in the unrolled stream," not an exact position match. This is a mild approximation that the design accepts.

**Risk 4 — empty `ref_graph.byte_key_writers` before unrolled walk lands**: `byte_key_writers` is populated by C3c. The dependency chain confirms C3c is a prerequisite of 67us (via `rocm-libraries-xxj4` which depended on `rocm-libraries-1rsy` = C3c). At this branch tip, C3c IS landed (confirmed by ScheduleCapture.py:2253 existence). No risk.

**Open question 1** — `CaptureConsistencyError` message in Phase 0: the Q4 design doc asks whether to reference `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3` by docname. The recommendation is yes. This commit uses that convention, consistent with the existing `PrefetchGlobalRead` assertion at KernelWriter:6289–6295 which cites a doc by name.

**Open question 2** — `EdgeRoutedDifferentlyFailure` disposition: hard-fail (same as `OverriddenInputFailure`). No warn-only mode. Already resolved per Q4.

---

## §10 New beads to file

None. All discovered items (classifier design, unrolled walk prerequisites, n7og fixture behavior) are already captured in existing beads. No (c)-class real bugs were surfaced during this planning investigation. Bead `rocm-libraries-67us` remains in_progress until C3e commit is merged.
