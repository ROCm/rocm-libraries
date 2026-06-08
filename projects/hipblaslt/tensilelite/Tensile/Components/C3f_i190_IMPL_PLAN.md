# C3f implementation plan — `diagnose_missing_edge` Phase 0/1 migration to byte-key + unrolled positions

**Bead:** `rocm-libraries-i190`
**Depends on:** `rocm-libraries-67us` (C3e, closed)
**Blocks:** `rocm-libraries-si5f` (C3h)
**Commit alias:** 1rsy + wg77

---

## §1 Scope

This commit migrates `diagnose_missing_edge` in `CMSValidator.py` from its legacy identity-tuple lookup model to the byte-key reverse-index + unrolled-position model that the rest of the validator adopted in C3b–C3e. Phase 0 replaces `next(n for n in reversed(subj_graph.nodes) if n.identity == p_id)` with a byte-key reverse-index query into `subj_graph.byte_key_writers[bk]` for the missing edge's producer write byte-key, picking the closest-prior writer strictly before the consumer's subj-side unrolled position. Phase 1 removes the `if p_node.body_label == c_node.body_label` gate and replaces the `p_node.position < c_node.position` within-body order check with `p_node.unrolled_position < c_node.unrolled_position` across the full unrolled timeline. No other phases (SCC clobber, quad-cycle dispatch, Phase 2 wait coverage) are touched. The `CaptureConsistencyError` raise currently in Phase 0 is replaced with a semantically richer path described in §3. Existing tests that pin the current identity-based Phase 0 error text change — enumerated in §5.

---

## §2 Investigation findings (A–H)

### A. Current `diagnose_missing_edge` shape

**Phase 0 (lines 3932–3944):** Extracts `p_id = ref_edge.producer.identity` and `c_id = ref_edge.consumer.identity`, then does two linear scans over `reversed(subj_graph.nodes)` — one for the producer, one for the consumer — matching by `.identity`. If either node is absent, raises `CaptureConsistencyError` with the message `"identity-coverage check at compare_graphs entry was bypassed. p_id=… c_id=…"`. This is the current gating raise that three tests explicitly pin (see §5).

**Legitimate-reorder defensive branch (lines 3967–3978):** Before Phase 1, walks `subj_graph.edges` to see if an edge with the same `(p_id, c_id, edge_kind, intra, src_slot, sink_slot)` already exists in the subject. If found, returns `[]` early. Under the byte-key edge_keys model this branch is redundant (identical byte-key tuples cancel in set-diff before reaching this function), but it is retained as a defensive identity-equality fallback. Post-C3f this branch should be evaluated for removal in C3g; for now it stays.

**Phase 1 (lines 3986–4005):** Reads `ref_p = ref_edge.producer` and `ref_c = ref_edge.consumer` (ref-side nodes). Then gates on `if p_node.body_label == c_node.body_label`. Inside that gate: computes `default_p_before_c = ref_p.position < ref_c.position` (ref-side SchedulePosition comparison) and `subj_p_before_c = p_node.position < c_node.position` (subj-side SchedulePosition comparison, within the same body). If default has p before c but subj has c before p: emits `OrderInvertedFailure`. If default has p before c and subj also has p before c: falls through to wait coverage. If default has p after c (kind_rank-induced): falls through with no failure. The gate skips the check entirely when `p_node.body_label != c_node.body_label`.

**Cross-subiter ALU-producer exemption:** Confirmed DELETED in C1 (no grep matches in the current file). The legacy lines that checked `_is_alu_producer` and compared subiter values are gone.

**Subsequent phases after Phase 1:**
- SCC clobber branch (lines 4022–4051): fires if `ref_resource.regType == "scc"`, scans for an intervening SCC writer in subj. Emits `OverriddenInputFailure` or falls through.
- Quad-cycle dispatch (lines 4053–4072): calls `_dispatch_quad_cycle_check(p_node, c_node, subj_graph)`. Returns `_PASSTHROUGH` (ALU/CVT exemption → return []), `TimingTooCloseFailure`, or `None` (fall through to Phase 2).
- Phase 2 (lines 4075–4195): wait/barrier coverage using `waits_in_window` and `barriers_in_window`. Emits `MissingWaitFailure`, `WaitInsufficientFailure`, `MissingBarrierFailure`. Raises `UnexplainedMissingEdgeError` on unclassified fall-through.

**Failure types produced:** `OrderInvertedFailure`, `OverriddenInputFailure`, `TimingTooCloseFailure`, `MissingWaitFailure`, `WaitInsufficientFailure`, `MissingBarrierFailure`, `UnexplainedMissingEdgeError` (raised).

### B. Byte-key Phase 0 migration design

The missing ref-edge carries `producer_write_byte_key` (a tuple of byte-key pairs, written by the producer) and `consumer_read_byte_key`. The relevant question for Phase 0 is: does subj have a node that "plays the role" of the producer for this specific byte-key + consumer position?

**Producer lookup:** `subj_graph.byte_key_writers` is `Dict[byte_key, List[(GraphNode, unrolled_position)]]`, appended in ascending position order. For `bk in ref_edge.producer_write_byte_key`, collect entries from `subj_graph.byte_key_writers.get(bk, [])`. The consumer's subj-side unrolled_position bounds the search: we want the most-recent writer strictly before the consumer's position — this is the writer subj's latest-writer resolver would have paired with the consumer for that byte-key, i.e., the node that "should have been" the ref producer's equivalent in subj.

**Consumer lookup:** The consumer's identity is known from `ref_edge.consumer.identity`; resolve to a subj node by scanning `subj_graph.nodes` for `.identity == c_id` (unchanged from current — the consumer lookup stays identity-based because we need the consumer's `unrolled_position` for the Phase 1 order check). Only the producer lookup migrates to byte-key.

**Algorithm for Phase 0:**
1. Resolve consumer via identity scan: `c_node = next((n for n in subj_graph.nodes if n.identity == c_id), None)`. If absent: raise `CaptureConsistencyError` (same contract as before — a consumer absent from subj is a capture-pipeline bug, not a CMS schedule defect; the consumer is a DATA-FLOW node whose identity must match).
2. For each `bk in ref_edge.producer_write_byte_key`: look up `subj_graph.byte_key_writers.get(bk, [])`. Pick the entry with the largest `unrolled_position` strictly less than `c_node.unrolled_position`. Call the result `p_node_candidate`. If no entries exist for ANY byte-key in the producer's footprint: that is a `CaptureConsistencyError` — ref's producer writes a byte-key that subj has no writer for, violating the same-instruction-set contract (parallel to `diagnose_extra_edge` Phase 0). If entries exist but all are AFTER the consumer (no prior writer): `p_node` is `None` for that byte-key — this is the "genuinely absent producer" case, falls through to Phase 1 unrolled-order check with the first available writer as a proxy, or directly to Phase 2.
3. If all byte-keys in the footprint have a closest-prior writer with the SAME identity: the candidates are consistent; pick any as `p_node`.
4. If byte-keys disagree on identity: pick the one that best aligns with `ref_edge.producer.identity`, or pick the first by byte-key order. Diagnostic only.

**Note on `p_node` = None handling:** When the producer write byte-key has no prior writer before the consumer in subj, the edge is "genuinely missing" — subj never produced the byte at or before the consumer's position. In this case the Phase 1 order check is vacuous (there's no subj producer to compare positions with). Skip Phase 1 and fall through directly to Phase 2 wait coverage with `p_node` = the ref producer's position as a proxy (or the first available writer). The existing Phase 2 logic does not depend on `p_node` being in subj — it uses `counter_for(p_node)` which can be derived from `ref_edge.producer` instead.

### C. Phase 1 body_label gate removal

**Current gate:** `if p_node.body_label == c_node.body_label` — skips the order check entirely for cross-body pairs. This was necessary because the per-body walk never formed cross-body edges in the old model; the body_label on p_node and c_node from different bodies would have incomparable `SchedulePosition` values (different `loop_index`).

**After C3f:** The unrolled walk assigns every node a `unrolled_position` (int, strictly monotone across all bodies). The order check becomes:
```python
default_p_before_c = ref_p.unrolled_position < ref_c.unrolled_position
subj_p_before_c = p_node.unrolled_position < c_node.unrolled_position
```
No body gate. Cross-body inversions surface naturally: if ref has ML-1 (iter=0) producer before ML (iter=1) consumer, but subj has them inverted in the unrolled stream, `OrderInvertedFailure` fires. This is correct behavior — the standing "validate as single timeline" rule requires it.

The `default_p_before_c = ref_p.position < ref_c.position` (SchedulePosition) comparison also changes: use `ref_p.unrolled_position` and `ref_c.unrolled_position`. The ref-side unrolled positions are on `ref_edge.producer_unrolled_position` and `ref_edge.consumer_unrolled_position` (populated by C3c on every DataflowEdge). Or equivalently, resolve ref-side nodes from `ref_edge.producer` directly — they are GraphNode objects with `.unrolled_position` set.

**`OrderInvertedFailure` field update:** The current constructor is:
```python
OrderInvertedFailure(
    producer=cms_node_label(p_node, subj_graph.body_for(p_node)),
    consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
    iter_delta=p_node.iter_delta_to(c_node),
    default_producer_position=ref_p.position,
    default_consumer_position=ref_c.position,
)
```
`default_producer_position` and `default_consumer_position` are typed `Optional[SchedulePosition]`. After the migration these should use `ref_p.unrolled_position` and `ref_c.unrolled_position` (ints). The `SchedulePosition` type annotation on `OrderInvertedFailure` must be updated to `Optional[int]`. This is a contained change — the failure data model is rich enough since body_label is already on both `producer` and `consumer` via `cms_node_label` (which includes body_label in its `FailureNodeLabel`).

### D. Symmetry with `diagnose_extra_edge` (C3e)

`diagnose_extra_edge` Phase 0: checks that every byte-key in `subj_edge.producer_write_byte_key` has at least one writer in `ref_graph.byte_key_writers`. If not, raises `CaptureConsistencyError` (same-instruction-set contract). Direction: subj producer vs ref writers.

`diagnose_missing_edge` Phase 0 mirror: checks that every byte-key in `ref_edge.producer_write_byte_key` has at least one writer in `subj_graph.byte_key_writers`. If not, raises `CaptureConsistencyError`. Direction: ref producer vs subj writers.

`diagnose_extra_edge` Phase 1: uses `ref_cons_node.unrolled_position` (ref-side consumer position) for the closest-prior-writer lookup in `ref_graph.byte_key_writers`. The docstring (lines 4228–4237) explains: using the ref-side position ensures the query answers "what did ref's walk record as the latest writer for this byte-key at the moment ref's consumer appeared" — not the subj consumer's position. Risk 3 from C3e_67us plan §9 applies here.

`diagnose_missing_edge` Phase 0 mirror: for the closest-prior-writer lookup in `subj_graph.byte_key_writers`, use `c_node.unrolled_position` where `c_node` is the subj-side consumer (resolved by identity). This is the correct analog: we ask "what does subj's walk record as the latest writer for this byte-key at the moment subj's consumer appeared." The subj-side consumer's position is the right anchor because we are querying the subj graph. This is the principled choice, parallel to C3e using the ref-side position for ref-graph queries.

### E. Existing tests that exercise `diagnose_missing_edge`

Tests by file that pin `OrderInvertedFailure` or call `diagnose_missing_edge`:

1. **`test_p39d_gr_orderinverted_minimal.py::TestP39dGrOrderInvertedMinimal::test_compare_graphs_surfaces_orderinverted_on_gr_pair`** — pins `len(failures) == 1` and `isinstance(failure, OrderInvertedFailure)` with `body_label == "ML"` on both nodes. Both nodes are in the ML body. Under C3f: Phase 1 now uses unrolled positions. Since both nodes are in the same body, their SchedulePosition comparison and unrolled_position comparison agree. This test PASSES unchanged — same-body inversions are still caught.

2. **`test_p39d_gr_orderinverted_minimal.py::TestP39dPhase1RealKernel::test_real_kernel_raises_capture_consistency_via_edge_layer`** — pins `CaptureConsistencyError` with `"identity-coverage check at compare_graphs entry was bypassed"` text. After C3f: the Phase 0 `CaptureConsistencyError` message text CHANGES (identity-based lookup replaced with byte-key-based lookup with a different error message). This test must be re-pinned: the error should still be `CaptureConsistencyError` but with new message text reflecting the byte-key consistency check.

3. **`test_oplb_register_naming_minimal.py`** — two test methods pin `"identity-coverage check at compare_graphs entry was bypassed"` text. Same situation as above: after C3f the Phase 0 error message changes. Both must be re-pinned.

4. **`test_approach_a_non_cms_reference.py`** — one test pins `"identity-coverage check at compare_graphs entry was bypassed"`. Same re-pin required.

5. **`test_validate_pack_graph.py`** — pins `OrderInvertedFailure` on several pack-related test methods (pack_before_swap, swap inversion tests). All pinned failures are within-body reorders. Under C3f same-body inversions remain detectable via unrolled positions. These tests PASS unchanged.

6. **`test_dataflow_graph_scc.py::TestSCCClobberFailure`** — pins `OverriddenInputFailure` on SCC clobber path. SCC branch is in Phase 2+ (after Phase 0/1), no change.

7. **`test_dataflow_graph_builder.py`** — references `OrderInvertedFailure` in imports. No direct pinning seen requiring re-fixture.

**Re-fixture summary:** 4 tests (3 files) need message-text re-pinning for Phase 0 error text change. 0 tests require assertion logic re-pinning for Phase 1 (all pinned inversions are same-body).

**Cross-body inversions newly detectable:** After removing the body_label gate, cross-body inversions become `OrderInvertedFailure` candidates. On current production fixtures: the unrolled walk puts PRO, ML-1 (iter=0), ML (iter=1), NGL, NLL in strict order. A legitimate cross-iter pipelined edge (ML iter=0 writes, ML iter=1 reads) has producer.unrolled_position < consumer.unrolled_position — `default_p_before_c=True` and if subj also preserves that order, `subj_p_before_c=True` → fall through. No false `OrderInvertedFailure`. Cross-body inversions only fire when subj genuinely puts the producer after the consumer in the unrolled stream, which is a real defect.

### F. The 5 SCC tests that resurfaced after 56e3 (Option E)

Per `56e3_OptionE_IMPL_PLAN.md §6`: 6 SCC test methods (TestValidateSCCOverlap) in `test_ValidateSCCoverlap.py` resurfaced as PASSING after 56e3. These tests exercise `compare_graphs` on SCC-producing kernels and pin `OverriddenInputFailure` outcomes. The SCC clobber branch in `diagnose_missing_edge` (Phase 2 range, lines 4022–4051) is NOT touched by C3f. C3f only changes Phase 0 (producer lookup) and Phase 1 (body gate). The SCC tests reach the SCC clobber branch only after Phase 0 finds p_node and c_node successfully. Under C3f: if both SCC producer and consumer are in the same body (which the code comments confirm they always are, per the "SCC cleared at body boundaries" note), the byte-key lookup finds them, Phase 1 sees `default_p_before_c=True` and `subj_p_before_c=True` (order preserved — the clobber inserts between them, not inverts them), falls through to SCC branch, emits `OverriddenInputFailure`. No change in behavior for these tests. They remain PASSING after C3f.

### G. `OrderInvertedFailure` data model richness

The current `OrderInvertedFailure` dataclass fields:
- `producer: FailureNodeLabel` — carries `body_label` as diagnostic info (via `cms_node_label` which sets it from `subj_graph.body_for(p_node)`)
- `consumer: FailureNodeLabel` — same
- `default_producer_position: Optional[SchedulePosition]` — ref-side position, diagnostic
- `default_consumer_position: Optional[SchedulePosition]` — ref-side position, diagnostic
- `iter_delta: int` — from `p_node.iter_delta_to(c_node)`

After C3f: `default_producer_position` and `default_consumer_position` change type to `Optional[int]` (unrolled position int instead of `SchedulePosition`). The `SchedulePosition` object carries `loop_index` and `stream_index`; the `unrolled_position` int is a global monotone counter. The format method (`_format_canonical`) prints `self.producer.position` and `self.consumer.position` which are subj-side positions in the `FailureNodeLabel` — these don't change. The `default_*_position` fields are supplementary diagnostics; their type change is a contained refactor. Update the `Optional[SchedulePosition]` annotation to `Optional[int]`.

**Test sites that construct `OrderInvertedFailure` with `SchedulePosition` arguments — these ALL need updating when the field type changes:**
- `test_failure_formatters.py` lines 112–113, 134–135, 501–502 (3 constructor calls, 6 field assignments)
- `test_graph_native_validation_base.py` lines 323–324, 338–339, 350–351 (3 constructor calls, 6 field assignments)
- `test_asm_source_failure_rendering.py` lines 106–107 (1 constructor call, 2 field assignments)

These tests will continue to pass at runtime (Python doesn't enforce dataclass field types), but the type annotation change must be made and callers updated to pass ints so that the fields carry correct values for Phase 2 debugging and any future typed consumers. Step 1 of the implementation must include auditing all 7 constructor call-sites.

`iter_delta = p_node.iter_delta_to(c_node)` uses `p_node.iter_index` and `c_node.iter_index` (set during the unrolled walk in C3c). This is already correct and unaffected.

**`iter_delta` in the `p_node is None` branch:** the Phase 1 pseudocode leaves `iter_delta=...` as a placeholder for the `p_node is None` path. With no subj-side `p_node`, use `ref_edge.producer.iter_delta_to(ref_edge.consumer)` (ref-side) to compute the inter-iteration delta — this gives the semantically correct value since the failure is synthesized from the ref-side producer.

### H. CaptureConsistencyError parallel

`diagnose_extra_edge` Phase 0 raises `CaptureConsistencyError` when subj's producer byte-key has no writer anywhere in ref. The symmetric case for `diagnose_missing_edge`: ref's producer byte-key has no writer anywhere in subj. This SHOULD raise `CaptureConsistencyError` — it means subj's capture pipeline dropped an instruction that ref has, violating the same-instruction-set contract.

**Current code:** The current Phase 0 raises `CaptureConsistencyError` only when the identity lookup finds no matching node. Under byte-key migration:
- If `subj_graph.byte_key_writers` has NO entry for ANY byte-key in `ref_edge.producer_write_byte_key`: this is the missing-piece case. The subj capture has no writer for the producer's byte footprint. `CaptureConsistencyError` is the correct raise.
- If entries exist for all byte-keys but none are prior to the consumer: the producer existed in subj but is misplaced. NOT a `CaptureConsistencyError` — this is either an `OrderInvertedFailure` (the only writer is AFTER the consumer) or requires further Phase 2 analysis.

The consumer identity-based lookup still raises `CaptureConsistencyError` when the consumer is absent from subj. This is a pre-existing, correct contract that stays.

**This IS a missing piece relative to the current code:** The current code raises on identity-mismatch for the producer, but the byte-key path needs to explicitly handle the "byte-key has no entry in subj" case as a `CaptureConsistencyError`. Document the condition clearly in the Phase 0 implementation.

---

## §3 Design — new Phase 0 byte-key lookup, new Phase 1 unrolled-position order check

### Phase 0 replacement (byte-key-driven)

```python
# Phase 0 — byte-key-based producer + consumer lookup.
#
# Consumer: resolved by identity (consumer must be a DATA-FLOW node present
# in both captures — its absence is a capture-pipeline bug).
c_id = ref_edge.consumer.identity
c_node = next((n for n in subj_graph.nodes if n.identity == c_id), None)
if c_node is None:
    raise CaptureConsistencyError(
        f"diagnose_missing_edge: consumer identity {c_id!r} is absent from "
        f"subj_graph — the data-flow node present in ref has no counterpart "
        f"in subj. Capture-pipeline bug (same-instruction-set contract violated)."
    )

# Producer: resolved via byte-key reverse-index.
# For each byte-key in the ref producer's write footprint, find subj's
# most-recent writer strictly before the consumer's unrolled position.
# This mirrors diagnose_extra_edge's closest-prior-writer logic (C3e),
# but applied to the subj graph: "what does subj say should be the
# producer for this byte_key at this consumer position?"
bks = ref_edge.producer_write_byte_key
cons_unrolled = c_node.unrolled_position

# Check: ref's producer byte-key must have at least one writer in subj.
# If subj has no writer at all for a byte-key the ref producer writes,
# the same-instruction-set contract is violated.
missing_in_subj = [bk for bk in bks
                   if not subj_graph.byte_key_writers.get(bk)]
if missing_in_subj:
    raise CaptureConsistencyError(
        f"diagnose_missing_edge: ref edge's producer writes byte_keys "
        f"{missing_in_subj!r} that have NO writer anywhere in subj. "
        f"Violates the same-instruction-set contract. Capture-pipeline bug."
    )

# Find the closest-prior writer in subj for each byte-key.
p_node = None
for bk in bks:
    writers = subj_graph.byte_key_writers[bk]  # List[(GraphNode, unrolled_pos)]
    priors = [(w, p) for (w, p) in writers if p < cons_unrolled]
    if priors:
        candidate, _ = max(priors, key=lambda wp: wp[1])
        if p_node is None:
            p_node = candidate
        # If multiple byte-keys give different candidates, use the first
        # (diagnostic-only; all candidates write the same byte footprint
        # from the same logical instruction).
```

If `p_node` is `None` after the loop (all writers for every byte-key are AFTER the consumer): the edge is "genuinely absent" — the producer hasn't fired yet when the consumer reads. This is a real ordering inversion: in subj, the consumer reads before any prior writer exists. Skip to Phase 1 with `p_node = None`; handle the None case there.

### Phase 1 replacement (unrolled-position order check, no body gate)

```python
# Phase 1 — order check over unrolled positions.
# ref_p is the ref-side GraphNode for the producer; its unrolled_position
# reflects the canonical (default-schedule) stream order.
ref_p = ref_edge.producer
ref_c = ref_edge.consumer

default_p_before_c = ref_p.unrolled_position < ref_c.unrolled_position

if p_node is None:
    # No prior writer in subj — the consumer fires with no available
    # producer. If default has p before c (the normal case), this IS
    # an inversion: the producer is absent from the pre-consumer window.
    if default_p_before_c:
        # Synthesize a label from the ref-side producer for the failure message.
        # Use ref body info since p_node is not in subj's pre-consumer window.
        return [OrderInvertedFailure(
            producer=...,   # label from ref_edge.producer
            consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
            iter_delta=...,
            default_producer_position=ref_p.unrolled_position,
            default_consumer_position=ref_c.unrolled_position,
        )]
    # If default also has no prior writer (p_node is None AND default_p_before_c=False),
    # this is a kind_rank-induced edge — fall through to Phase 2.
else:
    subj_p_before_c = p_node.unrolled_position < c_node.unrolled_position
    if default_p_before_c and not subj_p_before_c:
        return [OrderInvertedFailure(
            producer=cms_node_label(p_node, subj_graph.body_for(p_node)),
            consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
            iter_delta=p_node.iter_delta_to(c_node),
            default_producer_position=ref_p.unrolled_position,
            default_consumer_position=ref_c.unrolled_position,
        )]
    if default_p_before_c and subj_p_before_c:
        pass  # order preserved — fall through to Phase 2
    # default_p_before_c = False: kind_rank-induced edge — fall through
```

The body_label gate is completely gone. Cross-body and cross-iter inversions are detected naturally.

### OrderInvertedFailure field type update

Change `default_producer_position: Optional[SchedulePosition]` to `Optional[int]` and `default_consumer_position: Optional[SchedulePosition]` to `Optional[int]`. Update all constructors that pass `ref_p.position` to pass `ref_p.unrolled_position` instead. Update the 7 test-side constructor call-sites enumerated in §2G (test_failure_formatters.py ×3, test_graph_native_validation_base.py ×3, test_asm_source_failure_rendering.py ×1) to pass integer unrolled positions.

In the `p_node is None` Phase 1 branch, set `iter_delta=ref_edge.producer.iter_delta_to(ref_edge.consumer)` (ref-side nodes) since no subj-side `p_node` is available.

---

## §4 Symmetry with `diagnose_extra_edge` (C3e)

| Dimension | `diagnose_extra_edge` | `diagnose_missing_edge` after C3f |
|---|---|---|
| Direction | subj has extra edge | ref has edge subj lacks |
| Phase 0 consistency check | ref must have writers for subj prod's byte-keys | subj must have writers for ref prod's byte-keys |
| Position anchor for lookup | ref-side consumer's `unrolled_position` | subj-side consumer's `unrolled_position` |
| Graph queried | `ref_graph.byte_key_writers` | `subj_graph.byte_key_writers` |
| Closest-prior-writer result | ref's "expected producer" | subj's "actual closest producer" |
| Result when mismatched | `EdgeRoutedDifferentlyFailure` | `OrderInvertedFailure` or Phase 2 wait failure |
| Unresolved case (no prior writer) | ref has writers but all are after consumer → routing diverge | subj has writers but all are after consumer → order inverted |

The position anchor choice mirrors the C3e rationale (Risk 3 from C3e §9): we use the consumer's position IN THE GRAPH BEING QUERIED. For `diagnose_extra_edge` querying `ref_graph`, the ref consumer's position is used. For `diagnose_missing_edge` querying `subj_graph`, the subj consumer's position is used. This ensures the query answers "what was the closest prior writer in this graph's own timeline at the moment this graph's consumer appeared."

---

## §5 Re-fixture work

Three test files pin the current Phase 0 error message text that will change:

1. **`test_p39d_gr_orderinverted_minimal.py::TestP39dPhase1RealKernel::test_real_kernel_raises_capture_consistency_via_edge_layer`** — pins `"identity-coverage check at compare_graphs entry was bypassed"`. After C3f: consumer identity lookup still raises `CaptureConsistencyError` when the consumer is absent, but the message text changes. Re-pin to the new consumer-absent message text.

2. **`test_oplb_register_naming_minimal.py`** — two tests pin the same `"identity-coverage check at compare_graphs entry was bypassed"` substring. Same re-pin.

3. **`test_approach_a_non_cms_reference.py`** — one test pins the same text. Same re-pin.

All four test assertions need only the message substring updated. The `CaptureConsistencyError` exception class itself stays; only the message text changes. No logic inversion required — all four tests were asserting "Phase 0 raises on missing-identity" and after C3f they assert "Phase 0 raises on missing-byte-key-writer-in-subj" for the consumer's byte-keys (or on missing consumer identity).

**Assessment:** The underlying test scenario for `test_oplb_register_naming_minimal` involves T/X register-naming divergence that causes the ref edge's producer byte-keys to be absent from the subj graph (because the subj uses X registers, producing different byte indices than T registers). Under C3f, this surfaces as the new `CaptureConsistencyError` path ("ref edge's producer writes byte_keys that have NO writer in subj"). The test STILL raises `CaptureConsistencyError` — the error class doesn't change, only the message. Re-pin the message substring.

**No other re-fixture work.** Phase 1 behavior for same-body inversions is identical to today. Cross-body inversions that were previously skipped are now detectable, but on current production fixtures no cross-body inversions exist (the unrolled walk correctly sequences ML-1 before ML).

---

## §6 Step-by-step implementation order

**Step 1 — Update `OrderInvertedFailure` field types and all call-sites**

Change `default_producer_position: Optional[SchedulePosition]` and `default_consumer_position: Optional[SchedulePosition]` to `Optional[int]`. The format method does not use these fields directly in its canonical rendering (it uses `self.producer.position` and `self.consumer.position` which are subj-side FailureNodeLabel positions). Update the type annotations AND all call-sites — 7 test-side constructor calls in test_failure_formatters.py (×3), test_graph_native_validation_base.py (×3), and test_asm_source_failure_rendering.py (×1) that currently pass `SchedulePosition` objects must be updated to pass integer unrolled positions. Failing to update these sites leaves the annotation and the actual values mismatched.

**Step 2 — Implement new Phase 0 in `diagnose_missing_edge`**

Replace lines 3932–3944 with the byte-key-driven Phase 0 described in §3. The consumer lookup stays identity-based (scan `subj_graph.nodes`) but raises a new `CaptureConsistencyError` message. The producer lookup uses `subj_graph.byte_key_writers`. Handle the "no writers in subj for this byte-key" case as `CaptureConsistencyError`. Populate `p_node` as the closest-prior writer by subj-side consumer unrolled_position.

**Step 3 — Implement new Phase 1 in `diagnose_missing_edge`**

Replace lines 3986–4005. Remove the `if p_node.body_label == c_node.body_label` gate. Change `p_node.position < c_node.position` to `p_node.unrolled_position < c_node.unrolled_position`. Change `ref_p.position < ref_c.position` to `ref_p.unrolled_position < ref_c.unrolled_position`. Handle `p_node is None` case (all subj writers are after consumer) as described in §3. Update `OrderInvertedFailure` constructor to pass `unrolled_position` ints for `default_producer_position` and `default_consumer_position`.

**Step 4 — Update Phase 2 to handle `p_node` from byte-key lookup**

Phase 2 currently uses `p_node.position` for `waits_in_window(subj_graph, p_node.position, c_node.position, ...)`. After C3f, `p_node` comes from `subj_graph.byte_key_writers` (a `GraphNode` with `.position` set). The `GraphNode.position` is a `SchedulePosition` that `waits_in_window` accepts. No change needed IF `p_node` is a subj-side `GraphNode` (which it is — `byte_key_writers` stores `(node, unrolled_position)` where `node` is a `GraphNode` from the subj walk). Verify: `waits_in_window` uses `node.position` (a `SchedulePosition`) for ordering; `GraphNode.position` carries the body-local position for the wait window scan. This is correct.

If `p_node` is `None` after Phase 0/1 (genuinely no prior writer but `default_p_before_c=False` — kind_rank-induced edge, fell through): Phase 2 needs a non-None `p_node` for `counter_for(p_node)`. Use the ref-side producer node (`ref_edge.producer`) as the `p_node` proxy for Phase 2 counter and label derivation. This handles the kind_rank edge case without introducing a silent miss.

**Step 5 — Re-pin the 4 tests with new Phase 0 error message text**

Update the `assert "identity-coverage check at compare_graphs entry was bypassed" in msg` assertions to the new Phase 0 message text. Determine the exact new text after Step 2 is implemented. The three files: `test_p39d_gr_orderinverted_minimal.py`, `test_oplb_register_naming_minimal.py`, `test_approach_a_non_cms_reference.py`.

**Step 6 — Run the unit test suite, classify any unexpected failures**

Run `tox -e unit -- --ignore=.../test_MatrixInstructionConversion.py`. Compare failure count against the §7 prediction. Any unexpected failure gets classified per the (a)/(b)/(c) taxonomy.

---

## §7 Validation — expected failure delta vs end-of-C3e

**End-of-C3e baseline:** 20 FAILED + 2 ERROR (per 56e3_OptionE_IMPL_PLAN.md §I, which predicts 20 FAILED + 2 ERROR after Option E; the C3e bead reports "failure delta = 0 new failures on current fixtures" per §8, confirming the count stays at 20 FAILED + 2 ERROR through C3e).

**C3f changes:**

Phase 0 change: the 4 re-pinned tests currently FAIL (they pin `CaptureConsistencyError` with the old message text, but the old message is produced by the old identity-based lookup; after C3f the message text changes). Wait — these tests are NOT currently in the FAILED count because they pin a message that the current code DOES produce. After C3f the code produces a DIFFERENT message (or takes a different code path), so the message-text assertion fails. Net: these 4 tests go from PASS → FAIL unless re-pinned.

After re-pinning in Step 5: those 4 tests return to PASS.

Phase 1 change: no current FAILED test pins "cross-body inversions should not be flagged" behavior, because the existing body_label gate was transparent — it silently skipped cross-body pairs. Removing the gate exposes inversions that didn't exist before (requires a real cross-body inversion in the test data). On current production fixtures, no cross-body inversions exist. Net: 0 tests flip.

**Expected net delta from C3f:** 0 FAILED change after re-pinning (4 tests temporarily flip during implementation, then return to PASS in Step 5). Validator state: 20 FAILED + 2 ERROR unchanged.

**Note on environment variance:** The bead description says "20 + 2 or 21 + 2 depending on env" — the +1 uncertainty comes from a test that may or may not be counted as FAILED depending on the rocisa build state (the `test_MatrixInstructionConversion.py` excluded test or a borderline xfail). The C3f delta from this baseline is 0 in either environment.

---

## §8 Risks / open questions

**Risk 1 — p_node from byte-key may differ from identity-based p_node:** The current code finds `p_node` by identity. The new code finds it by closest-prior-writer via byte-key. For same-body edges where the ref producer IS the subj producer (no reorder), these should produce the same node. For reordered edges, they may differ: the identity-based lookup finds the node wherever it is in subj (could be before OR after the consumer); the byte-key lookup finds the closest-prior-writer (only writers BEFORE the consumer). If the identity-based lookup was finding a node AFTER the consumer (which would have given `subj_p_before_c = False`), the byte-key lookup finds `None` (no prior writer). Both paths reach the same conclusion (`OrderInvertedFailure`), just via different representation of `p_node`. Verify: the Phase 1 outcome is identical.

**Risk 2 — Phase 2 `p_node.position` used in `waits_in_window`:** The `SchedulePosition` comparison in `waits_in_window` uses `<=` and `<` against `SchedulePosition` objects. A `GraphNode` from `byte_key_writers` carries a valid `SchedulePosition` (set during graph construction). Verify that cross-body cases (p_node in ML-1, c_node in ML) produce a valid window scan — `all_nodes_in_order` already handles cross-body by concatenating bodies in order, and `waits_in_window` uses `all_nodes_in_order` internally. No issue.

**Risk 3 — `ref_p.unrolled_position` availability:** `ref_edge.producer` is a `GraphNode` from the ref graph. Its `unrolled_position` is set by C3c during graph construction. Since C3c is a prerequisite (already landed), this field is present. Confirm: the `DataflowEdge` diagnostic fields `producer_unrolled_position` and `consumer_unrolled_position` are also set (populated at edge-formation time in C3c). Either `ref_edge.producer.unrolled_position` or `ref_edge.producer_unrolled_position` can be used — prefer the direct GraphNode field.

**Risk 4 — `p_node is None` path in Phase 2:** When p_node is None (all subj writers after consumer) AND `default_p_before_c=False` (kind_rank edge), Phase 2 needs a counter. Using `ref_edge.producer` as proxy for `counter_for()`: `counter_for` dispatches on category. The ref-side GraphNode has the same category as the equivalent subj producer (same-instruction-set contract). This is safe.

**Risk 5 — Multi-byte-key footprint with disagreeing closest-prior writers:** When `bks` has multiple byte-keys that resolve to different `p_node` candidates, the current design picks the first. This is the same approach as `diagnose_extra_edge`. The disagreement itself could be surfaced as diagnostic info but is not currently a separate failure type. Accept for now; file a bead if multi-key disagreement surfaces unexpected failures in practice.

**Open question 1 — `p_node` label in `OrderInvertedFailure` when `p_node` is the closest-prior-writer but is NOT the identity-matching producer:** The current code uses the identity-matched `p_node` for the label. The new code uses the byte-key-matched closest-prior writer. These may be different nodes (same identity in different bodies under the unrolled walk, or a genuinely different instruction in the case of reordering). The label change is acceptable — the closest-prior writer IS the relevant node from the subj graph's perspective: it's what subj produced at that byte-key before the consumer. The identity match is less important for the diagnostic label than showing the user which subj instruction is the "current effective producer" at the time of the consumer.

**Open question 2 — Whether to remove the defensive identity-equality fallback (lines 3967–3978) in C3f:** The plan doc says "Defensive identity-equality fallback retained so a future test constructs an edge_keys variant that re-introduces a position-like discriminator without breaking this branch." Under byte-key edge_keys (C3d+), this branch should never fire — identical byte-key tuples cancel in set-diff before reaching `diagnose_missing_edge`. Recommend: keep in C3f (no-op, safe), remove in C3g as part of the broader body_label control-flow cleanup.

---

## §9 New beads to file

**No new beads required** based on this investigation. All discovered items are accounted for:

- Phase 0 message-text re-fixture work (4 assertions, 3 files): this is (b)-class re-fixture, contained in Step 5 of the implementation.
- `OrderInvertedFailure` field type change AND test call-site updates (7 test constructors in 3 files): contained in Step 1, not a separate bead.
- Multi-byte-key disagreement (Risk 5): not a discovered bug; a known design simplification. If it surfaces a real failure during Step 6, file a P0 bead per the standing rule at that time.
- Phase 2 `p_node is None` proxy handling: implementation detail resolved in §3/§6, no separate bead.
- Removal of the defensive identity-equality fallback in C3g: already in scope for C3g (body_label control-flow migration). If not already in C3g's bead scope, add it as a sub-item; no separate top-level bead.

**Scrutiny resolutions (critic, 2026-06-08):**
- "No prior writer before consumer" → `OrderInvertedFailure` is the correct classification. The writers exist in subj (so it is NOT `CaptureConsistencyError`); they are simply positioned after the consumer. `CaptureConsistencyError` is reserved for the case where NO writer for the byte-key exists anywhere in subj.
- Consumer identity-lookup robustness: confirmed adequate. The per-category count gate at `compare_graphs` entry guarantees data-flow nodes match across both captures before `diagnose_missing_edge` is called; a missing consumer would have fired the gate first.
- SCC edges are same-body by construction: verified against test_dataflow_graph_scc.py (all SCC clobber tests use BODY_LABEL_ML for both nodes) and the CMSValidator comment at lines 4024–4030.
