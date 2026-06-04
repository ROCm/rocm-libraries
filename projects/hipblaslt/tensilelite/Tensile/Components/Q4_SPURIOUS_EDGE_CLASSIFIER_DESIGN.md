# Design memo — `diagnose_extra_edge` classifier

**Status:** decision-ready (2026-06-04). Resolves Q4 of `UNROLLED_VALIDATION_ANSWERS.md`; gates commit 3 of `UNROLLED_VALIDATION_PLAN.md`.

## Verdict

Adopt **Option C — hybrid byte-key-driven multi-phase classifier**, structurally mirroring `diagnose_missing_edge` but with every branch decision driven by reverse-index byte-key lookups (no `canonical_render` / category / subiter pattern matching). The classifier produces exactly **three** typed terminal outcomes:

1. **`CaptureConsistencyError`** — raised, not returned. Subj emits a write at byte-keys that ref's capture has no writer for, anywhere in the unrolled stream. Per `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3` the contract is that both captures observe the same physical instructions; a CMS-only write is a capture-pipeline bug, not a CMS schedule defect.
2. **`EdgeRoutedDifferentlyFailure`** (new typed `Failure` subclass) — subj's consumer reads bytes that ref's consumer also reads, but subj resolves the consumer's most-recent-prior writer at a different unrolled position than ref does. This is the symmetric counterpart of `OverriddenInputFailure` from the missing-edge side: CMS inserted (or moved into position) an intervening write that displaced ref's producer. **All real CMS-extra edges after the unrolled walk land here.**
3. **`UnexplainedExtraEdgeError`** — raised, not returned. Classifier fall-through. Validator bug (the byte-key reverse index found a counterpart on ref, the position relationship is consistent with ref's, and yet edge_keys still disagreed — this should be impossible under the byte-key keying basis and unrolled walk).

There is no `SpuriousEdgeFailure` subclass in this design — its name presupposes "the edge is wrong," but under the §3 same-instruction-set contract every surviving CMS-extra after the unrolled walk is *either* a capture-pipeline bug (raise) *or* a routing divergence whose semantics match `OverriddenInputFailure`. Naming the routing case `EdgeRoutedDifferentlyFailure` rather than `SpuriousEdgeFailure` keeps the semantic intent (a clobber) clear and avoids the trap of treating legitimate re-routing as a defect.

**Key invariant:** classification is derived from byte-key reverse-index queries on the ref graph plus unrolled-position comparison. No instruction-class lists, no category-prefix matches, no subiter arithmetic, no `_is_alu_producer`-style guards. The m7o5 anti-pattern (a hardcoded class list growing per discovered case) is not introduced.

---

## The semantic contract

Per `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3` and `§1` (`CMSValidator.py` lines around the v5 design's same-instruction-set precondition):

> SHADOW and CMS captures see the **same physical rocisa instructions** because both pipelines run the default scheduler and the CMS scheduler over the **same CMS-derived codegen state**. They differ ONLY in stream ordering.

This contract has a sharp consequence for CMS-extra edges. A dataflow edge `(producer, consumer, byte_keys, slots, kind)` requires:

- A producer that writes those byte_keys somewhere in the stream
- A consumer that reads those byte_keys somewhere in the stream
- The producer to be the most-recent-prior writer in the unrolled walk

Under the §3 contract, **the set of (producer-writes-byte_key) facts is the same on both sides**: every physical write in CMS is also in SHADOW (because both captured the same physical instruction). The set of (consumer-reads-byte_key) facts is also the same. The only thing that can differ is *which producer is the most-recent-prior writer* — i.e., the position of writers relative to consumers in the unrolled stream.

So under the unrolled walk + byte-key edge_keys, a CMS-extra edge means exactly one thing semantically: **CMS placed a different writer in the most-recent-prior slot for this consumer than SHADOW did**. There are three sub-cases for *why* the writer differs:

1. **Pipelining re-routing** — CMS's most-recent-prior writer is a DIFFERENT iter copy of the SAME logical instruction (the rotating pack-buffer pattern from `n7og_PROBE_REPORT.md`). Under iter-blind identity, both writers carry the same `identity` tuple but live at different unrolled positions. **Cancellation note:** iter-blind identity + byte-key edge_keys means producer.identity is removed from the keying basis; the edge_keys (`producer_write_byte_keys, consumer_read_byte_keys, kind, intra, src_slot, sink_slot`) are byte-equal on both sides → set-diff cancels them → this sub-case doesn't reach `diagnose_extra_edge` at all. This is what `UNROLLED_VALIDATION_PLAN.md §3.3` ("Cross-body / cross-iter live-ins resolve naturally") asserts and what the 192 NLL extras + 16 NGL extras both reduce to under the rewrite.
2. **Clobber** — CMS placed a *different physical instruction* (different identity) writing the same byte_keys between SHADOW's producer and the consumer. The consumer now reads from the intervening writer, not from SHADOW's original producer. This IS a real CMS defect and is structurally identical to the SCC `OverriddenInputFailure` case from `diagnose_missing_edge:3874-3900` — the "displaced producer's value" semantics generalize from SCC to arbitrary byte_keys under the unrolled walk.
3. **Capture-pipeline bug** — CMS subj's producer writes bytes that ref's capture genuinely has no writer for. This violates the §3 contract and must surface loud, not be classified as a CMS defect.

The classifier's job is to discriminate (2) from (3). (1) doesn't reach the classifier under correct iter-blind identity + byte-key edge_keys.

---

## Empirical grounding

The 16 NGL "missing in SHADOW" edges (= CMS-extras in the symmetric direction) on BPG#11 (`n7og_PROBE_REPORT.md` per-body category breakdown):

| (p_cat, c_cat, kind) | SHADOW | CMS | Δ (CMS-extra) |
|---|---:|---:|---:|
| `(LRA3, MFMA, raw_intrawave)` | 0 | 8 | +8 |
| `(LRB3, MFMA, raw_intrawave)` | 0 | 8 | +8 |

Both LRA3 and LRB3 producers exist as captured GraphNodes on both sides (NGL node counts: SHADOW=28, CMS=28). The mechanism, per `n7og_PROBE_REPORT.md`:

- CMS NGL stream: LR3 producers placed BEFORE MFMA consumers → CMS's body-local `latest_writer` populated → CMS emits 16 edges.
- SHADOW NGL stream: same physical LR3 producers placed AFTER MFMA consumers → SHADOW's `latest_writer` empty → SHADOW emits 0 edges.

This is the exact mirror of the 192 NLL extras (CMS places consumers before producers in NLL → CMS emits 0 → SHADOW emits 192). Both are body-local-walk artifacts.

**Categorization under the design v5 sub-cases:**

All 16 are **sub-case 1 (pipelining re-routing)**: the SHADOW and CMS LR3 producers are the same physical instructions (NGL CVT-instance sharing per the probe's A3 finding generalizes to LR producers — both pipelines see the same codegen). Under the unrolled walk:

- The MFMA's read of these byte_keys in CMS NGL resolves to the LR3 from the immediately preceding iter (or the LR3 in ML's tail of the previous unrolled iter, depending on `ML_MAT_COUNT`).
- The same MFMA's read in SHADOW NGL resolves to the *same* LR3 (because the unrolled walk doesn't reset `latest_writer` at body boundaries, so SHADOW's NGL consumers can see ML's tail writers).
- Both sides' edges have byte-equal edge_keys → set-diff cancels them → these 16 don't reach `diagnose_extra_edge`.

**Net empirical finding:** ZERO of the currently-known CMS-extras on BPG#11 reach `diagnose_extra_edge` after the unrolled walk lands. The classifier exists for **future** CMS schedules that legitimately introduce sub-case 2 (clobber) — and for **defending** against sub-case 3 (capture-pipeline bug regressing the §3 contract). It will fire rarely-or-never on the current fixture corpus; this is acceptable and expected. The classifier's value is the loud failure mode when those cases DO arise.

> **Probe note:** the live BPG#11 SHADOW/CMS capture pipeline is currently broken on the branch tip (`_build_shadow_cms_pair` returns None for SHADOW — see the bead filed in `UNROLLED_VALIDATION_ANSWERS.md`'s "Beads filed" section). Empirical re-probing the 16 NGL extras through the unrolled walk to confirm "all 16 cancel" cannot be done in this session. The reasoning above is from the pre-existing report data + the design contract; the commit-3 acceptance criterion ("n7og fixtures resolve to 0 mismatches in both directions") is the empirical confirmation.

---

## Design space evaluation

### Option A — Byte-key-based, single-pass

**Sketch:**

```python
def diagnose_extra_edge(subj_edge, ref_graph):
    bk = subj_edge.producer_write_byte_key  # tuple of byte_keys
    ref_writers = ref_graph.byte_key_writers.get(bk[0], [])
    if not ref_writers:
        return [SpuriousEdgeFailure(...)]   # ref has no writer at all
    # ref has writers — figure out which one ref would have paired with the consumer
    cons_pos = subj_edge.consumer.unrolled_position
    ref_most_recent = max((w for w in ref_writers if w.position < cons_pos),
                          key=lambda w: w.position, default=None)
    if ref_most_recent is None or ref_most_recent.identity != subj_edge.producer.identity:
        return [EdgeRoutedDifferentlyFailure(...)]
    return [UnexplainedExtraEdgeError(...)]
```

**Pros:** simple, single pass, lookups are O(1).

**Failure modes I found:**

- Looks up byte_keys in `ref_graph.byte_key_writers` using `bk[0]`. **But byte_keys is a tuple** — if the consumer reads 4 bytes and the producer writes 4 bytes, `bk` is a 4-tuple. A producer that writes only bytes (31, 32) may match a ref writer for byte 31 but not byte 32. The single-byte-key probe loses the multi-byte structure of the edge.
- Treats the consumer's "most recent prior writer in ref" as definitionally the writer ref WOULD have paired with. But ref's actual edge structure is built by `build_dataflow_graph`'s Phase 2 with the same `latest_writer` walk; ref might have multiple writers for the same byte_keys at different positions. The "is the closest-prior" reasoning is correct, but Option A's pseudocode glosses over the per-byte fan-out (when consumer reads N bytes from M different writers, ref emits M edges, not 1).
- Conflates "ref has no writer" (sub-case 3 — capture bug) with "ref has a writer but at a position after the consumer" (which under the §3 same-instruction-set contract shouldn't happen because the same producers are in both streams at the same identities; if the producer is post-consumer in ref but pre-consumer in subj, that's a legitimate `OrderInvertedFailure`-symmetric case ref-side would have flagged when iterating its own edges). The single-pass branch doesn't distinguish them.

**Verdict:** correct intuition (byte-key-driven), but the "single pass + three branches" framing under-specifies the per-byte structure of edges and misclassifies "ref has writers but none pre-consumer" as "ref has no writer." The classifier needs more structure.

### Option B — Mirror `diagnose_missing_edge`'s Phase 0/1/2

**Sketch:**

```python
def diagnose_extra_edge(subj_edge, ref_graph):
    # Phase 0: identity lookup of subj's producer/consumer in ref
    p_node = ref_graph.lookup_by_identity(subj_edge.producer.identity)
    c_node = ref_graph.lookup_by_identity(subj_edge.consumer.identity)
    if p_node is None or c_node is None:
        raise CaptureConsistencyError(...)
    # Phase 1: order check
    if p_node.position > c_node.position and ...:  # ref has p-after-c
        return [OrderInvertedFailure-symmetric(...)]  # symmetric phrasing
    # Phase 2: wait coverage
    ...
    raise UnexplainedExtraEdgeError(...)
```

**Pros:** code shape mirrors the existing classifier; reviewers familiar with `diagnose_missing_edge` recognize it immediately.

**Failure modes I found:**

- **Phase 0 uses identity lookup. Under the unrolled walk + iter-blind identity, identity is no longer a primary key** (per `UNROLLED_VALIDATION_PLAN.md §2.2`, byte-key replaces identity-tuple-as-primary-key). Phase 0 falls back to "is there ANY node in ref with this identity" — but multiple iter copies of an ML producer share the same identity, so the lookup is ambiguous (which iter copy?). The plan resolves this at the `diagnose_missing_edge` site via byte-key reverse-index lookup; B's pure mirror would carry forward the pre-rewrite Phase 0 shape and undercuts the byte-key migration.
- **Phase 1 (order check) is the wrong question for extras.** A CMS-extra edge means subj has the edge AT ALL — not "subj has it in the wrong order." Under the unrolled walk, ordering is what `OrderInvertedFailure` from the missing-edge side already catches. The symmetric Phase 1 question is empty: if ref had the same producer-consumer order subj does, with the same byte_keys, the edge would cancel in set-diff (because edge_keys are byte-key-based, not position-based). The only way subj has an extra is when subj's *producer is different from ref's most-recent-prior writer for those byte_keys* — which is Phase 0's question phrased differently, not a Phase 1 question.
- **Phase 2 (wait coverage)** doesn't transfer either. Wait coverage answers "is the schedule legal," not "does this edge exist." Wait coverage of a CMS-extra is checked by `validate_edge_wait_coverage` on the CMS graph independently; the cross-graph extra classifier doesn't re-derive it.

**Verdict:** mirroring three phases for the sake of symmetry produces two phases (1 and 2) that don't ask the right questions for the extras direction. The single semantically-meaningful question is Option A's "did ref have a different most-recent-prior writer for those byte_keys" — Phase 0's question, sharpened.

### Option C — Hybrid (byte-key-driven, multi-phase)

This is the recommended design. Phase structure mirrors `diagnose_missing_edge` for reviewer familiarity, but every phase's *decision* is byte-key-driven (no instruction class lists, no subiter arithmetic, no `_is_alu_producer`-style pattern matching).

**Sketch (full pseudocode in "Recommended design" below):**

- **Phase 0 (capture-consistency gating):** for each byte_key in `subj_edge.producer_write_byte_key`, query `ref_graph.byte_key_writers[bk]`. If ref has NO writers at all for ANY of the byte_keys in subj_edge's footprint, raise `CaptureConsistencyError` — the §3 contract is violated.
- **Phase 1 (re-routing classification):** for each byte_key, find ref's most-recent-prior writer relative to subj_edge.consumer's unrolled position. Compare against subj_edge.producer. If they share identity, the edge SHOULD have canceled in set-diff already; surface as `UnexplainedExtraEdgeError` (validator bug — the byte-key reverse index and the edge_keys disagree). If they differ in identity, emit `EdgeRoutedDifferentlyFailure` carrying both writers' labels — this is the clobber-equivalent.
- **Phase 2 (fall-through):** unreachable under correct implementation; raises `UnexplainedExtraEdgeError`.

**Pros:**
- Single semantically-distinct decision per branch.
- Decisions are derived from byte-key reverse-index queries (a primitive the plan already builds eagerly per Q8). No new categorization machinery.
- Three terminal outcomes match the three structural sub-cases of "why CMS has an edge ref doesn't" (capture bug / re-routing / validator bug).
- Phase 0 is gating (raises if violated), like `diagnose_missing_edge`'s identity-coverage gate.
- `EdgeRoutedDifferentlyFailure` is the symmetric counterpart of `OverriddenInputFailure` — same shape (producer-clobbered-by-intervening-writer-then-consumer-reads), different framing (subj is what we're flagging vs. ref is what we're flagging).

**Failure modes I considered and rejected:**

- *"Per-byte fan-out: what if subj_edge covers 4 bytes and 2 of them have ref writers but 2 don't?"* — Under the §3 contract, all 4 bytes must have ref writers (same physical writes on both sides). Partial coverage means a capture-pipeline bug. Phase 0 must raise on the first missing byte; partial coverage is structurally `CaptureConsistencyError`.
- *"What if ref has writers at all byte_keys but the closest-prior for different bytes are different identities?"* — Possible if multiple smaller producers in ref cover the consumer's read (e.g., ref has 4 narrow CVTs at bytes 31, 32, 33, 34; subj has 1 wide PackB128 at all 4). But the same writers exist in both pipelines (§3); subj-side `_resolve_producers` would form 4 edges to the 4 narrow CVTs OR 1 edge to the wide PackB128, depending on the actual physical writes — not BOTH. If the physical writes differ between pipelines, that's a capture-pipeline bug (Phase 0). If the physical writes are the same but groupings differ, that's another representation issue subsumed by §3.
- *"What about pure ALU producers immediately visible to readers (no SWaitCnt drain)?"* — same answer. The classifier doesn't care about producer category; it only asks "is the closest-prior ref writer at this byte_key the same identity as subj's producer." Category-blindness IS the point.

**Verdict:** Option C survives scrutiny. Pseudocode below.

---

## Recommended design

### Pseudocode

```python
def diagnose_extra_edge(subj_edge: DataflowEdge, ref_graph: DataflowGraph) -> list[Failure]:
    """Classify a CMS-extra edge (present in subj, absent from ref under byte-key edge_keys).

    Terminal outcomes:
      - CaptureConsistencyError (raised): ref has no writer for one of subj's byte_keys.
      - EdgeRoutedDifferentlyFailure: ref's closest-prior writer for these byte_keys
        differs in identity from subj's producer (CMS inserted/moved an intervening
        writer between ref's producer and the consumer).
      - UnexplainedExtraEdgeError (raised): classifier fall-through (validator bug).
    """
    cons = subj_edge.consumer
    cons_pos = cons.unrolled_position
    subj_prod = subj_edge.producer
    subj_prod_identity = subj_prod.identity

    # Phase 0 — capture-consistency gating.
    # Every byte_key in subj's producer write must have at least one writer in ref;
    # otherwise the §3 same-instruction-set contract is violated.
    bks = subj_edge.producer_write_byte_key  # tuple of byte_keys
    missing_bks = [bk for bk in bks if not ref_graph.byte_key_writers.get(bk)]
    if missing_bks:
        raise CaptureConsistencyError(
            f"diagnose_extra_edge: subj emits an edge with producer writes at byte_keys "
            f"{missing_bks} that have NO writer anywhere in ref. "
            f"Violates the same-instruction-set contract "
            f"(DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3). Capture-pipeline bug."
        )

    # Phase 1 — closest-prior-writer comparison per byte_key.
    # For every byte_key in the footprint, identify ref's most-recent writer
    # whose unrolled_position is strictly less than the consumer's position
    # (the writer the unrolled walk's latest_writer would record for that byte
    # at the moment the consumer is resolved).
    ref_closest_priors: dict = {}  # bk -> (ref_writer_node, ref_writer_pos)
    for bk in bks:
        writers = ref_graph.byte_key_writers[bk]  # list of (node, pos)
        priors = [(w, p) for (w, p) in writers if p < cons_pos]
        if not priors:
            # ref has writers for this byte_key, but all of them are AFTER the consumer.
            # Under the §3 contract + iter-blind identity, this means ref's consumer
            # would also have no closest-prior writer — but then ref would not have
            # emitted an edge here, and subj-extras with no ref counterpart at the
            # same byte_keys is exactly what we're classifying. Fall through to
            # Phase 2 (treat as routing-differs with ref_writer=None).
            ref_closest_priors[bk] = (None, None)
            continue
        ref_closest_priors[bk] = max(priors, key=lambda np: np[1])

    # Identity comparison: does ref's closest-prior writer (per byte_key) match subj's producer?
    ref_writer_identities = {bk: (w.identity if w is not None else None)
                             for bk, (w, p) in ref_closest_priors.items()}
    all_match = all(ident == subj_prod_identity
                    for ident in ref_writer_identities.values())

    if all_match:
        # Ref's closest-prior writer is the SAME identity as subj's producer for every
        # byte_key. Under iter-blind identity + byte-key edge_keys, the edge SHOULD
        # have canceled in set-diff. That it didn't means the byte-key reverse index
        # and edge_keys disagree — a validator bug, not a CMS defect.
        raise UnexplainedExtraEdgeError(
            f"diagnose_extra_edge: ref's closest-prior writer matches subj's producer "
            f"identity for every byte_key in {bks}; this edge should have canceled in "
            f"set-diff. byte-key reverse index and edge_keys are inconsistent."
        )

    # Phase 1 verdict: ref's closest-prior writer for at least one byte_key differs
    # from subj's producer identity (or ref has no closest-prior writer for that
    # byte_key while subj does). CMS routed the consumer's read to a different
    # producer than SHADOW did. Structurally a clobber/displacement.
    #
    # Pick a representative ref writer for the failure message: the writer at the
    # first byte_key whose identity differs from subj's producer. Diagnostic only;
    # the per-byte detail is conveyed via the failure's `byte_key_routing` field.
    ref_repr_bk = next(bk for bk in bks if ref_writer_identities[bk] != subj_prod_identity)
    ref_repr_writer, ref_repr_pos = ref_closest_priors[ref_repr_bk]

    return [EdgeRoutedDifferentlyFailure(
        subj_producer=cms_node_label(subj_prod, subj_graph_unused_here=None, ...),
        subj_consumer=cms_node_label(cons, ...),
        ref_producer=(cms_node_label(ref_repr_writer, ...)
                      if ref_repr_writer is not None else None),
        byte_keys=bks,
        # iter_delta computed from the unrolled-position difference between
        # subj_prod and cons (per Failure base class convention).
        iter_delta=subj_prod.iter_delta_to(cons),
        # Per-byte routing diagnostic for diff'ing what ref would have paired with
        # vs. what subj did pair with.
        byte_key_routing={bk: (ref_writer_identities[bk], subj_prod_identity)
                          for bk in bks},
    )]
```

**Notes on the pseudocode:**
- `ref_graph.byte_key_writers` is the eager reverse index from `UNROLLED_VALIDATION_PLAN.md §3.2` / answer Q8. Type: `Dict[byte_key, List[Tuple[GraphNode, int]]]`.
- The "closest-prior" definition aligns with Q6's resolution (most-recent prior writer in unrolled order).
- `cms_node_label` already takes `(node, body_capture)`; the call sites in this pseudocode are abbreviated.

### Worked example on one of the 16 NGL extras (BPG#11)

Take `LRA3 → MFMA` in NGL, byte_key `('v', K)` (for some VGPR K covered by the rotating buffer):

- subj_edge.producer: CMS NGL's LRA3 producer (some `ds_read_b128`), unrolled_position = e.g. 20 (early in NGL).
- subj_edge.consumer: CMS NGL's MFMA, unrolled_position = e.g. 45.
- subj_edge.producer_write_byte_key: `(('v', K1), ('v', K2), ('v', K3), ('v', K4))` (4 bytes).

**Phase 0** — query `ref_graph.byte_key_writers[('v', K1)]`. Ref (SHADOW) under the unrolled walk has the SAME ds_read_b128 LRA3 producer (same identity — both pipelines see the same physical instruction per §3). The unrolled walk also has the ML body's prior LR producers and any earlier-iter LR3 producers in `byte_key_writers`. The list is non-empty for every byte_key. **No raise.**

**Phase 1** — find the closest-prior ref writer for each byte_key, position < 45 (the consumer's unrolled position in ref; ref's consumer has the same identity as subj's, and under the unrolled walk both pipelines' MFMA consumer at this identity lands at the same unrolled_position because the surrounding bodies are concatenated identically). The closest-prior writer in ref is the *same LRA3 producer identity as subj's* — because the unrolled walk doesn't reset latest_writer at body boundaries, both SHADOW NGL and CMS NGL see the same prior LR3 writer (either from the immediately-preceding ML iter or from earlier in NGL, depending on the schedule). **`all_match = True` → `UnexplainedExtraEdgeError` raised.**

Wait — that's the WRONG terminal outcome for this case. Let me re-examine.

**Correction:** under the unrolled walk + byte-key edge_keys, the edge from "the same LRA3 producer identity → the same MFMA consumer identity" with the same byte_keys / slots / kind produces the SAME edge_key on both sides. Set-diff cancels them at the `compare_graphs` level, BEFORE `diagnose_extra_edge` is ever called. So this case doesn't reach the classifier at all — and `all_match = True` happens to be unreachable for the same reason: if it were reachable, set-diff would have already canceled. The `UnexplainedExtraEdgeError` raise in the `all_match` branch is correct as a sanity check (catches the bug case where the reverse index and edge_keys disagree, which should be impossible under correct implementation).

So the 16 NGL extras' actual journey is:
1. They appear under TODAY's body-local walk (`n7og_PROBE_REPORT.md`).
2. Under the unrolled walk (§3.3 of the plan), they cancel at `compare_graphs` because the byte-key edge_keys are byte-equal on both sides.
3. `diagnose_extra_edge` is not invoked for any of them.

This is the empirically-correct outcome. The classifier fires only for cases that DON'T cancel — which means cases where ref's closest-prior writer for the relevant byte_keys at the consumer's position is a DIFFERENT identity than subj's producer. None of the current BPG#11 fixtures surface such a case.

**A hypothetical case where the classifier WOULD fire (constructed for completeness):** suppose CMS's schedule inserted a new `v_mov_b32` writing byte_key `('v', K)` between SHADOW's LR3 and the MFMA consumer. That `v_mov_b32` exists in CMS only (a real CMS scheduler bug — it shouldn't introduce instructions, only reorder them). The §3 contract would already be violated at the capture-comparison level (subj has a node ref doesn't), but suppose the contract check is permissive about scheduler-inserted control ops and only enforces it for dataflow categories. Then:
- subj_edge: `v_mov_b32 → MFMA`, byte_key `('v', K)`.
- Phase 0: query `ref_graph.byte_key_writers[('v', K)]` — non-empty (ref has LR3 writing it).
- Phase 1: ref's closest-prior writer for `('v', K)` at the MFMA's position is `LR3` (identity differs from subj's `v_mov_b32` identity). `all_match = False`.
- Emit `EdgeRoutedDifferentlyFailure(subj_producer=v_mov_b32, subj_consumer=MFMA, ref_producer=LR3, byte_key=('v', K), ...)`. The user reads: "CMS scheduled a v_mov_b32 between the LR3 producer and the MFMA consumer; SHADOW pairs the MFMA with the LR3."

That's the correct classification and the correct user-facing message. The clobber has been surfaced as a typed failure.

---

## Failure-class taxonomy

Three terminal outcomes:

| Outcome | Type | When | What it tells the user |
|---|---|---|---|
| `CaptureConsistencyError` | raised exception (existing class) | Phase 0: subj's producer writes byte_keys that have NO writer in ref. | The capture pipelines disagree on what physical instructions exist. This is a capture-layer bug per `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3`, NOT a CMS schedule defect. File a capture bead, not a CMS bead. |
| `EdgeRoutedDifferentlyFailure` | new `@dataclass` Failure subclass | Phase 1: ref has a closest-prior writer for the byte_keys, but its identity differs from subj's producer. | CMS placed a different writer in the most-recent-prior slot for the consumer than the default schedule did. Structurally a clobber. Mirror of `OverriddenInputFailure`. File a CMS bead. |
| `UnexplainedExtraEdgeError` | new exception class (mirrors `UnexplainedMissingEdgeError`) | Phase 1: ref's closest-prior writer matches subj's producer identity exactly (the edge SHOULD have canceled in set-diff) OR fall-through. | Validator bug. The byte-key reverse index and edge_keys are inconsistent; investigate the graph builder. |

**`EdgeRoutedDifferentlyFailure` schema:**

```python
@dataclass
class EdgeRoutedDifferentlyFailure(Failure):
    subj_producer: FailureNodeLabel = None
    subj_consumer: FailureNodeLabel = None
    ref_producer: Optional[FailureNodeLabel] = None  # may be None if ref had no prior writer at all
    byte_keys: tuple = ()                            # the byte_key footprint of the edge
    # Per-byte routing: bk -> (ref_writer_identity, subj_writer_identity).
    # Diagnostic for cases where different bytes of a wide read are routed
    # differently (rare but possible).
    byte_key_routing: dict = field(default_factory=dict)

    def _format_canonical(self) -> str:
        ref_part = (f"reference routes through {self.ref_producer.primary} {self.ref_producer.position}"
                    if self.ref_producer is not None
                    else "reference has no prior writer at this position")
        return (
            f"Subject's consumer {self.subj_consumer.primary} {self.subj_consumer.position} "
            f"reads from subject's producer {self.subj_producer.primary} {self.subj_producer.position} "
            f"at byte_keys {self.byte_keys}, but {ref_part}"
            f"{self._iter_suffix()}. The subject schedule inserted or moved an intervening "
            f"writer between the reference's producer and the consumer."
        )
```

**Why NOT `SpuriousEdgeFailure` as the failure name** — its name implies "the edge shouldn't exist." But the edge DOES exist in the CMS schedule's actual dataflow (the consumer IS reading from subj_producer at runtime). The defect isn't that the edge exists; it's that CMS introduced a clobber the default schedule didn't. `EdgeRoutedDifferentlyFailure` names the actual semantic problem ("the routing differs") rather than the surface artifact ("there's an edge we didn't expect").

---

## Edge cases I'm aware of

### SCC-clobber paths

`diagnose_missing_edge` has a dedicated SCC branch (`CMSValidator.py:3874-3900`) because SCC's "1-bit register, clears at body boundaries, easily clobbered" semantics surface as a recognizable pattern in the missing direction. Does the symmetric direction need an SCC branch?

**No.** Under the unrolled walk's byte-key model:
- SCC writes record byte_key `('s', 'scc')` (or whatever the sentinel byte_key is — `CMSValidator.py:3874` uses `getattr(ref_resource, "regType", None) == "scc"`, but byte_key is the abstraction the rewrite uses).
- A CMS-inserted SCC clobber between SHADOW's SCC writer and the SCC reader produces:
  - subj_edge: clobber → reader, byte_key `('s', 'scc')`.
  - Phase 0: ref has writers for `('s', 'scc')` (SHADOW's SCC writer). Non-empty. No raise.
  - Phase 1: ref's closest-prior writer for `('s', 'scc')` at the reader's position is SHADOW's SCC writer; its identity differs from subj's clobber identity. → `EdgeRoutedDifferentlyFailure`.

This is the right outcome. The byte-key model subsumes SCC handling; no special branch needed. The existing SCC branch in `diagnose_missing_edge` is also subsumed under the byte-key rewrite (per `UNROLLED_VALIDATION_PLAN.md §4.3` and answer Q3) — `OverriddenInputFailure` and `EdgeRoutedDifferentlyFailure` become the symmetric pair, both emitted from byte-key reverse-index queries.

### NOP-introduced waits

CMS might insert `s_nop` or `s_waitcnt` between SHADOW's producer and consumer for legitimate timing. These don't write any byte_keys; they don't create new dataflow edges. They surface as differences in the captured stream's instruction count (the `_data_flow_category_counts` gate in `compare_graphs:3696` filters them out), not as CMS-extra dataflow edges. **Not classifier-relevant.**

### Same-byte-different-writers within one edge

A consumer reads bytes (31, 32, 33, 34). Subj's producer writes all four. Ref's closest-prior writer for byte 31 is `LR3-iter-prev`, but for bytes 32-34 it's `LR3-iter-current` (different iter copies; under iter-blind identity they share identity but live at different positions). The `byte_key_routing` dict captures this; `all_match` is computed against `subj_prod_identity` (a single identity), so it correctly identifies that ref and subj share a single identity for the producer — and if subj's producer is the same identity (just at a different position), the edge cancels at edge_keys and never reaches `diagnose_extra_edge`. If subj's producer is a NEW identity (e.g., CMS inserted a wide PackB128 that writes all 4 bytes), `all_match` is False against any ref writer's identity, and `EdgeRoutedDifferentlyFailure` fires correctly.

### Multiple iter copies of the same logical writer

`ref_graph.byte_key_writers[bk]` may contain multiple entries with the same identity (different iter copies of an ML body's writer, distinct unrolled positions). Per Q6, the closest-prior is uniquely determined by unrolled-position comparison — pick the one with the largest position strictly less than the consumer's position. The classifier consistently uses this rule; no ambiguity.

### CMS writes to a byte_key that's the same logical resource but different iter

E.g., ML iter K's PackB writes the rotating-buffer half that subiter 0 of ML iter K+1 reads. Under iter-blind identity, the producer's identity in iter K and iter K+1 is the same; the edges in both pipelines are byte-equal and cancel. Doesn't reach the classifier. (This is the `n7og_PROBE_REPORT.md` 192 NLL + 16 NGL case.)

### CMS writes to byte_keys that are SHADOW's read targets but never SHADOW's write targets

Sub-case 3 (capture bug). Phase 0 catches this. Example: a hypothetical CMS scheduler bug that emits `v_mov_b32 v[K]` where SHADOW has no writer for `v[K]` at all in either dataflow or scheduler-control-op categories. The §3 contract is violated; `CaptureConsistencyError` is the right answer (file a capture bead).

---

## Open sub-questions (user-decision-required)

### Sub-question 1 — Should `CaptureConsistencyError` from Phase 0 mention the §3 contract by name?

The proposed error message references `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §3` as the contract being violated. This embeds doc-name dependence into the runtime error. **Recommendation:** yes, reference it — the runtime error is what users grep for when filing beads, and naming the contract surfaces "this is structural, not a CMS bug" without requiring the user to know the docs already. If the doc is later renamed, the runtime error gets updated alongside. **User decision:** OK with this docname reference, or prefer a contract-free wording?

### Sub-question 2 — Per-byte vs whole-edge routing in `EdgeRoutedDifferentlyFailure.byte_key_routing`

The pseudocode includes `byte_key_routing: dict[byte_key, tuple[ref_identity, subj_identity]]` for diagnostic completeness. If the consumer reads N bytes and they're all routed to the same different writer, this dict has N entries all pointing to the same (ref, subj) pair — possibly verbose. Three options:

- **Keep as-is:** N entries, possibly redundant, complete information.
- **Compress to unique (ref, subj) pairs:** `{(ref_id, subj_id): [bks where this routing applies]}`.
- **Omit entirely:** just store the representative ref writer; rely on diff tooling to recover per-byte info.

**Recommendation:** keep as-is (option 1). Diagnostics are read by humans investigating failures, not by automated tooling that needs compact representations. Redundancy is fine; ambiguity is not. **User decision:** OK with verbose dict, or prefer compressed?

### Sub-question 3 — Should `EdgeRoutedDifferentlyFailure` count as fatal (validator hard-fail) or as a finding to report?

`OverriddenInputFailure` (its missing-direction counterpart) is a hard failure that propagates through `ValidationError`. `EdgeRoutedDifferentlyFailure` describes the same structural defect (a clobber); consistency argues for the same disposition. **Recommendation:** hard fail. **User decision:** OK with hard-fail, or prefer warn-only?

---

## Beads filed (if any)

None new from this design pass. The existing P0 blockers stand:
- `rocm-libraries-67us` — `compare_graphs` symmetric direction needed (the bead this classifier implements).
- The capture-pipeline blocker surfaced in `UNROLLED_VALIDATION_ANSWERS.md` (SHADOW `_build_shadow_cms_pair` returns None) blocks empirical re-probing the 16 NGL extras through the unrolled walk; not new from this memo, already filed.

No new defects were surfaced during this design exercise that warrant additional beads beyond what the unrolled-validation plan already tracks.
