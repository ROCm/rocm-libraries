# Plan — teym: Resolve identity-set assertion semantics under iter-copy duplication

**Bead:** `rocm-libraries-teym`
**Status:** in_progress
**Blocks:** `rocm-libraries-1rsy` (C3c)
**Created:** 2026-06-05

---

## §1 Scope

This bead resolves one decision before C3c (`build_dataflow_graph` rewrite consuming `UnrolledCapture`) can be implemented: what assertion correctly expresses the validator invariant at the two sites in `test_dataflow_graph_comparison.py` (lines :154, :214) that currently read `{n.identity for n in g_ref.nodes} == {n.identity for n in g_subj.nodes}`? Under C3c, the same logical instruction appears twice in `nodes` (one per ML iter copy), so the set comprehension silently deduplicates, making the check vacuous as a multiplicity guard. The change is confined to those two test assertions; no production code and no other tests require modification as a direct result of this decision (see §4 for cascading sites that must be co-examined).

---

## §2 Investigation findings

### A — What the two tests actually verify

**Test :154** (`TestCleanComparison::test_redundant_swaits_and_barriers_in_subject_no_failures`):

The test builds a reference graph with one LR, one SWait, and one MFMA, and a subject graph with the same dataflow plus three redundant sync ops (two extra SWaits, one SBarrier). The primary assertion is `compare_graphs(g_ref, g_subj) == []`. The identity-set assertion at :154 is a *defensive guarantee*: it pins that SWait/SBarrier nodes were NOT admitted into `nodes` (the dataflow-node list). Since subject has 3 more sync ops, if they leaked into `nodes`, the subject identity set would be strictly larger — the assertion would fire. The test is NOT asserting that the two graphs have the same node count in general; it is asserting that the sync-op-exclusion filter is working.

**Test :214** (`TestCleanComparison::test_lcc_included_in_identity_set`):

The test builds two identical captures, each with LR + SWait + LCC + MFMA. The identity-set assertion at :214 confirms that LCC nodes (SSubU32 / SCmpEQI32 loop-counter code) ARE admitted into `nodes` — i.e. that `data_flow_instructions` includes them. The test is a positive inclusion pin, not a coverage-completeness check. Both graphs have the same logical instructions, so both sets are equal — the equality is expected to hold exactly.

**Original (pre-C3b) form of both assertions:**

`set(g_ref.nodes.keys()) == set(g_subj.nodes.keys())`

This was equivalent because `nodes` was a `Dict[identity_tuple, GraphNode]`: iterating `.keys()` gave the identity-tuple set. The C3b refactor replaced the dict with an ordered list, so `.keys()` no longer exists and the set-comprehension form was substituted mechanically to preserve the same semantics for the C3b era (no iter copies yet).

### B — The validator invariant under the unrolled stream

Per `UNROLLED_VALIDATION_PLAN.md §2.1`:

> "Multiple GraphNodes with the same `identity` (different iter copies of the same `TaggedInstruction`) coexist; no squashing."

Per `UNROLLED_VALIDATION_PLAN.md §1` (Constraints table):

> "Identity stays iter-blind | Pipelined instructions that move between iters must produce identical identity tuples so set-diff cancels them."

Per `UNROLLED_VALIDATION_PLAN.md §2.3`:

> `ML_iter[0]` and `ML_iter[1]` reuse the underlying `TaggedInstruction` objects from the captured ML body; each iter copy stamps a distinct `unrolled_position`; identity stays the same across copies.

Per `UNROLLED_VALIDATION_PLAN.md §2.2` (byte-key as primary keying basis):

> Identity becomes a label for diagnostics; the lookup basis migrates to byte-key reverse index.

Per `6QIB_DESIGN.md §0.7` and `UNROLLED_VALIDATION_PLAN.md §3.3` (cross-body / cross-iter live-ins resolve naturally via the single `latest_writer` walk):

> Both sides emit edges from their respective most-recent producers. Under iter-blind identity + byte-key `edge_keys`, those edges cancel in set-diff because the byte-keys are equal.

**Conclusion from the design docs:** "same node identities in ref and subj" means **same multiset of identities**. Under the unrolled stream, a logical instruction that appears `k` times (because it is in the ML body materialized `ML_MAT_COUNT = 2` times) must appear exactly `k` times on both sides for the graph structures to be comparable. If ref has 2 iter copies and subj has 1, the graphs are not structurally equivalent — even though the identity SET is the same. A set comparison cannot catch this difference; only a multiset comparison can.

The invariant is: **`Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)`**.

Quoting `UNROLLED_VALIDATION_PLAN.md §2.1`: "no squashing" — the invariant is explicitly multiplicity-preserving. The identity set captures unique logical instructions; the multiset captures how many times each appears (i.e., how many iter copies materialized). Both must agree between ref and subj for the graphs to be structurally comparable.

### C — Comparison of the three options

**Option (a) — `len(g_ref.nodes) == len(g_subj.nodes) and {ids_ref} == {ids_subj}`:**

Length equality catches "2 copies vs 1 copy" of any single identity, but cannot distinguish "2 copies of X vs 1 copy of X + 1 copy of Y" (different unique identities but same total count). This is a false negative: if ref has `[X, X]` and subj has `[X, Y]`, `len` matches, `set` matches on X (set is `{X}` vs `{X, Y}` — wait, that would catch it). Actually the combination is stronger than it appears on first reading: length equality + set equality together DO imply multiset equality WHEN the set has no internal multiplicity ambiguity. But they do NOT imply multiset equality in general: consider ref = `[X, X, Y]` and subj = `[X, Y, Y]` — both have length 3 and the same set `{X, Y}`, but different multiplicity distributions. Counter would catch this; the (a) combination would not. This is a real gap for the C3c world where `ML_MAT_COUNT = 2` but future fixture changes might add non-uniform multiplicities.

**Option (b) — `Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)`:**

Catches every difference in identity multiplicity by construction. Counter equality implies both set equality (same keys) and length equality (same total), so it is strictly stronger than (a). It also handles non-uniform multiplicity (where different identities appear different numbers of times) without any additional conditions. This is the correct principled answer: it expresses exactly "both graphs materialized the same logical instructions the same number of times each."

**Option (c) — "each identity appears exactly once OR exactly twice":**

This embeds the `ML_MAT_COUNT = 2` assumption as a positive constraint on multiplicity within a single graph. It cannot detect "ref has 2 copies of X, subj has 1" because it checks each graph independently. It is not a cross-graph comparison at all — it is a self-consistency check on multiplicity distribution within each graph. It is brittle (breaks if `ML_MAT_COUNT` changes) and does not answer the actual cross-graph question. Rejected.

**Decision: Option (b)** is the principled answer. It is the only option that directly expresses the multiset-equality invariant that the unrolled stream demands.

### D — Other test-suite sites with identity-set comparisons

The full grep across `Tensile/Tests/` and `Tensile/Components/` for `n.identity for n in` reveals:

| File | Line(s) | Context |
|---|---|---|
| `test_dataflow_graph_comparison.py` | :154, :214 | The two primary sites. Both are C3b-era set assertions that become vacuous under C3c. |
| `test_oplb_register_naming_minimal.py` | :586, :634 | Both are set comprehensions extracting identities from a SINGLE graph (for inspection of rendered content), NOT cross-graph equality comparisons. These are examining which identities exist, not comparing two graphs. They are unaffected by iter-copy duplication because a single graph's set comprehension is being used to filter for LR nodes and inspect renders — not for equality comparison. |
| `test_dataflow_graph_builder.py` | :519-523 | A LIST comprehension `[n.identity for n in g.nodes if n.category == "LRA0"]`, then `assert len(ids_a) == 1` and `assert ids_a[0] == ids_b[0]`. This explicitly uses a list and asserts length = 1 per side. Under C3c this would be `len(ids_a) == ML_MAT_COUNT` if LRA0 nodes are in the ML body. This is a correctness pin; the implementer of C3c must revisit it. **This site must be added to C3c's re-fixture scope but does NOT need to change under this bead** — it already uses list semantics. |

The two `.md` files containing set comprehensions (`EMISSION_ORDINAL_DESIGN.md`, `EXFW_MFMA_DIVERGENCE_INVESTIGATION.md`) reference the old `graph.nodes.values()` API (pre-C3b); they are investigation artifacts, not live code, and do not need updating.

**Conclusion for §D:** Only `test_dataflow_graph_comparison.py:154` and `:214` require changes under this bead. `test_dataflow_graph_builder.py:519-523` is a related site the C3c implementer must revisit.

### E — Production code identity-set assertions

The production code (`CMSValidator.py`) does NOT assert identity-set equality anywhere. The entry gate in `compare_graphs` uses **per-(rocisa-derived-InstructionCategory) count comparison** via `Counter`, not identity-set equality:

```python
ref_counts = _data_flow_category_counts(reference)  # Counter by InstructionCategory.value
subj_counts = _data_flow_category_counts(subject)
if ref_counts != subj_counts:
    raise CaptureConsistencyError(...)
```

This is a count comparison at the category level (LR, MFMA, GR, etc.), not at the identity level. It was introduced by `rocm-libraries-oplb` to replace a prior identity-tuple set-equality check that over-fired on register-naming differences.

Under C3c, this production-code gate must also be updated: with `ML_MAT_COUNT = 2` ML iter copies, the count of LR nodes and MFMA nodes will double from the ML body. The production-code Counter gate in `compare_graphs` will still PASS (both sides have the same doubled counts, because both ref and subj consume the same `UnrolledCapture`), so it does not produce false positives — but its semantics shift. Specifically:

- Pre-C3c: counts reflect the single-body instruction set.
- Post-C3c: counts reflect the unrolled stream (ML body instructions counted twice).

This is FINE because the gate's purpose is "both captures emitted N LR / M MFMA / etc." — doubling both sides preserves the equality. No change to the production-code gate is required under this bead.

**However:** `diagnose_missing_edge` at line :3789 does a linear scan `next((n for n in reversed(subj_graph.nodes) if n.identity == p_id), None)`. Under C3c, this will find the MOST-RECENT iter copy of the identity (due to `reversed`). This is correct behavior per `UNROLLED_VALIDATION_PLAN.md §4.3` which specifies that the byte-key reverse-index replaces this lookup entirely in C3f. The identity-based lookup is a C3b-era stopgap that C3f migrates away from. **No change needed under this bead** — the C3f bead owns this migration.

### F — Test intent and original assertion semantics

The original assertion `set(g_ref.nodes.keys()) == set(g_subj.nodes.keys())` (pre-C3b) was using the identity-keyed dict's key-set. Because the dict cannot have duplicate keys (identity uniqueness was enforced by the dict semantics), the assertion was equivalent to set equality. There was no multiplicity to track — the dict guaranteed each identity appeared at most once. The C3b refactor correctly translated to `{n.identity for n in g_ref.nodes}` which preserves the same semantics for C3b (where identities are still unique, just stored in a list).

The C3c transition changes the invariant: the `UnrolledCapture` materializer (C3a) stamps two distinct GraphNode objects for each ML instruction (distinct `unrolled_position`, same `identity`). The ordered list in `DataflowGraph.nodes` now contains duplicates by identity. The set comprehension silently deduplicates, and the cross-graph equality becomes vacuous as a multiplicity check.

The intent of both assertions is: "the two graphs contain the same logical instruction population." Under the unrolled stream, "same population" means same multiset — Counter equality.

---

## §3 Decision — picked option with rationale

**Pick option (b): `Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)`.**

Rationale: Counter equality is the direct mathematical expression of the validator invariant. It captures both "same unique identities" (same set of keys) and "same multiplicities" (same number of iter copies per identity). Options (a) and (c) each fail to capture one of these dimensions; option (b) captures both. It is also forward-compatible with any future change to `ML_MAT_COUNT` — if the count changes, the Counter test automatically reflects the new counts on both sides without embedding a specific numeric constant.

**Concrete pattern for both sites:**

```python
from collections import Counter
assert Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)
```

The `Counter` import is already present in `CMSValidator.py` (production code) but NOT in `test_dataflow_graph_comparison.py`. The implementation step must add `from collections import Counter` to the test file's imports.

**What "failure mode" this catches that the current set assertion misses:**

If C3c's `build_dataflow_graph` accidentally materializes only 1 ML iter copy instead of 2 (e.g. an off-by-one in `UnrolledCapture`), the Counter assertion would fire: ref has `Counter({ML_instr_identity: 2, ...})` but subj has `Counter({ML_instr_identity: 1, ...})`. The set assertion would pass silently.

**The tests at :154 and :214 are both C3b-correct for C3b but become the wrong assertion under C3c.** Changing them now (in this bead, before C3c) is the correct sequencing — the plan for this bead is to update them to Counter form, so C3c starts with the right assertion already in place.

---

## §4 Sites to update

### Primary sites (under this bead)

| File | Line | Current assertion | New assertion |
|---|---|---|---|
| `Tensile/Tests/unit/test_dataflow_graph_comparison.py` | :154 | `assert {n.identity for n in g_ref.nodes} == {n.identity for n in g_subj.nodes}` | `assert Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)` |
| `Tensile/Tests/unit/test_dataflow_graph_comparison.py` | :214 | `assert {n.identity for n in g_ref.nodes} == {n.identity for n in g_subj.nodes}` | `assert Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)` |

Also add `from collections import Counter` to the import block at the top of the test file (currently absent).

### Secondary sites (to note for C3c, NOT changed under this bead)

| File | Line(s) | Note |
|---|---|---|
| `Tensile/Tests/unit/test_dataflow_graph_builder.py` | :519-523 | Uses list semantics already; checks `len == 1`. Under C3c this must become `len == ML_MAT_COUNT` for ML-body nodes. C3c's re-fixture scope owns this. |
| `Tensile/Components/CMSValidator.py` | `compare_graphs` entry (`:3708-3714`) | Production Counter gate at InstructionCategory level. No change needed — doubles on both sides equally. Noted for C3c awareness. |
| `Tensile/Components/CMSValidator.py` | `diagnose_missing_edge` (`:3789`) | Identity-based linear scan. Migrated to byte-key reverse index in C3f. No change under this bead. |

### Sites explicitly not changed

- `test_oplb_register_naming_minimal.py:586, :634` — single-graph set comprehensions for content inspection, not cross-graph equality. Unaffected by iter-copy duplication.
- Production `CMSValidator.py` — no identity-set equality assertion exists in production code; the entry gate is already Counter-based at the category level.

---

## §5 Step-by-step implementation order

1. Add `from collections import Counter` to the top-level import block in `Tensile/Tests/unit/test_dataflow_graph_comparison.py`. The file already imports `pytest` and several `Tensile` symbols; insert Counter alongside or with the standard-library imports.

2. Replace line :154 (inside `test_redundant_swaits_and_barriers_in_subject_no_failures`):
   - Old: `assert {n.identity for n in g_ref.nodes} == {n.identity for n in g_subj.nodes}`
   - New: `assert Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)`

3. Replace line :214 (inside `test_lcc_included_in_identity_set`):
   - Old: `assert {n.identity for n in g_ref.nodes} == {n.identity for n in g_subj.nodes}`
   - New: `assert Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)`

4. Update the docstring comment at line :151-153 (immediately before :154) from "nodes_by_identity" to reflect the new semantics — the comment says "the SWait/SBarrier nodes must not have leaked into `nodes_by_identity`"; update to "the SWait/SBarrier nodes must not have leaked into the dataflow `nodes` list." This is a comment-only change to remove a stale reference to the removed dict.

5. Annotate the C3c bead (`rocm-libraries-1rsy`) to surface the `test_dataflow_graph_builder.py:519-523` cross-reference explicitly. The bead description currently does not mention this site; the C3c implementer must update the three `assert len(ids_X) == 1` assertions to `assert len(ids_X) == ML_MAT_COUNT` for any ML-body nodes, and `assert ids_a[0] == ids_b[0]` (cross-graph identity pin) needs re-examination under duplicated identities. Concrete action for the implementer: run `br amend rocm-libraries-1rsy` (or equivalent) and append a "Re-fixture scope" note to the C3c bead description naming `test_dataflow_graph_builder.py:519-523` and the `len == ML_MAT_COUNT` update.

6. No production code changes required.

7. Run the unit test suite for the affected file:
   ```bash
   pytest Tensile/Tests/unit/test_dataflow_graph_comparison.py -v \
     --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py
   ```
   All tests must pass. The Counter form is backward-compatible with C3b (where each identity appears exactly once; Counter equality reduces to set equality when all multiplicities are 1).

---

## §6 Validation

**The Counter assertion is backward-compatible with C3b:** When all identities are unique (C3b state, no iter copies), `Counter(n.identity for n in g_ref.nodes) == Counter(n.identity for n in g_subj.nodes)` is exactly equivalent to `{n.identity for n in g_ref.nodes} == {n.identity for n in g_subj.nodes}` (because each identity has count 1 in both Counters). No test will break.

**The Counter assertion is forward-correct for C3c:** When ML iter copies exist, Counter equality requires both sides to have the same number of copies of each identity. A mismatch in materialization count (e.g. 1 copy vs 2) surfaces immediately rather than being swallowed by set deduplication.

**Cross-validation:** After implementing, verify both changed tests still pass by running the suite per step 6. Also run the broader dataflow graph test suite (`test_dataflow_graph_builder.py`, `test_dataflow_graph_barriers.py`) to confirm no regressions from the `Counter` import.

---

## §7 Risks / open questions

**Risk 1 — Downstream confusion about what Counter failure means:**

A Counter mismatch at :154 or :214 under C3c would mean "one side materialized a different number of ML iter copies." The error message from `assert Counter(...) == Counter(...)` is bare Python — it prints the two Counter objects. This is readable but terse. Consider whether a better failure message is worth adding (e.g. `assert ... == ..., f"identity multiplicity mismatch: {Counter(ref) - Counter(subj)}"`). This is low priority; the Counter equality is self-documenting.

**Risk 2 — test_dataflow_graph_builder.py:519-523 is NOT fixed here:**

The list-based identity assertions in that file will need revisiting in C3c. §5 step 5 directs the implementer to annotate the C3c bead (`rocm-libraries-1rsy`) with an explicit re-fixture note naming this site; that annotation is part of this bead's implementation deliverables.

**Risk 3 — SWait/SBarrier exclusion semantics under C3c:**

Test :154 specifically verifies that sync ops do NOT leak into `nodes`. Under C3c, the `UnrolledCapture` materializer produces `TaggedInstruction` objects from the body. If SWait/SBarrier instructions are present in the ML capture, they would appear twice (two iter copies each) in the unrolled stream. The exclusion filter (`data_flow_instructions`) already gates them out of `nodes_list` in Phase 1 of `build_dataflow_graph`. This exclusion must remain active in C3c's rewrite of Phase 1. The Counter assertion at :154 correctly catches any leak: if two SWait copies both leaked into `nodes`, the Counter for that identity would be 2 on the subject side and 0 on the ref side (since ref has no SWait identity in its `nodes`). No special action required beyond verifying the exclusion filter is preserved in C3c's Phase 1 rewrite.

**Open question:** The current form of :154's setup has ref with 1 LR + 1 MFMA (in `nodes`) and subj also with 1 LR + 1 MFMA (in `nodes`, despite 3 extra sync ops). Under C3c, both would have 2 LR + 2 MFMA (ML iter copies). The Counter assertion would expect `Counter({lr_identity: 2, mfma_identity: 2})` on both sides. This is correct. The test logic doesn't need to change beyond the assertion form.

---

## §8 New beads to file

None. This bead is a prerequisite for C3c (`rocm-libraries-1rsy`) and resolves a decision point that was explicitly held open. The investigation found no new prerequisite work:

- The production code gate in `compare_graphs` does not need a parallel change (§E).
- `test_dataflow_graph_builder.py:519-523` is a C3c-era concern, not a new bead — §5 step 5 mandates that the implementer annotates `rocm-libraries-1rsy` with an explicit re-fixture scope note for this site. The annotation action is part of teym's own implementation deliverables, not deferred to C3c's planner.
- The `diagnose_missing_edge` identity-based scan is already owned by C3f (`rocm-libraries-i190`).

Bead `rocm-libraries-teym` stays `in_progress` until the implementation step closes it.
