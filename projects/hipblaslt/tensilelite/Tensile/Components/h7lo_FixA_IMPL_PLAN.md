# Fix A — Correct consumer-instance resolution in `diagnose_extra_edge` (and `diagnose_missing_edge`)

Bead: `rocm-libraries-h7lo` (Fix A only).

---

## §1 Scope

Fix A corrects the **failure messages** emitted by `diagnose_extra_edge` (and, symmetrically, `diagnose_missing_edge`) when a multi-body identity collision causes the bare-identity `next(...)` lookup to return the wrong body-instance of the ref/subj consumer.

Fix A does **not** cancel the 16 `EdgeRoutedDifferentlyFailure`s. Those failures reflect a genuine per-byte routing divergence (NGL pack-CVT vs NGL ds_read as last writer before the NLL consumer); that is Fix B (`rocm-libraries-uvrl`). Fix A's job is to make the failure messages name the correct ref-body instance — "routes through NGL `ds_read` @ idx=X (NLL body)" instead of the current "routes through `PackA0[9]` @ idx=-1 (PRO body)".

No feature flags. No backwards-compat shims. No LoC numbers.

---

## §2 Investigation findings

### A — `diagnose_extra_edge` consumer lookup (the primary bug)

`CMSValidator.py:4362–4364`:

```python
ref_cons_node = next(
    (n for n in ref_graph.nodes if n.identity == cons.identity), None
)
```

`GraphNode.identity` is `(canonical_render, source_module_id, emission_ordinal)` — **body-blind by construction** (line 919). For a canonical MFMA that appears in all 5 bodies (PRO, ML-1, ML, NGL, NLL), this identity recurs 5 times in `ref_graph.nodes`. `next()` returns the first match in node-list order, which is the **PRO copy** (lowest position). The correct instance is the NLL copy, matching the subj consumer's body.

`cons_pos` is then set to PRO's `unrolled_position` (≈26 in the v14 example), so the closest-prior-writer search scans ref's `byte_key_writers[('v',14)]` for writers before pos=26 — finding the prologue `PackA0` CVT (pos=16) instead of the NGL `ds_read` (pos=597). This makes the failure say "routes through PackA0[9] @ idx=-1 (PRO)" when the correct ref routing is `LRA3 ds_read` in NGL at pos=597.

### B — Correct resolution key

The subj consumer (`cons`) carries `body_label` and `iter_index` (line 924, 928 of `GraphNode`). CMS and SHADOW share the same body structure — body assignment is a codegen-emission attribute, not a scheduling one. The same canonical MFMA is emitted in NLL on **both** sides; only its position within NLL differs between the two schedules.

Resolution key: `(identity, body_label, iter_index)`.

- `identity` selects the canonical instruction.
- `body_label` selects the body (PRO / ML-1 / ML / NGL / NLL).
- `iter_index` distinguishes ML iter copies (0 vs 1) when `ML_MAT_COUNT > 1`.

This triple uniquely identifies one `GraphNode` in `ref_graph.nodes`. No other field is needed.

**Why identity alone does not suffice (emission_ordinal scoping confirmed)**: `emission_ordinal` is assigned by `assign_emission_ordinals` (`ScheduleCapture.py:791`), which resets its counter per body — each `LoopBodyCaptureBuilder` owns its own counter dict. Therefore the same canonical render-string gets `ordinal=0` in PRO, `ordinal=0` in ML-1, `ordinal=0` in NGL, and `ordinal=0` in NLL. All five body copies of the same MFMA share the same `identity` tuple. Adding `body_label` is required to break the five-way collision. `iter_index` further disambiguates the two ML copies (iter_index=0 for ML-1/PRO/NGL/NLL; iter_index=1 for ML). Confirmed by `ScheduleCapture.py:436-514` and `assign_emission_ordinals` implementation.

Concretely, the fix replaces:

```python
ref_cons_node = next(
    (n for n in ref_graph.nodes if n.identity == cons.identity), None
)
```

with:

```python
ref_cons_node = next(
    (n for n in ref_graph.nodes
     if n.identity == cons.identity
     and n.body_label == cons.body_label
     and n.iter_index == cons.iter_index),
    None
)
```

### C — `diagnose_missing_edge` has the symmetric bug

`CMSValidator.py:3946`:

```python
c_node = next((n for n in subj_graph.nodes if n.identity == c_id), None)
```

Here the ref edge's consumer is looked up in **subj_graph** by bare identity. `c_node.unrolled_position` (line 3962) is then used as `cons_unrolled` to search subj's `byte_key_writers` for the closest-prior writer. If the consumer identity recurs across bodies in subj, this picks the PRO copy and searches from pos=26, finding the wrong producer in subj — the same class of misattribution in the opposite diagnostic direction.

The reference's consumer body/iter is already available via `ref_edge.consumer.body_label` and `ref_edge.consumer.iter_index`. The fix is symmetric: add the same two predicates.

Fix A should fix **both** functions for consistency.

### D — No shared helper exists

There is no `graph.find_node(identity, body_label, iter_index)` helper. Every call site uses an inline `next(...)`. There are exactly **two** buggy call sites (one in `diagnose_extra_edge`, one in `diagnose_missing_edge`). A shared inline-capable helper is not needed — the fix is a three-predicate inline at both sites.

### E — Test impact

All 8 tests in `test_diagnose_extra_edge.py` plus the 2 in `TestDiagnoseExtraEdgeSCCAndPartialMatch` use **single-body** fixtures built by `_build_graph_lr_swait_mfma` (which calls `make_capture(BODY_LABEL_ML, ...)` wrapping into a `FourPartCapture`). In a single-body fixture, bare identity and `(identity, body_label, iter_index)` return the same node — no test behavior changes.

The fix does not alter what nodes qualify (still exactly one per lookup in a correct same-instruction-set pair); it only tightens the predicate to eliminate false PRO matches in multi-body fixtures.

### F — Validation approach

Fix A changes failure **messages**, not failure **counts**. Validation is two-pronged:

1. **Dumper regression**: re-run the existing `_h7lo_probe{2..6}.py` infrastructure (or write a new thin script against the live graphs) and inspect `compare_graphs_failures.txt`. After Fix A, each `EdgeRoutedDifferentlyFailure` message should cite the NLL-body ref instance (pos≈636, citing the NGL `ds_read` at pos≈597 as the ref closest-prior writer), not the PRO instance (pos=26).

2. **New unit test**: add a multi-body identity-collision test to `test_diagnose_extra_edge.py` that constructs a `FourPartCapture` with the same canonical MFMA in multiple bodies and asserts that `diagnose_extra_edge` resolves the NLL-body ref consumer, not the PRO-body one.

---

## §3 Design

**Corrected resolution key**: `(identity, body_label, iter_index)` — three-predicate inline `next(...)`.

**Both functions fixed**: `diagnose_extra_edge` (line 4362) and `diagnose_missing_edge` (line 3946). Same bug, same fix, same predicate shape.

**No shared helper**: two call sites, both inline. The three-predicate pattern is self-documenting; a helper would add indirection without reducing repetition at this scope.

**Body-label source**: for `diagnose_extra_edge`, use `cons.body_label` and `cons.iter_index` (subj consumer carries the correct body). For `diagnose_missing_edge`, use `ref_edge.consumer.body_label` and `ref_edge.consumer.iter_index` (the ref edge's consumer is the source of truth for which body to look up in subj).

---

## §4 Step-by-step implementation order

1. **Fix `diagnose_extra_edge`** (`CMSValidator.py:4362–4364`): replace the bare-identity `next(...)` with the three-predicate form using `cons.body_label` and `cons.iter_index`. Update the adjacent comment block (lines 4349–4361) to document that body+iter are now part of the resolution key.

2. **Fix `diagnose_missing_edge`** (`CMSValidator.py:3946`): replace the bare-identity `next(...)` with the three-predicate form using `ref_edge.consumer.body_label` and `ref_edge.consumer.iter_index`. Update the adjacent comment.

3. **Add unit test** in `test_diagnose_extra_edge.py`: new class `TestDiagnoseExtraEdgeMultiBodyIdentityCollision`. Construct a `FourPartCapture` where the same canonical consumer MFMA appears in both PRO and NLL bodies. Invoke `diagnose_extra_edge` with a subj edge whose consumer is the NLL instance. Assert the returned `EdgeRoutedDifferentlyFailure` cites the NLL ref consumer position (not PRO). Assert `ref_cons_node` (via the failure message or by inspecting the failure's `iter_delta`) reflects the NLL body.

4. **Run unit tests**: `tox -e unit -- test_diagnose_extra_edge.py --ignore=<slow file>`. All existing tests must pass; new test must pass.

5. **Dumper message check**: run `_h7lo_probe6.py` (or a new thin wrapper that calls `compare_graphs` on the live v14 fixture captures) and inspect failure messages. Confirm failures now reference NLL-body positions (≈636) not PRO (≈26).

---

## §5 Validation

### Message-correctness check via dumper

Re-run the live-kernel dumper against the BPG#11 TF32 4x4 TN canonical fixture (same as `_h7lo_probe{2..6}.py`). Before Fix A, failure 0 reads:

> subj consumer PackA3[11] @ idx=43 reads from PackA3[16] @ idx=46 at ('v',14), but reference routes through PackA0[9] @ idx=-1 (PRO body)

After Fix A, the same failure should read something like:

> subj consumer PackA3[11] @ idx=43 reads from PackA3[16] @ idx=46 at ('v',14), but reference routes through LRA3[...] @ idx=<NGL pos> (NLL body)

The failure count stays at 16. Only the cited ref producer changes.

### New unit test: multi-body identity collision

The test asserts two things:

- `diagnose_extra_edge` does **not** cite a PRO-body consumer position when the subj consumer is in NLL.
- `diagnose_extra_edge` resolves the ref consumer to the NLL copy (matching body_label and iter_index).

Construction strategy: build a `FourPartCapture` where `nll` contains the consumer MFMA and `main_loop_prev` (PRO stand-in) also contains a copy with the same canonical render. Use `make_capture(BODY_LABEL_NLL, ...)` for the NLL body and a filler for PRO. The NLL consumer's `body_label` must be `BODY_LABEL_NLL` and the PRO filler's consumer `body_label` must be `BODY_LABEL_ML_PREV`. Confirm `ref_cons_node.body_label == BODY_LABEL_NLL` by inspecting the failure's `iter_delta` or by introducing a thin assertion helper.

---

## §6 Test impact

All existing tests in `test_diagnose_extra_edge.py` use single-body fixtures (all nodes have the same `body_label`). In single-body fixtures the three-predicate form returns the same node as the bare-identity form — no behavior change, no test breakage.

`diagnose_missing_edge` tests (in `test_CMSValidator.py` or other unit files) similarly use single-body fixtures. Same conclusion.

The new test in §4 step 3 is the **only** test that exercises the multi-body collision path.

---

## §7 Risks

**R1 — body_label/iter_index mismatch between ref and subj**: the assumption that a canonical MFMA lands in the same body on both sides is stated explicitly in the design contract (body assignment is codegen-level, not schedule-level). If a future kernel violates this (e.g., a CMS that deliberately moves an instruction to a different body), Fix A would fail to find the ref consumer and fall back to `None` — which is already handled (falls back to `cons.unrolled_position`). The fallback is the same behavior as today for the non-collision case; it does not introduce a regression.

**R2 — iter_index semantics for ML**: `iter_index` is 0 for PRO / ML-1 / NGL / NLL and 1 for the second ML body copy (when `ML_MAT_COUNT > 1`). The fix correctly uses this field as a discriminator; the PRO and NLL copies both have `iter_index=0`, so they remain disambiguated solely by `body_label`. No risk of cross-iter confusion.

**R3 — `diagnose_missing_edge` source of body info**: `ref_edge.consumer.body_label` is populated at edge-formation time from the graph-builder (same pipeline as `diagnose_extra_edge`'s `cons.body_label`). No new plumbing is needed.

**R4 — SCC-branch `orig_producer_node` lookup in `diagnose_missing_edge` (line 4129) is out of scope but is another bare-identity lookup**:
`CMSValidator.py:4129` performs `next((n for n in subj_graph.nodes if n.identity == p_id), None)` to resolve the *original producer* in the SCC-clobber branch. This is the same class of body-blind lookup. Fix A does NOT fix this site — it is a producer resolution (not consumer), and `p_id` comes from `ref_edge.producer.identity`. If the same instruction appears in multiple bodies and the SCC clobber path is reached, it would also pick the wrong body copy. This case is dormant for the h7lo fixture (no SCC edges in the failing 16), but it is a latent defect. File as a follow-on bead; do not expand Fix A's scope to include it.

---

## §8 New beads

**h7lo-scc-probe (follow-on, non-blocking)**: `diagnose_missing_edge` SCC-branch at `CMSValidator.py:4129` has a third bare-identity `orig_producer_node` lookup (`next((n for n in subj_graph.nodes if n.identity == p_id), None)`) that is also body-blind. It is out of scope for Fix A (producer resolution, not consumer, and not exercised by the 16 h7lo failures), but is the same defect class. File a separate bead to fix it using the same three-predicate form with `ref_edge.producer.body_label` and `ref_edge.producer.iter_index`.

Fix B (`rocm-libraries-uvrl`) is the dependent bead for the substantive routing divergence and remains unchanged.
