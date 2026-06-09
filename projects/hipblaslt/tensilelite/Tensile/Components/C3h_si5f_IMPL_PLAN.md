# C3h (si5f) Implementation Plan
# n7og xfail removal + cross-iter/cross-body unit tests

**Status:** plan verified (2026-06-09)
**Bead:** rocm-libraries-si5f
**Branch:** users/alvasile/mxfp4_fast_ref_min (worktree: validator_long_term_plans)

---

## 1. Scope clarification

### 1.1 n7og xfail removal — already done in C3d

Verified by reading `Tensile/Tests/unit/test_n7og_edge_keys_multifixture.py`:
- `_FIXTURES` at line 217-221 uses plain `pytest.param(...)` with NO `marks=` arguments.
- The only `xfail` occurrences in the file are in docstring text (lines 82, 84, 289, 295), not in executable code.
- Commit `5b187061007c` (C3d / xxj4) removed the live markers inline.

**C3h does NOT re-do this step.** Scope: add new unit tests and provide the "validator state GREEN" lock-in.

### 1.2 What C3h adds

1. Cross-iter live-in unit tests (synthetic `FourPartCapture`, ML_PREV writes → ML reads)
2. Cross-body live-in unit tests (synthetic `FourPartCapture`, PRO writes → ML reads)
3. `EdgeRoutedDifferentlyFailure` classifier additional coverage (beyond the 8 tests already in C3e)
4. Symmetric edge_key equality lock-in for all 3 n7og fixtures (parametrized `compare_graphs` round-trip, both directions 0 mismatches)

---

## 2. Investigations

### A. xfail status — confirmed removed

See §1.1. No action.

### B. Cross-iter live-in test design

**Infrastructure:** `FourPartCapture` accepts `main_loop_prev` (ML[0], `BODY_LABEL_ML_PREV`,
`iter_index=0`) and `main_loop` (ML[1], `BODY_LABEL_ML`, `iter_index=1`). The
`UnrolledCapture.from_four_part_capture` materializer puts ML_PREV before ML in the
unrolled stream.

**Setup:** Use `make_lr` / `make_mfma` from `dataflow_fixtures.py`. Build a
`LoopBodyCapture` for ML_PREV where an LR writes vgpr range `[8..12)`. Build a separate
`LoopBodyCapture` for ML where an MFMA reads vgpr range `[8..12)`. Wrap both into a
`FourPartCapture` via `_wrap(ml_cap, ml_prev=cap_prev)` from `test_dataflow_graph_comparison.py`
(accepts explicit `ml_prev=` keyword) with `main_loop_prev={0: cap_prev}` and `main_loop={0: cap_ml}`.

NOTE: `test_dataflow_graph_builder.py::test_cross_iter_edge_carries_diagnostic_annotations`
already pins `producer_iter_index`, `consumer_iter_index`, and body labels for this exact
setup. Do NOT duplicate those field-level assertions. The new test's focus is the
`compare_graphs(g, g) == []` round-trip (confirming cross-iter edges cancel in set-diff)
and the negative `OrderInvertedFailure` assertion.

Call `build_dataflow_graph(fpc)`. The LR in ML_PREV and the MFMA in ML share no body —
the single `latest_writer` walk across the unrolled stream resolves the LR as the
producer for the MFMA's byte-keys because ML_PREV appears before ML in the stream.

**Assertions:**
- One edge exists with `edge_kind == "raw_intrawave"` spanning from ML_PREV's LR to ML's MFMA.
- `edge.producer_iter_index == 0` (ML_PREV's iter_index per UnrolledIterRecord).
- `edge.consumer_iter_index == 1` (ML's iter_index per UnrolledIterRecord).
- `edge.producer_body_label == BODY_LABEL_ML_PREV`.
- `edge.consumer_body_label == BODY_LABEL_ML`.
- `edge.producer_write_byte_key` and `edge.consumer_read_byte_key` cover the
  expected vgpr bytes.

**Negative assertion:** No `OrderInvertedFailure` in `compare_graphs` output when a
matching CMS graph is built from the same capture — the cross-iter edge appears in BOTH
graphs with the same `edge_keys()` tuple, so set-diff cancels it.

### C. Cross-body live-in test design

**Setup:** Use `_wrap_with_pro(prologue_cap, ml_cap)` from `test_dataflow_graph_hdem.py`,
which already constructs a `FourPartCapture` with a real PRO body and filler ML-1/NGL/NLL.
Build a PRO capture where an LR writes vgpr range `[16..20)`. Build an ML capture where
an MFMA reads vgpr range `[16..20)`.

NOTE: `_wrap_with_pro` is the correct helper to import or replicate — it was already
added for the hdem cross-body tests. The plan's earlier reference to constructing "a new
test helper" was wrong; this helper already exists.

Call `build_dataflow_graph(fpc)`. PRO appears first in the unrolled stream
(`records[0].body_label == BODY_LABEL_PROLOGUE`). The MFMA in ML_PREV or ML will have
the LR in PRO as its `latest_writer` for those byte-keys.

**Assertions:**
- One edge with `producer_body_label == BODY_LABEL_PROLOGUE` and
  `consumer_body_label` in `{BODY_LABEL_ML_PREV, BODY_LABEL_ML}`.
- `producer_iter_index == 0` (PRO is non-ML; iter_index=0 per UnrolledIterRecord).
- `edge.producer_write_byte_key` covers the expected vgpr bytes.

### D. EdgeRoutedDifferentlyFailure additional classifier coverage

`test_diagnose_extra_edge.py` (C3e) already has 8 tests covering:
- Phase 0 `CaptureConsistencyError` (1 test in class)
- Phase 1 `EdgeRoutedDifferentlyFailure` + field population (2 tests in class)
- Phase 1 `UnexplainedExtraEdgeError` (1 test)
- `compare_graphs` wiring via mock (1 test)
- `compare_graphs` identical graphs returns [] (1 test)
- `format()` required phrases (1 test)
- `format()` `ref_producer=None` path (1 test)

Missing coverage the C3e tests don't have:
- **SCC byte-key path:** construct a subj edge where `producer_write_byte_key = (("s", "scc"),)` and verify `diagnose_extra_edge` classifies it as `EdgeRoutedDifferentlyFailure` (SCC clobbers subsume into the byte-key model per plan §4.1).
- **Multi-byte-key partial match:** subj edge spans 4 bytes; ref has a writer for 2 of them but not the other 2. Verify `CaptureConsistencyError` is raised (Phase 0 fires on the first missing byte_key).

These two tests go into a new class `TestDiagnoseExtraEdgeSCCAndPartialMatch` in the same
file or a separate `test_diagnose_extra_edge_extended.py`.

### E. Symmetric edge_key equality lock-in for all 3 n7og fixtures

**Design:** A parametrized test in a new file
`test_n7og_compare_graphs_symmetric.py` (or appended to
`test_n7og_edge_keys_multifixture.py`). Each fixture runs through the full
`compare_graphs` call in BOTH directions:

```
failures_fwd = compare_graphs(ref=cms_graph, subj=default_graph)
failures_rev = compare_graphs(ref=default_graph, subj=cms_graph)
assert failures_fwd == []
assert failures_rev == []
```

All three fixtures (`bpg11-tf32-4x4-tn`, `oplb-tf32-6x8-tn`, `bf16-256x256x64-tn`) must
pass. This is the definitive "validator GREEN for principled reasons" pin for the n7og
work: if the edge_keys are truly byte-key symmetric, both directions produce zero
failures — not just the set-diff count.

The helper `_build_shadow_cms_pair` already exists in the multifixture file;
re-export it or import directly.

### F. "Validator GREEN" scope clarification

The bead acceptance criterion "validator state is GREEN for principled reasons" applies to:

1. The three n7og fixtures pass `compare_graphs` in both directions with 0 failures.
2. The new cross-iter and cross-body unit tests pass.
3. The new `EdgeRoutedDifferentlyFailure` coverage tests pass.

It does **not** mean the global suite is fully green. Current state:

- `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_validates_clean_with_carveout_engaged`
  and `::test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures` — these
  are **(b)-class** tests pinning the deleted exemption's silencing behavior. They have
  **no `@pytest.mark.xfail` annotation** — they are hard FAILED in the suite output,
  counted among the 20. Slated for re-fixture in bead `rocm-libraries-5ryl`. C3h does
  not touch them.
- `test_approach_a_non_cms_reference.py` Cycle 2 test — has its own `xfail(strict=True)`
  for `rocm-libraries-nyb5`; pre-existing, not C3-chain-introduced. C3h does not touch it.
- Any remaining failures are pre-existing (u6nn class) or C4-scope (b-class re-fixture).

None of the 20 current failures are addressable by C3h.

---

## 3. New test files

### 3.1 `test_c3h_cross_iter_live_in.py`

New file. Two test classes:

**`TestCrossIterLiveIn`:** ML_PREV writes → ML reads. Constructs a synthetic
`FourPartCapture` using `make_lr` / `make_mfma` / `make_capture` from `dataflow_fixtures`.
Verifies `producer_iter_index==0`, `consumer_iter_index==1`, body labels correct, byte
keys non-empty. Builds a CMS-side graph with the same instructions and asserts
`compare_graphs` returns [].

**`TestCrossBodyLiveIn`:** PRO writes → ML reads. Uses `FourPartCapture.prologue` to
supply a PRO capture. Verifies `producer_body_label==BODY_LABEL_PROLOGUE`,
`consumer_body_label` in ML or ML_PREV, `producer_iter_index==0`.

No real-kernel builds required — all synthetic.

### 3.2 `test_diagnose_extra_edge_extended.py` (or append to existing)

**`TestDiagnoseExtraEdgeSCCAndPartialMatch`:** Two tests:
- SCC byte-key: construct an edge with SCC byte-key; verify `EdgeRoutedDifferentlyFailure`.
- Partial byte-key miss: construct an edge spanning bytes where ref is missing half; verify
  `CaptureConsistencyError` on Phase 0.

Prefer appending to `test_diagnose_extra_edge.py` rather than creating a new file, since
the helpers are already there.

### 3.3 `test_n7og_compare_graphs_symmetric.py`

New file or a new parametrized test appended to `test_n7og_edge_keys_multifixture.py`.

Reuses `_build_shadow_cms_pair` and `_FIXTURES`. Calls `build_dataflow_graph` on both
captures, then calls `compare_graphs` in both directions, asserts `==[]` on all three
fixtures. Uses `isa_infrastructure` fixture (real kernel builds, hardware-path tests).

**This is the definitive C3h lock-in assertion.**

---

## 4. Step order

1. Read `test_diagnose_extra_edge.py` to confirm import surface before extending.
2. Write `test_c3h_cross_iter_live_in.py` — synthetic tests, no real builds. Run under
   tox unit env to confirm they pass.
3. Append SCC + partial-match tests to `test_diagnose_extra_edge.py`. Run.
4. Write `test_n7og_compare_graphs_symmetric.py` (or append to multifixture file). Run
   under `isa_infrastructure` fixture (requires real kernel). Confirm all three fixtures
   pass in both directions.
5. Confirm no regressions across the rest of the unit suite (excluding
   `test_MatrixInstructionConversion.py`).
6. Commit: "C3h (si5f): cross-iter/cross-body live-in tests + n7og compare_graphs
   symmetric lock-in".

---

## 5. Validation (expected failure delta)

**Before C3h:** 20 FAILED + 2 ERROR (per bead description). All are (b)-class or
pre-existing; none are addressable by C3h.

**After C3h:** same 20 FAILED + 2 ERROR unchanged (they are C4-scope or pre-existing).
Zero new failures. New tests (cross-iter, cross-body, SCC, partial-key, compare_graphs
symmetric) all pass.

**Green for principled reasons** means: n7og fixtures pass `compare_graphs` in both
directions; the new structural tests capture the cross-iter / cross-body live-in contract
that motivated the entire C3-chain rewrite.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| PRO capture constructor: `FourPartCapture.prologue` is `Optional[LoopBodyCapture]` — passing a PRO body requires constructing a valid `LoopBodyCapture` via `make_capture(BODY_LABEL_PROLOGUE, ...)`. If `make_capture` rejects BODY_LABEL_PROLOGUE (the label table check), the test setup fails. | Verify `BODY_LABEL_PROLOGUE` is in `BODY_LABEL_TO_LOOP_INDEX` before writing the test — it is (maps to -1). |
| ML_PREV-to-ML cross-iter: the current `_wrap` helper in `test_diagnose_extra_edge.py` puts a filler in `main_loop_prev`. A new test helper in `test_c3h_cross_iter_live_in.py` must construct `main_loop_prev` with the writer LR explicitly and `main_loop` with the reader MFMA. Ensure filler bodies don't introduce conflicting byte-key writes. | Use disjoint vgpr ranges for filler and payload. |
| `compare_graphs` category-count gate may fire if the synthetic cross-iter graph has mismatched MFMA counts between ref and subj. | Ensure both graphs are built from the same `FourPartCapture` (no separate ref/subj construction); use `compare_graphs(g, g)` for the no-failure assertion. |
| The n7og symmetric compare_graphs test requires real-kernel build (hardware emulation). If the capture pipeline raises an exception for any of the three fixtures post-C3g migration, the test fails at setup. | The fixtures already pass the `test_shadow_vs_cms_edge_keys_match` probe in their current state. The compare_graphs call is additive. |
| SCC byte-key fixture construction: `_byte_keys_for_resource` for an SCC register may return `(("s", "scc"),)` or a numeric key depending on the architecture profile. Test must construct an edge whose `producer_write_byte_key` contains an SCC key, which may require inspecting the `make_swait` output path. | Construct the SCC key manually by patching `producer_write_byte_key` on a fabricated edge (same mock strategy as Test 4 in `test_diagnose_extra_edge.py`). |

---

## 7. New beads

None required. All scope is contained within si5f. If during implementation an unexpected
failure surfaces that is (c)-class (real defect, not a (b)-class pin), file a new bead
with `br dep add r62g <new-bead>` per the standing rule. Do not defer.

---

## 8. Verifier corrections (2026-06-09)

### 8.1 xfail claim — CONFIRMED CORRECT
`grep -n "marks=pytest.mark.xfail"` on `test_n7og_edge_keys_multifixture.py` returns only
docstring lines (82, 84, 289, 295). No live executable `xfail` markers remain. §1.1 is
accurate.

### 8.2 Cross-iter test design — PARTIALLY SUPERSEDED
`test_dataflow_graph_builder.py::test_cross_iter_edge_carries_diagnostic_annotations`
(class inferred from file structure, added in a prior cycle) already constructs a
`FourPartCapture` with ML_PREV writing v8..v11 and ML reading them via MFMA, and asserts
`producer_iter_index==0`, `consumer_iter_index==1`, and both body labels. The `_wrap`
helper in `test_dataflow_graph_comparison.py` accepts `ml_prev=` explicitly.

**Correction:** The cross-iter test in `test_c3h_cross_iter_live_in.py::TestCrossIterLiveIn`
must NOT duplicate what builder already covers. Its distinctive value is:
(a) the compare_graphs round-trip assertion — `compare_graphs(g, g) == []` with a
graph that actually HAS a cross-iter edge, and (b) the negative assertion that no
`OrderInvertedFailure` appears. The `build_dataflow_graph` + field-level assertions
are already locked in; don't repeat them. Focus only on the comparison contract.

### 8.3 Cross-body test design — INFRASTRUCTURE CONFIRMED
`test_dataflow_graph_hdem.py` has `_wrap_with_pro` and `_wrap_no_pro` helpers that
already construct `FourPartCapture` with a PRO capture. `make_capture(BODY_LABEL_PROLOGUE, ...)`
is validated by `dataflow_fixtures.py` (BODY_LABEL_PROLOGUE maps to loop_index=-1 in
`BODY_LABEL_TO_LOOP_INDEX`). No missing infrastructure.

**Correction:** `TestCrossBodyLiveIn` in the new file can import `_wrap_with_pro` (or
replicate the same pattern locally). The plan's §B claim that `_wrap` is the only helper
and that a "new test helper" is needed is wrong — `_wrap_with_pro` already exists in the
test suite and covers exactly this setup. Import it directly.

### 8.4 EdgeRoutedDifferentlyFailure gaps — FINDINGS

The plan identifies two missing cases: SCC byte-key path and partial-key miss. Verified
against the actual `diagnose_extra_edge` code:

- **Empty byte_key footprint:** If `subj_edge.producer_write_byte_key == ()`, then
  `bks` is empty, `missing_bks` is `[]` (Phase 0 passes), the Phase 1 loop is empty,
  `all_match` is vacuously True, and `UnexplainedExtraEdgeError` is raised. This path is
  not covered by any existing test. The plan does not mention it. File note: this is an
  edge case that can only arise from a malformed synthetic edge (real edges always have
  non-empty `producer_write_byte_key`); not a blocker but worth a comment.

- **SCC byte-key path:** The plan says construct an edge with `producer_write_byte_key =
  (("s", "scc"),)`. Verified: `diagnose_extra_edge` does NOT have special SCC handling —
  SCC clearing happens in `build_dataflow_graph`'s body-boundary loop (line 2159), not in
  the classifier. An SCC-keyed edge entering `diagnose_extra_edge` is handled identically
  to any other byte_key. The plan's §D claim that "SCC clobbers subsume into the byte-key
  model per plan §4.1" is correct, but the test rationale is weak: it tests that the
  generic path works for an SCC key, not any SCC-specific branch. Low but non-zero value.

- **`lr_to_gr_lds_reuse` / `gr_to_lr_lds_reuse` edge kinds:** `diagnose_extra_edge`
  operates purely on `producer_write_byte_key` regardless of `edge_kind`. These edge kinds
  are not handled specially in the extra-edge path (unlike `diagnose_missing_edge` which
  has special LDS slot logic). No gap here — same code path for all kinds. Plan does not
  claim otherwise; confirmed.

- **WAW / WAR:** The graph only models RAW edges (`raw_intrawave`, `lds_raw_intrawave`,
  `lr_to_gr_lds_reuse`, `gr_to_lr_lds_reuse`). WAW/WAR do not exist as edge kinds. No gap.

- **Consumer with cross-body byte-key history:** Not a gap in `diagnose_extra_edge` —
  the consumer identity lookup in ref is by object identity, and the closest-prior-writer
  lookup uses `ref_graph.byte_key_writers` which is already the full unrolled history.
  Cross-body history is naturally included. No missing coverage.

**Net verdict on §D:** The two cases the plan adds (SCC key + partial miss) are the right
additions. The partial-miss test is the higher-value one (it actually tests a code path
that differs: Phase 0 fires when some byte_keys are absent from ref). The empty-bks case
is a gap the plan does not mention — add a comment in the implementation noting it is
unreachable from production code (no action required, just documentation).

### 8.5 Symmetric lock-in test — FEASIBLE

`compare_graphs(g, g)` for a graph that contains a cross-iter edge: the edge appears in
both `ref_keys` and `subj_keys` under the same `edge_keys()` tuple (iter-blind identity
+ byte_key), so it cancels in set-diff in both directions. No `missing_keys` and no
`extra_keys`. Result is `[]`. This holds by construction — it is a property of set-diff,
not of any hardware or schedule detail. The test is sound.

For the n7og 3-fixture round-trip: `_build_shadow_cms_pair` builds two SEPARATE captures
(default and CMS), not the same object. `compare_graphs(ref=cms_graph, subj=default_graph)`
and the reverse direction both returning `[]` is a genuine assertion about edge_key
symmetry. This is the definitive lock-in. The plan is correct here.

**Confirmed feasible.** No blocker.

### 8.6 "GREEN for principled reasons" scope — CONFIRMED WITH ONE CORRECTION

The plan claims "none of the 20 current failures are addressable by C3h." The two named
(b)-class tests (`test_real_kernel_validates_clean_with_carveout_engaged` and
`test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`) have NO
`@pytest.mark.xfail` annotation — they are genuinely FAILING tests (not xfail-pinned
failures). They fail because the exemption they relied on was deleted; they are (b)-class
re-fixture work for bead `rocm-libraries-5ryl`. The `nyb5` Cycle 2 test has
`@pytest.mark.xfail(strict=True)` at line 136 of `test_approach_a_non_cms_reference.py`.

**Correction to plan §F:** The two carveout tests are HARD FAILS (not xfailed), counted
among the 20. The plan's characterization as "(b)-class" is correct, but the reader
should not infer they are `xfail`-suppressed — they appear as red FAILED in the run.
This does not change the scope of C3h; it is a prose clarity issue. Update §F to say
"hard FAILED (no xfail marker), counted among the 20" for the carveout tests.

### 8.7 Re-fixture risk — NO CONFLICT FOUND
No existing test pins "no cross-iter edges exist." The only tests that assert on
`producer_iter_index` / `consumer_iter_index` are in `test_dataflow_graph_builder.py`
and `test_dataflow_graph_register_gaps.py`, and they assert positive cross-iter edge
properties (not absence). The new test files introduce only additive assertions.
No invalidation risk.
