# w5xw Correctness & Soundness Report

## Verdict

**CLEAN**

The fix is sound. SHADOW and CMS are produced by distinct code paths within Build #1 (different `ctx.default` vs `ctx.cms` writes at `Tensile/KernelWriter.py:6240` and `:6260`), so the re-route produces a meaningful 0-mismatch comparison rather than a tautological one. All 12 tests in the target file pass; full unit baseline matches expected exactly (1062 passed, 4 skipped, 2 xfailed, 1 pre-existing u6nn failure). No regressions, no orphaned fixtures, no new red flags. The n7og bead correctly preserves the unaddressed architectural concern at `Tensile/Components/CMSValidator.py:1300`.

## Per-Q

### Q1 — SHADOW genuinely differs from CMS?

**YES — genuinely distinct.** Confirmed by tracing the writer attribute setters:

- `_last_default_capture` (`Tensile/KernelWriter.py:451-457`) wraps `ctx.default` (a `CaptureContext.default` slot).
- `_last_cms_capture` (`Tensile/KernelWriter.py:460-465`) wraps `ctx.cms`.

Inside `_getKernelSource`'s `_captureNonCmsBuild` block:
- `ctx.default` is populated at `Tensile/KernelWriter.py:6240` from a `FourPartCapture(...)` built from the SHADOW machinery's instructions (driven by the `_captureDefaultSchedule` flag — references at lines 4185, 5103, 5618, 5854, 5954, 6012, 6170).
- `ctx.cms` is populated at `Tensile/KernelWriter.py:6260` from `build_cms_four_part_capture(...)` driven by the `customMainLoopSchedule` pending macro inputs (`self._pending_cms_capture_inputs`).

These are two `FourPartCapture` instances built from disjoint instruction-collection paths, populated at distinct sites within the same Build #1. The 0-mismatch property is a design outcome of dm4p Phase 2 (shared register-allocation + shared codegen branches), not the test trivially comparing an object to itself.

### Q2 — Could a regression go undetected?

**Bounded "yes" — documented, not a new defect.**

- A regression that broke SHADOW into emitting from the CMS-side instruction stream (e.g. accidentally aliasing `ctx.default = ctx.cms`) **would not** be caught by these 2 tests alone (0 == 0). However, sibling tests defend this independently: `test_lcc_invariant_per_body_use_loop_predicate` (line 857) asserts per-body LCC invariants on each side without cross-side reference, and `test_e293_scc_cross_build_identity_stable_via_source_module_id` (line 944) asserts source-module-id population independently.
- A regression in BOTH SHADOW and CMS producing the same wrong stream **would not** be caught — but this is the same limitation any "same-codegen reference" comparison has, and is exactly what the design v5 §1.5 contract accepts. The two retained Approach-A fixture tests (lines 858, 945) add independent invariants that would catch class (a) regressions.

This isn't a defect introduced by w5xw — it's a constitutional limitation of SHADOW-as-canonical, called out in design v5 and accepted. No bead filed.

### Q3 — Approach-A fixture orphaned?

**NO — still wired correctly.** `real_kernel_capture_pair_approach_a` (defined at `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:540`) is still consumed by:
- `test_lcc_invariant_per_body_use_loop_predicate` (line 858)
- `test_e293_scc_cross_build_identity_stable_via_source_module_id` (line 945)

Both pass (verified in Q6 run). The fixture's `build_non_cms_reference` path (`Tensile/Components/CustomSchedule/approach_a.py:216`) is exercised. Fixture retention is appropriate because these two tests legitimately need Approach A's unmutated `kernel` dict (line 562 docstring notes Approach A surfaces 3 GR-OrderInverted residuals tracked under rocm-libraries-3ija — those wouldn't surface under SHADOW).

### Q4 — n7og bead update preserves real defect?

**YES.** The architectural concern is genuinely unaddressed in code: `DataflowGraph.edge_keys()` at `Tensile/Components/CMSValidator.py:1300` still returns `{(e.producer.identity, e.consumer.identity, e.edge_kind, ...)}` — embedding the full identity tuples that include canonical render. The compare_graphs path also embeds them at line 3722.

The bead correctly:
- Distinguishes SHADOW-vs-CMS (0 mismatches empirically) from speculative future fixtures (could still trigger).
- Retains `dep_type: blocks` on r62g (Phase 3) — appropriate because Phase 3's multi-fixture coverage is the only mechanism that could empirically surface the speculative defect.
- Documents the 81/19 split honestly with the SHADOW-vs-Approach-A distinction.
- Remains `status: open`, `priority: 0` — not prematurely closed/downgraded.

### Q5 — Comment annotation correctness?

**ACCURATE.** `Tensile/Components/CMSValidator.py:3626-3654`:
- Correctly cross-references rocm-libraries-n7og as P0 Phase 3 blocker.
- Technical description matches the actual `edge_keys()` implementation at line 1300.
- Does not claim the defect is closed — explicitly states "the 81% T/X edge-layer concern remains real" and "future fixtures with cross-instance register-naming drift would still trigger edge-key mismatches here."
- Distinguishes SHADOW (0 mismatches) from Approach A's reference noise (642 mismatches, 81/19 split).
- References `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md §1.5, §6 oplb-row` correctly.

### Q6 — Baseline match?

**EXACT MATCH.** Run from `$WT` with `--ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py`:

- Target file: `12 passed in 5.09s`. All 3 affected tests pass: `test_real_kernel_per_render_counts_match PASSED`, `test_real_kernel_per_ordinal_logical_instruction_matches PASSED`, `test_example_yaml_no_spurious_order_inverted_failures PASSED`.
- Full unit suite: `1 failed, 1062 passed, 4 skipped, 2 xfailed in 22.49s`. Sole failure is `test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference` — pre-existing u6nn (also failed at parent).

Matches the agent's expected outcome bit-for-bit.

### Q7 — Corner cases?

- **SHADOW=None handling.** Fixture asserts at `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:534-535` that both `_last_default_capture` and `_last_cms_capture` are non-None before returning. If SHADOW were None (e.g. capture-window gap), the fixture would fail-loud (clear assertion error) rather than tests crashing in `_captures_per_body` later.
- **PGR=2 nmsx capture-window gap.** `CANONICAL_TF32_4X4_TN_CONFIG` at `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py:488` sets `PrefetchGlobalRead: 2`. The re-routed tests therefore DO exercise the same PGR=2 fixture that nmsx Fix 1 addressed. 0 mismatches on this fixture means nmsx's window/scope/walk fixes are working in concert with the SHADOW-based test (not just a PGR=1 happy path).

### Q8 — Diff failure lists pre/post?

Procedure: swapped `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py` and `Tensile/Components/CMSValidator.py` to parent (`7322d99c506`) versions, ran unit suite, then restored HEAD versions.

- **Pre-w5xw failures (`/tmp/pre_w5xw_failures.txt`):**
  ```
  FAILED test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_ordinal_logical_instruction_matches
  FAILED test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_render_counts_match
  FAILED test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference
  ```
  Counts: `3 failed, 1060 passed, 4 skipped, 2 xfailed`.
- **Post-w5xw failures (`/tmp/post_w5xw_failures.txt`):**
  ```
  FAILED test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference
  ```
  Counts: `1 failed, 1062 passed, 4 skipped, 2 xfailed`.
- **Diff:** Only the 2 emission_ordinal tests transition FAILED → PASSED. No new failures, no new skips/xfails. The pre-existing u6nn failure persists unchanged.

Confirmed: w5xw produces exactly the intended delta (+2 passes from the same fixture re-route), with no collateral regressions anywhere else in the unit suite.

Note for the record: at the parent commit, only 2 tests in the target file were failing (not 3). `test_example_yaml_no_spurious_order_inverted_failures` was already passing at parent (no skip marker, no failure). The intermediate w5xw commit `8dabfd4463d` added 3 skip markers; the final commit `7cd8510cc68` removed all 3 and re-routed 2 tests. Net diff from parent → HEAD is therefore: re-route 2 tests + comment rewrites, NOT "remove a 3rd test's false skip." This is a wording discrepancy in the round-2 adherence report but is harmless — the final state is correct.

## Bugs found

None. No P0 beads filed.

## Recommendation

**CLEAN → Step 9 squash-merge.**

The fix is principled and complete:
1. SHADOW and CMS are genuinely distinct captures from distinct code paths (Q1).
2. The 0-mismatch outcome on the canonical PGR=2 TF32 4x4 TN fixture is a real design property of dm4p Phase 2 + nmsx Phase 1, not a tautology.
3. The Approach-A fixture remains correctly wired for its two legitimate consumers (Q3).
4. The architectural concern at `CMSValidator.py:1300` is honestly preserved in n7og at P0 blocking r62g (Q4).
5. Source annotation at `CMSValidator.py:3626` is accurate and non-overstated (Q5).
6. Baseline matches agent claim exactly (Q6, Q8).
7. SHADOW=None is handled gracefully; PGR=2 capture-window gap is exercised (Q7).
