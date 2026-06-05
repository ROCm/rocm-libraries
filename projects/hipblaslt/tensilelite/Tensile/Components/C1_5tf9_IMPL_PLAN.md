# Implementation Plan — C1 (bead `rocm-libraries-5tf9`)
# Delete cross-subiter ALU-producer exemption + ML_MAT_COUNT + PrefetchGlobalRead assertion

**Bead:** `rocm-libraries-5tf9`
**Blocks:** `rocm-libraries-abgv` (C3a)
**Validator state after this commit:** RED (by design — validator-honest at every commit)

---

## §1 Scope

This commit executes three surgical changes and a postcondition check. (1) Delete the
cross-subiter ALU-producer exemption block from `diagnose_missing_edge` in
`CMSValidator.py`. (2) Add `ML_MAT_COUNT = 2` as a module-level constant in the same
file; it is unused in C1 and will be consumed by the `UnrolledCapture` materializer in
C3a. (3) Add `assert kernel["PrefetchGlobalRead"] == 2` at the SHADOW-block validation
entry in `KernelWriter.py`, citing `UNROLLED_VALIDATION_PLAN.md §9 Q1` in the message.
(4) Verify with a grep regression check that the exemption comment text is gone. No other
logic changes. The validator's output turns RED: 192 BPG#11 `OrderInvertedFailure`
instances (and analogous failures on other `UsePLRPack=True` fixtures) surface honestly.
These are expected and correct — the hack was hiding them.

---

## §2 Target sites

### 2.1 Exemption deletion — `CMSValidator.py:3831-3843`

**File:** `Tensile/Components/CMSValidator.py`

**Current state (lines 3830-3843, inside `diagnose_missing_edge`):**

```python
        if default_p_before_c and not subj_p_before_c:
            # Cross-subiter ALU-producer edges are a known false-positive
            # source: a PackA3 (subiter 3) writes a symbolic vgpr that an
            # earlier-subiter MFMA reads under the same symbolic name. The
            # default schedule emits all Packs before all MFMAs (linear
            # within-body); CMS pipelines so subiter-N+1's Pack issues after
            # subiter-N's MFMA — the order inversion across subiters is
            # legitimate pipelining, not a real reorder of a same-subiter
            # dependency. Mirrors the same-subiter gate
            # _classify_edge_coverage uses in within-graph mode.
            nmps = subj_graph.num_mfma_per_subiter
            if (_is_alu_producer(p_node)
                    and p_node.subiter(nmps) != c_node.subiter(nmps)):
                return []  # cross-subiter pipelined dependency — legitimate
            return [OrderInvertedFailure(
```

**Replacement state:** Delete lines 3831-3843 in their entirety. The comment, the
`nmps` assignment, and the `if _is_alu_producer(...)` guard all go. The resulting
structure under `if default_p_before_c and not subj_p_before_c:` is:

```python
        if default_p_before_c and not subj_p_before_c:
            return [OrderInvertedFailure(
                producer=cms_node_label(p_node, subj_graph.body_for(p_node)),
                consumer=cms_node_label(c_node, subj_graph.body_for(c_node)),
                iter_delta=p_node.iter_delta_to(c_node),
                default_producer_position=ref_p.position,
                default_consumer_position=ref_c.position,
            )]
```

**What stays:** The `_is_alu_producer` function definition at line 2776 is retained.
It is still imported and used by two external test files:
- `Tensile/Tests/unit/test_quad_cycle_dispatch_table.py:56` — imports and exercises
  `_is_alu_producer` directly as part of the dispatch-table correctness tests.
- `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py:1181` — imports and
  asserts on `_is_alu_producer` behavior for PackMFMA carve-out regression.

Do NOT delete `_is_alu_producer`. The function is no longer called from within
`CMSValidator.py` itself after the deletion, but it is a tested public symbol.
The comment at line 1555 (`# the ALU set (_is_alu_producer); see that function's
docstring...`) can be left unchanged — it's descriptive context for the
`_NON_ALU_CATEGORIES` frozenset and remains accurate.

**Companion GapRule — NOT part of C1 scope:**
The `_alu_cross_subiter_passthrough` GapRule in `_build_cdna4_gap_rules()` (around
line 728) uses `condition="cross_subiter_alu_artifact"` and mirrors the exemption's
intent for the within-graph `_classify_edge_coverage` path. This GapRule is NOT
deleted in C1. It governs a different code path (within-graph timing checks, not
cross-graph `diagnose_missing_edge`). Its deletion belongs to a future commit when
the unrolled walk makes the per-body subiter comparison meaningful on a single
timeline. Deleting it now would break within-graph validation on the same
`UsePLRPack=True` fixtures, creating noise that obscures the signal from the
cross-graph failures being surfaced by the exemption deletion.

### 2.2 `ML_MAT_COUNT = 2` constant — `CMSValidator.py` module level

**File:** `Tensile/Components/CMSValidator.py`

**Placement:** After the MFMA Type-Switch Threshold block and before the
`DataflowGraph` section header. Specifically, add it after line 879 (the blank
line following `_MFMA_TYPE_SWITCH_THRESHOLD_FROM_4X4`), before the `# =======`
section comment at line 882. The constant belongs near other module-level numeric
constants that parameterize the validator's behavior.

**Current state at line 880-882:**

```python
)


# =============================================================================
# Dataflow graph — GraphNode / DataflowEdge / DataflowGraph
```

**Replacement state:**

```python
)


# Number of main-loop iter copies materialized in the unrolled timeline.
# Consumed by UnrolledCapture (C3a). Hardcoded for PrefetchGlobalRead=2;
# the validator has only been empirically verified for this prefetch depth.
# See UNROLLED_VALIDATION_PLAN.md §9 Q1.
ML_MAT_COUNT = 2


# =============================================================================
# Dataflow graph — GraphNode / DataflowEdge / DataflowGraph
```

### 2.3 `PrefetchGlobalRead` assertion — `KernelWriter.py` SHADOW block

**File:** `Tensile/KernelWriter.py`

**The assertion site choice:**

There are two CMS validation blocks in `kernelBody`:

1. **SHADOW block** (lines ~6170-6319, gated by `_captureDefaultSchedule`): validates
   the SHADOW-default reference against the CMS schedule. This block runs FIRST on any
   CMS kernel. `kernel` is in scope as a parameter to the enclosing function.

2. **xj16 real-vs-real block** (lines ~6361-6483, gated by `_captureNonCmsBuild`): the
   second validation pass using a separately produced reference. Also uses `kernel`.

The assertion belongs in the SHADOW block, placed once, before the first validation
runs. This fires exactly once per production CMS kernel build. The escape hatch
`_capture_skip_internal_validate=True` (set via `enable_capture_default_schedule_no_assert`)
bypasses the entire validation block including the assertion; that method is documented
as test-only and MUST NOT be used by production callers. Adding the assertion to the
xj16 block too would produce a redundant double-fire. The SHADOW block is the canonical
entry because all production CMS kernels activate it (auto-activation via `kernelBody`
head at line 5617-5618).

**Current state in the SHADOW block (lines 6280-6295, inside the `if ctx.cms is not
None and not getattr(...)` guard):**

```python
            kernel_label = (
              f"{kernel['MacroTile0']}x{kernel['MacroTile1']}x{kernel['DepthU']}"
            )
            ref_graph = build_dataflow_graph(ctx.default)
```

**Replacement state:** Insert the assertion immediately before `kernel_label`:

```python
            assert kernel["PrefetchGlobalRead"] == 2, (
              "CMS validator has only been empirically verified for "
              "PrefetchGlobalRead=2. A kernel with a different prefetch depth "
              "reached the validator before the unrolled walk has been "
              "re-investigated for this depth. "
              "See UNROLLED_VALIDATION_PLAN.md §9 Q1."
            )
            kernel_label = (
              f"{kernel['MacroTile0']}x{kernel['MacroTile1']}x{kernel['DepthU']}"
            )
            ref_graph = build_dataflow_graph(ctx.default)
```

**Why this site, not the xj16 site:** The SHADOW block is the first validation
point for every CMS kernel. If PGR != 2 the assertion fires before any graph is
built, giving an actionable error with a citation. Placing it in the xj16 block
would miss kernels that don't reach that block (e.g. test fixtures that set
`_capture_skip_internal_validate`). Placing it in both blocks is redundant and
creates maintenance ambiguity about which is authoritative.

**Exact line reference:** The assertion inserts just before the current line 6289
(`kernel_label = ...`), inside the `if ctx.cms is not None and not getattr(
self, "_capture_skip_internal_validate", False):` condition (current line 6280-6281).

### 2.4 Regression check

**Command:**
```bash
grep "Cross-subiter ALU-producer edges" Tensile/Components/CMSValidator.py
```

**Expected return code:** 1 (no matches). The exact phrase "Cross-subiter
ALU-producer edges" appears only in the exemption comment at line 3831 and nowhere
else in the file. After deletion, zero matches remain.

**Why this grep pattern (not the plan's stated "cross-subiter ALU-producer
exemption"):** The word "exemption" does NOT appear in the code block being deleted;
it is a summary term from the plan document. The actual comment text at line 3831 is
`"# Cross-subiter ALU-producer edges are a known false-positive"`. The grep must
match text that IS in the block before deletion and CANNOT be present after. The
phrase `Cross-subiter ALU-producer edges` is unique to line 3831 and is gone after
deletion.

---

## §3 Step-by-step implementation order

Perform in this order (each step is a distinct, locally verifiable action):

**Step 1 — Add `ML_MAT_COUNT = 2` to `CMSValidator.py`.**
Insert after line 879. Verify: `grep "ML_MAT_COUNT" Tensile/Components/CMSValidator.py`
returns the new constant at the expected position.

**Step 2 — Delete the exemption block from `CMSValidator.py`.**
Remove lines 3831-3843 (the 13-line comment + `nmps` assignment + `if
_is_alu_producer(...)` guard). Leave line 3844 (`return [OrderInvertedFailure(`)
directly under line 3830 (`if default_p_before_c and not subj_p_before_c:`).
Verify indentation: `OrderInvertedFailure` must still be at 12 spaces (3 indent
levels inside `if ... and not ...:` inside `if p_node.body_label ...` inside
`diagnose_missing_edge`).

**Step 3 — Add `PrefetchGlobalRead` assertion to `KernelWriter.py`.**
Insert before `kernel_label` at line 6289 in the SHADOW block. The assertion
message cites `UNROLLED_VALIDATION_PLAN.md §9 Q1`.

**Step 4 — Run the regression grep.**
```bash
grep "Cross-subiter ALU-producer edges" Tensile/Components/CMSValidator.py
```
Must return exit code 1.

**Step 5 — Run the unit suite and classify every new failure.**
See §4 for the exact commands and §5 for the classification protocol.

---

## §4 Validation

### 4.1 Regression grep (pass-before-tests)

```bash
# Run from the worktree root: .worktrees/validator_long_term_plans/projects/hipblaslt/tensilelite/
grep "Cross-subiter ALU-producer edges" Tensile/Components/CMSValidator.py
echo "Exit code: $?"
```

Expected: no output, exit code 1.

```bash
grep "ML_MAT_COUNT" Tensile/Components/CMSValidator.py
```

Expected: one hit on the new constant line.

```bash
grep "PrefetchGlobalRead.*==.*2" Tensile/KernelWriter.py | head -5
```

Expected: the new assertion is visible (among other existing PGR comparisons in
the file).

### 4.2 Syntax check

```bash
python3 -c "import ast; ast.parse(open('Tensile/Components/CMSValidator.py').read()); print('OK')"
python3 -c "import ast; ast.parse(open('Tensile/KernelWriter.py').read()); print('OK')"
```

### 4.3 Unit suite

```bash
# If running inside a venv (adjust venv path per user environment):
python3 -m pytest Tensile/Tests/unit/ \
    --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
    -x --tb=short 2>&1 | tee /tmp/c1_pytest_run.log
```

**Expected outcome: RED (many failures).** This is correct and expected. The
exemption deletion surfaces 192 BPG#11 `OrderInvertedFailure` instances as
honest errors. Every test whose pass-state depended on the exemption silencing
those edges now fails.

**Expected failures (known in advance):**
- `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_validates_clean_with_carveout_engaged`
  — was pinning the "validator green with carve-out engaged" behavior. Now fails
  with 192 `OrderInvertedFailure` items. Classification: **(b)** — re-fixture in C4.
- `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`
  — was pinning the 192-failure count when the carve-out is neutralized via
  monkeypatch. After C1, the `diagnose_missing_edge` exemption is unconditionally
  deleted, so `compare_graphs` returns 192 `OrderInvertedFailure` instances whether
  or not the monkeypatch is applied. The monkeypatch is now irrelevant; the exemption
  path it was neutralizing no longer exists. The test still asserts `len(failures) == 192`
  — this assertion may **accidentally pass** (the count is unchanged, just via a
  different code path). The test's purpose — pinning the carve-out neutralization
  behavior — is superseded. Classification: **(b)** — re-fixture in C4 (the
  test's structural assumption about what causes the 192 failures is now wrong, even
  if the count happens to match). If the test accidentally passes, note it explicitly
  in the run log as a false green for C4 re-fixture purposes; do not treat it as a
  genuine green.

## CORRECTION (verifier 2026-06-05)
The original classification predicted this test would fail. It will instead likely
PASS accidentally after C1 because: (1) the exemption in `diagnose_missing_edge` is
deleted, making the 192 `OrderInvertedFailure` surfaces unconditional; (2) the test
asserts exactly 192, which still holds; (3) `compare_graphs` does not call
`_classify_edge_coverage` / the GapRule path, so the GapRule (still present in C1)
does not interfere. The test is still class (b) — it must be re-fixtured in C4 —
but it is NOT a reliable RED signal after C1. Use the first test and the
`test_cross_subiter_pack_artifact.py` fixture as the reliable RED indicators.
- Any fixture driven via `test_cross_subiter_pack_artifact.py` that relies on
  `carveout_suppresses_artifact` — classification **(b)**.
- `test_ScheduleCapture.py` production-kernel tests (`TestRealKernelCapture`) — the
  tests call `_getKernelSource`, which triggers `kernelBody`. For `UsePLRPack=True`
  CMS kernels, `kernelBody`'s auto-activated SHADOW block runs `compare_graphs` and
  hits `assert not graph_failures` when the 192 cross-subiter edges surface. The
  test fails with `AssertionError` raised from inside `_getKernelSource` — the test
  does NOT call `compare_graphs` directly; the failure comes from the inline
  kernelBody assertion. Classification: **(a)** (representational gap — unrolled
  walk resolves them). Note: tests that call `enable_capture_default_schedule_no_assert`
  are exempt from this assertion and will not fail via this path.

**Unexpected failure rule:** If a test that was NOT on the known-break list fails
AND its failure trace does NOT show `OrderInvertedFailure` involving
`PackA*/PackB*` → `MFMA` cross-subiter edges, apply the classification protocol
in §5. If it turns out to be class (c), file a new bead immediately.

### 4.4 Confirming the `PrefetchGlobalRead` assertion fires correctly

The assertion guards the production path. To verify it would fire on a PGR!=2
kernel without running a full build, inspect the insertion point visually: confirm
the `assert` is inside the `if ctx.cms is not None` guard and before
`ref_graph = build_dataflow_graph(ctx.default)`. No additional test is added in C1;
the assertion's correctness is self-evident by inspection and will be exercised
by production CMS builds.

---

## §5 Failure-classification protocol

Apply to every test failure that appears after C1. Every failure must be assigned
exactly one class:

### Class (a) — Representational gap, to be resolved by C3

**Criteria:** The failure is `OrderInvertedFailure` (or a missing-edge failure)
on a cross-subiter `Pack* → MFMA` or `LR* → Pack*` edge in a `UsePLRPack=True`
fixture. The producer is in subiter N, consumer in subiter M < N, and the body
is `NLL` or `NGL`. The unrolled-program walk (C3c–C3f) will resolve these because
the producer in the previous ML iter copy will be the true `latest_writer` at the
consumer's moment, making these edges disappear from the set-diff.

**Disposition:** Leave RED. Do NOT re-fixture. The test assertions that were
asserting `failures == []` will stay failing until C3h removes the xfail markers.
Note the test in the run log for the C4 re-fixture list.

### Class (b) — Test pinning the exemption's silencing behavior

**Criteria:** The test was explicitly designed to assert that the exemption fires
(e.g. `test_real_kernel_validates_clean_with_carveout_engaged` — asserts
`failures == []` because carve-out absorbs them) or to assert specific counts
produced by toggling the exemption (e.g.
`test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`). These
tests' expected behavior is structurally coupled to the exemption's existence.

**Disposition:** Leave RED. Note for re-fixture in C4 (`rocm-libraries-5ryl`). Do
NOT update the assertion or add `xfail` markers in this commit.

### Class (c) — Real bug the exemption was wrongly silencing

**Criteria:** The failure is NOT `OrderInvertedFailure` on a
cross-subiter Pack → MFMA edge. OR it is `OrderInvertedFailure` but on an edge
that is a genuine dataflow dependency violation (the CMS scheduler incorrectly
reordered a real RAW dependency, not a legitimate pipelined one). Distinguishing
mark: in the reference graph, the producer appears before the consumer at a
position that is NOT explained by cross-subiter pipelining. Equivalently: the
failure appears on a fixture that does NOT use `UsePLRPack=True`.

**Disposition:** File a new P0 bead immediately using:
```bash
ACTOR="${BR_ACTOR:-assistant}"
br --db /home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/.beads/beads.db \
    create --actor "$ACTOR" "C1 real bug: <description>" \
    --priority 0 --type bug \
    --description "..."
br --db /home/alvasile/rocm-libraries/.worktrees/validator_long_term_plans/projects/hipblaslt/.beads/beads.db \
    dep add <new-id> rocm-libraries-r62g
```

Do NOT re-fixture to make it pass. The bug must be filed and remain open until
independently fixed. The C1 commit lands with this test RED and classified as (c).

---

## §6 New beads to file

None are expected. The 192 BPG#11 failures are fully accounted for as class (a)
under the existing bead chain. The break-list tests from
`UNROLLED_VALIDATION_PLAN.md §6.1` are accounted for as class (b) under C4
(`rocm-libraries-5ryl`).

**Only file a new bead if a class (c) failure is found during §4.3.** If found,
file immediately before committing; do not defer.

---

## §7 Open questions and risks

### 7.1 The companion `_alu_cross_subiter_passthrough` GapRule is NOT deleted in C1

The within-graph timing path (`_classify_edge_coverage`) has its own
`cross_subiter_alu_artifact` condition in the gap-rule table. This GapRule remains
in C1. It governs a different comparison: within the CMS graph alone (not
cross-graph SHADOW-vs-CMS). Deleting it now would surface additional failures from
within-graph timing analysis that are mechanistically identical to the cross-graph
ones but come from a different path. Those would compound the RED state without
serving the plan's goal of making the cross-graph comparison honest.

The within-graph GapRule will be re-evaluated after the unrolled walk lands (C3c).
At that point, the subiter arithmetic it relies on may no longer apply (the unrolled
stream's concept of "subiter" changes meaning). If the GapRule is also a hack, it
should be deleted then — with its own bead and classification.

**Risk:** A reviewer may ask why the companion GapRule was not also deleted. The
answer: C1's scope, as stated in the bead, is specifically `CMSValidator.py:3831-3843`.
The GapRule is not in that range. Scope creep would mix two independent concerns.

### 7.2 The `_is_alu_producer` function becomes unreferenced within CMSValidator.py

After C1, `_is_alu_producer` is defined in `CMSValidator.py` but never called from
within the file. It IS imported by external test files. The function should NOT be
deleted in C1. No refactoring of its callers is needed in C1; that belongs to a
future commit (the QUAD_CYCLE_DISPATCH_AUDIT.md proposes collapsing it into
`shape_of()`, which is not part of this chain).

### 7.3 rocisa staleness

If running tests inside the worktree for the first time, rocisa may need to be
rebuilt before pytest can import it:

```bash
pip install -e ./rocisa
# or
invoke rocisa
```

The staleness check in `rocisa/__init__.py` compares `.cpp/.hpp` modification times
to the loaded `.so`. If any source file is newer than the `.so`, import raises. Run
the install once; subsequent test runs will not require it unless rocisa sources
change.

### 7.4 Grep pattern vs. plan document phrasing

The plan document says `grep "cross-subiter ALU-producer exemption"`. That string
does NOT appear in `CMSValidator.py` — it is the plan's name for the block, not
literal code. The implementer MUST use `grep "Cross-subiter ALU-producer edges"`
(matching the actual comment text at line 3831) as the regression check. Using the
plan's phrasing would always return 0 matches (false green) regardless of whether
the deletion happened.

---

*Plan written: 2026-06-05. Author: investigation agent for bead rocm-libraries-5tf9.*
