# C4 (5ryl) Implementation Plan
# Re-fixture (b)-class tests pinning the deleted exemption's silencing

**Bead:** `rocm-libraries-5ryl`
**Depends on:** `rocm-libraries-si5f` (C3h — closed)
**Blocks:** `rocm-libraries-r62g` (Phase 3 go/no-go gate)
**Branch:** users/alvasile/mxfp4_fast_ref_min (worktree: validator_long_term_plans)

---

## §1 Scope

C4 re-fixtures the three (b)-class test cases that were pinning the cross-subiter ALU-producer
exemption deleted in C1 (79c363031ada). Two of those tests are in
`test_cross_subiter_alu_carveout_real_kernel.py`, one is in `test_cross_subiter_pack_artifact.py`.
A fourth test (`test_carveout_suppresses_artifact_and_neutralization_surfaces_it`) contains a two-part
assertion, only part of which is obsolete. The companion tests in the same file
(`test_artifact_present_in_default_graph`, `test_correct_edge_present_in_cms_graph`) remain
structurally correct under the unrolled walk and require no logic changes — only their file-level
docstring needs updating to remove stale carve-out framing. No new validator logic is written.
The entire commit is test-layer only.

---

## §2 Investigation Findings

### A — Current failure enumeration

From the C3h plan (verified 2026-06-09): 20 FAILED + 2 ERROR going into C4. The 2 ERRORs are
collection failures caused by importing `OrderInvertedFailure` from the stale installed Tensile
package in the main project's tox unit env. In the worktree's correct environment they are FAIL
or PASS, not ERROR. The breakdown:

| Category | Count | Tests |
|---|---|---|
| (b)-class — C4 scope | 3 | carveout real kernel (2 FAILs), pack artifact (1 FAIL or ERROR depending on env) |
| Pre-existing u6nn | tracked | `test_prologue_capture.py` l1l6 breakage |
| Pre-existing nyb5 | tracked | `test_approach_a_non_cms_reference.py` Cycle 2 `xfail(strict=True)` |
| Other C3h residual | per C3h §5 | Non-zero in C3h plan; all pre-existing or C-chain resolved |

The 20-minus-3 non-(b) failures are pre-existing beads (u6nn, nyb5) and/or C-chain residuals not
addressable by C4. The acceptance criterion "ALL tests pass" should be read as: all C-chain-owned
tests pass; pre-existing P0 beads (u6nn, nyb5) remain in their current state (tracked separately).

### B — Per-test behavior under post-C3 validator

**Test 1: `test_real_kernel_validates_clean_with_carveout_engaged`**

Pre-C1 behavior: `compare_graphs(ref, subj) == []` because the cross-graph exemption in
`diagnose_missing_edge` absorbed the PackA3/PackB3 → MFMA artifact edges.

Post-C3 behavior: `compare_graphs(ref, subj) == []` for the correct principled reason —
the unrolled walk builds byte-key-based edge keys (8-field tuple including `source_module_id` +
`emission_ordinal` from Option E / 56e3) that cancel in set-diff because both ref and subj resolve
the same Pack3→MFMA byte-key flows from the same physical producer. The BPG#11 / n7og 192-edge
failures that prompted the exemption are confirmed to resolve to 0 after C3 (per C3h §F and the
n7og three-fixture symmetric compare_graphs lock-in added in C3h). The assertion remains `== []`
but the docstring must be updated to attribute it to the principled cancellation, not the carve-out.

**Decision: re-fixture** — update test docstring only; assertion unchanged.

**Test 2: `test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`**

Pre-C1 behavior: monkeypatching `GraphNode.subiter` to a constant 0 disabled both carve-out
sites (the deleted `diagnose_missing_edge` block AND the `cross_subiter_alu_artifact` GapRule
in `_classify_edge_coverage`), surfacing 192 `OrderInvertedFailure`s.

Post-C1/C3 behavior: the `diagnose_missing_edge` carve-out is GONE (deleted in C1). The
remaining `_alu_cross_subiter_passthrough` GapRule lives in the within-graph timing path
(`_classify_edge_coverage`), not in the cross-graph path. The 192 failures resolved to 0 via
byte-key set-diff cancellation (C3). Monkeypatching `GraphNode.subiter` to 0 now has no effect
on `compare_graphs` output because `diagnose_missing_edge` no longer contains any subiter check.
The test assertion `len(failures) == 192` would FAIL (actual: 0).

The semantics are fully obsolete: this test was documenting the behavior of the DELETED exemption.
There is no value in re-writing it to document something else; the correct behavior (0 failures
for principled reasons) is already pinned by Test 1.

**Decision: delete** — the test's entire purpose was the exemption neutralization probe, which
no longer exists in the cross-graph path.

**Test 3: `test_carveout_suppresses_artifact_and_neutralization_surfaces_it`** (in `test_cross_subiter_pack_artifact.py`)

This test has two parts:

*Part (a)* `failures_with_carveout == []`: using the SYNTHETIC fixture (3 instructions,
`num_mfma_per_subiter=0`). Post-C3 analysis:
- The ref edge key is `PackA1→MFMA` (by `emission_ordinal=1`); subj edge key is `PackA0→MFMA`
  (by `emission_ordinal=0`). These differ → `diagnose_missing_edge` is called.
- In `diagnose_missing_edge`: Phase 1 sees both `default_p_before_c=True` AND
  `subj_p_before_c=True` (PackA0 is a prior writer in the subj unrolled stream) → falls through.
- Phase 2 dispatch: `_dispatch_quad_cycle_check(PackA0, MFMA, subj_graph)`. The `ALU→MFMA`
  rule list is `[cross_subiter_alu_artifact, same_subiter, unconditional_passthrough]`. Since
  `num_mfma_per_subiter=0` (test fixture default), neither condition-gated rule fires; the
  unconditional passthrough rule applies → `_PASSTHROUGH` → `return []`.
- **Result: still `[]`. Part (a) assertion holds** — for the principled reason that the
  unconditional ALU→MFMA passthrough fallback (covering `nmps=0` fixtures) applies in Phase 2.

*Part (b)* `len(failures_neutralized) == 1` after monkeypatching `GraphNode.subiter = 0`:
- `_evaluate_gap_rule_condition` for `cross_subiter_alu_artifact` gates on `nmps > 0` first
  (`if nmps == 0: return False`, CMSValidator.py:2958). The synthetic fixture sets
  `num_mfma_per_subiter=0` (the `FourPartCapture` field default in `ScheduleCapture.py:683`).
  So the monkeypatch on `GraphNode.subiter` is never reached — `nmps=0` short-circuits
  before `subiter()` is called on either node.
- The `same_subiter` rule is gated the same way (`nmps=0` → False).
- The unconditional passthrough fires → `return []` → 0 failures.
- **The assertion `len(failures_neutralized) == 1` FAILS** (actual: 0).

The proximate reason part (b) fails is the `nmps=0` gate, not solely the C1 deletion of the
cross-graph exemption. Even if `diagnose_missing_edge` still had a subiter check, the synthetic
fixture's `nmps=0` would prevent the monkeypatch from mattering. The monkeypatch probe was doubly
dead: first because C1 deleted the cross-graph exemption, second because the synthetic fixture
never had a non-zero `num_mfma_per_subiter` to begin with. The part (b) assertion is dead.

**Decision: re-fixture** — remove the entire monkeypatch probe block (the `# Probe:` section
and the two assertions following it). Update the test docstring to reflect that part (a) passes
for a principled reason (unrolled walk + ALU→MFMA passthrough for same-subiter synthetic
fixtures), not because of the carve-out. Rename the test to remove "carveout" from its name.

**Companion tests: `test_artifact_present_in_default_graph`, `test_correct_edge_present_in_cms_graph`**

Both assertions still hold under the unrolled walk:
- Default stream `PackA0, PackA1, MFMA` → latest_writer resolution → `PackA1→MFMA` ✓
- CMS stream `PackA0, MFMA, PackA1` → latest_writer resolution → `PackA0→MFMA` ✓
- The `producer.position < consumer.position` assertion at line 217 uses `unrolled_position`
  under the new model; PackA1 at pos 1 < MFMA at pos 2 → True ✓

No assertion changes. The file-level module docstring ("the carve-out is the only mechanism
suppressing it") needs to be rewritten to describe the new understanding: the artifact still
exists at the graph level; it is no longer suppressed by a carve-out but by byte-key set-diff
cancellation in the cross-graph comparison. The companion tests document the artifact's continued
existence in the individual graphs; the re-fixtured test 3 documents why cross-graph comparison
still returns `[]`.

**Decision: no assertion changes; update module docstring only.**

### C — Module-level docstring state

`test_cross_subiter_alu_carveout_real_kernel.py`: the module docstring (lines 25–90) describes
the carve-out in detail, cites lines that no longer exist, and references the monkeypatch as
"the principled pivot point." All of this is stale. The docstring needs a rewrite: what the
test still pins (Test 1 only), why it passes for principled reasons, and a tombstone note that
Test 2 was deleted.

`test_cross_subiter_pack_artifact.py`: the module docstring (lines 25–61) lists item 3 and 4
as "carve-out engaged" and "carve-out neutralized" behaviors. Item 4 is now removed. Item 3's
claim ("the carve-out classifies cross-subiter ALU-producer order inversions") needs to become
"the unrolled walk's ALU→MFMA passthrough fallback handles the synthetic fixture; the artifact
is no longer hidden by a carve-out but is harmless because cross-graph comparison cancels it
via byte-key set-diff."

### D — Companion test infrastructure dependencies

No other test files import from `test_cross_subiter_alu_carveout_real_kernel.py` or
`test_cross_subiter_pack_artifact.py`. The `real_kernel_graphs` module-scoped fixture is
used only within `test_cross_subiter_alu_carveout_real_kernel.py`. Deleting Test 2 does not
break the fixture; Test 1 still uses it.

### E — Red-flag scan

Scanned across CMSValidator.py and the two (b)-class test files:

| Pattern | Findings | Action |
|---|---|---|
| `setdefault` in CMSValidator | 4 sites: `byte_key_writers`, `ref_edges_by_key`, `subj_edges_by_key`, `by_category` | All standard dict-building; none are defensive classifications. No action. |
| `@pytest.mark.skip` | None in (b)-class files | No action. |
| Comments mentioning "carveout" or "exemption" in test files | Multiple in module docstrings and inline | Rewrite module docstrings as described in §C. |
| `if isinstance(...) and ...: return []` defensive returns | None found — C1 deleted the only one | No action. |
| Defensive class labels / `setdefault` on classification dicts | None found | No action. |
| `_alu_cross_subiter_passthrough` GapRule (within-graph) | Still present at CMSValidator:731 | Out of C4 scope per C1 §7.1. File as open question (§7 below). |

### F — Expected final state after C4

After C4:
- `test_real_kernel_validates_clean_with_carveout_engaged` → **PASS** (docstring updated; assertion unchanged)
- `test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures` → **DELETED**
- `test_carveout_suppresses_artifact_and_neutralization_surfaces_it` → renamed + monkeypatch probe removed → **PASS**
- Companion tests `test_artifact_present_in_default_graph`, `test_correct_edge_present_in_cms_graph` → **PASS** (already correct; no assertion changes)

Pre-existing failures:
- `test_prologue_capture.py` (u6nn) — **unchanged**, tracked by own bead
- `test_approach_a_non_cms_reference.py` Cycle 2 (nyb5) — **xfail(strict=True)**, tracked by own bead

Net delta: −3 FAIL (the three (b)-class tests), 0 new failures.

---

## §3 Per-Test Decision

| Test | File | Decision | Rationale |
|---|---|---|---|
| `test_real_kernel_validates_clean_with_carveout_engaged` | `test_cross_subiter_alu_carveout_real_kernel.py` | **Re-fixture (docstring only)** | Assertion `== []` still correct; remove all carve-out attribution from docstring |
| `test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures` | `test_cross_subiter_alu_carveout_real_kernel.py` | **Delete** | Tests the deleted cross-graph exemption; no remaining semantics |
| `test_carveout_suppresses_artifact_and_neutralization_surfaces_it` | `test_cross_subiter_pack_artifact.py` | **Re-fixture** | Remove monkeypatch probe block; update docstring; rename to `test_compare_graphs_returns_no_failures_for_cross_subiter_artifact` |
| `test_artifact_present_in_default_graph` | `test_cross_subiter_pack_artifact.py` | **No assertion change; module docstring update** | Assertion still holds; artifact still exists in default graph |
| `test_correct_edge_present_in_cms_graph` | `test_cross_subiter_pack_artifact.py` | **No assertion change; module docstring update** | Assertion still holds; correct edge still in CMS graph |

---

## §4 Red-Flag Remediation

None required in C4's scope:
- No `setdefault` on classification dicts (the 4 sites in CMSValidator are standard dict-building, not defensive classification)
- No `@pytest.mark.skip` added by C4
- No new exemptions introduced
- "carve-out" and "exemption" language in test file docstrings is removed as part of the re-fixture
- The `_alu_cross_subiter_passthrough` GapRule is out of C4 scope — see §7

---

## §5 Step-by-Step Implementation Order

### Step 1 — `test_cross_subiter_alu_carveout_real_kernel.py`

1a. Rewrite the module-level docstring (lines 25–90). Replace the carve-out description with:
   - What the file now pins: a single test (`test_real_kernel_validates_clean_with_carveout_engaged`)
   - Why it passes: unrolled walk + byte-key set-diff cancellation (not the exemption)
   - Tombstone note: "the neutralization test was deleted in C4/5ryl; the exemption it probed
     no longer exists in `diagnose_missing_edge` (deleted C1/5tf9)"

1b. Update docstring of `test_real_kernel_validates_clean_with_carveout_engaged`. Remove all
   references to "carve-out engaged." Replace with: "Validates clean under the unrolled walk:
   cross-subiter Pack3→MFMA byte-key flows cancel in set-diff because both ref and subj resolve
   the same physical producer byte-keys. The `compare_graphs(ref, subj) == []` assertion holds
   for this principled reason as of C3 (si5f)."

1c. Delete the entire `test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`
   function, including its docstring and all helper functions/imports used exclusively by it.
   Verify that `_is_pack_n_to_mfma_artifact` and other helpers are not referenced by Test 1
   before deleting them. (They are only used by Test 2 — safe to delete.)

### Step 2 — `test_cross_subiter_pack_artifact.py`

2a. Rewrite the module-level docstring. The artifact (PackA1→MFMA in default graph) still exists;
   items 1 and 2 of the original 4-item list remain correct. Replace items 3 and 4 with:
   "3. Under the unrolled walk, `compare_graphs(ref, subj)` returns zero failures — not because
   of a carve-out, but because the byte-key edge keys differ by `emission_ordinal` (Option E /
   56e3), and `diagnose_missing_edge` Phase 2 routes the ALU→MFMA synthetic-fixture edge through
   the unconditional passthrough fallback (`num_mfma_per_subiter=0`). The artifact is real at the
   graph level but harmless in the cross-graph comparison."

2b. Rename `test_carveout_suppresses_artifact_and_neutralization_surfaces_it` to
   `test_compare_graphs_returns_no_failures_for_cross_subiter_artifact`.

2c. Rewrite the renamed test's docstring to describe the current behavior: part (a) holds because
   `diagnose_missing_edge` Phase 2 applies the unconditional ALU→MFMA passthrough fallback when
   `num_mfma_per_subiter=0` (synthetic fixture). No carve-out is involved.

2d. Delete the entire `# Probe:` section of the test body (from `# Probe: neutralize the
   carve-out...` to the end of the function). Keep only the part (a) assertion:
   ```python
   failures_with_carveout = compare_graphs(g_default, g_cms)
   assert failures_with_carveout == [], (...)
   ```

2e. Remove the now-unused `GraphNode` import in the test if it was only imported for the
   monkeypatch. Check: `GraphNode` is imported at line 261 inside the test function
   (`from Tensile.Components.CMSValidator import GraphNode`). After deleting the probe block,
   this import is gone automatically (it was local to the test body). Verify the module-level
   imports still compile (the `OrderInvertedFailure` import at line 67 is used by the companion
   tests via `isinstance` checks — keep it).

2f. Update companion test docstrings to remove stale carve-out references (inline comments like
   "which is exactly what the carve-out site (CMSValidator.py:2584) gates on" at line 215-216).

### Step 3 — Verify no other test imports these names

```bash
grep -rn "carveout_real_kernel\|cross_subiter_alu_carveout" \
  /path/to/worktree/Tensile/Tests/unit/ | grep -v "test_cross_subiter_alu_carveout_real_kernel.py"
```

Expected: 0 matches. If any file imports from the carveout file, investigate before deleting.

### Step 4 — Run the suite

Run the full unit suite (excluding `test_MatrixInstructionConversion.py`) against the worktree's
Tensile code. Expected: the 3 (b)-class failures are gone; no new failures appear.

```bash
# From worktree, using a correctly-configured env:
python -m pytest Tensile/Tests/unit/ \
  --ignore=Tensile/Tests/unit/test_MatrixInstructionConversion.py \
  --ignore=Tensile/Tests/unit/Common \
  -q --tb=short 2>&1 | tail -40
```

Verify:
- No FAILED from `test_cross_subiter_alu_carveout_real_kernel.py`
- No FAILED from `test_cross_subiter_pack_artifact.py`
- Pre-existing nyb5 xfail remains
- u6nn failures unchanged

### Step 5 — Commit

Commit with message:
```
C4 (5ryl): delete/re-fixture (b)-class carveout tests

Delete test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures
(tests the deleted cross-graph exemption; obsolete post-C1/5tf9).

Re-fixture test_real_kernel_validates_clean_with_carveout_engaged: assertion
== [] still holds, now for principled reason (unrolled walk byte-key
cancellation); update docstring.

Re-fixture test_carveout_suppresses_artifact_and_neutralization_surfaces_it:
rename + remove monkeypatch probe block; probe targeted the deleted exemption,
which no longer exists in diagnose_missing_edge post-C1.

Update module docstrings in both files to remove stale carve-out framing.

Closes rocm-libraries-5ryl.
```

---

## §6 Validation — Expected Failure Delta

| Metric | Before C4 | After C4 |
|---|---|---|
| (b)-class FAILs | 3 | 0 |
| pre-existing nyb5 | 1 xfail(strict) | unchanged |
| pre-existing u6nn | tracked | unchanged |
| New failures introduced | — | 0 |
| Net delta | — | −3 |

The bead acceptance criterion "ALL tests pass" is met for the C-chain scope. Pre-existing beads
u6nn and nyb5 are not C4's scope and are not counted against this acceptance.

---

## §7 Risks / Open Questions

### 7.1 `_alu_cross_subiter_passthrough` GapRule (within-graph path)

The `_alu_cross_subiter_passthrough` GapRule at CMSValidator:731 is the companion to the deleted
cross-graph exemption. It governs the within-graph timing analysis path (`_classify_edge_coverage`)
and fires for `(ALU, MFMA_STANDARD|MFMA_4x4)` pairs when `p.subiter(nmps) != c.subiter(nmps)`.

The C1 plan §7.1 explicitly deferred this rule's deletion to after the unrolled walk landed,
noting "those would compound the RED state without serving the plan's goal." Post-C3, the rule
is still live. Its correctness under the unrolled walk is **not evaluated in C4** — the
within-graph timing analysis is a different path from the cross-graph comparison that C4 re-fixtures.

**Risk**: if the GapRule is also a hack (i.e., it suppresses real timing violations), it should
be deleted with its own bead and failure classification. Leaving it in means C4's "no exemptions"
acceptance criterion is not fully met.

**Recommendation**: Filed as `rocm-libraries-37d3` (priority 1, depends on 5ryl). The within-graph
GapRule is principled for the same reason the deleted cross-graph exemption was: cross-subiter
ALU→MFMA edges in the within-graph walk arise from the same last-writer-wins resolver artifact
(Pack[subiter N+1] wins the byte-key; MFMA[subiter N] reads it — no real timing dependency).
Requiring 2 wait cycles for a non-dependency would be wrong. However the unrolled walk's
source-identity / byte-key model may make the within-graph subiter check redundant, and that
question requires a separate audit against the production kernel. 37d3 tracks that audit; it is
NOT a C4 blocker but must be resolved before r62g closes.

### 7.2 Production test env mismatch

The main project's tox unit env (`projects/hipblaslt/tensilelite/.tox/unit/`) has a stale Tensile
tarball install (pre-C-chain). Tests in the worktree need to run against the worktree's Tensile
code. The correct invocation is either:
- Run `tox -e unit` from inside the worktree (requires worktree's tox env to be populated)
- OR run pytest directly with the worktree on `sys.path` ahead of site-packages

If the implementer cannot reproduce the correct env, the test run in Step 4 will show ERRORs
(collection failures) instead of FAILs for the (b)-class tests. In that case:
- The FAIL/PASS distinction can still be observed by running only the files that don't have
  env-dependent imports (but both (b)-class files import `OrderInvertedFailure`, so both need
  the correct env)
- Alternatively, manually verify the `compare_graphs` call behavior by reading the code path

### 7.3 Companion test position assertion

`test_artifact_present_in_default_graph` asserts `producer.position < consumer.position` at
line 217. Under the unrolled walk, `position` maps to `unrolled_position` (int). In the
synthetic fixture: PackA1 is at `unrolled_position=1` < MFMA at `unrolled_position=2` → True.
If the field name changed during C3 migration, this assertion could fail. Verify that
`GraphNode.position` still resolves to `unrolled_position` (or is aliased to it) before claiming
the companion tests need no changes.

---

## §8 New Beads to File

No (c)-class promotions found during this investigation. All three (b)-class tests have
post-C3 behavior that is correct and matches the principled model — no real bugs were being
suppressed.

**One follow-up bead filed** (not a C4 blocker): `rocm-libraries-37d3` — evaluate
`_alu_cross_subiter_passthrough` GapRule under the unrolled walk (§7.1). Filed during plan
verification per no-deferred-discoveries rule. Blocked on 5ryl; must resolve before r62g closes.

No `br dep add r62g` action required — no (c)-class bugs surfaced.

---

*Plan written: 2026-06-09. Author: investigation agent for bead rocm-libraries-5ryl.*
