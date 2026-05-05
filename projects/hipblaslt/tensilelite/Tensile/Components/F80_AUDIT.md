# Bead `f80` — Inline-comment audit

Scope: inline `#` comments **inside function bodies** in Python files
touched on `users/alvasile/validator_long_term_plans` since branching from
`develop`. Docstrings, decorative module-level headers, bead-mention
comments (handled by `ont`), and signature-line comments (handled by `fnf`)
are out of scope.

## Summary counts

- DELETE: 2
- KEEP: ~440 (audited; not enumerated individually below — all KEEP entries
  fall under the documented categories: signposts a non-trivial block;
  explains a constraint, hazard, contract, or non-obvious tradeoff;
  archives a removed-feature pointer; or is a bead-mention SKIP)
- SKIP-bead-ref: handled by sibling bead `ont`; not touched here

## DELETE — landed as `[f80]`-prefixed commit

| File:line | Verdict | Reasoning |
|---|---|---|
| `Tensile/Components/CMSValidator.py:1565` | DELETE | `# Set mfma_reorder from schedule info` followed by `context.mfma_reorder = scheduleInfo.mfmaReorder or []` — pure restatement; the variable name and assignment expression already say everything the comment says. |
| `Tensile/Components/CMSValidator.py:1647` | DELETE | `# All rules passed, considered valid.` followed by `return True, ""` — the success-path return tuple is self-explanatory at the end of `isValid`; the comment restates the implicit semantic of `True`. |

## KEEP — representative judgement-call examples

These are the comments where deletion was considered but rejected — useful
to understand the audit's KEEP boundary.

| File:line | Verdict | Reasoning |
|---|---|---|
| `Tensile/Components/CMSValidator.py:1679` | KEEP | `# Save original values across all code paths.` — labels the first phase of a 3-phase mutate-test-restore loop. Removing it breaks the symmetric "Save / Mutate / Restore" navigational triplet at L1679/L1686/L1694. |
| `Tensile/Components/CMSValidator.py:1686` | KEEP | `# Mutate in-place.` — restates "in-place" but adds the load-bearing constraint that the schedule is being mutated rather than copied. Pairs with L1679/L1694. |
| `Tensile/Components/CMSValidator.py:1694` | KEEP | `# Restore original values.` — closing pair to "Save". |
| `Tensile/Components/CMSValidator.py:1698` | KEEP | `# Merge adjacent indices into ranges.` — labels a non-trivial 7-line algorithm phase. |
| `Tensile/Components/CMSValidator.py:461` | KEEP | `# Check needed_by constraint (GR must finish before LR1/3)` — adds the semantic ("GR must finish before LR1/3") not visible in the function name. |
| `Tensile/Components/CMSValidator.py:487-508` | KEEP | Numbered rule-case markers (`# 1. No SWait`, `# 2. No Barrier`, ...) — navigational + tag the validation-rule numbering. |
| `Tensile/Components/CMSValidator.py:514` | KEEP | `# Defensive fallback — string return; should not fire on valid logic.` — explains WHY a redundant-looking branch exists. |
| `Tensile/Components/CMSValidator.py:792` | KEEP | `# MFMAInstruction: destination is the accumulator` — explains why `.acc` is checked first; the priority is non-obvious. |
| `Tensile/Components/CMSValidator.py:1577` | KEEP | `# === Structural rules (no Timeline needed) ===` — section-marker inside a long function. |
| `Tensile/Components/ScheduleCapture.py:2741-2743` | KEEP | `# Phase 1 — node construction + sidecar.` (with bracketing dashes) — phase markers in a 100+ line function. |
| `Tensile/Components/ScheduleCapture.py:2788-2789` | KEEP | `# Skip when nothing was captured — e.g., the no-op build_dataflow_graph(None) contract holds...` — explains the early-out's contract. |
| `Tensile/Components/CustomSchedule.py:5745` (and many sibling lines in schedule data tables) | KEEP | `# LW: prev GR VGPRs → alt LDS buffer (3+4=7 writes)` — labels each `optSchedule` dict entry with semantic meaning (instruction class + register-flow direction + count). The next "code" line is just `'LWA': [[-1, 0, 1]]`; without the comment a reader has no way to know what the indices mean. ~140 such comments live in the new schedule additions; all KEEP. |
| `Tensile/Tests/unit/test_dataflow_graph_register_gaps.py` (whole file, 344 inline comments) | KEEP | Every comment either explains a test-scenario design decision, a domain invariant being pinned, or an arithmetic derivation (e.g. `# nk0 cycle-exact: producer at 0 with mfma_free=4; consumer max(1,4)=4 → gap=3`). All KEEP. |
| `Tensile/Tests/unit/test_dataflow_graph_comparison.py:296` | KEEP | `# SBarrier deleted` — tells the reader the structural difference between the `subj_cap` block and the `ref_cap` block above. Without this comment a reader has to manually diff the two captures to see what changed. |
| `Tensile/Tests/unit/test_capture_pipeline_checks.py:182` | KEEP | `# Should not raise.` — for negative-assertion tests, this comment names the implicit assertion (the `assert` line below is incidental; the real test is that the call before it doesn't raise). |
| `Tensile/Tests/unit/dataflow_fixtures.py:71-72`, etc. | KEEP | Field-level comments on dataclass fields (e.g. `dst: RegisterContainer       # vgpr range written`) — these add semantic role beyond the type. |
| `Tensile/Tests/unit/conftest.py:49-51` | KEEP | `# Note: do NOT call assignGlobalParameters here — it mutates global state and breaks test_validateParameterTypes.` — explains a constraint. |

## SKIP — bead-reference comments (out of scope, owned by `ont`)

Many comments mention beads (`8nz`, `ola.2`, `ola.3`, `ola.4`, `or9`, `35z`,
`e7w`, `4tw`, `mrj.1/2/4`, `wx9`, `nk0`, `2bu.3/4/5`, `vf4`, `arv`, `dpi`,
`ckj`, `cmw`, `s7k`, `d6e`, `pcz`, `hof`, `yh0`, `cpe`, ...). They are
SKIPPED here per the bead `f80` charter; bead `ont` owns them.

## Misleading comments

None found. No separate beads filed.

## Verification

- Worktree: `/home/alvasile/rocm-libraries/projects/hipblaslt/tensilelite/users/alvasile/.wt-f80`
- `python -m pytest Tensile/Tests/unit/` — 583 passed, 2 skipped, 1 xfailed
  (clean run after the deletions; identical to the pre-edit baseline).

## Files touched

- `Tensile/Components/CMSValidator.py` (2 inline comments removed)

## Files audited but not modified

- `Tensile/Components/CMSValidator.py` (besides the 2 deletes)
- `Tensile/Components/CustomSchedule.py`
- `Tensile/Components/ScheduleCapture.py`
- `Tensile/Tests/unit/cms_test_utils.py`
- `Tensile/Tests/unit/cms_validation_base.py`
- `Tensile/Tests/unit/conftest.py`
- `Tensile/Tests/unit/dataflow_fixtures.py`
- `Tensile/Tests/unit/graph_native_validation_base.py`
- All `Tensile/Tests/unit/test_*.py` files in the diff (~30 files)
