# dm4p Plan-Adherence Verification

## Verdict

**CLEAN** — with one filing addendum (w5xw filed by verifier per standing rule).

The Phase 2 swap is minimal, the BPG#11 build is end-to-end green, the lock-in test is load-bearing and source-level, the discovery the agent surfaced (u6nn) is properly filed and linked, the design doc is updated, and the broader pytest suite shows dm4p strictly improves the baseline (no regressions). One additional pre-existing failure pair was discovered during verification and has been filed as w5xw (P0, blocks r62g). No backwards-compat bloat introduced.

## Per-Q

### Q1 — Swap minimal?

**PASS.** Diff in `Tensile/KernelWriter.py:6376-6411` is exactly one-line of code change plus comment:
- Removed: `from Tensile.Components.CustomSchedule.approach_a import (build_non_cms_reference,)` import, the `isaInfoMap = getattr(...)` lookup, the `AssertionError` on missing `isaInfoMap`, and the `ctx.default = build_non_cms_reference(kernel, self.assembler, isaInfoMap)` call.
- Added: `ctx.default = self._last_default_capture` (KernelWriter.py:6399).
- Kept (per design v5 Phase 2 scope): the `is_cms_callsite` predicate (KernelWriter.py:6375), the `else` branch (shape b, KernelWriter.py:6400-6435), `_captureNonCmsBuild` flag.
- No new feature flags, no new kwargs, no shims, no setdefault, no test exclusions.

### Q2 — BPG#11 build works end-to-end?

**PASS.** Ran from worktree:
```
PYTHONPATH=$WT /home/alvasile/venv/bin/python3 Tensile/bin/Tensile \
  Tensile/Components/CustomSchedule/gfx950/test_yamls/6hk3_tf32_128x160x64_tn.yaml \
  /tmp/dm4p_verify --build-only --gpu-targets gfx950
```
- Exit code 0; build completes through `00_Final: End - 5.820s`.
- 2 / 2 solutions survive KernelWriter (`# Actual Solutions: 2 / 2 after KernelWriter`).
- No `isaInfoMap`-related errors.
- No `CaptureCategoryMissingError`, `CaptureConsistencyError`, or validator-class errors.
- The xj16 inline `compare_graphs` + `validate_edge_wait_coverage` assertions sit inside this build path and would have surfaced on failure; build clean implies they passed.

### Q3 — Lock-in test load-bearing?

**PASS.** `Tensile/Tests/unit/test_dm4p_shadow_as_canonical_reference.py:97-153` (`test_xj16_cms_callsite_consumes_shadow_capture`) does source-level inspection of `KernelWriter.kernelBody`:
- Locates the `if is_cms_callsite: ... else:` branch via regex.
- Strips line comments (`:128-130`) so narrative prose can mention `build_non_cms_reference` without false-positiving.
- Asserts `self._last_default_capture` is present (`:132-139`).
- Asserts `build_non_cms_reference` is absent (`:141-153`).
- Would fail loud if a future change restored Approach A here.

Includes the cwd-trap guard at `:62-82` (replicated from `test_capture_pipeline_checks.py:54-85`), so a stray `pytest` from a sibling tree won't silently assert against the wrong source.

A second test (`test_capture_context_default_survives_reset`, `:156-194`) pins the supporting invariant that `CaptureContext.reset()` preserves `default` and `cms` (otherwise the Phase 2 swap would silently propagate `None`).

Source-level rather than build-driven — appropriate because (a) the end-to-end build already exercises the path (Q2), (b) this test stays green in unit-only environments without rocisa/assembler.

### Q4 — Discovery handled per rule?

**PASS (with verifier addendum).**

Agent's filing `rocm-libraries-u6nn` (P0, bug):
- Exists, P0, correctly classifies `test_whole_kernel_cms_prologue_matches_non_cms_reference` failure as pre-existing.
- `br dep add` correctly created — u6nn `blocks` u89e (`br show rocm-libraries-u89e` shows u6nn in dependencies).
- Verified pre-existing by checking out `d606518ab21 -- Tensile/KernelWriter.py` and re-running pytest: reproduces `RuntimeError: build_non_cms_reference: Solution has no pre-CMS snapshot`. Agent's claim accurate.

**Verifier-filed addendum** (`rocm-libraries-w5xw`, P0, bug, blocks r62g):
- Two additional pre-existing failures in `test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_render_counts_match` and `::test_real_kernel_per_ordinal_logical_instruction_matches` (~642 mismatches each in ML body, SHADOW-vs-CMS render-count divergence).
- These were NOT in the agent's self-report. They reproduce at parent commit `d606518ab21`, so they pre-exist (not dm4p regressions).
- However, the standing rule requires filing such discoveries as P0 blockers on the next dependent bead. The natural blocker is `r62g` (Phase 3 hard go/no-go gate on the SHADOW-vs-CMS test surface) — exactly where these failures must be triaged before Phase 4 (u89e) can retire Approach A.
- Filed and linked.

### Q5 — No backwards-compat bloat?

**PASS.** Full diff scan (`git diff d606518ab21..HEAD`):
- No new feature flags, no new kwargs, no new shims, no new setdefault.
- No new test exclusions or `@pytest.mark.skip`.
- No new "defensive" classifications.
- Only "until Phase 4" reference in the swap comment is fully scoped to `rocm-libraries-u89e` (a real, open, properly-linked bead) — not a vague punt.
- Design doc update at `DEFAULT_SCHEDULER_REFERENCE_DESIGN.md:121-141` explicitly calls out "Scaffolding deferred to Phase 4" with the bead reference — appropriate hand-off, not punted work.
- The single line `ctx.default = self._last_default_capture` plus narrative comment block is the entirety of the executable change.

### Q6 — Baseline match?

**PASS (better than expected).**

Parent baseline (re-verified by checking out `d606518ab21 -- Tensile/KernelWriter.py` and deleting the new lock-in test):
```
19 failed, 1040 passed, 4 skipped, 2 xfailed, 2 errors
```
(matches verifier-supplied expectation exactly)

dm4p worktree (HEAD):
```
3 failed, 1060 passed, 4 skipped, 2 xfailed
```

Delta: −16 failed, −2 errors, +20 passed (= +2 lock-in tests + 18 tests that were failing/erroring on parent and now pass under dm4p).

The fixed tests are legitimately repaired by dm4p (not by suppression): they exercise the SHADOW-vs-Approach-A capture pipeline that Phase 2 is designed to fix. Examples now passing:
- `test_cms_from_default.py::TestCli::test_cli_runs_end_to_end`
- `test_cms_from_default.py::TestSoftFailDocstringWarning::test_warning_in_docstring`
- `test_prologue_capture.py::test_preloop_divergence_catches_useplrpack_change`
- `test_dataflow_graph_emission_ordinal.py::test_example_yaml_no_spurious_order_inverted_failures`
- `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_validates_clean_with_carveout_engaged`
- `test_cross_subiter_alu_carveout_real_kernel.py::test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`

All 3 remaining failures pre-exist at parent:
- `test_prologue_capture.py::test_whole_kernel_cms_prologue_matches_non_cms_reference` → u6nn (filed by agent)
- `test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_render_counts_match` → w5xw (filed by verifier)
- `test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_ordinal_logical_instruction_matches` → w5xw (same bead)

The verifier-supplied prediction of "1042 passed" was conservative; actual is 1060 because dm4p incidentally repairs cascade failures from Approach A's `build_non_cms_reference` brittleness. This is the explicit design intent (design v5 §4 Phase 2).

### Q7 — Design doc updated?

**PASS.** `Tensile/Components/DEFAULT_SCHEDULER_REFERENCE_DESIGN.md:124-141`:
- Section header changed from "Phase 2 — Wire SHADOW as `ctx.default`" to "Phase 2 — Wire SHADOW as `ctx.default` — **LANDED** (`rocm-libraries-dm4p`)".
- Reproducer site updated (no longer references the stale `KernelWriter.py:5772` line number; refers to `kernelBody`'s `_captureNonCmsBuild` block).
- Explicit acceptance note dated 2026-06-01 confirms BPG#11 builds end-to-end and the xj16 assertions pass.
- New paragraph "Scaffolding deferred to Phase 4" explicitly hands off the `is_cms_callsite` predicate and `else` branch retirement to `rocm-libraries-u89e`.
- No premature "DONE" claim — accurately reflects landing status of Phase 2 specifically.

## Concerns

- **Concern 1 (filed, not blocking dm4p):** Implementation agent's self-report omitted the two `test_dataflow_graph_emission_ordinal.py` failures. They are genuinely pre-existing, but per the standing rule they had to be filed. Verifier filed `rocm-libraries-w5xw` (P0, blocks r62g). Bead `rocm-libraries-w5xw`. No blocker on dm4p completion; blocker on Phase 3 (r62g) and transitively Phase 4 (u89e), which is the appropriate level.

- **Concern 2 (minor, not blocking):** The Phase 2 swap comment at `KernelWriter.py:6391-6398` claims `build_non_cms_reference` "remains importable for tests (Phase 4 ... will retire it)." Verified true — `approach_a.py` still exists and the helper is still imported by `test_prologue_capture.py:366` (which is itself the u6nn failure). No action; this is the expected scaffolding posture during Phase 2.

## Recommendation

**CLEAN → proceed to correctness verifier** (task #305).

The implementation is minimal, principled, end-to-end-green, doc-updated, and ships net-positive on the unit test baseline. The one discovery the agent missed has been filed by the verifier per the standing rule and is properly linked as a blocker on the appropriate next-phase bead (r62g). No code changes, no rollbacks, no fixups needed before correctness verification.
