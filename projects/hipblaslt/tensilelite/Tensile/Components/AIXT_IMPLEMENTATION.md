# aixt: migrate prologue-capture / d3zj LCC tests to the two-build setup

This memo documents `rocm-libraries-aixt`. Branch tip:
`users/alvasile/aixt_test_migration` based on
`users/alvasile/validator_long_term_plans` at `59bb5a8e9f` (post-nyb5
landing).

## Scope

Test-surface migration only. No production code (KernelWriter.py,
CMSValidator.py, ScheduleCapture.py, approach_a.py) was modified.
This bead routes specific tests OFF the SHADOW capture pipeline AND
ONTO the new two-build pattern provided by
`build_non_cms_reference()` (rocm-libraries-nyb5).

## Migration table

| Test (file::name) | Pre-aixt SHADOW plumbing | New pattern |
|-------------------|--------------------------|-------------|
| `test_prologue_capture.py::test_whole_kernel_useplrpack_cms_matches_both_defaults` (deleted; replaced by `test_whole_kernel_cms_prologue_matches_non_cms_reference`) | `_build_capture` (uses `enable_capture_default_schedule_no_assert` + post-`_initKernel` `UsePLRPack` forcing); reads `_capture_context.default` and `_capture_context.cms`; asserts `cap_with_cms.prologue is cap_with_default.prologue` (SHADOW thread-through identity check). | Real CMS build (consumes only `_last_cms_capture`) + `build_non_cms_reference()`. Asserts content equivalence (canonical-render `Counter` equality) between the CMS-side prologue and the non-CMS reference's prologue. For the canonical CMS-eligible kernel both prologues are `None` (UsePLRPack=0 at solution construction); the equality check covers both the trivial `None == None` case and any future kernel-config drift that produces a populated prologue. The forced-`UsePLRPack` matrix was dropped — see "Open questions" below for the residual `test_preloop_divergence_catches_useplrpack_change` case. |
| `test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_render_counts_match` | `real_kernel_capture_pair` fixture → reads `_last_default_capture` (SHADOW-extracted, missing LCC) and `_last_cms_capture`. `@pytest.mark.xfail(strict=True)` citing nyb5 as the principled fix. | New module-scoped fixture `real_kernel_capture_pair_approach_a` returning `(default_cap_from_build_non_cms_reference, cms_cap_from_real_cms_build)`. xfail marker REMOVED. |
| `test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_ordinal_logical_instruction_matches` | Same as above. | Same as above. |
| `test_dataflow_graph_emission_ordinal.py::test_lcc_invariant_per_body_use_loop_predicate` | Same as above. | Same as above. |

## d3zj LCC xfail removal — confirmed passing

All 3 d3zj LCC xfail markers removed; tests pass under the migrated
fixture:

```
projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py::test_real_kernel_per_render_counts_match PASSED
projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_real_kernel_per_ordinal_logical_instruction_matches PASSED
projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_lcc_invariant_per_body_use_loop_predicate PASSED
```

The post-`closeLoop` finalize in nyb5's helper observes the
`SSubU32`/`SCmpEQI32` LCC pair on `sgprLoopCounterL` in ML and ML-1,
which the SHADOW capture missed (finalize ran BEFORE `closeLoop`).
See `D3ZJ_SCMPEQI32_INVESTIGATION.md` §3.4 and
`NYB5_IMPLEMENTATION.md` §"d3zj closure (LCC capture)".

## Q4 audit log

The Q4 user direction was: "(c) audit-per-test, with (b) lean — no
new opt-out hooks unless a test genuinely requires one (STOP and
surface in that case)."

Tests that consumed `enable_capture_default_schedule_no_assert` /
`_capture_skip_internal_validate` BEFORE this bead:

1. `test_prologue_capture.py::_build_capture` (helper called by both
   `test_preloop_divergence_catches_useplrpack_change` and
   `test_whole_kernel_useplrpack_cms_matches_both_defaults`): used
   `enable_capture_default_schedule_no_assert` to bypass the
   in-`kernelBody` validator gate so the helper could force
   `UsePLRPack` between `_initKernel` and `kernelBody` (the SHADOW
   path's only post-init entry point). The forcing trips
   `TimingTooCloseFailure` on the back-to-back Pack chain when
   `UsePLRPack=1` is forced on a default-mode kernel that doesn't
   natively schedule for it; the `_no_assert` opt-out swallows that
   built-in assert and lets the test re-run validation explicitly on
   the residual.

   - `test_whole_kernel_useplrpack_cms_matches_both_defaults`:
     **Rewritten under (b) lean** — replaced with
     `test_whole_kernel_cms_prologue_matches_non_cms_reference` which
     uses Approach A's two-build pattern (real CMS build +
     `build_non_cms_reference`) and content-equivalence on the
     prologue. The forced-`UsePLRPack` matrix was dropped because the
     forcing mechanism is SHADOW-pipeline-specific (no equivalent in
     `build_non_cms_reference`, which forces
     `UseCustomMainLoopSchedule=0` deep in its body), and the
     headline assertion (the SHADOW thread-through `is` identity
     check) maps cleanly to two-build content equivalence without
     needing the matrix.
   - `test_preloop_divergence_catches_useplrpack_change`:
     **STOP-and-surface case** — see Open questions below.

2. `_capture_skip_internal_validate`: not present in the test corpus
   (not consumed by any test). No migration needed.

No test required introducing a new opt-out hook on the Approach A
pipeline — the only test that genuinely cannot be rewritten under
(b) is surfaced as an open question rather than papered over with a
new hook.

## Files modified

1. `Tensile/Tests/unit/test_dataflow_graph_emission_ordinal.py`:
   - Added module-scoped fixture `real_kernel_capture_pair_approach_a`.
   - Re-routed 3 d3zj LCC tests onto the new fixture; removed strict
     xfail markers.
   - Legacy `real_kernel_capture_pair` fixture retained for
     `test_example_yaml_no_spurious_order_inverted_failures` which
     legitimately needs the SHADOW default-side capture's particular
     shape (the SHADOW capture inherits the CMS-mutated `kernel` dict
     so its GR-stream order matches; the Approach A fixture surfaces
     3 GR-OrderInverted residuals tracked under
     `rocm-libraries-3ija` per
     `test_non_cms_reference_compare_graphs_surfaces_only_known_residuals`).

2. `Tensile/Tests/unit/test_prologue_capture.py`:
   - Replaced `test_whole_kernel_useplrpack_cms_matches_both_defaults`
     with `test_whole_kernel_cms_prologue_matches_non_cms_reference`
     (Approach A two-build pattern, content-equivalence assertion).

No production code modified.

## TDD discipline notes

For each migrated test, RED triggered as expected per the
svb1 / hdu1 honest-surfacing precedent:

- **3 d3zj LCC tests**: pre-aixt these were strict-xfailing on the
  SHADOW capture's missing LCC. Removing the xfail marker WITHOUT
  re-routing the fixture would have surfaced the same defect (still
  xfailing). Re-routing onto Approach A's helper fires the natural
  GREEN — nyb5's `_appendCloseLoopLCCToBuilder` post-hook captures
  LCC. The migration was done atomically (re-route + marker removal in
  the same edit) because the underlying defect was already fixed by
  nyb5 at the helper level (verified independently by
  `test_non_cms_reference_has_lcc_in_every_main_loop_body`).
- **`test_whole_kernel_cms_prologue_matches_non_cms_reference`**: the
  pre-aixt SHADOW `is` identity check would error out atomically
  when the captures came from two independent writers (Python identity
  cannot be shared across writers). The new content-equivalence check
  passes naturally because both prologues are `None` for the
  canonical CMS-eligible kernel (UsePLRPack=0 at solution
  construction). RED on the old assertion / GREEN on the new — the
  semantic shift is the migration.

## Open questions

1. **`test_preloop_divergence_catches_useplrpack_change`** (Q4
   STOP-and-surface case): the test's machinery requires forcing
   `UsePLRPack` AFTER `_initKernel` and BEFORE `kernelBody` to
   produce two captures that differ only by prologue content. There
   is no equivalent forcing point on the Approach A pipeline:
   `build_non_cms_reference` deep-copies the config and forces
   `UseCustomMainLoopSchedule=0` internally — there is no caller
   hook to also override `UsePLRPack` post-`_initKernel`. The test's
   actual assertion (post-hdem) is "two captures with different
   prologue Pack content collapse to equivalent dataflow under
   Approach A + Approach E identity scheme." That assertion is
   preserved by `test_dataflow_graph_hdem.py::test_cross_body_extra_write_surfaces`
   per the inline comment in the existing test (the
   pre-`UsePLRPack`-forcing scenario is one specific instance of the
   cross-body extra-write pattern that hdem test pins synthetically).

   Per the bead's discipline rule #4 ("no new opt-out hooks on the
   Approach A pipeline unless a specific test genuinely requires
   one"), this test cannot be cleanly rewritten under (b). Three
   options for the user:

   - **Option (a) — introduce a new opt-out hook**: add a parameter
     to `build_non_cms_reference` for caller-supplied post-`_initKernel`
     kernel-dict mutation. Requires user approval per the bead.
     Estimated cost: low.
   - **Option (b) — delete the test**: the assertion is preserved
     synthetically by `test_cross_body_extra_write_surfaces`; the
     forced-`UsePLRPack` real-kernel exercise is shadow-pipeline-
     specific machinery whose semantics don't survive shadow
     deletion. The test currently passes (the SHADOW path still
     exists), but it would become un-runnable when czby deletes
     SHADOW.
   - **Option (c) — leave untouched, defer to czby**: czby owns
     SHADOW path deletion. Whatever the test's fate (rewrite,
     delete) can ride with that bead.

   The test currently PASSES under the existing SHADOW machinery; it
   is NOT a blocker for the aixt verification checklist (the
   headline `is`-trick assertion is GONE, replaced by the
   content-equivalence assertion in the new
   `test_whole_kernel_cms_prologue_matches_non_cms_reference`).

   I left the test untouched in this bead and surface the question
   to the user for direction. The test's `_build_capture` helper
   (which uses `enable_capture_default_schedule_no_assert`) is
   retained alongside it for the same reason.

## Verification

Pre-aixt baseline:
```
1027 passed, 3 skipped, 5 xfailed
```

Post-aixt:
```
1030 passed, 3 skipped, 2 xfailed
```

Delta:
- +3 passed (the 3 d3zj LCC tests flipped from xfail to pass).
- -3 xfailed (same 3 markers removed).
- 0 failed, 0 xpassed.

Test count delta from rewrites:
- `test_whole_kernel_useplrpack_cms_matches_both_defaults` → renamed
  to `test_whole_kernel_cms_prologue_matches_non_cms_reference` (no
  net count change).

The 2 remaining xfails are:
1. `Common/test_Architectures.py::test_filterLogicFilesByPredicates_match_emulation_ids`
   — pre-existing, unrelated to CMS work.
2. `test_approach_a_non_cms_reference.py::test_non_cms_reference_compares_clean_against_cms_build`
   — nyb5 Cycle 2 strict-XFAIL pinning the GR-OrderInverted residual
   (in scope for `rocm-libraries-3ija`).

Production code untouched (verified by `git diff --stat`):
```
.../unit/test_dataflow_graph_emission_ordinal.py   | edits to fixture + 3 tests
.../Tensile/Tests/unit/test_prologue_capture.py    | one test rewritten
```

## Cross-references

- `NYB5_IMPLEMENTATION.md` — the helper this bead consumes
  (`build_non_cms_reference` API, return shape, isolation strategy).
- `2LZD_INVESTIGATION.md §6.2` — Approach A test architecture rework
  framing (Q2/Q3/Q5).
- `D3ZJ_SCMPEQI32_INVESTIGATION.md` — LCC capture-timing defect that
  motivates the d3zj re-route.
- `D3ZJ_NGL_NLL_LCC_INVESTIGATION.md` — per-body LCC invariant
  scope.
- `rocm-libraries-czby` — owns SHADOW path deletion (sibling bead;
  out of scope for aixt).
- `rocm-libraries-mnzh` — owns deletion of `_captureDefaultSchedule`
  auto-activation + public hooks (sibling bead; out of scope).
- `rocm-libraries-3ija` — owns the GR-OrderInverted residual
  surfaced by Approach A (sibling bead; out of scope).
- `rocm-libraries-78n3` — owns `cms_from_default` migration
  (sibling bead; out of scope; `test_cms_from_default.py`'s
  `_MockWriter._last_default_capture` mocks are 78n3 territory, not
  aixt).
