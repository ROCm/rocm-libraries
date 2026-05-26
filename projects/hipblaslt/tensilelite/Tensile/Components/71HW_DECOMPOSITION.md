# 71hw decomposition — letter→bead mapping

Bead: `rocm-libraries-71hw` (P0 meta — "Implement Approach A: real-build
CMS-vs-default validator"). This memo replaces the lost
`/tmp/71HW_DECOMPOSITION.md` that the sub-bead descriptions still cite.

The sub-beads' `## Dependencies` sections refer to other sub-beads by
letter (A, B, C, D, E, F, G). The original memo defined those letters;
that memo is gone. The mapping below is recovered from each sub-bead's
own description text, and verified against the validator-branch code
state at tip `44a17947c99` (post-gz0k + post-v01w).

## Mapping

| Letter | Bead | Scope |
|---|---|---|
| **A** | rocm-libraries-nyb5 | Real non-CMS Build #2 reference pipeline (`build_non_cms_reference`). See `approach_a.py:70`, `KernelWriter.py:526`. |
| **B** | rocm-libraries-3ija | Triage `compare_graphs` / wait-coverage residuals on real Build #2 across the gfx950 CMS test surface. See `3IJA_RESIDUAL_TRIAGE.md`. |
| **C** | rocm-libraries-aixt | Migrate prologue-capture / ScheduleCapture / dataflow-emission tests to the two-build setup. |
| **D** | rocm-libraries-78n3 | Migrate `cms_from_default` Phase 2 off `enable_capture_default_schedule` (default capture via second build, not shadow). |
| **E** | rocm-libraries-czby | Repoint `kernelBody` FourPartCapture assembly at real Build #2; delete shadow `default_main` / prologue feed sites. |
| **F** | rocm-libraries-mnzh | Remove `_captureDefaultSchedule` auto-activation + `enable_capture_default_schedule[_no_assert]` public hooks + dead writer-state fields. |
| **G** | rocm-libraries-9m05 | Retire `_last_default_capture` / `_last_cms_capture` writer-property accessors after all callers are migrated. |

## Recovery method

Every dependent bead's `## Dependencies` clause names its blockers by
letter and includes a one-phrase gloss (e.g. `aixt`: *"Blocked by
sub-bead A (the two-build helper depends on the real Build #2 path
existing) and sub-bead B (B's triage may surface tests whose pinned
property is wrong)"*). Cross-referencing the glosses with the visible
sub-bead titles produces a unique mapping with no remaining ambiguity.

Self-identification (the trailing `See /tmp/71HW_DECOMPOSITION.md for
the full decomposition memo (sub-bead X)` line that each bead's
description ends with) was used to verify the letter for each filed
bead.

## Status snapshot (2026-05-26)

| Letter | Bead | Status | Notes |
|---|---|---|---|
| A | nyb5 | closed | Landed in commit `59bb5a8e9f6`. 1027 tests pass. |
| B | 3ija | stale-in_progress → closing | Triage complete 2026-05-13; residuals filed as `jmfp`, `p39d`, `zvzu`, `6hk3`. Runner deleted in `2bww`. |
| C | aixt | open (partial) | Most test sites migrated in commit `bc72aec71a2`. One test (`test_preloop_divergence_catches_useplrpack_change`) surfaced because the helper does not yet support post-`_initKernel` `UsePLRPack` forcing. |
| D | 78n3 | open | Not started. `cms_from_default.py:120,123` still calls shadow plumbing. |
| E | czby | open | Not started. 8 `_captureDefaultSchedule` flag-check sites intact in `KernelWriter.py` (`:3890`, `:4572`, `:4674`, `:4776`, `:4803`, `:5190`, `:5290`, `:5348`, `:5506`). |
| F | mnzh | open | Not started. Must wait for E to delete flag-check sites or compilation breaks. |
| G | 9m05 | open | Not started. Marginal parallelism with C. |

## Dependency chain

```
nyb5 (A) ─── done ───┐
                     ├──► 3ija (B) ─── done ───► aixt (C) ──► 78n3 (D) ──► czby (E) ──► mnzh (F)
                     │                                │
                     └────────────────────────────────┴──► 9m05 (G)
```

Critical path now: **aixt → 78n3 → czby → mnzh**. Each is single-threaded
behind the previous because the shadow-deletion has to cascade in a
fixed order: kill the public hooks' last non-test caller (D), then the
shadow producers (E), then the public hooks themselves (F).

`9m05` (G) is strictly blocked on aixt, but the accessor retirement
touches different sites than the test-migration work, so it could run
in parallel with C if a separate agent picks it up.
