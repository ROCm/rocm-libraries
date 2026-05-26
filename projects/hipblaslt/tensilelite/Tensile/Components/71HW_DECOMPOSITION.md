# 71hw decomposition — letter→bead mapping

Bead: `rocm-libraries-71hw` (P0 meta — "Implement Approach A: real-build
CMS-vs-default validator"). This memo replaces the lost
`/tmp/71HW_DECOMPOSITION.md` that the sub-bead descriptions still cite.

The sub-beads' `## Dependencies` sections refer to other sub-beads by
letter (A, B, C, D, E, F, G, H). The original memo defined A-G; letter
**H** was added 2026-05-26 per user directive. That original memo is gone.
The mapping below is recovered from each sub-bead's own description text,
and verified against the validator-branch code state at tip `c13eb73c1c1`
(post-gz0k + post-v01w + post-strict-annotation-cleanup).

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
| **H** | rocm-libraries-xj16 | Wire real-vs-real `compare_graphs` + `validate_edge_wait_coverage` inline assertion in the `_captureNonCmsBuild` block at `KernelWriter.py:5656+`; auto-activate `_captureNonCmsBuild` for every CMS kernel mirroring the shadow auto-activation at `:4966`. **Added per user directive 2026-05-26 (see "Re-sequencing" below).** |

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
| B | 3ija | closed | Triage complete 2026-05-13; residuals filed as `jmfp`, `p39d`, `zvzu`, `6hk3`. Runner deleted in `2bww`. |
| C | aixt | open (partial) | Most test sites migrated in commit `bc72aec71a2`. One test (`test_preloop_divergence_catches_useplrpack_change`) surfaced because the helper does not yet support post-`_initKernel` `UsePLRPack` forcing. |
| D | 78n3 | open | Not started. `cms_from_default.py:120,123` still calls shadow plumbing. Blocked by C. |
| E | czby | open | Not started. 8 `_captureDefaultSchedule` flag-check sites intact in `KernelWriter.py` (`:3890`, `:4572`, `:4674`, `:4776`, `:4803`, `:5190`, `:5290`, `:5348`, `:5506`). **Now blocked by H.** |
| F | mnzh | open | Not started. Must wait for E to delete flag-check sites or compilation breaks. |
| G | 9m05 | open | Not started. Marginal parallelism with C. |
| **H** | **xj16** | **open (new)** | Filed 2026-05-26 per user directive. Blocked by `jmfp` / `p39d` / `zvzu` / `6hk3` (their failures will likely fire under the new assertion). |

## Re-sequencing per user directive 2026-05-26

The user directed two things that change the original ordering:

1. *"When running a yaml we MUST assert of failures. We should always be erroring out of a cms schedule is being created and the validation doesn't pass."*
2. *"I want us to move off the shadow path. It must be real vs real. This must be the next thing we directly work towards."*

The originally-planned czby (E) would have deleted the shadow inline-assert
WITHOUT replacing it — the comment at `KernelWriter.py:5667` (nyb5 era)
said `compare_graphs` is driven OUTSIDE the build. That decision is now
reversed. Instead, real-vs-real assertion must move INTO the build,
running inline alongside (and eventually replacing) the shadow assertion.

**New letter H = xj16** captures the missing first step: wire the
inline real-vs-real assertion at `KernelWriter.py:5656+` and
auto-activate `_captureNonCmsBuild` for every CMS kernel, mirroring the
shadow auto-activation at `:4966`. Both inline-assert paths run in
parallel during the transition. Only after xj16 lands AND real-YAML
smoke passes 100% under the new assertion may czby (E) delete the
shadow path.

`jmfp` / `p39d` / `zvzu` / `6hk3` (originally P1 ex-3ija residual
sub-beads) are bumped to **P0** because their failure shapes will
almost certainly surface under H's new assertion and block its
acceptance.

## Dependency chain

```
nyb5 (A) ─── done ───┐
                     ├──► 3ija (B) ─── done ───► aixt (C) ──► 78n3 (D) ──► czby (E) ──► mnzh (F)
                     │                                │                       ▲
                     │                                │                       │
                     │                                └──► 9m05 (G)           │
                     │                                                        │
                     └──► {jmfp, p39d, zvzu, 6hk3} ────► xj16 (H) ────────────┘
```

**Critical path now: `{jmfp, p39d, zvzu, 6hk3} → xj16 (H) → czby (E) → 78n3 (D) → mnzh (F)`**.

H is the new bottleneck. Until xj16 lands, the original shadow-deletion
chain (E → F) is frozen. The ex-3ija classifier/divergence fixes can
proceed in parallel (they fan out — each is independent of the others).

`9m05` (G) is strictly blocked on aixt, but the accessor retirement
touches different sites than the test-migration work, so it could run
in parallel with C if a separate agent picks it up.

`aixt` (C) is unblocked and remains startable — its one surfaced
test can be resolved independently of the H chain.
