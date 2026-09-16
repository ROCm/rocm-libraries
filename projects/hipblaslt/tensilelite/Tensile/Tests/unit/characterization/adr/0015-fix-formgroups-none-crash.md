# ADR 0015: Fix the `formGroups("None")` crash and flip ADR 0010's pinned golden

Status:  Accepted
Defect:  [AIHPBLAS-4409](https://amd-hub.atlassian.net/browse/AIHPBLAS-4409)

## Context
[ADR 0010](0010-pin-formgroups-none-crash.md) pinned a real crash rather than
fixing it, because the characterization suite is add-only with respect to
production source: `formForkParams(sol, skipMI=True)` — or any solution with
`EnableMatrixInstruction` falsy — assigned the *string* `"None"` and passed it
to `formGroups`, whose `.items()` call raised `AttributeError`. The whole
`--skipMI` CLI path was unusable. ADR 0010 recorded the exit condition
explicitly: "when that lands, flip this golden and supersede this ADR."

PR #11538 fixes it as a side effect of rewriting the extraction path. The
MI-disabled branch now builds a real group from the solution's `WorkGroup`,
which `formGroups` is the only emitter for (the fork loop above it skips every
key in `SOLUTION_KEYS_IN_GROUPS`).

## Decision
Flip the pin. `test_form_fork_params_skip_mi_raises`, which asserted the
`AttributeError`, is replaced by `test_form_fork_params_skip_mi_emits_workgroup`,
which asserts the emitted `{"WorkGroup": [...]}` group and that no
`MatrixInstruction` key appears. A companion test,
`test_form_fork_params_skip_mi_without_workgroup_raises`, pins the `KeyError`
that a solution carrying no `WorkGroup` still raises, so the fix is not
mistaken for making that case work.

This is the appended ADR that ADR 0010's Status now points at; ADR 0010's own
Context/Decision/Consequences prose is untouched, per the append-only rule.

## Consequences
- `--skipMI` and MI-disabled solutions now produce a usable config instead of
  crashing, so the flag is exercisable for the first time.
- The suite no longer documents a latent bug here; D14's entry in `DECISIONS.md`
  stays as the historical record and D26 records the flip.
- A solution dict with neither `MatrixInstruction` nor `WorkGroup` still fails,
  now with `KeyError` rather than `AttributeError`. That is a genuinely
  unusable input, and the companion test pins it so the difference is recorded
  rather than discovered.

**Rejected alternatives:**
- Leave the `AttributeError` pin in place and mark the test xfail — rejected:
  the crash no longer happens, so the golden would assert fiction.
- Delete the test outright — rejected: loses coverage of a path that has now
  gone from broken to working, which is exactly what a characterization suite
  should record.
