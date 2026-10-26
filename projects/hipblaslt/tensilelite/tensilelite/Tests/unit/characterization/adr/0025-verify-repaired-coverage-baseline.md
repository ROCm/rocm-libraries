# Verify the repaired coverage baseline

## Status

Accepted

## Context

The original rebaseline description listed nine reductions although its diff
lowered ten floors. ADR 0022 corrected that accounting and refreshed three
additional floors after unsupported MX exception-path tests were removed, but
the public run stopped before executing the coverage gate because saved-result
comparisons failed.

Before the final rebase, replacing those unstable comparisons allowed the
complete measurement to run in the `tensilelite-char:repro` container with a
freshly rebuilt `rocisa` extension. The characterization half completed with
3,261 passes, 2 expected failures, 1 unexpected pass, and 814 saved results. The
unit and extras half completed with 4,106 passes, 230 skips, and 16 expected
failures. Combined branch coverage was 84.13%.

The measured values reproduce the baseline for the disputed files:

- `Configuration.py`: 92.53%;
- `Solution.py`: 73.89% against a 73.92% floor;
- `PackData.py`: 85.04%;
- `LocalRead.py`: 76.68%; and
- `LraTileAssignment.py`: 93.17%.

The complete `coverage-gate` environment passed with the existing one-point
tolerance.

After rebasing the complete stack onto current `develop`, the characterization
half completed with 3,261 passes, 2 expected failures, 1 unexpected pass, and
814 saved results. The unit and extras half completed with 4,113 passes, 230
skips, and 16 expected failures. Combined branch coverage was 84.14%. The
rebased measurements for the disputed files were:

- `Configuration.py`: 92.53%;
- `Solution.py`: 73.92%;
- `PackData.py`: 85.04%;
- `LocalRead.py`: 76.68%; and
- `LraTileAssignment.py`: 93.17%.

The gate passed against `develop`'s unchanged 172-file baseline. Running the
monotonic updater from that report then raised 38 floors, added none, and
lowered none. A final gate run passed with those raised floors and the new 82%
whole-project floor.

## Decision

Keep `develop`'s newer baseline through every historical coverage-baseline
conflict and do not replay the stale reductions from ADRs 0018 and 0022. Raise
the 38 floors supported by the green rebased report. Raise the whole-project
floor from 75% to 82%, leaving slightly more than two percentage points below
the two stable 84.13% and 84.14% measurements.

## Consequences

The net baseline diff contains increases only. It captures the coverage gained
by the stack without accepting any unexplained regression, and the project-wide
floor now protects newly added files before the next per-file refresh.
