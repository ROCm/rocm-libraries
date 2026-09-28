# ADR 0027: Select problem groups by solution-generation inputs

Status:  Accepted
Defect:  none — behavior is intended

## Context

ADR 0024 identified a `BenchmarkProblems` entry by hashing its complete value.
That value includes `BenchmarkFinalParameters`, such as runtime problem sizes
and activation arguments. These fields do not participate in solution
derivation or kernel emission, but editing one changed the selector and failed
the set-cover test. Hashing only `ProblemType` would avoid that coupling but is
not unique in several multi-entry configuration files.

## Decision

Compute the selector from `ProblemType` and every problem-size-group field
except `BenchmarkFinalParameters`. This retains the parameters that define the
solution search while ignoring runtime-only benchmark inputs.

List selectors with:

```console
PYTHONPATH=. python Tensile/Tests/unit/characterization/_codegen/list_config_fingerprints.py <config.yaml>
```

When a selector changes, compare the selected problem type and generation
parameters with the prior entry, then update the table only after its fork
count, emitter statuses, and source assertions pass.

## Consequences

Reordering unrelated problem groups and editing runtime problem sizes no longer
redirect or invalidate a set-cover case. A change to the selected problem type
or solution-generation parameters still fails explicitly. Two entries with the
same selector are equivalent for this harness because they differ only in
ignored runtime benchmark inputs.
