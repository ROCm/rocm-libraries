# ADR 0039: Record the PAP LocalSplitU LDS offset

Status: Accepted
Defect: none — the state addition is intended

## Context

With PrefetchAcrossPersistent the next tile's prefetched A/B banks stay live
in LDS while the epilogue runs. The LocalSplitU reduction scratch used to
start at LDS 0, on top of those banks, so the derived solution now records
`LdsOffsetLSU`: the start of a reduction scratch placed after every compute
bank (0 when PAP or LocalSplitU is off). The SolutionClass fixture is a
nonpersistent gfx942 kernel, so the field is 0 there, but it belongs to the
state.

## Decision

Record the new key in `test_solution_construction` and increase the key counts
in that snapshot and `test_mapping_interface` by one. Keep the kernel name and
the existing stable field values unchanged.

## Consequences

The snapshots continue to check the complete solution schema. They do not
establish the correctness of the LDS placement; the PAP LocalSplitU tests in
`test_PrefetchAcrossPersistent.py` cover that.
