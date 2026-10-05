# ADR 0029: Record persistent vector epilogue LDS state

Status: Accepted
Defect: none — the state additions are intended

## Context

The persistent vector epilogue fix adds `_PersistentVectorEpilogueLds` and
`_SeparateEpilogueLds` to derived solutions. The SolutionClass fixture uses a
nonpersistent gfx942 kernel, so both fields are false, but they still belong
to its state. The saved construction and mapping summaries omit these fields.

## Decision

Record the two new keys in `test_solution_construction` and increase the key
counts in that snapshot and `test_mapping_interface` by two. Keep the kernel
name and existing stable field values unchanged.

## Consequences

The snapshots continue to check the complete solution schema. They do not
establish the correctness of the LDS allocation or synchronization; the
persistent vector epilogue tests in `test_PrefetchAcrossPersistent.py` cover
those requirements. Future schema additions must update these expectations
explicitly.
