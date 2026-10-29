# ADR 0033: Extend snapshots for W4A16 types and defaults

Status: Accepted
Bulk-Snapshot-Update: yes

## Context

Adding Int4 and W4A16 problem defaults changes five characterization golden
files, exceeding the snapshot guard's three-file threshold. The reviewed diff
contains 310 added lines and one replaced line, all attributable to these
additions.

## Decision

Accept the scoped updates to these snapshots:

- `DataType/__snapshots__/test_constructor_char.ambr`: add Int4 (index 24)
  and its constructor aliases and type representations.
- `DataType/__snapshots__/test_numeric_dunder_char.ambr`: add Int4 numeric
  properties, including half-byte storage and one-eighth register usage.
- `DataType/__snapshots__/test_predicates_char.ambr`: extend the predicate
  matrix with Int4 and the `isInt4` predicate.
- `ProblemType/__snapshots__/test_helpers_char.ambr`: add the keys and value
  types for `Int4EncodingA`, `ScaleBlockSizesA`, and `ScaleZeroPointA`.
- `ProblemType/__snapshots__/test_problemtype_char.ambr`: include their
  defaults (`Signed`, `[]`, and `False`) and increase the mapping length
  from 90 to 93.

These changes follow the Int4 entry in `Common/DataType.py` and the defaults
in `SolutionStructs/Problem.py`. Existing data-type values and existing
problem-field values remain unchanged. The golden extension records the
current helper outputs; it does not assert that every emitted type spelling
is supported by a kernel path.

## Consequences

The snapshot guard accepts this documented five-file extension. Its threshold
and the characterization assertions remain unchanged. Run the affected
DataType and ProblemType tests without `--snapshot-update` to validate the
recorded outputs. Future golden changes still require their own review.
