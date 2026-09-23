# Refresh config-driven emit results after develop changes

## Status

Superseded by [ADR 0026](0026-remove-setcover-basename-snapshots.md)

## Context

The original config-driven emit tests selected bounded groups of solutions from
existing Tensile YAML files, generated assembly on the CPU, and saved each
kernel's content-derived basename and emitter return code. Rebasing the tests
onto `develop` changed those basenames and changed four kernel counts:

- `f8f8s_cls_gfx1250`: 4 to 3;
- `spmm_tdm_all`: 4 to 8;
- `sk_bgemm_tdm_split`: 8 to 6; and
- `sk_mxf8gemm_tdm_split`: 4 to 2.

The retained kernels emitted with return code `0`. Repeating the four cases with
`MaxOccupancy` forced from its new default of 64 back to 40 produced the same
counts, so the count changes were not caused by the default change from
`be47443c8e9`.

## Decision

The historical decision was to re-record the 75 failing nodes in the three
set-cover snapshot files against `develop` and an in-tree `rocisa` build. ADR
0026 supersedes that decision: those snapshot files are absent from the final
tree because a basename does not observe emitted behavior.

## Consequences

This ADR records why the intermediate snapshot update existed in the commit
history; it prescribes no current saved-result maintenance. The final tests
assert kernel counts directly and use focused source assertions where a stable
semantic observable exists.
