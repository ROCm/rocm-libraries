# Refresh config-driven emit results after develop changes

## Status

Accepted

## Context

The config-driven emit tests select bounded groups of solutions from existing
Tensile YAML files, generate assembly on the CPU, and save each kernel's
content-derived basename and emitter return code. Rebasing the tests onto
current `develop` brought in the Stream-K admission changes from `3a558b0d4d4`
and other solution-derivation updates. A fresh in-tree `rocisa` build therefore
produces different kernel basenames from the saved results.

Seventy-one of the 75 set-cover cases preserve both their kernel count and the
multiset of emitter return codes. Four cases change kernel count:

- `f8f8s_cls_gfx1250`: 4 to 3;
- `spmm_tdm_all`: 4 to 8;
- `sk_bgemm_tdm_split`: 8 to 6; and
- `sk_mxf8gemm_tdm_split`: 4 to 2.

All kernels retained by those four cases emit with return code `0`. Repeating
the four cases with `MaxOccupancy` forced from its new default of 64 back to 40
produces the same counts, so the count changes are not caused by the default
change from `be47443c8e9`.

## Decision

Re-record only the 75 failing nodes in the three set-cover snapshot files
against current `develop` and an in-tree `rocisa` build. Keep the selected YAML
files and the eight-kernel limit unchanged. Do not change production source or
lower a coverage floor to preserve the previous results.

## Consequences

The saved results now describe the solutions accepted by current `develop`.
Future changes to solution derivation or validation may change the basenames or
counts again; reviewers must compare kernel counts and emitter return codes
before accepting another update.
