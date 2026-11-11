# ADR 0037: Rehash kernel names for the per-operand parameters

Status: Accepted
Bulk-Snapshot-Update: yes
Related: [ADR 0033](0033-rehash-kernel-names-for-interleavealpha.md), [ADR 0034](0034-distinguish-code-changing-kernel-modes.md)

## Context

Six per-operand parameters joined the solution parameter set:
`ClusterLocalReadA`, `ClusterLocalReadB`, `PrefetchLocalReadA`,
`PrefetchLocalReadB`, `TDMSplitA`, `TDMSplitB`. Each splits a control that was
previously one value shared by both operands, and each is settable, so per
ADR 0034 it belongs in the kernel name.

Adding them to the parameter set rehashes every kernel name, which moves the
`basename` field of every `{basename, err}` codegen golden. Fifty-four
characterization goldens change, far past the snapshot guard's three-file
threshold.

## Decision

Record the new names. The diff splits cleanly in two:

- **51 files, `basename` only.** The solution name hash moved because the
  parameter set grew. `err` is unchanged everywhere — every solution that
  generated before still generates, with the same status. No assembly is
  asserted by these goldens, so a codegen change cannot reach them.
- **3 files, new parameter keys.** `test_logiccontract_char.ambr`,
  `test_builders_char.ambr` and `test_solution_class_char.ambr` enumerate the
  parameter set itself, so they gain the six keys with their defaults
  (`-1`, `-1`, `-1`, `-1`, `0`, `0`). Nothing else in those files moves.

No golden flips a behavior, and no `err` changes from 0 to non-zero or back.

## Consequences

The goldens again match the shipped parameter set. The split above is the
review handle: a future diff to these files that touches `err`, or that adds a
field beyond the six keys, is a behavior change and needs its own ADR rather
than this one.

`ToolchainComponent/test_toolchain_component_char.py::test_get_rocm_version_rejects_info_version_only`
fails in containers whose ROCm layout makes the discovery chain succeed where
the test expects it to raise. It is not a golden, it fails identically before
and after this update, and it is out of scope here.
