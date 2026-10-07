# ADR 0002: Key TensileLogic known-bugs on solution_name, not solution_index

Status:  Accepted
Defect:  none - behavior is intended (motivating context: ROCM-7144)
Commit:  910c561 (PR #9355) — https://github.com/ROCm/rocm-libraries/commit/910c5616ec501f5ab750931e5064f0eff1eb89b8 — landed on develop via PR #9355's squash merge, d250cc8

## Context
`TensileLogic.KnownBugs` originally keyed each documented `--check-all` skip on
`(path, solution_index)`. `SolutionIndex` is positional: it shifts whenever the
library logic is re-tuned or regenerated, so every such PR risked hand-editing
`known_bugs.yaml`. Each solution already carries a `SolutionNameMin` - a
canonical, content-derived name (macro tile, MatrixInstruction, all kernel
params) that is stable across re-ordering and self-invalidates when the kernel
actually changes. TheRock is the only consumer and there is only ever one
`known_bugs.yaml`, so there is no external compatibility to preserve.

This changes observed behavior of the characterized surface: `load_known_bugs`
now returns `(path, solution_name)` keys and rejects a legacy `solution_index`
entry, and `is_known_bug(known, rel, solution_name)` matches on the name. The
`test_knownbugs_char.py` goldens (`test_is_known_bug_hit_and_miss`,
`test_load_roundtrip_multi`) change accordingly.

## Decision
Key known-bugs skips on `(path, solution_name)` only. Drop `solution_index`
support entirely (no backward-compatible dual key). Re-record only the two
affected golden nodes; the on-disk `known_bugs.yaml` entries were migrated from
indices to names as a one-time change in this PR.

## Consequences
Documented skips survive index churn without manual edits, and a genuinely
fixed/removed kernel stops matching (the correct signal to prune the entry).
This is an intended behavior change, not a pinned bug, so no `Defect:` fix is
tracked. If the key scheme is ever revised again, flip these goldens and
supersede this ADR. The `.ambr` edits here were made to match syrupy's amber
serialization by hand; they must be confirmed byte-identical by running
`pytest Tensile/Tests/unit/characterization/TensileLogic/test_knownbugs_char.py --snapshot-update`
in a build environment (rocisa present) and verifying no further diff, since the
`-m unit` lane cannot run without the compiled rocisa module.
