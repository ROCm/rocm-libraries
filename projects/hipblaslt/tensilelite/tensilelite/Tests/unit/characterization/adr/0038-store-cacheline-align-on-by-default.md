# ADR 0038: Four goldens move when StoreCachelineAlign is turned on by default

Status:  Accepted
Defect:  none — behavior is intended
Bulk-Snapshot-Update:  yes

## Context
This PR turns the `StoreCachelineAlign` solution parameter on by default
(`GlobalParameters.py`, `[False]` → `[True]`). Four `.ambr` goldens change as a
result, one more than `tools/check_snapshot_diff.py` allows without an override,
so CI's blanket-regeneration guard (AIHPBLAS-3876) fails the PR.

Nothing was regenerated wholesale. Each golden was updated by test id, and the
whole `.ambr` diff of the PR is 19 changed lines of two kinds:

| golden | what changed |
|---|---|
| `LibraryIO/test_logiccontract_char.ambr` | twelve parsed solutions record the default: `'StoreCachelineAlign': False` → `True` |
| `_codegen/test_mxf4_bf16_subtile_char.ambr` | four kernel `basename`s |
| `_codegen/test_mxf4_swizzlea_subtile_char.ambr` | one kernel `basename` |
| `_codegen/test_mxf4_swizzleb_subtile_char.ambr` | two kernel `basename`s |

The seven basenames belong to gfx950 MXFP4 subtile kernels emitted with default
parameters. `StoreCachelineAlign` is part of a kernel's name where it acts (the
`SCA1` token; see `Naming.py`), and these goldens record the hashed form of the
name, so the hash moves when the default does. No other field of those goldens
changes: `err` and every recorded property of the seven kernels are the same
lines as before.

The three `_codegen` goldens cannot be split from the first: they are the same
change seen from the codegen side, and a PR carrying only the default flip would
touch all four.

## Decision
Grant the override for this PR. The diff is the minimum golden movement that
changing this default can produce, and every changed line is one of the two
kinds above.

## Consequences
- A reviewer can check this cheaply: in the PR's `.ambr` diff, every changed
  line either contains `'StoreCachelineAlign':` or is a `'basename':` line.
- The goldens record the kernels' names, not their assembly, so this ADR does
  not vouch for the generated code. That is covered elsewhere: the unit tests in
  `Tests/unit/test_subtile_store_cacheline_align.py`, the on/off cases in
  `_codegen/test_mt320x256_plsin_weave_char.py`, and the GPU verification
  recorded in the PR.
- Kernels the parameter cannot act on (`MIWaveTile[0] < 4`, other
  architectures, other store paths) keep their names, so no golden of theirs
  moves. If one does in a later PR, that is a real change and not covered here.
- Flipping the default back would move the same four goldens the other way and
  would need its own ADR, for the same structural reason.
