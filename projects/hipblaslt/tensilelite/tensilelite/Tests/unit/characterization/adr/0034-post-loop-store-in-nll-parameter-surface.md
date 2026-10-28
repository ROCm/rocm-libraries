# ADR 0034: Four goldens move when one solution parameter is added

Status:  Accepted
Defect:  none — behavior is intended
Bulk-Snapshot-Update:  yes

## Context
This PR introduces one new solution parameter, `PostLoopStoreInNll`. Four
`.ambr` goldens change as a result, one more than
`tools/check_snapshot_diff.py` allows without an override, so CI's
blanket-regeneration guard (AIHPBLAS-3876) fails the PR.

The guard is right to ask and wrong about the cause. Three of the four are
parameter-surface goldens — they enumerate every key Tensile accepts, so they
cannot *not* move when a key is added:

| golden | what changed |
|---|---|
| `ValidParameters/test_builders_char.ambr` | the `PostLoopStoreInNll` entry and its `[False, True]` domain |
| `SolutionClass/test_solution_class_char.ambr` | the key, and the key count 351 → 352 |
| `LibraryIO/test_logiccontract_char.ambr` | twelve solutions now record the key |

Every changed line in all three names the new parameter or is the count bump
that follows from it; none is an unrelated re-baseline. The fourth,
`_codegen/test_mxf4_bf16_subtile_char.ambr`, is an **added** file — the golden
belonging to this PR's own new characterization test. Nothing was regenerated
to produce it, and the guard counts additions alongside modifications, so a PR
that adds one parameter and one test reaches four without ever running a
blanket `--snapshot-update`.

## Decision
Grant the override for this PR. The diff is not a mass regeneration: it is the
minimum golden movement that adding a parameter and a test can produce, and
each line is attributable to one of those two causes.

The three modified goldens were not regenerated wholesale; the diff was read
line by line to confirm that every one of them names `PostLoopStoreInNll` or is
the key count that follows it.

## Consequences
- A reviewer can check this claim cheaply: filter the `.ambr` diff for lines
  that do *not* mention `PostLoopStoreInNll`. Only the `num_keys` / `len`
  351 → 352 bumps and the new file's contents should remain.
- **This ADR grants the mechanical override, not a judgment that the twelve
  values in the logic-contract golden are correct.** Those twelve are gfx942
  non-fp4 solutions recording `PostLoopStoreInNll: True`, which is either a
  benign parse-level default or a gate that does not narrow as intended. That
  question is open and tracked separately; the golden pins what the code does
  today so the answer, when it lands, shows up as a diff rather than silence.
  If it turns out to be a gate miss, those twelve flip and this ADR is
  superseded.
- If a later PR adds another parameter, it will trip the same guard for the
  same structural reason. That is the guard working as designed — three
  parameter-surface goldens plus any new test's own golden is four — and the
  answer is another ADR, not a higher threshold.
