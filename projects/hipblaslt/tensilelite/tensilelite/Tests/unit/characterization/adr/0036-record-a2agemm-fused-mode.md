# ADR 0036: Record the A2AGemm fused all-to-all mode

Status:  Accepted
Defect:  none — behavior is intended
Bulk-Snapshot-Update:  yes
Related: [PR #14168](https://github.com/ROCm/rocm-libraries/pull/14168)

## Context
PR #14168 adds A2AGemm, a second fused all-to-all mode in which B's K shards
are gathered from the peers one shard at a time. Three changes in that PR reach the goldens:

- The boolean ProblemType key `FusedGemmA2A` is replaced by the string key
  `FusedA2A` (`Disabled`, `GemmA2A`, or `A2AGemm`). This moves
  `test_contractions_char.ambr` and `test_helpers_char.ambr` (key name and type)
  and `test_problemtype_char.ambr` (every pinned state: `False` → `'Disabled'`).
- A new solution parameter, `A2ASubRounds` (valid `1`, `2`, `4`; default `1`),
  joins the roster. This moves `test_builders_char.ambr` (the valid-parameter
  entry), `test_solution_class_char.ambr` (state keys 353 → 354), and
  `test_logiccontract_char.ambr` (the default in each pinned solution state).
  It enters the kernel name only for `A2AGemm` kernels, so no existing basename
  moves.
- The existing GemmA2A kernels' name suffix changes from `FusedA2A` to
  `GemmA2A`, which moves the two basenames in
  `test_fused_a2a_gfx950_char.ambr`; the new `test_a2a_gemm_loop_char.ambr`
  pins the two A2AGemm kernels.

Eight changed `.ambr` files exceed the threshold of three in
`check_snapshot_diff.py` (AIHPBLAS-3876).

## Decision
Accept the eight files together as one reviewed update. Each diff is confined
to the keys and basenames listed above; every `err` stays `0`, and no node
outside those keys changes. The eight test files passed two consecutive runs
without `--snapshot-update` (148 tests, 123 snapshots each run).

## Consequences
The override applies only to this PR. A later fused mode that adds a
`FusedA2A` value or a solution parameter will move the same goldens and needs
its own record. Passing these snapshots establishes naming and code-generation
status only; the A2AGemm kernels' numerical behavior is covered by the
multi-GPU comm tests, which do not run in this CI lane.
