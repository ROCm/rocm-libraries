# ADR 0039: Record the A2APeerGroups solution parameter

Status:  Accepted
Defect:  none — behavior is intended
Bulk-Snapshot-Update:  yes
Related: [ADR 0037](0037-record-a2agemm-fused-mode.md)

## Context
A2AGemm splits a batch of token blocks into peer groups that start on
different peer shards. The groups become contiguous runs of blocks, and the
default count changes from develop's min(W−1, count) interleaved groups to
ceil(count / ceil(count / (W−1))): 8 blocks at W = 8 form 4 groups of 2
instead of 7. A new solution parameter, `A2APeerGroups` (valid `0`-`7`;
default `0`, that rule), caps the number of groups. It is valid only
with `FusedA2A: A2AGemm` and enters the kernel name (`AAPG<k>`) only when
non-zero, so no existing basename moves. Four goldens change:

- `test_builders_char.ambr`: `A2APeerGroups` joins the valid-parameter key
  roster, and its entry pins a list of length 8 starting `0, 1, 2`.
- `test_solution_class_char.ambr`: the solution state gains the key
  (`len` and `num_keys` 366 → 367); the pinned kernel name is unchanged.
- `test_logiccontract_char.ambr`: each of the 12 pinned solution states gains
  `'A2APeerGroups': 0`.
- `test_a2a_gemm_loop_char.ambr`: a new golden, `test_a2a_gemm_gfx950_golden[7]`,
  pins the six kernels of the new `a2a_gemm_loop.yaml` group (A2APeerGroups 1, 2
  and 7 at A2ASubRounds 1 and 2), all with `err` 0. The other seven groups'
  goldens do not change.

Four changed `.ambr` files exceed the threshold of three in
`check_snapshot_diff.py` (AIHPBLAS-3876). As ADR 0037 anticipated, the first
three move for any new solution parameter, so they cannot be split from the
parameter; the fourth pins the kernels that the parameter's code-generation
tests run.

## Decision
Accept the four files together as one reviewed update. Each diff is confined
to the key and the new golden listed above, and no other node changes. The four
test files passed two consecutive runs without `--snapshot-update` (110 tests,
45 snapshots each run).

## Consequences
The override applies only to this PR. A later solution parameter or
`FusedA2A` value will move the same goldens and needs its own record. These
snapshots establish naming and code-generation status only; the batch-span and
group-walk interpreters in `test_a2a_gemm_loop_char.py` check the generated
group arithmetic, and the comm suite checks the kernels' results on GPUs.
