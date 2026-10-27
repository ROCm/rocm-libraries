# ADR 0033: Re-hash every kernel name to carry `InterleaveAlpha`

Status:  Accepted
Defect:  none — behavior is intended
Bulk-Snapshot-Update:  yes

## Context
`InterleaveAlpha` gains a third value, `2`, that defers each element's alpha
multiply into its own conversion+pack in the store loop. Value `2` emits
different assembly than `0`, so it must appear in the kernel name: added to
`getRequiredParametersMin()` (`Common/RequiredParameters.py`). Without it the
`0` and `2` kernels hash to the same name, one is dropped as a duplicate by
`Naming._getName`, and its dispatch fails at launch with a dangling `.s` and
"named symbol not found".

Adding a key to the required-parameter set changes the hashed suffix of **every
kernel basename**, on every architecture, whether or not that kernel's emitted
assembly moved. The characterization goldens record those basenames, so a single
one-line addition to the name set rewrites every `_emit_*`, `_seed_*`, and
`_codegen` `.ambr` that pins a kernel name — 52 golden files across gfx908,
gfx90a, gfx942, gfx950, gfx1100, gfx1201, and gfx1250. Most of that churn is the
basename hash alone: the gfx908 goldens, for instance, keep `err: 0` and every
assembly body unchanged and move only the `basename` string. The goldens that
actually exercise the value-`2` path (the gfx1250 emit/store/global-write nodes
and `GlobalWriteBatch` output) also change body, which is the real behavior under
test; the other ~45 files move only because the name hash did.

52 changed `.ambr` files is far over `check_snapshot_diff.py`'s threshold of 3
(AIHPBLAS-3876), which reads a diff this wide as a blanket `--snapshot-update`.

## Decision
Accept the 52-file golden update as a single reviewed bulk change and record it
here with `Bulk-Snapshot-Update: yes`. It cannot be split into sub-threshold
PRs: adding one parameter to the kernel-name set moves every kernel's basename at
once, so there is no scoping that touches ≤3 goldens. The diff is a pure
name-hash ripple everywhere except the gfx1250 value-`2` codegen nodes, which
carry the genuine assembly change.

## Consequences
- The override is scoped to this PR: the guard honors it only because this
  numbered ADR is added in the same diff. A later PR that again changes
  `getRequiredParametersMin()` (adding or removing a key) will likewise re-hash
  every basename and must ship its own ADR with the marker.
- A reviewer verifying the goldens can confirm the churn is benign by spot-check:
  outside the gfx1250 value-`2` nodes the only field that moves is `basename`,
  with `err` and all assembly bodies unchanged.
- If `InterleaveAlpha` is ever dropped from the name set again, the basenames
  revert and the goldens flip back under the same override, superseding this ADR.
