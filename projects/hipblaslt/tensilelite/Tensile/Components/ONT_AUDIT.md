# ONT_AUDIT — bead-citation sweep of Components/ + Tests/unit/

Bead `ont` audited every docstring and bead-mentioning comment under
`Tensile/Components/` and `Tensile/Tests/unit/` and resolved each
according to the three-bucket scheme:

  * **A. TODO / future-work** — citation points at an open bead the future
    reader still needs to follow. Status: leave the citation, verify the
    bead is open.
  * **B. Stale** — citation points at a CLOSED bead and the surrounding
    code is now self-documenting; the citation is dead history. Status:
    delete the citation; surrounding rationale (if any) survives.
  * **C. Rationale** — bead is named only as a historical anchor for a
    technical claim that still applies. Status: rewrite to drop the bead
    mention but KEEP the technical statement.

## Bead-status summary

Every referenced bead other than `hof`, `f80`, `fnf`, and `ont` itself is
CLOSED. `f80` and `fnf` are siblings to `ont` and out of scope.

The single retained KEEP is the open `hof` reference in
`ScheduleCapture.py:579` (note in `MissingWaitFailure` describing the
collapsed-from `WaitOnWrongCounterFailure` migration that landed under
`hof`'s scope).

## Disposition counts

| Bucket                                       | Count |
|----------------------------------------------|-------|
| A — TODO / future-work, bead still open      | 0     |
| B — stale citation deleted                   | ~95   |
| C — rationale rewritten, bead mention dropped| ~30   |
| KEEP because bead still open (hof)           | 4     |

Counts are approximate because some edits collapse multiple references
into a single rewritten passage (e.g. a deletion comment block that
mentioned three beads becomes one shorter comment).

## New beads filed

None. Every TODO encountered either pointed at a closed bead whose
follow-up work was already complete (deletion ⇒ bucket B, no new bead
needed) or restated rationale that the surrounding code now embodies.

## Per-file resolution notes

### `Tensile/Components/CMSValidator.py`

All bead references in this file are deletion-record comments tracing
the migration of structural rules to the graph-side validator
(`add_pack_constraints`, `add_local_read_constraints`,
`add_gr_finish_before_lr_constraints`, etc.). Every cited bead
(`ola.1` / `ola.2` / `ola.3` / `ola.4` / `4tw` / `8nz` / `dpi` / `nk0` /
`35z` / `or9`) is closed. Resolution: collapse each "[helper] was deleted
in bead X. Its replacement is [graph-side helper]" comment to "[helper]
was removed; its replacement is [graph-side helper]". The forward
pointer to the live graph-side code stays — the historical bead ID drops.

### `Tensile/Components/ScheduleCapture.py`

Mostly bucket B/C. The `nk0` references that document
`cumulative_issue_cycles` as the canonical issue-cycle simulator (post
deletion of the structural-side `precompute_issue_times` /
`estimate_quad_cycles` / `estimate_quad_cycles_precomputed` simulators)
fold into bucket C: keep the "this is the canonical implementation"
statement; drop the "(bead nk0)" and "post-bead 8nz" trailers.

The `2bu.3` / `2bu.4` / `2bu.5` "unification" leads on
`_quad_cycle_gap_ok` / `_cvt_to_mfma_gap_ok` / `_mfma_pack_to_cvt_gap_ok`
become bucket C: drop the "Bead 2bu.X unification:" prefix; keep the
"same-body and cross-body share one code path" statement and the warning
about the previous slot-delta double-counting (still load-bearing for
anyone tempted to revert).

The single `hof` mention in the `MissingWaitFailure` comment block stays
(`hof` is OPEN and the wording change documents an in-progress
collapse).

### `Tensile/Tests/unit/*.py`

Test docstrings reference bead IDs to label whose migration the test
ports (e.g. "Graph-native ports of test_ValidateLRsCompleteBeforeVMFMA.py
— bead ola.3"). Bucket C: drop the bead citation; keep the "Graph-native
port of test_X.py" reference. Test names and body assertions are
untouched.

The `2bu.X` / `nk0` / `35z` / `or9` / `e7w` test references in
`test_dataflow_graph_register_gaps.py` mostly become bucket B: the test
name itself documents what it covers; the "bead Y added this test"
preamble is dead history.

Off-by-one boundary tests that explicitly reference the bead which
created them (e.g. `test_mfma_pack_to_cvt1_one_short_pins_actual_4`'s
mention of `zpi`) are bucket C — keep the "off-by-one boundary sentinel"
contract documentation; drop the bead identifier.

### `Tensile/Tests/unit/dataflow_fixtures.py`

Single bucket-C edit: `make_lcc_pair`'s docstring drops the
"(bead 2bu.1)" mention while keeping the LCC_AUDIT.md cross-reference.

## Verification

* `pytest Tensile/Tests/unit/` — green pre- and post-edit.
* Re-run of the inventory ripgrep — only the four `hof` references
  remain, all of which point at the open bead and document migration
  context that is still in flight.

## Test result

PASS — see commits on `agent/ont`.
