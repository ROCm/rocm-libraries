# Lane: runtime integration (`IMPACT`)

Find changes that break the kernel ingestor at run time while the build stays green: a kernel's
argument list (its ABI) that no longer matches a copy, a dispatch predicate that drifts from the
kernel it selects, a descriptor or generic-target edit that changes what ships.

You judge; two scripts from the sibling skill
[hipdnn-ingestor-impact](../../hipdnn-ingestor-impact/SKILL.md) collect facts. Read its
[reference](../../hipdnn-ingestor-impact/reference.md) first: it holds the contracts, the row
statuses and rollup, the emitter's rules, how to judge each row, and the known gaps. This brief adds
what a review needs: where the evidence goes, when to stop, and how a judged row becomes a finding.
`impact_emit.py` imports and runs rocKE code from the reviewed trees: the same code execution as
`RUN` building the head.

Test code, CMake, CI test-category lists and `requirements*.txt` are not yours: they are `outside`
for status (the rollup names how many files it left to `BUILD` and `DEPS`), but tests are still
searched as copies and readers, so a stale pin or a stale import is yours.

## Method

1. Collect the facts. The report covers base to head and, when the target branch moved past the
   merge base, the head merged into `$TARGET_TIP`, where contract breaks between two PRs that each
   pass alone show up. `DRIFT` builds and tests that merge; you compare its contracts statically:

   ```bash
   . ./env.sh; mkdir -p "$SCRATCH/IMPACT" lanes/IMPACT-evidence
   IMPACT_SCRIPTS="$SKILL/../hipdnn-ingestor-impact/scripts"
   python3 "$IMPACT_SCRIPTS/impact.py" report --repo "$CLONE" --base "$BASE" --head "$HEAD" --target-tip "$TARGET_TIP" > lanes/IMPACT-evidence/report.md
   ```

   Sourcing `env.sh` moves to the round directory. Exit 1 means the script could not run: return
   `STOP:` with its message.
2. Run the emitter when the reference's *The emitter* says to:

   ```bash
   rm -f "$SCRATCH/IMPACT/emit.json"
   python3 "$IMPACT_SCRIPTS/impact_emit.py" run --repo "$CLONE" --base "$BASE" --head "$HEAD" --out "$SCRATCH/IMPACT/emit.json"
   python3 "$IMPACT_SCRIPTS/impact.py" report --repo "$CLONE" --base "$BASE" --head "$HEAD" --target-tip "$TARGET_TIP" --emit "$SCRATCH/IMPACT/emit.json" > lanes/IMPACT-evidence/report-emit.md
   ```

   The emitter exiting 1 (it could not run, or a side timed out) leaves no proof: return `STOP:`
   with its message, and never fold an `emit.json` from an earlier run. Record the flavor it used.
3. Judge every row whose status is not `inert` or `outside` with the reference's *Judging the
   rows*. Read both sides at `$HEAD` (and the merged tree when the row says `merged-only`) before
   filing.
4. End **Checked, no finding** with the rollup line the script printed, corrected by your
   judgment and saying why when it differs:
   `Runtime impact: BREAKS | NEEDS-DEVICE <arch: suite, ...> | NONE (<n> files inspected) | N/A (no surface file)`,
   keeping any `not judged here:` note the script added.

## From row to finding

| Judgment | Finding |
|---|---|
| A `mismatch` that holds | A `source-claim` at `BLOCKING` ([../policy.md](../policy.md): an ABI break without a migration path). **Where**: the producer line and every disagreeing copy. **Proof**: the command and both quoted lists. A `merged-only` mismatch: **Fix** is to rebase and update the named copy. The same mismatch at `$BASE` is pre-existing: `FUTURE WORK` at most |
| A dispatch divergence | A `source-claim` at `IMPORTANT` or higher, quoting both sides |
| A descriptor or generic-target value the loader rejects or ignores | A finding, citing the loader rule |
| No break (`changed` but consistent), `inert`, `outside` | No finding. Count them under **Checked, no finding**; list each `inert` reason once |

For every `changed` row without a finding, put the device run the reference says to name under
**Approval requests**, and add `AUTH: requires <arch> <suite> with the ingestor flags ON` under
**Handoffs**; `AUTH` decides whether a run executed.

## Handoffs

- `API`: a changed declaration in a plugin_sdk header, or a `.fbs` change.
- `BUILD`: flag matrix, option defaults, test registration, install rules.
- `ROCK`, `AUTH`: whether CI builds and tests the change; you supply the required pairs only.
- `DRIFT`: a `merged-only` row, so the merged build and tests cover the copy you name; and the
  conflicted files of a merged view the report marks `conflicted`, which it does not analyse.
- `TESTB`: a contract with no test that would fail on the break you describe.

Known gaps are in the reference's *Known gaps*; a row that one of them touches is not proof either
way.
