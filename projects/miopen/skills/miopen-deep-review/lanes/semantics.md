# Lane: semantics, fresh eyes (`SEMANTICS`)

Where this file and [common.md](common.md) conflict, this file wins. You are the reader a maintainer would be: you describe what
the changed entry points do for a user, without being told what the change was meant to do, and say what looks odd. Other lanes
check the change against its description; you check the behavior against what a user would expect.

## Inputs

Read: `pr/body.md` (title and description), the public docs and header comments of each changed entry point, the post-change
code of those entry points and of the paths they reach (`$REPO`), the test names. Do not read the diff, `facts.md`, any other
lane's report, `LEDGER.md`, `REPORT.md` or an earlier round: independence is the point. You may read `$REPO_BASE` to tell whether
a difference you found existed before the change.

## Method

1. List the user-visible values of each changed entry point: arguments (scale factors, attributes, flags), environment variables,
   returned sizes and status codes.
2. For each value, write a table of situations against what happens to it: data type, dimension, size above and below each
   internal dispatch threshold, backend or architecture. Each cell says whether the value is converted, rounded, clamped or
   ignored, and what the result is. Derive every cell from the code, with `path:line`.
3. For each row, ask what a user of the documented API would expect. Flag a cell where two situations the API treats alike give
   different results; where a value is converted to a type narrower than its declared one; where the result depends on a
   threshold the API does not document; and where the documentation and the code disagree.
4. For each flagged cell read `$REPO_BASE` for the same cell. A difference that is new is introduced by the change; one that
   existed before is pre-existing.

## Findings

| Tier | When |
|---|---|
| `IMPORTANT` | A difference the change introduced that the PR text does not state: a result that differs for the same logical call. Kind `defect`; also file it under **Needs human decision** with the options `fix the cause` and `document` ([../policy.md](../policy.md) *Review rules*) |
| `SUGGESTION` | An introduced difference with no effect on a result (naming, an undocumented but harmless conversion) |
| `FUTURE WORK` | A flagged cell that existed at `$BASE` and that the change does not alter |

Proof is the table row with its `path:line` anchors. A run is not required; when one call below and one above a threshold would
settle it cheaply, put it under **Approval requests** (the same input at both sides of the threshold; expected equal results).

## Cost

One agent, no build, no GPU; it runs in the first fan-out, so it never waits for another lane.

## Handoffs

`API:` a flagged public contract. `CORR:` a defect found while reading. `TESTB:` a flagged cell no test pins. `DOCS:` a
documented behavior the code contradicts.
