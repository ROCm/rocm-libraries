# Intake record, ledger and report

Formats the orchestrator writes; lanes use [findings.md](findings.md).

## PIN.md

`$EVIDENCE/r<N>/PIN.md`, one `Key: value` line per field; read one with `awk -F': *'
'/^<Key>:/{print $2}'`.

```markdown
Change: <PR URL | branch:<name> | <base>..<head>>
Round: <N>
Repo: <owner>/<repo>
<VAR>: <value>   one line each: SKILL EVIDENCE SCRATCH CLONE KEY TARGET TARGET_TIP HEAD HEAD_SHORT BASE REPO REPO_BASE KNOWLEDGE
Stack parent: <PR number and URL | none>
Description: pr/body.md | intent.md
Files: <count> (GitHub <count> | unverifiable (>3000))
Lines: <numstat total; a rename counts twice>
Checks: <pr/checks.tsv | none>
PR bot: pass | fail | absent
Build host: yes | no (<missing>)
Leak terms: <path | absent>
Knowledge: <$KNOWLEDGE>; <rule that chose it; why an earlier rule failed>
Previous head: <sha>   Previous base: <sha>     (round > 1)
```

## Final IDs

`ledger.py ids` assigns `<LANE>-<n>` (no `<k>`), continuing each lane's counter in
`LEDGER.md`; all correctness lanes share `CORR-<n>`. It writes `ID-MAP.md`, a table with the
columns `Lane / provisional | Final ID | SUPERSEDED-BY`; the first column is `<report
stem>/<provisional ID>` (`CORR2/CORR2-p3`, `TESTQ-f1/TESTQ-p4`), and `SUPERSEDED-BY` holds the
kept finding's final ID or is empty. A later run keeps assigned IDs and applies each
`SUPERSEDED-BY` cell to the ledger row's Status.

## LEDGER.md

`$EVIDENCE/LEDGER.md`, the complete machine record across rounds, written by `ledger.py` and
by hand. It holds only `##` headings and pipe tables (the scripts reject any other line). The
schema is stable; a row in another format is a candidate, not a record.

```markdown
## Coverage (round N)     table: File | Lines | Owner lane | Read depth | Tests reaching it
## Findings               table: ID | Tier (`<reported> -> <final>`) | Kind | Status | Verdict | Claim (narrowed
                          if PARTIAL; until a verdict, `Actual:` collapsed to one line, cut after its first
                          sentence and capped at 200 characters) | Where | Evidence |
                          Round opened | Round closed | known-open-item (empty or `K<n>`)
## Lanes (round N)        table: Lane | Ran / N/A / merged into / split / FAILED (reason) | Report | STOP
## Decisions              table: Round | Question | Decision | Findings affected
## Open questions         table: ID | Question | Options | Recommendation
```

`ledger.py seed` writes the Coverage and Lanes tables, `ids` the candidate Findings rows,
`verdicts` the verdict, status and final tier. A candidate has empty Status and Verdict. A second-verifier verdict that differs from the
first replaces it.

From round 2, append; never rewrite the file. The seed has one Coverage row per changed file (Read depth `pending`, Tests `-`) and one
Lanes row per lane; copy each lane's read depth and tests in as it reports.

A **candidate** is a reported finding with no verdict. A verdict sets the status:
`REFUTED` and `UNVERIFIED` keep their name; any other verdict gives `OPEN`. Other statuses:
`FIXED@<sha>`, `ACCEPTED-RISK` (human decision recorded), `SUPERSEDED-BY:<ID>`. `ledger.py set` records `FIXED`, `ACCEPTED-RISK` and reopening to `OPEN` ([RUNBOOK.md](RUNBOOK.md) *Iterate*). **Closed**
is every status except `OPEN` and `UNVERIFIED`. Never delete a row. A row closed by a fix
begins its Evidence cell with the evidence class ([findings.md](findings.md) *Verdicts*).

## Assessment and risk

`APPROVED`: policy met; a documented, accepted waiver does not block. `CHANGES REQUESTED`:
an `OPEN` or `UNVERIFIED` `BLOCKING` finding (unless a human decision records
`ACCEPTED-RISK`), a missing test, flag or waiver, absent tracking, a defect without a
regression plan, or a weak waiver. `REJECTED`: the approach is fundamentally wrong.

Risk 1: docs, comments or isolated non-shipping tests. 2: narrow, good coverage, no API,
schema or build impact. 3: core subsystem, new feature path, schema or API addition,
dispatch, build or test infrastructure. 4: broad behavior change, compatibility-sensitive
public API, performance-critical path, incomplete coverage. 5: cross-project or
architectural, default behavior change in a critical path, ABI break, large unproven
refactor, known unresolved failures. A required run that has not passed raises the level.

## REPORT.md

The human handoff: the reader acts without opening the ledger. Group findings by theme;
lead each bullet with the problem, not an ID (cite IDs sparingly, in parentheses). Every
statement traces to a ledger row; the REPORT may call a finding with a
`known-open-item` understated and cite the author's `K<n>`. Omit `REFUTED` findings and empty sections. Aim for about
80 lines.

```markdown
# Review of <owner>/<repo>#<pr> | <branch or range>: <title or intent summary>
<one line: commit and base or stack parent reviewed, size, whether the head moved>
Target drift: <`DRIFT` result: no trigger | merges clean, tests pass | conflicts resolved mechanically | breaks the merge (see fix list)>; target tip <12 hex digits>

## Verdict
<paragraph 1: assessment and risk (*Assessment and risk* above); open
BLOCKING and IMPORTANT counts; what to fix before merge; M-rule failures. End with
`Closure: OPEN | CLOSED` ([SKILL.md](SKILL.md) *Completion*); if OPEN, list what waits:
unverified findings, pending approvals, open decisions, `STOP:` rows, FAILED lanes.>
<paragraph 2: what was run and what was not>

## Decisions needed
| # | Question | Recommendation |
## Fix list, in priority order
### 1. <theme, stated as the consequence>
- **<one-line problem>** (`path:line`): consequence. Fix: <smallest change>.
<test gaps: group by theme; name the surviving mutation and the test to add>
## Lower priority (<n> suggestions, <m> future work)
<a few bullets batched by theme; `LEDGER.md` holds the full list>
## Earlier reviews
<round 2 on: what closed, what changed tier, corrections to earlier claims>
## What this review does not establish
<unrun hardware, platforms and builds; the runs that would close the gaps>
## Evidence
<`$EVIDENCE` paths: `LEDGER.md`, `r<N>/COMMENTS.md`, `r<N>/verify/`. Nothing posted.>
```

## COMMENTS.md

Drafts for the user to post. One `### <path>:<line> (<ID>, <tier>)` entry per `BLOCKING` or
`IMPORTANT` finding that is `OPEN` (problem, consequence, smallest fix). Batch `SUGGESTION` and
`FUTURE WORK` findings by theme into `## General`, one item per theme.

`comments.py draft` writes the first draft from the `OPEN` ledger rows. The anchor line must lie
inside a changed new-side hunk of the diff between `$BASE` and `$HEAD` for that path (`-U0`
`@@ +<start>,<count>` headers; a count of 0 is an empty range). A finding whose line is
outside every hunk, or about removed content (`$BASE:` anchors), goes under `## General`
with one `path:line` anchor and its one-line claim; a `$HEAD:path:line` anchor counts as `path:line`. Pasted text carries no code
fences, `$VAR`s, lane stems, provisional IDs or `$EVIDENCE` paths. Edit the draft before the user sees it.
