# Intake record, ledger and report

Formats the orchestrator writes; lanes use [findings.md](findings.md).

## PIN.md

`$EVIDENCE/r<N>/PIN.md`, one `Key: value` line per field; read one with `awk -F': *'
'/^<Key>:/{print $2}'`.

```markdown
Change: <PR URL | branch:<name> | <base>..<head>>
Round: <N>
Repo: <owner>/<repo>
<VAR>: <value>   one line each: SKILL EVIDENCE SCRATCH CLONE KEY TARGET HEAD BASE REPO REPO_BASE KNOWLEDGE
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

Assign `<LANE>-<n>` (no `<k>`), continuing each lane's counter in `LEDGER.md`; all
correctness lanes share `CORR-<n>`. Write `ID-MAP.md`: `Lane / provisional | Final ID |
SUPERSEDED-BY`.

## LEDGER.md

`$EVIDENCE/LEDGER.md`, the complete machine record across rounds. The schema is stable; a
row in another format is a candidate, not a record.

```markdown
## Coverage (round N): File | Lines | Owner lane | Read depth | Tests reaching it
## Findings: ID | Tier (reported -> final) | Kind | Status | Verdict | Claim (narrowed if
   PARTIAL) | Where | Evidence | Round opened | Round closed
## Lanes (round N): Lane | Ran / N/A / merged into / split / FAILED (reason) | Report | STOP
## Decisions: Round | Question | Decision | Findings affected
## Open questions: ID | Question | Options | Recommendation
```

From round 2, append; never rewrite the file. The seed has one Coverage row per changed file (Read depth `pending`, Tests `-`) and one
Lanes row per lane; copy each lane's read depth and tests in as it reports.

A **candidate** is a reported finding with no verdict. A verdict sets the status:
`REFUTED` and `UNVERIFIED` keep their name; any other verdict gives `OPEN`. Other statuses:
`FIXED@<sha>`, `ACCEPTED-RISK` (human decision recorded), `SUPERSEDED-BY:<ID>`. **Closed**
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
statement traces to a ledger row. Omit `REFUTED` findings and empty sections. Aim for about
80 lines.

```markdown
# Review of <owner>/<repo>#<pr> | <branch or range>: <title or intent summary>
<one line: commit and base or stack parent reviewed, size, whether the head moved>

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

Drafts for the user to post: one `### <path>:<line> (<ID>, <tier>)` entry per `OPEN`
finding (problem, consequence, smallest fix). The line is in the PR's diff at `$HEAD`; a
finding with none, or about removed content, goes under `## General`.
