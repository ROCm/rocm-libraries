---
name: hipdnn-review-comments
description: "Turn review findings into curated, fact-checked PR comments for the user to sign off, then post them. Findings come from a hipdnn-deep-review ledger round or from a plain findings file. Items that share a root cause collapse into one comment, every drop gets a distinct reason, usefulness is judged apart from severity, and every claim is checked at the PR head and anchored on a diff line. Posts inline comments plus an optional review body, only after the user approves the exact text. Never replies to threads. Use after the user has discussed a review result and asks for comments to post."
argument-hint: "<PR URL or number> findings:<ledger:<evidence>:<round> | FINDINGS.md> [out:<dir>] [checkout:<dir>] [terms:<file>] [Agents:<n>] [dry-run:<head>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task
---

# hipDNN Review Comments

`$SKILL` is the directory this skill was loaded from (the directory containing this
`SKILL.md`). You are the orchestrator: execute [RUNBOOK.md](RUNBOOK.md). You compile the
findings, present the drafts and post after approval; fresh agents curate, draft and review.
No agent checks its own output. Nothing is posted that the user has not approved word for word.

Prerequisites: network access, an authenticated `gh` (read access to the PR; write access
only for stage 6), `git`, and Python 3 (it runs `$SKILL/scripts/review_comments.py`).
Ledger mode also needs [hipdnn-deep-review](../hipdnn-deep-review/SKILL.md) installed alongside
this skill; a findings file needs no other skill.

## Entry contract

Ask for what is missing:

| Input | Notes |
|---|---|
| PR | URL or `<owner>/<repo>#<number>`; a bare number means `ROCm/rocm-libraries` |
| `findings:` | `ledger:<evidence>:<round>` (a deep-review round) or a `FINDINGS.md` ([RUNBOOK.md](RUNBOOK.md) *Formats*); in free-form mode compile one from the discussion |
| `out:` | The run directory `$OUT`. Ledger mode default: `<evidence>/r<round>/comments/c<k>/`, where the next deep-review round finds `posted.tsv`. Free-form mode: ask |
| `checkout:` | Optional git checkout holding the PR's base and head; anchors are checked with `git diff`, else from the PR's file patches |
| `terms:` | Optional file of fixed strings that must not appear in any posted text, one per line, kept outside the repository |
| `Agents:` | Optional maximum concurrent agents; stages 2-4 run one agent at a time |
| `dry-run:<head>` | Run stages 1-4 and check against that head; present and post nothing |

## Stages

| # | Stage | Runs in | Output |
|---|---|---|---|
| 1 | Compile | orchestrator | `FINDINGS.md`, `EXISTING.md` |
| 2 | Curate | fresh agent | `TRIAGE.md` |
| 3 | Draft | fresh agent | `COMMENTS.md`, `comments.json` |
| 4 | Review | fresh agent | `COMMENT-REVIEW.md` |
| 5 | Present | orchestrator | the user's sign-off |
| 6 | Post | orchestrator | `POSTED.md`, `posted.tsv` |

Curation never changes the deep-review ledger. Posting is a GitHub write: only the user's
approval of the exact inline bodies and review body authorises it, and the head is checked
again immediately before.
