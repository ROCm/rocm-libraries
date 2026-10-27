---
name: hipdnn-deep-review
description: "Multi-agent deep review of a hipDNN or dnn-providers PR, branch or commit range. Scouts map each affected area, focused reviewer lanes run in parallel, every finding is verified by source proof, reproduction or mutation, and a ledger carries findings across rounds until no blocking or important one stays open. For a PR body or a quick single-pass review, use hipdnn-pr-quality."
argument-hint: "[<PR URL or number> | branch:<name> | <base>..<head>] evidence:<dir> scratch:<dir> [clone:<dir>] [intent:<file>] [terms:<file>] [knowledge:<dir>] [round:<N>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task
---

# hipDNN Deep Review

`$SKILL` is the directory this skill was loaded from (the directory containing this
`SKILL.md`). You are the orchestrator: execute [RUNBOOK.md](RUNBOOK.md). You pin the change, dispatch
scouts, lanes and verifiers, merge their output and write the report; you do not review
code. Your formats are in [report.md](report.md); [RUNBOOK.md](RUNBOOK.md) *Fan out* says
what each agent reads, and `lanes/` holds the briefs. `RUN` builds with
[hipdnn-superbuild](../hipdnn-superbuild/SKILL.md) and tests with
[hipdnn-superbuild-test](../hipdnn-superbuild-test/SKILL.md); install those skills alongside this one.

Prerequisites: network access, an authenticated `gh` with read access, `git` 2.38 or later (`pin.py` uses `git merge-tree --write-tree`), and Python 3 (it runs the scripts under
`$SKILL/scripts/`); `pre-commit` and `pytest` for the lanes that run validators. `RUN` and verifiers also need
a ROCm install (Linux: `/opt/rocm` or `ROCM_PATH`; Windows: the wheel SDK that [hipdnn-superbuild](../hipdnn-superbuild/SKILL.md) provisions), CMake, Ninja and about 25 GB free in `$SCRATCH`; without them `defect` and
`test-gap` findings stay `UNVERIFIED` and the review closes only through human
`ACCEPTED-RISK` decisions, recorded with `ledger.py set` (`PIN.md` records `Build host`).

Platforms: Linux, macOS and Windows. On Windows run the skill from Git Bash (Git for Windows): the runbook's shell blocks need
`bash`, `awk`, `sed`, `grep` and `comm`. Blocks write the interpreter as `python3`; on Windows that name is the Microsoft Store
stub, so use `python` or `py -3` instead. Scripts read and write UTF-8 and LF whatever the host's locale.

## Entry contract

Ask for what is missing; record it in `PIN.md`:

| Input | Notes |
|---|---|
| Target | PR URL or number, `branch:<name>`, or `<base>..<head>`; `intent:<file>` (what it is for, how tested, `Known issues`; required without a PR) holds the intent document, whose `Known issues` section lists open items the author already knows, numbered `K1`, `K2` and so on, and never renumbered; without a PR, M4 and M5 are `N/A (no PR)` ([policy.md](policy.md) *M-rules*) |
| `evidence:<dir>`, `scratch:<dir>`, `clone:<dir>` | `$EVIDENCE` outside the product tree; disposable `$SCRATCH`; optional clone (default `$SCRATCH/clone`) |
| `terms:<file>`, `knowledge:<dir>` | Optional private terms, one fixed string per line, outside the repository; optional `$KNOWLEDGE` store |
| `round:<N>` | Default: one more than the highest `$EVIDENCE/r<k>/`; 1 when none |
| Agent limits | Ask the user or the host for `Agents:` (maximum concurrent agents), `Cores:` (CPU cores for builds and tests) and `Mutation:` (a host recipe for mutating a built tree, or none); the skill assumes no values. They go into `dispatch-context.md` ([RUNBOOK.md](RUNBOOK.md) *Fan out*) |

Local builds and host-only tests are allowed. GPU, remote or shared compute, and every write
to GitHub, Jira or a remote branch need the user's approval for that exact action. The
review is advisory: the user posts the draft comments. Write only `$EVIDENCE`, `$KNOWLEDGE`
and `$SCRATCH`, never the product tree.

## Completion

A round is complete when its gates hold and `REPORT.md` and `LEDGER.md` exist for the head.
The review is closed (`Closure: CLOSED`) when no `BLOCKING` or `IMPORTANT` finding is `OPEN`
or `UNVERIFIED` and no `STOP` or decision is unresolved. Otherwise `Closure: OPEN` lists what
blocks it and *Iterate* starts the next round; the user decides when to stop.
