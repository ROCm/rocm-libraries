---
name: miopen-deep-review
description: "Multi-agent deep review of a MIOpen PR, branch or commit range. Scouts map each affected area, focused reviewer lanes run in parallel, every finding is verified by source proof, reproduction or mutation, and a ledger carries findings across rounds until no blocking or important one stays open. For a PR body or a quick single-pass review, use rocm-pr-quality."
argument-hint: "[<PR URL or number> | branch:<name> | <base>..<head>] evidence:<dir> scratch:<dir> [clone:<dir>] [intent:<file>] [terms:<file>] [knowledge:<dir>] [round:<N>]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob, Task
---

# MIOpen Deep Review

`$SKILL` is the directory this skill was loaded from (the directory containing this
`SKILL.md`). You are the orchestrator: execute [RUNBOOK.md](RUNBOOK.md). You pin the change, dispatch
scouts, lanes and verifiers, merge their output and write the report; you do not review
code. Your formats are in [report.md](report.md); [RUNBOOK.md](RUNBOOK.md) *Fan out* says
what each agent reads, and `lanes/` holds the briefs. `RUN` builds MIOpen standalone from
`projects/miopen` and runs `miopen_gtest` ([lanes/build-validate.md](lanes/build-validate.md)).

Prerequisites: network access, an authenticated `gh` with read access, `git` 2.38 or later (`pin.py` uses `git merge-tree --write-tree`), and Python 3 (it runs the scripts under
`$SKILL/scripts/`); `pre-commit` and `pytest` for the lanes that run validators. `RUN` and verifiers also need
a ROCm install (`/opt/rocm` or `ROCM_PATH`) with the MIOpen dependencies `rbuild prepare` installs (`projects/miopen/requirements.txt`), CMake, Ninja, `dvc` for the kernel databases, and about 60 GB free in `$SCRATCH`; without them `defect` and
`test-gap` findings stay `UNVERIFIED` and the review closes only through human
`ACCEPTED-RISK` decisions, recorded with `ledger.py set` (`PIN.md` records `Build host`).
Most of `miopen_gtest` needs a GPU; the `CPU_*` suites and the configure, ABI and naming checks do not.

Platforms: Linux. MIOpen's Windows build is exercised only through TheRock; the `ROCK` lane reads it, nothing here builds it.

## Entry contract

Ask for what is missing; record it in `PIN.md`:

| Input | Notes |
|---|---|
| Target | PR URL or number, `branch:<name>`, or `<base>..<head>`; `intent:<file>` (what it is for, how tested, `Known issues`; required without a PR) holds the intent document, whose `Known issues` section lists open items the author already knows, numbered `K1`, `K2` and so on, and never renumbered; without a PR, M4 and M5 are `N/A (no PR)` ([policy.md](policy.md) *M-rules*) |
| `evidence:<dir>`, `scratch:<dir>`, `clone:<dir>` | `$EVIDENCE` outside the product tree; disposable `$SCRATCH`; optional clone (default `$SCRATCH/clone`) |
| `terms:<file>`, `knowledge:<dir>` | Optional private terms, one fixed string per line, outside the repository; optional `$KNOWLEDGE` store |
| `round:<N>` | Default: one more than the highest `$EVIDENCE/r<k>/`; 1 when none |
| Agent limits | Ask the user or the host for `Agents:` (maximum concurrent agents), `Cores:` (CPU cores for builds and tests), `GPU:` (the local `gfx` architecture, or none) and `Mutation:` (a host recipe for mutating a built tree, or none); the skill assumes no values. They go into `dispatch-context.md` ([RUNBOOK.md](RUNBOOK.md) *Fan out*) |
| Scope notes | Ask the user for scope notes the change's description does not state, such as areas being deprecated (the rocMLIR solvers), or `none`. Free text; it goes into `dispatch-context.md` as `Scope:` and is informational unless it says "do not file findings in <area>" ([lanes/common.md](lanes/common.md) *Rules*) |

Local builds and host-only tests are allowed. GPU, remote or shared compute, and every write
to GitHub, Jira or a remote branch need the user's approval for that exact action. The
review is advisory: the user posts the draft comments. Write only `$EVIDENCE`, `$KNOWLEDGE`
and `$SCRATCH`, never the product tree.

## Completion

A round is complete when its gates hold and `REPORT.md` and `LEDGER.md` exist for the head.
The review is closed (`Closure: CLOSED`) when no `BLOCKING` or `IMPORTANT` finding is `OPEN`
or `UNVERIFIED` and no `STOP` or decision is unresolved. Otherwise `Closure: OPEN` lists what
blocks it and *Iterate* starts the next round; the user decides when to stop.
