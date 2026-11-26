# Lane contract

Every lane reads this file, `dispatch-context.md` (its first message), its own brief and the engine brief (`Brief:`, including its
**code cards**). It reads [../findings.md](../findings.md) only when its task does not restate the report format. Triage and verify
lanes also read [../policy.md](../policy.md); search lanes do not need it.

## Agent kinds

Each brief names the kind of agent that runs the lane:

| Kind | Means |
|---|---|
| `scout` | A read-only research subagent (about 100K tokens of context): reads the source tree and the run files, edits nothing, runs no GPU command. Returns the report as its final message |
| `task` | A subagent with a much larger context that may also run commands (builds, GPU replays) within `Approved`, and write under `R/`. Use it for lanes that must read a whole plan, matcher and kernel |

Without subagents, the orchestrator plays each role in turn, keeping to the same limits.

## Variables (from `dispatch-context.md`)

| Name | Meaning |
|---|---|
| `R` | The round directory; write only `R/lanes/<LANE>.md`, `R/triage/<LANE>.md`, `R/verify/<ID>.md`, and scratch under `R/work/<LANE>/` |
| `WT`, `WT_BASE` | The PR and base worktrees (read-only) |
| `SKILL` | This skill's directory |
| `Brief`, `Ledger` | The engine brief and the engine ledger |
| `Approved` | The GPU plan the user approved; `none: read-only` means no GPU, build or machine command |

## Context budget

Your context is limited, and a lane that overflows it returns partial reads and `[INFERENCE]` claims. So:
- Read the brief and its code cards first. They quote the code you will keep needing.
- Find with `grep`, then read by `path:line` range. Never read a file over 300 lines whole; never re-read what a card already quotes.
- Do not open knowledge-base articles: the brief carries the rows that matter. If a row is missing, say so under `## Handoffs`.
- Stop after about 25 tool calls. Write the report, and under `## Handoffs` list what you did not read and why it might matter.
- A `task` agent writes each idea to `R/lanes/<LANE>.md` as it finds it, then fixes the format at the end.

## Rules

- **Read-only on the source.** Never edit `WT`, `WT_BASE`, the spaces, the generator or the harness. Propose changes in `Place:`
  or `Fix:`; the orchestrator makes them.
- **Cite what you read.** Every claim carries a `path:line` read at the pinned commit, or a run file under `R/`. Mark anything you
  did not observe `[INFERENCE]`. A guess is fine if labelled.
- **One concrete graph per idea.** Axis values, dims, strides, attributes: something a space row or a hand-built graph can be made
  from. "Large head dims" is not an idea; `hd=(264,264), sq=1, skv=257, fp32, bshd` is.
- **Judgeable or say why not.** Before proposing a graph for a graded space, check it against [../policy.md](../policy.md)
  *Judgeability* and the brief's conditioning section. An unjudgeable graph is proposed as a probe with judgeable inputs, or as
  `unreachable`.
- **Already known is not new.** An idea or finding that matches a ledger row cites that row's ID and adds only new evidence.
- **At most 10 ideas** for a search lane, most promising first; a triage lane files one finding per cause.
- **GPU use** only within `Approved`. One GPU per command; cap build parallelism at 8 jobs; delete any dump you create.
- **Public text.** Nothing a lane writes leaves the workspace, but proposed space descriptions and commit text are public: no host,
  site or user names in those.
- Write the report to the absolute path the dispatch gives, then stop. End early with `STOP: <reason>` as the report's first line when
  an input is missing.

## Report format (search lanes)

`ledger.py` parses it: plain lines, no bullets, no code fences, every label at the start of its line. In the template, `<LANE>`, `<space>`,
`<axis>` and `<family>` stand for the lane name, a space id, one of its axes and one of its probe families.

```
# <LANE>: <one-line scope>
Read:     the files and ranges read
## Ideas
ID: <LANE>-p1
Graph:     one concrete graph: op, axis values, dims, strides, attributes
Why:       why it could break, with path:line (alternatives and caveats go here)
Expect:    exactly ONE of PASS | DECLINED | WRONG | CRASH | ACCEPT-UNSERVABLE | DECLINE-SERVABLE | FRONTEND-GAP | BACKEND-MANGLE | KERNEL-UNREACHED
Judgeable: yes | bounded: <the input range that keeps it judgeable> | no: <why no fixed tolerance can judge it>
Covered:   yes: <space.axis> | partly: <space.axis, what is missing> | no
Place:     axis <space>.<axis>=<values> | probe <space>.<family> expect <declined|declined_or_pass> | kernel-rows <kernel> | hand-probe | unreachable: <reason>

ID: <LANE>-p2
...
## Checked, no idea
one line each: what was looked at and ruled out
## Handoffs
<LANE or ORCH>: what another lane or the orchestrator should look at, and what you did not read
```

The ID line is `ID: <LANE>-p<n>`, never the bare id. `Expect:` is one token with no trailing period or comment. Write `None: <why>`
under `## Ideas` when nothing was found. The full format of triage and verifier reports is in [../findings.md](../findings.md).
