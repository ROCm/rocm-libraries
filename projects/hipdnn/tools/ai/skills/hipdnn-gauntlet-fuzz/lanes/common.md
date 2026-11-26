# Lane contract

Every lane reads this file, `dispatch-context.md` (its first message), its own brief, the engine brief (`Brief:`), the ledger
(`Ledger:`) and [../findings.md](../findings.md). Triage and verify lanes also read [../policy.md](../policy.md).

## Agent kinds

Each brief names the kind of agent that runs the lane:

| Kind | Means |
|---|---|
| `scout` | A read-only research subagent: reads the source tree and the run files, edits nothing, runs no GPU command |
| `task` | A subagent that may also run commands (builds, GPU replays) within `Approved`, and write under `R/` |

Without subagents, the orchestrator plays each role in turn, keeping to the same limits.

## Variables (from `dispatch-context.md`)

| Name | Meaning |
|---|---|
| `R` | The round directory; write only `R/lanes/<LANE>.md`, `R/triage/<LANE>.md`, `R/verify/<ID>.md`, and scratch under `R/work/<LANE>/` |
| `WT`, `WT_BASE` | The PR and base worktrees (read-only) |
| `SKILL` | This skill's directory |
| `Brief`, `Ledger` | The engine brief and the engine ledger |
| `Approved` | The GPU plan the user approved; `none: read-only` means no GPU, build or machine command |

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
- Write the report to the absolute path the dispatch gives, in the format of [../findings.md](../findings.md) *Lane report*, then stop.
  End early with `STOP: <reason>` as the report's first line when an input is missing.
