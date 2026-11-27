---
name: split-to-stack
description: Decompose an oversized branch, a "from this branch up" range, or a whole stack of PRs into a lossless, linear stack of single-purpose branches. Plans the split from a hunk ledger without building or testing repo code, then stops for approval; with --implement it also builds the branches and proves the top tree equals the source after pre-commit fixes. Tests every branch only if asked. Local only unless publishing is explicitly requested. Use for /split-to-stack, or when asked to split, decompose, re-cut, or restructure a branch, PR, or stack that is too big to review.
---

# Split to Stack

**Announce at start:** "I'm using the split-to-stack skill to decompose this work into a linear stack of smaller, reviewable branches."

## Overview

Decompose a branch, a range of stacked branches, or a whole stack into one linear stack of single-purpose branches. Dependencies decide which orders are valid; among valid orders, fixes and cleanup go at the bottom and features last. The top of the stack has the same tree as the source after the pre-commit hooks, which proves nothing was lost. The product is a restructure plan and a stack of branches, delivered quickly, not a code review. It works on committed history only: uncommitted changes in the checkout are committed or stashed at start, as the user chooses. The default run stops after the plan for approval; the only repository code a plan-only run executes is the pre-commit hooks (Phase 0e).

    S  (source branch(es), never modified)
    │  Phase 0: scope, target, survey; copy S, run hooks, bring the copy up to date,
    │           run hooks again  →  S'
    │  Phase 1: linearize S' onto B  →  L        (always; a no-op, L = S', when S' is already linear on B)
    │  Phases 2–5: ledger, classify, group, order  →  plan files
    │  Phase 6: proposal to the user                (default run stops here)
    │  Phases 7–8: build the stack with scripts, verify, optionally test   (--implement or after approval)
    └  Phase 9: report, local gh stack, cleanup; publishing only on explicit request

## Invocation and scope

The scope is always explicit and never inferred.

| Request | Scope |
|---|---|
| `/split-to-stack <branch\|PR number\|PR URL>` ("split this branch") | only that branch |
| `/split-to-stack --from <branch\|PR>` ("split from this branch up") | that branch and every branch stacked above it |
| `/split-to-stack --stack <branch\|PR>` ("split this whole stack") | every branch of the stack the branch belongs to |

Flags: `--implement` (build and verify straight after the plan, no approval wait; see Stops), `--target <branch>` (target proposal for Phase 0b).

- Whenever the user says "stack", ask first: "Do you mean a `gh stack`, or plain stacked branches (PR base chains)?" Record the answer in `run.md`.
- For `--from` and `--stack` the Survey finds the branches (`references/survey-and-update.md`, 0a). The orchestrator shows the list with each branch's PR, author and review state and **waits for confirmation before continuing**. `--from` on the top branch is a single-branch scope; `--stack` on a branch that is in no stack reports that and asks what the user meant.
- Single branch: only that branch is re-cut. Branches stacked on top of it (dependents) are restacked onto the new top branch, with the user's permission, and stay members of the stack.
- The new branches go under `users/<you>/` (`references/orchestrator.md`).
- No PRs at all (a local branch) is fine: there is nothing to retire and no thread mapping.
- Tests run only if the user chooses them, asked once (`references/orchestrator.md`, Tests).

## Orchestration

The main session is the **orchestrator**; each phase's work is done by a **subagent** with a fresh context. Agents communicate only through files in the run directory. Symbols (`S`, `T`, `P`, `S'`, `B`, `L`, `$L`, `$SPLIT`, `$WT`, `$CO`, `$BASE_REMOTE`, `$PUSH_REMOTE`, `$RUN`, `$BR(NN)`, `<you>`) are defined in `references/common.md`.

**Invoking this skill is an explicit request for subagents.** The user asking for `/split-to-stack`, or naming this skill, authorizes the orchestrator to spawn the subagents below without asking again. The orchestrator does not do a named agent's phase work in its place.

**Orchestrator**
- The only agent that talks to the user. Records every decision in `run.md`.
- Never names candidate groupings, categories or layer labels (for example "C++" and "Python") in a subagent task. Commit readers, the Inventory and the Planner derive them from the diff; the orchestrator only passes inputs, outputs and the files to read.
- Passes the Planner any earlier review findings or notes about the branch that bear on grouping, marked "claims to check, not facts" (this is the only exception to the previous rule: facts and suspicions go in; groupings do not).
- Starts subagents; in every task names the exact files the agent must read (always `references/common.md`, its phase file, and the shared files listed for its role) plus its input and output paths.
- Starts every subagent with `fork_turns: "none"` and no `model`. The Survey, Updater, Linearizer, commit readers and Inventory get `reasoning_effort: "medium"`; the Planner, attack agents and Builder inherit the session's effort.
- Checks each output file before the next phase and enforces the gates below.
- Reads summaries, not raw diffs.

**Subagents**
- Read only their listed inputs; write only their own output files; return `DONE <file>` or `BLOCKED <file>`. Exceptions: the Builder may edit the plan files during the Phase 8 fix loop, and the orchestrator writes `commits.md` and `run.md`.
- Never ask the user anything: a question goes into the output file under `## Blocked`; the orchestrator asks, records the answer, and starts a new agent for the phase.
- At most one agent with write access runs at a time. Read-only agents may run in parallel (never more than the free agent slots).

**Run directory:** `RUN=$(git rev-parse --path-format=absolute --git-common-dir)/split-to-stack/<key>/` — always absolute. `<key>` is `<mode>-<name>`: `<mode>` is `branch`, `from` or `stack`; `<name>` is the head branch of the argument (a PR number or URL is resolved first with `gh pr view`; the remote prefix is removed), with `/` replaced by `__`. It lives inside `.git`. Print its path. Layout and file formats: `references/files.md`.

| Phase | Agent | Writes | Reads (besides common.md) | Output |
|---|---|---|---|---|
| 0a–0d | Survey | no | `survey-and-update.md` | `survey.md` (scope, target, merges, PRs, tools); `pr.md` when PRs are involved |
| decisions | orchestrator | `run.md`, user refs and their remote copies (approved fast-forward of the local target; approved Setup sync) | `orchestrator.md`, `workspace.md`, `survey.md`, `survey-and-update.md` (Setup sync) | scope, T, P, kept/re-cut PRs and ownership, sync, update answer |
| 0e | Updater | `$WT`, refs | `survey-and-update.md`, `workspace.md` | `update.md`: S', B, hook commits and findings, merge residue and duplicate scan, conflicts |
| 1 | Linearizer | `$WT`, refs | `linearize.md`, `concepts.md`, `workspace.md` | `linear.md` (`$L`, or "no-op: L = S'") |
| 2 | Commit readers (parallel batches) | no | `analyze-and-plan.md` | `commits/<sha>.md` |
| 2 | Inventory | `ledger.tsv` | `analyze-and-plan.md`, `concepts.md`, `files.md` | `ledger.tsv`, `inventory.md` |
| 3–5 | Planner (one agent) | plan files | `analyze-and-plan.md`, `concepts.md`, `files.md` | `groups.tsv`, `plan_assign.tsv`, `transitions.tsv`, `states/`, `plan.md` (incl. `## Grouping critique`) |
| 6 | orchestrator | `run.md` | `analyze-and-plan.md` (Phase 6) | proposal; approval recorded in `run.md` |
| 7–8 | Builder (one agent) | refs, `$WT`, plan files (8.2 fixes) | `build-and-test.md`, `concepts.md`, `files.md`, `workspace.md` | `stack.tsv`, `stack.md`, `tests.md` |
| 8.1 | orchestrator | — | `build-and-test.md` | independent `verify_stack.py` run |
| 9 | orchestrator | gh stack state and checked-out branch in `$CO` (approved), `$WT`, helper refs, remote (only on request) | `publish-and-cleanup.md` | final report, cleanup, publishing on request |

Wherever a phase text says "ask the user" or "stop and ask", a subagent writes it under `## Blocked`.

**Order at start:**
1. Resolve the argument to its head branch and compute `<key>`; check whether `$RUN` already exists: if so, follow `references/resume.md` before creating anything.
2. Read `references/orchestrator.md`. Find `$CO` (`git rev-parse --show-toplevel` of the invoking directory), resolve `$BASE_REMOTE`, `$PUSH_REMOTE` and then `<you>` (`orchestrator.md`), and record them. Create `$RUN` and `run.md` (scope, flags, `<you>`, the remotes, `$CO`). Settle the stack kind if the user said "stack".
3. With `--implement`: ask the Tests question now (`orchestrator.md`).
4. Run the Survey (read-only; it may update remote-tracking refs by fetching). Confirm with the user the branch list, the target (the orchestrator performs an approved fast-forward of the local target itself and records `T`), kept PRs and their ownership, and (if the copy is behind the target) whether to bring it up to date. Then make `$CO` clean and run the Setup sync (`survey-and-update.md`, Clean checkout and Setup sync): every branch of the stack equals its remote copy before anything else happens.
5. Now that S is confirmed, check `$SPLIT` and `$SPLIT/*`: if helper refs exist without a `run.md`, show them with their SHAs and let the user choose another name or delete them (Restart, `resume.md`).
6. Create an **empty** `$WT` directory (`mktemp -d`), record it in `run.md`, and start the Updater, which runs `git worktree add --detach` in it. After Phase 1, check the small-input path (`orchestrator.md`).

**Gates** (the orchestrator enforces them; scripts are in `scripts/`):
1. Phase 1: `tree(L) == tree(S')` (trivially true for the no-op). The Inventory starts only after this gate.
2. End of Phase 5 (and after every plan edit or renumbering): `check_plan.py` passes (every ledger entry assigned, steps and names valid, the ledger reproduces, no existing branch name is taken). Run it with `--check-remote "$PUSH_REMOTE"` before Phase 6, so names are checked on the remote before the user approves them.
2b. End of Phase 5: `plan.md` has a `## Grouping critique` section that passes the check in `analyze-and-plan.md` (no group is split on language, directory, test framework, size or a dependency alone). With three or more groups, the attack pass on the critique has run and its result is in `run.md`. Only then run Phase 6.
3. Phase 6: the plan is approved and recorded verbatim in `run.md` (`approval: user`, or `approval: auto (--implement)`), and `groups.tsv`, `plan_assign.tsv`, `transitions.tsv` and `states/` are copied to `$RUN/approved/`. The default run stops here.
4. Phase 7: `build_stack.py` exits 0 (`tree(top) == tree(L)`).
5. Phase 8.1: the orchestrator runs `verify_stack.py` itself (independent of the Builder).
6. If tests were chosen: the base must pass before anything is built (`build-and-test.md`).

**`--implement` stops** (it never skips these): the current user cannot be determined; the checkout has uncommitted changes; the Setup sync has any action or conflict; dependents are ready to be moved; a conflict is ambiguous (in the update, the Phase 1 replay or a dependent restack); Phase 1 finds an octopus merge or cannot reach `tree(L) == tree(S')` in 3 passes; the plan contains a hand-written `stored:` state (note `synthesized`) or any transition code, or a Phase 8 failure leads to a transition proposal (all are hand-written code nobody has reviewed; a `subtractive:` state that passes `check_plan.py` is not a stop, because it is proven to add no text); the base tests fail and the user has not opted out; a PR in scope or kept is queued for merge; the ledger is empty (everything already landed).

**Validation between phases:** every SHA in an output file exists, and the `$SPLIT/*` refs and `$BR(NN)` branches match `update.md`, `linear.md`, `stack.tsv` and `refs.tsv`.

Scripts are in `scripts/` (`references/files.md` lists them); `test_split_to_stack.py` is their pytest suite.
