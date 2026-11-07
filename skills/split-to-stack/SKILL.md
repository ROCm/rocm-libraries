---
name: split-to-stack
description: Decompose an oversized branch, a "from this branch up" range, or a whole stack of PRs into a lossless, linear stack of single-purpose branches. Plans the split from a hunk ledger without building or testing repo code, then stops for approval; with --implement it also builds the branches and proves the top tree equals the source after pre-commit fixes. Tests every branch only if asked. Local only unless publishing is explicitly requested. Use for /split-to-stack, or when asked to split, decompose, re-cut, or restructure a branch, PR, or stack that is too big to review.
---

# Split to Stack

**Announce at start:** "I'm using the split-to-stack skill to decompose this work into a linear stack of smaller, reviewable branches."

## Overview

Decompose a branch, a range of stacked branches, or a whole stack into one linear stack of smaller branches, each with a single purpose. Dependencies decide which orders are valid; among valid orders, independent low-risk changes (fixes, cleanup) go at the bottom so they land first and features go last. The top of the stack has the same tree as the source after the pre-commit hooks have been applied to it, which proves nothing was lost.

The product is a **restructure plan and a stack of branches**, delivered quickly. It is not a code review. The default run stops after the plan for approval. The only repository-configured code a plan-only run executes is the pre-commit hooks (Phase 0e).

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
- Single branch: branches stacked on top of it are not re-cut or rebased. They are listed as dependents left unchanged, and the user is told they need rebasing onto the new top branch.
- The new branches go under `users/<current user>/` (`references/common.md`, Naming).
- No PRs at all (a local branch) is fine: there is nothing to retire and no thread mapping.

**Tests.** Ask once: "Run the tests on every branch, or skip them (for example to inspect the branches first)?" With `--implement`, ask it in the order-at-start step below; in plan-only mode (the default), ask it when the user asks to implement. Record the answer in `run.md`. A user who skipped can ask later to test an existing stack: the orchestrator creates a fresh `$WT` and runs `references/build-and-test.md` 8.2 against the SHAs in `stack.tsv` (fixes then rebuild with `--refs update`; allowed only while the branches are unpublished).

**Small-input path.** After the Survey, the Updater and the Linearizer (always run as usual) the orchestrator measures: `git diff --numstat B $L` (hand-written = excluding the default generated globs of `concepts.md`), `git rev-list --count B..$L`, and the merges in `B..S'`. If all hold: ≤ 800 hand-written changed lines, ≤ 15 commits, no merges from other branches, one component (the first two path components under `projects/`, `shared/` or `dnn-providers/`), then a single Planner does the Inventory and the plan (reading commits itself, no commit readers, no separate Inventory), and one Builder does Phases 7–8. It follows the same files, scripts, gates and table. The Planner never runs hooks or asks the user; if it finds an entangled entry or a broken threshold it stops with `## Blocked: needs-full-pipeline`; the orchestrator then keeps `ledger.tsv` and `inventory.md`, deletes the plan files, and runs the commit readers and a fresh Planner.

## Symbols

`S` source branch (top of the scope, never modified); `T` resolved target SHA; `P` topmost kept (external) parent, if any; `S'` the copy of S after hooks and update; `B` base of the split; `L` the linear chain on B with `tree(L) == tree(S')` (equal to `S'` when the Phase 1 no-op check applies); `$L` the SHA of L, recorded in `run.md`; `$SPLIT` helper-branch prefix; `$WT` scratch worktree; `$RUN` run directory; `$BR(NN)` the final branch of group NN. Full definitions: `references/common.md`.

## Orchestration

The main session is the **orchestrator**; each phase's work is done by a **subagent** with a fresh context. Agents communicate only through files in the run directory.

**Invoking this skill is an explicit request for subagents.** The user asking for `/split-to-stack`, or naming this skill, authorizes the orchestrator to spawn the subagents below without asking again. The orchestrator does not do a named agent's phase work in its place.

**Orchestrator**
- The only agent that talks to the user. Records every decision in `run.md`.
- Never names candidate groupings, categories or layer labels (for example "C++" and "Python") in a subagent task. Commit readers, the Inventory and the Planner derive them from the diff; the orchestrator only passes inputs, outputs and the files to read.
- Passes the Planner any earlier review findings or notes about the branch that bear on grouping, marked "claims to check, not facts" (this is the only exception to the previous rule: facts and suspicions go in; groupings do not).
- Starts subagents; in every task names the exact files the agent must read (always `references/common.md`, its phase file, and the shared files listed for its role) plus its input and output paths.
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
| decisions | orchestrator | `run.md`, user refs (approved fast-forward of the local target only) | `survey.md` | scope, T, P, kept/re-cut PRs and ownership, update answer |
| 0e | Updater | `$WT`, refs | `survey-and-update.md` | `update.md`: S', B, hook commits and findings, merge residue and duplicate scan, conflicts |
| 1 | Linearizer | `$WT`, refs | `linearize.md`, `concepts.md` | `linear.md` (`$L`, or "no-op: L = S'") |
| 2 | Commit readers (parallel batches) | no | `analyze-and-plan.md` | `commits/<sha>.md` |
| 2 | Inventory | `ledger.tsv` | `analyze-and-plan.md`, `concepts.md`, `files.md` | `ledger.tsv`, `inventory.md` |
| 3–5 | Planner (one agent) | plan files | `analyze-and-plan.md`, `concepts.md`, `files.md` | `groups.tsv`, `plan_assign.tsv`, `transitions.tsv`, `states/`, `plan.md` (incl. `## Grouping critique`) |
| 6 | orchestrator | `run.md` | `analyze-and-plan.md` (Phase 6) | proposal; approval recorded in `run.md` |
| 7–8 | Builder (one agent) | refs, `$WT`, plan files (8.2 fixes) | `build-and-test.md`, `concepts.md`, `files.md` | `stack.tsv`, `stack.md`, `tests.md` |
| 8.1 | orchestrator | — | `build-and-test.md` | independent `verify_stack.py` run |
| 9 | orchestrator | `.git/gh-stack`, `$WT`, helper refs, remote (only on request) | `publish-and-cleanup.md` | final report, cleanup, publishing on request |

Wherever a phase text says "ask the user" or "stop and ask", a subagent writes it under `## Blocked`.

**Order at start:**
1. Resolve the argument to its head branch and compute `<key>`; check whether `$RUN` already exists: if so, handle **Resume** (below) before creating anything.
2. Resolve `<you>` (`common.md`, Naming) and record it. Create `$RUN` and `run.md` (scope, flags, `<you>`). Settle the stack kind if the user said "stack".
3. With `--implement`: ask the Tests question now.
4. Run the Survey (read-only; it may update remote-tracking refs by fetching). Confirm with the user the branch list, the target (the orchestrator performs an approved fast-forward of the local target itself and records `T`), kept PRs and their ownership, and (if the copy is behind the target) whether to bring it up to date.
5. Now that S is confirmed, check `$SPLIT` and `$SPLIT/*`: if helper refs exist without a `run.md`, show them with their SHAs and let the user choose another name or delete them (the Restart procedure).
6. Create an **empty** `$WT` directory (`mktemp -d`), record it in `run.md`, and start the Updater, which runs `git worktree add --detach` in it.

**Gates** (the orchestrator enforces them; scripts are in `scripts/`):
1. Phase 1: `tree(L) == tree(S')` (trivially true for the no-op). The Inventory starts only after this gate.
2. End of Phase 5 (and after every plan edit or renumbering): `check_plan.py` passes (every ledger entry assigned, steps and names valid, the ledger reproduces, no existing branch name is taken). Run it with `--check-remote origin` before Phase 6, so names are checked on the remote before the user approves them.
2b. End of Phase 5: `plan.md` has a `## Grouping critique` section that passes the check in `analyze-and-plan.md` (no group is split on language, directory, test framework, size or a dependency alone). With three or more groups, the attack pass on the critique has run and its result is in `run.md`. Only then run Phase 6.
3. Phase 6: the plan is approved and recorded verbatim in `run.md` (`approval: user`, or `approval: auto (--implement)`), and `groups.tsv`, `plan_assign.tsv`, `transitions.tsv` and `states/` are copied to `$RUN/approved/`. The default run stops here.
4. Phase 7: `build_stack.py` exits 0 (`tree(top) == tree(L)`).
5. Phase 8.1: the orchestrator runs `verify_stack.py` itself (independent of the Builder).
6. If tests were chosen: the base must pass before anything is built (`build-and-test.md`).

**`--implement` stops** (it never skips these): the current user cannot be determined; a conflict is ambiguous (in the update or in the Phase 1 replay); Phase 1 finds an octopus merge or cannot reach `tree(L) == tree(S')` in 3 passes; the plan contains a hand-written `stored:` state (note `synthesized`) or any transition code, or a Phase 8 failure leads to a transition proposal (all are hand-written code nobody has reviewed; a `subtractive:` state that passes `check_plan.py` is not a stop, because it is proven to add no text); the base tests fail and the user has not opted out; a PR in scope or kept is queued for merge; the ledger is empty (everything already landed).

**Validation between phases:** every SHA in an output file exists, and the `$SPLIT/*` refs and `$BR(NN)` branches match `update.md`, `linear.md`, `stack.tsv` and `refs.tsv`.

**Resume:** `run.md` records the last completed phase and the SHAs of S, T, P, B, S', L and the `$WT` path. If `$RUN` exists at start, show the last completed phase and ask: resume, restart, or abort. A run that stopped at "nothing to split" is recorded as such and restarts silently. When an agent stopped with `## Blocked` in the middle of an operation in `$WT` (a merge, a cherry-pick, a hook pass), its output file records the exact `$WT` state (HEAD SHA, the operation in progress, conflicted files, the hook commits made so far by SHA); the replacement agent starts by reading it and either continues with the user's answer or resets `$WT` to the recorded start SHA (`merge --abort` or `cherry-pick --abort`, then `switch --detach <start>`) and re-applies the recorded commits instead of re-running the hooks. Resume from the first phase without a valid output file, and only if the recorded SHAs still match the refs and the ledger's `base` and `tip` match `run.md` (`base` = B, `tip` = `$L`). A partially built stack is resumed with `build_stack.py --refs update` (missing branches are created; the script is all-or-nothing, so no half-moved stack exists). **Restart** has its own short procedure: print every `$SPLIT/*` ref, every branch recorded in the old `refs.tsv` and the `$WT` path, each with its SHA; ask per item; `git worktree remove` for `$WT`; `git branch -D` only for the helper refs and for names in the old `refs.tsv` whose SHA still matches (never any other `users/<you>/…` branch); say that deleting `$RUN` deletes `refs.tsv`, after which leftover `$BR(NN)` can no longer be updated by the new run, and that a deleted branch is recoverable only from the SHA printed.

## Reference files

| File | Read by |
|---|---|
| `references/common.md` | every agent |
| `references/concepts.md` | Linearizer, Inventory, Planner, Builder |
| `references/files.md` | Inventory, Planner, Builder |
| `references/survey-and-update.md` | Survey, Updater |
| `references/linearize.md` | Linearizer |
| `references/analyze-and-plan.md` | commit readers, Inventory, Planner, orchestrator (Phase 6) |
| `references/build-and-test.md` | Builder, orchestrator (8.1) |
| `references/publish-and-cleanup.md` | orchestrator (Phase 9) |
| `scripts/` | `build_ledger.py`, `check_plan.py`, `build_stack.py`, `verify_stack.py`, `render_plan.py`, `extract_state.py`, `selftest.py`, and the shared module `splitlib.py` |
