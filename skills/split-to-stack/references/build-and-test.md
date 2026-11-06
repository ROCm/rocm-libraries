# Phases 7–8: build, verify, and (optionally) test

Read by the Builder and, for Phase 8.1, the orchestrator. Also read `concepts.md` and `files.md`.

## Phase 7: Build

**Before building:** the plan is approved and recorded in `run.md` (`approval: user` or `auto (--implement)`); `scripts/check_plan.py` passes; every `$BR(NN)` name passes `git check-ref-format --branch` and is free locally and on `origin`, or is recorded in `refs.tsv` at its current SHA (a rebuild or resume; `check_plan.py` checks this); no `$BR(NN)` is checked out in any worktree.

**Build.** Run `scripts/build_stack.py --repo <repo> --run $RUN --prefix users/<you>/ --refs create`. The script computes every branch from the base and the plan (no patch application, no agent edits), creates one commit per group chained from B, and exits non-zero unless `tree(top) == tree(L)`; on a tree mismatch it records nothing and moves no ref. Otherwise it writes `stack.tsv` (the commits it built) and then moves all `$BR(NN)` branches in one atomic transaction; if that transaction is refused, no branch moved and `refs.tsv` (what was published) is unchanged. A non-zero exit is a plan problem: read the printed differing files or the file/directory clash, fix the plan (below) and rebuild. When `refs.tsv` exists (a rebuild or a resume), use `--refs update`, which creates missing branches and moves existing ones only if they still point at the SHA in `refs.tsv`.

Commit content comes from `groups.tsv`:
- **Author:** the author of the source commits the group's entries came from; if several contributed, the one with the most lines, plus `Co-authored-by:` trailers. The committer is whoever runs the skill.
- **Subject:** `type(scope): subject`, the type being the group's type.
- **Body:** the group's `summary`. **Trailers** and issue keys carried over from the source commits for the group's entries; none invented; no `Split-from:` trailer.
- One commit per branch. No merge commits.

Write `stack.md`: each branch with its SHA and notes. Keep it current after every rebuild.

## Phase 8: Verify

### 8.1 Losslessness

Run `scripts/verify_stack.py --repo <repo> --run $RUN --prefix users/<you>/ --tip $L --refs`. It must print `STACK VERIFY PASS`: the branches are the groups, linear chain on B, one parent each, no merges, between consecutive branches only paths the plan names change, `tree(top) == tree($L)`. The orchestrator runs it again itself after the Builder returns. After any rebuild, check it again.

### 8.2 Tests

The user's choice (asked at start, or at implementation start in plan-only mode) is **every branch** or **skip**.

**Skip.** Run no tests: no base gate, no source-tip run, no hooks. Record every branch as *not run* in `tests.md`; the final report says the stack is untested.

**Every branch.** Use the approved test plan.
0. **Hardware first.** Before choosing commands, check what the machine has: `rocminfo | rg gfx` and `rocm-smi --showproductname`. A GPU that `rocminfo` lists is usable without any permission escalation; if none is listed in a sandboxed session, say so and ask the user to grant device access instead of silently skipping GPU tests. Record the architecture in `tests.md`: a test written for another architecture (for example gfx1250 or the OCP FP8 formats) fails on the one you have, and belongs under "known-broken on the base", not to a branch. Do not assume the target architecture of the tests is the architecture of the machine.
1. **Base gate.** (After Phase 7 created the branches and before any branch is compiled or tested.) Run on B in `$WT` (`git -C "$WT" switch --detach B`), with the build directory in `$RUN/build`, the selected test commands **whose targets exist on B** (the union of the branches' commands; a command for a target a branch adds is recorded as "n/a on base"). Hooks are not part of the base gate. The base must pass. If any test fails, **stop** and report it under `## Blocked`: the user either fixes the base first, or gives an explicit opt-out that **names the failing tests**. Record an opt-out in `run.md` and `tests.md`; opted-out tests are known-broken and are never blamed on a branch. Nothing is built or walked until the base passes or the opt-out is recorded, also under `--implement`.
2. **Source-tip run.** Run the same tests on L (the same tree as the top). Record the failing tests as **source problems**.
3. **Walk, bottom to top.** For each branch: `git -C "$WT" switch --detach $BR(NN)`, build, run its tests and its hooks. Collect the failures of the **whole stack** first; do not stop at the first failing branch. Classify each failure at branch NN:
   - fails on the base (opted out): known-broken, ignore;
   - also fails on the source tip: **source problem**: record and report it, do not fix it (a fix would change the final tree), no fix attempt is spent;
   - passes on the source tip: **split error**: this branch's cut is wrong.
4. **Fix all split errors, then rerun.** Find the branch that causes each split error, make every plan edit needed, rebuild once (below), and rerun the walk. Do not rerun after each single fix. Branches whose SHA did not change and already passed need no rerun. **Max 3 attempts** per problem, then show the remaining failure and ask the user.
5. Report what was built and run per branch, and every fix and every transition proposal.

**Hooks in the walk.** The source-tip run also runs the hooks over the union of all branches' changed files; its findings (hook id + path) are the source problems. On each branch run `pre-commit run --files <the branch's changed files>`, record the findings, then discard anything the hooks changed: `git -C "$WT" restore --staged --worktree .` and a `git -C "$WT" clean -fd -- <the branch's changed paths>`. The source is hook-clean (0e), so a finding on a branch is a split error unless the same hook id and path also appear in the source-tip run or in `update.md` under `## Hook findings` (source problems). Formatter output that differs only because a branch has an intermediate state is fixed like any split error: move the entry, or correct the stored state.

**Refactor, chore and style branches** must pass the existing tests without changing what they assert. Test-file changes in such a branch are limited to mechanical edits (renames, moved files, updated imports); check the branch's diff under test directories. If an assertion or expected value changes, the branch is not a refactor: change its type in `groups.tsv` (fix or feat), which may change its position, and rebuild.

A test that could not run (no GPU, missing dependency) is recorded as **not run**, never as passed. The final report says which branches were fully tested.

### Fixes: plan edit, then rebuild

A fix changes the plan, never a built branch. No hand edits, no amend, no restack. The allowed fixes are:
- move ledger entries between branches (for example a missing header include moves down to the branch that needs it) by editing `plan_assign.tsv`;
- reorder or merge branches (edit `groups.tsv` and `plan_assign.tsv`);
- correct a stored intermediate state under `$RUN/states/` (a wrong hand-written state is fixed by correcting the text).

After editing, regenerate any transition `stored` file whose path gained or lost an entry inside its window. Then re-run `check_plan.py` and `build_stack.py --refs update`: it rebuilds from the plan, keeps every branch below the first changed one at the same SHA, and moves a branch only if it still points at the SHA recorded in `refs.tsv`. Re-check 8.1.

**Transition code.** If a branch truly cannot stand alone without temporary compatibility code, do not add it: write the proposal under `## Blocked` as described in `concepts.md` (Transition code). If the user approves, add it to `transitions.tsv` and rebuild. A failure caused by a bad split is fixed by changing the split.

### `tests.md`

Per branch: commands, result (`pass`, `fail`, `not run`), classification of each failure (known-broken, source problem, split error), fixes made, transition proposals, rebuilds with the old and new SHAs, and any opt-out with the tests it names.

## Deviations from the approved plan

Changing a type, moving entries, merging groups or reordering after approval is allowed, but every fix is listed in `tests.md`, and the final report shows the deviations as a diff of `groups.tsv`, `plan_assign.tsv`, `transitions.tsv` and `states/` against `$RUN/approved/`. In the default (approved) mode a merge or reorder of more than one group needs the user first: the Builder writes it under `## Blocked`. Under `--implement` the deviations are listed but do not stop the run, except transition proposals.

## Original branch

The source branch is preserved untouched throughout.
