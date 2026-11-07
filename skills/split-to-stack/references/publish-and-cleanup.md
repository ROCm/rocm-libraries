# Phase 9: report, local stack, cleanup, and publishing (orchestrator reads this file)

By default the skill creates local refs and registers a local `gh stack` only. Nothing is pushed, and no PR or comment is created, closed or edited, unless the user explicitly asks.

## 9.1 Report

List every ref created with its SHA: each `$BR(NN)` (the deliverable), `$SPLIT/00-updated` and `$SPLIT/00-linear` if they exist. Say which branches were fully tested, which were *not run*, any opt-outs, source problems found, the hook delta and unfixed hook findings, every deviation from the approved plan, and (single-branch scope) the dependents left unchanged that now need rebasing onto the new top branch. `$L` is the zero-diff linear version of S', useful for review and for any re-split.

## 9.2 Local gh stack (if `gh stack` is available)

Local only. `gh stack` has no `-C`, so run it as `(cd "$WT" && gh stack …)`. Before registering, check every name with `git rev-parse --verify refs/heads/<name>`: `gh stack init` **creates** a branch that does not exist, so a typo would silently create one.

1. **An existing local stack.** Only if `gh stack init` would conflict (the trunk, or a member of the new stack, is already tracked in a local gh stack in `.git/gh-stack`), copy `.git/gh-stack` to `$RUN/gh-stack.backup` first. Removing local tracking drops the whole old stack, so list every other branch that loses local tracking (kept parents, dependents) before asking. Removing local tracking needs `gh stack unstack --local` run with a branch of the old stack checked out, which would switch a checkout. **Never pass a stack number**: with a number, `unstack` acts "directly through the GitHub API", and the `--local` combination is not documented. So ask the user to run it (print the exact command and the checkout it needs), or ask to approve doing it in a scratch worktree they name. Do not edit `.git/gh-stack` by hand. If registering the new stack then fails, tell the user to copy the backup back (`cp $RUN/gh-stack.backup <git-common-dir>/gh-stack`).
2. **Register the new stack:** `(cd "$WT" && gh stack init --base <trunk> <new branches, bottom first>)`.
   - `<trunk>` is the target by default; register only if `git merge-base --is-ancestor B <trunk>` holds, otherwise tell the user to update the local target first.
   - When kept PRs exist and any of them is not the current user's, `<trunk>` is the **topmost kept branch** and the kept branches are not members (`gh stack submit`, `push` and `sync` push every member branch and update PR bases, so a member would be written to). `--base` takes one branch; the lower kept PRs stay in its history but are not managed by gh stack.
   - If all kept PRs are the user's and the user chose to include them, list them first, bottom first, then the new branches.
   - `init` may leave a branch checked out in `$WT`. Before any later rebuild, `git -C "$WT" switch --detach`; `build_stack.py` refuses a branch that is checked out in a worktree.
3. Print the `gh stack submit --auto` command for the user. Do not run it (see Publishing).

After the kept PRs land, GitHub retargets PRs whose base branch was deleted on merge, but the local trunk must be re-pointed to the target: tell the user to run `gh stack init --base <target> …` again (gh stack has no visible trunk-change command; this is unverified).

## 9.3 Worktree

Remove `$WT` only when no publishing is intended in this session: if the user said they will publish, or asks for it next, keep `$WT` (Publishing recreates it when it is gone). To remove: `git -C "$WT" status --porcelain`; untracked files left by hooks are removed with `git -C "$WT" clean -fd` (this acts only inside the scratch worktree; add `-x` if ignored files block removal), then `git worktree remove "$WT"`. If the run stopped early or `$WT` holds uncommitted state, keep it, say where it is, and say why.

## 9.4 Cleanup prompt

Only after the orchestrator's `verify_stack.py` run (8.1) has passed and the test results (or "not run") are recorded in `tests.md`, ask **one** prompt listing every helper item that exists, each with its recommended answer, and wait. Skip items that do not exist. Never ask about, or delete, `$BR(NN)`, the source branch `S`, or anything not listed.

- **a. `$SPLIT/00-updated` (<SHA>)**: the copy of `<S>` with the hooks applied and brought up to date. Recommended: keep until the PRs are merged, so it can be inspected.
- **b. `$SPLIT/00-linear` (<SHA>)**: the zero-diff linear chain; `tree($BR(top)) == tree(00-linear)` is verified (tests: <summary>). Needed only to re-split or compare. Recommended: keep until the PRs are merged.
- **c. Run directory `<absolute RUN path>`** (plan, ledger, stack and test notes, and the `gh-stack.backup` copy if one was made; deleting it deletes them). Recommended: keep until the PRs are merged, but delete its `build/` subdirectory now (builds can be many GB).

For each "delete" answer, resolve and print the exact target first (`git branch -D <ref>`, `rm -r "<absolute RUN>"`), delete only that item, then say what was removed. **A deleted branch cannot be restored from the reflog** (`git branch -D` removes its reflog). The skill prints each helper's SHA in the report; as long as the object has not been garbage-collected, `git branch <name> <sha>` brings it back. For "no" or no answer, keep it and print the one-line command for the user to run later. The Resume "restart" option uses this same procedure.

## Publishing: only on explicit request

If the user explicitly asks to publish, follow these rules.

- **Draft first.** Show the exact commands, PR titles and bodies, and comment text, and get approval before any external write. Mutate only what was requested, then verify. PR titles are the groups' `type(scope): subject` and bodies their `summary` (plus the dependency note "stacked on #<N>").
- **Fetch first.** Fetch the remote refs involved and record their SHAs. Re-check the PR states from 0d (auto-merge, merge queue): they may have changed since the survey.
- **`$WT`.** Recreate it if it was removed (`git worktree add --detach "$WT" <the top branch's SHA>`), check with `git worktree list --porcelain` that `$BR(top)` is not checked out elsewhere, then `git -C "$WT" switch $BR(top)`: `gh stack submit` acts on the stack of the checked-out branch. No rebuild is allowed after this point.
- **No takeover, no force-push of existing PR branches.** Every new branch is a new branch with a new name (`$BR(NN)`) and gets a new PR. Never push to the target branch, a protected branch, or a kept PR's branch. Never delete a remote branch unless the user separately asks.
- **Other people's PRs.** Disabling auto-merge on, unstacking, or closing a PR the current user does not own needs its own explicit confirmation per PR ("close #<n> owned by <login>?"). The default for such a PR is to leave it open and only draft the comment.

Sequence (skip the steps that do not apply):

1. **Disable auto-merge** on every PR that will be retired: `gh pr merge <pr> --disable-auto`. This is first, so an old PR cannot merge before it is retired. Record who enabled it (`autoMergeRequest.enabledBy`) and say so if it was someone else. Never re-enable it. A PR queued for merge is kept, not retired, unless the user explicitly opted in to dequeuing it (GraphQL `dequeuePullRequest`), with the trade-off stated.
2. **Unstack** the old GitHub stack with `gh stack unstack <stack number from pr.md>` — only if **every** member of that stack is in the scope or is a retired PR. Otherwise stop: list the PRs (kept parents, dependents left unchanged) that would lose their stack membership and ask. `unstack` removes the stack object only; the PRs stay open. GitHub leaves PRs that are queued or have auto-merge stacked, which is why step 1 comes first.
3. **Push and create PRs:** `(cd "$WT" && gh stack submit --auto)` with the new stack registered (9.2). It pushes the new branches, creates their PRs as drafts, sets the bases, and creates the stack. Do not pass `--open` without asking: it marks PRs ready for review. `--auto` uses generated titles, so then set the approved ones: `gh pr edit <n> --title "<title>" --body "<body>"` for each new PR. Without gh stack, plain `git push -u origin <branch>` and `gh pr create --draft --base <previous branch> --title … --body …`, only if the user says so.
4. **Verify:** each remote SHA equals the local one; each PR's base is the previous branch (the first one's is the trunk); stack membership is as planned; every kept PR is unchanged (except the base update `submit` makes for kept PRs that were made members).
5. **Close the retired PRs, only after step 4 passes:** `gh pr close <n> --comment <approved text>`. The comment names the new PRs that now hold the code. Closing does not delete the remote branch.
6. **Reply drafts** (`analyze-and-plan.md`) are posted only if the user asks, after approving the exact text with the placeholders resolved to real PR numbers.

**Reviews and comments:** drafts stay `PENDING` unless the user explicitly asks to submit. The user's checkout stays untouched; publishing runs from `$WT` or with explicit refspecs.
