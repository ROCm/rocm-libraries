# Phase 9: report, local stack, cleanup, and publishing (orchestrator reads this file)

By default the skill creates local refs and registers a local `gh stack` only. Nothing is pushed, and no PR or comment is created, closed or edited, unless the user explicitly asks.

## 9.1 Report

If changes were stashed at start, first remind the user: switch back to the recorded branch, then `git stash apply <sha>`. List every ref created with its SHA: each `$BR(NN)` (the deliverable), `$SPLIT/00-updated` and `$SPLIT/00-linear` if they exist. Say which branches were fully tested, which were *not run*, any opt-outs, source problems found, the hook delta and unfixed hook findings, every deviation from the approved plan, and (single-branch scope) each restacked dependent with its original and new SHA. `$L` is the zero-diff linear version of S', useful for review and for any re-split.

## 9.2 Local gh stack (if `gh stack` is available)

Local only. Older `gh stack` versions keep their state per worktree (`<git-dir>/gh-stack`), so it runs in the invoking checkout: `(cd "$CO" && gh stack …)`. Ask **one** permission prompt listing what happens: the old local stack loses local tracking (step 1), `$CO` is switched to `$BR(top)`, and the new stack is registered. `$CO` must be clean (`git -C "$CO" status --porcelain` empty); otherwise ask the user to commit or stash first. Record the branch `$CO` had checked out and print the command to switch back. Before registering, check every name with `git rev-parse --verify refs/heads/<name>`: `gh stack init` **creates** a branch that does not exist, so a typo would silently create one.

1. **An existing local stack.** If `gh stack init` would conflict (the trunk or a member is already tracked in `$CO`'s gh stack state), copy that state file (`$(git -C "$CO" rev-parse --git-dir)/gh-stack`, else `<git-common-dir>/gh-stack`) to `$RUN/gh-stack.backup`, list every other branch that loses local tracking, then run `(cd "$CO" && gh stack unstack --local)` while `$CO` is still on its old stack branch. **Never pass a stack number**: with a number, `unstack` acts "directly through the GitHub API". Do not edit the state file by hand. If registering then fails, tell the user to copy the backup back.
2. **Register the new stack:** `git -C "$CO" switch $BR(top)`, then `(cd "$CO" && gh stack init --base <trunk> <members, bottom first>)`.
   - `<trunk>` is the target by default; register only if `git merge-base --is-ancestor B <trunk>` holds, otherwise tell the user to update the local target first.
   - Kept parents outside the stack of S: `<trunk>` is the **topmost** of them, and they are not members (`gh stack submit`, `push` and `sync` push every member branch and update PR bases, so a member would be written to). `--base` takes one branch; the lower ones stay in its history but are not managed by gh stack.
   - Kept parents in the stack of S are members, whoever owns them: list them first, bottom first, then the new branches, then the restacked dependents. The Setup sync made them equal to the remote, so `submit` pushes nothing to them.
   - `build_stack.py` refuses a branch that is checked out in a worktree, so a rebuild after this point needs `$CO` switched off `$BR(top)` first.
3. Print the `gh stack submit --auto --remote "$PUSH_REMOTE"` command for the user. Do not run it (see Publishing).

After the kept PRs land, GitHub retargets PRs whose base branch was deleted on merge, but the local trunk must be re-pointed to the target: tell the user to run `gh stack init --base <target> …` again (gh stack has no visible trunk-change command; this is unverified).

## 9.3 Worktree

Publishing does not use `$WT`. To remove it: `git -C "$WT" status --porcelain`; untracked files left by hooks are removed with `git -C "$WT" clean -fd` (this acts only inside the scratch worktree; add `-x` if ignored files block removal), then `git worktree remove "$WT"`. If the run stopped early or `$WT` holds uncommitted state, keep it, say where it is, and say why.

## 9.4 Cleanup prompt

Ask it after the orchestrator's `verify_stack.py` run (8.1) has passed and the test results (or "not run") are recorded in `tests.md`, or at the plan-only stop (Phase 6). Ask **one** prompt listing every helper item that exists, each with its recommended answer, and wait. Skip items that do not exist. Never ask about, or delete, `$BR(NN)`, the source branch `S`, or anything not listed. At the plan-only stop, also list `$WT` (path; removed as in 9.3, recreated when needed), and recommend keeping a, b and c if the user may implement this plan: deleting any of them means implementing later starts over from the Survey.

- **a. `$SPLIT/00-updated` (<SHA>)**: the copy of `<S>` with the hooks applied and brought up to date. Recommended: keep until the PRs are merged, so it can be inspected.
- **b. `$SPLIT/00-linear` (<SHA>)**: the zero-diff linear chain; `tree($BR(top)) == tree(00-linear)` is verified (tests: <summary>). Needed only to re-split or compare. Recommended: keep until the PRs are merged.
- **c. Run directory `<absolute RUN path>`** (plan, ledger, stack and test notes, and the `gh-stack.backup` copy if one was made; deleting it deletes them). Recommended: keep until the PRs are merged, but delete its `build/` subdirectory now (builds can be many GB).

For each "delete" answer, resolve and print the exact target first (`git branch -D <ref>`, `rm -r "<absolute RUN>"`), delete only that item, then say what was removed. **A deleted branch cannot be restored from the reflog** (`git branch -D` removes its reflog). The skill prints each helper's SHA in the report; as long as the object has not been garbage-collected, `git branch <name> <sha>` brings it back. For "no" or no answer, keep it and print the one-line command for the user to run later. The Resume "restart" option uses this same procedure.

## Publishing: only on explicit request

If the user explicitly asks to publish, follow these rules.

- **Draft first.** Show the exact commands, PR titles and bodies, and comment text, and get approval before any external write. Mutate only what was requested, then verify. PR titles are the groups' `type(scope): subject` and bodies their `summary` (plus the dependency note "stacked on #<N>").
- **Fetch first.** Fetch the remote refs involved and record their SHAs. Re-check the PR states from 0d (auto-merge, merge queue): they may have changed since the survey.
- **Checkout.** `gh stack submit` acts on the stack of the checked-out branch, so publishing runs in `$CO` with `$BR(top)` checked out (9.2; do 9.2 first if it was skipped). No rebuild is allowed after this point.
- **No takeover.** Every new branch is a new branch with a new name (`$BR(NN)`) and gets a new PR. Never push to the target branch or a protected branch. The only writes to existing PR branches are the approved Setup sync (`survey-and-update.md`) the approved push of the restacked dependents, and the approved force-push of rebuilt published branches (step 3); never push to any other existing PR branch. Never delete a remote branch unless the user separately asks.
- **Other people's PRs.** Disabling auto-merge on, unstacking, or closing a PR the current user does not own needs its own explicit confirmation per PR ("close #<n> owned by <login>?"). The default for such a PR is to leave it open and only draft the comment.

Sequence (skip the steps that do not apply):

1. **Disable auto-merge** on every PR that will be retired: `gh pr merge <pr> --disable-auto`. This is first, so an old PR cannot merge before it is retired. Record who enabled it (`autoMergeRequest.enabledBy`) and say so if it was someone else. Never re-enable it. A PR queued for merge is kept, not retired, unless the user explicitly opted in to dequeuing it (GraphQL `dequeuePullRequest`), with the trade-off stated.
2. **Unstack** the old GitHub stack. GitHub cannot remove one PR from a stack, and `submit` does not replace an existing stack (it only warns), so the stack is dissolved here and recreated in step 3 with its kept parents and dependents as members (9.2). Look up its number now: `gh api "repos/<base owner>/<repo>/stacks?pull_request=<n>"` for any PR in `pr.md`; if none is in a stack, skip this step. If it has open members that are neither in the new stack nor retired, stop, list them and ask. Run `gh stack unstack <number>`, then query again and stop if the stack still exists: a partial unstack exits 0. `unstack` removes the stack object only; the PRs stay open. GitHub leaves PRs that are queued or have auto-merge stacked, which is why step 1 comes first.
3. **Push and create PRs.** If there are restacked dependents or rebuilt published branches, first get explicit permission to force-push them, listing each with its remote SHA and new SHA. Then `(cd "$CO" && gh stack submit --auto --remote "$PUSH_REMOTE")` with the new stack registered (9.2). It pushes the new branches and the dependents, creates the new PRs as drafts, sets the bases, and creates the stack. Do not pass `--open` without asking: it marks PRs ready for review. `--auto` uses generated titles, so then set the approved ones: `gh pr edit <n> --title "<title>" --body "<body>"` for each new PR. Without gh stack, plain `git push -u "$PUSH_REMOTE" <branch>` (dependents with `--force-with-lease=<branch>:<remote SHA>`) and `gh pr create --draft --base <previous branch> --title … --body …`, only if the user says so.
4. **Verify:** each remote SHA equals the local one; each PR's base is the previous branch (the first one's is the trunk); the stacks API returns the new stack with the planned PRs in order (show the user its PR list); every kept PR is unchanged except for a base update.

**Fork mode** (`common.md`, Remotes) replaces 9.2, step 2 and step 3: push with `git push "$PUSH_REMOTE" <branches>` (dependents and rebuilt published branches with `--force-with-lease=<branch>:<remote SHA>`, after the same permission), then create each new PR, bottom first, with `gh pr create --draft --repo <base owner/name> --head <fork owner>:<branch> --base <target> --title … --body …`. Every PR targets the target branch and so shows the whole stack below it; the body of each one above the bottom ends with "Stacked on #<previous PR>; review only commit <its SHA>." In step 4 each PR's base is the target. Tell the user that after a lower PR merges, the branches above it need rebasing.
5. **Close the retired PRs, only after step 4 passes:** `gh pr close <n> --comment <approved text>`. The comment names the new PRs that now hold the code. Closing does not delete the remote branch.
6. **Reply drafts** (`analyze-and-plan.md`) are posted only if the user asks, after approving the exact text with the placeholders resolved to real PR numbers.

**Reviews and comments:** drafts stay `PENDING` unless the user explicitly asks to submit. Apart from the approved switch in 9.2, the user's checkout is untouched.
