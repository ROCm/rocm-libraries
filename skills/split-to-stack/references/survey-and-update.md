# Phase 0: scope, survey, and update (Survey and Updater read this file)

**Goal:** the confirmed scope, a resolved target, the facts about PRs and tools, and a copy `S'` of the source that has the pre-commit hooks applied and is up to date with the target. The source `S` itself is never modified. The definition of `B` is in `common.md`.

## Survey (read-only except for remote-tracking fetches; writes `survey.md` and `pr.md`)

`survey.md` sections: `Scope`, `Target candidates`, `Merges` (table), `Unlanded refs and aliases of S`, `PRs` (also in `pr.md`), `Tools`, `Questions for the user`, then `Guesses`, `Blocked`. "Hand-written size" is the `B..S` shortstat without the paths that are obviously generated (the Inventory decides the exact sets later).

### 0a. Scope discovery

The orchestrator passes the requested scope (`branch`, `--from`, or `--stack`) and, for a stack, the kind the user chose.

- **Resolving a branch or PR.** For a PR number or URL, take its head branch from `gh pr view`. Fetch only the remote-tracking ref: `git fetch origin <head>`; never `gh pr checkout` and never switch the user's checkout. If a local branch of that name exists and differs from `origin/<head>`, report ahead/behind and ask which is S. A cross-repository (fork) PR is fetched as `git fetch origin pull/<n>/head:refs/remotes/origin/pr/<n>` and never becomes a local branch; say so in `survey.md`. Apply the same rule to every scoped branch, not only to S.
- **Single branch** (`<branch>`). Scope = that branch. Find the branches stacked above it (PRs whose base is the branch, then recursively) and list them as **dependents left unchanged**.
- **`--from <branch>`**: scope = the branch and every branch above it. Use `gh stack` metadata if the branch is in a gh stack; otherwise walk the PR base chain upward. If nothing is above it, it is a single-branch scope.
- **`--stack`**: scope = the whole stack. Kind **gh stack**: `gh stack view` and `.git/gh-stack`; record the stack number (needed for publishing). Kind **plain stacked branches**: walk the PR base chain down from the branch until a PR's base is the target branch (or no PR exists), and up from it until no PR has the current top as its base. If the branch is in no stack, report that and ask the user what they meant.
- Derive PR chains per branch, never from one unbounded list (`gh pr list` returns only 30 by default and this repository has many more open PRs): `gh pr list --state open --head <branch> --limit 200 --json number,headRefName,headRefOid,baseRefName,author,isDraft,url` and `--base <branch>` the same way; if a result count equals the limit, say the list may be incomplete. Cross-check with local ancestry (`git merge-base --is-ancestor`) and, for a gh stack, with `.git/gh-stack`. `gh stack view` saying "not part of a stack" is inconclusive. Report every disagreement. **Without `gh`:** discover by local ancestry only, treat every kept branch as someone else's (trunk mode), and disable publishing.
- **A local branch that is a copy of someone else's PR head.** If S has no counterpart on `origin` but another ref has the same SHA (for example `users/<other>/<name>`), or an open PR's `headRefOid` equals S, report those aliases and PRs. They are PRs this split would re-cut (with their owner): ownership rules in 0c apply, and the orchestrator asks the user.
- Output, bottom first: each branch, its PR, author, review state, CI state, hand-written size, and whether it is the current user's. `S` is the top branch. Unlanded branches below the lowest scoped branch are **external parents** (0c).
- The orchestrator shows the list and **waits for the user to confirm** before anything else.

### 0b. Target branch

1. Propose a target: for a single branch, `develop`; for a slice or a stack, the base branch of the bottom branch in scope. Always confirm with the user; when the PR base differs from the proposal, show it in the prompt. `--target` overrides the proposal but is still confirmed. If the PR's base is not the repository's integration branch, treat that base branch as an external parent (0c).
2. The Survey fetches (`git fetch origin <target>`, remote-tracking refs only), compares local `<target>` with `origin/<target>`, and reports both candidates with ahead/behind and the question; it asks and changes nothing. The **orchestrator** asks and records `T`:
   - equal: use it;
   - local behind by N: ask "Local `<target>` is N commits behind `origin/<target>`. Pull the newest version before splitting?" Answers: *pull* (the orchestrator fast-forwards only: `git fetch origin <target>:<target>` when it is not checked out anywhere, otherwise `git -C <that-worktree> merge --ff-only origin/<target>` only if that worktree is clean, else ask); *use `origin/<target>` as T without touching the local branch*; or *use local* and note it in the proposal;
   - local has its own commits or diverged: never rewrite it; report the counts and ask whether to use local or `origin/<target>`;
   - fetch fails: use local and say freshness could not be checked.
3. No local `<target>`: ask whether to use `origin/<target>` or whether the name is wrong.
4. The orchestrator records the resolved target SHA `T` in `run.md`; the Updater reads it from there, and the "behind by" distance is recomputed against it.

### 0c. Merges and external parents

List every merge in `T..S` (`git rev-list --merges T..S`) and classify each by its second parent `P2`:

| Merge kind | Detection | Meaning |
|---|---|---|
| Target sync | `git merge-base --is-ancestor P2 T` | the user merges the target into the branch; only the conflict resolution is new content |
| Own remote sync | P2 is on `origin/<branch>` history | local/remote sync; noise |
| Other branch, already landed | P2 not in T, but its patches are in T (`git cherry T P2` all `-`, or a tree check; squash merges break ancestry) | cosmetic; disappears after the update |
| Other branch, unlanded | otherwise | external parent or absorb (below) |

Style: target-sync merges present = **merge style**; linear history over a recent target commit = **rebase style**; mixed = follow the most recent integration, or ask.

**Nothing to split:** after the user has decided about external parents, estimate B (`common.md`, read-only). If `B..S` has no non-merge commits, or none remain after external parents are excluded, report it; the orchestrator records `stopped (nothing to split)` in `run.md` and stops before any ref or worktree is created. (Work that landed through squash merges is caught later, by an empty ledger.)

**Unlanded other branches.** Find them both ways: merged in (the table) and stacked by rebase with no merge commit (`git for-each-ref --merged S --no-merged T refs/heads refs/remotes`). Exclude S itself and every ref at S's SHA (they are *aliases of S*, listed separately in `survey.md`, and are not parents); the "own remote sync" row does not apply when S has no counterpart on `origin`. Branches below the confirmed scope are external parents. For each other unlanded branch, check for an open PR (`gh pr list --head <branch> --state open`) and ask: **external parent** (default when a PR is open: not part of the split, the new stack is based on it), **absorb** (default when no PR exists: its work is linearized and re-split with S), or **abort**. An absorbed branch that has an open PR is treated as a re-cut PR (retired at publishing, with the ownership rules below). External parents must form a chain (`git merge-base --is-ancestor`); the topmost, `P`, becomes the base; if they are not a chain, explain that a linear stack has one base and ask which to absorb. Resolve local vs remote for P as in 0b and record its SHA. Report when P is behind T; updating P is outside the split.

**Ownership.** Record the author of every kept PR and of every PR that will be re-cut. If any kept PR is not the current user's (default whenever it is someone else's), the new stack is registered with the topmost kept branch as the stack's **trunk**, never as a member (`publish-and-cleanup.md`). If all are the user's, ask whether they should be members of the new gh stack. Re-cut PRs owned by someone else are flagged for Phase 6: retiring them needs a separate confirmation.

### 0d. Tools and PR state

- `command -v gh`, `gh auth status 2>&1 | rg Logged` (the full output shows part of the token), `gh extension list | rg gh-stack`, `gh stack --version`; `git --version` (require ≥ 2.36 only if the source has merges, for `git show --remerge-diff`; ≥ 2.31 for `--path-format`); `pre-commit --version` and whether `.pre-commit-config.yaml` exists. Record each. If something is missing, say so, give the install or login command, and continue without it: the split still works and the proposal lists the branches and PR-base order manually. Without pre-commit or its config, the hook passes of 0e are skipped and the proposal says so.
- For each PR in scope or kept (`pr.md`): number, URL, author, base and head, draft state, review decision, open review threads (file, line, original line, commit, outdated or not, author, text; `gh pr view --json` has no thread field, so use GraphQL: `gh api graphql` with `pullRequest(number: N) { reviewThreads(first: 100) { nodes { isResolved isOutdated path line originalLine comments(first: 1) { nodes { author { login } body commit { oid } } } } } }`), CI status, and:
  - **auto-merge**: `autoMergeRequest` (with `enabledBy`) from `gh pr view --json`;
  - **merge queue** state: not available from `gh pr view --json`; use GraphQL `pullRequest(number: N) { isInMergeQueue mergeQueueEntry { state } }` via `gh api graphql`.
  A PR that is **queued for merge** and is kept defaults to *kept*: let it land, then base the plan on it; dequeuing is an explicit opt-in the orchestrator asks about, stating the trade-off. A queued PR **inside the scope** (including S itself) is a stop: ask whether to dequeue it or abort. Under `--implement` any queued PR is a stop.

## Updater (writes `$WT` and refs; writes `update.md`)

### 0e. Hooks, update, hooks

1. The orchestrator passed an empty directory `$WT`: run `git worktree add --detach "$WT" <S>` in it (do not create another one) and work on its detached HEAD. After step 4, if HEAD differs from S, name it with `git -C "$WT" switch -c $SPLIT/00-updated`; if HEAD equals S, `S' = S` and no ref is created.
2. **Hook pass 1 (before the update).** Files = those changed by the branch (`git diff --name-only $(git merge-base <T or P> S) S`, existing files only). Run `pre-commit run --files <files>` in `$WT`. Then `git -C "$WT" status --porcelain`: commit every modified tracked file as `style: apply pre-commit hook fixes` (author = the current user), report any modified file that was not in the list, and stop with `## Blocked` if untracked files remain (a hook created them). Hook failures that did not change anything (linters) are *findings*: record them in `update.md` under `## Hook findings` (hook id, path, message); do not fix them.
3. **Update.**
   - External parent `P` chosen: update onto **P**, not T (merging T would pull all of the target's newer changes into `P..S'`). If S already contains P's recorded tip, skip the update; otherwise updating onto P is required.
   - No external parent: `git log --oneline $(git merge-base T S)..T | wc -l` is how far S is behind T. If 0, skip. Otherwise follow the user's answer (the orchestrator asked, using the distance from the Survey). If they declined, skip the update.
   - Update in the user's style: merge style, or any merge from another branch in S: `git merge --no-ff T` (or `P`); rebase style with linear history: `git rebase T` (or `P`). Use `--no-edit` on a merge (keep git's default message); `GIT_EDITOR=true` for everything else. Rebasing history that contains merges risks silently losing earlier conflict resolutions.
   - Conflicts are resolved by the intent of both sides; do not abort. Report every conflicted file and how it was resolved. A genuinely ambiguous semantic conflict: stop with `## Blocked`.
4. **Compute `B`** (`common.md`, Base) now. Then **hook pass 2 (after the update)**: files = those changed in `B..HEAD`; run and commit as in step 2. The hooks depend only on the tree, so running them on `S'` equals running them on `L`; Phase 1 does not run them again.
5. **Scan for Phase 1** (read-only): for each merge in `B..HEAD`, `git show --remerge-diff --format= <merge>` (git ≥ 2.36; empty output = no residue); same-patch duplicates and commits already in B (`git cherry B HEAD` compares patch-ids; do not run a separate pairwise `git patch-id` pass, which is very slow in a partial clone) and rewritten-patch pairs (same author email and subject, different patch-id). This scan is the Linearizer's input (Phase 1 always runs next).
6. Report in `update.md` (if the hooks changed nothing say "no hook commits"): S' SHA; B; the hook commits (their SHAs as they are in S', after the update: a rebase-style update rewrites hook pass 1's commit; `git show --stat`) so Phase 6 can show the **hook delta** (files and lines); hook findings; conflicts; the scan. Nothing is pushed.

The first hook run may install hook environments (network and time); a failure to run the hooks at all is a `## Blocked`. This phase runs in plan-only mode too.
