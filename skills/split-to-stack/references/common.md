# Common rules (every agent reads this file)

## Symbols

| Symbol | Meaning |
|---|---|
| `S` | the source: the top branch of the confirmed scope. Never modified. |
| `T` | the resolved target SHA (0b). |
| `P` | the topmost kept (external) parent branch, if the scope sits on other unlanded branches (0c). |
| `S'` | the copy of S after the pre-commit hooks ran and the copy was brought up to date (0e). Equals S when nothing had to change. |
| `B` | base of the split, see Base below. Owned by the Updater, recorded in `update.md` and `run.md`. |
| `L` | a linear, merge-free chain on B with `tree(L) == tree(S')`, made by Phase 1, which always runs. When its no-op check applies (S' already is a clean linear chain on B), `L = S'`. The ledger, the commit readers and the intermediate states all work from L. |
| `$L` | the **SHA** of L, recorded in `run.md` and `linear.md` (refs are for humans; a moving branch such as S itself is never used as `$L`). |
| `$SPLIT` | helper-branch prefix, see Naming. |
| `$WT` | the scratch worktree. |
| `$CO` | the invoking checkout: the worktree the user ran the skill from, recorded in `run.md` at start. |
| `$BASE_REMOTE`, `$PUSH_REMOTE` | the remote of the repository the PRs target, and the remote the stack's branches live in (resolved by the orchestrator, `orchestrator.md`). |
| `$RUN` | the run directory. |
| `$BR(NN)` | the final branch of group NN: `users/<you>/<suffix>`. |
| `<you>` | the namespace the person running the skill already uses for their branches (`users/<you>/`), resolved by the orchestrator (`orchestrator.md`) and recorded in `run.md`. |

## Base `B`

`B` is the commit the new stack is built on, and the commit the ledger diff starts from.
- If an external parent `P` was chosen: `B` is P's recorded tip, and L contains only the work in `P..S'`.
- Otherwise: the newest target commit contained in `S'`: `T` if S' was updated onto T, else `git merge-base S' T` (for a merge-style user, the newest target commit they merged in).
- If the user declined the update: `git merge-base --all S T`. Exactly one commit is B. More than one (criss-cross history): stop, show the candidates, and recommend updating (0e), which makes B unambiguous.
The Survey only *estimates* B (for the "Nothing to split" check); the Updater computes the real B before hook pass 2 and records it.

## Naming

- **Final branches** are `users/<you>/<suffix>`, written `$BR(NN)`. `<suffix>` is a short, descriptive, unique, kebab-case name chosen by the Planner, optionally with a common topic prefix and the position in the stack (for example `cache-03-evict-on-resize`). The split branches exist under exactly one name each; there are no second, scratch copies.
- **Helper branches** are ordinary local branches: `$SPLIT/00-updated` (only if S' differs from S) and `$SPLIT/00-linear` (only if Phase 1 was not a no-op), with `SPLIT=<source branch name with its remote prefix removed>-split`. Remove a leading remote name (`<remote>/`) from the source before deriving names. The `-split/` prefix avoids the leaf-versus-directory ref collision with the source branch itself.
- Validate every name with `git check-ref-format --branch`. Before creating anything, check that neither a ref `$SPLIT` (leaf) nor any `$SPLIT/*` ref exists. The final names are checked by `check_plan.py --prefix users/<you>/ --check-remote "$PUSH_REMOTE"` before the user approves the plan: it fails if a `$BR(NN)` exists locally (unless this run set it) or on the remote, or if two suffixes cannot coexist (`feat` and `feat/more`). If something exists, handle it as in `resume.md` or choose another name.

## Workspace

`$WT` is the scratch worktree of S'. The agents that write `$WT`, refs or builds (orchestrator, Updater, Linearizer, Builder) follow `workspace.md`; every other agent only reads `$WT`, as Searching says, and never writes it.

## Scope and speed

The product is a restructure plan and a stack of branches, delivered quickly. It is not a code review.
- Analysis agents (Survey, Commit readers, Inventory, Planner) decide *structure*: what belongs together, what depends on what, in what order. They do not judge correctness, hunt bugs, assess style, or list reviewer risks beyond what affects grouping, order, or whether a step will build.
- **Analysis agents never execute repository code.** No tests, no builds, no repo scripts, no importing repo modules, no copying the tree elsewhere to run it. They use read-only git/gh commands, reads of the files in `$WT` (Searching), the bundled `scripts/`, and their own helpers that only parse git output. A claim that needs running code ("this test fails", "this builds") is marked *unverified* and left to Phase 8.
- The only agents that run repository-configured code are the Updater (the pre-commit hooks, 0e) and the Builder (builds, tests, hooks in Phase 8, only if the user chose tests). The Survey may run `pre-commit --version` and read-only git and gh commands, and may update remote-tracking refs by fetching; it changes nothing else.
- In a partial clone (`blob:none`) read-only git commands may fetch blobs lazily and be slow (a `git patch-id` scan took minutes); that is expected, not a failure.
- **Searching.** Never run `git grep <rev>` or `git log -S` without a range on a whole directory: inflating historical blobs costs minutes, and missing blobs are fetched one at a time. While nothing writes to `$WT` (Phases 2–5) and its tree equals L (`git -C "$WT" rev-parse HEAD^{tree}` equals `L^{tree}`, and `git -C "$WT" status --porcelain --untracked-files=no` is empty):
  - Tip: `git -C "$WT" grep -n -E 'pat' -- <paths>`, or `git -C "$WT" ls-files -z -- <paths> | xargs -0 rg -n 'pat'`. Do not use plain `rg` on `$WT`: it searches untracked build output and skips hidden tracked files.
  - Base B: tip results are valid at B except for paths in `git diff --name-only --no-renames B L`; read those with `git diff --no-renames B L -- <paths>` or `git show B:<path>`.
  - Other revisions: `git grep <rev> -- <paths that commit changed>`; for history, `git log -S'x' B..L -- <paths>`.
- Read hand-written diffs, but treat generated and bulk files by `--stat` or file lists only. Prefer commit messages, `--numstat` and `git blame` over reading every line.
- An analysis agent that finds itself at ~15 minutes writes what it has and returns.

## Subagent protocol

- Read only the inputs listed in your task; write only your own output files; return `DONE <file>` or `BLOCKED <file>`.
- Batch independent reads: read several files, or several `git show` paths, in one command, or issue several tool calls in one turn, instead of one per turn.
- Every markdown output file has the headings `## Inputs` (file paths and SHAs it was based on), `## Result`, and `## Blocked`.
- Ownership: each file in the run directory has one writer (`files.md`). The Builder may edit the plan files during the Phase 8 fix loop, and the orchestrator writes `run.md` and `commits.md` (it copies what other agents report, for example the generated sets from `inventory.md` and `$L` from `linear.md`).
- When you stop with `## Blocked` in the middle of an operation in `$WT`, record the exact state: HEAD SHA, the operation in progress, conflicted files, and the commits you made so far (by SHA).
- Never ask the user anything. A question goes under `## Blocked`.
- Conflicts in any phase: resolve by intent and record the resolution in the phase's output file. Write `## Blocked` only for a genuinely ambiguous semantic conflict.

## Constraints

- The source `S` and the user's local target are never modified, except a user-approved fast-forward of the local target in 0b and the user-approved Setup sync, which runs before S is recorded. Dependents are moved only by the approved restack (`build-and-test.md`).
- No publishing (`git push`, `gh stack submit/push/sync`, `gh pr create/edit/close/merge`, comments, reviews) unless the user explicitly asks; then follow `references/publish-and-cleanup.md`.
- No interactive editor sessions: export `GIT_EDITOR=true` and `GIT_SEQUENCE_EDITOR=true`, use `--no-edit` on `merge` and `--continue`.
- No merge commits in the output; the result is one linear stack with one commit per branch.
- `tree(L) == tree(S')` and `tree(top of stack) == tree(L)`. Never use `--whitespace=fix`; it rewrites content.
- Use `rg` rather than `grep` for searching (including in pipelines, for example `gh extension list | rg gh-stack`).
