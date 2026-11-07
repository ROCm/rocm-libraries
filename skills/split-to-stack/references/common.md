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
| `$RUN` | the run directory. |
| `$BR(NN)` | the final branch of group NN: `users/<you>/<suffix>`. |
| `<you>` | the namespace the person running the skill already uses for their branches (`users/<you>/`), see Naming. |

## Base `B`

`B` is the commit the new stack is built on, and the commit the ledger diff starts from.
- If an external parent `P` was chosen: `B` is P's recorded tip, and L contains only the work in `P..S'`.
- Otherwise: the newest target commit contained in `S'`: `T` if S' was updated onto T, else `git merge-base S' T` (for a merge-style user, the newest target commit they merged in).
- If the user declined the update: `git merge-base --all S T`. Exactly one commit is B. More than one (criss-cross history): stop, show the candidates, and recommend updating (0e), which makes B unambiguous.
The Survey only *estimates* B (for the "Nothing to split" check); the Updater computes the real B before hook pass 2 and records it.

## Naming

- **`<you>`.** It is the namespace the user already uses, which is not always their GitHub login (here the login is `Alex-Vasile` but every branch is under `users/alvasile/`). Collect candidates: `gh api user --jq .login`, `git config github.user`, and the `users/<x>/` namespaces of the local branches whose tip is authored by `git config user.email`. Keep the candidates for which `git ls-remote --heads origin 'users/<x>/*'` is non-empty. If exactly one remains, use it; if several or none, ask the user. Never derive it from the source branch (it may be someone else's). Record it in `run.md`; the user may also pin it with `git config split-to-stack.user <x>`, which wins.
- **Final branches** are `users/<you>/<suffix>`, written `$BR(NN)`. `<suffix>` is a short, descriptive, unique, kebab-case name chosen by the Planner, optionally with a common topic prefix and the stack number (for example `kfa-rr-mx-05-ranker-without-rocroller-headers`). The split branches exist under exactly one name each; there are no second, scratch copies.
- **Helper branches** are ordinary local branches: `$SPLIT/00-updated` (only if S' differs from S) and `$SPLIT/00-linear` (only if Phase 1 was not a no-op), with `SPLIT=<source branch name with its remote prefix removed>-split`. Remove a leading `origin/` or any other remote name from the source before deriving names. The `-split/` prefix avoids the leaf-versus-directory ref collision with the source branch itself.
- Validate every name with `git check-ref-format --branch`. Before creating anything, check that neither a ref `$SPLIT` (leaf) nor any `$SPLIT/*` ref exists. The final names are checked by `check_plan.py --prefix users/<you>/ --check-remote origin` before the user approves the plan: it fails if a `$BR(NN)` exists locally (unless this run published it) or on the remote, or if two suffixes cannot coexist (`feat` and `feat/more`). If something exists, handle it as in **Resume** (SKILL.md) or choose another name.

## Workspace

Do all mutating work that needs a checkout in a dedicated scratch worktree so the user's checkout and its dirty state are never touched: `WT=$(mktemp -d)`; `git worktree add --detach "$WT" <start>`. Create each helper ref inside it with `git -C "$WT" switch -c <ref> <start>`. Never check out S or the target in `$WT`.
- Every git command that reads or changes `$WT` runs as `git -C "$WT" …`. Never `cd` into `$WT` or the repo root, with one exception: commands that have no `-C` (`gh stack`, `pre-commit`, builds, tests) run as a subshell `(cd "$WT" && …)`; never in the user's checkout. Before running hooks, check that `git -C "$WT" rev-parse --show-toplevel` equals `$WT`.
- Every file edit uses an absolute path under `$WT`. Never edit files in the user's checkout.
- Keep build directories **outside** `$WT` (under `$RUN/build`, or a symlink there to a `mktemp -d` directory: a large build does not belong inside `.git/`), so removing the worktree is never blocked by build output.
- The orchestrator creates `$WT` as an empty `mktemp -d` directory and passes it; the Updater runs `git worktree add --detach "$WT" <start>` in it. A replacement agent never creates a second one (see Resume in SKILL.md).
- `build_stack.py` writes branch trees with plumbing and a private index, from stored objects without any filter or line-ending conversion (`hash-object --no-filters`): it needs no worktree, and writes only objects and, with `--refs`, the `$BR(NN)` branches. Never check files out to "help" it.
- Writes to `$WT`, the `$SPLIT/*` refs and the `$BR(NN)` branches are serial: one agent at a time, handed over explicitly by the orchestrator. Read-only agents may run in parallel.
- Before the Builder creates or moves a `$BR(NN)` branch, check that no `$BR(NN)` is checked out in any worktree (`git worktree list --porcelain`); `build_stack.py` refuses otherwise.

## Scope and speed

The product is a restructure plan and a stack of branches, delivered quickly. It is not a code review.
- Analysis agents (Survey, Commit readers, Inventory, Planner) decide *structure*: what belongs together, what depends on what, in what order. They do not judge correctness, hunt bugs, assess style, or list reviewer risks beyond what affects grouping, order, or whether a step will build.
- **Analysis agents never execute repository code.** No tests, no builds, no repo scripts, no importing repo modules, no copying the tree elsewhere to run it. They use read-only git/gh commands, the bundled `scripts/`, and their own helpers that only parse git output. A claim that needs running code ("this test fails", "this builds") is marked *unverified* and left to Phase 8.
- The only agents that run repository-configured code are the Updater (the pre-commit hooks, 0e) and the Builder (builds, tests, hooks in Phase 8, only if the user chose tests). The Survey may run `pre-commit --version` and read-only git and gh commands, and may update remote-tracking refs by fetching; it changes nothing else.
- In a partial clone (`blob:none`) read-only git commands may fetch blobs lazily and be slow (a `git patch-id` scan took minutes); that is expected, not a failure.
- Read hand-written diffs, but treat generated and bulk files by `--stat` or file lists only. Prefer commit messages, `--numstat` and `git blame` over reading every line.
- An analysis agent that finds itself at ~15 minutes writes what it has and returns.

## Subagent protocol

- Read only the inputs listed in your task; write only your own output files; return `DONE <file>` or `BLOCKED <file>`.
- Every markdown output file has the headings `## Inputs` (file paths and SHAs it was based on), `## Result`, and `## Blocked`.
- Ownership: each file in the run directory has one writer (`files.md`). The Builder may edit the plan files during the Phase 8 fix loop, and the orchestrator writes `run.md` and `commits.md` (it copies what other agents report, for example the generated sets from `inventory.md` and `$L` from `linear.md`).
- When you stop with `## Blocked` in the middle of an operation in `$WT`, record the exact state: HEAD SHA, the operation in progress, conflicted files, and the commits you made so far (by SHA).
- Never ask the user anything. A question goes under `## Blocked`.
- Conflicts in any phase: resolve by intent and record the resolution in the phase's output file. Write `## Blocked` only for a genuinely ambiguous semantic conflict.

## Constraints

- The source `S` and the user's local target are never modified, except a user-approved fast-forward of the local target in 0b.
- No publishing (`git push`, `gh stack submit/push/sync`, `gh pr create/edit/close/merge`, comments, reviews) unless the user explicitly asks; then follow `references/publish-and-cleanup.md`.
- No interactive editor sessions: export `GIT_EDITOR=true` and `GIT_SEQUENCE_EDITOR=true`, use `--no-edit` on `merge` and `--continue`.
- No merge commits in the output; the result is one linear stack with one commit per branch.
- `tree(L) == tree(S')` and `tree(top of stack) == tree(L)`. Never use `--whitespace=fix`; it rewrites content.
- Use `rg` rather than `grep` for searching (including in pipelines, for example `gh extension list | rg gh-stack`).
