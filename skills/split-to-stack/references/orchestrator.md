# Orchestrator setup (the orchestrator reads this at start)

## Remotes

Resolve two remotes at start and record them in `run.md`; no command names any other remote. Match remote URLs to repositories by `owner/name`, ignoring protocol and `.git`; no match or several matches: ask.
- `$BASE_REMOTE`: the remote of the repository the PRs target (`gh pr view --json baseRepository`; without a PR, the repository `gh repo view` resolves). The target is fetched from it.
- `$PUSH_REMOTE`: the remote of the repository the stack's branches live in (`headRepository`; without a PR, the branch's upstream remote). The Setup sync, `--check-remote` and every push use it.

When they are the same, every `gh stack` command gets `--remote "$PUSH_REMOTE"` (without it, gh stack prompts when several remotes exist). When they differ, the run is in **fork mode**, detected automatically and stated in the proposal: GitHub stacks need every branch in the base repository, so gh stack is not used and publishing follows the fork-mode rules in `publish-and-cleanup.md`.

## `<you>`

The namespace the user already uses for branches, which is not always their GitHub login. If the branch checked out in `$CO` is `users/<x>/…` and is the user's (its open PR's author is `gh api user --jq .login`; without a PR, its tip commit's author name or email is `git config user.name` or `git config user.email`), use `<x>`. Otherwise ask the user. `git config split-to-stack.user <x>` overrides both. Record it in `run.md`.

## Tests

Ask once: "Run the tests on every branch, or skip them (for example to inspect the branches first)?" With `--implement`, ask it at start; in plan-only mode, when the user asks to implement. Record the answer in `run.md`. A user who skipped can ask later to test an existing stack: create a fresh `$WT` and run `build-and-test.md` 8.2 against the SHAs in `stack.tsv` (fixes then rebuild with `--refs update`; a published branch is moved only with the user's explicit permission, `build-and-test.md` Fixes).

## Small-input path

After the Survey, the Updater and the Linearizer (always run as usual), measure `git diff --numstat B $L` (hand-written = excluding the default generated globs of `concepts.md`), `git rev-list --count B..$L`, and the merges in `B..S'`. If all hold: ≤ 800 hand-written changed lines, ≤ 15 commits, no merges from other branches, one component (the first two path components under `projects/`, `shared/` or `dnn-providers/`), then a single Planner does the Inventory and the plan (reading commits itself, no commit readers, no separate Inventory), and one Builder does Phases 7–8, with the same files, scripts, gates and table. The Planner never runs hooks or asks the user; if it finds an entangled entry or a broken threshold it stops with `## Blocked: needs-full-pipeline`; then keep `ledger.tsv` and `inventory.md`, delete the plan files, and run the commit readers and a fresh Planner.
