# Resume and Restart (the orchestrator reads this only when `$RUN` already exists at start)

`run.md` records the last completed phase and the SHAs of S, T, P, B, S', L and the `$WT` path. Show the last completed phase and ask: **resume**, **restart**, or **abort**. A run that stopped at "nothing to split" is recorded as such and restarts silently.

## Resume

- Resume from the first phase without a valid output file, and only if the recorded SHAs still match the refs and the ledger's `base` and `tip` match `run.md` (`base` = B, `tip` = `$L`).
- An agent that stopped with `## Blocked` in the middle of an operation in `$WT` (a merge, a cherry-pick, a hook pass) recorded the exact `$WT` state in its output file: HEAD SHA, the operation in progress, conflicted files, and the hook commits made so far by SHA. The replacement agent reads it first, then either continues with the user's answer or resets `$WT` to the recorded start SHA (`merge --abort` or `cherry-pick --abort`, then `switch --detach <start>`) and re-applies the recorded commits instead of re-running the hooks.
- A partially built stack is resumed with `build_stack.py --refs update`: missing branches are created, and the script is all-or-nothing, so no half-moved stack exists.

## Restart

1. Print every `$SPLIT/*` ref, every branch recorded in the old `refs.tsv`, and the `$WT` path, each with its SHA, and ask per item.
2. `git worktree remove` for `$WT`; `git branch -D` only for the helper refs and for names in the old `refs.tsv` whose SHA still matches (never any other `users/<you>/…` branch).
3. Say that deleting `$RUN` deletes `refs.tsv`, after which leftover `$BR(NN)` can no longer be updated by the new run, and that a deleted branch is recoverable only from the SHA printed.
