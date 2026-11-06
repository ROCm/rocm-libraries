# Phase 1: linearize (Linearizer reads this file; also `concepts.md`)

**Goal:** a clean, linear chain of commits `L` (ref `$L`) on top of `B`, with **zero diff** against `S'`.

## When this phase runs

Always: after the Updater (0e) and before the Inventory. Every later phase works from L: the ledger is built with `--tip $L`, the commit readers read `B..L`, and intermediate states of entangled code come from L's history. Because every commit of L sits on B, the state of a file at any commit of L is B plus a prefix of the work, which is what an intermediate branch needs (in a merge-heavy S', a commit that sits on an older base would carry stale target text). The Updater's scan in `update.md` (0e step 5) is this phase's input. When S' already is a clean linear chain on B, the no-op check below sets `L = S'` and creates nothing, so the phase costs nothing for linear branches.

**1-0. No-op check (read-only).** If `B` is an ancestor of S', `git rev-list --merges B..S'` is empty, `git cherry B S'` shows no `-` lines, and there are no same-patch or rewritten-patch duplicates, then S' already is a clean linear chain on B. Set `L = S'`, record "no-op: L = S'" in `linear.md`, skip 1c and 1d, and do **not** create `$SPLIT/00-linear`. Otherwise `$L = $SPLIT/00-linear` and 1a–1e run.

The orchestrator copies `$L` (the SHA, never a ref name) from `linear.md` into `run.md`.

## 1a. Base

`B` is the one recorded by the Updater in `update.md` (definition in `common.md`, Base).

## 1b. Commits to replay

1. `git rev-list --reverse --topo-order --no-merges B..S'` walks all parents, so commits that arrived through other-branch merges are included. Topo order keeps stack-parent commits before child commits. The hook commits (`style: apply pre-commit hook fixes`) are ordinary commits in this list.
2. Deduplicate repeated changes and report every case. A commit and its exact revert (an add/revert pair, same files, inverse patch) are **not** duplicates: keep both as ordinary commits, they cancel in the aggregate diff, and `linear.md` names the pair so the Planner knows they belong to no group.
   - **Same patch** (`git patch-id --stable` equal, SHAs differ): keep the first occurrence in topo order; drop the rest.
   - **Rewritten patch** (same author email and subject, different patch-id): a candidate pair. Confirm with `git range-diff <old>^! <new>^!`. Keep the newest version (committer date) at the earliest position and drop the others. If the interdiff shows unrelated changes, the commits are not a pair; keep both. If unclear, write `## Blocked`.
3. Commits already in B: `git cherry B S'` marks a commit `-` when an equivalent patch is already in B. Drop those and record "dropped: already in B". `git cherry` misses changes that landed in modified form (for example squash merges); those are caught during replay in 1c.
4. Merge commits are not replayed. Their hand-written content (conflict resolutions and edits made inside the merge) is the **merge residue**, used as attribution evidence in 1d. For each merge `M`: `git show --remerge-diff --format= M` (git ≥ 2.36). Empty output means no residue; non-empty output is the residue. Octopus merges (`M^3` exists): stop and report; do not guess. Do not use `git diff-tree --cc`; it hides resolutions that match one parent.

## 1c. Replay

1. Create `$L` (`$SPLIT/00-linear`) at B in `$WT`.
2. Use the user's rerere configuration as-is; do not override it. Your own resolutions will be recorded in the shared `rr-cache` as a side effect you cannot avoid; say so in the report. If rerere pre-resolves a conflict (git prints "Resolved '<file>' using previous resolution"; it may already have staged the file), review that file like any other resolution before you continue: no markers, the result matches the commit's intent with S' as the end-state reference, and it contains no change that belongs to a different commit (a resolution recorded from a merge can hold other commits' work). List rerere-resolved files in the report.
3. For each commit `c` in order: `git -C "$WT" cherry-pick -x c` (authorship preserved).
   - If it stops and `git -C "$WT" diff --cached --quiet` succeeds with no unmerged paths, the patch is already present: `git -C "$WT" cherry-pick --skip` and record "dropped: already applied".
   - If it stops with conflicts, resolve them toward that commit's purpose (use S' as the end-state reference but do not pull later commits' changes forward), then `git -C "$WT" cherry-pick --continue`. If the resolution leaves nothing staged, treat it as already applied.
   - If the target already holds the commit's purpose in a different form (a later upstream change to the same code), keep B's form and add only what the commit contributes on top of it; if the commit's change is a later revert of something that is no longer there in that form, resolve so the end state matches S'. Record each such decision under "landed in modified form".
   - Do not rely on `--empty=drop`, which needs git ≥ 2.45.

## 1d. Close the gap (automatic folding)

1. Compare trees: `git rev-parse $L^{tree}` vs `S'^{tree}`.
2. For each remaining hunk of `git diff $L S'`, attribute it to the commit that last touched those lines (blame on L, cross-checked against the merge residues from 1b.4). Lines that exist only in S' have no blame in L: use the merge residue they came from, else step 4. Commit it at the top of L as `git -C "$WT" commit --fixup=<that commit>`, then fold all fixups with `GIT_SEQUENCE_EDITOR=true git -C "$WT" rebase -i --autosquash B`.
3. **If the autosquash rebase conflicts:** `git -C "$WT" rebase --abort`; then fold the conflicting fixup into the nearest later commit that touches the same lines (change `--fixup=<c>` to that commit), record the decision, and retry the autosquash. Never resolve such a conflict by dropping text.
4. Hunks that cannot be attributed (for example new lines written only inside a merge) are folded into the earliest commit whose purpose they serve; list each such decision.
5. Repeat until `tree(L) == tree(S')`. This is a hard requirement; if it cannot be reached in 3 passes, stop and show the remaining diff under `## Blocked`.
6. **No per-commit builds** in L. Only tree equality is required here; buildability is enforced on the split branches in Phase 8 (if tested).

## 1e. Report (`linear.md`)

Commits replayed, duplicates dropped (with patch-id matches), commits already in B, conflicts resolved, folded residue hunks and their target commits, rerere-resolved files, and the tree-equality proof (`L^{tree} == S'^{tree}`, both SHAs).
