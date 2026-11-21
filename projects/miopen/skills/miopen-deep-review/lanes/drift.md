# Lane: target drift (`DRIFT`)

Find what breaks when the change lands on the target's current tip. Every other lane reviews `$HEAD` against `$BASE`; this lane
reviews the merge of `$TARGET_TIP` into `$HEAD`. Work in `$SCRATCH/DRIFT/`.

## Inputs

`drift.txt` (listed in your packet): the target tip, whether it merges cleanly with `$HEAD`, the conflicted paths, and the commits
since `$BASE` that touch a changed path or a file beside one. `pr/pr.json` holds GitHub's `mergeStateStatus` and `mergeable`;
read them as a cross-check, not as the answer.

## Steps

1. **Merge.** Own the merged tree; never reuse `$SCRATCH/RUN/`. Delete a tree from an earlier attempt first: its pending merge
   blocks the checkout.

   ```bash
   . ./env.sh; rm -rf "$SCRATCH/DRIFT/merge"; fullclone "$SCRATCH/DRIFT/merge" "$HEAD" && git -C "$SCRATCH/DRIFT/merge" merge --no-commit "$TARGET_TIP"
   ```

2. **Resolve.** Resolve each conflict in the scratch tree so that both sides keep their intent. Record the file, the choice and
   why. A resolution that needs a decision only the author can make (two designs for the same code, a removed symbol the change
   still uses) is not yours to settle.
3. **Read the touching commits.** `git -C "$CLONE" show <sha>` for each: renamed or removed symbols, changed signatures, option
   defaults, CMake targets, test fixtures, schema changes. Check each against the PR's code, tests and CMake.
4. **Test the merge.** Select and run tests by [build-validate.md](build-validate.md) steps 2 and 3, in the merged tree, with the
   same build rules (clang-tidy off, a unique `TMPDIR`, `Cores:`). A selection that fails at `$HEAD` is `RUN`'s finding; one
   that fails only on the merge is yours.

## Findings

| Tier | When |
|---|---|
| `BLOCKING` | The merged tree fails to build or a selected test fails, or a target-side contract change breaks code the PR depends on. Kind `defect`; **Proof** is the merged-tree command and its log |
| `IMPORTANT` | A conflict whose resolution needs an author decision. Kind `process` |

A conflict that resolves mechanically and builds and passes is no finding: list it under **Checked, no finding** with
its resolution. A selection that cannot run here (no build host, a GPU device) stays `UNVERIFIED`, as in `RUN`.
