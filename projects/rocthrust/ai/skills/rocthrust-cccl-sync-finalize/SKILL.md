---
name: rocthrust-cccl-sync-finalize
description: Read-only audit of a CCCL-into-rocThrust sync branch once every item in todo.md is ticked - completeness and lint gate, file-rename, counterpart, CMake-wiring and copyright-header checks. Reports findings only; never edits files, touches the CHANGELOG or version numbers, or commits. Use when every rocthrust-cccl-sync-resolve item is done and the sync needs a final review.
---

# CCCL → rocThrust Sync (finalize: audit)

Invoked once every checkbox in `todo.md` is ticked and every item has its
own local commit on `$SYNC_BRANCH` (`rocthrust-cccl-sync-resolve` step 8).
This skill audits the sync and reports what it finds. It is **read-only**:

- Don't edit, stage, or commit anything, and don't push.
- Don't write or check the CHANGELOG, and don't bump or check
  `THRUST_VERSION` or any other version number.
- Don't fix findings yourself. Each one goes back to the human, who
  decides whether to reopen a `todo.md` item through
  `rocthrust-cccl-sync-resolve` or handle it some other way.

"The sync diff" below means `git diff "$SYNC_BASE"..HEAD`: everything the
item commits changed, relative to where the sync branch started.

Run every check, even if an earlier one fails, so the human gets the whole
picture in one report. Each finding is either **blocking** (the sync isn't
ready to land as is) or **informational**.

## Check 1 — Completeness (blocking)

- Every line in `todo.md`'s commit list is `- [X]`, not `- [ ]`.
- Every ticked item has its commit: for each item SHA, `git log --oneline
  "$SYNC_BASE"..HEAD --grep "port CCCL <sha11>"` returns exactly one
  commit. A missing one means the item was ticked but never committed; a
  duplicate means it was committed twice.
- `git status --short -- projects/hipccl/hipccl3/rocthrust/` is empty (no staged or
  unstaged work outside an item commit).
- No git merge is in progress (`git status` doesn't mention `MERGE_HEAD`).
  There never should have been one at any point in this pipeline.
- The counterpart-disposition linter reports zero violations:

  ```bash
  rocthrust-cccl-sync-resolve/scripts/rocthrust-todo-lint.sh --repo "$ROCTHRUST_REPO" --todo todo.md
  ```

  This exists because the informal version of this check has already
  failed silently once in a completed sync: PR 12112's `todo.md` used the
  CUDA -> HIP / `testing/` -> `test/` counterpart checks correctly for its
  first several commits, then recorded no disposition at all for the
  remaining ~80 — including the exact commits a later diff against the
  human-authored PR 11296 proved had left `test/test_*.cpp` files behind.
  Report each flagged item as one to reopen in `rocthrust-cccl-sync-resolve`
  (step 2/3, to record the missing disposition).

## Check 2 — File renames (blocking)

Same command the investigate skill already runs to scope the sync, reused
here to double check nothing was missed while porting:

```bash
git diff --find-renames --name-status --diff-filter=R "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/
```

For every rename found, report any `projects/hipccl/hipccl3/rocthrust/CMakeLists.txt` or
test/benchmark `CMakeLists.txt` that still references the old path.

## Check 3 — Counterpart sweep, CUDA -> HIP and testing/ -> test/ (informational)

`rocthrust-cccl-sync-resolve` flags each AMD-only counterpart file's status
per-commit, as it's processed (see that skill's "Why the counterpart checks
don't de-duplicate"). This check is the one-time end-of-sync equivalent for
both counterpart types: a summary over the *whole* sync's final state, not
a new detection mechanism.

**CUDA -> HIP**: list every `system/cuda/**` file present in the sync
diff, and for each, its HIP counterpart's diff status relative to
`$SYNC_BASE`:

```bash
git diff --name-only "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/rocthrust/thrust/system/cuda/
```

For each file listed, translate `system/cuda/` -> `system/hip/` (same
direct-path-then-basename-fallback logic as
`rocthrust-show-upstream-commit.sh`), and check
`git diff --stat "$SYNC_BASE"..HEAD -- <hip-counterpart>`.

**`testing/` -> `test/`**: list every top-level `testing/*.cu` file present
in the sync diff, and for each, its `test/test_<name>.cpp` counterpart's
diff status relative to `$SYNC_BASE`:

```bash
git diff --name-only "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/rocthrust/testing/ | grep -E '^projects/hipccl/hipccl3/rocthrust/testing/[^/]+\.cu$'
```

For each file listed, translate `testing/<name>.cu` -> `test/test_<name>.cpp`
(same direct-path-then-basename-fallback logic as
`rocthrust-show-upstream-commit.sh`), and check
`git diff --stat "$SYNC_BASE"..HEAD -- <test-counterpart>`.

Report both full lists. The point is to catch anything waved through
mid-sync without a look at the aggregate picture (e.g. several small
per-commit "no change needed" verdicts that, taken together across the
whole sync, still leave a counterpart file substantively behind its
sibling).

## Check 4 — Test/example CMake wiring (blocking)

`rocthrust-cccl-sync-todo`'s commit scope (all of `thrust/`, not just
`thrust/thrust/`) means this sync can add new test or example source files,
not just header changes. (New benchmarks need no registration:
`benchmark/CMakeLists.txt` globs `bench/<algo>/*.cu`.) Upstream always
registers a new test/example file in the sibling CMake list in the same
commit; rocThrust needs the same pairing on the local side. Check:

```bash
git diff --name-status "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/rocthrust/testing/ projects/hipccl/hipccl3/rocthrust/examples/ projects/hipccl/hipccl3/rocthrust/test/
```

For every newly-added (`A`) source file in that list, confirm the sync
diff also registers it in the relevant `CMakeLists.txt`
(`projects/hipccl/hipccl3/rocthrust/testing/CMakeLists.txt`,
`projects/hipccl/hipccl3/rocthrust/test/CMakeLists.txt`, or
`projects/hipccl/hipccl3/rocthrust/examples/CMakeLists.txt`). PR #11296 is the concrete
precedent: it added `testing/reduce_into.cu` alongside a 3-line change to
`testing/CMakeLists.txt` and an 8-line change to `test/CMakeLists.txt` in
the same commit. A new test file with no matching CMake registration
silently never runs, so report each one.

## Check 5 — `examples/` copyright header year (blocking)

Confirmed, narrow convention from PR #11296: every touched file under
`examples/` had its AMD copyright header's end year bumped to the year the
sync landed (`// Copyright (c) 2020-2025 ...` → `// Copyright (c) 2020-2026
...`), 19 of 19 touched example files, no exceptions. This does **not**
extend to `thrust/`, `testing/`, or `test/` — of the 302 other touched files
carrying the same header style in that PR, none were bumped. Don't
generalize this into a blanket "every touched file's header" rule; it is
`examples/`-specific until evidence says otherwise.

```bash
git diff --name-only "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/rocthrust/examples/
```

For each file listed, check its `// Copyright (c) <start>-<end> Advanced
Micro Devices, Inc.` header line, and report every file whose `<end>` is
not the current year.

## Report

Give the human one report:

1. A verdict up front: ready to land, or the number of blocking findings.
2. Blocking findings, grouped by check, each with the file or `todo.md`
   item and what's wrong.
3. Informational findings (check 3).
4. `git log --oneline "$SYNC_BASE"..HEAD` (one commit per `todo.md` item,
   plus any `fix(rocthrust): fix port of ...` commits) and `git diff --stat
   "$SYNC_BASE"..HEAD`.

Writing the CHANGELOG, bumping versions, building and testing, committing
any fixes, pushing, opening the PR, and choosing between landing the
per-item history as-is or squashing it are all outside this skill.

`todo.md` can be discarded or attached to the tracking ticket once the sync
lands — it's untracked scratch state and is never committed. The item
commits carry its tick-notes.
