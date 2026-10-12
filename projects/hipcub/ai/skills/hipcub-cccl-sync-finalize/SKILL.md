---
name: hipcub-cccl-sync-finalize
description: Read-only audit of a CCCL-into-hipCUB sync branch once every item in todo.md is ticked - completeness and lint gate, upstream header renames vs hipCUB's cub includes, backend parity sweep, CMake and umbrella-header wiring, and a list of rocPRIM follow-ups and CCCL-minimum requirements. Reports findings only; never edits files, touches the CHANGELOG or version numbers, or commits. Use when every hipcub-cccl-sync-resolve item is done and the sync needs a final review.
---

# CCCL → hipCUB Sync (finalize: audit)

Invoked once every checkbox in `todo.md` is ticked and every item has its
own local commit on `$SYNC_BRANCH` (`hipcub-cccl-sync-resolve` step 8).
This skill audits the sync and reports what it finds. It is **read-only**:

- Don't edit, stage, or commit anything, and don't push.
- Don't write or check the CHANGELOG, and don't bump or check
  `HIPCUB_CCCL_VERSION_*`, `CCCL_MINIMUM_VERSION`, `HIPCUB_VERSION` or any
  other version number.
- Don't fix findings yourself. Each one goes back to the human, who
  decides whether to reopen a `todo.md` item through
  `hipcub-cccl-sync-resolve` or handle it some other way.

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
- `git status --short -- projects/hipccl/hipccl3/hipcub/` is empty (no staged or unstaged
  work outside an item commit). If any item edited `projects/hipccl/hipccl3/rocprim/`,
  check that too.
- No git merge is in progress (`git status` doesn't mention `MERGE_HEAD`).
- The disposition linter reports zero violations:

  ```bash
  hipcub-cccl-sync-resolve/scripts/hipcub-todo-lint.sh --repo "$HIPCUB_REPO" --todo todo.md
  ```

  Report each flagged item as one to reopen in `hipcub-cccl-sync-resolve`
  (step 2/3, to record the missing disposition).

## Check 2 — Upstream header renames and deletions (blocking)

`backend/cub/` includes CUB headers by path (`#include <cub/...cuh>`). An
upstream rename or deletion breaks the CUB backend as soon as it's built
against the new CCCL, and nothing on the AMD side would notice.

```bash
git diff --find-renames --name-status --diff-filter=RD "$CURRENT_TAG..$TO_TAG" -- cub/cub/
```

For every renamed (`R`) or deleted (`D`) header, strip the `cub/` prefix
(`cub/cub/x/y.cuh` → `cub/x/y.cuh`) and grep hipCUB at `HEAD`:

```bash
git grep -n '<cub/x/y.cuh>' HEAD -- projects/hipccl/hipccl3/hipcub/
```

Report every hit, with the new path for renames. Also report hipCUB
headers named after a deleted upstream header (`.cuh`→`.hpp`, any of the
three locations) that the sync left in place, so the human can decide
whether hipCUB keeps the API anyway.

## Check 3 — Backend parity sweep (informational)

`hipcub-cccl-sync-resolve` checks backend parity per commit. This is the
end-of-sync summary over the whole diff, to catch per-item "no change
needed" verdicts that together leave one backend behind.

```bash
git diff --name-only "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/backend/
```

Group the files by their path below `backend/rocprim/` or `backend/cub/`
(e.g. `device/device_reduce.hpp`), and for each one report whether the
sync changed the other backend's file of the same name, and whether that
file exists. Expected one-sided cases (a `block/` or `warp/` header with no
`backend/cub/` file because `backend/cub/hipcub.hpp` re-exports it, or an
`detail/` helper only one backend has) are fine; list them separately so
the human can skim past them.

## Check 4 — Wiring of new files (blocking)

New files that aren't wired in never build or run. For every file the sync
added:

```bash
git diff --name-status --diff-filter=A "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/hipcub/
```

- **Tests**: a new `test/hipcub/test_hipcub_*.cpp` (or `.cpp.in`) must be
  registered with `add_hipcub_test(...)` or `add_hipcub_test_parallel(...)`
  in `test/hipcub/CMakeLists.txt`.
- **Benchmarks**: a new `benchmark/benchmark_*.cpp` must be registered with
  `add_hipcub_benchmark(...)` in `benchmark/CMakeLists.txt` (there is no
  glob).
- **Examples**: a new `examples/<sub>/*` source must be registered with
  `add_hipcub_example(...)` in `examples/<sub>/CMakeLists.txt`.
- **Public headers**: a new `hipcub/include/hipcub/backend/rocprim/<dir>/<x>.hpp`
  or `backend/cub/<dir>/<x>.hpp` needs (a) the top-level forwarder
  `hipcub/include/hipcub/<dir>/<x>.hpp` that picks the backend by
  `__HIP_PLATFORM_AMD__`, (b) the matching file in the other backend unless
  the other backend gets the API another way (report which), and (c) an
  include in `backend/rocprim/hipcub.hpp` / `backend/cub/hipcub.hpp` if
  sibling headers in the same directory are included there.

Report each missing piece.

## Check 5 — rocPRIM follow-ups and CCCL minimum (informational)

Collect from the tick-notes in `todo.md`:

- every `rocPRIM follow-up:` line that isn't `none (...)`, with its item
  SHA: these are open work items for rocPRIM;
- every note that a `backend/cub/` forward needs
  `CCCL_MINIMUM_VERSION >= <tag>`: report the highest tag named. Say
  plainly that the human bumps it; don't compare it with the current value
  in `cmake/Dependencies.cmake`.
- every item ticked "Needs rocPRIM" and skipped, and any item that edited
  `projects/hipccl/hipccl3/rocprim/` (from `git diff --name-only "$SYNC_BASE"..HEAD --
  projects/hipccl/hipccl3/rocprim/`).

## Report

Give the human one report:

1. A verdict up front: ready to land, or the number of blocking findings.
2. Blocking findings, grouped by check, each with the file or `todo.md`
   item and what's wrong.
3. Informational findings (checks 3 and 5).
4. `git log --oneline "$SYNC_BASE"..HEAD` (one commit per `todo.md` item,
   plus any `fix(hipcub): fix port of ...` commits) and `git diff --stat
   "$SYNC_BASE"..HEAD`.

Writing the CHANGELOG, bumping versions, building and testing, committing
any fixes, pushing, opening the PR, and choosing between landing the
per-item history as-is or squashing it are all outside this skill.

`todo.md` can be discarded or attached to the tracking ticket once the sync
lands — it's untracked scratch state and is never committed. The item
commits carry its tick-notes.
