---
name: hipcub-cccl-sync-resolve
description: Works through a hipcub-cccl-sync-todo todo-hipcub.md one upstream commit at a time, porting each commit's CUB changes by hand into hipCUB's rocPRIM and CUB backends. Use when asked to continue, resume, or work the next item of a CCCL-into-hipCUB sync.
---

# CCCL → hipCUB Sync (per-commit port loop)

This is the workhorse stage of the CCCL → hipCUB sync pipeline. It consumes
the `todo-hipcub.md` produced by `hipcub-cccl-sync-todo` and ports upstream commits
into `projects/hipccl/hipccl3/hipcub/` one at a time, in the exact order listed.

hipCUB is not a copy of CUB. It exposes CUB's API under `hipcub::` through
two backends: `backend/rocprim/` reimplements it on rocPRIM (the AMD path),
and `backend/cub/` forwards to NVIDIA's CUB (the nvcc path). Porting a
commit means deciding what its API-visible effect is and expressing that in
each backend, not applying its diff. Most of CUB's implementation layers
(dispatch, agent, detail, tuning) have no hipCUB counterpart at all.

## Assumptions to verify before starting (STOP if any are false)

- `todo-hipcub.md` exists at the repo root.
- The current branch is the `$SYNC_BRANCH` named in `todo-hipcub.md`'s "Sync
  parameters" section.
- **No git merge is in progress.** There is no `MERGE_HEAD` to check for and
  no conflict markers to grep for — hipCUB syncs never use `git merge`.
- Only the **first unticked** (`- [ ]`) item in `todo-hipcub.md`'s commit list may
  be worked next. Do not jump ahead to a later commit even if it looks
  simpler — later commits may assume earlier ones are already applied.
- Every ticked item already has its local commit (step 8), and nothing is
  staged or modified under `projects/hipccl/hipccl3/hipcub/` (`git status --short --
  projects/hipccl/hipccl3/hipcub/`). Leftover changes mean an earlier item was never
  committed or a later one was started early; ask the human which.

If any of these don't hold, STOP and ask the human before proceeding.

## Per-commit loop

1. **Read the next item, then STOP.** Take the first unticked line, e.g.
   `- [ ] <sha> <subject> [SCOPE: ...]`. Tell the human which item is next
   (SHA, subject, `SCOPE`, any `BASELINE` mark or ⚠ flag) and wait for them
   to say go before running step 2 or looking at the commit at all. Do this
   at the start of every item, including straight after committing the
   previous one; never roll on into the next item on your own (committing
   the previous item in step 8 doesn't count as the go).

2. **Show the upstream commit**, scoped to the CUB subtree, with hipCUB
   counterparts already looked up:

   ```bash
   $SKILL_DIR/scripts/hipcub-show-upstream-commit.sh --repo "$HIPCUB_REPO" --sha <sha> --sync-base "$SYNC_BASE"
   ```

   This prints the commit's message, its diff scoped to `cub/`, a table
   mapping each touched upstream path to the hipCUB file(s) it may
   correspond to (`[no local file]` where none was found), and four
   counterpart checks:
   - **Backend parity**: for every touched public `cub/cub/` header, the
     top-level forwarder, `backend/rocprim/` and `backend/cub/` files, each
     with its existence and, given `--sync-base`, whether it changed during
     this sync. A missing `backend/cub/` file for block/warp headers is
     normal: `backend/cub/hipcub.hpp` re-exports those.
   - **rocPRIM follow-up**: for every touched dispatch/agent/detail file,
     the rocPRIM device header that most likely implements the same
     algorithm.
   - **Test counterpart**: for every touched `catch2_test_*.cu`, the
     `test/hipcub/test_hipcub_*` file(s) that cover it.
   - **Benchmark counterpart**: for every touched `bench/<algo>/`, the
     `benchmark/benchmark_device_*` file(s).

   The mappings are name heuristics. A miss means "look for it by hand",
   not "hipCUB doesn't have it". All checks run on every commit, with no
   de-duplication against earlier items; see "Why the counterpart checks
   don't de-duplicate" below.

3. **Classify the commit.** Start from its `SCOPE` and `BASELINE`:
   - `BASELINE` `=` or `~`: the change is probably already in
     `$CURRENT_TAG` through a backport. Check whether hipCUB already
     reflects it; if so it's N/A (`porting-categories.md` category 10).
   - `HEADER`: a backend parity decision (category 1, 2, 9).
   - `INTERNAL` only: N/A in hipCUB, plus a rocPRIM follow-up decision
     (category 3).
   - `TEST`: port the test intent to Google Test (category 4). Catch2/c2h
     plumbing is N/A.
   - `BENCH`: port to hipCUB's benchmark harness (category 5);
     nvbench/tuning-only changes are N/A.
   - `EXAMPLE`: port (category 6).
   - `CMAKE`: hipCUB's CMake is its own. Usually N/A; check whether the
     intent (a new test target, a dropped workaround) needs an equivalent.
   - `OTHER`: e.g. `cub/README.md`, docs. hipCUB's own files aren't
     ports; never delete them. Usually N/A.

   Then pick one disposition:
   - **Port** — the API-visible change is expressed in hipCUB (one or both
     backends, tests, benchmarks, examples).
   - **N/A on AMD** — nothing API-visible changes for hipCUB (internal
     refactor, CUDA-only tuning/PTX/PDL, Catch2 plumbing). Record why.
   - **Needs rocPRIM** — the change can't be expressed without a rocPRIM
     change. STOP and ask the human whether to edit `projects/hipccl/hipccl3/rocprim/` in
     this item, record it as a follow-up, or skip.
   - **Conflicts with local AMD-only changes** — hipCUB has diverged from
     upstream here for its own reasons. STOP and ask the human.

   Whatever the disposition, macro renames/removals need the
   content-triggered sweep in category 7.

   The tick-note (step 7) **must** record a disposition for each check from
   step 2 that applied, even when it's "no change needed":
   - `Backend parity: ...` for a touched public header (name the backend
     files, or say why neither changes);
   - `rocPRIM follow-up: ...` for a touched dispatch/agent/detail file
     (`none (<reason>)` or what rocPRIM would need);
   - `Test counterpart: ...` for a touched `catch2_test_*.cu`;
   - `Benchmark counterpart: ...` for a touched `bench/**/*.cu`.

   `hipcub-todo-lint.sh` checks for these phrases.

4. **Check whether upstream already deleted or renamed something you're
   about to touch.** Before assuming a symbol, file, or code path still
   exists the way this commit references it, check the target tag:

   ```bash
   git show "$TO_TAG:cub/cub/<path>" | grep -n '<identifier>'
   ```

   If it's gone or moved by the time you reach `$TO_TAG`, a later commit in
   the list will handle it — don't pre-emptively "fix" it out of order.

5. **STOP and confirm the classification and porting approach with the
   human before editing anything.**

6. **Apply the change** to the hipCUB file(s) and `git add` them by name.
   Never use `git add -A`/`git add .`, and never stage `todo-hipcub.md` (or the
   rocThrust sync's `todo-rocthrust.md`, if it shares the branch) or the
   investigation report. Changes are hand-written to match the upstream
   commit's intent in hipCUB/rocPRIM idiom, not a mechanical patch apply.
   New tests, benchmarks and examples are registered in their
   `CMakeLists.txt` in the same item.

   Don't compile, build, or go looking for toolchains/dependencies on your
   own: ask the human first, saying what you'd compile and which
   configuration it covers (the rocPRIM backend needs ROCm; the CUB backend
   needs nvcc and a CCCL at least `$TO_TAG` for new forwards). In the same
   question, ask whether to also run the tests, naming which ones: the
   `test_hipcub_*` targets for the files this item touched, not the whole
   suite. Look up their exact names with `ctest -N` in the build directory
   (they're registered as e.g. `hipcub.DeviceReduce`, and may carry a
   GPU-target suffix), then select them with `ctest -R`. Tests need a
   successful build and a GPU. If the human says yes, run only those tests.
   If any fail, show the failures and stop; don't tick the item or work
   around them yourself. Record both outcomes in the tick-note, e.g.
   "Compiled (rocPRIM backend, gfx942); hipcub.DeviceReduce passed" or "Not
   compiled, tests not run (human skipped)".

7. **Tick the checkbox**, adding an indented rationale note:

   ```
   - [X] <sha> <subject>
     - port — added `DeviceReduce::Foo` overload
     - Backend parity: backend/rocprim/device/device_reduce.hpp implements it
       on rocprim::reduce; backend/cub/device/device_reduce.hpp forwards
       (needs CCCL_MINIMUM_VERSION >= 3.1.0)
     - Test counterpart: added Foo cases to test_hipcub_device_reduce.cpp
   ```

   or, for a skip:

   ```
   - [X] <sha> <subject>
     - N/A on AMD — PDL launch plumbing in dispatch_reduce.cuh
     - rocPRIM follow-up: none (CUDA launch detail, no behavior change)
   ```

8. **Commit the item.** Every item gets exactly one local commit on
   `$SYNC_BRANCH`, so the branch history maps 1:1 onto `todo-hipcub.md`:
   - Check `git diff --cached --name-only` lists only this item's files.
   - N/A items with no file changes still get a commit
     (`--allow-empty`), so the reason lives in git history, not only in
     `todo-hipcub.md`, which is scratch state.
   - Message: subject `feat(hipcub): port CCCL <sha11> - <upstream
     subject>`. Rewrite upstream's trailing `(#NNNN)` as
     `(NVIDIA/cccl#NNNN)`, because a bare `#NNNN` links to the wrong
     rocm-libraries PR. Body: the upstream commit URL
     (`https://github.com/NVIDIA/cccl/commit/<sha>`), a blank line, then
     the step 7 tick-note bullets verbatim.

   ```bash
   git commit [--allow-empty] -F - <<'EOF'
   feat(hipcub): port CCCL <sha11> - <subject> (NVIDIA/cccl#NNNN)

   Upstream: https://github.com/NVIDIA/cccl/commit/<sha>

   - <tick-note bullets>
   EOF
   ```

   Then add `- Local commit: <local-sha>` to the item's tick-note in
   `todo-hipcub.md`.
   - Don't push.
   - Don't amend, rebase or squash earlier item commits unless the human
     asks. A fix to an earlier item goes in a new commit (see "When a
     build failure drives this skill").
   - Avoid `git reset --hard` on this branch.
   - Never touch `CCCL_MINIMUM_VERSION`, `HIPCUB_CCCL_VERSION_*` or
     `CHANGELOG.md` in an item commit.

## Periodic self-check: `hipcub-todo-lint.sh`

Every 5-10 commits (and always before handing off to
`hipcub-cccl-sync-finalize`), re-run:

```bash
$SKILL_DIR/scripts/hipcub-todo-lint.sh --repo "$HIPCUB_REPO" --todo todo-hipcub.md
```

This re-derives, from each ticked commit's upstream diff, which step 3
dispositions it needed, and flags any tick-note that records none. It
exists because the same requirement failed silently in the rocThrust
family: an AI-driven sync recorded counterpart dispositions for its first
few items and then stopped, including on the commits later shown to have
left AMD-only files behind. Treat any violation as reopening that item: go
back to step 2/3 for it and record the missing disposition.

This is a presence check only: it doesn't judge whether a recorded
disposition was the right call.

## No 3-way diff tool

A 3-way view would need an open merge to snapshot against, and hipCUB's
files aren't copies of upstream's anyway. Use
`hipcub-show-upstream-commit.sh` (step 2) as the two-way alternative:
upstream commit vs. the current hipCUB counterparts.

## Why the counterpart checks don't de-duplicate

If five upstream commits each touch `cub/cub/device/device_reduce.cuh`, all
five get a backend parity flag for `backend/rocprim/device/device_reduce.hpp`,
even after a human dispositioned it once. That's deliberate. In the rocThrust
family, a plausible "no other backend needs a change" verdict
(`reduce_into`) turned out to miss an AMD-side addition the real human port
made; a check that stopped asking after the first "no change needed" would
have hidden every later chance to revisit it. Re-confirming costs seconds.

## `porting-categories.md`

See the sibling `porting-categories.md` for the (DRAFT) structural
categories to check a commit against before porting. Treat it as a
starting checklist, not an exhaustive rulebook.

## When a build failure drives this skill

If you were invoked because a build broke on a specific file, find which
`todo-hipcub.md` item introduced the change to that file (`git log -- <file>` on
the sync branch points at the item commit), treat it as reopened (un-tick
it, note the failure), and work it again from step 3. Commit the fix as a
new commit, `fix(hipcub): fix port of CCCL <sha11> - <what broke>`, rather
than rewriting the original item commit, and add its SHA to the item's
tick-note. There is no dedicated build-verification skill for hipCUB —
build/test verification happens outside this skill family, whenever the
human runs it.
