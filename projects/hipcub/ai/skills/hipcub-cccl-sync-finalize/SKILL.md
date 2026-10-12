---
name: hipcub-cccl-sync-finalize
description: Read-only audit of a CCCL-into-hipCUB sync branch once every item in todo-hipcub.md is ticked - completeness and lint gate, upstream header renames/additions vs hipCUB's cub includes and re-exports, backend parity sweep, CMake and umbrella-header wiring, rocPRIM follow-ups and CCCL-minimum requirements, and residual parity debt. Reports findings only; never edits files, touches the CHANGELOG or version numbers, or commits. Use when every hipcub-cccl-sync-resolve item is done and the sync needs a final review.
---

# CCCL → hipCUB Sync (finalize: audit)

Invoked once every checkbox in `todo-hipcub.md` is ticked and every item has its
own local commit on `$SYNC_BRANCH` (`hipcub-cccl-sync-resolve` step 8).
This skill audits the sync and reports what it finds. It is **read-only**:

- Don't edit, stage, or commit anything, and don't push.
- Don't write or check the CHANGELOG, and don't bump or check
  `HIPCUB_CCCL_VERSION_*`, `CCCL_MINIMUM_VERSION`, `HIPCUB_VERSION` or any
  other version number.
- Don't fix findings yourself. Each one goes back to the human, who
  decides whether to reopen a `todo-hipcub.md` item through
  `hipcub-cccl-sync-resolve` or handle it some other way.

"The sync diff" below means `git -C "$HIPCUB_REPO" diff "$SYNC_BASE"..HEAD --
projects/hipccl/hipccl3/hipcub/ projects/hipccl/hipccl3/rocprim/`:
everything the item commits changed in hipCUB (and rocPRIM, if any item
edited it), relative to where the sync branch started. Always keep the
path filter. The rocThrust sync (`rocthrust-cccl-sync-*`) may share the
branch, and its changes are not part of this audit.

Run every check, even if an earlier one fails, so the human gets the whole
picture in one report. Each finding is either **blocking** (the sync isn't
ready to land as is) or **informational**.

## Paths and working directory

Nothing in this skill assumes a current directory, and shell state (`cd`,
exported variables) may not carry over from one command to the next.

- `$SKILL_DIR` is the absolute path of the directory holding this
  `SKILL.md`; sibling skills are at `"$SKILL_DIR"/../<skill>/`.
- `$HIPCUB_REPO` is the absolute path of the `rocm-libraries` working tree.
  Git commands run as `git -C "$HIPCUB_REPO" ...`, other repo paths are
  written `"$HIPCUB_REPO"/projects/...`, and bundled scripts take
  `--repo "$HIPCUB_REPO"`.
- Substitute these (and `$SYNC_BASE`, `$CURRENT_TAG`, `$TO_TAG`, ...) with
  their values, or set them in the same command. An empty `$HIPCUB_REPO`
  makes `git -C ""` silently use the current directory; write
  `"${HIPCUB_REPO:?}"` if unsure it is set.

## Check 1 — Completeness (blocking)

- Every line in `todo-hipcub.md`'s commit list is `- [X]`, not `- [ ]`.
- Every ticked item has its commit: for each item SHA, `git -C "$HIPCUB_REPO" log --oneline
  --fixed-strings --grep "feat(hipcub): port CCCL <sha11>" "$SYNC_BASE"..HEAD`
  returns exactly one commit. A missing one means the item was ticked but
  never committed; a duplicate means it was committed twice. Keep the
  `feat(hipcub)` prefix in the grep: when the rocThrust sync shares the branch, its
  commits can cite the same upstream SHA.
- Every `[DEBT]` item is ticked and has its
  `feat(hipcub): align with CCCL <CURRENT_TAG> - ...` commit (or is marked
  `[DEFERRED: ...]` by the human). A `todo-hipcub.md` with no "Baseline
  parity debt" section at all is a finding too: run check 6 to see whether
  any was missed.
- `git -C "$HIPCUB_REPO" status --short -- projects/hipccl/hipccl3/hipcub/` is empty (no staged or unstaged
  work outside an item commit). If any item edited `projects/hipccl/hipccl3/rocprim/`,
  check that too.
- No git merge is in progress (`git -C "$HIPCUB_REPO" status` doesn't mention `MERGE_HEAD`).
- The disposition linter reports zero violations:

  ```bash
  "$SKILL_DIR"/../hipcub-cccl-sync-resolve/scripts/hipcub-todo-lint.sh --repo "$HIPCUB_REPO" --todo "$HIPCUB_REPO"/projects/hipccl/hipccl3/hipcub/todo-hipcub.md
  ```

  Report each flagged item as one to reopen in `hipcub-cccl-sync-resolve`
  (step 2/3, to record the missing disposition).

## Check 2 — Upstream header renames, deletions and additions (blocking)

`backend/cub/` includes CUB headers by path (`#include <cub/...cuh>`). An
upstream rename or deletion breaks the CUB backend as soon as it's built
against the new CCCL, and nothing on the AMD side would notice. An
upstream *addition* that `backend/cub/hipcub.hpp` doesn't re-export leaves
the new API unreachable through `hipcub::` on NVIDIA (3.1's
`cub/thread/thread_simd.cuh`).

```bash
git -C "$HIPCUB_REPO" diff --find-renames --name-status --diff-filter=RDA "$CURRENT_TAG..$TO_TAG" -- cub/cub/
```

For every added (`A`) header outside `detail/`, `agent/`,
`device/dispatch/` and `*/specializations/`, check that
`backend/cub/hipcub.hpp` includes `<cub/...>` or that a
`backend/cub/<dir>/<x>.hpp` wrapper exists; report each one that has
neither.

For every renamed (`R`) or deleted (`D`) header, strip the `cub/` prefix
(`cub/cub/x/y.cuh` → `cub/x/y.cuh`) and grep hipCUB at `HEAD`:

```bash
git -C "$HIPCUB_REPO" grep -n '<cub/x/y.cuh>' HEAD -- projects/hipccl/hipccl3/hipcub/
```

Report every hit, with the new path for renames. Also report hipCUB
headers named after a deleted upstream header (`.cuh`→`.hpp`, any of the
three locations) that the sync left in place, so the human can decide
whether hipCUB keeps the API anyway.

## Check 3 — Backend parity sweep (informational, plus two blocking compile checks)

`hipcub-cccl-sync-resolve` checks backend parity per commit. This is the
end-of-sync summary over the whole diff, to catch per-item "no change
needed" verdicts that together leave one backend behind.

```bash
git -C "$HIPCUB_REPO" diff --name-only "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/backend/
```

Group the files by their path below `backend/rocprim/` or `backend/cub/`
(e.g. `device/device_reduce.hpp`), and for each one report whether the
sync changed the other backend's file of the same name, and whether that
file exists. Expected one-sided cases (a `block/` or `warp/` header with no
`backend/cub/` file because `backend/cub/hipcub.hpp` re-exports it, or an
`detail/` helper only one backend has) are fine; list them separately so
the human can skim past them.

Also report, from the sync diff:
- tests that branch on the platform to reach an API
  (`git -C "$HIPCUB_REPO" diff "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/hipcub/test/ | grep -E '^\+.*#\s*if.*__HIP_PLATFORM_(NVIDIA|AMD)__'`):
  usually a missing `hipcub::detail::` forwarder in `backend/cub/`
  (porting-categories.md category 1, namespace moves);
- public function overloads the sync changed whose deprecated
  `debug_synchronous` sibling in the same class still has the old
  parameter types.

Two compile checks follow. Nothing in this pipeline compiles, so these
are the only guard against the breaks below. Each hit that fails is
**blocking**.

- **Names that only exist in `cub::`.** On NVIDIA, `hipcub::detail::X`
  doesn't fall through to `cub::detail::X`, because hipCUB declares its
  own `hipcub::detail` (see porting-categories.md category 1). Every
  `hipcub::detail::` name used by tests, benchmarks or examples must be
  declared somewhere other than the rocPRIM backend:

  ```bash
  H=projects/hipccl/hipccl3/hipcub
  for n in $(git -C "$HIPCUB_REPO" grep -ohE 'hipcub::detail::[A-Za-z_][A-Za-z0-9_]*' HEAD -- $H/test $H/benchmark $H/examples \
             | sed 's/.*:://' | sort -u); do
    git -C "$HIPCUB_REPO" grep -qw "$n" HEAD -- $H/hipcub/include ":!$H/hipcub/include/hipcub/backend/rocprim" \
      || echo "not reachable on the CUB backend: hipcub::detail::$n"
  done
  ```

  Look at each hit before reporting it. Hits that occur only in comments
  or inside `#ifdef __HIP_PLATFORM_AMD__` code are fine; say so. In 3.1
  this caught `hipcub::detail::ThreadScanInclusive` (the human port added
  `backend/cub/thread/thread_scan.hpp`).
- **Function types that name a changed signature.** For every public
  function whose parameter types the sync changed (from the backend diff
  above), find code that spells the function's type instead of calling
  it, and check that the parameter list matches the new signature:

  ```bash
  git -C "$HIPCUB_REPO" grep -nE '&\s*hipcub::<Class>::<Function>\b|decltype\(\s*&?\s*hipcub::<Class>::<Function>' HEAD -- \
    projects/hipccl/hipccl3/hipcub/{benchmark,test,examples}
  ```

  Then read the function-pointer alias each hit is assigned to. In 3.1
  this would have caught the `sort_func` alias in
  `benchmark_device_segmented_radix_sort.cpp`, which still had `int, int`
  after `DeviceSegmentedRadixSort` moved to `int64_t`.

## Check 4 — Wiring of new files (blocking)

New files that aren't wired in never build or run. For every file the sync
added:

```bash
git -C "$HIPCUB_REPO" diff --name-status --diff-filter=A "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/hipcub/
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

Collect from the tick-notes in `todo-hipcub.md`:

- every `rocPRIM follow-up:` line that isn't `none (...)`, with its item
  SHA: these are open work items for rocPRIM;
- every note that a `backend/cub/` forward needs
  `CCCL_MINIMUM_VERSION >= <tag>`: report the highest tag named. Say
  plainly that the human bumps it; don't compare it with the current value
  in `cmake/Dependencies.cmake`. Put the libhipcxx minimum next to it:
  every "needs libhipcxx >= <tag>" note, plus a note that
  `_HIPCUB_REQUIRED_LIBCXX_VERSION_*` in `hipcub/include/hipcub/libcxx.hpp`
  usually moves together with `CCCL_MINIMUM_VERSION` (3.1: both went to
  3.1.4). Also list items whose diff added a use of `_HIPCUB_STD_EXEC`,
  `stream_ref`, `memory_resource` or other libcu++ facilities newer than
  3.0, even if the note doesn't say so. As above, report this for the
  human to act on; don't check the current values.
- every item ticked "Needs rocPRIM" and skipped, and any item that edited
  `projects/hipccl/hipccl3/rocprim/` (from `git -C "$HIPCUB_REPO" diff --name-only "$SYNC_BASE"..HEAD --
  projects/hipccl/hipccl3/rocprim/`).
- every range guard the sync added in place of a capability
  (`git -C "$HIPCUB_REPO" diff "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/hipcub/hipcub/include | grep -nE '^\+.*numeric_limits<\s*unsigned int\s*>::max'`),
  each with the rocPRIM change that would remove it. In 3.1 the human
  port made those rocPRIM changes instead of adding guards.
- every item deferred because "libhipcxx lacks X": say whether the
  tick-note shows libhipcxx was actually checked.
- every non-`none` `rocPRIM follow-up:` on an item ticked N/A or
  "Needs rocPRIM" whose upstream change adds a public overload or entry
  point (not a behavior or size change in an existing one). Say whether
  the note gives a "hipCUB-only considered: <reason>". Flag notes without
  one as likely missed hipCUB ports (porting-categories.md category 3).
  The 3.1 env-based `DeviceReduce` overloads are the example: the human
  port did them in hipCUB alone.

## Check 6 — Residual parity debt and idiom leftovers (informational)

Re-run the baseline debt check against the sync branch, with `$TO_TAG` as
the baseline: anything it still prints is debt the next sync inherits.

```bash
"$SKILL_DIR"/../hipcub-cccl-sync-investigate/scripts/hipcub-baseline-debt.sh --repo "$HIPCUB_REPO" \
  --base HEAD --from "$TO_TAG" --to "$TO_TAG"
```

Skip hipCUB-only extensions in its check 1 (`IteratorWrapper`,
`DigitExtractorAdopter`, ...). Then grep for idioms the range swept
upstream that hipCUB still has, scoped to what the sync's tick-notes said
was swept:

```bash
git -C "$HIPCUB_REPO" grep -nE '^\s*\(void\)\s*[A-Za-z_][A-Za-z0-9_]*\s*;' HEAD -- projects/hipccl/hipccl3/hipcub/hipcub/include
git -C "$HIPCUB_REPO" grep -nE '::cuda::std\b' HEAD -- projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/backend/cub
git -C "$HIPCUB_REPO" grep -nE '#include <(type_traits|cstdint|cstddef|limits|utility|iterator)>' HEAD -- \
  projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/backend
```

Report the counts per pattern and the files, not every line. Treat each
count as a pointer to look at, not a defect count. The 3.1 human port
left 0 `(void)` casts but still had 18 spelled-out `::cuda::std` in
`backend/cub/`, where `_HIPCUB_STD` isn't always the right spelling.

Then narrow the tree-wide counts to files the sync touched. These are the
leftovers resolve should have fixed while the item was open
(porting-categories.md category 1, "CUB backend include hygiene"):

```bash
for f in $(git -C "$HIPCUB_REPO" diff --name-only "$SYNC_BASE"..HEAD -- \
             projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/backend/cub); do
  n=$(git -C "$HIPCUB_REPO" grep -cE '(^|[^_A-Za-z:])(::)?std::|::cuda::std\b|#include <(type_traits|cstdint|cstddef|limits|utility|iterator)>' HEAD -- "$f" \
      | cut -d: -f3)
  [ -n "$n" ] && echo "$f: $n"
done
```

For each file listed, name the item commit(s) that touched it (`git -C "$HIPCUB_REPO" log
--oneline "$SYNC_BASE"..HEAD -- <file>`) and whether their tick-note
"Identifier sweep:" line explains the leftovers. Without an explanation,
the human can reopen the item to fix them.

## Report

Give the human one report:

1. A verdict up front: ready to land, or the number of blocking findings.
2. Blocking findings, grouped by check, each with the file or `todo-hipcub.md`
   item and what's wrong.
3. Informational findings (checks 3, 5 and 6).
4. This family's commits and diff stat:

   ```bash
   git -C "$HIPCUB_REPO" log --oneline -E --grep '^(feat|fix)\(hipcub\): ' "$SYNC_BASE"..HEAD
   git -C "$HIPCUB_REPO" diff --stat "$SYNC_BASE"..HEAD -- projects/hipccl/hipccl3/hipcub/ projects/hipccl/hipccl3/rocprim/
   ```

   The log should hold one commit per `todo-hipcub.md` item plus any
   `fix(hipcub): fix port of ...` commits. Filter the log by prefix, not by
   path: a path filter drops the empty commits of N/A items. Report any
   commit in `"$SYNC_BASE"..HEAD` that touches `projects/hipccl/hipccl3/hipcub/`
   but matches neither prefix (`git -C "$HIPCUB_REPO" log --oneline "$SYNC_BASE"..HEAD --
   projects/hipccl/hipccl3/hipcub/`) as informational.

Writing the CHANGELOG, bumping versions, building and testing, committing
any fixes, pushing, opening the PR, and choosing between landing the
per-item history as-is or squashing it are all outside this skill.

`todo-hipcub.md` can be discarded or attached to the tracking ticket once the sync
lands — it's untracked scratch state and is never committed. The item
commits carry its tick-notes.
