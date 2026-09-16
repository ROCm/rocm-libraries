---
name: rocthrust-cccl-sync-finalize
description: Completes a CCCL-into-rocThrust sync once every item in todo.md is ticked - bumps THRUST_VERSION, writes the CHANGELOG entry, runs build/test verification, and creates the single landing commit. Use when every rocthrust-cccl-sync-resolve item is done and the sync is ready to land.
---

# CCCL → rocThrust Sync (finalize)

Invoked once every checkbox in `todo.md` is ticked and every ported commit's
changes are staged (`git add`), but nothing has been committed yet. This
skill's job is to turn that staged state into a single, real landing commit.

## Step 1 — Readiness check

- Confirm every line in `todo.md`'s commit list is `- [X]`, not `- [ ]`.
- Confirm `git diff --cached --stat` is non-empty.
- Confirm there is **no** git merge in progress (`git status` should not
  mention `MERGE_HEAD`) — there never should have been one at any point in
  this pipeline.
- Run the counterpart-disposition linter, and confirm it reports zero
  violations:

  ```bash
  rocthrust-cccl-sync-resolve/scripts/rocthrust-todo-lint.sh --repo "$ROCTHRUST_REPO" --todo todo.md
  ```

  This is a **hard gate**, unlike step 3's end-of-sync counterpart sweep
  below. It exists because the informal version of this check has already
  failed silently once in a completed sync: PR 12112's `todo.md` used the
  CUDA -> HIP / `testing/` -> `test/` counterpart checks correctly for its
  first several commits, then recorded no disposition at all for the
  remaining ~80 — including the exact commits a later diff against the
  human-authored PR 11296 proved had left `test/test_*.cpp` files behind.
  If this reports any violations, treat each flagged item as reopened (back
  to `rocthrust-cccl-sync-resolve` step 2/3 to record the missing
  disposition) rather than proceeding to Step 2.

If any of these don't hold, STOP — hand back to `rocthrust-cccl-sync-resolve`
rather than proceeding.

## Step 2 — File-rename audit

Same command the investigate skill already runs to scope the sync, reused
here to double check nothing was missed while porting:

```bash
git diff --find-renames --name-status --diff-filter=R "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/
```

For every rename found, check whether `projects/rocthrust/CMakeLists.txt` or
any test/benchmark `CMakeLists.txt` references the old path, and update it.

## Step 3 — End-of-sync counterpart sweep (CUDA -> HIP, testing/ -> test/)

`rocthrust-cccl-sync-resolve` flags each AMD-only counterpart file's status
per-commit, as it's processed (see that skill's "Why the counterpart checks
don't de-duplicate"). This step is the one-time end-of-sync equivalent for
both counterpart types: a final summary sweep over the *whole* sync's final
state, not a new detection mechanism.

**CUDA -> HIP**: list every `system/cuda/**` file present in the final
staged diff, and for each, its HIP counterpart's diff status relative to
`$SYNC_BASE`:

```bash
git diff --cached --name-only -- projects/rocthrust/thrust/system/cuda/
```

For each file listed, translate `system/cuda/` -> `system/hip/` (same
direct-path-then-basename-fallback logic as
`rocthrust-show-upstream-commit.sh`), and check
`git diff --stat "$SYNC_BASE" -- <hip-counterpart>`.

**`testing/` -> `test/`**: list every top-level `testing/*.cu` file present
in the final staged diff, and for each, its `test/test_<name>.cpp`
counterpart's diff status relative to `$SYNC_BASE`:

```bash
git diff --cached --name-only -- projects/rocthrust/testing/ | grep -E '^projects/rocthrust/testing/[^/]+\.cu$'
```

For each file listed, translate `testing/<name>.cu` -> `test/test_<name>.cpp`
(same direct-path-then-basename-fallback logic as
`rocthrust-show-upstream-commit.sh`), and check
`git diff --stat "$SYNC_BASE" -- <test-counterpart>`.

Report both full lists to the human as a final human-facing summary — this
is not a gate, and it does not block completing the sync. Its purpose is to
catch anything a human waved through mid-sync without a final look at the
aggregate picture (e.g. several small per-commit "no change needed"
verdicts that, taken together across the whole sync, still leave a
counterpart file substantively behind its sibling).

## Step 4 — Test/example CMake-wiring check

`rocthrust-cccl-sync`'s widened commit scope (`thrust/testing/`,
`thrust/examples/`, not just `thrust/thrust/`) means this sync can add new
test or example source files, not just header changes. Upstream always
registers a new test/example file in the sibling CMake list in the same
commit; rocThrust needs the same pairing on the local side. Check:

```bash
git diff --cached --name-status -- projects/rocthrust/testing/ projects/rocthrust/examples/ projects/rocthrust/test/
```

For every newly-added (`A`) source file in that list, confirm the same
staged diff also touches the relevant `CMakeLists.txt`
(`projects/rocthrust/testing/CMakeLists.txt`,
`projects/rocthrust/test/CMakeLists.txt`, or
`projects/rocthrust/examples/CMakeLists.txt`). PR #11296 is the concrete
precedent: it added `testing/reduce_into.cu` alongside a 3-line change to
`testing/CMakeLists.txt` and an 8-line change to `test/CMakeLists.txt` in
the same commit — a new test file with no matching CMake registration would
silently never run. If a new file is missing its CMake registration, add it
rather than treating the omission as acceptable; this is a gate, unlike
step 3's end-of-sync sweep.

## Step 5 — `examples/` copyright header year bump

Confirmed, narrow convention from PR #11296: every touched file under
`examples/` had its AMD copyright header's end year bumped to the year the
sync landed (`// Copyright (c) 2020-2025 ...` → `// Copyright (c) 2020-2026
...`), 19 of 19 touched example files, no exceptions. This does **not**
extend to `thrust/`, `testing/`, or `test/` — of the 302 other touched files
carrying the same header style in that PR, none were bumped. Do not
generalize this into a blanket "bump every touched file's header" rule; it
is `examples/`-specific until evidence says otherwise.

```bash
git diff --cached --name-only -- projects/rocthrust/examples/
```

For each file listed, check its `// Copyright (c) <start>-<end> Advanced
Micro Devices, Inc.` header line. If `<end>` is not the current year, bump
it. This is a gate, like step 4 — a stale header on a touched example file
is not an acceptable omission, the same way a missing CMake registration
isn't.

## Step 6 — No `net_ib`-equivalent step

RCCL's finalize skill has a dedicated step for hand-porting NCCL's
`net_ib.cc` into RCCL's `net_ib_rocm.cc`. **rocThrust has no equivalent** —
there is no single hot, hand-maintained file that every sync must specially
reconcile. This is a deliberate, permanent gap in this skill, not an
oversight — do not invent a replacement step.

## Step 7 — CHANGELOG entry

Read the real, current top few entries of `projects/rocthrust/CHANGELOG.md`
as the style guide — headings follow `## rocThrust <X.Y.Z> for ROCm <A.B.C>`,
with `### Added`, `### Changed`, `### Removed`, `### Resolved issues`, and
`### Upcoming changes` sections as needed. For example, the entry that
recorded the (later-reverted) CCCL 3.0 work reads:

```markdown
## rocThrust 5.0.0 for ROCm 10.0.0

### Added

* Largely in feature parity with CCCL/thrust v3.0.3.
  - `thrust::tuple`, `thrust::pair` and `thrust::zip_iterator` fall back to
    rocThrust 4.4.0 implementations when a libhipcxx counterpart
    corresponding to CCCL/libcudacxx >= v3.0.3 is unavailable, ...
```

Sources for the new entry:
- CCCL's own release notes per ported tag:
  `gh release view <tag> --repo NVIDIA/cccl`
- `todo.md`'s tick-notes — anything marked "N/A on AMD" or "skip" is a
  candidate for a `### Removed`/known-gap callout, not silence.
- The staged diff itself (`git diff --cached --stat`).

Write in rocThrust's own voice — this is not a republication of upstream's
release notes. State what changed for a rocThrust user, not what changed in
CCCL generally.

## Step 8 — Version numbers

**Manually** bump `THRUST_VERSION` in
`projects/rocthrust/thrust/version.h` to encode `$TO_TAG`, per the file's own
documented formula (`major*100000 + minor*100 + patch`, e.g. `v3.1.0` →
`300100`). Unlike RCCL's `version.mk`, which the merge commit updates
automatically as part of the subtree merge, nothing here does this for
you — **it is easy to forget, and every future
`rocthrust-cccl-sync-investigate` run depends on it being correct** (it's
Signal D, the primary version signal).

Also check `_THRUST_REQUIRED_LIBCXX_VERSION_MAJOR`/`_MINOR` in
`thrust/detail/config/libcxx.h` for consistency with the new tag if this
sync changed the required libcu++/libhipcxx version.

Do **not** touch `projects/rocthrust/CMakeLists.txt`'s `VERSION_STRING`
(e.g. `"5.0.0"`) — that's rocThrust's own ROCm-release-cadence product
version, entirely decoupled from `THRUST_VERSION`/CCCL tags. Bumping it is
a separate, unrelated release decision, not part of this sync.

Verify by re-running the investigate skill's version-delta script and
confirming Signal D now reports `ok`, no drift:

```bash
rocthrust-cccl-sync-investigate/scripts/cccl-version-delta.sh --repo "$ROCTHRUST_REPO"
```

## Step 9 — Build and test verification

There is no `rccl-build`-equivalent skill for rocThrust yet — this is a
documented gap, not an oversight. Invoke the build directly:

```bash
python rmake.py -ci -a <arch>
```

Then, from the build directory:

```bash
ctest --output-on-failure
```

(`rtest.py` is the CI-style alternative if that's the human's preferred
entry point instead.)

Iterate failures one at a time. For each failure, check whether it maps back
to a specific `todo.md` item:
- If it does, **reopen that item** (change it back to `- [ ]`, note the
  failure), and hand it back to `rocthrust-cccl-sync-resolve`.
- If it doesn't map to anything in `todo.md` — i.e. it's a pre-existing or
  out-of-scope failure — surface it to the human rather than silently
  patching around it.

## Step 10 — Complete the sync

There is no `git merge --continue` here — there was never a merge. Instead:

```bash
git add -A
git commit
```

Follow the real historical commit-message convention (from `1eb022d06b`,
"feat(rocthrust): CCCL 3.0.x changes (#3773)"):
`feat(rocthrust): CCCL <version> changes (#<pr>)`, with a body noting that
this pulls in the specific upstream commits listed in `todo.md`, plus fixes
and workarounds attached to those commits.

This produces exactly **one** commit with a single parent — no second
parent, no subtree-merge lineage to preserve, unlike RCCL.

Do not push or open a PR unless the human asks.

Report:
- The new commit SHA.
- `git show --stat HEAD`.
- That `todo.md` can now be discarded or attached to the tracking ticket —
  it's uncommitted scratch state, not part of the landed history.
