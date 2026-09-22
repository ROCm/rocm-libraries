---
name: rocthrust-cccl-sync-resolve
description: Works through a rocthrust-cccl-sync todo.md one upstream commit at a time, porting each commit's Thrust changes by hand into rocThrust. Use when asked to continue, resume, or work the next item of a CCCL-into-rocThrust sync.
---

# CCCL → rocThrust Sync (per-commit port loop)

This is the workhorse stage of the CCCL → rocThrust sync pipeline. It
consumes the `todo.md` produced by `rocthrust-cccl-sync` and ports upstream
commits into `projects/rocthrust/` one at a time, in the exact order listed.

## Assumptions to verify before starting (STOP if any are false)

- `todo.md` exists at the repo root.
- The current branch is the `$SYNC_BRANCH` named in `todo.md`'s "Sync
  parameters" section.
- **No git merge is in progress.** There is no `MERGE_HEAD` to check for and
  no conflict markers to grep for — rocThrust syncs never use `git merge`.
  If you find yourself looking for conflict markers, you are following the
  wrong playbook.
- Only the **first unticked** (`- [ ]`) item in `todo.md`'s commit list may
  be worked next. Do not jump ahead to a later commit even if it looks
  simpler — later commits may assume earlier ones are already applied.

If any of these don't hold, STOP and ask the human before proceeding.

## Per-commit loop

1. **Read the next item.** Take the first unticked line, e.g.
   `- [ ] <sha> <subject>`.

2. **Show the upstream commit**, scoped to the Thrust subtree and with
   destination paths already translated:

   ```bash
   $SKILL_DIR/scripts/rocthrust-show-upstream-commit.sh --repo "$ROCTHRUST_REPO" --sha <sha> --sync-base "$SYNC_BASE"
   ```

   This prints the commit's message, its diff scoped to `thrust/thrust/`,
   `thrust/testing/`, and `thrust/examples/`, a table mapping each touched
   upstream path to its local `projects/rocthrust/...` equivalent, and two
   counterpart checks for AMD-only files that have no upstream commit trail
   of their own:
   - **CUDA -> HIP counterpart check**: for every touched path under
     `thrust/system/cuda/`, reports whether a same-named file exists under
     `thrust/system/hip/` and, given `--sync-base`, whether that HIP file
     has already changed during this sync.
   - **`testing/` -> `test/` counterpart check**: for every touched
     top-level `thrust/testing/*.cu` path, reports whether a
     `test_<name>.cpp` file exists under `projects/rocthrust/test/` (a
     separate, hand-maintained GTest suite with no upstream equivalent at
     all — distinct from `testing/`, which is a 1:1 translated port of
     upstream's Catch2 suite) and, given `--sync-base`, whether it has
     already changed during this sync. Does not trigger on
     `thrust/testing/cuda/`, `cpp/`, `omp/`, or `unittest/` subdirectory
     touches (no `test/` counterpart pattern exists for those), nor on
     header-only commits that don't also touch a top-level `testing/*.cu`
     file.

   Both checks are unconditional and run the same way on every commit —
   neither suppresses itself if an earlier `todo.md` item already flagged
   (and dispositioned) the same counterpart file. Expect repeats when
   several commits touch the same CUDA file or the same `testing/*.cu`
   file; that's by design, not a bug — see "Why the counterpart checks
   don't de-duplicate" below.

3. **Classify the commit** into one of. First check the item's `SCOPE` tag
   (set by `rocthrust-commit-list.sh`, carried into `todo.md`): a commit
   scoped to `TEST` or `EXAMPLE` alone (no `HEADER`) usually needs no
   CUDA/HIP source-level porting judgment at all — Thrust's test and example
   code dispatches through the same execution-policy templates regardless of
   backend, so a clean port is the default expectation. The one exception is
   `thrust/testing/cuda/`, which contains literal CUDA-API tests; it already
   has a same-named local counterpart (`projects/rocthrust/testing/cuda/`),
   so it is still a direct port, not an adaptation case — just to the CUDA
   test file's own counterpart rather than a HIP-side analogue. Do not skip
   the classification step below just because a commit is `TEST`/`EXAMPLE`
   scoped; it still needs a disposition and a tick-note, the same as any
   other item.
   - **Clean port** — applies to rocThrust with no CUDA-specific content.
   - **Needs CUDA→HIP adaptation** — touches CUDA intrinsics, PTX, or
     `thrust/system/cuda/` structure that has a HIP analogue under
     `thrust/system/hip/`. Cross-check `sensitive-files.md` (sibling of
     `rocthrust-cccl-sync-investigate`) and `porting-categories.md` (sibling
     of this file) for known-risky patterns.
   - **N/A on AMD** — the commit only affects a CUDA-only code path with no
     HIP equivalent (e.g. CUDA-graph-specific execution policies). Skip it,
     but record why.
   - **Conflicts with local AMD-only changes** — the commit touches a
     region rocThrust has already diverged from upstream on for its own
     reasons. STOP and ask the human how to reconcile.

   If step 2's CUDA -> HIP or `testing/` -> `test/` counterpart check
   reported any counterpart file as `UNCHANGED since $SYNC_BASE`, the
   tick-note (step 7) **must** record that disposition explicitly — e.g.
   "HIP counterpart: no change needed, matches upstream's own scope",
   "HIP counterpart: ported, see `system/hip/detail/<file>:<line>`", or
   "test/ counterpart: updated `test/test_<name>.cpp` to match" — rather
   than leaving it implicit. This is not optional even when the disposition
   is "no change needed": the point is a reviewable trail, not just a
   correct decision. (An upstream commit touching only `thrust/system/cuda/`
   or `thrust/testing/*.cu` does not, by itself, prove no counterpart-side
   change is warranted — see the `reduce_into` case in "Why the counterpart
   checks don't de-duplicate" below.)

4. **Check whether upstream already deleted or renamed something you're
   about to touch.** Before assuming a symbol, file, or code path still
   exists the way this commit references it, check the target tag:

   ```bash
   git show "$TO_TAG:thrust/thrust/<path>" | grep -n '<identifier>'
   ```

   If it's gone or moved by the time you reach `$TO_TAG`, a later commit in
   the list will handle it — don't pre-emptively "fix" it out of order.

5. **STOP and confirm the classification and porting approach with the
   human before editing anything**, unless the human has explicitly asked
   to run through several commits autonomously in this session.

6. **Apply the change** to the translated local path(s) and `git add` them.
   For adaptation cases, the change is hand-written to match the upstream
   commit's intent using HIP/rocThrust idiom, not a mechanical patch apply.

7. **Tick the checkbox**, adding an indented rationale note:

   ```
   - [X] <sha> <subject>
     - clean port — applied verbatim to projects/rocthrust/thrust/<path>
   ```

   or, for a skip:

   ```
   - [X] <sha> <subject>
     - N/A on AMD — CUDA-graph-only execution policy, no HIP equivalent exists
   ```

8. **Do not commit.** Everything ported across every item in `todo.md`
   lands in a single commit, created by `rocthrust-cccl-sync-finalize` once
   every checkbox is ticked. This is a discipline this skill family imposes
   on itself — nothing here stops you from committing early by mistake, so
   be deliberate about it. Because staged-but-uncommitted state
   can span many sessions, avoid `git reset --hard` on this branch, and
   consider a backup branch if the sync is long-running.

## Periodic self-check: `rocthrust-todo-lint.sh`

Every 5-10 commits (and always before handing off to
`rocthrust-cccl-sync-finalize`), re-run:

```bash
$SKILL_DIR/scripts/rocthrust-todo-lint.sh --repo "$ROCTHRUST_REPO" --todo todo.md
```

This re-derives, from each already-ticked commit's own upstream diff,
whether it touched `thrust/system/cuda/` or a top-level `thrust/testing/*.cu`
file, and flags any ticked item whose tick-note recorded no HIP or `test/`
counterpart disposition at all — a mechanical version of step 3's "must
record that disposition explicitly" requirement, not a new check.

This exists because that requirement has already failed silently once, in
practice: the real `todo.md` behind PR 12112 used the counterpart checks
correctly for its first several commits, then stopped recording dispositions
for the remaining ~80 — including the exact commits a later side-by-side
diff against the human-authored PR 11296 proved had left `test/test_*.cpp`
files behind. The counterpart-check mechanism existed and was demonstrably
used at the start of that sync; the gap was a mid-session discipline lapse
nothing caught until the sync was already done. Treat any violation this
script reports as reopening that `todo.md` item — go back to step 2/3 for it
and record the missing disposition — not as something to wave through
because the sync has already moved on.

This is a presence check only: it does not judge whether a recorded
disposition was the *right* call, only that one was written down. A
violation here does not necessarily mean a file was left un-ported — it
means the tick-note doesn't show the reasoning either way.

## No 3-way diff tool

A meld-able 3-way view ("ours"/"theirs"/"working" trees) would require an
open subtree merge to snapshot against. rocThrust has no merge in progress,
so there is nothing to snapshot that way — this is a known, permanent gap,
not an oversight. Use `rocthrust-show-upstream-commit.sh` (step 2 above) as
the two-way alternative: upstream commit vs. current local file.

## Why the counterpart checks don't de-duplicate

`rocthrust-show-upstream-commit.sh`'s CUDA -> HIP and `testing/` -> `test/`
counterpart checks (step 2) deliberately do not remember which counterpart
files earlier `todo.md` items already flagged. If five different upstream
commits each touch `thrust/system/cuda/detail/reduce.h`, all five get an
independent `system/hip/detail/reduce.h` flag — even after a human has
already reviewed and dispositioned it once. Same for `testing/*.cu` files
against their `test/test_*.cpp` counterpart.

This is a deliberate choice, not an oversight: this exact scenario already
happened once, informally, before the CUDA -> HIP check existed. Item
`7eedb6e38ef188d8a60e513fb1c287bbb12550a3` ("Add `reduce_into`") reasoned
that no other backend, including HIP, needed a change, "matching upstream's
own scope (only CUDA gets the optimized path)". That reasoning was
plausible and locally correct — the upstream commit genuinely only touched
CUDA — but the real historical AMD port (PR #11296) made a different,
AMD-specific call: it added a `reduce_into` forward-declaration to
`system/hip/detail/reduce.h` anyway, as an addition beyond upstream's own
scope. A one-shot check that stops re-asking after the first "no change
needed" verdict would have hidden exactly this kind of case from every
subsequent commit that touched the same CUDA file — de-duplicating would
optimize for less noise at the cost of silently trusting the first
judgment call forever. Repeating the flag costs a human a few seconds of
re-confirming "yeah, still no change needed" most of the time, in exchange
for never losing the chance to revisit that judgment as more of the
surrounding code lands.

The `testing/` -> `test/` check exists for the same reason, discovered the
same way: diffing a completed AI-driven sync (which made zero changes
anywhere under `projects/rocthrust/test/` for the whole `v3.0.3..v3.1.0`
range) against PR #11296, whose real diff touched 104 files there
(+2446/-1118 lines) — comparable in size to `testing/` itself. Concretely,
upstream commit `80c1a65d438785ae1e75c12925993e7af0b75d9a` ("Add variadic
ctor and CTAD to `zip_iterator`") touches `thrust/testing/zip_iterator.cu`;
PR #11296's hand-maintained counterpart, `test/test_zip_iterator.cpp`,
picked up a matching `TestZipIteratorConstructionFromIterators` addition
that nothing in the old pipeline would ever have prompted a human to add.

## `porting-categories.md`

See the sibling `porting-categories.md` file for the (currently small,
DRAFT) set of structural categories to check a commit against before
porting. Building a mature, incident-history-backed catalog takes years;
this one has none yet; treat it as a starting checklist, not an exhaustive
rulebook.

## When a build failure drives this skill

If you were invoked because a build broke on a specific file, find which
`todo.md` item introduced the change to that file, treat it as reopened
(un-tick it, note the failure), and work it again from step 3 above. There
is no dedicated build-verification skill for rocThrust — build/test
verification happens outside this skill family, whenever the human runs it.
