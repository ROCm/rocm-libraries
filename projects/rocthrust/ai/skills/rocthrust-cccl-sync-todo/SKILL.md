---
name: rocthrust-cccl-sync-todo
description: Sets up a sync branch and enumerates the upstream CCCL/Thrust commits that need to be ported into rocThrust. Does not port any code itself. Use when asked to start, kick off, or set up a CCCL sync into rocThrust.
---

# CCCL → rocThrust Sync (driver)

rocThrust lives at `projects/rocthrust/` inside the `ROCm/rocm-libraries` monorepo.
**rocThrust has no subtree-merge mechanism** for tracking upstream CCCL/Thrust.
The real historical sync,
[`1eb022d06b`](https://github.com/ROCm/rocm-libraries/commit/1eb022d06b)
("feat(rocthrust): CCCL 3.0.x changes (#3773)"), is a single-parent,
normal squash-merged PR. Its own description says it plainly: *"Most of the
changes here are pulling in the commits from CCCL 3.0.3, with the fixes and
workarounds attached to these commits"* — upstream commits were manually
ported one at a time, not merged via git. There is no git-native conflict
state to lean on here; the discipline has to be imposed by this skill family
instead.

This skill drives **one** stage of the sync: creating the sync branch and
producing the `todo.md` handoff file that enumerates, in strict chronological
order, every upstream commit that needs to be ported. It does not port any
code. It is preceded by an investigation stage and followed by two further
stages, each with their own skill:

- **`rocthrust-cccl-sync-investigate`** (Step 0, run first) — works out the
  CCCL/Thrust version delta from git history and the four-signal version
  check, and catalogs what changed in the targeted release(s): new features,
  new CMake options/feature macros, test/benchmark coverage gaps, and
  commits likely to be problematic for rocThrust. Produces a
  `cccl-investigation-<tag>.md` report that scopes and de-risks this driver.
- **`rocthrust-cccl-sync-resolve`** — works through `todo.md` one commit at a
  time, in order, porting each upstream commit by hand.
- **`rocthrust-cccl-sync-finalize`** — once every item in `todo.md` is
  ticked, runs the pre-landing checks, bumps `thrust/version.h`, and updates
  the CHANGELOG, leaving the sync staged and ready to commit.

## Conventions

- One sync branch per confirmed `$CURRENT_TAG..$TO_TAG` range.
- Commits are ported **strictly oldest-first, and are never reordered**: a
  later upstream commit may assume an earlier one has already landed (it may
  reference symbols the earlier commit introduces, or fix a bug the earlier
  commit itself introduced). Only the first unticked item in `todo.md` may
  ever be worked next.
- No `git merge` of any kind is run or in progress at any point in this
  pipeline. Do not look for `MERGE_HEAD` or conflict markers — they will
  never exist here.
- Do **not** push to origin or create a PR at this stage.
- `todo.md` is uncommitted scratch state, not part of the eventual landing
  commit. It can be discarded or attached to a ticket once the sync lands.

## Locating the repository

Before doing anything else, establish the path to the `rocm-libraries`
working tree and store it in `$ROCTHRUST_REPO`.

1. **Check if there is an obvious existing clone** (e.g. the user has
   mentioned a path, or `$ROCTHRUST_SKILLS_WORK_ROOT` is already set in the
   environment). If so, confirm with the user that it is the right one.
2. **If still unclear**, STOP and ask the user:
   > "Where is your `rocm-libraries` working tree? Please provide the full
   > path."

Do not guess or fall back to a hardcoded path. Once `$ROCTHRUST_REPO` is
established, pass it to every script invocation with
`--repo "$ROCTHRUST_REPO"`.

## Confirm the tag range

> If `rocthrust-cccl-sync-investigate` (Step 0) was already run, reuse its
> reported `CURRENT_TAG`/`TO_TAG` (or the first tag of `PENDING_TAGS`) and
> the `cccl-investigation-<tag>.md` report instead of recomputing — the
> check below should agree with it. If it disagrees, STOP and reconcile
> before proceeding.

If no investigation report exists yet, re-run the same version-delta script
the investigate skill uses — do not fork or reimplement it:

```bash
$SKILL_DIR/../rocthrust-cccl-sync-investigate/scripts/cccl-version-delta.sh --repo "$ROCTHRUST_REPO"
```

Go through the same STOP-and-confirm gate the investigate skill uses before
trusting a derived current-tag guess: none of its four signals is
authoritative on its own, so an explicit human confirmation of both
`$CURRENT_TAG` and `$TO_TAG` is required before creating a sync branch.

Remember the path-alignment gotcha the investigate skill documents: upstream
CCCL's Thrust source lives at `thrust/thrust/*.h`; rocThrust's ported copy
lives at `projects/rocthrust/thrust/*.h` (one extra nested `thrust/`
segment). Every upstream path must be translated by prepending
`projects/rocthrust/` when comparing against or writing into the local tree.

## Prep the repo

Before doing anything else, make sure the working tree is clean and the
`cccl` remote (`https://github.com/NVIDIA/cccl.git`) and `origin` are up to
date.

If there is in-progress local work on another branch, confirm with the human
before discarding or switching away from it.

If the working tree is not clean, suggest creating a separate worktree.

## Sync branch name

Use the git conventions to create a sync branch name, `$SYNC_BRANCH`. Ask
for a JIRA ticket if one exists, but don't require one — rocThrust syncs
have no fixed JIRA project (the historical CCCL 3.0 revert,
[PR #10464](https://github.com/ROCm/rocm-libraries/pull/10464), cites both
`ROCM-29174` and `EXSWSTRHPC-300` on the same PR). If the branch already
exists, confirm with the human before reusing or recreating it.

## Status check with the user

Ask the user to confirm what will happen. Print the following information:
- The JIRA ticket for this sync, if any
- The branch the sync will start from (`$SYNC_BASE`)
- The confirmed `$CURRENT_TAG` and `$TO_TAG`
- The name of the new branch that will be created (`$SYNC_BRANCH`)
- The number of commits that will need to be ported (see below)

The user must confirm that this looks ok before proceeding.

## Set up the sync branch

```bash
git checkout -b "$SYNC_BRANCH" --no-track "$SYNC_BASE"
```

There is no merge command here: the branch simply starts as an exact copy
of `$SYNC_BASE`; every subsequent change comes from hand-porting individual
commits in `rocthrust-cccl-sync-resolve`.

## Determine the commit list

Get the ordered list of upstream commits to port, scoped to the Thrust
subtree and already translated/flagged against `sensitive-files.md`:

```bash
$SKILL_DIR/scripts/rocthrust-commit-list.sh --repo "$ROCTHRUST_REPO" --from "$CURRENT_TAG" --to "$TO_TAG"
```

This is a thin wrapper around:

```bash
git log --no-merges --reverse "$CURRENT_TAG..$TO_TAG" -- thrust/
```

The whole `thrust/` subtree, not a subset. rocThrust's parity list includes
every upstream commit that touches Thrust, even ones that turn out to need
no change (CMake, benchmark-only, README/.gitignore removals). Narrower
scans made those commits silently vanish from `todo.md`. Listing a commit
doesn't mean it changes rocThrust: `rocthrust-cccl-sync-resolve` decides
that, and ticks it as N/A with a reason when it doesn't.

Upstream paths map to `projects/rocthrust/<same path>`, except
`thrust/benchmarks/` (plural), which maps to `projects/rocthrust/benchmark/`
(singular).

Each row the script prints also carries a `SCOPE` column, a comma-joined
subset of:

| Tag | Upstream path | Usual outcome in rocThrust |
|-----|---------------|----------------------------|
| `HEADER` | `thrust/thrust/` | port; may need CUDA→HIP adaptation |
| `TEST` | `thrust/testing/` | port to `testing/` and `test/` |
| `EXAMPLE` | `thrust/examples/` | port |
| `BENCH` | `thrust/benchmarks/` | port to Google Benchmark; nvbench-only changes are N/A |
| `CMAKE` | `thrust/cmake/`, other `CMakeLists.txt` | usually N/A: rocThrust's CMake is its own |
| `OTHER` | anything else, e.g. `thrust/README.md`, `thrust/.gitignore` | usually N/A; never delete rocThrust's own README/.gitignore |

See `rocthrust-cccl-sync-resolve/SKILL.md` step 3 for how the tag affects
classification.

There is no MERGED/NOT_MERGED classification step here — the range was
already human-confirmed in the step above, so every commit the script
returns needs to be ported.

## Write `todo.md`

Create a `todo.md` at the repo root by copying `todo.md.template` (sibling
of this SKILL.md) and filling in the placeholders. Append one
`- [ ] <sha> <subject>` line per commit returned by
`rocthrust-commit-list.sh`, in the **exact order printed** — do not sort,
group, or otherwise reorder them. Where the script flags a commit as
touching a `sensitive-files.md` pattern, add an inline `⚠` note after the
subject rather than moving the item.

## Do not port anything

This skill's job ends at producing `todo.md`. Do not open, read the diff of,
or apply any of the listed commits — that is `rocthrust-cccl-sync-resolve`'s
job. Do not touch `thrust/version.h`, `CHANGELOG.md`, or run any build —
that is `rocthrust-cccl-sync-finalize`'s job.

## Handoff

Report to the user:
- Path to the generated `todo.md`.
- Total commit count.
- The two follow-on skills: `rocthrust-cccl-sync-resolve` for the per-commit
  porting loop, `rocthrust-cccl-sync-finalize` once every checkbox is
  ticked.
