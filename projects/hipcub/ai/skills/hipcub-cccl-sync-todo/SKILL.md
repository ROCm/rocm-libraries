---
name: hipcub-cccl-sync-todo
description: Sets up a sync branch and enumerates the upstream CCCL/CUB commits that need to be ported into hipCUB. Does not port any code itself. Use when asked to start, kick off, or set up a CCCL sync into hipCUB.
---

# CCCL → hipCUB Sync (driver)

hipCUB lives at `projects/hipccl/hipccl3/hipcub/` inside the `ROCm/rocm-libraries` monorepo,
next to rocPRIM at `projects/hipccl/hipccl3/rocprim/`. Both used to live at
`projects/hipcub/` and `projects/rocprim/`, so historical commits cited in this
family use those paths. `projects/hipccl/hipccl2/` is a frozen compatibility
snapshot: never port into it.
**hipCUB has no subtree-merge mechanism** for tracking upstream CCCL/CUB, and
unlike rocThrust it is not a copy of upstream at all: it's a wrapper with
two backends, `backend/rocprim/` (the CUB API reimplemented on rocPRIM) and
`backend/cub/` (thin forwarding to NVIDIA's CUB when built with nvcc). The
last historical sync,
[`72b6de5f86e`](https://github.com/ROCm/rocm-libraries/commit/72b6de5f86e)
("feat(hipcub): Add CCCL 3.0.x support (copy) (#9931)"), is a single-parent,
squash-merged PR that hand-applied the CUB 3.0 API changes to both backends.
There is no git-native conflict state to lean on here; the discipline has to
be imposed by this skill family instead.

This skill drives **one** stage of the sync: creating the sync branch and
producing the `todo.md` handoff file that enumerates, in strict chronological
order, every upstream commit that needs to be ported. It does not port any
code. It is preceded by an investigation stage and followed by two further
stages, each with their own skill:

- **`hipcub-cccl-sync-investigate`** (Step 0, run first) — works out the
  CCCL/CUB version delta and catalogs what changed in the targeted
  release(s): new features, CMake options/feature macros, test/benchmark
  coverage gaps, and commits likely to be problematic for hipCUB or to need
  a rocPRIM follow-up. Produces a `cccl-investigation-hipcub-<tag>.md`
  report that scopes and de-risks this driver.
- **`hipcub-cccl-sync-resolve`** — works through `todo.md` one commit at a
  time, in order, porting each upstream commit by hand and committing it as
  one local commit per item.
- **`hipcub-cccl-sync-finalize`** — once every item in `todo.md` is ticked,
  audits the sync branch and reports findings, without changing anything.

## Conventions

- One sync branch per confirmed `$CURRENT_TAG..$TO_TAG` range.
- Commits are ported **strictly oldest-first, and are never reordered**: a
  later upstream commit may assume an earlier one has already landed. Only
  the first unticked item in `todo.md` may ever be worked next.
- No `git merge` of any kind is run or in progress at any point in this
  pipeline. Do not look for `MERGE_HEAD` or conflict markers.
- Do **not** push to origin or create a PR at this stage.
- `todo.md` is untracked scratch state and is never committed.
- Scratch files go under `$HOME/.cache/hipcub-sync/`, not `/tmp`.

## Locating the repository

Before doing anything else, establish the path to the `rocm-libraries`
working tree and store it in `$HIPCUB_REPO`.

1. **Check if there is an obvious existing clone** (e.g. the user has
   mentioned a path, or `$HIPCUB_REPO` / `$HIPCUB_SKILLS_WORK_ROOT` is
   already set in the environment). If so, confirm with the user that it is
   the right one.
2. **If still unclear**, STOP and ask the user:
   > "Where is your `rocm-libraries` working tree? Please provide the full
   > path."

Do not guess or fall back to a hardcoded path. Once `$HIPCUB_REPO` is
established, pass it to every script invocation with
`--repo "$HIPCUB_REPO"`.

## Confirm the tag range

> If `hipcub-cccl-sync-investigate` (Step 0) was already run, reuse its
> confirmed `CURRENT_TAG`/`TO_TAG` and the
> `cccl-investigation-hipcub-<tag>.md` report instead of recomputing — the
> check below should agree with it. If it disagrees, STOP and reconcile
> before proceeding.

If no investigation report exists yet, re-run the same version-delta script
the investigate skill uses — do not fork or reimplement it:

```bash
$SKILL_DIR/../hipcub-cccl-sync-investigate/scripts/cccl-version-delta.sh --repo "$HIPCUB_REPO" --base "$SYNC_BASE"
```

Go through the same STOP-and-confirm gate the investigate skill uses: none
of its signals is authoritative on its own, so an explicit human
confirmation of both `$CURRENT_TAG` and `$TO_TAG` is required before
creating a sync branch.

If the script's `ANCESTRY_STATUS` starts with `NOT ancestor`, tell the
human: CCCL tags releases on `branch/X.Y.x`, so `$CURRENT_TAG..$TO_TAG`
also lists main-line commits whose backports are already in
`$CURRENT_TAG`. The commit-list script marks them in its `BASELINE` column;
they stay in `todo.md` (never silently dropped) and are usually ticked N/A.

## Prep the repo

Make sure the working tree is clean and the `cccl` remote
(`https://github.com/NVIDIA/cccl.git`) and `origin` are up to date.

If there is in-progress local work on another branch, confirm with the human
before discarding or switching away from it.

If the working tree is not clean, suggest creating a separate worktree.

## Sync branch name

Use the git conventions to create a sync branch name, `$SYNC_BRANCH`. Ask
for a JIRA ticket if one exists, but don't require one (the 3.0 sync used
`EXSWSTRHPC-300`; its revert,
[PR #10464](https://github.com/ROCm/rocm-libraries/pull/10464), also cites
`ROCM-29174`). If the branch already exists, confirm with the human before
reusing or recreating it.

## Status check with the user

Ask the user to confirm what will happen. Print the following information:
- The JIRA ticket for this sync, if any
- The branch the sync will start from (`$SYNC_BASE`)
- The confirmed `$CURRENT_TAG` and `$TO_TAG`
- The name of the new branch that will be created (`$SYNC_BRANCH`)
- The number of commits that will need to be ported, and how many of them
  carry a `BASELINE` mark (see below)

The user must confirm that this looks ok before proceeding.

## Set up the sync branch

```bash
git checkout -b "$SYNC_BRANCH" --no-track "$SYNC_BASE"
```

There is no merge command here: the branch starts as an exact copy of
`$SYNC_BASE`; every subsequent change comes from hand-porting individual
commits in `hipcub-cccl-sync-resolve`.

## Determine the commit list

Get the ordered list of upstream commits to port, scoped to the CUB subtree
and flagged against `sensitive-files.md`:

```bash
$SKILL_DIR/scripts/hipcub-commit-list.sh --repo "$HIPCUB_REPO" --from "$CURRENT_TAG" --to "$TO_TAG"
```

This is a thin wrapper around:

```bash
git log --no-merges --reverse "$CURRENT_TAG..$TO_TAG" -- cub/
```

The whole `cub/` subtree, not a subset: test-only, benchmark-only and
CMake-only commits must not silently vanish from `todo.md`. Listing a
commit doesn't mean it changes hipCUB: `hipcub-cccl-sync-resolve` decides
that, and ticks it as N/A with a reason when it doesn't.

Output columns: `SHA SUBJECT PR FLAG SCOPE BASELINE`. `SCOPE` is a
comma-joined subset of:

| Tag | Upstream path | Usual outcome in hipCUB |
|-----|---------------|-------------------------|
| `HEADER` | `cub/cub/` public headers | backend parity decision: rocPRIM backend, CUB backend wrapper, or both |
| `INTERNAL` | `cub/cub/device/dispatch/`, `cub/cub/agent/`, `cub/cub/detail/` | N/A in hipCUB; possibly a rocPRIM follow-up |
| `TEST` | `cub/test/` | port the test intent to `test/hipcub/test_hipcub_*.cpp` (Google Test); Catch2/c2h plumbing is N/A |
| `EXAMPLE` | `cub/examples/` | port to `examples/` |
| `BENCH` | `cub/benchmarks/` | port to hipCUB's benchmark harness; nvbench/tuning-only changes are N/A |
| `CMAKE` | `cub/cmake/`, other `CMakeLists.txt` | usually N/A: hipCUB's CMake is its own |
| `OTHER` | anything else, e.g. `cub/README.md`, docs | usually N/A |

`BASELINE` is `=` (patch-identical commit already in `$CURRENT_TAG`), `~`
(a commit in `$CURRENT_TAG` cites the same PR number) or `-`.

See `hipcub-cccl-sync-resolve/SKILL.md` step 3 for how these affect
classification.

## Write `todo.md`

Create a `todo.md` at the repo root by copying `todo.md.template` (sibling
of this SKILL.md) and filling in the placeholders. Append one
`- [ ] <sha> <subject> [SCOPE: ...]` line per commit returned by
`hipcub-commit-list.sh`, in the **exact order printed** — do not sort,
group, drop, or otherwise reorder them. Add `[BASELINE: =]` or
`[BASELINE: ~]` when the column isn't `-`, and an inline `⚠` note when the
script flags a sensitive path, rather than moving the item.

## Do not port anything

This skill's job ends at producing `todo.md`. Do not open, read the diff of,
or apply any of the listed commits — that is `hipcub-cccl-sync-resolve`'s
job. Do not touch `CCCL_MINIMUM_VERSION`, `HIPCUB_CCCL_VERSION_*`, or
`CHANGELOG.md`, and don't run any build; none of the skills in this family
do those.

## Handoff

Report to the user:
- Path to the generated `todo.md`.
- Total commit count, and the `BASELINE`-marked count.
- The two follow-on skills: `hipcub-cccl-sync-resolve` for the per-commit
  porting loop, `hipcub-cccl-sync-finalize` once every checkbox is ticked.
