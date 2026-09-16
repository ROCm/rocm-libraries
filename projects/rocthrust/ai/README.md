# rocThrust CCCL-sync skills

This directory (`ai/skills/rocthrust-cccl-sync*`) is the canonical, live copy
of the skill family that ports upstream NVIDIA CCCL/Thrust commits into
rocThrust (`projects/rocthrust/` in `ROCm/rocm-libraries`). Use this copy —
not any other checkout of these skills — for both running them and editing
them.

## Prerequisites for a new session

- A local clone of `ROCm/rocm-libraries` with a `cccl` remote pointing at
  `https://github.com/NVIDIA/cccl.git` (the skills add this remote
  themselves if it's missing).
- Know the path to that clone. Every skill asks for it up front and stores
  it as `$ROCTHRUST_REPO` — have it ready, or be ready to answer when asked.
  Nothing here guesses a path.

## The five skills, in pipeline order

1. **`rocthrust-cccl-sync-investigate`** — Step 0. Read-only. Works out
   which CCCL version rocThrust currently tracks (via `THRUST_VERSION` plus
   three weaker corroborating signals), confirms a target tag with you, then
   catalogs new features, new CMake options/macros, test/benchmark coverage
   gaps, and commits likely to be risky for rocThrust. Produces
   `cccl-investigation-<tag>.md` at the repo root.
2. **`rocthrust-cccl-sync-tickets`** *(optional, independent)* — turns an
   investigation report's work breakdown into JIRA Epic/Story/Task tickets.
   Can be run before, during, or never; nothing downstream depends on it.
3. **`rocthrust-cccl-sync`** — the driver. Creates the sync branch and
   writes `todo.md` at the repo root: the ordered, checkbox list of every
   upstream commit that needs porting. Does not port any code itself.
4. **`rocthrust-cccl-sync-resolve`** — the workhorse. Works through
   `todo.md` one commit at a time, **strictly in order** (never skip ahead
   or batch), porting each commit's Thrust changes by hand and ticking it
   off. Everything ported stays staged (`git add`), never committed.
5. **`rocthrust-cccl-sync-finalize`** — run once every `todo.md` item is
   ticked. Runs a readiness/lint gate, bumps `THRUST_VERSION`, writes the
   CHANGELOG entry, runs `rmake.py`/`ctest` build+test verification, and
   creates the single landing commit.

## Starting a new session

Just describe what you want in plain language — each `SKILL.md` declares
its own trigger phrasing. For example:

- *"What changed in the next CCCL release we haven't synced yet?"* →
  `rocthrust-cccl-sync-investigate`
- *"Kick off a CCCL sync into rocThrust."* → `rocthrust-cccl-sync`
- *"Continue the sync"* / *"work the next todo.md item"* →
  `rocthrust-cccl-sync-resolve`
- *"Everything in todo.md is ticked, land it."* →
  `rocthrust-cccl-sync-finalize`
- *"File tickets for this sync."* → `rocthrust-cccl-sync-tickets`

There's no need to name a skill explicitly or read `SKILL.md` yourself
first — reference it by describing the task, and it will be picked up from
this directory.

## Key conventions to know before you start

- **No git merge, ever.** rocThrust has no subtree-merge mechanism. There
  is never a `MERGE_HEAD`, never conflict markers. Every commit is
  hand-ported and staged; one single commit lands the whole sync at the end
  (`rocthrust-cccl-sync-finalize`).
- **`todo.md` order is strict.** Only the first unticked item may be worked
  next — later commits may assume earlier ones already landed.
- **Three upstream paths, not one.** Every skill scopes to upstream
  `thrust/thrust/` (headers), `thrust/testing/`, and `thrust/examples/` —
  each translates to the same-named path under `projects/rocthrust/`.
- **Two AMD-only counterpart trees have no upstream commit trail of their
  own**: `thrust/system/hip/` (CUDA's HIP counterpart) and
  `projects/rocthrust/test/` (a hand-maintained GTest suite mirroring
  `testing/`, naming convention `testing/<name>.cu` →
  `test/test_<name>.cpp`). `rocthrust-cccl-sync-resolve` checks both for
  every commit and requires the disposition to be recorded in the
  `todo.md` tick-note, even when the answer is "no change needed" —
  `rocthrust-todo-lint.sh` (see `rocthrust-cccl-sync-resolve/scripts/`)
  mechanically enforces this and is a hard gate in `-finalize`'s readiness
  check.
- **`todo.md` and any investigation/report `.md` files are scratch state**,
  not part of the eventual landing commit — safe to discard once a sync
  lands, or attach to a tracking ticket.

## Where things live

```
ai/skills/
  rocthrust-cccl-sync-investigate/   Step 0: scope + version delta
    SKILL.md, scripts/, sensitive-files.md, version-fingerprints.tsv,
    report.md.template
  rocthrust-cccl-sync-tickets/       optional JIRA planning bridge
    SKILL.md
  rocthrust-cccl-sync/               driver: branch + todo.md
    SKILL.md, scripts/, todo.md.template
  rocthrust-cccl-sync-resolve/       per-commit port loop
    SKILL.md, scripts/, porting-categories.md
  rocthrust-cccl-sync-finalize/      version bump, CHANGELOG, build, land
    SKILL.md
```

Each `SKILL.md`'s frontmatter `description` is what triggers it — read the
file itself for the full step-by-step if you want the details behind any
particular phase.
