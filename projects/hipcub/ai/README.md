# hipCUB CCCL-sync skills

This directory (`ai/skills/hipcub-cccl-sync*`) holds the skill family that
ports upstream NVIDIA CCCL/CUB changes into hipCUB (`projects/hipccl/hipccl3/hipcub/` in
`ROCm/rocm-libraries`). It's adapted from the rocThrust family in
`projects/rocthrust/ai/`, and follows the same pipeline and conventions,
with hipCUB-specific mappings. Not yet used for a real sync: treat
`sensitive-files.md`, `porting-categories.md` and the scripts' path
heuristics as drafts.

## Prerequisites for a new session

- A local clone of `ROCm/rocm-libraries` with a `cccl` remote pointing at
  `https://github.com/NVIDIA/cccl.git`.
- Know the path to that clone. Every skill asks for it up front and stores
  it as `$HIPCUB_REPO`. Nothing here guesses a path.

## How hipCUB differs from rocThrust

rocThrust is a copy of Thrust with a HIP backend; most upstream commits
apply to the same path. hipCUB is a **wrapper** with two backends:

- `hipcub/include/hipcub/<dir>/<x>.hpp` picks a backend by
  `__HIP_PLATFORM_AMD__`;
- `backend/rocprim/` reimplements the CUB API on rocPRIM (the AMD path);
- `backend/cub/` forwards to NVIDIA's CUB (the nvcc path), and only works
  with CCCL ≥ `CCCL_MINIMUM_VERSION` (`cmake/Dependencies.cmake`).

So porting a commit means deciding its API-visible effect and expressing it
in each backend. CUB's dispatch/agent/detail layers have no hipCUB copy;
commits there are N/A, or a follow-up for rocPRIM.

## The four skills, in pipeline order

1. **`hipcub-cccl-sync-investigate`** — Step 0. Read-only. Works out which
   CCCL version hipCUB tracks (`HIPCUB_CCCL_VERSION_*` in the root
   `CMakeLists.txt`, plus `CCCL_MINIMUM_VERSION`, the CHANGELOG and
   fingerprints), confirms a target tag with you, then catalogs features,
   CMake options/macros, test/benchmark gaps, risky commits and expected
   rocPRIM follow-ups. Produces `cccl-investigation-hipcub-<tag>.md`.
2. **`hipcub-cccl-sync-todo`** — the driver. Creates the sync branch and
   writes `todo.md` at the repo root: the ordered checkbox list of every
   upstream commit touching `cub/`. Does not port any code.
3. **`hipcub-cccl-sync-resolve`** — the workhorse. Works through `todo.md`
   one commit at a time, **strictly in order**, stopping for your go at
   each item and before editing, asking before any build or test run, and
   committing each item as one local commit (never pushed).
4. **`hipcub-cccl-sync-finalize`** — run once every item is ticked. A
   read-only audit: completeness/lint, upstream header renames vs hipCUB's
   `<cub/...>` includes, backend parity sweep, CMake/umbrella-header wiring,
   and a list of rocPRIM follow-ups and CCCL-minimum requirements. Changes
   nothing — no CHANGELOG, no version bump, no commit.

## Starting a new session

Describe what you want in plain language:

- *"What changed in CUB since hipCUB's last sync?"* →
  `hipcub-cccl-sync-investigate`
- *"Kick off a CCCL sync into hipCUB."* → `hipcub-cccl-sync-todo`
- *"Continue the hipCUB sync"* → `hipcub-cccl-sync-resolve`
- *"Everything in todo.md is ticked, audit the hipCUB sync."* →
  `hipcub-cccl-sync-finalize`

## Key conventions to know before you start

- **Layout.** rocThrust, hipCUB and rocPRIM live under
  `projects/hipccl/hipccl3/` (previously `projects/<name>/`; historical
  commits cited in these skills use the old paths). `projects/hipccl/hipccl2/`
  is a frozen compatibility snapshot: never port into it.

- **No git merge, ever.** Every upstream commit is hand-ported and
  committed locally as its own commit (`feat(hipcub): port CCCL <sha11> -
  <subject>`, empty for N/A items), so the sync branch maps 1:1 onto
  `todo.md`. Nothing is pushed by the skills.
- **`todo.md` order is strict.** Only the first unticked item may be worked
  next.
- **All of upstream `cub/`.** Every commit touching CUB is listed, with a
  `SCOPE` tag (`HEADER`/`INTERNAL`/`TEST`/`EXAMPLE`/`BENCH`/`CMAKE`/`OTHER`);
  those that don't apply are ticked N/A with a reason.
- **Release branches.** CCCL tags releases on `branch/X.Y.x`, so the
  current tag is often not an ancestor of the target and the range includes
  commits already backported into the current tag. They're kept in
  `todo.md` with a `BASELINE` mark (`=` patch-identical, `~` same PR) and
  usually ticked N/A after a check.
- **Recorded dispositions.** For each item, the tick-note records backend
  parity, rocPRIM follow-up, test counterpart and benchmark counterpart
  decisions where they apply, even "no change needed".
  `hipcub-todo-lint.sh` enforces their presence; finalize reports any
  violation as blocking.
- **Test and benchmark parity.** hipCUB keeps `test/hipcub/` and
  `benchmark/` at parity with upstream: Catch2 tests become Google Test
  `test_hipcub_*.cpp`, nvbench benchmarks become `benchmark_*.cpp` on
  hipCUB's harness. New files are registered in CMake in the same item.
- **Versions are the human's.** No skill bumps `HIPCUB_CCCL_VERSION_*`,
  `CCCL_MINIMUM_VERSION` or the CHANGELOG. Finalize lists the CCCL minimum
  the ported CUB-backend forwards need.
- **rocPRIM is a separate library.** Changes that need rocPRIM are recorded
  as follow-ups; `projects/hipccl/hipccl3/rocprim/` is edited only if you say so.
- **Scratch state** — `todo.md`, the investigation report, and files under
  `$HOME/.cache/hipcub-sync/` — is never committed.

## Where things live

```
ai/skills/
  hipcub-cccl-sync-investigate/   Step 0: scope + version delta
    SKILL.md, scripts/, sensitive-files.md, version-fingerprints.tsv,
    report.md.template
  hipcub-cccl-sync-todo/          driver: branch + todo.md
    SKILL.md, scripts/, todo.md.template
  hipcub-cccl-sync-resolve/       per-commit port loop
    SKILL.md, scripts/, porting-categories.md
  hipcub-cccl-sync-finalize/      read-only audit of the sync branch
    SKILL.md
```
