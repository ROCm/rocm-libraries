---
name: hipcub-cccl-sync-investigate
description: Step 0 of the CCCL→hipCUB sync pipeline (investigate → todo → resolve → finalize). Investigates exactly what changed in the targeted CCCL/CUB release(s) BEFORE any code is ported. Determines which CCCL version hipCUB currently tracks (from HIPCUB_CCCL_VERSION plus three corroborating signals), the next CCCL release(s) to sync, then catalogs new features, new CMake options/feature macros, test/benchmark coverage gaps, rocPRIM follow-ups, and commits likely to be problematic for hipCUB. Produces an investigation report. Use when asked to investigate, scope, or survey an upcoming CCCL/CUB sync into hipCUB (i.e. "what changed in CUB 3.1?").
---

# CCCL → hipCUB Sync — Step 0: Investigate

This skill runs **before** any code is ported. It produces a written
investigation of everything that changed in the CCCL/CUB release(s) about
to be synced into hipCUB, so the port can be planned and scoped. It is
purely descriptive: it doesn't port, resolve, or build anything.

It is the first stage of the sync pipeline, followed by:

- **`hipcub-cccl-sync-todo`** — creates the sync branch and writes
  `todo-hipcub.md`, the ordered list of upstream commits to port. It reuses this
  report's confirmed `CURRENT_TAG`/`TO_TAG` rather than recomputing them.
- **`hipcub-cccl-sync-resolve`** — ports the `todo-hipcub.md` commits one at a
  time, strictly in order.
- **`hipcub-cccl-sync-finalize`** — read-only audit of the sync branch
  once every `todo-hipcub.md` item is ticked.

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

## How hipCUB relates to CUB

hipCUB is not a copy of CUB. It is a wrapper with two backends, selected
at compile time:

| hipCUB path (under `projects/hipccl/hipccl3/hipcub/hipcub/include/hipcub/`) | Role |
|---|---|
| `<dir>/<name>.hpp` (top level) | Forwarding header: includes the rocPRIM backend on `__HIP_PLATFORM_AMD__`, the CUB backend otherwise |
| `backend/cub/<dir>/<name>.hpp` | nvcc backend: forwards to real CUB (`#include <cub/<dir>/<name>.cuh>`), built against the CCCL release in `CCCL_MINIMUM_VERSION` |
| `backend/rocprim/<dir>/<name>.hpp` | AMD backend: implements the CUB API on top of rocPRIM (`projects/hipccl/hipccl3/rocprim/`) |

The CUB backend re-exports many upstream headers (`block/`, `warp/`, most
of `thread/` and `iterator/`) directly from `backend/cub/hipcub.hpp`
without a wrapper file. So an upstream commit can need:

- nothing (dispatch, agents, tuning, kernels, `detail/`: hipCUB has no
  copy; rocPRIM has its own algorithms),
- a CUB-backend wrapper change (a forwarded signature changed),
- a rocPRIM-backend implementation (a new or changed API),
- a rocPRIM change (the rocPRIM backend can't express it otherwise).

Upstream path → hipCUB path:

| Upstream | hipCUB |
|----------|--------|
| `cub/cub/<dir>/<name>.cuh` | `hipcub/include/hipcub/{,backend/cub/,backend/rocprim/}<dir>/<name>.hpp` |
| `cub/cub/device/dispatch/**`, `cub/cub/agent/**`, `cub/cub/detail/**` | none (rocPRIM: `projects/hipccl/hipccl3/rocprim/rocprim/include/rocprim/device/**`) |
| `cub/test/**/catch2_test_<x>.cu` | `test/hipcub/test_hipcub_<x>.cpp` (or `.cpp.in`; often a shorter `<x>`) |
| `cub/benchmarks/bench/<algo>/<file>.cu` | `benchmark/benchmark_device_<algo>.cpp` (flat; nvbench → hipCUB's harness) |
| `cub/examples/block/<x>.cu` | `examples/block/<x>.cu` |
| `cub/examples/device/<x>.cu` | `examples/device/<x>.cpp` |
| `cub/test/test_util.h` (host helpers the examples use) | `examples/example_utils.hpp` |

## What this skill mutates

Almost nothing. The `rocm-libraries` working tree is **read-only** here. The
only side effect is adding the upstream `cccl` git -C "$HIPCUB_REPO" remote (if missing) and
fetching it. No branch is created.

## Locating the repository

Establish the path to the `rocm-libraries` working tree and store it in
`$HIPCUB_REPO`:

1. If the user gave a path, or `$HIPCUB_SKILLS_WORK_ROOT` is set, confirm it.
2. Otherwise STOP and ask: *"Where is your `rocm-libraries` working tree?"*

Do not guess. Pass `--repo "$HIPCUB_REPO"` to every script invocation.

The base branch is `origin/develop` unless the human names another. Store it
as `$SYNC_BASE`. Ask if unsure: the CCCL 3.0 port
([PR #9931](https://github.com/ROCm/rocm-libraries/pull/9931)) was reverted
on `develop` by [PR #10464](https://github.com/ROCm/rocm-libraries/pull/10464),
so `develop` and a branch carrying the 3.0 port report different baselines.

## Phase A — Determine the version delta (one primary signal, three corroborating)

```bash
"$SKILL_DIR"/scripts/cccl-version-delta.sh --repo "$HIPCUB_REPO" --base "${SYNC_BASE:-origin/develop}"
```

This prints:
- **Signal D (primary)**: `HIPCUB_CCCL_VERSION_MAJOR/MINOR/PATCH` from
  `projects/hipccl/hipccl3/hipcub/CMakeLists.txt`. hipCUB's own declared CCCL-compatible
  version, exported to users as `HIPCUB_CCCL_VERSION` (e.g. `300003` =
  3.0.3). Drives `CURRENT_TAG` when present.
- **Signal A (corroboration/fallback)**: `CCCL_MINIMUM_VERSION` from
  `projects/hipccl/hipccl3/hipcub/cmake/Dependencies.cmake`, the CCCL release the CUB
  backend finds or downloads. A minimum, so it can lag Signal D within a
  minor line (3.0.0 vs 3.0.3) without anything being wrong. Drives
  `CURRENT_TAG` only when Signal D is missing.
- **Signal B (corroboration)**: a "CCCL/CUB X.Y.Z" mention near the top of
  `CHANGELOG.md` (e.g. "Feature parity with CCCL/CUB 3.0.0."). Known to lag.
- **Signal C (corroboration)**: a curated, local-only fingerprint check
  against `version-fingerprints.tsv` (sibling of this SKILL.md). Empty for
  hipCUB so far; add a row whenever you find hipCUB already contains an
  upstream change newer than Signals A/D claim. In rocThrust, this signal
  caught a baseline that two agreeing signals had both understated.
- **Tag ancestry**: whether `CURRENT_TAG` is an ancestor of `TO_TAG`. CCCL
  tags every release, including `X.Y.0`, on its `branch/X.Y.x` maintenance
  branch, so normally it isn't (v3.0.0 and v3.0.3 are not ancestors of
  v3.1.4). Then `CURRENT_TAG..TO_TAG` also contains main-line commits that
  were backported into `CURRENT_TAG` under different SHAs.
  `hipcub-commit-list.sh` marks those in its `BASELINE` column.
- An eval-able summary (`CURRENT_TAG`, `NEXT_TAG`, `TO_TAG`, `PENDING_TAGS`,
  `HIPCUB_CCCL_VERSION_TAG`, `CCCL_MINIMUM_VERSION`, `SIGNAL_D_STATUS`,
  `VERSION_SIGNAL_AGREEMENT`, `SIGNAL_C_FLOOR_TAG`, `SIGNAL_C_STATUS`,
  `ANCESTRY_STATUS`).

`HIPCUB_VERSION` / `VERSION_STRING` (e.g. 5.0.0) is hipCUB's product
version, not a CCCL signal.

Pin the target with `--to <tag>` to stop short of the latest release. Capture
the eval-able block for the rest of the session:

```bash
eval "$("$SKILL_DIR"/scripts/cccl-version-delta.sh --repo "$HIPCUB_REPO" \
        --base "${SYNC_BASE:-origin/develop}" 2>/dev/null \
        | sed -n '/eval-able summary/,$p' | grep -E '^[A-Z_]+=')"
```

### Confirm the version delta with the user (required gate)

**STOP here.** Report what was found, be explicit about any disagreement,
and let the human decide. Signal D being present is not grounds to skip
the gate. For example:

> "hipCUB's `CMakeLists.txt` declares `HIPCUB_CCCL_VERSION` **3.0.3**;
> `CCCL_MINIMUM_VERSION` is 3.0.0 and the CHANGELOG says 'parity with
> CCCL/CUB 3.0.0', which is the same minor line. Upstream has released
> **3.1.0** through **3.5.0** since.
>
> v3.0.3 is not an ancestor of the targets, so the commit list will include
> some commits already backported into 3.0.x; they'll be marked.
>
> Can you confirm v3.0.3 is the real baseline, and which target you want?"

If the signals disagree across minor lines (e.g. Signal D says 3.0.3 but
`CCCL_MINIMUM_VERSION` says 2.8.2), say so plainly instead of averaging it
away, and ask which one reflects the code.

Also confirm the JIRA ticket for the report if one hasn't been provided.

Then wait for the answer and adjust before continuing:
- If the user corrects the current version, re-run the script with `--from <tag>`.
- If they pick a target short of the latest, re-run with `--to <tag>`.

Only proceed to Phase B once **both** the current version and target tag(s)
are confirmed.

## Phase B — Make upstream CCCL available

```bash
git -C "$HIPCUB_REPO" remote get-url cccl >/dev/null 2>&1 || git -C "$HIPCUB_REPO" remote add cccl https://github.com/NVIDIA/cccl.git
git -C "$HIPCUB_REPO" fetch cccl --tags -q
```

> If `rocm-libraries` is a partial/sparse clone and `git diff` fails with
> `could not fetch <oid> from promisor remote`, re-fetch with
> `git -C "$HIPCUB_REPO" fetch cccl --tags --refetch -q`.

### Maintenance-branch drift check

CCCL keeps receiving backports on `branch/X.Y.x` after `$TO_TAG` is cut,
sometimes untagged. Run the drift check right after fetching tags:

```bash
"$SKILL_DIR"/scripts/cccl-branch-drift-check.sh --repo "$HIPCUB_REPO" --to "$TO_TAG"
```

Corroboration only; `NO MAINTENANCE BRANCH FOUND` isn't a failure. If it
reports `DRIFT`, **STOP and tell the human** before Phase C: don't silently
widen `$TO_TAG`, and don't silently ignore the commits. Typical resolutions:
re-run Phase A with `--to <later patch tag>`, or record the exclusion in the
Summary's drift line.

Sanity-check the range size:

```bash
git -C "$HIPCUB_REPO" log --no-merges --oneline "$CURRENT_TAG..$TO_TAG" -- cub/ | wc -l
git -C "$HIPCUB_REPO" diff --stat "$CURRENT_TAG..$TO_TAG" -- cub/ | tail -1
git -C "$HIPCUB_REPO" diff --dirstat=files,2 "$CURRENT_TAG..$TO_TAG" -- cub/
```

When `ANCESTRY_STATUS` is `NOT ancestor`, count the commits already in the
baseline too:

```bash
git -C "$HIPCUB_REPO" log --no-merges --oneline --cherry-mark --right-only "$CURRENT_TAG...$TO_TAG" -- cub/ | grep -c '^='
```

## Phase C — The five investigations

Do this **per tag** in `PENDING_TAGS` where it matters (features, CMake
options), and **across the whole range** where a roll-up is clearer
(problematic commits). Use the previous tag as the lower bound (`$PREV..$TAG`).

### 1. New features

Primary source is CCCL's release notes; corroborate with the commit log.

```bash
gh release view "$TAG" --repo NVIDIA/cccl 2>/dev/null \
  || curl -fsSL "https://api.github.com/repos/NVIDIA/cccl/releases/tags/$TAG" | jq -r '.body'

git -C "$HIPCUB_REPO" diff --name-status --diff-filter=A "$PREV..$TAG" -- cub/cub/

# Full commit-subject scan — do not skip this. New APIs often land inside
# existing headers (a new DeviceReduce overload, an env-based overload) and
# don't show up as added files. Read every subject:
git -C "$HIPCUB_REPO" log --no-merges --oneline "$PREV..$TAG" -- cub/cub/
```

(The rocThrust version of this skill once skipped the full subject scan and
missed a new algorithm, four backend ports and a behaviour change, all
inside existing headers.)

Sort each tag's items into the report's `####` categories (New API, Behavior
changes, Deprecations, Removals, Fixes, Performance/tuning, NVIDIA-only).
For each item record what it is and what hipCUB needs, split by place:

- **rocPRIM backend**: implement or adapt in `backend/rocprim/`.
- **CUB backend**: wrapper change in `backend/cub/` (only when the
  forwarded API changed; re-exported headers need nothing).
- **rocPRIM follow-up**: the rocPRIM backend can't express it without a
  rocPRIM change.
- **Nothing**: tuning, kernels, PTX, SM-specific paths (NVIDIA-only).

Correctness fixes go under `#### Fixes`: a fix to a `cub::` algorithm
usually doesn't port (rocPRIM's algorithm is different code), but say
whether rocPRIM could have the same bug. Performance/tuning commits
(`dispatch/tuning/`, policy changes) are normally "Nothing" for hipCUB;
list them in one line rather than one bullet each.

Cite and hyperlink the introducing PR:
`[PR #<n>](https://github.com/NVIDIA/cccl/pull/<n>)`.

### 2. New CMake options / feature macros

```bash
git -C "$HIPCUB_REPO" diff "$CURRENT_TAG..$TO_TAG" -- cub/cub/ \
  | grep -E '^\+' | grep -oE '(_CCCL|CUB)_[A-Z0-9_]+' | sort -u

git -C "$HIPCUB_REPO" diff "$CURRENT_TAG..$TO_TAG" -- cub/CMakeLists.txt cub/cmake/ 2>/dev/null \
  | grep -E '^\+' | grep -oE 'option\([A-Za-z0-9_]+' | sed 's/option(//' | sort -u
```

For each new user-facing macro (e.g. a `CUB_DISABLE_*` switch), say whether
hipCUB should mirror it as a `HIPCUB_*` macro, whether the rocPRIM backend
can honour it, or N/A on AMD. Eyeball the results; the `^\+` filter is
approximate.

Mirror under the `HIPCUB_*` name, never by defining or reading the CCCL
name in hipCUB: 3.1's `CCCL_AVOID_SORT_UNROLL` became
`HIPCUB_AVOID_SORT_UNROLL` in the human port, and hipCUB's own fallback
definitions of `_CCCL_PRAGMA_*` became `HIPCUB_PRAGMA_*` (see §5). Also list
macros the range removes from upstream entirely that hipCUB's own code
still uses (none in 3.0.3..3.1.4):

```bash
git -C "$HIPCUB_REPO" diff "$CURRENT_TAG..$TO_TAG" -- cub/cub/ | grep -E '^-' \
  | grep -oE '\b_?(CCCL|CUB)_[A-Z0-9_]+' | sort -u \
  | while read -r m; do
      git -C "$HIPCUB_REPO" grep -qw "$m" "$TO_TAG" -- cub/ libcudacxx/ || \
        git -C "$HIPCUB_REPO" grep -lw "$m" "$SYNC_BASE" -- projects/hipccl/hipccl3/hipcub/ | sed "s|^|$m: |"
    done
```

### 3. Test / benchmark coverage gaps

hipCUB keeps its suites at parity with upstream. That is policy; don't raise
it as an open question.

| hipCUB | Mirrors | Framework |
|--------|---------|-----------|
| `test/hipcub/test_hipcub_<level>_<algo>.cpp` (some `.cpp.in`) | upstream `cub/test/**/catch2_test_<level>_<algo>*.cu` | Google Test (upstream uses Catch2) |
| `benchmark/benchmark_<level>_<algo>.cpp` | upstream `cub/benchmarks/bench/<algo>/*.cu` | hipCUB's benchmark harness (upstream uses nvbench) |

Upstream splits one algorithm across several Catch2 files
(`catch2_test_device_reduce.cu`, `..._reduce_api.cu`, `..._reduce_env.cu`)
and hipCUB usually has one. Match by algorithm, not by file name. Upstream's
non-Catch2 tests (`test_*_fail.cu`, `test_nvtx_*.cu`, `link_*.cu`,
`ptx-json/`) test CUDA-only build mechanics and are normally N/A.

For each new feature and fix from step 1, check whether the matching hipCUB
test exists and needs new cases. For benchmarks, compare paths:

```bash
git -C "$HIPCUB_REPO" diff --name-status "$CURRENT_TAG..$TO_TAG" -- cub/benchmarks/bench/
git -C "$HIPCUB_REPO" ls-tree --name-only "$SYNC_BASE" -- projects/hipccl/hipccl3/hipcub/benchmark/
```

New tests and benchmarks need explicit CMake registration
(`add_hipcub_test` / `add_hipcub_test_parallel` in
`test/hipcub/CMakeLists.txt`, `add_hipcub_benchmark` in
`benchmark/CMakeLists.txt`); there's no glob.

### 4. Potentially problematic commits

Flag commits touching files listed in `sensitive-files.md` (sibling of this
SKILL.md, a draft):

```bash
# Upstream counterparts of sensitive hipCUB files:
git -C "$HIPCUB_REPO" log --oneline "$CURRENT_TAG..$TO_TAG" -- cub/cub/config.cuh cub/cub/util_type.cuh \
  cub/cub/util_ptx.cuh cub/cub/util_macro.cuh cub/cub/thread/ cub/cub/iterator/

# Public header renames/moves/additions. backend/cub/ includes upstream
# headers by path, so a rename breaks the CUB backend's build, and an added
# public header must join backend/cub/hipcub.hpp's re-export list:
git -C "$HIPCUB_REPO" diff --find-renames --name-status --diff-filter=RDA "$CURRENT_TAG..$TO_TAG" -- cub/cub/

# Removed public APIs (both backends may still expose them):
git -C "$HIPCUB_REPO" log --oneline "$CURRENT_TAG..$TO_TAG" -- cub/cub/ | grep -iE 'remove|drop|deprecat'

# CUDA intrinsics / PTX in public block/warp/thread headers:
git -C "$HIPCUB_REPO" diff "$CURRENT_TAG..$TO_TAG" -- cub/cub/block cub/cub/warp cub/cub/thread \
  | grep -E '^\+' | grep -nE '__CUDA_ARCH__|asm[[:space:]]*\(|__shfl|__ldg|NV_IF_TARGET' || true
```

For every renamed or deleted public header, check whether hipCUB includes it:

```bash
git -C "$HIPCUB_REPO" grep -n '<cub/<old path>>' "$SYNC_BASE" -- projects/hipccl/hipccl3/hipcub/
```

For each flagged commit, note **why** it is risky, and list every rocPRIM
follow-up in the report's own subsection.

For every rocPRIM follow-up that is a new capability or a lifted limit
(64-bit sizes, more segments, a new mode), recommend doing it in rocPRIM
during this sync rather than deferring it: the human 3.1 port
([PR #13083](https://github.com/ROCm/rocm-libraries/pull/13083)) did
exactly that for RLE, segmented radix sort and segmented reduce.

When an item looks blocked on libhipcxx ("libhipcxx has no execution
environments / memory resources / X"), grep libhipcxx for X before writing
that down (`projects/hipccl/hipccl3/libhipcxx/` is a submodule and may be empty; then
look for an installed copy, or put it under Open questions as unverified).
The 3.1 AI run deferred the env-based `DeviceReduce` overloads on this
unverified claim; the human port implemented them.

### 5. Baseline parity debt

Earlier syncs can leave hipCUB behind upstream *at `$CURRENT_TAG`*: APIs
upstream already removed, legacy template parameters, internals hipCUB
still exposes, CCCL-namespace macros hipCUB defines itself, public headers
the CUB backend never re-exported. No commit in the range mentions them, so
the per-commit loop never sees them. The 3.1 human port cleaned up all of
these and the AI run missed every one.

```bash
"$SKILL_DIR"/scripts/hipcub-baseline-debt.sh --repo "$HIPCUB_REPO" --base "$SYNC_BASE" \
  --from "$CURRENT_TAG" --to "$TO_TAG"
```

Read each hit: check 1 also lists hipCUB-only extensions
(`IteratorWrapper`, `DigitExtractorAdopter`, ...), which aren't debt. For
each real hit, say what upstream did and when (e.g. "`Equality`,
`Sum`, `Max`, ... removed upstream before 3.0.3; still deprecated in both
hipCUB backends → remove"), and whether it's a removal (a breaking change
for hipCUB users, so name it as one). Also look by hand for internal
helpers hipCUB exposes that upstream dropped (in 3.1: the pointer-based
`ThreadReduce<Length>(T*)` overloads); the script can't find those.

Hyperlink every commit SHA: upstream CCCL commits as
`[<sha>](https://github.com/NVIDIA/cccl/commit/<sha>)`, rocm-libraries
commits as `[<sha>](https://github.com/ROCm/rocm-libraries/commit/<sha>)`.
Check which repo each SHA comes from.

## Phase D — Write the investigation report

Copy `report.md.template` (sibling of this SKILL.md) to the `rocm-libraries`
hipCUB project directory, `$HIPCUB_REPO/projects/hipccl/hipccl3/hipcub/` (absolute path), as
`cccl-investigation-hipcub-<TO_TAG>.md` and fill in every
placeholder. Delete the template's `<!-- ... -->` comments once followed.

### Writing style

The readers are hipCUB developers planning the port. They want to know
what changed and what they have to do about it:

- **Every item answers two questions**: what the change is, and what hipCUB
  needs to do (or "nothing"). One or two sentences each.
- **Leave out methodology and tooling commentary.** Don't say which command
  found something, or mention this skill.
- **Keep the baseline discussion out of the Summary.** One Baseline line,
  plus a sentence only if the signals disagreed in a way that matters.
- **Use real headings for groups**, not bold lines.
- **State each fact once.** Refer back to a section rather than repeating.
- **Keep diff statistics out** unless they change what someone has to do.

Put anything inferred rather than verified (e.g. "rocPRIM probably has the
same bug") under Open questions.

Hyperlink every PR: CCCL PRs →
`[PR #<n>](https://github.com/NVIDIA/cccl/pull/<n>)`; rocm-libraries PRs →
`[PR #<n>](https://github.com/ROCm/rocm-libraries/pull/<n>)`. No bare
`PR #<n>` or bare commit SHA anywhere in the report.

## Handoff

Report to the user:
- Absolute path to `cccl-investigation-hipcub-<TO_TAG>.md`.
- The version delta (`CURRENT_TAG` → `PENDING_TAGS`), signal agreement and
  tag ancestry.
- Counts: new features, new CMake options/macros, test/benchmark coverage
  gaps, rocPRIM follow-ups, problematic commits, baseline parity debt.
- **Next step**: once the human has reviewed the report, run
  `hipcub-cccl-sync-todo` to create the sync branch and `todo-hipcub.md` for the
  confirmed `CURRENT_TAG..TO_TAG` range.
