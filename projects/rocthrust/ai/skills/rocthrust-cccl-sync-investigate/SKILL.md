---
name: rocthrust-cccl-sync-investigate
description: Step 0 of the CCCL→rocThrust sync pipeline (investigate → todo → resolve → finalize). Investigates exactly what changed in the targeted CCCL release(s) BEFORE any code is merged. Determines which CCCL version rocThrust is currently aligned with (from a primary signal plus three weaker corroborating signals, since no version.mk-equivalent exists), the next CCCL release(s) to sync, then catalogs new features, new CMake options/feature macros, test/benchmark coverage gaps, and commits likely to be problematic for rocThrust. Produces an investigation report. Use when asked to investigate, scope, or survey an upcoming CCCL sync into rocThrust (i.e. "what changed in CCCL 3.1?").
---

# CCCL → rocThrust Sync — Step 0: Investigate

This skill runs **before** any code is ported. It produces a written
investigation of everything that changed in the CCCL release(s) about to be
synced into rocThrust, so the code sync can be planned and scoped. It is
purely **descriptive intelligence** — it does not port, resolve, or build
anything.

It is the first stage of the sync pipeline, followed by:

- **`rocthrust-cccl-sync-todo`** — creates the sync branch and writes
  `todo.md`, the ordered list of upstream commits to port. It reuses this
  report's confirmed `CURRENT_TAG`/`TO_TAG` rather than recomputing them.
- **`rocthrust-cccl-sync-resolve`** — ports the `todo.md` commits one at a
  time, strictly in order.
- **`rocthrust-cccl-sync-finalize`** — pre-landing checks, `THRUST_VERSION`
  bump, and CHANGELOG entry once every `todo.md` item is ticked.

## What this skill mutates

Almost nothing. The `rocm-libraries` working tree is **read-only** here. The
only side effect is adding the upstream `cccl` git remote (if missing) and
fetching it. No branch is created, no merge is started.

## Locating the repository

Establish the path to the `rocm-libraries` working tree and store it in
`$ROCTHRUST_REPO`:

1. If the user gave a path, or `$ROCTHRUST_SKILLS_WORK_ROOT` is set, confirm it.
2. Otherwise STOP and ask: *"Where is your `rocm-libraries` working tree?"*

Do not guess. Pass `--repo "$ROCTHRUST_REPO"` to every script invocation.

The base branch is `origin/develop` unless the human names another. Store it
as `$SYNC_BASE`.

## Phase A — Determine the version delta (one primary signal, three corroborating)

rocThrust has **no dedicated version-delta file** as machine-truth ground
truth — but it does have one strong first-party signal (Current version
signal D, below) plus three weaker corroborating
ones. None of the corroborating signals is authoritative on its own, and (per
dogfooding below) even Signals A and B agreeing is not enough to trust the
result on their own — which is exactly why Current version signal D now drives the derived
tag whenever it's available.

```bash
scripts/cccl-version-delta.sh --repo "$ROCTHRUST_REPO" --base "${SYNC_BASE:-origin/develop}"
```

This prints:
- **Current version signal A (corroboration/fallback)**: `_THRUST_REQUIRED_LIBCXX_VERSION_MAJOR/MINOR`
  from `projects/rocthrust/thrust/detail/config/libcxx.h` — the libcu++/
  libhipcxx version rocThrust currently requires. **Assumption, not
  guarantee**: this is read as tracking the overall CCCL release
  major.minor, since libcu++/CUB/Thrust are versioned together upstream as of
  CCCL. This has held historically but is not asserted as certain. Only used
  to derive `CURRENT_TAG` when Current version signal D is unavailable (e.g. an older
  rocThrust tree that predates the `THRUST_VERSION` macro).
- **Current version signal B (corroboration)**: a "CCCL X.Y" mention in the top of
  `CHANGELOG.md` (e.g. "CCCL 2.8.x compatibility is deprecated..."). Known to
  drift stale — a CHANGELOG mention can easily lag the real current version.
- **Current version signal C (corroboration)**: a curated, **local-only** fingerprint check
  against `version-fingerprints.tsv` (sibling of this SKILL.md). Each row
  pairs a known upstream code pattern with the CCCL tag it was introduced in;
  the script greps rocThrust's current tree for each pattern (no `cccl`
  remote or network needed) and reports the highest tag whose fingerprint is
  already present locally.

  This signal exists because Signals A and B **agreeing is not sufficient**:
  dogfooding this script found that rocThrust's `develop` already contained
  the `transform_input_iterator_t` fix from CCCL **v2.8.1** verbatim, while
  both Current version signal A (`libcxx.h`) and Current version signal B (CHANGELOG) confidently agreed the
  current baseline was **v2.8.0**. A later dogfooding pass found Current version signal D
  (once added) told the fuller story: the true baseline was **v2.8.5**,
  newer than even Current version signal C's floor. `version-fingerprints.tsv` is a
  **curated, growing list, not exhaustive** — the same DRAFT/expand-over-time
  discipline as `sensitive-files.md`. Add a new row whenever you (or a future
  run of this skill) discover rocThrust's tree already contains a
  fingerprinted upstream fix ahead of what the other signals reported.
- **Current version signal D (primary)**: the `THRUST_VERSION` macro from
  `projects/rocthrust/thrust/version.h`, decoded per its own documented
  formula (`major = value/100000`, `minor = value/100%1000`, `patch =
  value%100`) into an exact tag. Thrust's own doc comment calls this file
  "the only Thrust header that is guaranteed to change with every thrust
  release" — a first-party, exact, self-declared version, categorically
  stronger evidence than a libcxx-requirement guess, stale CHANGELOG prose,
  or curated fingerprints. `CURRENT_TAG` is set from this signal whenever it's
  present; Signals A/B/C above only apply as fallback/corroboration.
- A **Current version signal D status** (`ok` / `DRIFT` / `no THRUST_VERSION macro found`,
  possibly with a note if Current version signal C's floor tag is newer than Current version signal D's
  declared version), a **signal agreement** verdict for A/B (`ok` / `DRIFT` /
  `unknown`), and a separate Current version signal C **status** (`no fingerprints matched` /
  `ok (no contradiction)` / `DRIFT (local code already contains
  fingerprint(s) for: ...)`).
- A derived current-tag **guess** (exact, when Current version signal D is present) and the
  candidate pending CCCL tags up to the latest release, plus an eval-able
  summary (`CURRENT_TAG`, `NEXT_TAG`, `TO_TAG`, `PENDING_TAGS`,
  `VERSION_SIGNAL_AGREEMENT`, `SIGNAL_C_FLOOR_TAG`, `SIGNAL_C_STATUS`,
  `THRUST_VERSION_RAW`, `THRUST_VERSION_TAG`, `SIGNAL_D_STATUS`).

Pin the target with `--to <tag>` to stop short of the latest release. Capture
the eval-able block for the rest of the session:

```bash
eval "$(scripts/cccl-version-delta.sh --repo "$ROCTHRUST_REPO" \
        --base "${SYNC_BASE:-origin/develop}" 2>/dev/null \
        | sed -n '/eval-able summary/,$p' | grep -E '^[A-Z_]+=')"
```

### Confirm the version delta with the user (required gate)

**STOP here.** Even with Current version signal D as a strong first-party
signal, there is no file as reliable as a dedicated version-tracking file
would be, so this confirmation gate matters — and Current version signal D
being present is **not** grounds to skip it either: it is the best evidence
available, not infallible. Report what was found, be explicit about the
uncertainty, and let the human decide — do not silently trust the derived
guess.

If Current version signal D found an exact tag and it isn't contradicted by Current version signal C, phrase
it plainly, substituting the actual values, e.g. (the real result of this
skill's second dogfooding pass):

> "rocThrust's `thrust/version.h` declares `THRUST_VERSION 200805`, which
> decodes to **v2.8.5** exactly — this is Thrust's own self-declared release
> version, so it's stronger evidence than the older libcxx.h/CHANGELOG
> guesses (which only agreed on '2.8.x', not an exact patch). Current version signal C's
> fingerprint check doesn't contradict it either.
>
> Upstream CCCL has released **3.0.0** through **3.4.2** since v2.8.5. Can
> you confirm v2.8.5 is really the current baseline, and which target you
> want — 3.0.0, the latest, or somewhere in between?"

If Current version signal D is unavailable and the script fell back to Signals A/B, phrase it
as before:

> "rocThrust's `libcxx.h` requires libcu++/libhipcxx **2.8**, which I'm
> guessing corresponds to CCCL **2.8.x** — but that's an assumption, not a
> verified fact (there's no `THRUST_VERSION` macro to check exactly). The
> CHANGELOG corroborates this (it mentions 'CCCL 2.8.x compatibility is
> deprecated... brought forward to CCCL 3.0.x'), so Signals A and B agree,
> and Current version signal C's fingerprint check found nothing that contradicts it.
>
> Upstream CCCL has released **3.0.0** and **3.1.0** since. Can you confirm
> 2.8.x is really the current baseline, and which target you want — 3.0.0,
> 3.1.0, or the latest?"

If Current version signal C reports **DRIFT** against whichever signal is driving
`CURRENT_TAG` (the real case an earlier dogfooding run hit, before Current version signal D
existed), say so plainly instead of averaging it away:

> "Signals A and B both say rocThrust is currently at CCCL **2.8.0** — but
> Current version signal C's fingerprint check found that `find.h` already contains a fix that
> was only introduced in CCCL **v2.8.1**. That means the two 'agreeing'
> signals are likely both stale, and the real baseline is at least v2.8.1.
>
> Can you confirm the true current baseline before I pick a target? I'd
> suggest re-running with `--from v2.8.1` once you've confirmed, rather than
> trusting the v2.8.0 guess."

Also confirm the JIRA ticket for the report if one hasn't been provided.

Then wait for the answer and adjust before continuing:
- If the user corrects the current version, re-run the script with `--from <tag>`.
- If they pick a target short of the latest, re-run with `--to <tag>`.

Only proceed to Phase B once **both** the current version and target tag(s)
are confirmed. If there is no confident pending range, say so plainly and stop
rather than guessing a scope.

## Phase B — Make upstream CCCL available

Add the upstream remote (idempotent) and fetch tags:

```bash
cd "$ROCTHRUST_REPO"
git remote get-url cccl >/dev/null 2>&1 || git remote add cccl https://github.com/NVIDIA/cccl.git
git fetch cccl --tags -q
```

> **Path alignment gotcha.** CCCL's
> Thrust source lives at `thrust/thrust/*.h` inside the upstream repo (an
> extra nested `thrust/` directory), while rocThrust's is the flatter
> `projects/rocthrust/thrust/*.h`. A direct `git diff` between the two trees
> will show every file as added/removed due to the path mismatch. Scope all
> diffs to upstream's `thrust/thrust/` subtree and compare paths with that
> prefix mentally stripped, e.g.:
>
> ```bash
> git diff "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/ | less
> git diff --stat "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/ | tail -1
> ```
>
> If a rocm-libraries local clone of the corresponding rocThrust path is
> needed side-by-side, construct it by prefixing/stripping `thrust/` as
> appropriate rather than assuming the paths line up.

> If `rocm-libraries` is a partial/sparse clone, a `blob:none` gotcha may
> apply — if `git diff` fails with `could not fetch <oid> from promisor
> remote`, re-fetch with `git fetch cccl --tags --refetch -q`.

### Maintenance-branch drift check (confirmed gap, not hypothetical)

CCCL keeps a per-minor-version maintenance branch upstream (`branch/X.Y.x`,
e.g. `branch/3.1.x`) that keeps receiving backports **after** `$TO_TAG` is
cut — sometimes rolled into a later patch tag (`v3.1.1`, `v3.1.2`, ...),
sometimes landed but not yet tagged at all. Confirmed directly against a
real clone: `branch/3.1.x`'s tip sat 31 commits (~7 months) ahead of
`v3.1.0`, including a version-bump commit for an as-yet-untagged `v3.1.5`.
Of those 31, **8 touched `thrust/thrust`, `thrust/testing`, or
`thrust/examples`**, and at least 3 were substantive fixes, not version
bumps. Nothing about this is visible from tags alone.

Run the drift check right after fetching tags:

```bash
scripts/cccl-branch-drift-check.sh --repo "$ROCTHRUST_REPO" --to "$TO_TAG"
```

This is corroboration only — like Current version signals B/C in Phase A,
not authoritative — and a `NO MAINTENANCE BRANCH FOUND` result is not itself
a failure (older releases may predate the per-minor branch convention). But
if it reports `DRIFT`, **STOP and tell the human** before continuing to
Phase C: do not silently widen `$TO_TAG` to absorb the drifted commits, and
do not silently ignore them either. Typical resolutions: re-run Phase A with
`--to <later patch tag>` if one already exists and covers them, or record
the exclusion explicitly in the report's open questions (Phase D).

Sanity-check the range size:

```bash
git log --no-merges --oneline "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/ | wc -l
git diff --stat "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/ | tail -1
```

## Phase C — The four investigations (retargeted)

Do this **per tag** in `PENDING_TAGS` where it matters (features, CMake
options), and **across the whole range** where a roll-up is clearer
(problematic commits). Use the previous tag as the lower bound (`$PREV..$TAG`).

### 1. New features

Primary source is CCCL's release notes; corroborate with the commit log
scoped to `thrust/thrust/` and newly-added files/APIs.

```bash
gh release view "$TAG" --repo NVIDIA/cccl 2>/dev/null \
  || curl -fsSL "https://api.github.com/repos/NVIDIA/cccl/releases/tags/$TAG" | jq -r '.body'

git shortlog --no-merges "$PREV..$TAG" -- thrust/thrust/
git diff --name-status --diff-filter=A "$PREV..$TAG" -- thrust/thrust/

# Full commit-subject scan — do not skip this. --diff-filter=A only catches
# features that arrived as brand-new files; a new algorithm added inside an
# existing header (thrust::transform_n landed in an existing transform
# header) or a default-behavior change to an existing policy (cuda::par_nosync
# switching to async-allocation-by-default) touches zero new files and is
# invisible to the two commands above. Read every subject line for
# feature-shaped verbs (Add/Implement/Port/Perform/Deprecate), not just the
# added-files list:
git log --no-merges --oneline "$PREV..$TAG" -- thrust/thrust/
```

**Confirmed gap, not hypothetical**: a real dogfooding run of this skill
(`cccl-investigation-v3.1.4.md`) followed only the release-notes-skim +
added-files-diff steps above and missed `thrust::transform_n`, four
CUDA-backend iterator ports, the `par_nosync` behavior change, three
backend/performance commits, and a Thrust-facility deprecation commit — all
real, in-range, `thrust/thrust/`-scoped commits that simply modified existing
files. A human cross-checking the report against an independently-written
summary of the same release caught the gap after the fact. The full
commit-subject scan above is the fix; do not silently skip it as redundant
with `git shortlog` — `shortlog` groups by author and elides subjects a human
skimming for content actually needs to read.

This same full-log scan also surfaces notable **correctness fixes**, not just
new features — CCCL's own release notes commonly bucket both together, and a
bug fix to an existing `thrust::` API (e.g. a nullptr-deref fix in
`device_reference`, or a `malloc<void>` correctness fix) is exactly the kind
of change rocThrust's HIP backend needs to independently verify or port, the
same as a genuinely new feature. Record these with a `Fix:`-style note
distinct from feature bullets rather than filtering them out for not being
"new."

For each feature record: what it is, how it is turned on (CMake option /
feature macro / API), and its rocThrust relevance (AMD equivalent? likely
needs libhipcxx work? disabled?). Cite the introducing PR when the commit
log/release notes name one, and hyperlink it —
`[PR #<n>](https://github.com/NVIDIA/cccl/pull/<n>)` — so the report is
clickable without a separate lookup.

### 2. New CMake options / feature macros

Replaces "new env vars" — Thrust/CCCL surfaces new capabilities mostly via
CMake `option()`s and `_CCCL_*` / `USE_LIB*CXX`-style preprocessor macros, not
runtime env vars. Extract ones **added** in the range:

```bash
git diff "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/ \
  | grep -E '^\+' | grep -oE '_CCCL_[A-Z0-9_]+|USE_LIB[A-Z0-9_]+' | sort -u

git diff "$CURRENT_TAG..$TO_TAG" -- CMakeLists.txt cmake/ 2>/dev/null \
  | grep -E '^\+' | grep -oE 'option\([A-Za-z0-9_]+' | sed 's/option(//' | sort -u
```

For each new option/macro, find its default and read-site and note the
rocThrust disposition (new feature gate / needs libhipcxx integration / N/A on
AMD). Caveat: the `^\+` filter is approximate — eyeball the results.

### 3. Test / benchmark coverage gaps

rocThrust ports its own `testing/` and `benchmark/` suites from upstream
Thrust's. For each new feature from step 1, check whether rocThrust already
has a ported test/benchmark:

```bash
# What does upstream test/benchmark for this feature?
git log --oneline -20 "$PREV..$TAG" -- thrust/testing/ thrust/benchmarks/ 2>/dev/null

# Does rocThrust already have the equivalent ported?
git grep -l "<feature symbol>" "${SYNC_BASE}" -- projects/rocthrust/testing/ projects/rocthrust/benchmark/ 2>/dev/null
```

Output: which features need a ported test/benchmark, whether rocThrust has
one, and follow-up (port it / file a ticket). Mark uncertain cases for human
confirmation rather than guessing.

### 4. Potentially problematic commits

Flag commits touching files listed in `sensitive-files.md` (sibling of this
SKILL.md) — a **draft**, not-yet-battle-tested list seeded from
[PR #10464](https://github.com/ROCm/rocm-libraries/pull/10464) (the last
real, later-reverted CCCL sync attempt). There is no institutional incident
history backing this list yet.

```bash
SENSITIVE="thrust/system/hip thrust/detail/libcxx_wrapper thrust/detail/config/libcxx.h \
thrust/detail/config/cpp_dialect.h thrust/tuple.h cmake/Dependencies.cmake cmake/FindROCMCmake.cmake"

# Commits touching rocThrust-sensitive areas (scoped conceptually — adjust
# paths for the thrust/thrust/ vs projects/rocthrust/thrust/ prefix mismatch):
git log --oneline "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/tuple.h thrust/thrust/detail/config

# File renames/moves (build-break risk):
git diff --find-renames --name-status --diff-filter=R "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/

# CUDA intrinsics added to backend headers (need HIP equivalents):
git diff "$CURRENT_TAG..$TO_TAG" -- thrust/thrust/system/cuda \
  | grep -E '^\+' | grep -nE '__CUDA_ARCH__|__CUDACC__|asm[[:space:]]*\(|__ldg' || true
```

For each flagged commit, note **why** it is risky. When a commit deletes a
file rocThrust has an AMD-only equivalent for (or the reverse), state that
explicitly — don't assume either side is authoritative. Also call out public
API/ABI changes
(required libcu++/libhipcxx version bumps, new `thrust::` API surface).
Hyperlink every commit SHA cited in this section — most are upstream CCCL
commits, so default to
`[<sha>](https://github.com/NVIDIA/cccl/commit/<sha>)` — but check each one:
a SHA found by `git log`/`git show` against the local `rocm-libraries` clone
(e.g. the sync commit itself, or a provenance commit like the one behind
`libcxx.h`) is a rocm-libraries-local commit and must instead link to
`[<sha>](https://github.com/ROCm/rocm-libraries/commit/<sha>)`. Same
two-repo rule applies everywhere else a commit SHA is cited in the report —
Summary/"Deriving the Current Version of rocThrust", §1 New features, §5's upstream bug
fixes — not just this section.

## Phase D — Write the investigation report

Copy `report.md.template` (sibling of this SKILL.md) to the `rocm-libraries`
repo root as `cccl-investigation-<TO_TAG>.md` and fill in every placeholder
from the phases above (version delta signals, agreement verdict, the four
sections, and the handoff block).

Be honest about confidence: anything inferred rather than verified (especially
the version-delta guess and "is this disabled on AMD?") should be marked as a
question for the human, not asserted.

Hyperlink every PR reference in the report, not just the ones in §1 — this
includes any mentioned in §4 (problematic commits) and Handoff. Use the
correct repo per PR: CCCL/Thrust PRs →
`[PR #<n>](https://github.com/NVIDIA/cccl/pull/<n>)`; rocm-libraries PRs
(e.g. a past or planned sync attempt) →
`[PR #<n>](https://github.com/ROCm/rocm-libraries/pull/<n>)`. Don't leave a
bare `PR #<n>` in the final report.

The same applies to bare commit SHAs: don't leave one un-hyperlinked anywhere
in the report, including the Summary's "Deriving the Current Version of rocThrust"
sub-section (e.g. the libcxx.h provenance commit, or the sync commit found
while verifying the derived tag guess) and §1's New features prose. Apply
the same CCCL-vs-rocm-libraries disambiguation described in §4 above.

## Phase E — Make the report ticket-ready (forward-compatible scaffolding)

There is no `rocthrust-cccl-sync-tickets` skill yet to consume this, but
capture the decisions a ticket-filing skill would need, so the report is
ready whenever that skill exists:

1. **Per-feature disposition (decided, not open).** For every §1 feature:
   `port/validate`, `enable` (AMD path needs real porting), or
   `track/N-A/deferred` with a reason.
2. **NVIDIA-only exclusion list.** Items genuinely NV-only.
3. **Headline scope roll-up, per tag.**
4. **Enumerated upstream bug-fix list** warranting new/extended test coverage.

Where a disposition genuinely depends on the human, keep it in Open questions.

## Handoff

Report to the user:
- Path to `cccl-investigation-<TO_TAG>.md`.
- The version delta (`CURRENT_TAG` → `PENDING_TAGS`) and signal-agreement status.
- Counts: new features, new CMake options/macros, test/benchmark coverage
  gaps, problematic commits.
- **Next step**: once the human has reviewed the report, run
  `rocthrust-cccl-sync-todo` to create the sync branch and `todo.md` for the
  confirmed `CURRENT_TAG..TO_TAG` range.
- Whether the report is ticket-ready per Phase E, for whenever a tickets skill
  exists.
