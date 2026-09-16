# rocThrust/CCCL porting categories (DRAFT)

> **Status: DRAFT, unvalidated.** RCCL's 26-rule conflict-resolution catalog
> (`rccl-nccl-sync-resolve/SKILL.md`) was built from years of real merge
> incident history. Nothing like that exists yet for rocThrust — the one
> real historical sync (`1eb022d06b`) predates this skill family, and PR
> #10464 was reverted for a management/scheduling reason, not a documented
> technical failure, so it doesn't supply incident-driven rules either. What
> follows is a small set of structural categories only, meant to be expanded
> once several commits have actually been ported through
> `rocthrust-cccl-sync-resolve` and produced real lessons.
>
> Do not treat this as exhaustive. When you hit a case not covered here,
> STOP and ask the human rather than guessing at a rule that isn't written
> down.

## Relationship to `sensitive-files.md`

`sensitive-files.md` (sibling of `rocthrust-cccl-sync-investigate`) lists
*where* AMD-specific logic concentrates — directory/file patterns, not
resolution guidance. This file is about *what to do* once a commit touches
one of those areas. Don't duplicate the pattern list here; when a category
below needs an example path, link back to `sensitive-files.md` instead of
restating it.

## Categories

### 1. AMD-only files — leave untouched
Some files under `projects/rocthrust/` have no upstream CCCL equivalent at
all (ROCm-specific CMake tooling, HIP-only backend files). If an upstream
commit's diff doesn't correspond to anything on the AMD side, there is
nothing to port — record it as N/A, don't invent a change to force a parallel.

### 2. CUDA→HIP translation points
Structural changes to upstream's `thrust/system/cuda/` usually require a
parallel, non-mechanical port to `thrust/system/hip/` rather than a direct
patch — see `sensitive-files.md`'s `thrust/system/hip/**` entry. The same
applies to `thrust/detail/libcxx_wrapper/**` (libcu++ vs. libhipcxx shims)
and `thrust/detail/config/libcxx.h` (required-library-version macros and the
`USE_LIBCUDACXX`/`USE_LIBHIPCXX` toggle). Treat any commit touching these
areas as "needs CUDA→HIP adaptation" by default, not "clean port."

This is no longer just prose guidance to remember unprompted:
`rocthrust-show-upstream-commit.sh`'s CUDA -> HIP counterpart check (see
`rocthrust-cccl-sync-resolve/SKILL.md`, step 2) automatically surfaces every
`thrust/system/cuda/**` file a commit touches alongside its
`thrust/system/hip/**` counterpart's status, and step 3 requires recording
that disposition in the tick-note explicitly, even when the answer is "no
HIP change needed." Before this check existed, a plausible-looking "no
other backend needed a change" verdict (the `reduce_into` item,
`7eedb6e38ef188d8a60e513fb1c287bbb12550a3`) turned out to miss an
AMD-specific addition the real historical port made anyway — see
`rocthrust-cccl-sync-resolve/SKILL.md`'s "Why the counterpart checks don't
de-duplicate" for the full account.

### 3. Build-system cross-check
If a commit adds, removes, or restructures upstream CMake targets or
dependencies, check whether `cmake/Dependencies.cmake` or the root
`CMakeLists.txt` need a corresponding update — new upstream CMake
dependencies typically need an AMD equivalent resolved there (see
`sensitive-files.md`'s `cmake/Dependencies.cmake` entry). Don't assume a
new upstream CMake option needs a rocThrust-side toggle without checking
whether it's CUDA-only first.

### 4. Copyright / file-header convention
**Confirmed for `examples/`, still open elsewhere.** PR #11296 bumped the
AMD copyright header's end year (`// Copyright (c) 2020-2025 ...` → `...
2020-2026 ...`) on every one of the 19 touched `examples/*.cu` files, no
exceptions — but did **not** do the same for any of the 302 other touched
files under `thrust/`, `testing/`, or `test/` that carry the same header
style. Treat the bump as an `examples/`-specific, mechanical gate (see
`rocthrust-cccl-sync-finalize/SKILL.md`'s step 5), not a rule to apply
elsewhere. For a ported file outside `examples/` that needs a new or
updated header for some other reason, ask the human rather than assuming
either upstream's or rocThrust's existing convention applies by default.

### 5. Never assume upstream deleted something
Before "fixing" a symbol, file, or code path that looks stale or removed,
confirm against the actual target tag rather than assuming the commit in
front of you already reflects the final state:

```bash
git show "$TO_TAG:thrust/thrust/<path>" | grep -n '<identifier>'
```

If it's still there at `$TO_TAG`, a subsequent commit in the ordered
`todo.md` list — not this one — is responsible for removing or moving it.
Porting out of order to "clean up early" breaks the strict-ordering
guarantee `rocthrust-cccl-sync`'s `todo.md` depends on.

### 6. Widened commit scope: testing/ and examples/, not just headers
`rocthrust-commit-list.sh` (and `rocthrust-show-upstream-commit.sh`) scan
three sibling upstream paths — `thrust/thrust/` (headers), `thrust/testing/`,
`thrust/examples/` — not just headers. This was widened after diffing an
AI-driven sync against the real historical port (PR #11296): scanning
headers alone made any commit whose *entire* diff lived under
`thrust/testing/` or `thrust/examples/` invisible to `todo.md`, not skipped
or flagged, simply never enumerated. A commit's `SCOPE` tag
(`HEADER`/`TEST`/`EXAMPLE`, comma-joined) tells you which of the three it
touches; see `rocthrust-cccl-sync-resolve/SKILL.md`'s step 3 for how that
tag changes the default classification.

Deliberately **not** widened further, and do not add these without a real
reason:
- `thrust/benchmarks/` — upstream is plural, rocThrust's local directory is
  `projects/rocthrust/benchmark/` (singular). The generic
  strip-"thrust/"-and-prefix path translation both scripts rely on would
  silently produce a nonexistent path.
- `thrust/cmake/`, `thrust/internal/`, `thrust/scripts/` — upstream CI/build
  tooling. rocThrust has same-named local directories, but they are
  independently-maintained AMD tooling, not ports of upstream's — including
  these would risk the same silent-wrong-translation trap as `benchmarks/`,
  just semantic instead of a naming typo.

New test/example files this widened scope surfaces need their own
CMake-registration check — see `rocthrust-cccl-sync-finalize/SKILL.md`'s
step 4 (test/example CMake-wiring check), which is a gate, not just a
summary.

### 7. `test/` — an AMD-only mirror of `testing/`, not a copy
`projects/rocthrust/test/` (singular) is a separate, hand-maintained GTest
suite with no upstream CCCL equivalent at all — unlike `testing/` (plural),
which is a 1:1 translated port of upstream's Catch2 suite. There is a
reliable naming convention (`testing/<name>.cu` → `test/test_<name>.cpp`)
but no git history ever links the two: a change to a `testing/<name>.cu`
file (or a header it exercises) can leave `test/test_<name>.cpp` behind with
nothing to notice.

Discovered the same way as category 2's `reduce_into` case: diffing a
completed AI-driven sync (zero changes anywhere under
`projects/rocthrust/test/` for the whole `v3.0.3..v3.1.0` range) against PR
#11296 (104 files changed there, +2446/-1118 — comparable in size to
`testing/` itself).

`rocthrust-show-upstream-commit.sh`'s `testing/` -> `test/` counterpart
check (see `rocthrust-cccl-sync-resolve/SKILL.md`, step 2) surfaces every
top-level `thrust/testing/*.cu` file a commit touches alongside its
`test/test_<name>.cpp` counterpart's status. It does **not** trigger on
`thrust/testing/cuda/`, `cpp/`, `omp/`, or `unittest/` subdirectory touches
(no `test/` counterpart pattern exists for those — `testing/cuda/` in
particular is a direct port to `projects/rocthrust/testing/cuda/`, see
category 1), nor on header-only commits that don't also touch a top-level
`testing/*.cu` file — the same structural limitation category 2's CUDA ->
HIP check has for header-only commits.

### 8. Macro rename/removal — content-triggered, not path-triggered
Unlike every other category here, this one has no single upstream path to
key off of: a commit that renames or drops a CCCL macro (e.g.
`_CCCL_INLINE_VAR`, `_CCCL_NODISCARD`, `_CCCL_HAS_CUDA_COMPILER`) can look
like a clean "N/A on AMD, no functional difference" call — the macro's
*behavior* on the HIP side is genuinely unaffected — while still leaving
stale AMD-side **references** to the old macro name behind. The trigger is
in the *content* of the commit's diff (does it touch a macro definition?),
not a path. Three known sub-cases, all confirmed by real historical PR
#11296:

- **Stale TODO comments.** A comment like
  `// TODO(libhipcxx): replace inline with _CCCL_INLINE_VAR once libhipcxx
  gets ready` becomes meaningless once upstream removes `_CCCL_INLINE_VAR`
  entirely — there is nothing left to eventually adopt. Confirmed by the
  real port of upstream's "Remove `_CCCL_INLINE_VAR`" commit
  (`1a3ecb199a8e0a7ec97b21b8d5b0f5da5e9845d8`): PR #11296 deleted this exact
  comment from all 4 files it appeared in
  (`detail/raw_reference_cast.h`, `detail/type_traits/iterator/is_output_iterator.h`,
  `iterator/detail/iterator_facade_category.h`, `iterator/detail/retag.h`),
  even though the AI-driven sync had (correctly, on functional grounds)
  marked that upstream commit N/A on AMD.
- **rocThrust's own translated macro aliases.** When upstream drops a macro
  in favor of a native language feature, rocThrust's own equivalently-named
  wrapper macro (`THRUST_NODISCARD`, a local translation of
  `_CCCL_NODISCARD`) needs the same textual replacement at its own call
  sites. Confirmed by the real port of upstream's "Drop `_CCCL_NODISCARD`"
  commit (`66973d47f32208aa3813a483a0d800a1e1bc795b`): PR #11296 replaced
  `THRUST_NODISCARD` with `[[nodiscard]]` directly in all 8 files that used
  it (`device_allocator.h`, `mr/allocator.h`, `mr/disjoint_pool.h`,
  `mr/disjoint_sync_pool.h`, `mr/fancy_pointer_resource.h`,
  `mr/memory_resource.h`, `mr/pool.h`, `mr/sync_pool.h`) — again, despite
  the AI-driven sync's N/A verdict being functionally correct.
- **`.clang-format`'s `AttributeMacros` list.** `.clang-format` is a
  local-only file, never touched by an upstream commit directly, so
  nothing prompts a human to check it. Confirmed by PR #11296: its
  `.clang-format` diff added `_CCCL_API` and removed `THRUST_NODISCARD`,
  `_CCCL_NODISCARD`, and `_LIBCUDACXX_HIDE_FROM_ABI[_AFTER_V1]` from
  `AttributeMacros`, tracking the same macro renames visible in that
  range's header diffs.

All three are invisible to the counterpart checks (2 and 7) and to the
`sensitive-files.md` pattern list (which only matches translated upstream
*paths*, and none of these three have one). No automated check exists for
any of them yet — whenever a commit's diff introduces, renames, or removes
a macro used as a `THRUST_HOST_DEVICE`-style attribute or feature-detection
annotation (patterns like `_CCCL_*`, `THRUST_*`, `_LIBCUDACXX_*`), a human
should, by hand:
1. grep already-ported files for TODO/comment references to the old macro
   name and remove ones that no longer make sense,
2. grep for rocThrust's own translated alias of that macro and update its
   call sites the same way upstream updated its own,
3. grep `.clang-format` for the old and new spellings.

An "N/A on AMD" verdict on functional grounds does not exempt a commit from
this sweep — the two confirmed cases above were both, correctly, N/A
functionally, and still needed a textual follow-up.

## What's deliberately not here yet

- Content-level CUDA-intrinsic-swap rules (RCCL's resolve skill has an
  extensive GIN/device-intrinsic shim catalog). rocThrust has no equivalent
  catalog yet — build one incrementally as real ported commits reveal
  recurring patterns, rather than guessing upfront.
- Any per-specific-file rule (RCCL's resolve skill has ~13 of these). None
  exist for rocThrust because no sync has been completed through this
  pipeline yet.
