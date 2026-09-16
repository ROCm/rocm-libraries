# rocThrust/CCCL sensitive files (DRAFT)

> **Status: DRAFT, needs domain-expert review.** This is a first-pass,
> directory/pattern-granularity list, not an exhaustive per-file audit. It
> exists so Phase C.4 (cross-referencing upstream commits against known
> AMD-customized areas) has *something* to check against — but this list
> was built without any institutional incident history yet. Treat matches
> here as "look closer," not "this will conflict."
>
> `rocthrust-cccl-sync-resolve/porting-categories.md` is the per-rule
> conflict-resolution guidance counterpart to this pattern list, and it
> starts from nothing too. Expand both together as real ported commits
> produce real lessons; keep the pattern list here and the resolution
> guidance there rather than duplicating either direction.

## Provenance

Seeded from the changed-file set of
[`ROCm/rocm-libraries` PR #10464](https://github.com/ROCm/rocm-libraries/pull/10464)
— the last real attempt at a CCCL 3.0 sync into rocThrust/hipCUB, which was
reverted (management deferred the work, not a technical failure of the
approach). That PR touched 1123 files under `projects/rocthrust/` (880 of
them under `projects/rocthrust/thrust/` specifically), 185 under
`projects/hipcub/`, and 5 under `projects/rocprim/`. The patterns below are
the directories/files that concentrated the bulk of AMD-specific logic and
conflict risk in that attempt, not a full file listing.

## Patterns (relative to `projects/rocthrust/`)

- `thrust/system/hip/**` — AMD's HIP backend execution policies and kernels;
  the rocThrust analogue of upstream's `thrust/system/cuda/`. Structural
  changes upstream to `thrust/system/cuda/` usually require a parallel,
  non-mechanical port here, not a direct merge.
- `thrust/detail/libcxx_wrapper/**` — shims bridging Thrust's internal usage
  to either libcu++ or libhipcxx depending on platform. Sensitive to upstream
  changes in how Thrust selects/includes its C++ standard library shim.
- `thrust/detail/config/libcxx.h` — the required-libcu++/libhipcxx version
  macros this skill's version-delta script reads (`Phase A`). Also the
  `USE_LIBCUDACXX`/`USE_LIBHIPCXX` toggle logic and `THRUST_HAS_INCLUDE`
  fallback-to-`::std` behavior. Any upstream restructuring here directly
  affects version detection and libhipcxx integration.
- `thrust/detail/config/cpp_dialect.h` — C++ standard/dialect detection and
  gating; upstream bumps to minimum supported C++ standard ripple through
  build config on the AMD side too.
- `thrust/tuple.h` — historically a hotspot for libcu++/libhipcxx tuple
  implementation swaps; changes here tend to be structural rather than
  additive.
- `cmake/Dependencies.cmake` — declares the HIP → rocPRIM (required) and
  rocRAND/GTest/Benchmark/TBB/SQLite (test/benchmark-only) dependency chain,
  plus the `ROCTHRUST_USE_LIBHIPCXX`/`ROCTHRUST_USE_LIBCUDACXX` options. New
  upstream CMake dependencies typically need an AMD equivalent resolved here.
- `cmake/FindROCMCmake.cmake` — ROCm-specific CMake tooling discovery with no
  upstream CCCL equivalent; upstream build-system changes elsewhere in
  `CMakeLists.txt` can require corresponding updates here to keep discovery
  working.

## Known gaps not yet reflected here

- hipCUB's own sensitive-file set (this list is rocThrust-only; PR #10464
  also touched 185 hipCUB files and 5 rocPRIM files that would need separate
  triage if/when a hipCUB/CUB sync is prototyped).
- Incident-driven numbered resolver rules (e.g. a specific pointer-arithmetic
  or memory-drain gotcha) — none exist yet for rocThrust because no
  completed sync has produced that institutional knowledge. The reverted
  PR #10464 was deferred for management reasons, not because of a documented
  technical failure, so it doesn't yet supply that knowledge either.
