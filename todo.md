# CCCL → rocThrust Sync: v3.0.3 → v3.1.0 — TODO

We are in the middle of a manual-port sync of upstream CCCL/Thrust commits
into rocThrust. There is no git merge in progress and never will be — this
sync is a series of hand-ported commits, landed as a single commit at the
end by `rocthrust-cccl-sync-finalize`. This file tracks which upstream
commits still need to be ported.

## Sync parameters

- **JIRA ticket**: none
- **Repo**: `ROCm/rocm-libraries` — rocThrust lives at `projects/rocthrust/`
- **Sync base branch** (`$SYNC_BASE`): `cccl3`
- **Confirmed current tag** (`$CURRENT_TAG`): `v3.0.3`
- **Target tag** (`$TO_TAG`): `v3.1.0`
- **Sync branch** (`$SYNC_BRANCH`, currently checked out): `users/jmahovsk/cccl-v3.1.0-sync`
- **Investigation report**: `cccl-investigation-v3.1.0-6.md`
- **Path-prefix reminder**: upstream commits touch `thrust/thrust/*`; the
  equivalent local path is `projects/rocthrust/thrust/*` (one extra nested
  `thrust/` segment). Translate every path before applying.

## ⚠ Ordering rule

Commits below are listed **strictly oldest-first**. Only the first unticked
(`- [ ]`) item may be worked next. Do not skip ahead, reorder, or batch
multiple commits together — a later commit may assume an earlier one is
already applied (it may reference symbols the earlier commit introduces, or
fix a bug the earlier commit itself introduced).

## State of the sync

- No `git merge` is in progress. There is no `MERGE_HEAD`, and no conflict
  markers will ever appear as part of this process.
- Ticked (`- [X]`) items below mean the commit's changes are **staged**
  (`git add`), not committed. Everything lands in a single commit at the end
  of `rocthrust-cccl-sync-finalize`.
- Because this state can persist across sessions with nothing enforcing it
  (unlike RCCL's open merge, which git itself blocks you from abandoning
  silently), avoid `git reset --hard` on this branch, and consider a backup
  branch if the sync spans many sessions.

## Commits to port

- [X] 4e5a7a46de21f180998c114b4ff20e6e3839f3bf Simplify Thrust iterator functions (#4178)
  - needs CUDA→HIP adaptation — upstream collapses advance.h/distance.h to
    `using ::cuda::std::...` forwarding declarations and deletes the now-redundant
    detail/.inl and system/detail/generic/{advance,distance}.{h,inl} files.
    Translated to rocThrust's existing portability macros instead of
    hardcoding libcu++: `#include _THRUST_STD_INCLUDE(iterator)` +
    `using _THRUST_STD::...` (matches the idiom already used by
    system/detail/generic/advance.inl for `_THRUST_STD_INCLUDE(type_traits)`).
    Verified no other rocThrust file references
    `system::detail::generic::advance`/`distance` before deleting those 4
    files, and confirmed all 6 deleted files are genuinely gone at v3.1.0
    upstream (not reintroduced by a later commit).
- [X] 62970a104bb5907c936a9a52f77f3afbd6a3d711 Avoid instantiating discard_iterator while parsing (#4180)
  - clean port — templatized `make_discard_iterator` on `System = use_default`
    in thrust/iterator/discard_iterator.h (translating `_CCCL_HOST_DEVICE` to
    rocThrust's `THRUST_HOST_DEVICE`, matching the file's existing usage);
    added the missing `thrust/iterator/iterator_traits.h` include to
    thrust/system/cuda/detail/core/load_iterator.h; qualified
    `detail::allocator_aware_execution_policy` as `thrust::detail::...` in
    thrust/system/cuda/detail/execution_policy.h. All three local files were
    byte-identical to their upstream pre-commit versions, so no HIP
    adaptation was needed — rocThrust keeps its own `system/cuda/` backend
    tree separate from `system/hip/`, and this commit only touches the
    former upstream. Noted but deliberately left untouched: the parallel
    unqualified `detail::allocator_aware_execution_policy<hip_rocprim::execution_policy>`
    in thrust/system/hip/detail/execution_policy.h:66 has the same latent
    pattern this commit fixes on the CUDA side, but upstream has no HIP
    backend and never touched it in this commit, so fixing it here would be
    out of scope.
- [X] 1a3ecb199a8e0a7ec97b21b8d5b0f5da5e9845d8 Remove `_CCCL_INLINE_VAR` (#4192)
  - N/A on AMD — no code change needed. All 4 touched files
    (raw_reference_cast.h, is_output_iterator.h, iterator_facade_category.h,
    retag.h) already use plain `inline constexpr` locally instead of
    `_CCCL_INLINE_VAR`, each with a
    `// TODO(libhipcxx): replace inline with _CCCL_INLINE_VAR once libhipcxx
    gets ready` comment — i.e. rocThrust's pre-existing libhipcxx-readiness
    workaround already matches the exact outcome this commit produces
    upstream (confirmed `_CCCL_INLINE_VAR` is fully gone from
    thrust/detail/raw_reference_cast.h at v3.1.0). Left the 9 now-stale TODO
    comments untouched per human direction — cosmetic cleanup is out of
    scope for this line item, but note for later: those TODOs can never be
    "resolved" as literally written since the macro they reference no
    longer exists upstream at all.
- [X] 60074e145c041609144bc1abed4d6cbc0598a5eb Make compiler version comparisons safer (#4185)
  - N/A on AMD — no code change needed across all 41 touched files. This
    commit converts `_CCCL_HAS_CUDA_COMPILER` from a bare object-like macro
    to a called function-like macro `_CCCL_HAS_CUDA_COMPILER()` upstream.
    39/41 touched files are under thrust/system/cuda/detail/** (rocThrust's
    CUDA-only backend mirror tree) and already use the raw upstream macro
    name with parens (`_CCCL_HAS_CUDA_COMPILER()`), i.e. already at the
    post-commit state. The remaining 2 files
    (thrust/detail/malloc_and_free.h, thrust/random/detail/normal_distribution_base.h)
    use rocThrust's own translated macro names
    (`THRUST_HAS_CUDA_COMPILER()`/`THRUST_CUDA_COMPILER(...)`), which were
    already function-like/parenthesized from the start, so the bare-vs-called
    distinction this commit fixes upstream never applied to the translated
    side. Verified by grepping every one of the 41 files for both spellings
    before ticking. (The commit body also mentions removing an MSVC2017
    check and swapping `_CCCL_CUDACC()` for `_CCCL_HAS_CUDA_COMPILER()`, but
    those hunks fall outside thrust/thrust/ and aren't part of this
    line item.)
- [X] 45cb0b2110b0c2a7ad4a1543fc3f5188f4f1a6c0 Allow cuda::par*.on() to take cuda::stream_ref (#4225)
  - clean port — applied verbatim to
    projects/rocthrust/thrust/system/cuda/detail/par.h: added
    `#include <cuda/stream_ref>` and four `on(::cuda::stream_ref const& s)`
    forwarding overloads (execute_on_stream_base,
    execute_on_stream_nosync_base, par_t, par_nosync_t), each calling
    `on(s.get())`. This file is entirely within rocThrust's CUDA-only
    backend mirror tree and already used raw upstream `_CCCL_HOST_DEVICE`
    conventions directly (not the translated HIP macro layer), so no
    adaptation was needed — matched upstream's own choice to leave the new
    overloads unannotated (no `__host__`/`__device__`/
    `THRUST_RUNTIME_FUNCTION`) since `cuda::stream_ref` is host-only.
- [X] 1595d252c0f2b6124d52a4dbe5cb02c943d571b9 Drop `_CCCL_NO_VARIABLE_TEMPLATES` (#4229)
  - N/A on AMD — no code change needed. rocThrust's
    thrust/type_traits/logical_metafunctions.h already unconditionally
    exposes all 6 symbols (`conjunction`, `conjunction_v`, `disjunction`,
    `disjunction_v`, `negation`, `negation_v`) via `_THRUST_STD::...`, with
    no `_CCCL_NO_VARIABLE_TEMPLATES` guard present at all — already exactly
    the post-commit state this commit produces upstream. Confirmed
    `_CCCL_NO_VARIABLE_TEMPLATES` has zero references anywhere in
    projects/rocthrust/.
- [X] 56e8daccc5a9964a480aa7bc47a674c35b9c6cd7 Simplify Thrust assign_value (#4227)
    - Needs CUDA→HIP adaptation. Rewrote `thrust/system/cuda/detail/assign_value.h`
      to match upstream's simplification, translating `NV_IF_TARGET`/`NV_IS_HOST`
      to this file's existing `_THRUST_IF_TARGET`/`_THRUST_IS_HOST` convention
      (confirmed via diff against pre-change local state — this file already used
      the translated macros, so only the new logic needed the same treatment).
      Also applied upstream's comment-only FIXME to
      `thrust/system/cuda/detail/internal/copy_cross_system.h` (no functional
      change). Verified all other referenced symbols (`cross_system.h`,
      `trivial_copy_device_to_device`, `trivial_copy_from_device`,
      `trivial_copy_to_device`, `triple_chevron`, `stream()`, `throw_on_error`,
      `synchronize_optional`, `it_value_t`, `derived_cast`,
      `CCCL_DETAIL_KERNEL_ATTRIBUTES`) already exist locally with matching
      signatures before porting verbatim.
- [X] 57f72a1cef71933747777a6d20eec32bfb043806 Use `[[maybe_unused]]` in whole cccl (#4207)
    - Clean port. Mechanical `(void) x;`/`static_cast<void>(dummy)` ->
      `[[maybe_unused]]` cleanup across 9 files (catrig.h,
      contiguous_storage.h, device_malloc_allocator.h,
      device_new_allocator.h, mr/new.h,
      random/detail/linear_congruential_engine_discard.h,
      system/cuda/detail/core/agent_launcher.h,
      system/cuda/detail/core/triple_chevron_launch.h,
      system/cuda/memory_resource.h). Pure C++ attribute syntax, no
      CUDA/HIP-specific macro translation needed. Confirmed all 9 local
      files matched the pre-change upstream state (with rocThrust's usual
      translated macro names, e.g. THRUST_HOST_DEVICE, THRUST_COMPILER)
      before applying.
- [X] 8b4d386c91454c052c0e51aa32370d94909db875 Make `cuda::stream_ref` constructible on device (#4243)
    - Clean port. Applied verbatim to `thrust/system/cuda/detail/par.h`:
      collapsed each pair of `on(cudaStream_t const&)` +
      `on(::cuda::stream_ref const&)` overloads (added by commit 45cb0b21,
      already ported as item 5) into a single `on(::cuda::stream_ref s) const`
      taken by value, in `execute_on_stream_base`,
      `execute_on_stream_nosync_base`, `par_t`, `par_nosync_t`. Used raw
      `_CCCL_HOST_DEVICE` (not `THRUST_HOST_DEVICE`) matching this file's own
      pre-existing convention for these methods' host/device annotation.
      Also dropped a stray `_CCCL_EXEC_CHECK_DISABLE` above
      `execute_on_stream_base`'s constructor per upstream.
- [X] 66973d47f32208aa3813a483a0d800a1e1bc795b Drop `_CCCL_NODISCARD` (#4265)
    - N/A, deferred. Upstream drops `_CCCL_NODISCARD` at 8 call sites in
      favor of raw `[[nodiscard]]` (C++17 guaranteed). Locally, all 8
      corresponding files use rocThrust's own `THRUST_NODISCARD` macro
      instead, 100% consistently (no raw `[[nodiscard]]` usage exists
      anywhere else in the tree). The macro's definition file
      (`thrust/detail/libcxx_wrapper/std/__cccl/attributes.h`) carries an
      explicit `TODO(libhipcxx): remove this file and replace
      THRUST_DECLSPEC_EMPTY_BASES, THRUST_NODISCARD* ... with
      _CCCL_NODISCARD* ... in rocThrust once libhipcxx gets ready` — the
      planned local migration path is `THRUST_NODISCARD`->`_CCCL_NODISCARD`,
      not straight to `[[nodiscard]]` as upstream did here. Left all 8 files
      untouched per user decision, to avoid diverging from that planned
      path ahead of libhipcxx readiness.
- [X] 9e8b4cb340397345f86ba96b19cd5c341d95b8c3 [Version] Update main to v3.1.0 (#4175)
    - Deferred to -finalize, ticked here. Upstream's own version bump
      (`THRUST_VERSION` 300000 -> 300100 in `thrust/version.h`). Per the
      sync design, the version-number bump is reserved for the -finalize
      step, applied once at the very end to encode the final `$TO_TAG`
      after all 72 commits are ported (so `cccl-version-delta.sh`'s Signal D
      only flips to `ok` when the sync is actually complete, not
      prematurely mid-sync). No file edited for this item; -finalize must
      still bump `thrust/version.h` to 300100 before landing.
- [X] ee57a8e2d419b070fe32a152460ee4b59a10f1b8 Use more libcu++ includes in thrust (#4316)
    - Mixed: 89 of 91 touched files ported and staged; 2 deferred pending
      dedicated review (see below). This commit's upstream diff is large
      (91 files under thrust/thrust/), so it was split into sub-batches by
      pattern rather than one mechanical sweep:
      - 24 files under system/cuda/detail/** already used `::cuda::std::`
        unconditionally (rocThrust's CUDA-only backend mirror), so upstream's
        diff applied verbatim with no translation.
      - ~65 remaining files needed indirection through rocThrust's portability
        macros instead of upstream's raw libcu++ spellings: bare
        `#include <LIB>` -> `#include _THRUST_STD_INCLUDE(LIB)`,
        `std::foo` -> `_THRUST_STD::foo` (move, decay_t, add_lvalue_reference_t,
        numeric_limits, initializer_list, malloc/free, etc.) for symbols with
        a portable libcu++/HIP-std equivalent. Two exceptions used literal
        `::std::` (not `_THRUST_STD::`) matching upstream exactly, since these
        have no libcu++ equivalent: `thrust/complex.h` and
        `detail/complex/complex.inl`'s use of `::std::complex<T>` and
        `::std::basic_istream`/`::std::basic_ostream` (guarded in both files
        by a new `#if !_CCCL_COMPILER(NVRTC)` block, matching upstream's own
        NVRTC guard, using the existing portable `_CCCL_COMPILER(NVRTC)`
        macro already used elsewhere in the tree — no new include needed).
      - `detail/complex/{catrig,catrigf,cproj,csinh,csinhf,csqrt,csqrtf}.h`
        each call the local custom `infinity<T>()` helper (defined in
        c99math.h, deferred below); upstream replaces each call site with
        `::cuda::std::numeric_limits<T>::infinity()`, which translates cleanly
        to `_THRUST_STD::numeric_limits<T>::infinity()` — this does **not**
        touch or delete c99math.h's `infinity<T>()` definition itself, so it
        carries none of that file's AMD-guard conflict and was applied.
      - `detail/complex/{ctanh,ctanhf}.h` only needed the trivial
        `#include <cmath>` -> `_THRUST_STD_INCLUDE(cmath)` swap (no
        `infinity<T>()` usage in either file) — applied.
      - **Deferred, needs dedicated review, NOT applied**:
        - `thrust/detail/complex/c99math.h` — upstream deletes the entire
          FreeBSD-derived `isinf`/`isnan`/`signbit`/`isfinite`/`copysign`/
          `copysignf`/`log1p`/`log1pf`/`hypotf`/`hypot`/`infinity<T>()` MSVC
          fallback block (~150 lines) in favor of
          `using ::cuda::std::{...};` declarations. That exact block contains
          AMD-specific HIP-on-Windows guards
          (`__HIP__`, `THRUST_HAS_HIP_COMPILER()`) interleaved with the code
          being deleted — applying upstream's diff verbatim would silently
          drop those guards. Human explicitly chose to defer this file for
          dedicated review rather than resolve it inline (asked via
          AskUserQuestion; answer: "Defer this file, continue others").
        - `thrust/system/detail/sequential/trivial_copy.h` — upstream drops
          the `NV_IF_TARGET(NV_IS_HOST, (std::memmove...), (general_copy_n
          fallback))` host/device branch entirely in favor of an
          unconditional `::cuda::std::memmove(...)` call (assumes device-side
          memmove now works via libcu++ — a behavioral/portability judgment
          call, not a mechanical translation). Local file uses AMD-renamed
          `_THRUST_IF_TARGET`/`_THRUST_IS_HOST`/`_THRUST_IS_DEVICE` macros for
          this same branch. Deferred by the same reasoning pattern the human
          approved for c99math.h (structurally identical: upstream deletes a
          host/device branch that carries AMD-specific naming/behavior),
          though not separately asked about — flag for confirmation.
      - Version bump line (`THRUST_VERSION` in `thrust/version.h`, part of
        this same PR upstream) intentionally NOT applied here — reserved for
        -finalize per the same policy as item 9's note above.
- [X] 72692d32d26ead0dbf1ee53883fde9e5de0e10c5 Remove extra semicolons (#4426)
    - Clean port. Mechanical trailing-semicolon removal across 6 files
      (detail/contiguous_storage.h, detail/range/head_flags.h,
      detail/range/tail_flags.h, detail/vector_base.h [2 sites],
      system/detail/bad_alloc.h, system/system_error.h): stray `;` after a
      macro invocation (`THRUST_SYNTHESIZE_SEQUENCE_ACCESS`/
      `THRUST_SYNTHESIZE_SEQUENCE_REVERSE_ACCESS` — rocThrust's renamed
      spelling of upstream's `_CCCL_SYNTHESIZE_SEQUENCE_ACCESS`/
      `_CCCL_SYNTHESIZE_SEQUENCE_REVERSE_ACCESS`, already ending in `}` /
      not requiring one) and after `noexcept {}` destructor bodies
      (`~bad_alloc()`, `~system_error()`). No CUDA/HIP-specific content;
      applied verbatim with only the macro-name translation.
- [X] bd60840e829b3f3355d103acfd3854af097230cc Attempt to recover from upstream OOM in disjoint_pool. (#4420)
    - Clean port (functional change, not mechanical) to
      thrust/mr/disjoint_pool.h: added `pool_idx` field to
      `chunk_descriptor`; added new `squeeze()` method that scans
      `m_allocated` for chunks with no blocks still in use and returns them
      to `m_upstream`, then drops all `m_cached_oversized` entries; split
      `do_allocate` into a thin try/catch wrapper (catches `std::bad_alloc`,
      calls `squeeze()`, retries once) plus the original logic moved into a
      new `do_allocate_impl`; replaced the hand-rolled min/max clamping
      logic for chunk growth with `_THRUST_STD::max`/`_THRUST_STD::min`
      (translating upstream's `::cuda::std::max`/`::cuda::std::min`, adding
      the granular `_THRUST_STD_INCLUDE(__algorithm/max.h)`/
      `_THRUST_STD_INCLUDE(__algorithm/min.h)` includes gated by
      `_THRUST_HAS_DEVICE_SYSTEM_STD`, mirroring the existing precedent in
      thrust/extrema.h and system/detail/sequential/stable_merge_sort.inl —
      also moved the pre-existing unconditional
      `#include <thrust/detail/algorithm_wrapper.h>` behind
      `#if !_THRUST_HAS_DEVICE_SYSTEM_STD` to match that same precedent);
      added `#include <thrust/find.h>` for the new unqualified `find(...)`
      calls in `squeeze()` (resolved via ADL, same pattern this file already
      relied on for pre-existing `find_if`/`lower_bound` calls); renamed
      `bucket_idx` -> `pool_idx` in both do_allocate_impl and do_deallocate
      per upstream. New `do_allocate_impl` uses `THRUST_NODISCARD` (not
      `[[nodiscard]]`) for consistency with this file's pre-existing
      `do_allocate` and with item 4's established local convention of
      keeping `THRUST_NODISCARD` pending libhipcxx readiness (this
      occurrence wasn't literally in that item's file list since it's new
      code introduced by this commit, but the same reasoning applies).
      `catch (std::bad_alloc&)` kept as upstream's literal spelling (real
      host-only C++ exception type, no libcu++ equivalent, no ambiguity
      with thrust's own `system::detail::bad_alloc` since fully qualified).
      Pre-existing untouched `(std::max)(bytes, m_options.smallest_block_size)`
      calls in do_allocate_impl/do_deallocate left as-is, matching upstream's
      own choice not to touch them in this commit.
      **Out of scope, not ported**: upstream's companion change to
      `thrust/testing/mr_disjoint_pool.cu` (new squeeze test, ~230 lines) —
      test files under `thrust/testing/` are outside this sync's tracked
      path (`-- thrust/thrust/`), consistent with how this todo list itself
      was generated; test coverage for `squeeze()` is not carried over by
      this line item.
- [X] c796a7ffff37b713cda08b607c2ed444f4e6b190 Move libcudacxx endian macros to cccl (#4429) ⚠ touches sensitive-files.md pattern
    - Ported, with a structural extension beyond a single mechanical edit —
      confirmed with the human before applying (AskUserQuestion; answer:
      "Extend the shim"). Upstream's only change under thrust/thrust/ is to
      detail/config/cpp_dialect.h: it collapses the MSVC/NVRTC-vs-clang/gcc
      `THRUST_COMP_DEPR_IMPL` branch into a single line using a new
      `_CCCL_WARNING` macro. Verified via `git show v3.1.0:libcudacxx/include/
      cuda/std/__cccl/compiler.h` that `_CCCL_WARNING` is genuinely new at
      v3.1.0 (absent at v3.0.3) — this is why sensitive-files.md's flag on
      cpp_dialect.h was warranted: the simplification depends on a macro that
      does not yet exist on rocThrust's side.
      Applied in two parts:
      - `thrust/detail/libcxx_wrapper/std/__cccl/compiler.h` (rocThrust's
        THRUST_-prefixed mirror of libcudacxx/libhipcxx's `_CCCL_*` macros,
        itself a sensitive-files.md pattern: `thrust/detail/libcxx_wrapper/**`):
        added `THRUST_WARNING(MSG)` to both branches — a passthrough
        `#define THRUST_WARNING(MSG) _CCCL_WARNING(MSG)` in the
        `_THRUST_HAS_DEVICE_SYSTEM_STD` branch (same pattern as the existing
        THRUST_PRAGMA/THRUST_COMPILER passthroughs immediately above it), and
        a literal copy of upstream's new MSVC/else `_CCCL_WARNING` definition
        (using the already-local `THRUST_PRAGMA`/`THRUST_TO_STRING`/
        `THRUST_COMPILER` equivalents) in the fallback branch. No deletions,
        no AMD-guard interaction — purely additive, so treated as safe to
        extend rather than deferred outright.
      - `thrust/detail/config/cpp_dialect.h`: applied upstream's diff
        verbatim, substituting `THRUST_WARNING` for `_CCCL_WARNING` (same
        translation convention as every other THRUST_-prefixed macro in this
        file); the dropped `THRUST_COMPILER(NVRTC)` branch is intentional —
        upstream's own new `_CCCL_WARNING` only distinguishes MSVC vs.
        everything else, so the port preserves that upstream behavior change
        rather than working around it.
      Confirmed `THRUST_COMPILER`/`THRUST_WARNING` reachability: cpp_dialect.h
      is only ever included via `thrust/detail/config/config.h`, which
      includes `thrust/detail/libcxx_wrapper/__cccl_config.h` (pulls in this
      shim) before `thrust/detail/config/cpp_dialect.h` itself — same
      resolution path THRUST_COMPILER already relied on pre-port, so no new
      include was needed in cpp_dialect.h.
- [X] 25a904960386202f2d8bd0da566833d35eb48b1e Improve some thrust iterators (#4461)
    - Clean port across 10 files. `iterator/constant_iterator.h`: default
      constructor changed to `= default` (dropping the explicit
      `THRUST_HOST_DEVICE`/base-init body, since defaulted special members
      need no annotation); removed the now-unused protected
      `value_reference()` accessors (verified via repo-wide grep: no callers
      anywhere in rocThrust, matches upstream deleting them outright);
      `Value m_value;` -> `Value m_value{}` (value-initialize, needed now
      that the default ctor no longer explicitly initializes it); added a
      CTAD deduction guide `constant_iterator(ValueT) -> constant_iterator<ValueT>`
      using `THRUST_HOST_DEVICE` in place of upstream's `_CCCL_HOST_DEVICE`.
      `iterator/zip_iterator.h`: added the new variadic constructor
      (`zip_iterator(Iterators&&...)`, SFINAE-gated on non-empty pack +
      `IteratorTuple` constructibility) and its CTAD deduction guide, both
      translated to `_THRUST_STD::`/`THRUST_HOST_DEVICE`; **both new pieces
      were placed inside this file's pre-existing
      `#if _THRUST_HAS_DEVICE_SYSTEM_STD` branch, not added unconditionally**
      -- this file already maintains a dual implementation (a modern
      `_THRUST_STD::tuple`-based path guarded by
      `_THRUST_HAS_DEVICE_SYSTEM_STD`, and a legacy `thrust::tuple`-based
      fallback marked `TODO(libhipcxx): remove all the code in the path of
      !_THRUST_HAS_DEVICE_SYSTEM_STD once libhipcxx gets ready`), and every
      other _THRUST_STD-only addition in this file already follows that same
      per-branch split (e.g. `dereference_impl`/`advance_impl`/
      `increment_impl`/`decrement_impl`). The fallback `thrust::tuple`-based
      `zip_iterator` does not gain the new constructor -- consistent with it
      being legacy/scheduled for removal, not a capability regression upstream
      introduced. The two `make_zip_iterator` overloads' device-std-branch
      bodies were updated to construct via `{...}` (brace-init, using the new
      variadic ctor directly) matching upstream; their `#else` (fallback)
      bodies were left untouched (still routing through
      `thrust::make_tuple`), since the multi-arg `make_zip_iterator(Iterators...)`
      overload itself already pre-existed in both branches before this
      commit -- this commit only changed its internal implementation where
      the new ctor is available.
      Remaining 8 files are pure call-site simplifications enabled by that
      pre-existing multi-arg `make_zip_iterator` overload (removing a
      redundant inner `thrust::make_tuple(...)` wrapper): `system/cuda/detail/
      unique.h`, `system/detail/generic/{copy,extrema,merge,mismatch,sort,
      transform}.inl`, and a doc-comment-only update in `zip_function.h`.
      These are backend-agnostic call sites (no CUDA/HIP-specific content)
      and applied verbatim identical to upstream's diff.
- [X] 9da054a4238d3b149f466cca403e3226e87a9d86 Deprecate more `Thrust` facilities in favor of `libcu++` ones (#4334)
      -- touches the identical 99-file set as item 17 above (same headline
      libcu++-migration effort, split across two upstream commits). Found
      already fully ported in the working tree (applied in a prior session
      before this commit was reached in order), so no new edits were made
      here -- this tick only records verification. Deprecates thrust's own
      functor/algorithm names (`plus`, `minus`, `multiplies`, `divides`,
      `modulus`, `negate`, `equal_to`, `not_equal_to`, `greater`,
      `greater_equal`, `less`, `less_equal`, `logical_and`, `logical_or`,
      `logical_not`, `bit_and`, `bit_or`, `bit_xor`, `distance`, `advance`,
      `next`, `prev`) in favor of direct `_THRUST_STD::` usage, marking the
      old names `THRUST_DEPRECATED_BECAUSE("Use _THRUST_STD::X instead")`.
      `maximum`/`minimum`/`not_fn` correctly remain routed through
      `::internal::` (the pre-existing local helper namespace), not
      `_THRUST_STD::`, matching established convention. Verified via an
      automated exact-line-diff audit (parsing the saved upstream diff's
      removed lines per file and confirming none survive verbatim in the
      current local file) across all 99 files -- clean except one benign
      false positive (`HAS_IDENTITY_ZERO =` in `reduce_by_key.h`, a
      multi-line-statement artifact, not a missed edit) -- plus manual
      spot-checks of `advance.h`, `distance.h`, `functional.h`,
      `type_traits/is_operator_less_or_greater_function_object.h`,
      `type_traits/is_operator_plus_function_object.h`, and the internal
      call sites in `detail/allocator/copy_construct_range.inl` and
      `detail/temporary_array.inl`. All 99 files already staged (same `git
      add` pass performed for item 17, since the file lists are identical).
- [X] fbf517d567badc1da166baa674c741353f119f0b [Thrust] Perform asynchronous allocations by default for the `par_nosync` policy (#4204)
      -- single file (`system/cuda/detail/temporary_buffer.h`), replacing
      the previous stub ("this system has no special temporary buffer
      functions") with `par_nosync`/`execute_on_stream_nosync`
      `get_temporary_buffer`/`return_temporary_buffer` overloads backed by
      `cudaMallocAsync`/`cudaFreeAsync` (falling back to
      `cudaMalloc`/`cudaFree` on failure, e.g. on platforms without async
      allocator support). Found already present verbatim at HEAD (`git diff
      HEAD` on this file is empty, byte-identical to upstream's post-commit
      content) -- landed via the prior `1eb022d06b` "feat(rocthrust): CCCL
      3.0.x changes (#3773)" sync commit despite this upstream commit being
      dated after `v3.0.3`, so no new edits or staging were needed here.
      Raw `cuda*` runtime API names (`cudaError_t`, `cudaMalloc`,
      `cudaFree`, `cudaSuccess`, `cudaGetLastError`) are left untranslated,
      consistent with the pre-existing convention across every other file
      in `system/cuda/detail/` (`util.h`, `parallel_for.h`, `error.inl`,
      `malloc_and_free.h`, etc. all use the same raw `cuda*` names) --
      not part of the `_THRUST_STD::` migration pattern from items 17/22,
      since these are runtime API calls, not libcu++ functor/algorithm
      usages.
- [X] 8da9b8fcbdec6878e03c55da62bbd08468e8be27 Use `cudaStream_t` for `thrust::device.on(...)`. (#4451)
      -- single file (`system/cuda/detail/par.h`), reverting `on(...)`'s
      parameter type on `execute_on_stream_base`, `execute_on_stream_nosync_base`,
      `par_t`, and `par_nosync_t` from `::cuda::stream_ref` back to raw
      `::cudaStream_t` (removing the now-unneeded `#include <cuda/stream_ref>`
      and the `.get()` unwrap at each of the 4 call sites), since
      `stream_ref` broke implicit conversions from custom stream wrapper
      types (nvbench, rmm, etc.). Applied verbatim identical to upstream's
      diff; confirmed no other `stream_ref` usages remain anywhere in
      `projects/rocthrust/thrust/`. Staged.
- [X] ceba82d43c01681c9aa051dbf030f3fe1bc8fd6b Missing forward include in iterator facade category (#4512)
      -- trivial single-line missing-include fix (adds
      `#include <thrust/iterator/detail/iterator_category_to_system.h>` to
      `iterator/detail/iterator_facade_category.h`). Already present at
      HEAD (`git diff HEAD` on this file is empty) -- landed via the prior
      `1eb022d06b` CCCL 3.0.x sync commit, same as item 23. No new edits or
      staging needed.
- [X] 7eedb6e38ef188d8a60e513fb1c287bbb12550a3 [Thrust] Add `reduce_into` (#4355)
      -- new API, `reduce_into` (a `reduce` variant returning `void` and
      writing the result through an output iterator instead of returning
      it by value; asynchronous under `par_nosync`, avoids a
      device-to-host copy of the reduction result). Ported across all 5
      touched files: `thrust/reduce.h` (6 new public declarations + full
      doxygen docs, 3 exec-policy overloads + 3 free-function overloads),
      `thrust/detail/reduce.inl` (matching definitions dispatching to
      `system::detail::generic::reduce_into`/`select_system`),
      `system/detail/generic/reduce.h`/`.inl` (generic fallback
      declarations/definitions -- the no-BinaryFunction overloads
      delegate down to `reduce_into(..., T(0))`/`plus<T>()`, the
      full-args overload falls back to plain `*output = thrust::reduce(...)`
      for backends without a specialized override), and
      `system/cuda/detail/reduce.h` (CUDA-specific optimization: splits
      the pre-existing `reduce_n_impl` into a shared
      `get_reduce_n_temporary_storage_size` helper + `reduce_n_impl` +
      new `reduce_n_into_impl`/`reduce_n_into`, which skips the extra `T`
      slot in the temporary allocation, writes directly to `output`, and
      synchronizes via `synchronize_optional` instead of `synchronize` so
      it is genuinely async under `par_nosync`). No other backend
      (`sequential`/`omp`/`tbb`/`hip`) gained a `reduce_into` override,
      matching upstream's own scope (only CUDA gets the optimized path;
      others use the generic fallback). Also applied this commit's
      incidental renames in the pre-existing generic `reduce`
      implementations (`InputType`/`RandomAccessIterator`/`OutputType` ->
      `T`/`InputIterator`/`T`) to match upstream. Translated per
      established convention: `_CCCL_HOST_DEVICE`/`::cuda::std::plus`/
      `::cuda::std::distance` -> `THRUST_HOST_DEVICE` (non-cuda-detail
      files) or `_CCCL_HOST_DEVICE` (cuda-detail file, matching that
      file's own pre-existing style) / `_THRUST_STD::plus` /
      `_THRUST_STD::distance`; doc-comment code examples using
      `cuda::maximum{}` + `#include <cuda/functional>` -> `::internal::
      maximum<int>()` + `#include <thrust/functional.h>`, matching this
      exact file's own pre-existing `reduce` doc examples. Verified via
      full `git diff` review against the saved upstream diff -- structure,
      control flow, and comments match line-for-line modulo the above
      substitutions. Staged.
- [X] 58e329d12cae8bd73bead7524247f8f24e013a48 Implement `shuffle_iterator` iterator type (#4564)
      New feature: extracted the pre-existing local `feistel_bijection`
      class out of `system/detail/generic/shuffle.inl` into a new public
      header `thrust/detail/random_bijection.h` (adding an `index_type`
      alias, a new `size()` accessor, and switching the key-generation loop
      to go through `thrust::uniform_int_distribution<std::uint32_t>`
      instead of calling `g()` directly), plus a brand-new
      `random_bijection<IndexType, Bijection>` adaptor class that iterates
      a bijection until the result is back in range (for non-power-of-two
      problem sizes). Built a new public fancy iterator
      `thrust/iterator/shuffle_iterator.h` (`shuffle_iterator<IndexType,
      BijectionFunc>` + `make_shuffle_iterator`) on top of it, following the
      same `make_*_iterator_base` + `iterator_adaptor` structure as sibling
      fancy iterators. Updated `shuffle.inl` to remove its local
      `feistel_bijection` and reference `thrust::detail::feistel_bijection`
      instead (3 call sites: `construct_key_flag_op`'s member/ctor,
      `shuffle_copy`'s local variable), matching upstream's diff exactly.
      Both new files live outside `system/cuda/detail/**`, so applied the
      established non-cuda-detail translation conventions throughout:
      `_CCCL_HOST_DEVICE` -> `THRUST_HOST_DEVICE` (confirmed against sibling
      `thrust/detail/*.h` files like `contiguous_storage.h`/`copy.h` and
      sibling `thrust/iterator/*.h` files like `counting_iterator.h`/
      `offset_iterator.h`, all of which use `THRUST_HOST_DEVICE`
      exclusively -- also cross-checked that `shuffle.inl`'s own
      pre-existing `feistel_bijection` already used `THRUST_HOST_DEVICE`,
      confirming this is the correct macro for this whole feature, not just
      convention-by-analogy); `::cuda::std::is_integral_v`/
      `is_convertible_v`/`is_constructible_v`/`enable_if_t`/`forward`/
      `ptrdiff_t` -> `_THRUST_STD::` equivalents (matching
      `zip_iterator.h`'s established precedent); `#include <cuda/std/X>` ->
      `#include _THRUST_STD_INCLUDE(X)` (a previously-undocumented but
      well-established convention, confirmed via `constant_iterator.h`,
      `random.h`, and `detail/alignment.h` all using this macro for their
      `cuda/std` includes instead of the raw upstream spelling). One
      genuinely new translation needed: upstream's `::cuda::std::_If<Cond,
      IfRes, ElseRes>` (a libcu++-internal alias used to pick the
      `difference` type based on `sizeof(IndexType)`) has no
      `_THRUST_STD::_If` equivalent yet -- found the dedicated wrapper
      `::internal::If<Cond, IfRes, ElseRes>` in
      `thrust/detail/libcxx_wrapper/std/__type_traits/conditional.h` (which
      carries an explicit `TODO(libhipcxx)` comment saying to switch to
      `_THRUST_STD::_If` once libhipcxx supports it), and confirmed the
      exact same `::internal::If<sizeof(X) < sizeof(int), int,
      _THRUST_STD::ptrdiff_t>` pattern already used for an equivalent
      `difference` type calculation in `counting_iterator.h`'s
      `counting_iterator_difference_type` alias -- reused verbatim rather
      than inventing a new pattern. Also confirmed via a repo-wide grep that
      raw `::cuda::std::_If` is still the correct choice inside
      `system/cuda/detail/**` (e.g. `reduce_by_key.h:966`), so this
      wrapper-vs-raw distinction is layer-specific, matching the general
      `_THRUST_STD::`-vs-raw rule already established for other traits.
      `_CCCL_ASSERT` and bare `assert()` left untranslated as-is (no
      thrust-specific wrapper exists anywhere in the repo for either; both
      are treated as generic CCCL/libc infrastructure, consistent with
      other untouched `_CCCL_*` macros like the system-header pragma
      guards). `std::move` -> `_THRUST_STD::move` in the second
      constructor, matching `offset_iterator.h`'s existing precedent for
      the same pattern. Verified the full upstream diff (508 lines, 3
      files, 2 new) doesn't touch anything else (no umbrella header needs
      the new files added). Verified via `git diff` that the `shuffle.inl`
      edit matches upstream's diff line-for-line modulo the
      `thrust::detail::` qualification. Staged all 3 files.
- [ ] 83ea8015adfe80bad055067e6af4022b550aad15 Move NVTX to libcu++ and add support for Thrust (#4537)
- [ ] 6887634f24fd5a27f271b84d1517a16cd8789bd2 Replace use of `__CUDACC__` with `_CCCL_CUDA_COMPILATION()` (#4587)
- [ ] a0a7b3856e9110722fec1d9ed474648251de033d Reduce the use of `__CUDA_ARCH__` (#4589)
- [ ] 1f5498642c12bb6d5ec528e5efc023d50107fb04  Improve defence against the external macros (#4635)
- [ ] eb7b68775a92b36307db023e6351b6025d6e2d8e c.parallel: reuse CUB agent policies for reduce (#4286)
- [ ] 1b13308ef2e3849dddcf74ac5db230f480021915 Use list init for test data in iterator docs (#4738)
- [ ] a3a00b9164a9c3893aca63dc5186ee72bb98c3ae Make `device_reference<T>::operator=` `const` (#4740)
- [ ] 0c5d8878687f4851516d8e0f013b696368fe7856 Simplify thrust::cuda_cub::swap_ranges (#4182)
- [ ] 4e7e7cd5cf4878123ea9c2fbaac8515bf74a7a7d Add thrust::transform_n (#4750)
- [ ] ab1ba00183c55b4e61e2ce95faf748718700ebec Allow default-initializing and skipping initialization of Thrust vectors (#4183)
- [ ] 525ef99f44f572a7956c6a1c8ac49616766780d7 Add `thrust::strided_iterator` and a step for `thrust::counting_iterator` (#4014)
- [ ] e1f04c7d907707918233a5bb54c46fbeb5b6f81a Small refactorings in Thrust CUDA (#4764)
- [ ] 3e2f6e9a6e0436f9d1d2ea8b407dd76d3dc94dc4 Make sure that `cuda` iterators play nicely with the thrust system and traversal machinery (#4806)
- [ ] c38a6dd5a1d6ad6f3e36bfd745854506312af975 fix cudax's vector_add example that was broken by #4795 (#4814)
- [ ] 035da530641bff6c2d78b324294bd3e6c20ab9d6 Refactor Thrust select_system (#4762)
- [ ] 4d5040efa532a3e9e21bee0ec1d42ed741b3fe48 Port `thrust::strided_iterator` to cuda (#4808)
- [ ] 584fedba37077b0040cccf532229daa2d80ba312 Refactor Thrust internal_functional (#4810)
- [ ] 85694c049e5af7b5c0453f272d32f6f74a99f9a0 Refactor `thrust::minimum_type|minimum_system` (#4042)
- [ ] 5973d44bd59e1c438ccba4c09725df4f023dcf35 Port `thrust::constant_iterator` to cuda (#4812)
- [ ] 6210162082bb2064e4e76b8632b42708f3e250af Use `cuda::std::type_identity` instead of *identity-like* types (#4893)
- [ ] 599d21f090af655d2b0061f3202ed59eaa3517d1 Drop cuda::std::__identity (#4887)
- [ ] 7371801fb23623fb1f536a23d9749a519fc2de79 Fix RadixEncoder<int> operator() signature for radix sort (#4921)
- [ ] 9d5b5c5508cf4ba7348914ed7541d6c7321fe415 Improve RFA PR 2234 (#4888)
- [ ] 450622a74f6ca55e6884a02fb179cf9583e06c1a Drop dead code in Thrust reduce (#4969)
- [ ] 5e610b0ce8f3175ca01b4211917f9197ed4d97a5 replace `_LIBCUDACXX_HIDE_FROM_ABI` with `_CCCL_API inline` (#4936)
- [ ] a0ac7eadf8470ffdd68a4ea858c389e229230252 Port `thrust::tabulate_output_iterator` to `cuda` (#4879)
- [ ] 3314a70aa01e477c8f645da810e9810002f80483 Refactor thrust::sequential::sort (#4925)
- [ ] 3935bb637c036bac4c817d4a7b369e12aa020e9a Add load-bearing semicolon for MSVC in openMP sort (#5024)
- [ ] 6b7ab3b6d8b7445c436b0fe066ec0d9c594041a1 Refactor around `thrust::vector` (#5044)
- [ ] a6282e9cf4f8d2a20a3328b9473edf7b7cdbe9cd Handle upcoming vector type change (#5036)
- [ ] cd1b8dcbb1c7d679a625ef0def9be54448d81d1d Apply `remove_cvref` in `thrust::is_contiguous_iterator` and refactor all uses (#5050)
- [ ] 71da83a25c2451bf8db7db2ac1e45b63c0742d80 `thrust::cuda::pinned_memory_resource` should dispatch to the host system (#5045)
- [ ] 6ccee554169d39f19105d6ef65f68f4e9e220a5e Add vectorized `cub::DeviceTransform` algorithm (#4815)
- [ ] c16a2e1623e632c986d20ba5715d00f098b77b45 Port `thrust::transform_output_iterator` to `cuda` (#5051)
- [ ] 0788bf61e2b586703f8c839f75c73f64a0fbfdbf Fix `thrust::make_discard_iterator` (#5093)
- [ ] 0cf03e41303bb49d3822ec0d2aa4fbc9f9acb489 Port `thrust::permutation_iterator` (#4835)
- [ ] 11486dea25276ac29b3bee5eeaf8c1d03a186f0d Refactor generic sequence (#5156)
- [ ] e703c48ced0b5ba23daf4bb8d87eb328e483ed72 Update doc errors set_operations.h (#5177)
- [ ] 9e4cd939db12d7d7ae8027c0df5bf2f89a3c8851 [pre-commit.ci] pre-commit autoupdate (#4365)
- [ ] 74dac60f7db268d2a221e42e218c283d9ef3ce34 Refactor thrust cuda replace (#5180)
- [ ] 77eb9a6414c62799914c8983455a2cb2897309a1 Implement transform[_if]_n in the generic system (#5182)
- [ ] 961efabbc7a60b892d93e9c91af5fdddc1f1adaa Replace cuda version checks with `_CCCL_CTK_XXX()` macro (#5204)
- [ ] 9b34fe9249cbb1b308d4d2b9d99faacc4cf86983 Implement `thrust::swap_ranges` via `transform` in CUDA system (#5243)
- [ ] 33900a1fb128269148bfa78edd8d53d4ca3557b4 Add a macro to disable PDL (#5316) (#5329)
- [ ] 7296b79e16f148cd2c8c3114971d9a9ecbb59372 [Backport branch/3.1.x] PR #5396 and #5566 (#5611)
- [ ] 7edde877f8f4b15c55568beb973a7654e48ba06b Fix `thrust::malloc` for `void` (#5698) (#5718)
- [ ] 62b14d19b77cfc5b549c97177ea80030e25ba5ec [BACKPORT 3.1] Use forward declarations of extended floating point types instead of including the headers (#5846) (#5978)
- [ ] 5fefb30bc8e597a288352372e9786496bb2f733f [Backport 3.1]  Backport iterator fixes (#5977)
- [ ] ecce794c363cf65c97e59faf5caf376abf8da340 [Backport to 3.1] Fix dereferencing nullptr in thrust::device_reference (#4226) (#6002)
- [ ] 30ae2aca9d54c51262f06e5ebce598562ceb884b [Backport 3.1]: [CUB] Replace several direct uses of `__clz` (#6099) (#6202)
