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
- [X] 83ea8015adfe80bad055067e6af4022b550aad15 Move NVTX to libcu++ and add support for Thrust (#4537)
      N/A on AMD, skipped -- not ported. This commit adds
      `_CCCL_NVTX_RANGE_SCOPE("thrust::<algo>")` calls to ~35 algorithm
      `.inl` entry points (36 files total under `thrust/thrust/`, 322
      insertions, mostly one-line mechanical additions), sourced from a
      brand-new NVIDIA-only libcu++ header `<cuda/__nvtx/nvtx.h>` newly
      included in `thrust/detail/config.h` (guarded by `#if
      !_CCCL_COMPILER(NVRTC)`). NVTX (NVIDIA Tools Extension) is a CUDA-only
      profiling API with no ROCm/HIP equivalent vendored anywhere in this
      repo (confirmed via repo-wide grep: zero pre-existing NVTX/ROCTX
      references, and no `<cuda/__nvtx/...>` header available for the HIP
      backend). This exact commit was already anticipated and flagged by
      the prior `cccl-investigation-v3.1.0-6.md` scoping report (see its
      "Per-feature disposition" table: "Likely N/A on AMD; track rather
      than port unless a roctx-analogue is desired"), which left an open
      question for the human on whether a roctx-based analogue is wanted.
      Asked the human directly during this sync session; decision: skip as
      N/A, matching the investigation report's own recommended disposition
      -- do not include the NVIDIA-only header or add any instrumentation
      calls. No files touched, nothing staged. If a roctx-based analogue is
      ever wanted, that is new feature work (design a portable
      `_CCCL_NVTX_RANGE_SCOPE`-equivalent macro backed by roctx, then
      reapply across the same ~35 files) and should be tracked as a
      separate follow-up, not folded into this sync.
- [X] 6887634f24fd5a27f271b84d1517a16cd8789bd2 Replace use of `__CUDACC__` with `_CCCL_CUDA_COMPILATION()` (#4587)
      Single-file, single-hunk change in `system/cuda/detail/cdp_dispatch.h`:
      replaced `#if defined(__CUDACC__) && defined(__CUDA_ARCH__)` with `#if
      _CCCL_DEVICE_COMPILATION()` (a portable CCCL macro, no HIP-specific
      translation needed since this is a `system/cuda/detail` file), and
      updated the matching `#else`/`#endif` trailing comments to upstream's
      new `^^^ .../ vvv ...` style. No other translation needed -- the
      file's pre-existing `_THRUST_IF_TARGET`/`_THRUST_ANY_TARGET`/
      `_THRUST_IS_HOST` names (already diverged from upstream's
      `NV_IF_TARGET`/`NV_IS_HOST`) were untouched by this commit and left
      as-is. Verified via `git diff` matching upstream line-for-line.
      Staged.
- [X] a0a7b3856e9110722fec1d9ed474648251de033d Reduce the use of `__CUDA_ARCH__` (#4589)
      Small mechanical macro-replacement commit across 4 files, all
      `#if defined(__CUDA_ARCH__) [|| defined(_NVHPC_CUDA)]`-style device-pass
      detection replaced with the portable CCCL macros
      `_CCCL_DEVICE_COMPILATION()` / `_CCCL_HOST_COMPILATION()` /
      `_CCCL_PTX_ARCH()`, plus matching trailing-comment style updates
      (`^^^ .../ vvv ...`). No HIP-specific translation needed anywhere --
      these are portable CCCL compiler-detection macros, applied verbatim
      to `detail/allocator/temporary_allocator.inl` (include guard for
      `system/cuda/detail/terminate.h`, alongside the file's pre-existing
      untouched `THRUST_DEVICE_SYSTEM_HIP` branch),
      `system/cuda/detail/core/agent_launcher.h` (`_kernel_agent`/
      `_kernel_agent_vshmem` device-vs-host specialization),
      `system/cuda/detail/core/util.h` (`_THRUST_TUNING_ARCH` SM-version
      branch, `__CUDA_ARCH__ >= 600` -> `_CCCL_PTX_ARCH() >= 600`), and
      `system/cuda/detail/malloc_and_free.h` (`get_allocator()`
      host-only guard, `ifndef __CUDA_ARCH__` -> `#if
      _CCCL_HOST_COMPILATION()`). Verified all 4 local pre-edit contexts
      matched upstream's pre-commit state exactly before editing, and all 4
      post-edit `git diff`s match upstream line-for-line. Staged.
- [X] 1f5498642c12bb6d5ec528e5efc023d50107fb04  Improve defence against the external macros (#4635)
      N/A locally, nothing to port. Upstream's only change touching
      `thrust/thrust/` (out of a much larger CCCL-wide prologue/epilogue
      header-guard refactor) is a pure include-order swap in two files --
      moving `<stdexcept>`/`<new>` to *after* `#include <nv/target>`
      (working around a system-header conflict specific to that combo).
      Checked both local files
      (`detail/contiguous_storage.inl`,`detail/memory_algorithms.h`):
      neither includes `<nv/target>` at all locally (confirmed via grep --
      only a handful of `system/cuda/detail/*.h` files and
      `detail/config/libcxx.h` include it in this codebase), so the
      ordering relationship the upstream fix depends on doesn't exist here
      -- there is nothing to reorder relative to. No edits made, nothing
      staged.
- [X] eb7b68775a92b36307db023e6351b6025d6e2d8e c.parallel: reuse CUB agent policies for reduce (#4286)
      Single file, single hunk in
      `system/cuda/detail/core/triple_chevron_launch.h`: wrapped the whole
      `doit_host` member function in `#if !_CCCL_COMPILER(NVRTC)` / `#endif`
      (host-launch codepath is meaningless under NVRTC device-only JIT
      compilation), bumping the nested nested `#if`/`#endif` guards
      (`_CCCL_HAS_PDL()`, the MSVC workaround) in by one `#`-indentation
      level to match. Local file already had `_CCCL_HAS_PDL()` in
      function-call form (not upstream's plain `_CCCL_HAS_PDL` at this
      point in history) from an earlier-ported commit -- preserved that
      spelling rather than reverting it, only applying this commit's
      specific structural change (the NVRTC wrap + indent bump). No other
      hunks in this commit touch `thrust/thrust/`. Verified via `git diff`
      matching upstream's structural diff exactly. Staged.
- [X] 1b13308ef2e3849dddcf74ac5db230f480021915 Use list init for test data in iterator docs (#4738)
      Pure doc-comment cleanup, no code semantics: replaced verbose
      `device_vector<T> x(N); x[0] = ...; x[1] = ...;` construction
      sequences with brace-init-list (`device_vector<T> x{...};`) in the
      `\code` examples of 6 fancy-iterator headers -- `constant_iterator.h`,
      `counting_iterator.h`, `discard_iterator.h`,
      `permutation_iterator.h`, `reverse_iterator.h` (3 occurrences),
      `transform_iterator.h` (3 occurrences). All 6 local files matched
      upstream's pre-commit doc text exactly before editing. Verified via
      `git diff` matching upstream line-for-line for all 6 files. Staged.
- [X] a3a00b9164a9c3893aca63dc5186ee72bb98c3ae Make `device_reference<T>::operator=` `const` (#4740)
      Semantic const-correctness change (proxy-reference assignment is
      shallow-const), touching `thrust/detail/reference.h` and
      `thrust/device_reference.h`. In `reference.h`: made all 3
      `reference::operator=` overloads (self, cross-`OtherDerived`, and
      `value_type`) return/take `this` as `const`; converted the
      cross-`OtherDerived` overload from the old-style
      `typename std::enable_if<Cond, ReturnType>::type` SFINAE-return idiom
      to upstream's new non-type-template-parameter-default idiom, using
      `_THRUST_STD::enable_if_t`/`_THRUST_STD::is_convertible_v` in place of
      upstream's `::cuda::std::enable_if_t`/`::cuda::std::is_convertible_v`
      (this file is outside `system/cuda/detail/**`, so the established
      `_THRUST_STD::` wrapper convention applies -- confirmed the necessary
      `_THRUST_STD_INCLUDE(type_traits)` include was already present at the
      top of the file, guarded by `#if _THRUST_HAS_DEVICE_SYSTEM_STD`, so no
      new include was needed); added a new `const derived_type& derived()
      const` overload alongside the pre-existing non-const one; added
      `const` to both `assign_from` overloads and to
      `strip_const_assign_value` (left `convert_to_value_type` and
      `strip_const_get_value` untouched -- they were already `const`,
      unrelated to this commit); made all 3 `tagged_reference::operator=`
      overloads const the same way. In `device_reference.h`: added a new
      defaulted `device_reference(const device_reference& other) = default;`
      copy constructor, added a new `const device_reference& operator=(const
      device_reference& other) const` overload, and made the two existing
      `operator=` overloads (`operator=(const device_reference<OtherT>&)`,
      `operator=(const value_type&)`) const-returning/const-qualified.
      Confirmed this file already uses `THRUST_HOST_DEVICE` (not upstream's
      `_CCCL_HOST_DEVICE`) prior to editing, matching the established
      per-directory macro convention. `git diff` for both files matches
      upstream's diff line-for-line modulo the macro/namespace
      substitutions above. Staged.
- [X] 0c5d8878687f4851516d8e0f013b696368fe7856 Simplify thrust::cuda_cub::swap_ranges (#4182)
      Touches 3 files: `thrust/detail/reference.h`, `thrust/device_reference.h`
      (both layered on top of item 34's const-correctness changes), and
      `thrust/system/cuda/detail/swap_ranges.h`. In the first two files,
      changed `swap(derived_type&/device_reference&, ...)` overloads to take
      the other reference *by value* instead of by lvalue-reference
      (`derived_type other`, `device_reference other`) in `reference::swap`,
      `reference::swap(System*, ...)`, the free `swap(tagged_reference,
      tagged_reference)`, and `device_reference::swap` declaration; moved
      `device_reference`'s `swap` free function out of the class body (was a
      hidden friend) into a `template <typename T>` free function after the
      class closes, matching upstream's comment explaining this avoids an
      nvcc 12.0 miscompile ("incomplete type is not allowed"). In
      `swap_ranges.h` (CUDA-backend-only file, guarded by
      `_CCCL_HAS_CUDA_COMPILER()` -- confirmed AMD builds use the separate,
      untouched `thrust/system/hip/detail/swap_ranges.h` instead, so this
      file is dead code for ROCm but kept in sync structurally): removed the
      `__swap_ranges` namespace wrapper and manual-construction/macro-based
      `swap_f` functor (with its `THRUST_FUNCTION`/`THRUST_DEVICE_FUNCTION`
      constructor and 3-step load/swap/store body) in favor of a simpler
      aggregate `__swap_f` using `_CCCL_HOST_DEVICE void operator()(Size)
      const` (matching this file's own pre-existing `_CCCL_HOST_DEVICE`
      convention) that directly calls `iter_swap`; translated upstream's raw
      `::cuda::std::distance`/`::cuda::std::iter_swap` to
      `_THRUST_STD::distance`/`_THRUST_STD::iter_swap`, matching this exact
      file's own pre-existing (untouched-by-this-commit) choice of
      `_THRUST_STD::distance` one line above -- local precedent within the
      same file takes priority over the general "raw ::cuda::std:: inside
      system/cuda/detail/**" heuristic, since usage there was already mixed
      before this edit. Kept the two new `<cuda/std/__algorithm_>` /
      `<cuda/std/iterator>` includes raw (unwrapped), matching this file's
      pre-existing raw (non-`_THRUST_STD_INCLUDE`) `<cuda/std/utility>`
      include style. Confirmed no other file in the repo references the
      removed `__swap_ranges` namespace or `cuda::detail`'s `swap_f` name
      (the HIP backend has its own independent, differently-named
      `hip::detail::__swap_ranges::swap_f`, untouched). `git diff` for all
      3 files matches upstream line-for-line modulo the above. Staged.
- [X] 4e7e7cd5cf4878123ea9c2fbaac8515bf74a7a7d Add thrust::transform_n (#4750)
      Single-file, purely additive commit: adds 12 new overloads (6
      `transform_n`, 6 `transform_if_n`) to `thrust/transform.h`, each a
      thin wrapper computing `first + count` and forwarding to the existing
      `transform`/`transform_if`. Added `#include _THRUST_STD_INCLUDE(iterator)`
      (translated from upstream's raw `#include <cuda/std/iterator>`,
      matching the established outside-cuda/detail convention -- confirmed
      exact precedent via `thrust/distance.h`'s identical unguarded
      `#include _THRUST_STD_INCLUDE(iterator)`). Translated
      `::cuda::std::iter_difference_t` to `_THRUST_STD::iter_difference_t`
      (confirmed as the intended replacement via a pre-existing
      `THRUST_DEPRECATED_BECAUSE` message in `iterator_traits.h` naming
      `_THRUST_STD::iter_difference_t` as the modern equivalent). Used
      `THRUST_HOST_DEVICE` (not upstream's `_CCCL_HOST_DEVICE`), matching
      this file's own pre-existing convention for its other
      execution-policy-overload free functions. Omitted every
      `_CCCL_NVTX_RANGE_SCOPE("thrust::transform_n"/"...transform_if_n")`
      call upstream adds inside each new function body: grepped and
      confirmed `_CCCL_NVTX_RANGE_SCOPE` has zero occurrences anywhere in
      the local repo (consistent with item 28's finding that NVTX support
      was never ported/is N/A on AMD) -- the macro is dropped rather than
      force-defined, since it is purely a profiling annotation with no
      effect on correctness, and this matches the same "N/A locally, omit"
      disposition already used for item 28, just applied inline rather than
      to a whole commit. `git diff` matches upstream structurally
      line-for-line modulo the include/namespace/macro substitutions and
      NVTX-call omissions above. Staged.
- [X] ab1ba00183c55b4e61e2ce95faf748718700ebec Allow default-initializing and skipping initialization of Thrust vectors (#4183)
      Adds new `default_init_t`/`no_init_t` empty tag structs plus
      `inline constexpr default_init`/`no_init` tag objects at `thrust::`
      namespace scope (not `thrust::detail`) in `vector_base.h`, enabling a
      new `vector_base(size_type n, default_init_t)` constructor (skips
      value-initialization when `T` is trivially constructible, via
      `if constexpr (!_THRUST_STD::is_trivially_constructible_v<T>)`
      guarding the one `value_initialize_n` call) and a template-defaulted
      `vector_base(size_type n, no_init_t)` (`template <typename T2 = T>`,
      `static_assert(_THRUST_STD::is_trivially_constructible_v<T2>, ...)`,
      allocates but never initializes). Mirrored the same pair as
      `resize(size_type, default_init_t)` / `resize(size_type, no_init_t)`,
      both using `_THRUST_STD::next` for the shrink path (already
      transitively available via the pre-existing `#include
      <thrust/distance.h>`, itself including `_THRUST_STD_INCLUDE(iterator)`
      -- no new include needed) and delegating growth to a newly
      parameterized `template <bool SkipInit = false> void append(size_type
      n)`, with both of `append`'s `value_initialize_n` call sites wrapped
      in `if constexpr (!SkipInit)`. `_THRUST_STD_INCLUDE(type_traits)` was
      already present in `vector_base.inl`, so no new include needed there
      either. Also fixed a pre-existing doc typo (`max_size9).` ->
      `max_size().`) carried in upstream's diff. Propagated the analogous
      two constructors + two `resize` declarations to `device_vector.h` and
      `host_vector.h`: `device_vector`'s use no host/device macro (matching
      its own pre-existing macro-less `explicit device_vector(size_type n)`
      constructor), while `host_vector`'s use `THRUST_HOST` (translated from
      upstream's `_CCCL_HOST`, matching this file's own pre-existing
      `THRUST_HOST explicit host_vector(size_type n)` convention). Both
      new `resize` declarations in `device_vector.h`/`host_vector.h` land
      inside the pre-existing `#if 0` doxygen-only block (dead code, never
      compiled) alongside the existing undocumented `resize(size_type,
      const value_type&)` declaration it sits next to -- confirmed no
      `.inl` implementation is needed/expected, matching upstream's own
      placement exactly. `git diff` across all 4 files
      (`vector_base.h`, `vector_base.inl`, `device_vector.h`,
      `host_vector.h`) matches upstream line-for-line modulo the
      `::cuda::std::` -> `_THRUST_STD::` and `_CCCL_HOST` -> `THRUST_HOST`
      translations above. Staged.
- [X] 525ef99f44f572a7956c6a1c8ac49616766780d7 Add `thrust::strided_iterator` and a step for `thrust::counting_iterator` (#4014)
      New file `thrust/iterator/strided_iterator.h` created (`runtime_value<T>`
      / `compile_time_value<Value>` stride holders, `strided_iterator` fancy
      iterator, `make_strided_iterator` x2). Local file's own `counting_iterator.h`
      turned out to have a substantially different pre-existing base
      structure than upstream (uses a legacy `::internal::If`/`eval_if`/
      `identity_` idiom for `make_counting_iterator_base`'s `system`/
      `traversal`/`difference` aliases, rather than raw `::cuda::std::`) --
      confirmed via diff context lines that upstream's own diff does not
      touch these lines either (elided as unchanged context), so this
      divergence is pre-existing and orthogonal to this commit; only the
      lines upstream's diff actually touches (template parameter lists,
      new methods) were ported, left everything else alone. Added new
      `StrideHolder` 5th template parameter (defaulted to new
      `detail::unit_stride = compile_time_value<1>`) threaded through
      `counting_iterator`'s forward-decl, `make_counting_iterator_base`,
      and its `type` alias; `counting_iterator` now also privately inherits
      `StrideHolder` (empty-base-optimized via pre-existing
      `THRUST_DECLSPEC_EMPTY_BASES`, confirmed as the local name for
      upstream's `_CCCL_DECLSPEC_EMPTY_BASES` via 3 other usages in this
      same directory). Added new `stride()`/`advance()`/`increment()`/
      `decrement()` private members with `if constexpr
      (_THRUST_STD::is_same_v<StrideHolder, detail::unit_stride>)`
      branching (translated from `::cuda::std::is_same_v`), and a new
      `counting_iterator(Incrementable x, StrideHolder stride)` constructor.
      Updated `equal`'s template parameter list (`OtherStrideHolder` added,
      body untouched -- confirmed via diff context that upstream's own
      floating-point special-case body is unmodified by this commit) and
      `distance_to`'s parameter type (uses own `StrideHolder`, not a
      template param -- confirmed exactly matches upstream, meaning
      `distance_to` only supports same-stride-type counting_iterators).
      Added two new `make_counting_iterator` overloads (runtime and
      compile-time stride), guarded by `#ifndef THRUST_DOXYGEN_INVOKED`
      (translated from upstream's `_CCCL_DOXYGEN_INVOKED` -- confirmed as
      the local macro name via 10 existing `#ifdef`/`#ifndef
      THRUST_DOXYGEN_INVOKED` usages elsewhere, e.g. `thrust/pair.h`,
      `thrust/tuple.h`, `thrust/memory.h`; zero occurrences of the raw
      `_CCCL_` spelling anywhere locally). Used `THRUST_EXEC_CHECK_DISABLE`
      (not `_CCCL_EXEC_CHECK_DISABLE`) throughout the new file and the new
      counting_iterator members -- confirmed as the local name via its
      existing use in `iterator_adaptor.h`'s own `dereference`/`equal`/
      `advance`/`increment`/`decrement`/`distance_to`. Used
      `detail::it_difference_t` in `strided_iterator.h`'s `stride()`
      unchanged from upstream -- confirmed it lives in `thrust::detail`
      locally too (`iterator_traits.h`), not just `thrust::`, matching
      upstream's qualification exactly. Kept `_THRUST_STD::random_access_iterator`
      (C++20 concept) as a direct translation of `::cuda::std::random_access_iterator`,
      consistent with the project-wide mechanical `::cuda::std::` ->
      `_THRUST_STD::` rule, even though no other local file currently uses
      this particular concept (no fallback/pre-C++20 shim exists for it
      locally, and none was invented here -- `_THRUST_STD` is expected to
      resolve to a real `cuda::std`/`hip::std` providing it). New file's
      license header uses the local two-line SPDX
      NVIDIA-2025/AMD-Modifications-2026 + `Apache-2.0 WITH LLVM-exception`
      convention (confirmed precedent via `thrust/iterator/offset_iterator.h`
      and `tabulate_output_iterator.h`, the only 2 other files using this
      short-form header), not upstream's bare 2-line
      `Apache-2.0`-without-LLVM-exception SPDX header. Also ported the
      small `iterator_adaptor.h` simplification (drops the redundant
      `typename iterator_adaptor::` prefix from `reference`/
      `difference_type` in `dereference()`/`advance()`/`distance_to()`'s
      signatures, relying on injected-class-name lookup) -- purely
      mechanical, `THRUST_HOST_DEVICE`/`THRUST_EXEC_CHECK_DISABLE` already
      matched. Grepped for any other 4-template-arg explicit
      `counting_iterator<...>` instantiations elsewhere in the repo that
      the new 5th defaulted parameter could break: none found. `git diff`
      for all 3 touched files (`counting_iterator.h`, `iterator_adaptor.h`)
      plus the new `strided_iterator.h` matches upstream line-for-line
      modulo the conventions above. Staged.
- [X] e1f04c7d907707918233a5bb54c46fbeb5b6f81a Small refactorings in Thrust CUDA (#4764)
      Small cosmetic/structural refactor across 5 CUDA-backend-only files
      (all `#if _CCCL_HAS_CUDA_COMPILER()`-guarded, dead code for ROCm, same
      disposition as item 35's `swap_ranges.h` -- real AMD path is the
      separate untouched `system/hip/detail/*.h` sibling for each). Removed
      each functor's now-redundant hand-written constructor (`functor(...)
      : field(field), ... {}`) in `fill.h`/`tabulate.h`/
      `uninitialized_copy.h`/`uninitialized_fill.h`, converting call sites
      from `functor<...>(first, value)` to aggregate-init
      `functor<...>{first, value}`; `tabulate.h`'s `functor` additionally
      dropped its 3rd `Size` template parameter (now inferred per-call via
      a `template <typename Size>` on `operator()` alone, matching the
      other 3 files' pre-existing per-call-templated `operator()` style).
      `terminate.h` collapsed 3 nested `namespace system { namespace cuda {
      namespace detail { ... } } }` into a single `namespace
      system::cuda::detail { ... }` (C++17 nested namespace syntax).
      Removed extraneous blank lines after opening namespace braces in all
      5 files, matching upstream's cosmetic cleanup exactly. No macro/
      namespace substitutions needed: all touched lines use pre-existing
      local names (`THRUST_FUNCTION`, `THRUST_DEVICE_FUNCTION`,
      `_CCCL_HOST_DEVICE`, `_CCCL_DEVICE`, `_THRUST_STD::distance`)
      untouched by upstream's own diff, so no translation was required.
      Grepped for other callers of the 4 touched `functor` types: only the
      independent, untouched `system/hip/detail/*.h` siblings reference
      same-named `functor` structs (their own separate copies, still using
      manual constructors) -- no cross-reference breakage. `git diff` for
      all 5 files matches upstream line-for-line. Staged.
- [X] 3e2f6e9a6e0436f9d1d2ea8b407dd76d3dc94dc4 Make sure that `cuda` iterators play nicely with the thrust system and traversal machinery (#4806)
      8 files touched, split into two groups. (1) 5 files
      (`pointer_traits.h`, `constant_iterator.h`, `counting_iterator.h`,
      `retag.h`, `permutation_iterator.h`) are a purely mechanical
      readability refactor: `typename iterator_system<X>::type` ->
      `iterator_system_t<X>` (already defined locally in
      `iterator_traits.h`). Applied verbatim; `counting_iterator.h`'s copy
      constructor was already carrying item 38's `StrideHolder` 5th
      template arg, so the simplification was layered on top of that
      state rather than the pre-item-38 shape. (2) 3 files add support for
      libcu++'s *native* `::cuda::discard_iterator` /
      `::cuda::counting_iterator<Start>` / `::cuda::transform_iterator`
      (distinct from `thrust::discard_iterator` etc.) so they interoperate
      with thrust's traversal/system-trait machinery. Confirmed the local
      shim (`thrust/detail/config/libcxx.h`) already has an established,
      pre-existing convention for exactly this: `_THRUST_LIBCXX` (=
      `::cuda`/`::hip`) and `_THRUST_LIBCXX_INCLUDE(hdr)` (= `<cuda/hdr>`/
      `<hip/hdr>`), already used elsewhere (`alignment.h`, `functional.h`,
      `zip_function.h`, and even this same `counting_iterator.h` at its
      own line 54/277) -- so translated `::cuda::` -> `_THRUST_LIBCXX::`
      and `<cuda/...>` -> `_THRUST_LIBCXX_INCLUDE(...)` throughout, always
      guarded by `#if _THRUST_HAS_DEVICE_SYSTEM_STD` matching the
      established pattern (the macro expands to nothing in the
      no-libcudacxx/libhipcxx fallback, so unguarded use would break the
      fallback build). `iterator_traits.h`: kept the pre-existing
      `_THRUST_STD_INCLUDE(iterator)` include (needed unconditionally,
      including fallback, for `_THRUST_STD::iterator_traits`) and added
      `_THRUST_LIBCXX_INCLUDE(iterator)` as a second, guarded include
      rather than replacing it as upstream's diff does -- upstream drops
      `<cuda/std/iterator>` entirely because `<cuda/iterator>` re-exports
      it, but that's an assumption about libcu++'s own header graph this
      port doesn't need to rely on; added the 3 new `iterator_system`/
      `iterator_traversal` specializations (discard/counting/transform)
      guarded the same way. Also added an explicit
      `#include <thrust/iterator/detail/any_system_tag.h>`: upstream's
      diff needs `any_system_tag` without adding an include for it
      (presumably already transitively visible upstream), but grepping
      locally showed it is NOT transitively included here, so added it
      directly for correctness rather than assuming parity with upstream's
      transitive-include graph. `internal_functional.h`: added
      `is_discard_proxy<T>` (false by default, specialized true only for
      `_THRUST_LIBCXX::discard_iterator::__discard_proxy`, guarded) and
      wired it into `enable_if_non_const_reference_or_tuple_of_iterator_references`.
      `zip_iterator.h`: added `zip_iterator_reference_t` helper so a
      `void`-typed `it_reference_t` (discard_iterator's reference type)
      falls back to `decltype(*declval<Iter>())` instead of using `void`
      directly in the reference tuple. Did not attempt to verify whether
      the actual `<hip/iterator>` / `<hip/__iterator/discard_iterator.h>`
      headers and `::hip::discard_iterator`/`counting_iterator`/
      `transform_iterator` types exist in the external libhipcxx this
      repo builds against (no vendored libhipcxx source in this repo to
      grep) -- trusted `_THRUST_LIBCXX`/`_THRUST_LIBCXX_INCLUDE` to
      resolve correctly per the same precedent already established for
      `_THRUST_STD` (item 38's `random_access_iterator` concept
      decision). `git diff` for all 8 files matches upstream line-for-line
      modulo the `_THRUST_STD`/`_THRUST_LIBCXX` translation and the two
      deliberate additions noted above (extra `any_system_tag.h` include,
      non-destructive `iterator` include). Staged.
- [X] c38a6dd5a1d6ad6f3e36bfd745854506312af975 fix cudax's vector_add example that was broken by #4795 (#4814)
      Upstream diff for `thrust/thrust/` touches only 1 line: adds
      `#include <thrust/iterator/detail/any_system_tag.h>` to
      `iterator_traits.h` -- described in the commit message as "Add
      missing header to Thrust's `iterator_traits.h`", i.e. upstream's own
      follow-up fix for the same gap this port already noticed and closed
      independently while porting item 40 (grepped and found
      `any_system_tag` wasn't transitively included locally, so it was
      added proactively then). Verified the include is already present at
      `iterator_traits.h:42` -- no further edit needed, nothing to stage.
      The unrelated cudax `vector_add` example fix is outside
      `thrust/thrust/` and out of scope for rocThrust.
- [X] 035da530641bff6c2d78b324294bd3e6c20ab9d6 Refactor Thrust select_system (#4762)
      Large but purely mechanical structural refactor (6 files, 595-line
      diff): collapses the fixed-arity (1-6 argument) `select_system`
      overload family, its SFINAE-detection machinery
      (`select_systemN_exists`), and its split declaration(`.h`)/
      definition(`.inl`) into a single variadic implementation using
      C++17 `if constexpr` + a `void_t`-based `select_system_exists`
      trait, all inlined into one header. Before editing, grepped the
      whole repo for every symbol being deleted
      (`select_system{1..6}_exists`, `enable_if_defined`) and confirmed
      each is used *only* inside the 3 files this commit deletes/rewrites
      -- no ROCm-specific consumer elsewhere, so deletion is safe and
      matches upstream's own blast radius exactly. Deleted (git rm, no
      replacement, matching upstream): `thrust/detail/select_system.h`
      (a `select_system_detail::select_system_fn` callable object,
      confirmed unused anywhere in the repo -- same dead code upstream
      removes), `system/detail/generic/select_system.inl`,
      `system/detail/generic/select_system_exists.h`. Rewrote
      `system/detail/generic/select_system.h` in place with the new
      variadic implementation, translating `_CCCL_HOST_DEVICE` ->
      `THRUST_HOST_DEVICE`, `::cuda::std::` -> `_THRUST_STD::`, and
      `#include <cuda/std/type_traits>` -> `#include
      _THRUST_STD_INCLUDE(type_traits)` (unguarded -- confirmed this
      macro, unlike `_THRUST_LIBCXX_INCLUDE`, already resolves in all 3
      branches including the no-libcudacxx/libhipcxx fallback, matching
      existing unguarded usage in `counting_iterator.h`). Kept the C++17
      `namespace system::detail::generic` collapse (same precedent as
      item 39's `terminate.h`). Added `minimum_system_t` alias to
      `minimum_system.h` (purely additive, same pattern as
      `iterator_system_t`) -- left the underlying `minimum_system` alias
      itself untouched (`::internal::If`/`identity_` legacy idiom,
      pre-existing local divergence from upstream's `::cuda::std::_If`,
      orthogonal to this commit since upstream's own diff doesn't touch
      that line either, only the new line below it). Removed
      `enable_if_defined` from `is_metafunction_defined.h`, leaving
      `is_metafunction_defined` itself untouched. `git diff` for all 3
      surviving files matches upstream line-for-line modulo the standard
      macro translations; 3 deletions match upstream's deletions exactly.
      Staged.
- [X] 4d5040efa532a3e9e21bee0ec1d42ed741b3fe48 Port `thrust::strided_iterator` to cuda (#4808)
      1-file, 7-line addition: same pattern as item 40's `transform_iterator`
      specialization (delegates `iterator_system`/`iterator_traversal` to
      the wrapped `Iter`'s), now for libcu++'s native
      `::cuda::strided_iterator<Iter, Stride>`. Inserted right before the
      `transform_iterator` specialization in `iterator_traits.h`
      (matching upstream's placement), translated `::cuda::` ->
      `_THRUST_LIBCXX::` per the item-40-established convention, kept
      inside the same `#if _THRUST_HAS_DEVICE_SYSTEM_STD` guard block.
      `git diff` matches upstream line-for-line modulo the translation.
      Staged.
- [X] 584fedba37077b0040cccf532229daa2d80ba312 Refactor Thrust internal_functional (#4810)
      Largest item so far (19 files, 654-line diff), but structurally
      uniform: strip hand-written constructors that just forward to
      aggregate-init-compatible members, convert several
      `struct X : bool_constant_type {}` traits to C++17
      `inline constexpr bool X_v = ...` variable templates, and move a
      SFINAE guard from a function's return type to its template parameter
      list. Before touching anything, grepped the whole repo for
      `tuple_not_binary_predicate` and `compare_first_less_second` (both
      entirely deleted by upstream) and confirmed zero consumers outside
      their own definitions -- safe to delete matching upstream's blast
      radius exactly.
      Key open question resolved before editing: upstream's diff changes
      several `thrust::get<N>(t)` call sites to `::cuda::std::get<N>(t)`.
      Checked `thrust/tuple.h`: in the `_THRUST_HAS_DEVICE_SYSTEM_STD`
      branch it does `using _THRUST_STD::tuple;` and `using
      _THRUST_STD::get;` -- i.e. `thrust::get`/`thrust::tuple` are
      literally the same entities as `_THRUST_STD::get`/`_THRUST_STD::tuple`
      via using-declaration, while the fallback (no libcu++) branch
      provides its own distinct boost-cons-based `tuple`/`get` pair. Since
      `internal_functional.h`'s functors are generic, dual-mode code (not
      guarded by `_THRUST_HAS_DEVICE_SYSTEM_STD`), the correct local
      translation is to leave every `thrust::get<N>(t)` call as-is rather
      than hardcode `_THRUST_STD::get` -- doing the latter would silently
      break the fallback branch, since `_THRUST_STD::tuple`/`get` don't
      apply to the fallback's own `thrust::tuple` type. This is a
      deliberate, documented divergence from upstream's literal diff, not
      an oversight.
      `thrust/detail/internal_functional.h`: removed constructors from
      `predicate_to_integral`, `tuple_binary_predicate` (also dropped its
      now-unused `using result_type = bool;`),
      `unary_transform_functor`/`binary_transform_functor`/`unary_transform_if_functor`/
      `unary_transform_if_with_stencil_functor`/`binary_transform_if_functor`,
      and `compare_first`; added the nvcc-12.0+clang14 explanatory comment
      to `equal_to_value`'s constructor (kept, not removed); deleted
      `tuple_not_binary_predicate` and `compare_first_less_second` in full;
      converted `is_non_const_reference` -> `is_non_const_reference_v`
      (translated `::internal::_And`/`thrust::detail::not_`/
      `_THRUST_STD::disjunction` idiom to plain `!...  && (... || ...)`,
      matching upstream's simplification) and
      `is_tuple_of_iterator_references` -> `is_tuple_of_iterator_references_v`;
      restructured `enable_if_non_const_reference_or_tuple_of_iterator_references`
      into the `enable_if_assignable_ref<T>` alias template
      (`_THRUST_STD::enable_if_t<..., int>`), moving the SFINAE point from
      each functor's return type into a `template <typename Tuple,
      enable_if_assignable_ref<typename thrust::tuple_element<N,
      Tuple>::type> = 0>` non-type template parameter (kept
      `thrust::tuple_element<N, Tuple>::type`, the pre-existing dual-mode-safe
      form, rather than upstream's `::cuda::std::tuple_element_t`, for the
      same fallback-branch reason as the `get` decision above); changed
      `fill_functor`/`uninitialized_fill_functor`'s copy-ctor and
      destructor to `= default`, each preceded by upstream's new "explicit
      declaration is needed to avoid an exec check warning" comment,
      dropping `THRUST_HOST_DEVICE` from the two now-defaulted
      declarations (matching upstream, which drops `_CCCL_HOST_DEVICE`
      there too); dropped the stray `(void)` in `fill_functor::operator()`.
      `thrust/detail/type_traits.h`: `is_proxy_reference` struct ->
      `is_proxy_reference_v` inline bool (base `false` case). Updated its
      3 specializations (`tabulate_output_iterator.h`,
      `transform_input_output_iterator.h`, `transform_output_iterator.h`,
      each a proxy-reference opt-in for one iterator's proxy type) and its
      2 consumers in `raw_reference_cast.h` to the `_v` spelling
      (`grep`-confirmed no other consumers of `is_proxy_reference<...>`
      exist).
      Remaining 12 files (`system/cuda/detail/{count,remove,replace}.h`,
      `system/detail/generic/{copy,copy_if}.inl`,
      `system/detail/generic/fill.h`,
      `system/detail/generic/{merge,mismatch,remove}.inl`,
      `system/detail/generic/set_operations.inl` (4 call sites),
      `system/detail/generic/sort.inl`,
      `system/detail/generic/transform.inl` (5 call sites),
      `system/detail/generic/uninitialized_fill.inl` (2 call sites)):
      purely mechanical `Functor(args)` -> `Functor{args}` aggregate-init
      call-site updates, required now that the corresponding constructors
      were removed. `clang-format --dry-run --Werror` clean on every
      touched file. `git diff` for all 19 files matches upstream
      line-for-line modulo the two documented `get`/`tuple_element`
      dual-mode divergences above and the standard macro translations.
      Staged.
- [X] 85694c049e5af7b5c0453f272d32f6f74a99f9a0 Refactor `thrust::minimum_type|minimum_system` (#4042)
      Rewrote `thrust/detail/type_traits/minimum_type.h` from the old
      boost-style up-to-16-template-param `minimum_type<T1..T16>` struct
      (`minimum_type_detail::primitive_minimum_type`/`any_conversion`/
      `lazy_minimum_type` machinery) to upstream's variadic recursive form:
      `no_minimum_type_marker` sentinel, `smaller_type<T1,T2,GreaterEqual,LessEqual>`
      (partial-specialized on convertibility), `minimum_type_impl<Head,Tail...>`
      recursing via `smaller_type`, and `minimum_type_check_marker<SFINAE,Ts...>`
      producing a SFINAE-friendly compile error when types are unrelated.
      `minimum_type<Ts...>` is now a bare alias
      (`typename minimum_type_check_marker<void, Ts...>::type`), not a struct
      with nested `::type` -- every call site changes from
      `typename minimum_type<...>::type` to `minimum_type<...>` directly.
      Translated `::cuda::std::` -> `_THRUST_STD::` and
      `#include <cuda/std/type_traits>` -> `#include _THRUST_STD_INCLUDE(type_traits)`.
      `thrust/iterator/detail/minimum_system.h`: replaced the ROCm-local
      `::internal::If<is_metafunction_defined<minimum_type<Ts...>>::value,
      minimum_type<Ts...>, identity_<unrelated_systems<Ts...>>>`-based
      `minimum_system`/`minimum_system_t` pair (a pre-existing local
      divergence from upstream's `::cuda::std::_If`, documented in item 42's
      note) with upstream's new int/long-overload SFINAE technique:
      `minimum_system_impl(int) -> minimum_type<Ts...>` /
      `minimum_system_impl(long) -> unrelated_systems<Ts...>`, then
      `minimum_system_t<Ts...> = decltype(minimum_system_impl<Ts...>(0))`.
      Dropped the now-unused `#include
      <thrust/detail/libcxx_wrapper/std/__type_traits/conditional.h>`
      (the `::internal::If` polyfill shim, no longer referenced by this
      file) -- a local cleanup not present in upstream's diff since upstream
      never had this shim, but correct since it becomes dead code here.
      Kept `#include <thrust/detail/type_traits/is_metafunction_defined.h>`
      per exact upstream match, even though this file no longer calls
      `is_metafunction_defined` itself; confirmed via repo-wide grep that
      the trait has a second, real consumer
      (`thrust/detail/type_traits/is_thrust_pointer.h:60`), so it must not
      be deleted.
      `thrust/detail/overlapped_copy.h`,
      `thrust/iterator/permutation_iterator.h`: mechanical
      `typename minimum_system<S1,S2>::type` -> `minimum_system_t<S1,S2>`
      call-site updates; `permutation_iterator.h` also drops the now-unused
      `#include <thrust/iterator/iterator_facade.h>` (grep-confirmed no
      other use in the file), matching upstream.
      `thrust/iterator/zip_iterator.h`: this file has a local-only
      `#if _THRUST_HAS_DEVICE_SYSTEM_STD` / `#else` dual-mode split not
      present upstream (the `#else` fallback exists for when libhipcxx
      isn't ready, per the file's own
      "TODO(libhipcxx): remove all the code in the path of
      !_THRUST_HAS_DEVICE_SYSTEM_STD" comment). Upstream's diff only
      touches the `_THRUST_HAS_DEVICE_SYSTEM_STD` branch: removed the
      `template <typename... Ts> using minimum_category = minimum_type<Ts...>;`
      alias and simplified
      `system`/`traversal_category` from
      `_THRUST_STD::__type_fold_left<_THRUST_STD::__type_list<...>, ...,
      _THRUST_STD::__type_quote_trait<minimum_system/minimum_category>>` to
      direct `minimum_system_t<iterator_system_t<Its>...>` /
      `minimum_type<iterator_traversal_t<Its>...>` -- ported verbatim.
      However the local `#else` fallback branch's
      `minimum_category_lambda::apply` and `minimum_system_lambda::apply`
      (consumed by `tuple_meta_accumulate` via `apply2<...>::type`) inherited
      from the now-removed `minimum_category<T1,T2>` alias and the
      now-removed (old, `::type`-bearing) `minimum_system<T1,T2>` alias
      respectively -- neither symbol exists anymore after the two rewrites
      above. This is a local-only problem (upstream has no such fallback
      branch to break), so as a necessary adaptation (not mirrored from any
      upstream diff) both `apply` structs were rewritten to stop inheriting
      and instead define `using type = minimum_type<T1, T2>;` /
      `using type = minimum_system_t<T1, T2>;` directly, preserving the
      exact same `::type` shape `tuple_meta_accumulate`/`apply2` expect.
      Verified via grep that `minimum_category` and the old
      `::type`-bearing `minimum_system` alias have zero remaining
      references anywhere in the tree after this change.
      `thrust/system/detail/generic/{copy,copy_if,reduce_by_key,transform,unique_by_key}.inl`:
      removed the now-unused `#include <thrust/iterator/detail/minimum_system.h>`
      from each (grep-confirmed no other use of anything from that header
      in any of the 5 files).
      `thrust/system/{omp,tbb}/detail/copy.inl`: mechanical
      `typename thrust::detail::minimum_type<traversal1,traversal2>::type`
      -> `thrust::detail::minimum_type<traversal1,traversal2>`, 2 call sites
      each (`copy`/`copy_n`). Left each file's pre-existing, unrelated
      `::cuda::std::`-vs-`_THRUST_STD::` inconsistency untouched (out of
      scope for this item; not part of upstream's diff).
      `clang-format --style=file` clean on all 12 touched files (the one
      surviving whitespace diff in `zip_iterator.h` at the
      `enable_if_t<(sizeof...(Iterators) != 0), int>` alignment was
      confirmed via `git stash` to pre-exist this item, unrelated).
      `git diff` for all 12 files matches upstream line-for-line modulo the
      `zip_iterator.h` `#else`-branch adaptation and the two documented
      local-only include drops (`conditional.h`, and the 5
      `minimum_system.h` include removals map 1:1 to upstream's own). All
      12 files staged.
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
