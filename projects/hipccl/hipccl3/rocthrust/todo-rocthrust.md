# CCCL → rocThrust Sync: v3.1.0 → v3.2.1 — TODO

We are in the middle of a manual-port sync of upstream CCCL/Thrust commits
into rocThrust. There is no git merge in progress and never will be — this
sync is a series of hand-ported commits, one local commit per item below,
audited at the end by `rocthrust-cccl-sync-finalize`. This file tracks
which upstream commits still need to be ported.

## Sync parameters

- **JIRA ticket**: none (no fixed project exists for
  rocThrust syncs)
- **Repo**: `ROCm/rocm-libraries` — rocThrust lives at `projects/hipccl/hipccl3/rocthrust/`
- **Sync base branch** (`$SYNC_BASE`): `users/matyas-streamhpc/hipccl/rocthrust_cccl31`
- **Confirmed current tag** (`$CURRENT_TAG`): `v3.1.0`
- **Target tag** (`$TO_TAG`): `v3.2.1`
- **Sync branch** (`$SYNC_BRANCH`, currently checked out): `users/amd-jmahovsky/hipccl/rocthrust_cccl32`
- **Investigation report**: `/home/AMD/jmahovsk/hipcclsync_rocthrust_3.2/rocm-libraries/projects/hipccl/hipccl3/rocthrust/cccl-investigation-v3.2.1.md`
- **Path-prefix reminder**: upstream `thrust/<path>` translates to
  `projects/hipccl/hipccl3/rocthrust/<path>` (e.g. `thrust/testing/*` →
  `projects/hipccl/hipccl3/rocthrust/testing/*`), except `thrust/benchmarks/*` →
  `projects/hipccl/hipccl3/rocthrust/benchmark/*` (singular). Translate every path before
  applying.
- **Sync decisions** (human, 2026-10-19):
  - libhipcxx will be at least 3.2. Port `cuda::` iterator interop and
    libcu++ 3.2 macros directly via `_THRUST_LIBCXX::`/`_THRUST_STD::`, with
    no libhipcxx-3.0 fallback. Raise the libhipcxx version required in
    `thrust/detail/config/libcxx.h` from 3.0 to 3.2.
  - Compatibility over performance. Performance-only rewrites go into
    `thrust/system/cuda/` only, with HIP disposition "no change:
    performance-only". Behavior and correctness fixes still go into
    `thrust/system/hip/`.

## ⚠ Ordering rule

Commits below are listed **strictly oldest-first**. Only the first unticked
(`- [ ]`) item may be worked next. Do not skip ahead, reorder, or batch
multiple commits together — a later commit may assume an earlier one is
already applied (it may reference symbols the earlier commit introduces, or
fix a bug the earlier commit itself introduced).

## State of the sync

- No `git merge` is in progress. There is no `MERGE_HEAD`, and no conflict
  markers will ever appear as part of this process.
- Ticked (`- [X]`) items below are **committed** on the sync branch: one
  local commit per item, including N/A items (empty commit), whose subject
  contains `port CCCL <sha11>`. The tick-note records it as `Local commit:
  <sha>`.
- This file itself is untracked and is never committed. Nothing is pushed
  until the human decides to. Avoid `git reset --hard` on this branch.

## Commits to port

- [X] 543b245c62c Bump main to 3.2.0. (#5286) [SCOPE: HEADER]
  - N/A — version bump only; rocThrust's THRUST_VERSION is set outside the port loop (target 300201 after finalize, as the 3.1 sync did in e4702647598).
  - HIP counterpart: none (no thrust/system/cuda/ paths touched).
  - test/ counterpart: none (no testing/*.cu paths touched).
  - Local commit: 08d5fcb66fa
- [X] 76fc4572271 Extend and refactor transform overloads in CUDA system (#5238) [SCOPE: HEADER]
  - clean port — applied to projects/hipccl/hipccl3/rocthrust/thrust/system/cuda/detail/transform.h (now identical to upstream; the local `_THRUST_STD::distance` spellings were all on lines upstream rewrote).
  - HIP counterpart: no change — `thrust::transform_n`/`transform_if_n` already reach the HIP backend through the generic implementations in `system/detail/generic/transform.h`; the new CUDA overloads only shortcut onto the CUDA fast path (performance-only, per sync decision).
  - test/ counterpart: none (no testing/*.cu paths touched).
  - Not compiled, tests not run (per human).
  - Local commit: 00e73adf2a5
- [X] 6d3ffc900fb Port `thrust::transform_input_output_iterator` to `cuda` (#5113) [SCOPE: HEADER,TEST]
  - Adaptation: libcxx transform_input_output_iterator include placed inside local `#if _THRUST_HAS_DEVICE_SYSTEM_STD` guard; iterator_traits specializations use `_THRUST_LIBCXX::` spelling, local `transform_output_iterator<Fn, Iter>` order kept.
  - HIP counterpart: none (no system/cuda paths touched).
  - test/ counterpart: catch2 "transform_input_output_iterator" case implemented as GTest in test/test_libcxx_iterators.cpp (TestLibcxxTransformInputOutputIterator) and legacy testing/cuda_iterators.cu.
  - Not compiled, tests not run (per human).
  - Local commit: 61d047f9782
- [X] 7825c40c8a6 Add a macro to disable PDL (#5316) [SCOPE: HEADER]
  - N/A: already backported (verified — rocThrust triple_chevron_launch.h already uses `_CCCL_HAS_PDL()`; 3-way merge is a no-op).
  - HIP counterpart: none (hip backend has no PDL launch path).
  - test/ counterpart: none (no test changes).
  - Not compiled, tests not run (per human).
  - Local commit: 39141fa23ca
- [X] daf84b3d7f5 Makes `thrust::unique` use `cub::DeviceSelect::Unique` (#5396) [SCOPE: HEADER,TEST]
  - N/A: already backported (verified — 3-way merge of system/cuda/detail/unique.h and testing/cuda/unique.cu is a no-op; local unique.cu additionally keeps local custom-equality tests).
  - HIP counterpart: none (cuda DeviceSelect::Unique routing is performance-only; hip backend already uses rocprim unique).
  - test/ counterpart: none needed (TestUniqueWithLargeNumberOfItems already present in testing/cuda/unique.cu; cuda-dir test, no test/test_ counterpart).
  - Not compiled, tests not run (per human).
  - Local commit: af79728c93f
- [X] 4db1ed081a3 Disable NVHPC optimization that leads to error (#5416) [SCOPE: TEST]
  - Ported NVHPC noinline workaround on FixedVector::operator+ in testing/unittest/special_types.h, keeping local THRUST_HOST_DEVICE spelling.
  - HIP counterpart: none (test-infrastructure only, NVHPC-specific).
  - test/ counterpart: none (NVHPC compiler workaround; GTest test/ is not built with NVHPC).
  - Not compiled, tests not run (per human).
  - Local commit: bd3ed37dd9c
- [X] 228c8379b80 Generate negative numbers in Thrust unit tests and fix them (#4923) [SCOPE: HEADER,TEST]
  - Clean port: uniform_real_distribution now uses `_THRUST_STD::lerp` (fixes overflow for [lowest, max] ranges); legacy testing/unittest/random.h generator refactored to `if constexpr` and generates negative values.
  - HIP counterpart: none (system-independent random header).
  - test/ counterpart: none (GTest test/ uses its own data generators in test/test_utils.hpp; legacy testing/unittest/random.h updated).
  - Not compiled, tests not run (per human).
  - Local commit: 8f47253fb79
- [X] 3b6e4a6e911 Port `thrust::zip_iterator` to namespace `cuda` (#5429) [SCOPE: HEADER,TEST]
  - Clean port: iterator_system/iterator_traversal specializations for `_THRUST_LIBCXX::zip_iterator<Iterators...>` (requires libhipcxx >= 3.2 zip_iterator/zip_function).
  - HIP counterpart: none (no system/cuda paths touched).
  - test/ counterpart: catch2 "zip_iterator" case implemented as GTest TestLibcxxZipIterator in test/test_libcxx_iterators.cpp and TestCUDAZipIterator in legacy testing/cuda_iterators.cu.
  - Not compiled, tests not run (per human).
  - Local commit: 57d74ebd10c
- [X] f22dae63e20 Drop all usages of `_CCCL_TRAIT` (#5466) [SCOPE: HEADER]
  - Clean port (after spelling normalization): transform_iterator::operator= uses `_THRUST_STD::is_*_v` instead of local `THRUST_TRAIT` (upstream `_CCCL_TRAIT`). Other local THRUST_TRAIT uses left as upstream did not touch them in thrust/.
  - HIP counterpart: none (no system/cuda paths touched).
  - test/ counterpart: none (no test changes).
  - Not compiled, tests not run (per human).
  - Local commit: 66fc2141646
- [X] c8e0441d799 Simplify thrust::device_malloc (#5477) [SCOPE: HEADER]
  - Clean port (conflict was doc-comment rewrite; local side was identical to upstream base): device_malloc folded into a single `template <typename T = void>` in device_malloc.h; detail/device_malloc.inl deleted (no other local includes).
  - HIP counterpart: none (system-independent; dispatches via thrust::malloc).
  - test/ counterpart: none (no test changes; existing device_malloc tests cover both overloads).
  - Not compiled, tests not run (per human).
  - Local commit: 15c13247bdb
- [X] 30c02fd29dd Refactor thrust::pointer (#5478) [SCOPE: HEADER]
  - Adaptation: pointer.inl folded into pointer.h per upstream; kept local `::internal::remove_cvref_t` shim (C++17 fallback) and moved the local hipcc-linkage `pointer(Element*)` constructor (THRUST_HAS_HIP_COMPILER) definition inline; `_CCCL_*` wrapper macros spelled `THRUST_*`.
  - HIP counterpart: none (system-independent pointer type; hip `pointer` derives from it).
  - test/ counterpart: none (no test changes).
  - Not compiled, tests not run (per human).
  - Local commit: 5596ad85080
- [X] 8b6974ed415 Add test to ensure we can use `cuda::std::reverse_iterator` with thrust APIs (#5486) [SCOPE: HEADER,TEST]
  - Clean port: iterator_system/iterator_traversal specializations for `_THRUST_STD::reverse_iterator<Iter>` so thrust dispatches on the wrapped iterator's system.
  - HIP counterpart: none (no system/cuda paths touched).
  - test/ counterpart: catch2 "reverse_iterator" case implemented as GTest TestLibcxxReverseIterator in test/test_libcxx_iterators.cpp and TestCUDAReverseIterator in legacy testing/cuda_iterators.cu.
  - Not compiled, tests not run (per human).
  - Local commit: 717f3a94918
- [X] f5a40bf5e45 Drop `thrust::LoadIterator`/`make_load_iterator` (#5480) [SCOPE: HEADER]
  - Clean port into system/cuda: load_iterator.h/make_load_iterator.h deleted; reduce/reduce_by_key/set_operations use cub::try_make_cache_modified_iterator (requires CCCL 3.2 CUB in the CUDA build).
  - HIP counterpart: none (no system/hip usage of LoadIterator; hip backend uses rocPRIM).
  - test/ counterpart: none (no test changes).
  - Not compiled, tests not run (per human).
  - Local commit: dab7e6173a3
- [X] 82335a826ed Extend `cub::DeviceTransform` by a predicate and port `thrust::transform_if` (#5198) [SCOPE: HEADER,BENCH]
  - Clean port into system/cuda/detail/transform.h: transform_if routed through cub::DeviceTransform with predicate (performance-only).
  - HIP counterpart: no change: performance-only (hip backend transform_if keeps its rocPRIM path; per sync decision compatibility over performance).
  - Benchmark: new nvbench benchmarks/bench/transform/if.cu rewritten for primbench as benchmark/bench/transform/if.cu (negate_if over int8..int64, 3 entropies) and registered in benchmark/CMakeLists.txt.
  - test/ counterpart: none (no test changes).
  - Not compiled, tests not run (per human).
  - Local commit: 558df7bfe99
- [X] 4ea5d23ae76 [Thrust] Make documentation behind `#if 0` visible (#5455) [SCOPE: HEADER]
  - Port (docs-only): `#if 0` documentation blocks switched to `#if THRUST_DOXYGEN_INVOKED` (upstream `_CCCL_DOXYGEN_INVOKED`) across device_reference/device_vector/host_vector/memory/retag/execution_policy and cpp/omp/tbb execution_policy.h. Conflicts resolved to upstream doc text; kept local hipcc-linkage `pointer(Element*)` ctor in detail/pointer.h.
  - HIP counterpart: none (system/hip/execution_policy.h has no `#if 0` doc block; no system/cuda paths touched).
  - test/ counterpart: none (no test changes).
  - Not compiled, tests not run (per human).
  - Local commit: dd2c8abc604
- [X] 4b2201184c4 Use `thrust::copy` in `thrust::uninitialized_copy[_n]` in CUDA system when possible (#5181) [SCOPE: HEADER,BENCH]
  - Clean port into system/cuda: uninitialized_copy_n delegates to copy_n when output is trivially constructible from input (performance-only); added `thrust::detail::raw_reference_t` alias in detail/raw_reference_cast.h (generic).
  - HIP counterpart: no change: performance-only (system/hip uninitialized_copy keeps its parallel_for path per sync decision).
  - Benchmark: new nvbench bench/uninitialized_copy/basic.cu rewritten for primbench as benchmark/bench/uninitialized_copy/basic.cu (uint8..uint64, no_copy, no_construct) and registered in benchmark/CMakeLists.txt.
  - test/ counterpart: none (no test changes).
  - Not compiled, tests not run (per human).
  - Local commit: 9d7099b1e54
- [X] 7b52284299d Rename our namespace macros (#5538) [SCOPE: HEADER,TEST]
  - Adaptation: local `_THRUST_STD_NAMESPACE_BEGIN/END` (thrust/detail/config/libcxx.h) already abstracts the libcu++ std namespace macros; kept local spelling in testing/copy.cu, testing/scan.cu, testing/unittest/testframework.h, tuple_of_iterator_references.h, zip_iterator.h. libcxx.h now maps to `_CCCL_BEGIN/END_NAMESPACE_CUDA_STD` when defined (CCCL >= 3.2), else the old `_LIBCUDACXX_*` names; added `_THRUST_LIBCXX_NAMESPACE_BEGIN/END` (cuda namespace) and used it in zip_function.h.
  - HIP counterpart: none (no system/cuda paths changed locally)
  - test/ counterpart: none (testing/copy.cu, testing/scan.cu namespace-macro rename only, kept local abstraction; test/test_copy.cpp / test/test_scan.cpp unaffected)
  - Not compiled, tests not run (per human)
  - Local commit: 09162f5157b
- [X] 854290f8b5f Add a benchmark for transform_if with stencil (#5571) [SCOPE: BENCH]
  - Ported as primbench benchmark: new `benchmark/bench/transform/if_stencil.cu` (negate_if_stencil, modeled on bench/transform/if.cu; stencil generated with seed+1; reads n + selected, writes selected) + `add_thrust_benchmark(bench/transform/if_stencil.cu)` in benchmark/CMakeLists.txt
  - HIP counterpart: none (benchmark only)
  - test/ counterpart: none (benchmark only)
  - Not compiled, tests not run (per human)
  - Local commit: 7530717d999
- [X] bc455b2d440 Remove repo-docs dependency (#5568) [SCOPE: HEADER] ⚠ touches `thrust/tuple.h`
  - Ported Doxygen comment fixes in thrust/ headers (group open/close markers, \param fixes, offset_iterator deduction guide `#ifndef THRUST_DOXYGEN_INVOKED`). offset_iterator.h: kept local deduction guide without `THRUST_HOST_DEVICE` (local spelling; guard already present). Non-thrust (docs/, cub/, .github) parts N/A.
  - HIP counterpart: none (doc comments only; system/cuda/vector.h comment only)
  - test/ counterpart: none (no test changes)
  - Not compiled, tests not run (per human)
  - Local commit: fc0ff525183
- [X] b2a8d0e88f8 Replace our qualification macros with plain `cuda::std::` (#5573) [SCOPE: HEADER,TEST] ⚠ touches `thrust/tuple.h`
  - N/A: local code already uses `_THRUST_STD::` (maps to `::cuda::std`/`::hip::std`) instead of `_CUDA_VSTD`/`_CCCL_*` qualification macros in tuple.h, pair.h, zip_iterator.h, tuple_of_iterator_references.h and testing/transform_iterator.cu; 3-way merge produced no change
  - HIP counterpart: none (no system/cuda paths)
  - test/ counterpart: none (testing/transform_iterator.cu already uses `_THRUST_STD::forward`; test/test_transform_iterator.cpp unaffected)
  - Not compiled, tests not run (per human)
  - Local commit: 4113417ddf3
- [X] 419d327e959 Documentation fixes (#5468) [SCOPE: HEADER]
  - Ported thrust/ part: shuffle_iterator.h Doxygen guard tweaks (`#ifndef THRUST_DOXYGEN_INVOKED` instead of `\cond`). Non-thrust doc fixes N/A.
  - HIP counterpart: none (documentation only)
  - test/ counterpart: none (documentation only)
  - Not compiled, tests not run (per human)
  - Local commit: 7b5b4a05ee8
- [X] 2e8d11a0e36 Avoids invoking custom equality operator for out-of-bounds items (#5566) [SCOPE: TEST]
  - testing/cuda/unique.cu: N/A, already backported (TestUniqueWithCustomEqualityOp + check_valid_item_op present; 3-way merge only conflicted on later local NonConstAdapter test, kept ours).
  - HIP counterpart: none (test-only commit; no header change)
  - test/ counterpart: added `TEST(UniqueTests, TestUniqueWithCustomEqualityOp)` in test/test_unique.cpp (unique_copy over counting_iterator with an equality op that counts out-of-bounds invocations) to cover the hip backend
  - Not compiled, tests not run (per human)
  - Local commit: c6f0a6ccb63
- [X] 8b68d958473 Drop internal uses of `thrust::reverse_iterator` (#5616) [SCOPE: HEADER,TEST,EXAMPLE,BENCH]
  - Ported: `thrust::reverse_iterator` / `thrust::make_reverse_iterator` are now aliases of `_THRUST_STD::reverse_iterator` / `_THRUST_STD::make_reverse_iterator` (local class removed); internal uses switched to `_THRUST_STD::` in detail/vector_base.h, system/detail/generic/reverse.inl, system/tbb/detail/reduce_by_key.inl, system/cuda/detail/reverse.h. iterator_system/iterator_traversal for `_THRUST_STD::reverse_iterator` already present (item 12). In the no-libhipcxx fallback this resolves to `::std::reverse_iterator` (constexpr members, usable on device under hipcc).
  - examples/cuda/range_view.cu kept `cuda::std::` spelling to match surrounding code; testing/cuda/sort.cu and benchmark/bench/partition/basic.cu (primbench, kept ours) use `_THRUST_STD::make_reverse_iterator`.
  - testing/*_iterator.cu comment-only conflicts resolved to upstream wording with `_THRUST_STD`; testing/reverse_iterator.cu adds `_THRUST_STD_INCLUDE(iterator)` and keeps local `unittest.h` include.
  - HIP counterpart: system/hip/detail/reverse.h mirrored (`_THRUST_STD::reverse_iterator{...}` / `_THRUST_STD::make_reverse_iterator`, unconditional `_THRUST_STD_INCLUDE(iterator)`, dropped thrust/iterator/reverse_iterator.h include)
  - test/ counterpart: none needed (test/test_reverse_iterator.cpp uses `thrust::reverse_iterator`, which now aliases `_THRUST_STD::reverse_iterator`; its assertions match upstream's)
  - Not compiled, tests not run (per human)
  - Local commit: 331c611fcb9
- [X] d7fbce513a7 Fixes `thrust::unique` for non-const `equality_op` (#5652) [SCOPE: TEST]
  - testing/cuda/unique.cu: N/A, already backported (NonConstAdapter + TestUniqueWithCustomEqualityOpMutable present; 3-way merge no-op).
  - HIP counterpart: none (test-only commit)
  - test/ counterpart: added `TEST(UniqueTests, TestUniqueWithCustomEqualityOpMutable)` in test/test_unique.cpp (unique with a non-const `__device__` equality op) to cover the hip backend
  - Not compiled, tests not run (per human)
  - Local commit: 3f86335ac62
- [X] ed7b88d9441 Fix `cuda::shuffle_iterator` not properly working with thrust algorithms (#5686) [SCOPE: HEADER,TEST]
  - Ported: iterator_system (any_system_tag) / iterator_traversal (random_access) specializations for `_THRUST_LIBCXX::shuffle_iterator` in thrust/iterator/iterator_traits.h.
  - HIP counterpart: none (no system/cuda paths; traits are backend-agnostic)
  - test/ counterpart: catch2_test_cuda_iterators.cu changes ported to GTest test/test_libcxx_iterators.cpp (new TestLibcxxShuffleIterator + fake_bijection; constant/counting/strided/transform/transform_output cases now check results) and legacy testing/cuda_iterators.cu (TestCUDAShuffleIterator + same strengthened checks)
  - Not compiled, tests not run (per human)
  - Local commit: ec576853b41
- [X] 10823d8cf51 Move nvbench_helper out of CUB for easier reuse. (#5692) [SCOPE: BENCH]
  - N/A: moves nvbench_helper out of CUB and updates thrust/benchmarks/CMakeLists.txt; rocThrust benchmarks use primbench + local bench_utils.hpp, no nvbench_helper
  - HIP counterpart: none (benchmark build infrastructure only)
  - test/ counterpart: none (benchmark build infrastructure only)
  - Not compiled, tests not run (per human)
  - Local commit: 4e6c4a360a9
- [X] d046198a7ee Fix `thrust::malloc` for `void` (#5698) [SCOPE: HEADER]
  - N/A: already backported (verified — system/detail/generic/memory.inl already has the `if constexpr (is_void_v<T>)` byte-allocation path; 3-way merge only differed by a blank line, kept ours)
  - HIP counterpart: none (generic backend, shared by hip)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: 68c5023135a
- [X] 065a6a742d6 Unify the libcudacxx header test infrastructure with the other projects (#5735) [SCOPE: CMAKE]
  - N/A: upstream drops Clang-CUDA flags from thrust/cmake/ThrustCudaConfig.cmake (moved to cccl compiler interface); rocThrust has no ThrustCudaConfig.cmake (HIP CMake build)
  - HIP counterpart: none (CMake only)
  - test/ counterpart: none (CMake only)
  - Not compiled, tests not run (per human)
  - Local commit: f1a38744694
- [X] d2453265b96 Address Sphinx warnings, populate Thrust's group pages (#5759) [SCOPE: HEADER]
  - Ported thrust/ Doxygen group fixes (host_vector.h containers group, partition.h missing `\{`). docs/ part N/A.
  - HIP counterpart: none (documentation only)
  - test/ counterpart: none (documentation only)
  - Not compiled, tests not run (per human)
  - Local commit: 0c14aa13adc
- [X] 8a95f4c8051 Fix Thrust header tests and remove unused defines. (#5764) [SCOPE: HEADER,CMAKE]
  - N/A (kept local): temporary_allocator.inl switches from `THRUST_DEVICE_SYSTEM == CUDA` to `_CCCL_CUDA_COMPILATION()` guards for header-test fixes; local file dispatches on THRUST_DEVICE_SYSTEM with a dedicated HIP branch (hip/detail/terminate.h, `_THRUST_IF_TARGET`), so the guard rewrite does not apply. thrust/cmake/ThrustHeaderTesting.cmake has no local counterpart.
  - HIP counterpart: none (local HIP branch already present and unchanged)
  - test/ counterpart: none (header-test CMake only)
  - Not compiled, tests not run (per human)
  - Local commit: 31863203b06
- [X] e6c67785ca6 Drop unused header (#5802) [SCOPE: HEADER]
  - Ported: deleted unused thrust/system/cpp/detail/vector.inl (not included anywhere locally; only spelling diffs vs upstream)
  - HIP counterpart: none (cpp system only)
  - test/ counterpart: none (unused header removal)
  - Not compiled, tests not run (per human)
  - Local commit: 53a7fc4578c
- [X] 5ff11d47308 Use nested namespace specifier in Thrust cpp system (#5801) [SCOPE: HEADER]
  - Clean port: C++17 nested namespace specifiers (`namespace system::detail::generic {`, etc.) across thrust/system/cpp/, system/detail/generic/, system/detail/sequential/, system/detail/internal/; all 3-way merges clean.
  - HIP counterpart: none (cpp/generic/sequential systems only; style-only change, system/hip not touched upstream)
  - test/ counterpart: none (no behavior change)
  - Not compiled, tests not run (per human)
  - Local commit: 7d8e82be9ea
- [X] ee141c6b175 Fix uninitialized read in uninitialized_copy_n (#5811) [SCOPE: HEADER]
  - Clean port (system/cuda): uninitialized_copy_n only delegates to copy_n when construction AND assignment are trivial.
  - HIP counterpart: no change: the copy_n fast path in uninitialized_copy_n was ported to system/cuda only (performance-only, item 4b2201184c4); hip uninitialized_copy still always placement-constructs, so the bug does not exist there
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: c91e8ff7919
- [X] 50929671faf Use DeviceTransform::Fill in uninitialized_fill[_n] when possible (#5813) [SCOPE: HEADER]
  - Clean port (system/cuda): uninitialized_fill_n delegates to fill_n (DeviceTransform::Fill) for trivially constructible+assignable types.
  - HIP counterpart: no change: performance-only
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: 46a9a9cc4ba
- [X] e93c9c33537 Port thrust::generate[_n] to cub::DeviceTransform::Fill (#5807) [SCOPE: HEADER,BENCH]
  - Clean port (system/cuda): generate/generate_n route through cub::DeviceTransform::Fill.
  - HIP counterpart: no change: performance-only
  - test/ counterpart: none (no upstream test change)
  - Benchmark: added primbench `benchmark/bench/generate/basic.cu` (modeled on bench/fill/basic.cu) and registered in benchmark/CMakeLists.txt
  - Not compiled, tests not run (per human)
  - Local commit: f500eddf9cc
- [X] d668cd199be Slim down Thrust CUDA core utils (#5845) [SCOPE: HEADER]
  - Clean port (system/cuda): core/util.h drops unused cub/thrust includes and the core::detail::BlockLoad helper; reduce_by_key.h and set_operations.h define a local BlockLoad alias and include cub/block/block_load.cuh.
  - HIP counterpart: no change: CUDA-only compile-time cleanup (HIP backend uses rocPRIM)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: 451a6a29843
- [X] 5222e2921fb Improve Thrust iterator documentation (#5833) [SCOPE: HEADER]
  - Ported Doxygen improvements; moved `thrust/iterator/detail/iterator_traversal_tags.h` to `thrust/iterator/iterator_traversal_tags.h` (git mv) and updated all includes (pointer.h, iterator_categories.h, iterator_category_to_traversal.h, iterator_facade_category.h, iterator_traits.h).
  - Conflict (iterator_traits.h x2): local `#ifndef THRUST_DOXYGEN_INVOKED` guards around iterator_traversal_t/iterator_system_t replaced by upstream doc comments (aliases are now documented upstream).
  - Upstream docs/thrust/api_docs/*.rst changes not ported (rocThrust docs are structured differently).
  - HIP counterpart: none (docs/header layout only)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: 9e6d4a0708c
- [X] 49b642e53dd Use a custom `iter_swap` kernel in Thrust (#5843) [SCOPE: HEADER]
  - Ported (system/cuda): iter_swap host path launches a dedicated iter_swap_kernel instead of swap_ranges.
  - Conflict (1): took upstream body with local `_THRUST_IF_TARGET`/`_THRUST_IS_HOST` spelling (matches get_value.h/assign_value.h).
  - HIP counterpart: no change: performance-only (compile-time)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: 3dd96159c3c
- [X] 0b700f32b1e Use forward declarations of extended floating point types instead of including the headers (#5846) [SCOPE: HEADER]
  - N/A: already backported (local commit 2d08c87c83c removed the cuda_fp16/cuda_bf16 includes from system/cuda/detail/sort.h); libcudacxx parts belong to libhipcxx.
  - HIP counterpart: none
  - test/ counterpart: none
  - Not compiled, tests not run (per human)
  - Local commit: 02b1624af57
- [X] bf56172e512 Test Thrust iterator system propagation (#5875) [SCOPE: TEST]
  - Upstream adds Catch2 `testing/catch2_test_iterator_traits.cu` (rocThrust has no Catch2 harness): not added; ported the three iterator-system-propagation cases (any / device / any+device via counting, transform, zip, device_vector iterators) as compile-time checks.
  - test/ counterpart: `test/test_type_traits.cpp` gains `TEST(TypeTraitsTests, TestIteratorSystemPropagation{AnySystem,DeviceSystem,AnyAndDeviceSystem})` (+ thrust/device_vector.h include).
  - testing/ counterpart: same three as `void TestIteratorSystemPropagation*()` + DECLARE_UNITTEST in `testing/type_traits.cu`.
  - HIP counterpart: none (test-only)
  - Not compiled, tests not run (per human)
  - Local commit: ddb5a95186d
- [X] abcb70514b7 Refactor thrust::iterator_facade_category (#5877) [SCOPE: HEADER]
  - Ported refactor: dropped unused ValueParam/Reference parameters from iterator_facade_default_category*; `iterator_facade_category<...>::type` replaced by `iterator_facade_category_t<System, Traversal>` (sole user iterator_facade.h updated).
  - Conflicts (4): took upstream structure, kept local `::internal::If` in place of `::cuda::std::_If` (not available via _THRUST_STD fallback).
  - cub/ iterator parts: not applicable (rocThrust uses hipCUB/rocPRIM).
  - HIP counterpart: none (generic iterator header)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: af24d6d1659
- [X] a94f5f38596 Add tests for host system propagation (#5881) [SCOPE: TEST]
  - Upstream extends Catch2 catch2_test_iterator_traits.cu (not present locally; see bf56172e512): ported the host-system propagation case (host_vector, transform, zip iterators -> host_system_tag).
  - test/ counterpart: `TEST(TypeTraitsTests, TestIteratorSystemPropagationHostSystem)` in `test/test_type_traits.cpp` (+ thrust/host_vector.h include).
  - testing/ counterpart: `TestIteratorSystemPropagationHostSystem` + DECLARE_UNITTEST in `testing/type_traits.cu` (+ thrust/host_vector.h include).
  - HIP counterpart: none (test-only)
  - Not compiled, tests not run (per human)
  - Local commit: 34f5f9cd0b3
- [X] af6a7158195 Drop obsolete is_discard_iterator (#5884) [SCOPE: HEADER]
  - `git rm thrust/detail/type_traits/iterator/is_discard_iterator.h` (unused locally, verified by grep across thrust/, test/, testing/, benchmark/).
  - HIP counterpart: none (no system/hip user)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: d85cfa3e55d
- [X] a0c7db9916c Fix `iterator_category_to_system` for device iterator tags (#5880) [SCOPE: HEADER,TEST]
  - Clean port (correctness fix, generic header): iterator_category_to_system.h gains forward/bidirectional/random_access host+device cat_to_system_impl overloads and `iterator_category_to_system_t`; iterator_category_to_traversal.h gains `iterator_category_to_traversal_t`.
  - HIP counterpart: none needed (generic iterator header shared by all backends; fix applies to HIP automatically)
  - test/ counterpart: Catch2 cases ported as `TEST(TypeTraitsTests, TestIteratorCategoryToSystem)` / `TestIteratorCategoryToTraversal` in `test/test_type_traits.cpp`.
  - testing/ counterpart: same two tests + DECLARE_UNITTEST in `testing/type_traits.cu`.
  - Not compiled, tests not run (per human)
  - Local commit: 10668f20f46
- [X] 025ed18d66e Remove unused include of grid/grid_queue (#5887) [SCOPE: HEADER]
  - Clean port (system/cuda): reduce.h now includes cub/grid/grid_queue.cuh directly (upstream dropped the transitive include from CUB dispatch headers; cub/ parts not applicable).
  - HIP counterpart: none (system/hip reduce uses rocPRIM, no GridQueue)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: 13e6c4a19a4
- [X] d91b32f1bf6 Add missing stream synchronization in thrust::cuda_cub::generate (#5889) [SCOPE: HEADER]
  - Clean port (system/cuda): generate_n adds `synchronize_optional` after the cub::DeviceTransform::Fill path (from e93c9c33537).
  - HIP counterpart: no change needed: system/hip generate_n goes through hip_rocprim::for_each_n -> parallel_for, which already calls synchronize_optional (parallel_for.h:157).
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: aea15d9a004
- [X] 4d1fc6ae38d Clarify missing Reference and ValueType (#5888) [SCOPE: HEADER]
  - Clean port: comment-only clarification in iterator_facade_category.h.
  - HIP counterpart: none (comment only)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: a023d858630
- [X] f4867bd8982 Make thrust::fill[_n] use cub::DeviceTransform (#5805) [SCOPE: HEADER]
  - Ported (system/cuda): fill_n now dispatches cub::detail::transform (DeviceTransform::Fill-style, `__return_constant`) instead of parallel_for; conflict (1, closing of removed `__fill` namespace) resolved to upstream. Result identical to upstream except local `_THRUST_STD::distance`.
  - cub/test/warp part: not applicable.
  - HIP counterpart: no change: performance-only
  - test/ counterpart: none (no upstream thrust test change)
  - Not compiled, tests not run (per human)
  - Local commit: 09034620cf3
- [X] da893945e45 Modernize Thrust examples (#5670) [SCOPE: EXAMPLE]
  - Ported example modernizations (initializer lists, range-for, host_vector staging, .size()) to 33 `examples/*.cu` files.
  - Deviations from upstream:
    - Kept explicit `#include "include/host_device.h"` in every example, and kept the local `examples/include/host_device.h` and `examples/CMakeLists.txt` unchanged. Upstream instead force-includes host_device.h through the CMake `thrust_add_example`, which rocThrust's example CMake does not have.
    - Kept the local `!_THRUST_HAS_DEVICE_SYSTEM_STD` `<iterator>` fallbacks in repeated_range/strided_range/tiled_range.
    - Spelling: `cuda::std::` -> `_THRUST_STD::` and `cuda::maximum` -> `_THRUST_LIBCXX::maximum`; `<cuda/functional>` -> `_THRUST_LIBCXX_INCLUDE(functional)`.
    - Kept `<cmath>` in max_abs_diff.cu (needed for fabsf on HIP).
  - HIP counterpart: none (examples only)
  - test/ counterpart: none (no upstream test change)
  - Not compiled, tests not run (per human)
  - Local commit: 55dceb1d0fb
- [X] ca57b79824c Inherit `thrust::transform_iterator` traversal from base iterator traversal (#5883) [SCOPE: HEADER,TEST]
  - Clean port (correctness fix, generic header): transform_iterator.h passes `use_default` for traversal so it is deduced from the base iterator instead of the raw std iterator_category (restores random-access traversal propagation).
  - HIP counterpart: none needed (generic iterator header; applies to HIP automatically)
  - test/ counterpart: the four `TestIteratorSystemPropagation*` GTests in `test/test_type_traits.cpp` are replaced by `TestIteratorSystemAndTraversalPropagation{AnySystem,Vectors,VectorsAndAnySystem}`. These mirror upstream's TEMPLATE_LIST cartesian products: thrust:: and `_THRUST_LIBCXX::` make_*_iterator wrappers, crossed with device_vector, host_vector and std::vector. The tests check both the system and random-access traversal. The libcxx combinations sit under `_THRUST_HAS_DEVICE_SYSTEM_STD`. Added the thrust/functional.h, <vector> and libcxx iterator includes.
  - testing/ counterpart: the same replacement in `testing/type_traits.cu`, using DECLARE_UNITTEST.
  - Not compiled, tests not run (per human)
  - Local commit: 91471d5c62d
- [X] 4eb027ce476 Drop `thrust::counting_iterator` in favor of `cuda::counting_iterator` (#5839) [SCOPE: HEADER,TEST,EXAMPLE,CMAKE]
  - N/A: reverted upstream by 07e250154a8 (#5925); the pair nets to zero in thrust/ (verified `git diff 4eb027ce476^ 07e250154a8 -- thrust/` is empty). No local change.
  - HIP counterpart: none
  - test/ counterpart: none
  - Not compiled, tests not run (per human)
  - Local commit: 1b2d7a69d64
- [X] 07e250154a8 Revert "Drop `thrust::counting_iterator` in favor of `cuda::counting_iterator` (#5839)" (#5925) [SCOPE: HEADER,TEST,EXAMPLE,CMAKE]
  - N/A: revert of 4eb027ce476 (#5839), which was not applied locally; the pair nets to zero in thrust/.
  - HIP counterpart: none
  - test/ counterpart: none
  - Not compiled, tests not run (per human)
  - Local commit: 8f51763a320
- [X] bb718514624 Drop Thrust mpl math (#5897) [SCOPE: HEADER,TEST]
  - `git rm thrust/detail/mpl/math.h`.
  - xor_combine_engine_max.h was rewritten to constexpr functions, using `_THRUST_STD` spelling. Its conflict came from the local include block; I used a single `_THRUST_STD_INCLUDE(algorithm)` in place of the `__algorithm/max.h`/`min.h` detail headers, which the `::std` fallback does not have.
  - system/cuda reduce_by_key/set_operations: mpl::min/max replaced by `::cuda::std::min/max` (conflict 1, took upstream). Dropped the unused mpl/math.h includes from scan_by_key/unique_by_key.
  - HIP counterpart: removed the unused `#include <thrust/detail/mpl/math.h>` from system/hip/detail/{scan_by_key,set_operations,unique_by_key}.h. No mpl::math uses remained.
  - test/ counterpart: git mv `test/test_metaprogramming.cpp` -> `test/test_xor_combine_engine_max.cpp`. It now has `TEST(XorCombineEngineMaxTests, TestLog2)` as static_asserts on `thrust::random::detail::log2`. The CMake registration "metaprogramming" was replaced by "xor_combine_engine_max".
  - testing/ counterpart: `git rm testing/metaprogramming.cu` and added upstream `testing/xor_combine_engine_max.cu`. Updated testing/CMakeLists.txt to match.
  - Not compiled, tests not run (per human)
  - Local commit: aad74167a62
- [X] e2dda739257 Modernize iterator machinery (#5928) [SCOPE: HEADER,TEST]
    - N/A: thrust side already uses `_THRUST_STD::__has_input_traversal` / `__has_random_access_traversal` (vector_base.h/.inl, 11 testing/*_iterator.cu static_asserts); upstream renamed these to `__is_cpp17_*` here, then renamed back to `__has_*_traversal` in e20f6a68c8a before v3.2.1, so the local spelling matches v3.2.1. All conflicts resolved to ours; no diff. libcudacxx parts belong to libhipcxx.
    - HIP counterpart: none.
    - test/ counterpart: none (GTest has no equivalent static_asserts).
    - Not compiled, tests not run (per human).
  - Local commit: 49907612445
- [X] feb641f2a59 Fix iterator adaptor sample (#5957) [SCOPE: HEADER]
    - Doc-only: iterator_adaptor.h sample drops `const` from the `begin` member so the sample iterator is assignable.
    - HIP counterpart: none (doc comment only).
    - test/ counterpart: none; testing/ counterpart: none.
    - Not compiled, tests not run (per human).
  - Local commit: 7efc1a73f8f
- [X] 5197e4b8a58 Refactor and condense thrust::copy implementation (#5491) [SCOPE: HEADER]
    - cuda `copy.h` merged cleanly (now identical to upstream): cross-system and device-to-device copy folded in; device-to-device routes through `cuda_cub::transform` with `::cuda::std::identity`. `git rm` cuda `internal/copy_cross_system.h` and `internal/copy_device_to_device.h` (only included by cuda copy.h).
    - HIP counterpart: no change: performance-only refactor; `system/hip/detail/copy.h` keeps its `internal/copy_{cross_system,device_to_device}.h` (per compatibility-over-performance decision).
    - test/ counterpart: none; testing/ counterpart: none (no upstream test changes).
    - Not compiled, tests not run (per human).
  - Local commit: d7763108212
- [X] 1ceac3731d5 Improve `thrust::cuda_cub::replace` functor handling (#5949) [SCOPE: HEADER]
    - Generic `detail/internal_functional.h`: added `new_value_if_f` (`_CCCL_DEVICE_API` spelled `THRUST_DEVICE`) and `proclaims_copyable_arguments` specializations for `predicate_to_integral`, `tuple_binary_predicate`, `new_value_if_f`; specializations placed in `_THRUST_LIBCXX_NAMESPACE_BEGIN/END` or `namespace internal` (fallback), matching `zip_function.h`; address_stability pulled in via `thrust/detail/libcxx_wrapper/__functional/address_stability.h` instead of the cuda include.
    - cuda `replace.h` merged cleanly: uses `thrust::detail::new_value_if_f` with `cuda_cub::transform`, dropping its local functor.
    - HIP counterpart: no change: performance-only (copyable-argument proclamation only feeds CUDA DeviceTransform); `system/hip/detail/replace.h` keeps `__replace::new_value_if_f`. cub `device_transform.cuh` change belongs to hipCUB.
    - test/ counterpart: none; testing/ counterpart: none (no upstream test changes).
    - Not compiled, tests not run (per human).
  - Local commit: 3d25701be6a
- [X] 2f39a342b93 Refactor Thrust execution policies (#5821) [SCOPE: HEADER,TEST]
    - Generic: cpp/omp/tbb `tag`/`execution_policy` moved out of `system::X::detail` into `system::X`, and `par_t`/`par` folded into each `detail/execution_policy.h`; `git rm` `system/{cpp,omp,tbb}/detail/par.h`. Upstream SPDX header taken where the file had no AMD modification line; AMD-modified license headers kept (`seq.h`, `execution_policy.h`, `sequential/execution_policy.h`). `_CCCL_*` spelled `THRUST_*`.
    - `system/detail/sequential/execution_policy.h`: kept the local converting constructor on `sequential::tag` (needed for minimum_system), took upstream's `operator tag()` and TODO comment.
    - `thrust/execution_policy.h`: Doxygen-only `#if THRUST_DOXYGEN_INVOKED` blocks replaced by upstream doc comments; `host` is now `inline constexpr`; device docs keep "CUDA or HIP" wording; kept `#include <thrust/detail/execute_with_allocator.h>` (upstream dropped it) so `thrust::device(alloc)` keeps working for users including only this header.
    - cuda: `par.h`/`par_to_seq.h` folded into `cuda/detail/execution_policy.h` (`git rm` both; local leftover `<cuda/stream_ref>` include dropped with par.h); algorithm headers include `execution_policy.h` instead of `par_to_seq.h`; `util.h` forward-declares `execution_policy`.
    - HIP counterpart: no change: `system/hip/detail/par.h` and `par_to_seq.h` kept as-is (pure restructuring, no behavior fix; `hip/execution_policy.h` still includes `hip/detail/par.h`, `thrust::system::hip::detail::par_t` still exists for `device_t`).
    - testing/ counterpart: `allocator_aware_policies.cu` includes `{cpp,omp,tbb}/detail/execution_policy.h` (kept `hip/detail/par.h`, no cuda block); `gather.cu` gains gcc>=11 `THRUST_DISABLE_BROKEN_GCC_VECTORIZER` on `TestGatherCountingIterator`; `transform_input_output_iterator.cu` workaround widened to gcc>=12 (spelled `THRUST_COMPILER`).
    - test/ counterpart: `test_allocator_aware_policies.cpp` same include/policy_info updates (`thrust::system::X::execution_policy`); `test_gather.cpp` adds the gcc>=11 vectorizer workaround on `TestGatherCountingIterator`; `test_transform_input_output_iterator.cpp` widened to gcc>=12.
    - Not compiled, tests not run (per human).
  - Local commit: 9b393ddebdc
- [X] dcc24cd082b Try to use _CCCL_API in Thrust and CUB (#5953) [SCOPE: HEADER]
    - cuda `config.h`: `THRUST_{DEVICE,HOST}_FUNCTION`/`THRUST_FUNCTION` now built on `_CCCL_{DEVICE,HOST}_API`/`_CCCL_API` (merged cleanly). cub `detect_cuda_runtime.cuh` belongs to hipCUB.
    - HIP counterpart: none (`system/hip/config.h` defines these on HIP attributes; `_CCCL_*_API` visibility macros are CUDA/libcu++-specific).
    - test/ counterpart: none; testing/ counterpart: none.
    - Not compiled, tests not run (per human).
  - Local commit: a0f51ec0236
- [X] 9e73f14e8f1 Improve interoperability of `cuda` iterators with thrust and std (#5929) [SCOPE: HEADER]
    - Partially backported: `transform_output_iterator<Fn, Iter>` / `transform_iterator<Fn, Iter>` specializations were already in the libcu++ 3.2 order; this ports the remaining `transform_input_output_iterator<InputFn, OutputFn, Iter>` reorder in `iterator_traits.h` (spelled `_THRUST_LIBCXX::`), matching libhipcxx >= 3.2. libcudacxx iterator changes belong to libhipcxx.
    - HIP counterpart: none (generic header).
    - test/ counterpart: none; testing/ counterpart: none (upstream only added libcudacxx tests).
    - Not compiled, tests not run (per human).
  - Local commit: e1ea5da54e2
- [X] 03552de3825 Drop thrust detail seq policy global (#5964) [SCOPE: HEADER]
    - Dropped the `thrust::system::detail::sequential::seq` global; tbb `for_each.inl` now uses `thrust::seq` (includes `thrust/detail/seq.h`). Merged cleanly; no other local users of `sequential::seq` (checked `system/hip`, tests, benchmarks).
    - HIP counterpart: none (HIP backend never referenced the sequential global).
    - test/ counterpart: none; testing/ counterpart: none.
    - Not compiled, tests not run (per human).
  - Local commit: 1d2185ddd56
- [X] 8979072c17a Deprecate and replace THUST_[HOST|DEVICE]_FUNCTION (#5972) [SCOPE: HEADER,BENCH]
    - cuda: `THRUST_{DEVICE,HOST}_FUNCTION`/`THRUST_FUNCTION` marked deprecated in `cuda/config.h`; all cuda backend uses replaced by `_CCCL_{DEVICE,HOST}_API _CCCL_FORCEINLINE`. Conflicts in `reduce.h`/`reduce_by_key.h` took upstream signatures, keeping local `_THRUST_STD::true_type`/`false_type` spelling.
    - HIP counterpart: none (macros are only defined/used under `system/cuda`; HIP backend uses `THRUST_HOST_DEVICE`/`THRUST_DEVICE`).
    - Benchmark counterpart: `benchmark/bench/generate/basic.cu` already uses `THRUST_DEVICE` (no `THRUST_DEVICE_FUNCTION`); conflict resolved to ours, no change.
    - test/ counterpart: none; testing/ counterpart: none.
    - Not compiled, tests not run (per human).
  - Local commit: cf4cf22097b
- [X] 9fbba017026 Fix dereferencing nullptr in `thrust::device_reference` (#4226) [SCOPE: HEADER,TEST]
    - N/A: already backported on the base branch (`reference.h`, `seq.h`, `minimum_system.h`, `select_system.h`, `sequential/execution_policy.h` merged with no change).
    - HIP counterpart: none (generic headers).
    - testing/ counterpart: already present as `testing/minimum_system.cu` (`MinimumSystem`); upstream Catch2 file not added.
    - test/ counterpart: already present as `test/test_minimum_system.cpp` (`MinimumSystemTests.MinimumSystem`).
    - Not compiled, tests not run (per human).
  - Local commit: 789aeb4361a
- [X] 36ec00be03f Ensure non-trivial thrust::copy from D2D uses UBLKCP (#5954) [SCOPE: HEADER]
    - cuda `copy.h`: non-trivial device-to-device copy wraps `::cuda::std::identity` in `::cuda::proclaim_copyable_arguments` so DeviceTransform can use UBLKCP (merged cleanly). cub `kernels/transform.cuh` belongs to hipCUB.
    - HIP counterpart: no change: performance-only.
    - test/ counterpart: none; testing/ counterpart: none.
    - Not compiled, tests not run (per human).
  - Local commit: 52d443083d8
- [X] fe943f23ced Ensure thrust::scatter_if uses UBLKCP (#5952) [SCOPE: HEADER]
    - cuda `scatter.h`: `scatter`/`scatter_if` wrap `::cuda::std::identity` in `::cuda::proclaim_copyable_arguments` (UBLKCP, performance) and the predicate-less `scatter_if` overload drops its non-deducible `class Predicate` template parameter (merged cleanly).
    - HIP counterpart: ported the overload fix only: `system/hip/detail/scatter.h` predicate-less `scatter_if` drops the non-deducible `class Predicate` parameter, so the overload is actually viable instead of falling back to the generic path. `proclaim_copyable_arguments` wrapping: no change: performance-only.
    - test/ counterpart: none; testing/ counterpart: none (existing scatter_if tests cover the overload).
    - Not compiled, tests not run (per human).
  - Local commit: 61193836ea2
- [X] e20f6a68c8a Rename the trait checks to `__has_meow_traversal` (#5968) [SCOPE: HEADER,TEST]
    - N/A: rename back to `__has_*_traversal` matches what rocThrust already uses (see e2dda739257); all thrust-side files merged with no change. libcudacxx parts belong to libhipcxx.
    - HIP counterpart: none.
    - test/ counterpart: none; testing/ counterpart: none (already spelled `__has_*_traversal`).
    - Not compiled, tests not run (per human).
  - Local commit: 7eb4bed0ab4
- [X] 168e81561d4 Try to clean up align utilities (#5950) [SCOPE: HEADER,TEST]
  Deleted thrust/detail/util/align.h (align_up/align_down/is_aligned); its only user testing/cuda/cudart.cu now uses ::cuda::std::is_sufficiently_aligned<alignof(T)> with <cuda/std/memory> (CUDA-only test, kept raw cuda spelling; AMD license header kept). cuda reduce.h ReduceAgent::is_aligned now uses ::cuda::std::is_sufficiently_aligned<alignof(Vector)>. HIP: no change (system/hip reduce does not use align.h; cleanup only). test/: no counterpart (no test uses align.h). testing/: cuda/cudart.cu updated. Rest of upstream commit is libcudacxx/cub/c only. Not compiled, tests not run (per human)
  - Local commit: 8bcfade550e
- [X] ef51c79d683 Replace `_LIBCUDACXX_DEPRECATED` with `CCCL_DEPRECATED` (#6024) [SCOPE: HEADER]
  Thrust side of upstream is clang-format-only reflow after the _LIBCUDACXX_DEPRECATED→CCCL_DEPRECATED rename (rest is libcudacxx/cub). advance.h, functional.h, iterator/iterator_traits.h: kept ours (local already uses CCCL_/THRUST_DEPRECATED_BECAUSE with hip/_THRUST_STD wording). distance.h: took upstream placement of [[nodiscard]] before CCCL_DEPRECATED_BECAUSE. HIP: no change (no system/hip usage). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: c1b26f255a3
- [X] ff8a0d96b7f Unwrap cuda::zip_iterator/zip_function in thrust::transform (#6039) [SCOPE: HEADER,TEST]
  cuda transform.h: new cub_transform_many overload unwraps ::cuda::zip_iterator/::cuda::zip_function (via __iterators()/__fun()) for cub::DeviceTransform, plus includes of cuda/__iterator/zip_{function,iterator}.h. HIP: no change: performance-only (DeviceTransform unwrapping; system/hip transform unaffected). testing/: cuda/transform.cu merged (CUDA-only test: renamed TestTransformThrustZipIteratorUnwrapping, added zip_function<sum_five> specialization and TestTransformCudaZipIteratorUnwrapping, include via _THRUST_LIBCXX_INCLUDE(iterator)). test/: no counterpart (test is CUDA-backend specific). Not compiled, tests not run (per human)
  - Local commit: 199324aace2
- [X] d9ebd3eb183 Add more tests for `thrust::reduce_by_key` (#6063) [SCOPE: TEST]
  Test-only upstream commit. testing/: cuda/reduce_by_key.cu merged (AMD license header kept; adds check_valid_item_op/check_accumulator_t_op, TestReduceByKeyWithCustomEqualityOp, TestReduceByKeyWithDifferentAccumulatorT; kept the thrust/iterator/counting_iterator.h include upstream dropped since thrust::counting_iterator is still used). test/: ported both tests to test/test_reduce_by_key.cpp as TEST(ReduceByKeyTests, ...) using thrust::make_counting_iterator/make_constant_iterator, thrust::equal_to<> and std:: types. HIP: no change (test-only). Not compiled, tests not run (per human)
  - Local commit: e51faba04d1
- [X] ad70011e990 Support fancy iterators in vectorized transform and port thrust::tabulate to it (#6012) [SCOPE: HEADER,BENCH]
  cuda tabulate.h: took upstream (drop __tabulate::functor/parallel_for; implement via cuda_cub::transform_n over ::cuda::counting_iterator with ::cuda::proclaim_copyable_arguments(tabulate_op)); fancy-iterator support itself is in cub DeviceTransform. HIP: no change: performance-only (system/hip tabulate keeps its parallel_for path). benchmark/: bench/tabulate/basic.cu adapted to primbench: added sequence_benchmark (subalgo "sequence"), seg_size_t static_cast, output vectors use thrust::no_init, types widened to int8/16/32/64 (upstream integral_types); kept subalgo "basic" for the seg_size benchmark (upstream renamed it seg_size). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 291f437712f
- [X] 578ff26c307 Refactor `thrust::[try_]unwrap_contiguous_iterator[_t]` (#6065) [SCOPE: HEADER]
  type_traits/unwrap_contiguous_iterator.h: took upstream (drop detail::contiguous_iterator_traits and detail::try_unwrap_contiguous_iterator_impl; unwrap_contiguous_iterator uses deduced return type, *_t aliases are decltype of the functions, try_unwrap uses if constexpr; THRUST_EXEC_CHECK_DISABLE on both). Local file had a partially-applied state where the trailing return referenced the alias before its declaration; now consistent. No local users of the removed detail helpers. HIP: no change (generic header). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: ebaf03faf22
- [X] 57b16c2fc64 [CUB] Replace several direct uses of `__clz` (#6099) [SCOPE: HEADER]
  N/A: thrust/detail/integer_math.h already matches upstream (clz replacement via countl_zero was already backported); rest of upstream commit is cub. HIP: no change. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: a57a8a48a28
- [X] fa13b90c53b Implement `cuda::zip_transform_iterator` (#5982) [SCOPE: HEADER,TEST,BENCH]
  iterator/iterator_traits.h: added iterator_system/iterator_traversal specializations for _THRUST_LIBCXX::zip_transform_iterator (libhipcxx >= 3.2 provides it). cuda copy.h: device_to_device unwraps ::cuda::zip_transform_iterator into __transform::cub_transform_many (forward-declared; includes cub tuning_transform, cuda/__fwd/zip_iterator.h). HIP counterpart (system/hip): no change: performance-only (HIP copy handles zip_transform_iterator via the generic transform path thanks to the iterator_traits specializations). testing/: cuda_iterators.cu gained TestCUDAZipTransformIterator (upstream catch2_test_cuda_iterators.cu has no local copy). test/: test_libcxx_iterators.cpp gained TestLibcxxZipTransformIterator. benchmark/: kept local transform/basic.cu (primbench; already contains both fib and babelstream, so upstream's split into babelstream.cu/fib.cu not mirrored); added primbench transform/zip_transform.cu (subalgo zip_transform.nstream) and registered it in benchmark/CMakeLists.txt. Not compiled, tests not run (per human)
  - Local commit: f7d7e8ea011
- [X] a1683b97d87 Remove fully qualified ::cuda::std:: from examples (#6130) [SCOPE: EXAMPLE]
  examples/: upstream replaces ::cuda::std:: with cuda::std:: and reflows. Local generic examples already use _THRUST_STD:: (constant_iterator.cu, monte_carlo.cu, monte_carlo_disjoint_sequences.cu kept ours; merged reflow would have reintroduced raw cuda::std::plus); other touched examples unchanged by merge. examples/cuda/range_view.cu (CUDA-only example): took upstream removal of the unneeded "../include/host_device.h" include. HIP: no change (examples only). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 6ae37b28d8e
- [X] ab99d87e866 Move nvhpc header wrappers to libcudacxx. (#6167) [SCOPE: HEADER,TEST]
  N/A (kept ours): upstream deletes thrust/detail/{algorithm,memory,numeric}_wrapper.h and includes the libcudacxx copies <cuda/std/__cccl/*_wrapper.h> instead (nvhpc include-cycle guard moved). Locally that would map to _THRUST_STD_INCLUDE(__cccl/memory_wrapper.h), which expands to <__cccl/memory_wrapper.h> in builds without libhipcxx and breaks them; for compatibility kept the thrust-local wrapper headers and all existing includes (allocate_unique.h, allocator/*, internal_functional.h, memory_algorithms.h, temporary_array.h, host_vector.h, mr/pool.h, mr/disjoint_pool.h unchanged). testing/cmake/check_source_files.cmake has no local copy. HIP: no change. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 28719cf7f00
- [X] 700f2a867a7 Refactor Thrust `destroy_range` and `device_[new|delete|free]` (#6134) [SCOPE: HEADER]
  Took upstream refactor: destroy_range.inl, device_delete.inl, device_free.inl, device_new.inl folded into their headers (git rm of the .inl files); destroy_range now uses if constexpr over has_effectful_member_destroy (variable template, std::allocator specialization false) instead of enable_if cases. Local spellings kept (THRUST_HOST_DEVICE, THRUST_EXEC_CHECK_DISABLE, _THRUST_STD::is_trivially_destructible_v); destroy_range.h keeps <thrust/detail/memory_wrapper.h> (see ab99d87e866) and the !_THRUST_HAS_DEVICE_SYSTEM_STD <type_traits> fallback. HIP: no change (generic headers). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 325a086b899
- [X] fbb9479172c Refactor Thrust allocator internals (#6136) [SCOPE: HEADER]
  Took upstream allocator-internals refactor: allocator_traits, copy_construct_range, fill_construct_range, malloc_allocator, tagged_allocator, temporary_allocator and value_initialize_range .inl files folded into their .h (git rm of the .inl files), enable_if dispatch replaced by if constexpr / variable templates. Local adaptations kept: <thrust/detail/memory_wrapper.h> instead of <cuda/std/__cccl/memory_wrapper.h> (see ab99d87e866), !_THRUST_HAS_DEVICE_SYSTEM_STD <type_traits> fallbacks, _THRUST_STD_INCLUDE(limits); temporary_allocator.h restores the HIP device path (THRUST_DEVICE_SYSTEM_HIP -> _THRUST_IF_TARGET + thrust::system::hip::detail::terminate_with_message, plus hip/detail/terminate.h include) and the AMD Modifications Copyright line from the deleted .inl. HIP: no change (generic headers; HIP terminate path preserved). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 2a537d64af3
- [X] dbd58724133 Bypass allocator in `thrust::device_delete` (#6198) [SCOPE: HEADER,TEST]
  Correctness fix, generic headers: device_delete now runs destructors on the device via thrust::for_each_n(device, ptr, n, detail::gozer{}) instead of destroy_range with an empty allocator; device_new(p, n) value-initializes in place on the device via device_allocator + detail::value_initialize_range (no host-constructed exemplar copy), and device_new(p, exemplar, n) passes the device policy to uninitialized_fill. Applies to HIP via the generic headers. testing/: device_delete.cu re-enabled TestDeviceDeleteDestructorInvocation (no KNOWN_FAILURE/QNX guard), Foo is device-only and non-copyable (THRUST_DEVICE spelling). test/: test_device_delete.cpp updated the same way (enabled test, same Foo). HIP: covered via generic headers, no system/hip change. Not compiled, tests not run (per human)
  - Local commit: 06a541c41ff
- [X] 01cd9b491ed Support polymorphic types in thrust::device_delete (#6140) [SCOPE: TEST]
  Test-only. testing/: device_delete.cu merged (Foo member renamed to destroyed; new base/derived polymorphic types and TestDeviceDeleteVirtualDestructorInvocation; _CCCL_HOST_DEVICE -> THRUST_HOST_DEVICE). test/: test_device_delete.cpp mirrored (TEST(DeviceDelete, TestDeviceDeleteVirtualDestructorInvocation) plus the Foo update). HIP: no change (test-only; relies on generic device_delete from dbd58724133). Not compiled, tests not run (per human)
  - Local commit: 973de3f4425
- [X] 1c3c7ddee12 Refactor iterator usage in thrust/cuda find() (#6019) [SCOPE: HEADER,TEST]
  HIP: system/hip/detail/mismatch.h transform_pair_of_input_iterators_t got the iterator correctness fixes (defaulted default ctor, prefix operator++ returns self_t&, operator<); the find() zip/transform/counting_iterator rewrite is performance-only, so HIP find.h is unchanged. test/: no counterpart (no test change upstream beyond CUDA-only file). testing/: cuda/mismatch.cu unused-parameter cleanup taken. Not compiled, tests not run (per human)
  - Local commit: 5b9b72ee172
- [X] 73d8738e860 Try different formulation for thrust::ccosh (#6200) [SCOPE: HEADER]
  HIP: generic complex header, applies to HIP automatically (taken as upstream). test/: no counterpart (no upstream test change). testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 47bf963b955
- [X] c859e948453 Improve `invoke` machinery (#6227) [SCOPE: HEADER]
  HIP: generic thrust/detail/type_traits.h; libhipcxx branch now uses _THRUST_STD::invoke_result instead of the internal __invoke_of (non-libhipcxx rocprim/std fallbacks unchanged), applies to HIP automatically. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 33e87b68197
- [X] bfb875c8f73 Drop all uses of `thrust::tabulate_output_iterator` in favor of `cuda::tabulate_output_iterator` (#6001) [SCOPE: HEADER,TEST,CMAKE]
  HIP: generic iterator header; thrust::tabulate_output_iterator / make_tabulate_output_iterator now CCCL_DEPRECATED_BECAUSE (THRUST_HOST_DEVICE spelling kept), no internal rocThrust users remain, applies to HIP automatically. test/: test_tabulate_output_iterator.cpp defines CCCL_IGNORE_DEPRECATED_API. testing/: tabulate_output_iterator.cu defines CCCL_IGNORE_DEPRECATED_API (AMD license header kept). CMake: ThrustHeaderTesting.cmake has no rocThrust counterpart (no header-test targets), skipped. Not compiled, tests not run (per human)
  - Local commit: 8742e2e52dc
- [X] 13886f72f16 Add host standard library detection (#6244) [SCOPE: HEADER]
  N/A: upstream switches is_contiguous_iterator.h to _CCCL_HOST_STD_LIB(LIBCXX/LIBSTDCXX/STL); rocThrust has no THRUST-spelled equivalent and the macro is unavailable without libhipcxx, so the existing defined(_LIBCPP_VERSION)/defined(__GLIBCXX__)/defined(_MSC_VER) checks (behaviourally equivalent) are kept. HIP: no change, generic header. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: f36d487a3e8
- [X] c99060c19e9 Deprecate `<cuda/stream_ref>` header (#6266) [SCOPE: TEST]
  HIP: no counterpart (CUDA-only test include change, <cuda/stream_ref> -> <cuda/stream>, raw spelling kept as in the CUDA-only test). test/: no counterpart (no stream_ref usage in gtest tests). testing/: cuda/stream_wrapper.cu updated. Not compiled, tests not run (per human)
  - Local commit: d323feb3b7f
- [X] 344641b3b3c Limit deprecation exclusions to targeted headers. (#6275) [SCOPE: CMAKE]
  N/A: only touches thrust/cmake/ThrustHeaderTesting.cmake, which has no rocThrust counterpart (rocThrust has no per-header test targets). HIP: no change. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: e36b7111d56
- [X] 5303f65de2d Dropped duplicated math function from Thrust (#6188) [SCOPE: HEADER]
  Partial: upstream deletes thrust/detail/integer_math.h and switches execute_with_allocator.h / mr/{pool,disjoint_pool,pool_options}.h to ::cuda::ceil_div / ceil_ilog2 / is_power_of_two. Kept integer_math.h and the thrust::detail helpers in generic headers for compatibility (those ::cuda:: cmath utilities are unavailable without libhipcxx; behaviour is identical). system/cuda: dropped the now-redundant integer_math.h include from scan.h/sort.h (is_negative still reaches scan.h via dispatch.h); cuda dispatch.h keeps including integer_math.h instead of defining its own is_negative, which would collide with integer_math.h's overloads. HIP counterpart (system/hip): no change (hip dispatch.h still uses thrust::detail::is_negative from integer_math.h). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 65b0b4d28af
- [X] 1238f287f4f add parallel scan support for TBB and OMP (#6178) [SCOPE: HEADER]
  HIP: host OMP/TBB backends are shared by HIP builds, applies automatically. New omp/detail/scan.inl adapted for builds without libhipcxx: umbrella _THRUST_STD_INCLUDE(algorithm/iterator/numeric) instead of libcu++ internal __numeric/* headers, ::internal::accumulator_t (libcxx_wrapper invoke.h, as in sequential/scan.h) instead of _THRUST_STD::__accumulator_t, and an inline integer ceil-div instead of _THRUST_LIBCXX::ceil_div; SPDX NVIDIA + AMD Modifications header. Upstream's later omp scan bugfix (26c79caf60e) follows in order. test/: no counterpart (no upstream test change). testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 2243c7505ba
- [X] 75f88e5a5db Ignore OOM failures for large size unique thrust test. (#6304) [SCOPE: TEST]
  HIP: no counterpart (test-only change). test/: no counterpart (test_unique.cpp has no large-magnitude unique test). testing/: cuda/unique.cu TestUniqueWithLargeNumberOfItems now skips on std::bad_alloc. Not compiled, tests not run (per human)
  - Local commit: b17ac75d7af
- [X] 638c03d54f7 Replace __popc with cude::std::popcounter (#6213) [SCOPE: HEADER]
  HIP counterpart (system/hip): no change; hip/detail/set_operations.h keeps the HIP __popc intrinsic (valid on all HIP targets; switching to _THRUST_STD::popcount would require libhipcxx and gives no correctness benefit). system/cuda: __popc -> ::cuda::std::popcount. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 134adcc71b4
- [X] f1b4a17019b Always include `<new>` when we need operator new for clang-cuda (#6310) [SCOPE: HEADER]
  HIP counterpart (system/hip): no change needed in hip/detail/uninitialized_{copy,fill}.h (no clang-cuda assignment workaround there; hip-clang already supports device placement new). Generic internal_functional.h / generic uninitialized_copy.inl include _THRUST_STD_INCLUDE(__new/device_new.h) only under _THRUST_HAS_DEVICE_SYSTEM_STD (libcu++-internal header; non-libhipcxx builds keep <thrust/detail/memory_wrapper.h> / <type_traits>). system/cuda: device_new.h includes added and the _CCCL_CUDA_COMPILER(CLANG) assignment fallbacks removed. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 083584467f0
- [X] 325b8851fe8 Fix thrust system dependend includes (#6311) [SCOPE: HEADER]
  HIP: generic dispatch headers; new "#if 0" build-system include hints (execution_policy.h, device/host_system_tag.h, mr/{device,host}_memory_resource.h, universal_{allocator,vector}.h) also list the thrust/system/hip/... counterparts alongside cuda; adl/*.h comment rewording taken (existing hip entries kept). universal_{allocator,vector}.h hint blocks placed before the local THRUST_DOXYGEN_INVOKED guard. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: d0ffe60bd04
- [X] 52eb28883c4 Simplify thrust::zip_function (#6321) [SCOPE: HEADER,TEST]
  Partial: zip_function.h keeps the local detail::zip_detail::apply (upstream switches to ::cuda::std::apply, which is not usable with non-libhipcxx tuples on device; behaviour identical). HIP: no change (generic header kept). test/: no counterpart. testing/: cuda/transform.cu zip_function<sum_five> specialization now returns double instead of the decltype(zip_detail::apply(...)) trailing return, as upstream. Not compiled, tests not run (per human)
  - Local commit: df5274e3c2d
- [X] 8ca3dc31e90 Refactor thrust::mismatch to use CUDA iterators (#6018) [SCOPE: HEADER]
  HIP counterpart (system/hip): no change: performance-only / internal refactor (cuda mismatch drops transform_pair_of_input_iterators_t for ::cuda::zip_iterator + ::cuda::zip_function); hip/detail/mismatch.h keeps its iterator (already carries the 1c3c7ddee12 correctness fixes). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: d656f2535fa
- [X] 3ec95622aff Move the `throw_meow_error` functions into their own header and drop the `stdexcept` include (#6335) [SCOPE: HEADER]
  HIP counterpart (system/hip): no change; hip/detail/dispatch.h uses the libcxx_wrapper stdexcept.h shim (::internal::__throw_runtime_error), which works with and without libhipcxx. system/cuda: dispatch.h includes <cuda/std/__exception/throw_error.h> instead of the libcxx stdexcept header (other include additions from upstream belong to the is_negative rewrite not ported in 5303f65de2d). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: e50e7c3cbed
- [X] 81b9aa5f1cc Move the implementation of `tuple` into its own file (#6336) [SCOPE: HEADER]
  HIP: generic header; the libhipcxx-only (_THRUST_HAS_DEVICE_SYSTEM_STD) specialization now targets the __is_tuple_of_iterator_references_v variable template, matching the libcu++ tuple refactor in libhipcxx >= 3.2; non-libhipcxx path unchanged. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: d63ad167556
- [X] 85bf1b00c4d Refactor generic thrust scan dispatch (#6360) [SCOPE: HEADER]
  HIP: generic dispatch header; scan.inl folded into scan.h and the backend-required binary_op overloads are now "= delete" (HIP backend provides them), applies to HIP automatically. Uses <thrust/functional.h> instead of _THRUST_LIBCXX_INCLUDE(functional) so builds without libhipcxx still get _THRUST_STD::plus; THRUST_HOST_DEVICE spelling kept. git rm generic/scan.inl. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 53ef5ee1437
- [X] e73cc82956a Move CPP, OMP and TBB exec policies to detail (#6361) [SCOPE: HEADER,TEST]
  HIP counterpart (system/hip): no change; hip tag/execution_policy already live in thrust::hip_rocprim alongside the HIP algorithms, so ADL already finds them (the cuda change is only a comment). cpp/omp/tbb tag + execution_policy moved into system::<sys>::detail with using-declarations in system::<sys>, as upstream. omp/detail/scan.inl: P0571 accumulator (no-init tag, wrapped_function, init applied in block-sum scan), keeping the libhipcxx-free includes (_THRUST_STD_INCLUDE(type_traits) added, libcxx_wrapper invoke.h include dropped as accumulator_t is no longer used). test/: test_scan.cpp only_set_when_expected_it gains postfix operator++(int). testing/: scan.cu same (THRUST_HOST_DEVICE spelling). Not compiled, tests not run (per human)
  - Local commit: 6f8eb446d11
- [X] c22298179e0 Refactor Thrust OMP system headers (#6372) [SCOPE: HEADER]
  HIP counterpart (system/hip): no change; OMP host backend restructuring only (shared by HIP builds, applies automatically). Inlined the 14 omp/detail/*.inl files into their .h and git rm'd them; SPDX headers and nested namespaces as upstream. Local edits preserved: omp/detail/scan.h keeps the libhipcxx-free includes (_THRUST_STD_INCLUDE(algorithm/iterator/numeric/type_traits)) and inline integer ceil-div, plus AMD Modifications SPDX line; memory.h keeps _THRUST_STD_INCLUDE(limits); reduce_intervals.h uses _THRUST_STD_INCLUDE(cstdint). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 764fd88c08b
- [X] 88b95bb9a2c Inline the Thrust ADL layer (#6377) [SCOPE: HEADER]
  HIP counterpart (system/hip): covered via __THRUST_DEVICE_SYSTEM_ROOT (resolves to thrust/system/hip), so the new __THRUST_{HOST,DEVICE}_SYSTEM_ALGORITH_{,DETAIL_}HEADER_INCLUDE macros pull in hip/detail headers directly; verified every referenced system/hip/detail/*.h and system/cpp/detail/*.h exists. Added thrust/system/hip lines to all upstream `#if 0` build-system hint blocks. git rm'd all 43 thrust/system/detail/adl/*.h; no remaining includers. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 6a4ae1580ed
- [X] a8c74132b74 Drop unused file to detect CUDA archs (#6374) [SCOPE: CMAKE]
  N/A: thrust/cmake/detect_compute_archs.cu has no local counterpart (rocThrust cmake/ has no CUDA arch detection). HIP counterpart (system/hip): none needed. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 857b08611c0
- [X] e2fcf893086 Make some if constexpr (#6382) [SCOPE: HEADER]
  Took upstream: cuda set_operations.h / sort.h `if` -> `if constexpr` plus [[maybe_unused]] items_first.
  HIP counterpart (system/hip): no change; hip set_operations.h/sort.h have no matching runtime `if (HAS_VALUES::value)` / `if (SORT_ITEMS::value)` / `if (!is_contiguous_iterator...)` branches (rocPRIM-based dispatch). test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 44ce4e405dc
- [X] c2f0b68245e Refactor Thrust TBB system headers (#6394) [SCOPE: HEADER]
  Took upstream TBB refactor: nested namespaces, SPDX headers, inlined copy/for_each/merge/partition/reduce/scan/sort .inl into their .h and git rm'd the 7 .inl files (no remaining includers). Local .inl edits were spelling-only (::cuda::std:: -> _THRUST_STD::, <cuda/std/type_traits> -> _THRUST_STD_INCLUDE(type_traits)) and are preserved in the inlined headers. Stats match upstream (61 files).
  HIP counterpart (system/hip): no change; refactor is TBB-backend-only. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 5b0e2305076
- [X] dc30031642b Fix wrong namespace (#6395) [SCOPE: HEADER]
  Took upstream: tbb/detail/remove.h namespace/include fixed from omp to tbb.
  HIP counterpart (system/hip): no change; TBB-only fix. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: bac3a8ff69a
- [X] e11761cc1b8 Fix wrongly rewritten license headers in Thrust OMP/TBB backend (#6406) [SCOPE: HEADER]
  Took upstream: restored original NVIDIA Corporation / Apache-2.0 license headers in OMP/TBB backend files (102 files, stats match upstream). omp/detail/scan.h conflict: took upstream header, kept the AMD Modifications SPDX line.
  HIP counterpart (system/hip): no change; OMP/TBB-only. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 8fde352f9c0
- [X] 865e3d6429f Inline remaining `*.inl` files in tbb and seq backends (#6437) [SCOPE: HEADER]
  Took upstream: inlined sequential copy/merge/sort/stable_merge_sort/stable_primitive_sort/stable_radix_sort .inl and tbb copy_if/memory/reduce_by_key/remove/unique/unique_by_key .inl into their headers; git rm'd the 12 .inl files (no remaining includers).
  Preserved local .inl edits in the inlined headers: THRUST_HOST_DEVICE/THRUST_EXEC_CHECK_DISABLE/_THRUST_STD spellings; sequential/copy.h keeps _THRUST_STD::{true,false}_type; sequential/sort.h keeps the conjunction.h wrapper include, `<type_traits>` fallback and _THRUST_IF_TARGET/_THRUST_IS_HOST (nv/target dropped); stable_merge_sort.h keeps the _THRUST_HAS_DEVICE_SYSTEM_STD-guarded includes (algorithm_wrapper.h fallback) and _THRUST_IF_TARGET(_THRUST_IS_DEVICE, "CUDA or HIP threads"); stable_primitive_sort.h keeps the `<type_traits>` fallback; AMD Modifications lines carried from sort.inl/stable_merge_sort.inl. tbb reduce_by_key.h: libcu++-internal includes guarded by _THRUST_HAS_DEVICE_SYSTEM_STD with algorithm_wrapper.h/<type_traits> fallback (compatibility without libhipcxx).
  HIP counterpart (system/hip): no change; sequential backend is shared and picked up by HIP through the generic dispatch, TBB is host-only. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: ab00525afc9
- [X] 79060c84eb6 Fix offset_iterator tests (#6436) [SCOPE: TEST]
  Took upstream: testing/cuda/offset_iterator.cu device_only_iterator gains operator+ and operator- now returns difference_type (CUDA-only test, raw spelling kept).
  HIP counterpart (system/hip): no change; test-only fix. test/: test_offset_iterator.cpp has no device_only_iterator, no change. Not compiled, tests not run (per human)
  - Local commit: c6b601bd94e
- [X] 106665c64f4 Use SPDX license identifiers in CUB (#6441) [SCOPE: HEADER,BENCH]
  Took upstream SPDX license headers for cuda/detail/{merge.h,core/triple_chevron_launch.h} and the primbench-style benchmark/bench/ counterparts (adjacent_difference, copy, fill, for_each, inner_product, merge, partition, reduce, scan, shuffle, sort, tabulate, transform_reduce, unique, vectorized_search); AMD Modifications lines preserved as SPDX-FileCopyrightText lines. Kept ours for AMD-only MIT-licensed set_operations benchmarks. NOLOCAL: set_operations/base.cuh, by_key.cuh, transform/babelstream.cu, transform/fib.cu, vectorized_search/base.cu. CUB part not applicable (hipCUB is separate).
  HIP counterpart (system/hip): no change; license-only. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 899e186bb7f
- [X] 31dd1d76e72 [clang-format] WrapNamespaceBodyWithEmptyLines: Never (#6439) [SCOPE: HEADER,TEST]
  Took upstream whitespace-only change (empty lines removed after namespace open / before close) across 324 thrust/ and testing/ files. Not ported: the `WrapNamespaceBodyWithEmptyLines: Never` option for rocthrust/.clang-format, since the monorepo pre-commit pins clang-format v18.1.4 and that key needs v20+. Conflicts: kept ours (whitespace trimmed) in testing/unittest/{assertions,random}.h, mr/{fancy_pointer_resource,polymorphic_adaptor}.h doc comments, cuda/detail/sort.h TODO, type_traits/is_operator_{plus,less_or_greater}_function_object.h `#endif`, and iterator_facade_category.h (::internal::If); kept ours (no addition) for cuda/detail/dispatch.h is_negative (collides with integer_math.h, see 5303f65de2d) and cuda/detail/reduce.h public reduce_into overloads (absent locally).
  HIP counterpart (system/hip): no change; upstream does not touch system/hip and the whitespace style change is not mirrored there. test/ counterpart: none (whitespace-only style change, not mirrored into gtest test/*.cpp). testing/: whitespace in unittest/ and tests merged. Not compiled, tests not run (per human)
  - Local commit: 42c97ed3778
- [X] 93e1f87c252 Ensure that `detect_wrong_difference` is a valid output iterator (#6450) [SCOPE: TEST]
  Took upstream: testing/cpp/adjacent_difference.cu detect_wrong_difference gains output-iterator typedefs (iterator_category spelled _THRUST_STD::output_iterator_tag).
  HIP counterpart (system/hip): no change; test-only. test/: test_adjacent_difference.cpp has no detect_wrong_difference, no change. Not compiled, tests not run (per human)
  - Local commit: d7c4653fc34
- [X] e6ad5079192 Nested namespace resolve for thrust & libcudacxx (#6465) [SCOPE: HEADER]
  Took upstream: nested namespace definitions (namespace a::b) in 40 thrust headers (complex math, functional, random detail, cpp/cuda/omp system headers); merged cleanly. libcudacxx part not applicable.
  HIP counterpart (system/hip): no change; style-only, the corresponding system/hip headers keep their existing namespace layout. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 0b59b09e572
- [X] d89402658b2 Various CMake cleanups. (#6346) [SCOPE: TEST,EXAMPLE,CMAKE]
  N/A: upstream adds cccl_configure_target(... DIALECT ...) calls to CCCL's ThrustHeaderTesting.cmake, examples/, testing/ and testing/unittest/ CMake. rocThrust uses its own CMake (no cccl_configure_target / thrust_add_test definitions locally; ThrustHeaderTesting.cmake absent); the vestigial testing/unittest/CMakeLists.txt is left unchanged.
  HIP counterpart (system/hip): none needed. test/: no counterpart (test/CMakeLists.txt is rocThrust-native). testing/: no change. Not compiled, tests not run (per human)
  - Local commit: fd53b07b864
- [X] cb07e970a01 Remove old C++ version checks (#6494) [SCOPE: HEADER] ⚠ touches `thrust/detail/config/cpp_dialect.h`
  Took upstream comment fix in thrust/detail/config/cpp_dialect.h: `#endif // THRUST_STD_VER < 2017` (local spelling of _CCCL_STD_VER). The rest of the upstream commit (old C++ version checks) is in CUB/libcudacxx only.
  HIP counterpart (system/hip): no change. test/: no counterpart. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: 279e1e7302c
- [X] bcb2bcb106d [clang-format] KeepEmptyLines only at EOF (#6440) [SCOPE: HEADER,TEST]
  Took upstream: removed leading empty line in system/detail/generic/select_system.h. detail/pointer.h already had no leading blank line; testing/offset_iterator.cu and zip_function.h have local AMD license headers without a leading blank line, kept ours. Not ported: .clang-format KeepEmptyLines block (monorepo pins clang-format v18.1.4; KeepEmptyLines needs v19+).
  HIP counterpart (system/hip): no change; whitespace-only. test/ counterpart: test/test_offset_iterator.cpp unchanged (whitespace-only). testing/: offset_iterator.cu kept ours. Not compiled, tests not run (per human)
  - Local commit: 9f57de356b0
- [X] a3cf1182949 Replace custom equal_to functors by _1 (#6515) [SCOPE: TEST]
  Took upstream: testing/find.cu and testing/cuda/find.cu replace the custom equal_to/not_equal_to/less_than value predicates with thrust::placeholders::_1 expressions (find.cu conflict: dropped the structs, kept local _THRUST_STD_INCLUDE(cstdint) placement).
  HIP counterpart (system/hip): no change; test-only. test/: mirrored in test/test_find.cpp: removed the three *_value_pred structs, added file-scope `using thrust::placeholders::_1;` and replaced uses with `_1 ==/!=/< T(x)` (same conversion semantics). Not compiled, tests not run (per human)
  - Local commit: 3bf99859680
- [X] 24deb145aec Cleanup `thrust::complex` math includes and functions (#6546) [SCOPE: HEADER]
  Took upstream complex math cleanup: deleted detail/complex/c99math.h (git rm; no remaining includers) and replaced unqualified/ADL math calls with qualified _THRUST_STD:: calls across detail/complex/*.h; removed the abs float/double specializations in arithmetic.h. Compatibility adjustments: upstream's libcu++-internal includes (_THRUST_STD_INCLUDE(__cmath/...), __type_traits/common_type.h) replaced by a single _THRUST_STD_INCLUDE(cmath) per file (works with ::std when libhipcxx is absent); arithmetic.h keeps _THRUST_STD_INCLUDE(cfloat) (FLT_MIN/DBL_MIN) and <thrust/detail/type_traits.h>; cpow.h keeps <thrust/detail/type_traits.h>; kept local ::internal::common_type_t instead of _THRUST_STD::common_type_t; complex.h kept ours (_THRUST_STD_INCLUDE(cmath)); csinhf.h kept cmath include; catrig.h additionally qualifies the remaining `isnan(rx)` as _THRUST_STD::isnan since c99math's using-declarations are gone.
  HIP counterpart (system/hip): no change; generic complex headers are used by HIP directly. test/: test_complex_various.cpp references thrust::detail::complex::hypot/hypotf only under `_MSC_VER <= 1500 && !defined(__clang__)` (dead for hipcc), no change. testing/: no counterpart. Not compiled, tests not run (per human)
  - Local commit: e1ffc7b63d8
- [X] 4b10fd26255 Cleanup includes in thrust (#6547) [SCOPE: HEADER]
  HIP counterpart (system/hip): none needed; system/cuda include cleanup taken as upstream. Generic headers: took trait modernization (is_x_v, enable_if_t, add_lvalue_reference_t) and binary_search_less -> _THRUST_STD::less<> (added _THRUST_STD_INCLUDE(functional)); kept local include blocks instead of upstream's granular libcu++-internal includes (would break non-libhipcxx builds); kept ::internal::is_integral/::internal::If wrappers and dual less list in cuda sort.h. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: 9b7abf25a94
- [X] 369f2f72eca Fix merge conflicts from dropping headers (#6563) [SCOPE: HEADER]
  HIP counterpart (system/hip): none. N/A: upstream adds granular libcu++ __cmath/* includes to catrig.h/catrigf.h; local already includes the _THRUST_STD_INCLUDE(cmath) umbrella (from 24deb145aec port), which is compatible with non-libhipcxx builds. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: c308d88e449
- [X] 2845628f12e Add gersemi CMake formatter (#6557) [SCOPE: TEST,EXAMPLE,BENCH,CMAKE] ⚠ touches `CMakeLists.txt`
  HIP counterpart (system/hip): none. N/A: gersemi formatting-only reflow of CMake files; local CMakeLists.txt/*.cmake are rocThrust-owned (HIP build) and the monorepo does not use gersemi, so no reformat ported; no functional change. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: 5510a5eabe8
- [X] 3a4892e1fb9 [Thrust]: New "sum rows" and "sum columns" examples (#4462) [SCOPE: TEST,EXAMPLE,OTHER]
  HIP counterpart (system/hip): none. Ported rewritten examples/sum_rows.cu and new examples/sum_columns.cu (registered in examples/CMakeLists.txt), adapted for non-libhipcxx compatibility: cuda::std::mdspan replaced by row-major raw pointer indexing, cuda::counting_iterator by thrust::counting_iterator<int>; added missing permutation_iterator include; AMD MIT header kept on sum_rows. CMake sidecars (sum_*.cmake, GCC7/nvcc-only), sum_columns filecheck and gersemi-only address_stability.cmake reflow not ported (local build has no per-example sidecar/FileCheck support). test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: 5ef6089282f
- [X] 06f2025d169 Replace enum by static constexpr in CUB/Thrust (#6480) [SCOPE: HEADER]
  HIP counterpart (system/hip): none; system/hip has no anonymous-enum constants of this form. Ported enum -> static constexpr in system/cuda (agent_launcher.h, core/util.h, reduce.h, reduce_by_key.h, set_operations.h); util.h conflict resolved to static constexpr bool value while keeping local ::cuda::std::true_type/false_type. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: e4cf20cf111
- [X] 8b263079605 Make Thrust/CUB ABI namespace resilient against user-defined macros (#6564) [SCOPE: HEADER]
  HIP counterpart (system/hip): mirrored in namespace.h HIP branch: ABI namespace now THRUST_PP_CAT(THRUST_PP_CAT(_V_, THRUST_VERSION), THRUST_PP_SPLICE_WITH(_, _R, ROCTHRUST_VERSION)) so user macros named THRUST/NS cannot break it (namespace name changes from THRUST_<ver>_<rocver>_NS to _V_<ver>_R_<rocver>; ABI-only, inline namespace). CUDA branches take upstream. cub/libcudacxx parts not applicable. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: 8162b02e909
- [X] 91434d76ab3 Replace uses of `thrust::pair` with `cuda::std::pair` (#6616) [SCOPE: HEADER,TEST,EXAMPLE,BENCH]
  HIP counterpart (system/hip): none; system/hip keeps thrust::pair. Ported only system/cuda (extrema, merge, mismatch, partition, reduce_by_key, set_operations, temporary_buffer, unique_by_key) and CUDA-only testing/cuda tests, where thrust::pair is an alias of cuda::std::pair. Generic/public headers, sequential/omp/tbb backends, top-level tests, examples and benchmarks keep thrust::pair: in the non-libhipcxx fallback thrust::pair is a distinct struct (pair.h), so switching signatures to _THRUST_STD::pair (std::pair) would break the fallback and diverge from system/hip return types (compatibility over upstream parity). Remaining upstream changes in those files were formatting-only. test/ counterpart: none (test/ uses thrust::pair; unchanged). Not compiled, tests not run (per human)
  - Local commit: 5555313e802
- [X] fba9f834103 Publicly expose `<cuda/std/algorithm>` (#3741) [SCOPE: TEST]
  HIP counterpart (system/hip): none. testing/counting_iterator.cu now includes _THRUST_STD_INCLUDE(algorithm) instead of the private __algorithm_ header; testing/zip_iterator_sort.cu qualifies thrust::stable_sort (avoids ADL ambiguity with now-public cuda::std algorithms). test/ counterpart: mirrored in test/test_counting_iterator.cpp (algorithm include) and test/test_zip_iterator_sort.cpp (thrust::stable_sort). Not compiled, tests not run (per human)
  - Local commit: 3c05a78861b
- [X] b934754d607 Refactor cub::detail::AliasTemporaries (#6617) [SCOPE: HEADER]
  HIP counterpart (system/hip): none; system/hip uses rocPRIM temp storage, not cub AliasTemporaries. Ported call-site updates in system/cuda (copy_if, core/util, extrema, partition, reduce_by_key, unique, unique_by_key). test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: 461edd016d8
- [X] 99976dcd665 MVP for disabling nvtx ranges for `thrust::seq` (#6415) [SCOPE: HEADER]
  HIP counterpart (system/hip): none. N/A: adds nvtx_policy.h and _CCCL_NVTX_RANGE_SCOPE_IF to algorithm entry points to skip NVTX for thrust::seq; rocThrust carries no NVTX ranges in generic entry points (kept ours, nvtx_policy.h not added). test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: 8f48aa3616e
- [X] 2f82c4d1be1 Port thrust complex.cu tests to catch2_test_complex.cu (#6625) [SCOPE: TEST]
  HIP counterpart (system/hip): none. N/A: upstream moves testing/complex.cu to Catch2 testing/catch2_test_complex.cu; rocThrust has no Catch2 harness, so the catch2 file is not added and legacy testing/complex.cu is kept (same coverage). test/ counterpart: test/test_complex.cpp already covers these cases; unchanged. Not compiled, tests not run (per human)
  - Local commit: c914c9efc71
- [X] 26c79caf60e fix omp scan bug (#6560) (#6704) [SCOPE: HEADER,TEST]
  HIP counterpart (system/hip): none (rocPRIM scan unaffected). Ported OMP scan fix in system/omp/detail/scan.h: blocks reduce with their first element as init (no accum_t{} identity assumption), no-init inclusive path seeds from block_sums[0] and thread 0 scans without prefix, serial fallback when n < max(threshold, num_threads); _CCCL_ASSERT dropped (not available locally; replaced by a comment). testing/scan.cu and testing/transform_scan.cu gain the new tests (THRUST_HOST_DEVICE spelling; kept local include guards, added <numeric>). test/ counterpart: mirrored in test/test_scan.cpp (Int with friend ops, TestScanWithUserDefinedTypeAndInit replacing TestInclusiveScanWithUserDefinedType, TestScanBug6317 using get_random_data, TestScanEdgeCases) and test/test_transform_scan.cpp (TransformInt, TestTransformScanEdgeCases). Not compiled, tests not run (per human)
  - Local commit: b30d2c07630
- [X] 2f529edf653 Ensure that we properly warn about device lambdas that need to query the return type (#6765) (#6782) [SCOPE: HEADER]
  HIP counterpart (system/hip): none. transform_iterator_reference now uses _THRUST_STD::invoke_result_t<UnaryFunc&, ...> (also for functional::actor) so the device-lambda return-type diagnostic fires; kept local include block (type_traits umbrella provides invoke_result_t; upstream's granular __functional/__type_traits includes not added). libcudacxx invoke.h part not applicable. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: 0988f1197e8
- [X] 783ebe64a0e CMake backports for 3.2 (#6850) [SCOPE: TEST,EXAMPLE,BENCH,CMAKE] ⚠ touches `CMakeLists.txt`
  HIP counterpart (system/hip): none. N/A: upstream CCCL CMake backports (Thrust*.cmake, DetectSupportedStandards, ThrustRunTest, gersemi-formatted CMakeLists, async_reduce.cmake removal); rocThrust's CMake (top-level, benchmark, examples, testing) is independent of upstream's and these files are absent or AMD-owned locally. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: fa12cc643c2
- [X] 2c7c41e2026 [thrust] Ignore CUDA free errors in thrust memory resource (#7002) (#7033) [SCOPE: HEADER]
  HIP counterpart (system/hip): mirrored in system/hip/memory_resource.h: do_deallocate no longer throws on hipFree errors (status discarded with [[maybe_unused]]), matching system/cuda/memory_resource.h. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: 6a3c654cb90
- [X] 7407ca152b0 Use cudaMemcpyDefault for trivial copies (#7006) (#7134) [SCOPE: HEADER,TEST]
  HIP counterpart (system/hip): mirrored in system/hip/detail/util.h: trivial_copy_from_device/trivial_copy_to_device use hipMemcpyDefault (both hipMemcpyWithStream and legacy hipMemcpyAsync paths) so the runtime picks the direction when a device-accessible pointer is host memory. system/cuda/detail/util.h and CUDA-only testing/cuda/trivial_copy_memcpy_default.cu taken as upstream. test/ counterpart: added TestTrivialCopyFromDeviceHostSource/TestTrivialCopyToDeviceHostDest to test/test_copy.cpp. Not compiled, tests not run (per human)
  - Local commit: 2edab5b61d2
- [X] 5281dcaefdc Align local vector storage arrays in vec transform (#7162) (#7163) [SCOPE: HEADER]
  HIP counterpart (system/hip): none; the Alignment parameter only serves CUB's vectorized transform (cub part not applicable), and system/hip uses rocPRIM transform. Ported the uninitialized_array Alignment template parameter in system/cuda/detail/core/util.h. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: b0f307773ad
- [X] b85d41e36d4 Fixes for thrust::shuffle (#7172) (#7189) [SCOPE: HEADER,TEST]
  HIP counterpart (system/hip): none (generic shuffle/random_bijection shared by all backends). Took upstream's new Feistel bijection (L/R split, size() = 1 << (L_bits+R_bits), min 256, nearest_power_of_two/mulhilo/get_cipher_bits/round_state removed) and generic/shuffle.inl now uses bijection.size(). Replaced cuda::std::bit_width with a portable bit-count loop (C++20 std::bit_width unavailable without libhipcxx); kept local umbrella includes instead of granular __type_traits/__bit includes. testing/shuffle.cu taken as upstream (Cephes and EvenSpacing/EvenDistribution tests removed, z-test key-position test, chi-squared critical value 145.461) with cuda::std:: -> _THRUST_STD:: and _THRUST_STD_INCLUDE(algorithm/cmath). test/ counterpart: mirrored in test/test_shuffle.cpp (size_t(256) bound, FeistelBijectionLength 345->512/256->256/1->256, inverse_erf z-test, critical value 145.461, Cephes/EvenSpacing/EvenDistribution removed, <cmath> added). Not compiled, tests not run (per human)
  - Local commit: 6c4b6a6fc1c
- [X] 64bf7077610 Bump branch/3.2.x to 3.2.1. (#7329) [SCOPE: HEADER]
  HIP counterpart (system/hip): none. N/A: upstream version bump only touches thrust/version.h / CCCL version files; rocThrust versioning (thrust/version.h, CHANGELOG.md) is rocThrust-owned and not modified by the sync. test/ counterpart: none. Not compiled, tests not run (per human)
  - Local commit: d2976ad27c2
