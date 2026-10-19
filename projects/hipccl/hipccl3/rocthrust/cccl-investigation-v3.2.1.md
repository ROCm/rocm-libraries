# CCCL → rocThrust Sync Investigation: v3.1.0 → v3.2.1

What changes in rocThrust when syncing CCCL v3.1.0 → v3.2.1, and
what work that implies. No code has been ported yet.

## Summary

- **JIRA ticket**: none
- **Base branch**: `users/matyas-streamhpc/hipccl/rocthrust_cccl31` (the unmerged CCCL 3.1 sync, 86 commits ahead of `origin/develop`)
- **Baseline**: `v3.1.0` (confirmed by the human). `libcxx.h` on the base branch still requires libcu++/libhipcxx 3.0; this sync raises it to 3.2.
- **Target**: `v3.2.1` — tags covered: `v3.1.1 v3.1.2 v3.1.3 v3.1.4 v3.2.0 v3.2.1`
- **Upstream commits to port**: 136 touching `thrust/` (810 files changed). 8 are already on the base branch via the 3.1.0 backports, and 2 more are partly there (see §1).
- **Maintenance-branch drift**: `branch/3.2.x` has one commit past v3.2.1 that touches `thrust/`, [`21b3532227f`](https://github.com/NVIDIA/cccl/commit/21b3532227f) "Bump branch/3.2.x to 3.2.2" ([PR #7651](https://github.com/NVIDIA/cccl/pull/7651)). It's a version bump for an untagged release and is deliberately left out.

v3.2 adds little new public Thrust API. It is mostly a large internal restructuring of Thrust. The ADL dispatch layer is inlined and `system/detail/adl/` is deleted. Execution policies, the allocator internals, and the OMP/TBB/sequential backends are rewritten, and most `*.inl` files are folded into headers. `thrust::pair` and other Thrust facilities are replaced internally by `cuda::std::` and `cuda::` equivalents. On top of that come repo-wide clang-format, SPDX-license and include-cleanup sweeps. The CUDA backend moves several algorithms onto `cub::DeviceTransform`; that part is NVIDIA-only for rocThrust's HIP/rocPRIM backend but must still land in `thrust/system/cuda/`. The main risk is the dependency on libcu++ 3.2 features. These include `cuda::zip_iterator`, `cuda::zip_transform_iterator`, `cuda::transform_input_output_iterator` and new `_CCCL_*` macros. The bundled libhipcxx is still at CCCL 3.0.2 and has none of them. libhipcxx will be bumped to at least 3.2, so these commits are ported directly and `libcxx.h` requires 3.2. The genuinely new features are OMP/TBB parallel scan and the new iterator interop. The fixes that matter for HIP are the memcpy-kind, memory-resource free and `shuffle` fixes.

---

## 1. New features

v3.1.1–v3.1.4 were cut from `branch/3.1.x`. Their only Thrust content is three fixes (`<new>` for clang-cuda, `offset_iterator` tests, `detect_wrong_difference`), and the main-line originals of all three are in this range, listed under v3.2.0. v3.2.1's only Thrust change is the version bump. Everything below is v3.2.0.

#### Already on the base branch

The 3.1.0 backports already brought in these commits, so they should be ticked N/A after checking that nothing is missing:
- [`7825c40c8a6`](https://github.com/NVIDIA/cccl/commit/7825c40c8a6) PDL macro ([PR #5316](https://github.com/NVIDIA/cccl/pull/5316))
- [`daf84b3d7f5`](https://github.com/NVIDIA/cccl/commit/daf84b3d7f5) and [`2e8d11a0e36`](https://github.com/NVIDIA/cccl/commit/2e8d11a0e36) `unique` via `DeviceSelect::Unique` ([PR #5396](https://github.com/NVIDIA/cccl/pull/5396), [PR #5566](https://github.com/NVIDIA/cccl/pull/5566))
- [`d7fbce513a7`](https://github.com/NVIDIA/cccl/commit/d7fbce513a7) `unique` with non-const `equality_op` ([PR #5652](https://github.com/NVIDIA/cccl/pull/5652))
- [`d046198a7ee`](https://github.com/NVIDIA/cccl/commit/d046198a7ee) `malloc<void>` ([PR #5698](https://github.com/NVIDIA/cccl/pull/5698))
- [`0b700f32b1e`](https://github.com/NVIDIA/cccl/commit/0b700f32b1e) extended-float forward declarations ([PR #5846](https://github.com/NVIDIA/cccl/pull/5846))
- [`9fbba017026`](https://github.com/NVIDIA/cccl/commit/9fbba017026) `device_reference` nullptr ([PR #4226](https://github.com/NVIDIA/cccl/pull/4226))
- [`57b16c2fc64`](https://github.com/NVIDIA/cccl/commit/57b16c2fc64) `__clz` ([PR #6099](https://github.com/NVIDIA/cccl/pull/6099))

Two commits were only partly backported by the 3.1 "iterator fixes" backport ([PR #5977](https://github.com/NVIDIA/cccl/pull/5977)). Only the remainder of each needs porting:
- [`e2dda739257`](https://github.com/NVIDIA/cccl/commit/e2dda739257) ([PR #5928](https://github.com/NVIDIA/cccl/pull/5928)): roughly 6 of 10 changed lines are already present.
- [`9e73f14e8f1`](https://github.com/NVIDIA/cccl/commit/9e73f14e8f1) ([PR #5929](https://github.com/NVIDIA/cccl/pull/5929)): roughly 16 of 24 changed lines are already present.

### v3.2.0

#### New API
- **`cuda::zip_iterator` / `cuda::zip_function` interop** ([PR #5429](https://github.com/NVIDIA/cccl/pull/5429), [PR #6039](https://github.com/NVIDIA/cccl/pull/6039)) — Thrust specializes `iterator_system`/`iterator_traversal` for `cuda::zip_iterator`, and `thrust::transform` unwraps `cuda::zip_iterator`/`zip_function`. **To do:** add `_THRUST_LIBCXX::` specializations under `_THRUST_HAS_DEVICE_SYSTEM_STD`, as the 3.1 branch did for `permutation_iterator`. The types come from libhipcxx ≥ 3.2.
- **`cuda::transform_input_output_iterator` interop** ([PR #5113](https://github.com/NVIDIA/cccl/pull/5113)) — the same kind of trait specializations, plus tests. **To do:** port the same way as the zip iterators above.
- **`cuda::zip_transform_iterator` interop** ([PR #5982](https://github.com/NVIDIA/cccl/pull/5982)) — trait specializations, `thrust::copy` support, and a new benchmark. **To do:** port the same way; the benchmark is covered in §3.
- **Parallel scan for the OMP and TBB backends** ([PR #6178](https://github.com/NVIDIA/cccl/pull/6178), with the follow-up fix [PR #6704](https://github.com/NVIDIA/cccl/pull/6704)) — `inclusive_scan`/`exclusive_scan`/`*_by_key` now run in parallel under OMP and TBB. **To do:** port into `system/omp/` and `system/tbb/`, after the OMP/TBB restructuring in §4.
- **`transform_n` / `transform_if_n` overloads and a `DeviceTransform`-based `transform_if` in the CUDA system** ([PR #5238](https://github.com/NVIDIA/cccl/pull/5238), [PR #5198](https://github.com/NVIDIA/cccl/pull/5198)). **To do:** port into `system/cuda/`. Check whether `system/hip/detail/transform.h` needs matching overloads; the generic `transform_n` is already on the base branch.
- **New `sum_columns` example** ([PR #4462](https://github.com/NVIDIA/cccl/pull/4462)) — rocThrust already has `sum_rows.cu`. **To do:** add `examples/sum_columns.cu` and update `sum_rows.cu`.

#### Behavior changes
- **CUDA algorithms routed through `cub::DeviceTransform`**: `fill`/`fill_n` ([PR #5805](https://github.com/NVIDIA/cccl/pull/5805)), `uninitialized_fill` ([PR #5813](https://github.com/NVIDIA/cccl/pull/5813)), `generate` ([PR #5807](https://github.com/NVIDIA/cccl/pull/5807)), `tabulate` ([PR #6012](https://github.com/NVIDIA/cccl/pull/6012)), `scatter_if` ([PR #5952](https://github.com/NVIDIA/cccl/pull/5952)), and non-trivial device-to-device `copy` ([PR #5954](https://github.com/NVIDIA/cccl/pull/5954)). In addition, `uninitialized_copy` now uses `copy` ([PR #5181](https://github.com/NVIDIA/cccl/pull/5181)), and `copy` ([PR #5491](https://github.com/NVIDIA/cccl/pull/5491)), `find` ([PR #6019](https://github.com/NVIDIA/cccl/pull/6019)), `mismatch` ([PR #6018](https://github.com/NVIDIA/cccl/pull/6018)) and `iter_swap` ([PR #5843](https://github.com/NVIDIA/cccl/pull/5843)) are rewritten. **To do:** port into `system/cuda/` only. The HIP disposition is "no change: performance-only", unless a commit also changes behavior.
- **Trivial copies use `cudaMemcpyDefault`** ([PR #7006](https://github.com/NVIDIA/cccl/pull/7006) via [PR #7134](https://github.com/NVIDIA/cccl/pull/7134)) — this makes copies correct when host and device pointers are swapped or the memory is managed. **To do:** switch `system/hip/detail/util.h` (lines 201 and 226) to `hipMemcpyDefault`, and port the new test as a HIP test.
- **`thrust::shuffle` bijection reworked to match libcu++'s Feistel bijection** ([PR #7172](https://github.com/NVIDIA/cccl/pull/7172) via [PR #7189](https://github.com/NVIDIA/cccl/pull/7189)) — the shuffled output sequence changes. **To do:** port `random_bijection.h` and the rewritten `shuffle.cu`, and update `test/test_shuffle.cpp`.
- **`device_delete` bypasses the allocator and supports polymorphic types; allocator and `device_[new|delete|free]` internals rewritten** ([PR #6134](https://github.com/NVIDIA/cccl/pull/6134), [PR #6136](https://github.com/NVIDIA/cccl/pull/6136), [PR #6198](https://github.com/NVIDIA/cccl/pull/6198), [PR #6140](https://github.com/NVIDIA/cccl/pull/6140)). **To do:** port; 7 of the allocator `.inl` files being deleted carry rocThrust edits (see §4).
- **`transform_iterator` traversal now comes from the base iterator's traversal, not its category** ([PR #5883](https://github.com/NVIDIA/cccl/pull/5883)). **To do:** port with its tests.
- **Internal uses of `thrust::pair`, `thrust::reverse_iterator`, `thrust::tabulate_output_iterator` and Thrust's duplicated math replaced by `cuda::std::` / `cuda::`** ([PR #6616](https://github.com/NVIDIA/cccl/pull/6616), [PR #5616](https://github.com/NVIDIA/cccl/pull/5616), [PR #6001](https://github.com/NVIDIA/cccl/pull/6001), [PR #6188](https://github.com/NVIDIA/cccl/pull/6188)). **To do:** translate to `_THRUST_STD::` / `_THRUST_LIBCXX::`. The non-libhipcxx fallback build needs care (see §4).
- **NVTX ranges disabled for `thrust::seq`** ([PR #6415](https://github.com/NVIDIA/cccl/pull/6415)) — adds `thrust/detail/nvtx_policy.h`. **To do:** rocThrust has no NVTX ranges; add the header for parity and drop the `_CCCL_NVTX_RANGE_SCOPE_IF` call sites, or tick N/A.
- **ABI inline-namespace spelling hardened against user macros** ([PR #6564](https://github.com/NVIDIA/cccl/pull/6564)). **To do:** apply the same `_CCCL_PP_CAT` construction to rocThrust's own `ROCTHRUST_VERSION` branch in `detail/config/namespace.h`.

#### Deprecations
- **`THRUST_HOST_FUNCTION`, `THRUST_DEVICE_FUNCTION` and `THRUST_FUNCTION`** ([PR #5972](https://github.com/NVIDIA/cccl/pull/5972)) — marked deprecated in `system/cuda/config.h`, with internal uses replaced by `_CCCL_API`/`_CCCL_HOST_API`. **To do:** port to `system/cuda/`. Decide whether `THRUST_HIP_FUNCTION` in `system/hip/config.h` follows suit.
- **`_LIBCUDACXX_DEPRECATED` replaced by `CCCL_DEPRECATED`** ([PR #6024](https://github.com/NVIDIA/cccl/pull/6024)). **To do:** port. The macro comes from libhipcxx ≥ 3.2.

#### Fixes
- **Uninitialized read in `uninitialized_copy_n`** ([PR #5811](https://github.com/NVIDIA/cccl/pull/5811)). **To do:** port, and check `system/hip/detail/uninitialized_copy.h` for the same pattern.
- **Missing stream synchronization in `cuda_cub::generate`** ([PR #5889](https://github.com/NVIDIA/cccl/pull/5889)). **To do:** port, and check `system/hip/detail/generate.h`.
- **`iterator_category_to_system` wrong for device iterator tags** ([PR #5880](https://github.com/NVIDIA/cccl/pull/5880)). **To do:** port with its tests.
- **CUDA memory resource no longer throws on free** ([PR #7002](https://github.com/NVIDIA/cccl/pull/7002) via [PR #7033](https://github.com/NVIDIA/cccl/pull/7033)) — avoids throwing from destructors and at process teardown. **To do:** apply the same change to `system/hip/memory_resource.h:79` ("HIP free failed").
- **`thrust::ccosh` reformulated** ([PR #6200](https://github.com/NVIDIA/cccl/pull/6200)) — works around a GCC 14 segfault on Arm. **To do:** port.
- **`<new>` included for clang-cuda** ([PR #6310](https://github.com/NVIDIA/cccl/pull/6310)), **`offset_iterator` tests** ([PR #6436](https://github.com/NVIDIA/cccl/pull/6436)), **`detect_wrong_difference` output iterator** ([PR #6450](https://github.com/NVIDIA/cccl/pull/6450)), and **OOM tolerated in the large `unique` test** ([PR #6304](https://github.com/NVIDIA/cccl/pull/6304)). **To do:** port.

#### NVIDIA-only
- **UBLKCP in copy/scatter_if** ([PR #5954](https://github.com/NVIDIA/cccl/pull/5954), [PR #5952](https://github.com/NVIDIA/cccl/pull/5952)), **vectorized-transform storage alignment** ([PR #7163](https://github.com/NVIDIA/cccl/pull/7163)), **removal of `LoadIterator`** ([PR #5480](https://github.com/NVIDIA/cccl/pull/5480)), **trimmed CUDA core utilities** ([PR #5845](https://github.com/NVIDIA/cccl/pull/5845)), **nvhpc header wrappers moved to libcudacxx** ([PR #6167](https://github.com/NVIDIA/cccl/pull/6167)), **an NVHPC optimization workaround** ([PR #5416](https://github.com/NVIDIA/cccl/pull/5416)) and **device-lambda diagnostics** ([PR #6782](https://github.com/NVIDIA/cccl/pull/6782)). **To do:** port into `system/cuda/` for parity where the file exists; nothing is needed on the HIP side.

---

## 2. New CMake options / feature macros

None of the new upstream CMake options are Thrust-specific. `CCCL_ENABLE_*` and `CCCL_USE_LIBCXX` come from top-level CCCL CMake, and rocThrust has its own CMake, so they are N/A. All the new macros below are libcu++ 3.2 macros that rocThrust gets from libhipcxx ≥ 3.2 (the bundled 3.0.2 lacks them).

| Option / macro | Default | Introduced in | Defined/read in | Notes / rocThrust disposition |
|----------------|---------|----------------|------------------|--------------------------------|
| `_CCCL_HOST_STD_LIB(X)` | — | v3.2.0 ([PR #6244](https://github.com/NVIDIA/cccl/pull/6244)) | libcu++; read in a Thrust header | Port as-is; provided by libhipcxx ≥ 3.2 |
| `_CCCL_BEGIN/END_NAMESPACE_CUDA[_STD]` | — | v3.2.0 ([PR #5538](https://github.com/NVIDIA/cccl/pull/5538)) | libcu++; Thrust iterator headers and tests | Translate to `_THRUST_STD_NAMESPACE_BEGIN/END` |
| `_CCCL_HOST_API`, `_CCCL_DEVICE_API`, `_CCCL_API` | — | v3.2.0 ([PR #5953](https://github.com/NVIDIA/cccl/pull/5953), [PR #5972](https://github.com/NVIDIA/cccl/pull/5972)) | libcu++; `system/cuda/` | Already used in `system/cuda/` (3 uses); keep them CUDA-side only |
| `_CCCL_TEMPLATE` / `_CCCL_REQUIRES` | — | v3.2.0 ([PR #6188](https://github.com/NVIDIA/cccl/pull/6188)) | libcu++ concept emulation | Port as-is; provided by libhipcxx ≥ 3.2 |
| `_CCCL_PP_CAT` | — | v3.2.0 ([PR #6564](https://github.com/NVIDIA/cccl/pull/6564)) | `detail/config/namespace.h` | Use rocThrust's `THRUST_PP_*` equivalent |
| `CCCL_DEPRECATED` | — | v3.2.0 ([PR #6024](https://github.com/NVIDIA/cccl/pull/6024)) | libcu++ | Port as-is; provided by libhipcxx ≥ 3.2 |
| `_CCCL_NVTX_RANGE_SCOPE_IF` | — | v3.2.0 ([PR #6415](https://github.com/NVIDIA/cccl/pull/6415)) | libcu++ NVTX | N/A on AMD (no NVTX in rocThrust) |
| `CCCL_DETAIL_KERNEL_ATTRIBUTES` | — | v3.2.0 ([PR #5843](https://github.com/NVIDIA/cccl/pull/5843)) | `system/cuda/detail/iter_swap.h` | CUDA-only |
| `__THRUST_{DEVICE,HOST}_SYSTEM_ALGORITH[_DETAIL]_HEADER_INCLUDE(f)` | — | v3.2.0 ([PR #6377](https://github.com/NVIDIA/cccl/pull/6377)) | `detail/config/{device,host}_system.h` | Replaces the ADL layer; must resolve to `system/hip/` for the HIP device system (see §4) |
| `THRUST_INCLUDING_ALGORITHMS_HEADER`, `_CCCL_TRAIT` (removed) | — | v3.2.0 ([PR #5764](https://github.com/NVIDIA/cccl/pull/5764), [PR #5466](https://github.com/NVIDIA/cccl/pull/5466)) | Thrust headers | Remove; check `system/hip/` for remaining uses |

---

## 3. Test / benchmark coverage gaps

### Tests

| Feature / fix | Upstream test | rocThrust `testing/` / `test/` today | Action needed |
|---------------|---------------|--------------------------------------|---------------|
| `cuda::zip_iterator` / `transform_input_output_iterator` / `zip_transform_iterator` interop | `catch2_test_cuda_iterators.cu`, `zip_iterator.cu`, `transform_input_output_iterator.cu` | `testing/cuda_iterators.cu` + `test/test_libcxx_iterators.cpp` exist | Extend both, in Google Test |
| Iterator system/traversal propagation ([PR #5875](https://github.com/NVIDIA/cccl/pull/5875), [PR #5880](https://github.com/NVIDIA/cccl/pull/5880), [PR #5881](https://github.com/NVIDIA/cccl/pull/5881), [PR #5883](https://github.com/NVIDIA/cccl/pull/5883)) | new `catch2_test_iterator_traits.cu` | missing | Implement as a new Google Test, e.g. `test/test_iterator_traits.cpp`, plus a `testing/` counterpart |
| Trivial copy with `cudaMemcpyDefault` | new `cuda/trivial_copy_memcpy_default.cu` | missing | Add `testing/cuda/` copy and a HIP Google Test |
| `log2` in `xor_combine_engine_max` (replaces mpl math, [PR #5897](https://github.com/NVIDIA/cccl/pull/5897)) | new `xor_combine_engine_max.cu`; deletes `metaprogramming.cu` | `metaprogramming.cu` + `test_metaprogramming.cpp` exist | Add the new test to both suites; keep `metaprogramming` cases that still apply |
| OMP/TBB parallel scan | `omp/` scan tests ([PR #6704](https://github.com/NVIDIA/cccl/pull/6704)) | `testing/omp/` exists | Port |
| `device_delete` polymorphic / allocator bypass | `device_delete.cu` | both exist | Extend both |
| `reduce_by_key` | `reduce_by_key.cu` (+136 lines, [PR #6063](https://github.com/NVIDIA/cccl/pull/6063)) | both exist | Extend both |
| `shuffle` | `shuffle.cu` rewritten | both exist | Port and update the Google Test |
| Pair-, set-, sort- and iterator-test churn ([PR #6616](https://github.com/NVIDIA/cccl/pull/6616), [PR #5616](https://github.com/NVIDIA/cccl/pull/5616), [PR #4923](https://github.com/NVIDIA/cccl/pull/4923), [PR #6515](https://github.com/NVIDIA/cccl/pull/6515)) | ~60 `testing/*.cu` files | all exist | Port to `testing/`; mirror into `test/` where it changes behavior, not just spelling |

### Catch2 migration
- `complex.cu` → `catch2_test_complex.cu` ([PR #6625](https://github.com/NVIDIA/cccl/pull/6625)). rocThrust keeps `testing/complex.cu` and `test/test_complex.cpp`, and adds any new cases with Google Test.

### Benchmarks
- New upstream benchmarks that rocThrust lacks; port with Google Benchmark:
  - `bench/generate/basic.cu`
  - `bench/uninitialized_copy/basic.cu`
  - `bench/transform/fib.cu`
  - `bench/transform/if.cu`
  - `bench/transform/if_stencil.cu`
  - `bench/transform/zip_transform.cu`
- Renamed upstream: `bench/transform/basic.cu` → `bench/transform/babelstream.cu`. rocThrust still has `basic.cu`; rename it.
- Deleted upstream: none.
- Modified: about 40 benchmarks. The changes are mostly the `thrust::pair` → `cuda::std::pair` swap, `nvbench_helper` relocation, and SPDX headers ([PR #6441](https://github.com/NVIDIA/cccl/pull/6441)); there are no functional changes.

---

## 4. Potentially problematic commits

| Commit | Subject | Why risky | What to do |
|--------|---------|-----------|------------|
| [`88b95bb9a2c`](https://github.com/NVIDIA/cccl/commit/88b95bb9a2c) | Inline the Thrust ADL layer ([PR #6377](https://github.com/NVIDIA/cccl/pull/6377)) | Deletes all 43 `system/detail/adl/*.h`, which in rocThrust explicitly include `system/hip/detail/*`. 91 files are affected. | Make the new `__THRUST_DEVICE_SYSTEM_*_HEADER_INCLUDE` macros resolve to `system/hip` when the device system is HIP; build-test both the HIP and CUDA device systems |
| [`2f39a342b93`](https://github.com/NVIDIA/cccl/commit/2f39a342b93) | Refactor Thrust execution policies ([PR #5821](https://github.com/NVIDIA/cccl/pull/5821)) | Deletes `par.h`/`par_to_seq.h` for cuda/cpp/omp/tbb. rocThrust has AMD-only `system/hip/detail/par.h` and `par_to_seq.h`. | Port the same restructuring into `system/hip/` by hand |
| [`e73cc82956a`](https://github.com/NVIDIA/cccl/commit/e73cc82956a) | Move CPP, OMP and TBB exec policies to detail ([PR #6361](https://github.com/NVIDIA/cccl/pull/6361)) | Follow-up to #5821; touches `system/cuda/` | Check whether `system/hip/` needs the same move |
| [`fbb9479172c`](https://github.com/NVIDIA/cccl/commit/fbb9479172c), [`700f2a867a7`](https://github.com/NVIDIA/cccl/commit/700f2a867a7) | Allocator internals; `destroy_range`/`device_*` refactors ([PR #6136](https://github.com/NVIDIA/cccl/pull/6136), [PR #6134](https://github.com/NVIDIA/cccl/pull/6134)) | Deletes 11 `.inl` files; 7 of the allocator `.inl` files contain rocThrust `_THRUST_STD`/`::hip::` edits | Carry the rocThrust edits over into the merged headers |
| [`c22298179e0`](https://github.com/NVIDIA/cccl/commit/c22298179e0), [`c2f0b68245e`](https://github.com/NVIDIA/cccl/commit/c2f0b68245e), [`865e3d6429f`](https://github.com/NVIDIA/cccl/commit/865e3d6429f), [`e11761cc1b8`](https://github.com/NVIDIA/cccl/commit/e11761cc1b8) | OMP / TBB / seq header restructuring ([PR #6372](https://github.com/NVIDIA/cccl/pull/6372), [PR #6394](https://github.com/NVIDIA/cccl/pull/6394), [PR #6437](https://github.com/NVIDIA/cccl/pull/6437), [PR #6406](https://github.com/NVIDIA/cccl/pull/6406)) | About 250 files; most `.inl` files are deleted | Mostly mechanical; diff rocThrust's copies against v3.1.0 first to find local edits that must survive |
| [`91434d76ab3`](https://github.com/NVIDIA/cccl/commit/91434d76ab3) | Replace `thrust::pair` with `cuda::std::pair` ([PR #6616](https://github.com/NVIDIA/cccl/pull/6616)) | 115 files. In the fallback build (no libhipcxx) `_THRUST_STD::pair` is `std::pair`, whose device-side usability is unverified. | Translate to `_THRUST_STD::pair`; build-test the fallback configuration |
| [`31dd1d76e72`](https://github.com/NVIDIA/cccl/commit/31dd1d76e72), [`bcb2bcb106d`](https://github.com/NVIDIA/cccl/commit/bcb2bcb106d), [`4b10fd26255`](https://github.com/NVIDIA/cccl/commit/4b10fd26255), [`e6ad5079192`](https://github.com/NVIDIA/cccl/commit/e6ad5079192), [`5ff11d47308`](https://github.com/NVIDIA/cccl/commit/5ff11d47308) | clang-format, include cleanup, nested namespaces ([PR #6439](https://github.com/NVIDIA/cccl/pull/6439), [PR #6440](https://github.com/NVIDIA/cccl/pull/6440), [PR #6547](https://github.com/NVIDIA/cccl/pull/6547), [PR #6465](https://github.com/NVIDIA/cccl/pull/6465), [PR #5801](https://github.com/NVIDIA/cccl/pull/5801)) | 600+ files of whitespace and include churn; a high chance of conflicting context | Apply to the upstream-shared files; extend to `system/hip/` only if you want consistent formatting |
| [`b2a8d0e88f8`](https://github.com/NVIDIA/cccl/commit/b2a8d0e88f8), [`7b52284299d`](https://github.com/NVIDIA/cccl/commit/7b52284299d) | Plain `cuda::std::`; renamed namespace macros ([PR #5573](https://github.com/NVIDIA/cccl/pull/5573), [PR #5538](https://github.com/NVIDIA/cccl/pull/5538)) | rocThrust spells these `_THRUST_STD::` (1238 uses) | Translate rather than copy verbatim |
| [`3b6e4a6e911`](https://github.com/NVIDIA/cccl/commit/3b6e4a6e911), [`6d3ffc900fb`](https://github.com/NVIDIA/cccl/commit/6d3ffc900fb), [`fa13b90c53b`](https://github.com/NVIDIA/cccl/commit/fa13b90c53b), [`ff8a0d96b7f`](https://github.com/NVIDIA/cccl/commit/ff8a0d96b7f), [`bfb875c8f73`](https://github.com/NVIDIA/cccl/commit/bfb875c8f73) | `cuda::` iterator interop | The types don't exist in libhipcxx 3.0.2 | Port via `_THRUST_LIBCXX::`; requires the libhipcxx ≥ 3.2 bump |
| [`4eb027ce476`](https://github.com/NVIDIA/cccl/commit/4eb027ce476) + [`07e250154a8`](https://github.com/NVIDIA/cccl/commit/07e250154a8) | Drop `thrust::counting_iterator`, then revert it ([PR #5839](https://github.com/NVIDIA/cccl/pull/5839), [PR #5925](https://github.com/NVIDIA/cccl/pull/5925)) | Net zero across 83 files | Tick both N/A together |
| [`8b263079605`](https://github.com/NVIDIA/cccl/commit/8b263079605) | ABI namespace spelling ([PR #6564](https://github.com/NVIDIA/cccl/pull/6564)) | Changes the mangled inline-namespace name, and rocThrust has its own `ROCTHRUST_VERSION` branch | Apply to all three branches of `namespace.h` |
| [`2845628f12e`](https://github.com/NVIDIA/cccl/commit/2845628f12e), [`783ebe64a0e`](https://github.com/NVIDIA/cccl/commit/783ebe64a0e), [`d89402658b2`](https://github.com/NVIDIA/cccl/commit/d89402658b2), [`8a95f4c8051`](https://github.com/NVIDIA/cccl/commit/8a95f4c8051), [`065a6a742d6`](https://github.com/NVIDIA/cccl/commit/065a6a742d6), [`a8c74132b74`](https://github.com/NVIDIA/cccl/commit/a8c74132b74), [`344641b3b3c`](https://github.com/NVIDIA/cccl/commit/344641b3b3c) | Upstream CMake changes | rocThrust's CMake is independent of upstream's | Mostly N/A; check `testing/` and `examples/` CMake additions |

### File renames / moves (build-break risk)
- `thrust/iterator/detail/iterator_traversal_tags.h` → `thrust/iterator/iterator_traversal_tags.h` ([PR #5833](https://github.com/NVIDIA/cccl/pull/5833)). Fix up includes in `system/hip/`.
- 110 upstream-deleted files still exist in rocThrust. They are the 43 `adl/*.h`, 11 allocator/`device_*` `.inl` files, `util/align.h`, `mpl/math.h`, `integer_math.h`, `complex/c99math.h`, `{algorithm,memory,numeric}_wrapper.h`, `cuda/detail/core/{load,make_load}_iterator.h`, `cuda/detail/{par,par_to_seq}.h`, `cuda/detail/internal/copy_*.h`, the OMP/TBB/sequential `.inl` files, and `generic/scan.inl`. Any `system/hip/` or rocThrust-only file that includes them must be updated. The `system/hip/detail/internal/copy_*.h` and `par*.h` counterparts need an explicit keep-or-delete decision.
- `testing/complex.cu` and `testing/metaprogramming.cu` are deleted upstream; keep them (see §3).

### Public API / ABI changes
- The ABI inline namespace name changes ([PR #6564](https://github.com/NVIDIA/cccl/pull/6564)), and `THRUST_VERSION` becomes `300201`.
- Thrust 3.2 is written against libcu++ 3.2, while `libcxx.h` on the base branch still requires 3.0 and the bundled libhipcxx is 3.0.2. Raise the requirement to 3.2 alongside the libhipcxx bump.
- `THRUST_*_FUNCTION` macros are deprecated ([PR #5972](https://github.com/NVIDIA/cccl/pull/5972)).

---

## Handoff

- **Next step**: `rocthrust-cccl-sync-todo` for `v3.1.0..v3.2.1` on top of `users/matyas-streamhpc/hipccl/rocthrust_cccl31`, then `rocthrust-cccl-sync-resolve` and `rocthrust-cccl-sync-finalize`.
- **Decisions (2026-10-19)**:
  - libhipcxx will be at least 3.2. Port the `cuda::` iterator interop and the libcu++ 3.2 macros as-is, translated to `_THRUST_LIBCXX::`/`_THRUST_STD::`, with no libhipcxx-3.0 fallback. Raise the required libhipcxx version in `libcxx.h` from 3.0 to 3.2.
  - Compatibility over performance. Performance-only rewrites (the `DeviceTransform` routing, UBLKCP, the `copy`/`find`/`mismatch`/`iter_swap`/`replace` rewrites, vectorized-transform alignment) are ported into `system/cuda/` only. Their `system/hip/` disposition is "no change: performance-only". Behavior and correctness fixes still go into `system/hip/`.
- **Open questions**:
  - Should the HIP backend's `par.h`, `par_to_seq.h` and `internal/copy_*.h` follow #5821's deletion, or stay as AMD-only?
  - The `THRUST_HIP_FUNCTION` deprecation: follow #5972 or not?
  - Is `std::pair` usable in device code in the fallback (no libhipcxx) build? This affects #6616.
