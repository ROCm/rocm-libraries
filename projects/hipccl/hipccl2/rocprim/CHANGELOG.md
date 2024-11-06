# Changelog for rocPRIM

Full documentation for rocPRIM is available at [https://rocm.docs.amd.com/projects/rocPRIM/en/latest/](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/).

## (Unreleased) rocPRIM 3.4.0 for ROCm 6.4.0

### Added

* Added the parallel `find_first_of` device function with autotuned configurations, this function is similar to `std::find_first_of`, it searches for the first occurrence of any of the provided elements.

### Changed

### Resolved issues

* Fixed an issue where `rocprim::partial_sort_copy` would yield a compile error if the input iterator is const.
* Fixed incorrect 128-bit signed and unsigned integers type traits.
* Fixed compilation issue when `rocprim::radix_key_codec<...>` is specialized with a 128-bit integer.

### Upcoming changes

## rocPRIM 3.3.0 for ROCm 6.3.0

### Added

* The `--test smoke` option has been added to `rtest.py`. When `rtest.py` is called with this option it runs a subset of tests such that the total test time is 5 minutes. Use `python3 ./rtest.py --test smoke` or `python3 ./rtest.py -t smoke` to run the smoke test.
* The `--seed` option has been added to `run_benchmarks.py`. The `--seed` option specifies a seed for the generation of random inputs. When the option is omitted, the default behavior is to use a random seed for each benchmark measurement.
* Added configuration autotuning to device partition (`rocprim::partition`, `rocprim::partition_two_way`, and `rocprim::partition_three_way`), to device select (`rocprim::select`, `rocprim::unique`, and `rocprim::unique_by_key`), and to device reduce by key (`rocprim::reduce_by_key`) to improve performance on selected architectures.
* Added `rocprim::uninitialized_array` to provide uninitialized storage in local memory for user-defined types.
* Added large segment support for `rocprim:segmented_reduce`.
* Added a parallel `nth_element` device function similar to `std::nth_element`. `nth_element` places elements that are smaller than the nth element before the nth element, and elements that are bigger than the nth element after the nth element.
* Added deterministic (bitwise reproducible) algorithm variants `rocprim::deterministic_inclusive_scan`, `rocprim::deterministic_exclusive_scan`, `rocprim::deterministic_inclusive_scan_by_key`, `rocprim::deterministic_exclusive_scan_by_key`, and `rocprim::deterministic_reduce_by_key`. These provide run-to-run stable results with non-associative operators such as float operations, at the cost of reduced performance.
* Added a parallel `partial_sort` and `partial_sort_copy` device functions similar to `std::partial_sort` and `std::partial_sort_copy`. `partial_sort` and `partial_sort_copy` arrange elements such that the elements are in the same order as a sorted list up to and including the middle index.

### Changed

* Modified the input size in device adjacent difference benchmarks. Observed performance with these benchmarks might be different.
* Changed the default seed for `device_benchmark_segmented_reduce`.
* Changed `test_utils_hipgraphs.hpp` to be a class `GraphHelper` with internal graph and graph instances

### Removed

* `rocprim::thread_load()` and `rocprim::thread_store()` have been deprecated. Use `dereference()` instead.

### Resolved issues

* Resolved an issue in `rtest.py` where it crashed if the `build` folder was created without `release` or `debug` subdirectories.
* Resolved an issue with `rtest.py` on Windows where passing an absolute path to `--install_dir` caused a `FileNotFound` error.
* rocPRIM functions are no longer forcefully inlined on Windows. This significantly reduces the build
  time of debug builds.
* `block_load`, `block_store`, `block_shuffle`, `block_exchange`, and `warp_exchange` now use placement `new` instead of copy assignment (`operator=`) when writing to local memory. This fixes the behavior of custom types with non-trivial copy assignments.
* Fixed a bug in the generation of input data for benchmarks, which caused incorrect performance to be reported in specific cases. It may affect the reported performance for one-byte types (`uint8_t` and `int8_t`) and instantiations of `custom_type`. Specifically, device binary search, device histogram, device merge and warp sort are affected.
* Fixed a bug for `rocprim::merge_path_search` where using `unsigned` offsets would produce incorrect results.
* Fixed a bug for `rocprim::thread_load` and `rocprim::thread_store` where `float` and `double` were not cast to the correct type, resulting in incorrect results.
* Resolved an issue where tests where failing when they were compiled with `-D_GLIBCXX_ASSERTIONS=ON`.
* Resolved an issue where algorithms that used an internal serial merge routine caused a memory access fault that resulted in potential performance drops when using block sort, device merge sort (block merge), device merge, device partial sort, and device sort (merge sort).
* Fixed memory leaks in unit tests due to missing calls to `hipFree()` and the incorrect use of hipGraphs.
* Fixed an issue where certain inputs to `block_sort_merge()`, `device_merge_sort_merge_path()`, `device_merge()`, and `warp_sort_stable()` caused an assertion error during the call to `serial_merge()`.

## rocPRIM 3.2.1 for ROCm 6.2.1

### Optimizations

* Improved performance of `block_reduce_warp_reduce` when warp size equals block size.

## rocPRIM-3.2.0 for ROCm 6.2.0

### Additions

* New overloads for `warp_scan::exclusive_scan` that take no initial value. These new overloads will write an unspecified result to the first value of each warp.
* The internal accumulator type of `inclusive_scan(_by_key)` and `exclusive_scan(_by_key)` is now exposed as an optional type parameter.
  * The default accumulator type is still the value type of the input iterator (inclusive scan) or the initial value's type (exclusive scan).
    This is the same behaviour as before this change.
* New overload for `device_adjacent_difference_inplace` that allows separate input and output iterators, but allows them to point to the same element.
* New public API for deriving resulting type on device-only functions:
  * `rocprim::invoke_result`
  * `rocprim::invoke_result_t`
  * `rocprim::invoke_result_binary_op`
  * `rocprim::invoke_result_binary_op_t`
* New `rocprim::batch_copy` function added. Similar to `rocprim::batch_memcpy`, but copies by element, not with memcpy.
* Added more test cases, to better cover supported data types.
* Updated some tests to work with supported data types.
* An optional `decomposer` argument for all member functions of `rocprim::block_radix_sort` and all functions of `device_radix_sort`.
  To sort keys of an user-defined type, a decomposer functor should be passed. The decomposer should produce a `rocprim::tuple`
  of references to arithmetic types from the key.
* New `rocprim::predicate_iterator` which acts as a proxy for an underlying iterator based on a predicate.
  It iterates over proxies that holds the references to the underlying values, but only allow reading and writing if the predicate is `true`.
  It can be instantiated with:
  * `rocprim::make_predicate_iterator`
  * `rocprim::make_mask_iterator`
* Added custom radix sizes as the last parameter for `block_radix_sort`. The default value is 4, it can be a number between 0 and 32.
* New `rocprim::radix_key_codec`, which allows the encoding/decoding of keys for radix-based sorts. For user-defined key types, a decomposer functor should be passed.

### Optimizations

* Improved the performance of `warp_sort_shuffle` and `block_sort_bitonic`.
* Created an optimized version of the `warp_exchange` functions `blocked_to_striped_shuffle` and `striped_to_blocked_shuffle` when the warpsize is equal to the items per thread.
* Improved the performance of `device_transform`.

### Fixes

* Fixed incorrect results of `warp_exchange::blocked_to_striped_shuffle` and `warp_exchange::striped_to_blocked_shuffle` when the block size is
  larger than the logical warp size. The test suite has been updated with such cases.
* Fixed incorrect results returned when calling device `unique_by_key` with overlapping `values_input` and `values_output`.
* Fixed incorrect output type used in `device_adjacent_difference`.
* Hotfix for incorrect results on the GFX10 (Navi 10/RDNA1, Navi 20/RDNA2) ISA and GFX11 ISA (Navi 30 GPUs) on device scan algorithms `rocprim::inclusive_scan(_by_key)` and `rocprim::exclusive_scan(_by_key)` with large input types.
* `device_adjacent_difference` now considers both the input and the output type for selecting the appropriate kernel launch config. Previously only the input type was considered, which could result in compilation errors due to excessive shared memory usage.
* Fixed incorrect data being loaded with `rocprim::thread_load` when compiling with `-O0`.
* Fixed a compilation failure in the host compiler when instantiating various block and device algorithms with block sizes not divisible by 64.

### Deprecations

* The internal header `detail/match_result_type.hpp` has been deprecated.
* `TwiddleIn` and `TwiddleOut` have been deprecated in favor of `radix_key_codec`.
* The internal `::rocprim::detail::radix_key_codec` has been deprecated in favor of the new public utility with the same name.

## rocPRIM-3.1.0 for ROCm 6.1.0

### Additions

* New primitive: `block_run_length_decode`
* New primitive: `batch_memcpy`

### Changes

* Renamed:
  * `scan_config_v2` to `scan_config`
  * `scan_by_key_config_v2` to `scan_by_key_config`
  * `radix_sort_config_v2` to `radix_sort_config`
  * `reduce_by_key_config_v2` to `reduce_by_key_config`
  * `radix_sort_config_v2` to `radix_sort_config`
* Removed support for custom config types for device algorithms
* `host_warp_size()` was moved into `rocprim/device/config_types.hpp`; it now uses either `device_id` or
  a `stream` parameter to query the proper device and a `device_id` out parameter
  * The return type is `hipError_t`
* Added support for `__int128_t` in `device_radix_sort` and `block_radix_sort`
* Improved the performance of `match_any`, and `block_histogram` which uses it

### Deprecations

* Removed `reduce_by_key_config`, `MatchAny`, `scan_config`, `scan_by_key_config`, and
  `radix_sort_config`

### Fixes

* Build issues with `rmake.py` on Windows when using VS 2017 15.8 or later (due to a breaking fix with
  extended aligned storage)
* Fix tests for `block_histogram`, `block_exchange`, `device_histogram` and `device_reduce_by_key` for various types

### Known Issues
* `device_run_length_encode`, `warp_exchange` and `warp_load` tests fail with `rocprim::half`

## rocPRIM-3.0.0 for ROCm 6.0.0

### Additions
- `block_sort::sort()` overload for keys and values with a dynamic size, for all block sort algorithms. Additionally, all `block_sort::sort()` overloads with a dynamic size are now supported for `block_sort_algorithm::merge_sort` and `block_sort_algorithm::bitonic_sort`.
- New two-way partition primitive `partition_two_way` which can write to two separate iterators.

### Optimizations
- Improved the performance of `partition`.

### Fixes
- Fixed `rocprim::MatchAny` for devices with 64-bit warp size. The function `rocprim::MatchAny` is deprecated and `rocprim::match_any` is preferred instead.

## rocPRIM-2.13.1 for ROCm 5.7.0

### Changes
- Deprecated configuration `radix_sort_config` for device-level radix sort as it no longer matches the algorithm's parameters. New configuration `radix_sort_config_v2` is preferred instead.
- Removed erroneous implementation of device-level `inclusive_scan` and `exclusive_scan`. The prior default implementation using lookback-scan now is the only available implementation.
- The benchmark metric indicating the bytes processed for `exclusive_scan_by_key` and `inclusive_scan_by_key` has been changed to incorporate the key type. Furthermore, the benchmark log has been changed such that these algorithms are reported as `scan` and `scan_by_key` instead of `scan_exclusive` and `scan_inclusive`.
- Deprecated configurations `scan_config` and `scan_by_key_config` for device-level scans, as they no longer match the algorithm's parameters. New configurations `scan_config_v2` and `scan_by_key_config_v2` are preferred instead.

### Fixes
- Fixed build issue caused by missing header in `thread/thread_search.hpp`.

## rocPRIM-2.13.0 for ROCm 5.5.0

### Additions

* New block level `radix_rank` primitive
* New block level `radix_rank_match` primitive
* Added a stable block sorting implementation, which can be used with `block_sort` by adding the `block_sort_algorithm::stable_merge_sort` algorithm

### Changes

* Improved the performance of:
  * `block_radix_sort`
  * `device_radix_sort`
  * `device_merge_sort`
* Updated the `docs` directory structure to match the standard of
  [rocm-docs-core](https://github.com/RadeonOpenCompute/rocm-docs-core)

### Known Issues

* Disabled GPU error messages relating to incorrect warp operation usage with Navi GPUs on
  Windows (due to GPU `printf` performance issues on Windows)
* When `ROCPRIM_DISABLE_LOOKBACK_SCAN` is set, `device_scan` fails for input sizes larger than
  `scan_config::size_limit`, which defaults to `std::numeric_limits<unsigned int>::max()`

## rocPRIM-2.12.0 for ROCm 5.4.0

### Changes

* `device_partition`, `device_unique`, and `device_reduce_by_key` now support problem sizes larger than
  2^32 items
* Device algorithms now return `hipErrorInvalidValue` if the amount of passed temporary memory is
  insufficient
* Lists of sizes for tests are unified, restored scan and reduce tests for `half` and `bfloat16` values

### Removals

* `block_sort::sort()` overload for keys and values with a dynamic size
  * This overload was documented but the implementation is missing; to avoid further confusion, the
    documentation is removed until a decision is made on implementing the function

## rocPRIM-2.11.1 for ROCm 5.3.3

### Fixes

* Fixed the compilation failure in `device_merge` when the two key iterators don't match

## rocPRIM-2.11.0 for ROCm 5.3.2

### Known Issues

* `device_merge` doesn't correctly support different types for `keys_input1` and `keys_input2` (as of the
  5.3.0 release)

## rocPRIM-2.11.0 for ROCm 5.3.0

### Additions

* New functions `subtract_left` and `subtract_right` in `block_adjacent_difference` to apply functions on
  pairs of adjacent items distributed between threads in a block
* New device-level `adjacent_difference` primitives
* Experimental tooling for automatic kernel configuration tuning for various architectures
* Benchmarks collect and output more detailed system information
* CMake functionality improves build parallelism of the test suite that splits compilation units by
  function or by parameters
* Reverse iterator
* Support for problem sizes over `UINT_MAX` in device functions `inclusive_scan_by_key` and
  `exclusive_scan_by_key`

## Changes

* Improved the performance of warp primitives using the swizzle operation on Navi
* Improved build parallelism of the test suite by splitting up large compilation units
* `device_select` now supports problem sizes larger than 2^32 items
* `device_segmented_radix_sort` now partitions segments to group small, medium, and large segments
  * Each segment group can be sorted by specialized kernels to improve throughput
* Improved histogram performance for the case of highly uneven sample distribution

## rocPRIM-2.10.14 for ROCm 5.2.0

### Additions

* Packages for tests and benchmark executables on all supported operating systems using CPack
* Added file and folder reorganization changes with backward compatibility support using wrapper
  headers

## rocPRIM-2.10.13 for ROCm 5.1.0

### Fixes

* Fixed Radix Sort `int64_t` bug introduced in version 2.10.11

### Additions

* Future value
* Device `partition_three_way` to partition input to three output iterators based on two predicates

### Changes

* The reduce/scan algorithm precision issues in the tests has been resolved for half types
* The device Radix Sort algorithm supports indexing with 64-bit unsigned integers
  * The indexer type is chosen based on the type argument of parameter `size`
  * If `sizeof(size)` is not larger than 4 bytes, the indexer type is 32-bit unsigned int, otherwise, the
    indexer type is 64-bit unsigned int
  * The maximum problem size is based on the compile time configuration of the algorithm according to the following formula:
    * `max_problem_size = (UINT_MAX + 1) * config::scan::block_size * config::scan::items_per_thread`

### Deprecations

* Flags API of `block_adjacent_difference`

### Known issues

* `device_segmented_radix_sort` unit test is failing for HIP on Windows

## rocPRIM-2.10.12 for ROCm 5.0.0

### Fixes

* Enable bfloat16 tests and reduce threshold for bfloat16
* Fix device scan `limit_size` feature
* Non-optimized builds no longer trigger local memory limit errors

### Additions

* Scan size limit feature
* Reduce size limit feature
* Transform size limit feature
* `block_load_striped` and `block_store_striped`
* `gather_to_blocked` to gather values from other threads into a blocked arrangement
* The block sizes for device merge sorts initial block sort and its merge steps are now separate in its
  kernel config
  * The block sort step supports multiple items per thread

### Changes

* you can now set the `size_limit` for scan, reduce, and transform in the config struct instead of using a
  parameter
* `device_scan` and `device_segmented_scan`: `inclusive_scan` now uses the `input-type` as
  `accumulator-type`; `exclusive_scan` uses `initial-value-type`
  * This changes the behavior of small-size input types with large-size output types (e.g., `short` input,
    `int` output) and low-res input with high-res output (e.g., `float` input, `double` output)
* Revert an old Fiji workaround because they solved the issue at the compiler side
* Update README CMake minimum version number
* Added block sort support for multiple items per thread
  * Currently only powers of two block sizes, and items per threads are supported and only for full blocks
* Bumped the minimum required version of CMake to 3.16

### Known issues

* `device_segmented_radix_sort` and `device_scan` unit tests failing for HIP on Windows
* `ReduceEmptyInput` causes random failure with bfloat16

## rocPRIM-2.10.11 for ROCm 4.5.0

### Additions

* Initial HIP on Windows support
* bfloat16 support added

### Changes

* Packaging has been split into a runtime package (`rocprim`) and a development package
  (`rocprim-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.
  * Because rocPRIM is a header-only library, the runtime package is an empty placeholder used to aid
    in the transition. This package is also a deprecated feature and will be removed in a future rocm
    release.

### Known issues

* Unit tests may soft hang on MI200 when running in `hipMallocManaged` mode

## rocPRIM-2.10.11 for ROCm 4.4.0

### Additions

* Code coverage tools build option
* AddressSanitizer build option
* gfx1030 support added
* Experimental [HIP-CPU](https://github.com/ROCm-Developer-Tools/HIP-CPU) support; build using
  GCC/Clang/MSVC on Windows and Linux (this is work in progress and many algorithms are known to
  fail)

### Optimizations

* Added single tile Radix Sort for smaller sizes
* Improved performance for Radix Sort for larger element sizes

## rocPRIM-2.10.10 for ROCm 4.3.0

### Fixes

* Bug fix and minor performance improvement for `merge_sort` when input and output storage are the
  same

### Additions

* gfx90a support added

### Deprecations

* `warp_size()` function; use `host_warp_size()` and `device_warp_size()` for host and device references,
  respectively

## rocPRIM-2.10.9 for ROCm 4.2.0

### Fixes

* Size zero inputs are now properly handled with newer ROCm builds that no longer allow zero-size
  kernel grid and block dimensions

### Changes

* Minimum CMake version required is now 3.10.2

### Known issues

* Device scan unit test is currently failing due to an LLVM bug

## rocPRIM-2.10.8 for ROCm 4.1.0

### Fixes

* Texture cache iteration support has been re-enabled
* Benchmark builds have been re-enabled
* Unique operator is no longer called on invalid elements

### Known issues

* Device scan unit test is currently failing because of an LLVM bug

## rocPRIM-2.10.7 for ROCm 4.0.0

* No new features

## rocPRIM-2.10.6 for ROCm 3.10

### Optimizations

* Updates to DPP instructions for warp shuffle

### Known issues

* Benchmark builds are disabled due to compiler bug

## rocPRIM-2.10.5 for ROCm 3.9.0

### Additions

* HIP CMake dependency

### Optimizations

* Updates to warp shuffle for gfx10
* Disabled DPP functions on gfx10++

### Known issues

* Benchmark builds are disabled due to compiler bug

## rocPRIM-2.10.4 for ROCm 3.8.0

### Fixes

* Fix for rocPRIM texture cache iterator

## rocPRIM-2.10.3 for ROCm 3.7.0

### Fixes

* Package dependency correct to `hip-rocclr`

### Known issues

* rocPRIM texture cache iterator functionality is broken in the runtime (this will be fixed in the next
  release); you can use the prior release if calling this function

## rocPRIM-2.10.2 for ROCm 3.6.0

* No new features

## rocPRIM-2.10.1 for ROCm 3.5.1

### Fixes

* Point release with compilation fix

## rocPRIM-2.10.1 for ROCm 3.5.0

### Additions

* Improved tests with fixed and random seeds for test data
* Network interface improvements with API v3

### Changes

* Switched to HIP-Clang as the default compiler
* CMake searches for rocPRIM locally first; if t's not found, CMake downloads it from GitHub
