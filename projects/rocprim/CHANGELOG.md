# Changelog for rocPRIM

Full documentation for rocPRIM is available at [https://rocm.docs.amd.com/projects/rocPRIM/en/latest/](https://rocm.docs.amd.com/projects/rocPRIM/en/latest/).

## rocPRIM 4.1.0 for ROCm 7.1

### Added

* Added `get_sreg_lanemask_lt`, `get_sreg_lanemask_le`, `get_sreg_lanemask_gt` and `get_sreg_lanemask_ge`.
* Added `rocprim::transform_output_iterator` and `rocprim::make_transform_output_iterator`.
* Added experimental support for SPIR-V, to use the correct tuned config for part of the appliable algorithms.
* Added a new cmake option, `BUILD_OFFLOAD_COMPRESS`. When rocPRIM is build with this option enabled, the `--offload-compress` switch is passed to the compiler. This causes the compiler to compress the binary that it generates. Compression can be useful in cases where you are compiling for a large number of targets, since this often results in a large binary. Without compression, in some cases, the generated binary may become so large symbols are placed out of range, resulting in linking errors. The new `BUILD_OFFLOAD_COMPRESS` option is set to `ON` by default.
* Added a new CMake option `-DUSE_SYSTEM_LIB` to allow tests to be built from `ROCm` libraries provided by the system.

### Changed

* Changed tests to support `ptr-to-const` output in `/test/rocprim/test_device_batch_memcpy.cpp`.

### Optimizations

* Improved performance of many algorithms, by updating their tuned configs.
  * 891 specializations have been improved.
  * 399 specializations have been added.

### Upcoming changes

* Deprecated the `->` operator for the `zip_iterator`.

### Resolved issues

* Fixed `device_select`, `device_merge`, and `device_merge_sort` not allocating the correct amount of virtual shared memory on the host.
* Fixed the `->` operator for the `transform_iterator`, the `texture_cache_iterator` and the `arg_index_iterator`, by now returning a proxy pointer.
  * The `arg_index_iterator` also now only returns the internal iterator for the `->`.

## rocPRIM 4.0.0 for ROCm 7.0

### Added

* Added `rocprim::accumulator_t` to ensure parity with CCCL.
* Added test for `rocprim::accumulator_t`
* Added `rocprim::invoke_result_r` to ensure parity with CCCL.
* Added function `is_build_in` into `rocprim::traits::get`.
* Added virtual shared memory as a fallback option in `rocprim::device_merge` when it exceeds shared memory capacity, similar to `rocprim::device_select`, `rocprim::device_partition`, and `rocprim::device_merge_sort`, which already include this feature.
* Added initial value support to device level inclusive scans.
* Added new optimization to the backend for `device_transform` when the input and output are pointers.
* Added `LoadType` to `transform_config`, which is used for the `device_transform` when the input and output are pointers.
* Added `rocprim:device_transform` for n-ary transform operations API with as input `n` number of iterators inside a `rocprim::tuple`.
* Added gfx950 support.
* Added `rocprim::key_value_pair::operator==`.
* Added the `rocprim::unrolled_copy` thread function to copy multiple items inside a thread.
* Added the `rocprim::unrolled_thread_load` function to load multiple items inside a thread using `rocprim::thread_load`.
* Added `rocprim::int128_t` and `rocprim::uint128_t` to benchmarks for improved performance evaluation on 128-bit integers.
* Added `rocprim::int128_t` to the supported autotuning types to improve performance for 128-bit integers.
* Added the `rocprim::merge_inplace` function for merging in-place.
* Added initial value support for warp- and block-level inclusive scan.
* Added support for building tests with device-side random data generation, making them finish faster. This requires rocRAND, and is enabled with the `WITH_ROCRAND=ON` build flag.
* Added tests and documentation to `lookback_scan_state`. It is still in the `detail` namespace.

### Optimizations

* Improved performance of `rocprim::device_select` and `rocprim::device_partition` when using multiple streams on the MI3XX architecture.

### Changed

* Changed the parameters `long_radix_bits` and `LongRadixBits` from `segmented_radix_sort` to `radix_bits` and `RadixBits` respectively.
* Marked the initialisation constructor of `rocprim::reverse_iterator<Iter>` `explicit`, use `rocprim::make_reverse_iterator`.
* Merged `radix_key_codec` into type_traits system.
* Renamed `type_traits_interface.hpp` to `type_traits.hpp`, rename the original `type_traits.hpp` to `type_traits_functions.hpp`.
* The default scan accumulator types for device-level scan algorithms have changed. This is a breaking change.
The previous default accumulator types could lead to situations in which unexpected overflow occured, such as
when the input or inital type was smaller than the output type. 
  * This is a complete list of affected functions and how their default accumulator types are changing:
    * `rocprim::inclusive_scan`
      * Previous default: `class AccType = typename std::iterator_traits<InputIterator>::value_type>`
      * Current default: `class AccType = rocprim::accumulator_t<BinaryFunction, typename std::iterator_traits<InputIterator>::value_type>`
    * `rocprim::deterministic_inclusive_scan`
      * Previous default: `class AccType = typename std::iterator_traits<InputIterator>::value_type>`
      * Current default: `class AccType = rocprim::accumulator_t<BinaryFunction, typename std::iterator_traits<InputIterator>::value_type>`
    * `rocprim::exclusive_scan`
      * Previous default: `class AccType = detail::input_type_t<InitValueType>>`
      * Current default: `class AccType = rocprim::accumulator_t<BinaryFunction, rocprim::detail::input_type_t<InitValueType>>`
    * `rocprim::deterministic_exclusive_scan`
      * Previous default: `class AccType = detail::input_type_t<InitValueType>>`
      * Current default: `class AccType = rocprim::accumulator_t<BinaryFunction, rocprim::detail::input_type_t<InitValueType>>`
* Undeprecated internal `detail::raw_storage`.
* A new version of `rocprim::thread_load` and `rocprim::thread_store` replace the deprecated `rocprim::thread_load` and `rocprim::thread_store` functions. The versions avoid inline assembly where possible, and don't hinder the optimizer as much as a result.
* Renamed `rocprim::load_cs` to `rocprim::load_nontemporal` and `rocprim::store_cs` to `rocprim::store_nontemporal` to express the intent of these load and store methods better.
* All kernels now have hidden symbol visibility. All symbols now have inline namespaces that include the library version, for example, `rocprim::ROCPRIM_300400_NS::symbol` instead of `rocPRIM::symbol`, letting the user link multiple libraries built with different versions of rocPRIM.
    
### Upcoming changes

* `rocprim::invoke_result_binary_op` and `rocprim::invoke_result_binary_op_t` are deprecated. Use `rocprim::accumulator_t` now.

### Removed

* Removed `rocprim::detail::float_bit_mask` and relative tests, use `rocprim::traits::float_bit_mask` instead.
* Removed `rocprim::traits::is_fundamental`, please use `rocprim::traits::get<T>::is_fundamental()` directly.
* Removed the deprecated parameters `short_radix_bits` and `ShortRadixBits` from the `segmented_radix_sort` config. They were unused, it is only an API change.
* Removed the deprecated `operator<<` from the iterators.
* Removed the deprecated `TwiddleIn` and `TwiddleOut`. Use `radix_key_codec` instead.
* Removed the deprecated flags API of `block_adjacent_difference`. Use `subtract_left()` or `block_discontinuity::flag_heads()` instead.
* Removed the deprecated `to_exclusive` functions in the warp scans.
* Removed the `rocprim::load_cs` from the `cache_load_modifier` enum. Use `rocprim::load_nontemporal` instead.
* Removed the `rocprim::store_cs` from the `cache_store_modifier` enum. Use `rocprim::store_nontemporal` instead.
* Removed the deprecated header file `rocprim/detail/match_result_type.hpp`. Include `rocprim/type_traits.hpp` instead.
  * This header included `rocprim::detail::invoke_result`. Use `rocprim::invoke_result` instead.
  * This header included `rocprim::detail::invoke_result_binary_op`. Use `rocprim::invoke_result_binary_op` instead.
  * This header included `rocprim::detail::match_result_type`. Use `rocprim::invoke_result_binary_op_t` instead.
* Removed the deprecated `rocprim::detail::radix_key_codec` function. Use `rocprim::radix_key_codec` instead.
* Removed `rocprim/detail/radix_sort.hpp`, functionality can now be found in `rocprim/thread/radix_key_codec.hpp`.
* Removed C++14 support, only C++17 is supported.
* Due to the removal of `__AMDGCN_WAVEFRONT_SIZE` in the compiler, the following deprecated warp size-related symbols have been removed:
  * `rocprim::device_warp_size()`
    * For compile-time constants, this is replaced with `rocprim::arch::wavefront::min_size()` and `rocprim::arch::wavefront::max_size()`. Use this when allocating global or shared memory.
    * For run-time constants, this is replaced with `rocprim::arch::wavefront::size().`
  * `rocprim::warp_size()`
    * Use `rocprim::host_warp_size()`, `rocprim::arch::wavefront::min_size()` or `rocprim::arch::wavefront::max_size()` instead.
  * `ROCPRIM_WAVEFRONT_SIZE`
    * Use `rocprim::arch::wavefront::min_size()` or `rocprim::arch::wavefront::max_size()` instead.
  * `__AMDGCN_WAVEFRONT_SIZE`
    * This was a fallback define for the compiler's removed symbol, having the same name. 
* This release removes support for custom builds on gfx940 and gfx941.

### Resolved issues

* Fixed an issue where `device_batch_memcpy` reported benchmarking throughput being 2x lower than it was in reality.
* Fixed an issue where `device_segmented_reduce` reported autotuning throughput being 5x lower than it was in reality.
* Fixed device radix sort not returning the correct required temporary storage when a double buffer contains `nullptr`.
* Fixed constness of equality operators (`==` and `!=`) in `rocprim::key_value_pair`.
* Fixed an issue for the comparison operators in `arg_index_iterator` and `texture_cache_iterator`, where `<` and `>` comparators were swapped.
* Fixed an issue for the `rocprim::thread_reduce` not working correctly with a prefix value.

### Known issues
* When using `rocprim::deterministic_inclusive_scan_by_key` and `rocprim::deterministic_exclusive_scan_by_key` the intermediate values can change order on Navi3x
  * However if a commutative scan operator is used then the final scan value (output array) will still always be consistent between runs

## rocPRIM 3.4.1 for ROCm 6.4.2

### Upcoming changes
* Changes to the template parameters of warp and block algorithms will be made in an upcoming release.

* Due to an upcoming compiler change the following warp size-related symbols will be removed in the next major release and are thus marked as deprecated:
  * `rocprim::device_warp_size()`
    * For compile-time constants, this is replaced with `rocprim::arch::wavefront::min_size()` and `rocprim::arch::wavefront::max_size()`. Use this when allocating global or shared memory.
    * For run-time constants, this is replaced with `rocprim::arch::wavefront::size().`
  * `rocprim::warp_size()`
  * `ROCPRIM_WAVEFRONT_SIZE`
  
* The default scan accumulator types for device-level scan algorithms will be changed in an upcoming release, resulting in a breaking change. Previously, the default accumulator type was set to the input type for the inclusive scans and to the initial value type for the exclusive scans. This could lead to unexpected overflow if the input or initial type was smaller than the output type when the accumulator type was't explicitly set using the `AccType` template parameter. The new default accumulator types will be set to the type that results when the input or initial value type is applied to the scan operator.  

    The following is the complete list of affected functions and how their default accumulator types are changing:
    
    * `rocprim::inclusive_scan`
        * current default: `class AccType = typename std::iterator_traits<InputIterator>::value_type>`
        * future default: `class AccType = rocprim::invoke_result_binary_op_t<typename std::iterator_traits<InputIterator>::value_type, BinaryFunction>`
    * `rocprim::deterministic_inclusive_scan`
        * current default: `class AccType = typename std::iterator_traits<InputIterator>::value_type>`
        * future default: `class AccType = rocprim::invoke_result_binary_op_t<typename std::iterator_traits<InputIterator>::value_type, BinaryFunction>`
    * `rocprim::exclusive_scan`
        * current default: `class AccType = detail::input_type_t<InitValueType>>`
        * future default: `class AccType = rocprim::invoke_result_binary_op_t<rocprim::detail::input_type_t<InitValueType>, BinaryFunction>`
    * `rocprim::deterministic_exclusive_scan`
        * current default: `class AccType = detail::input_type_t<InitValueType>>`
        * future default: `class AccType = rocprim::invoke_result_binary_op_t<rocprim::detail::input_type_t<InitValueType>, BinaryFunction>`

* `rocprim::load_cs` and `rocprim::store_cs` are deprecated and will be removed in an upcoming release. Alternatively, you can use `rocprim::load_nontemporal` and `rocprim::store_nontemporal` to load and store values in specific conditions (like bypassing the cache) for `rocprim::thread_load` and `rocprim::thread_store`.

## rocPRIM 3.4.0 for ROCm 6.4.0

### Added

* Added extended tests to `rtest.py`. These tests are extra tests that did not fit the criteria of smoke and regression tests. These tests will take much longer to run relative to smoke and regression tests.
 * Use `python rtest.py [--emulation|-e|--test|-t]=extended` to run these tests.
* Added regression tests to `rtest.py`. Regression tests are a subset of tests that caused hardware problems for past emulation environments.
  * Can be run with `python rtest.py [--emulation|-e|--test|-t]=regression`
* Added the parallel `find_first_of` device function with autotuned configurations, this function is similar to `std::find_first_of`, it searches for the first occurrence of any of the provided elements.
* Added `--emulation` option added for `rtest.py`
  * Unit tests can be run with `[--emulation|-e|--test|-t]=<test_name>`
* Added tuned configurations for segmented radix sort for gfx942 to improve performance on this architecture.
* Added a parallel device-level function, `rocprim::adjacent_find`, similar to the C++ Standard Library `std::adjacent_find` algorithm.
* Added configuration autotuning to device adjacent find (`rocprim::adjacent_find`) for improved performance on selected architectures.
* Added rocprim::numeric_limits which is an extension of `std::numeric_limits`, which includes support for 128-bit integers.
* Added rocprim::int128_t and rocprim::uint128_t which are the __int128_t and __uint128_t types.
* Added the parallel `search` and `find_end` device functions similar to `std::search` and `std::find_end`, these functions search for the first and last occurrence of the sequence respectively.
* Added a parallel device-level function, `rocprim::search_n`, similar to the C++ Standard Library `std::search_n` algorithm.
* Added new constructors and a `base` function, and added `constexpr` specifier to all functions in `rocprim::reverse_iterator` to improve parity with the C++17 `std::reverse_iterator`.
* Added hipGraph support to device run-length-encode for nontrivial runs (`rocprim::run_length_encode_non_trivial_runs`).
* Added configuration autotuning to device run-length-encode for nontrivial runs (`rocprim::run_length_encode_non_trivial_runs`) for improved performance on selected architectures.
* Added configuration autotuning to device run-length-encode for trivial runs (`rocprim::run_length_encode`) for improved performance on selected architectures.
* Added a new type traits interface to enable users to provide additional type trait information to rocPRIM, facilitating better compatibility with custom types.

### Changed

* Changed the subset of tests that are run for smoke tests such that the smoke test will complete with faster run-time and to never exceed 2GB of vram usage. Use `python rtest.py [--emulation|-e|--test|-t]=smoke` to run these tests.
* The `rtest.py` options have changed. `rtest.py` is now run with at least either `--test|-t` or `--emulation|-e`, but not both options.
* Changed the internal algorithm of block radix sort to use rank match to improve performance of various radix sort related algorithms.
* Disabled padding in various cases where higher occupancy resulted in better performance despite more bank conflicts.

* Removed HIP-CPU support. HIP-CPU support was experimental and broken.
* Changed the C++ version from 14 to 17. C++14 will be deprecated in the next major release.
* You can use CMake HIP language support with CMake 3.18 and later. To use HIP language support, run `cmake` with `-DUSE_HIPCXX=ON` instead of setting the `CXX` variable to the path to a HIP-aware compiler.

### Resolved issues

* Fixed an issue where `rmake.py` would generate wrong CMAKE commands while using Linux environment
* Fixed an issue where `rocprim::partial_sort_copy` would yield a compile error if the input iterator is const.
* Fixed incorrect 128-bit signed and unsigned integers type traits.
* Fixed compilation issue when `rocprim::radix_key_codec<...>` is specialized with a 128-bit integer.
* Fixed the warp-level reduction `rocprim::warp_reduce.reduce` DPP implementation to avoid undefined intermediate values during the reduction.
* Fixed an issue that caused a segmentation fault when `hipStreamLegacy` was passed to some API functions.

### Upcoming changes
* Using the initialisation constructor of `rocprim::reverse_iterator` will throw a deprecation warning. It will be marked as explicit in the next major release.

* Using the initialisation constructor of rocprim::reverse_iterator will throw a deprecation warning. It will be marked as explicit in the next major release.

## rocPRIM 3.3.0 for ROCm 6.3.0

### Added

* Changed the default value of `rmake.py -a` to `default_gpus`. This is equivalent to `gfx906:xnack-,gfx1030,gfx1100,gfx1101,gfx1102,gfx1151,gfx1200,gfx1201`.
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

* Fixed an issue in `rmake.py` where the list storing cmake options would contain individual characters instead of a full string of options.
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


