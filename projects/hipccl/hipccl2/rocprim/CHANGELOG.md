# Change Log for rocPRIM

Full documentation for rocPRIM is available at [https://rocprim.readthedocs.io/en/latest/](https://rocprim.readthedocs.io/en/latest/)

## [Unreleased rocPRIM-2.13.1 for ROCm 5.7.0]
### Changed
- Deprecated configuration `radix_sort_config` for device-level radix sort as it no longer matches the algorithm's parameters. New configuration `radix_sort_config_v2` is preferred instead.
- Removed erroneous implementation of device-level `inclusive_scan` and `exclusive_scan`. The prior default implementation using lookback-scan now is the only available implementation.
- The benchmark metric indicating the bytes processed for `exclusive_scan_by_key` and `inclusive_scan_by_key` has been changed to incorporate the key type. Furthermore, the benchmark log has been changed such that these algorithms are reported as `scan` and `scan_by_key` instead of `scan_exclusive` and `scan_inclusive`.
- Deprecated configurations `scan_config` and `scan_by_key_config` for device-level scans, as they no longer match the algorithm's parameters. New configurations `scan_config_v2` and `scan_by_key_config_v2` are preferred instead.
### Fixed
- Fixed build issue caused by missing header in `thread/thread_search.hpp`.

## [rocPRIM-2.13.0 for ROCm 5.5.0]
### Added
- New block level `radix_rank` primitive.
- New block level `radix_rank_match` primitive.
- Added a stable block sorting implementation. This be used with `block_sort` by using the `block_sort_algorithm::stable_merge_sort` algorithm.
### Changed
- Improved the performance of `block_radix_sort` and `device_radix_sort`.
- Improved the performance of `device_merge_sort`.
- Updated `docs` directory structure to match the standard of [rocm-docs-core](https://github.com/RadeonOpenCompute/rocm-docs-core). Contributed by: [v01dXYZ](https://github.com/v01dXYZ).
### Known Issues
- Disabled GPU error messages relating to incorrect warp operation usage with Navi GPUs on Windows, due to GPU printf performance issues on Windows.
- When `ROCPRIM_DISABLE_LOOKBACK_SCAN` is set, `device_scan` fails for input sizes bigger than `scan_config::size_limit`, which defaults to `std::numeric_limits<unsigned int>::max()`.

## [rocPRIM-2.12.0 for ROCm 5.4.0]
### Changed
- `device_partition`, `device_unique`, and `device_reduce_by_key` now support problem
  sizes larger than 2^32 items.
- Device algorithms now return `hipErrorInvalidValue` if the amount of passed temporary memory is insufficient.
- Lists of sizes for tests are unified, restored scan/reduce tests for `half` and `bfloat16` values.
### Removed
- `block_sort::sort()` overload for keys and values with a dynamic size. This overload was documented but the
  implementation is missing. To avoid further confusion the documentation is removed until a decision is made on
  implementing the function.
### Fixed
- Fixed the compilation failure in `device_merge` if the two key iterators don't match.

## [rocPRIM-2.11.1 for ROCm 5.3.3]
### Fixed
- Fixed the compilation failure in device_merge if the two key iterators don't match.

## [rocPRIM-2.11.0 for ROCm 5.3.2]
### Known Issue
- device_merge no longer correctly supports using different types for `keys_input1` and `keys_input2` (starting from the 5.3.0 release).

## [rocPRIM-2.11.0 for ROCm 5.3.0]
### Added
- New functions `subtract_left` and `subtract_right` in `block_adjacent_difference` to apply functions
  on pairs of adjacent items distributed between threads in a block.
- New device level `adjacent_difference` primitives.
- Added experimental tooling for automatic kernel configuration tuning for various architectures
- Benchmarks collect and output more detailed system information
- CMake functionality to improve build parallelism of the test suite that splits compilation units by
function or by parameters.
- Reverse iterator.
- Support for problem sizes over `UINT_MAX` in device functions `inclusive_scan_by_key` and `exclusive_scan_by_key`.
## Changed
- Improved the performance of warp primitives using the swizzle operation on Navi
- Improved build parallelism of the test suite by splitting up large compilation units
- `device_select` now supports problem sizes larger than 2^32 items.
- `device_segmented_radix_sort` now partitions segments to groups small, medium and large segments.
  Each segment group can be sorted by specialized kernels to improve throughput.
- Improved performance of histogram for the case of highly uneven sample distribution.

## [rocPRIM-2.10.14 for ROCm 5.2.0]
### Added
- Packages for tests and benchmark executable on all supported OSes using CPack.
- Added File/Folder Reorg Changes and Enabled Backward compatibility support using wrapper headers.

## [Released rocPRIM-2.10.13 for ROCm 5.1.0]
### Fixed
- Fixed radix sort int64_t bug introduced in [2.10.11]
### Added
- Future value
- Added device partition_three_way to partition input to three output iterators based on two predicates
### Changed
- The reduce/scan algorithm precision issues in the tests has been resolved for half types.
- The device radix sort algorithm supports indexing with 64 bit unsigned integers.
  - The indexer type is chosen based on the type argument of parameter `size`.
  - If `sizeof(size)` is not larger than 4 bytes, the indexer type is 32 bit unsigned int,
  - Else the indexer type is 64 bit unsigned int.
  - The maximum problem size is based on the compile time configuration of the algorithm according to the following formula:
    - `max_problem_size = (UINT_MAX + 1) * config::scan::block_size * config::scan::items_per_thread`.
- The flags API of `block_adjacent_difference` is now deprecated and will be removed in a future
  version.
### Known issues
- device_segmented_radix_sort unit test failing for HIP on Windows

## [Released rocPRIM-2.10.12 for ROCm 5.0.0]
### Fixed
- Enable bfloat16 tests and reduce threshold for bfloat16
- Fix device scan limit_size feature
- Non-optimized builds no longer trigger local memory limit errors
### Added
- Added scan size limit feature
- Added reduce size limit feature
- Added transform size limit feature
- Add block_load_striped and block_store_striped
- Add gather_to_blocked to gather values from other threads into a blocked arrangement
- The block sizes for device merge sorts initial block sort and its merge steps are now separate in its kernel config
    - the block sort step supports multiple items per thread
### Changed
- size_limit for scan, reduce and transform can now be set in the config struct instead of a parameter
- Device_scan and device_segmented_scan: `inclusive_scan` now uses the input-type as accumulator-type, `exclusive_scan` uses initial-value-type.
  - This particularly changes behaviour of small-size input types with large-size output types (e.g. `short` input, `int` output).
  - And low-res input with high-res output (e.g. `float` input, `double` output)
- Revert old Fiji workaround, because they solved the issue at compiler side
- Update README cmake minimum version number
- Block sort support multiple items per thread
    - currently only powers of two block sizes, and items per threads are supported and only for full blocks
- Bumped the minimum required version of CMake to 3.16
### Known issues
- Unit tests may soft hang on MI200 when running in hipMallocManaged mode.
- device_segmented_radix_sort, device_scan unit tests failing for HIP on Windows
- ReduceEmptyInput cause random faulire with bfloat16

## [rocPRIM-2.10.11 for ROCm 4.5.0]
### Added
- Initial HIP on Windows support. See README for instructions on how to build and install.
- bfloat16 support added.
### Changed
- Packaging split into a runtime package called rocprim and a development package called rocprim-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
    - As rocPRIM is a header-only library, the runtime package is an empty placeholder used to aid in the transition. This package is also a deprecated feature and will be removed in a future rocm release.
### Known issues
- Unit tests may soft hang on MI200 when running in hipMallocManaged mode.
### Deprecated
- The warp_size() function is now deprecated; please switch to host_warp_size() and device_warp_size() for host and device references respectively.

## [rocPRIM-2.10.11 for ROCm 4.4.0]
### Added
- Code coverage tools build option
- Address sanitizer build option
- gfx1030 support added.
- Experimental [HIP-CPU](https://github.com/ROCm-Developer-Tools/HIP-CPU) support; build using GCC/Clang/MSVC on Win/Linux. It is work in progress, many algorithms still known to fail.
### Optimizations
- Added single tile radix sort for smaller sizes.
- Improved performance for radix sort for larger element sizes.
### Deprecated
- The warp_size() function is now deprecated; please switch to host_warp_size() and device_warp_size() for host and device references respectively.

## [rocPRIM-2.10.10 for ROCm 4.3.0]
### Fixed
- Bugfix & minor performance improvement for merge_sort when input and output storage are the same.
### Added
- gfx90a support added.
### Deprecated
- The warp_size() function is now deprecated; please switch to host_warp_size() and device_warp_size() for host and device references respectively.

## [rocPRIM-2.10.9 for ROCm 4.2.0]
### Fixed
- Size zero inputs are now properly handled with newer ROCm builds that no longer allow zero-size kernel grid/block dimensions
### Changed
- Minimum cmake version required is now 3.10.2
### Known issues
- Device scan unit test currently failing due to LLVM bug.

## [rocPRIM-2.10.8 for ROCm 4.1.0]
### Fixed
- Texture cache iteration support has been re-enabled.
- Benchmark builds have been re-enabled.
- Unique operator no longer called on invalid elements.
### Known issues
- Device scan unit test currently failing due to LLVM bug.

## [rocPRIM-2.10.7 for ROCm 4.0.0]
### Added
- No new features

## [rocPRIM-2.10.6 for ROCm 3.10]
### Optimizations
- Updates to DPP instructions for warp shuffle
### Known issues
- Benchmark builds are disabled due to compiler bug.

## [rocPRIM-2.10.5 for ROCm 3.9.0]
### Added
- Added HIP cmake dependency
### Optimizations
- Updates to warp shuffle for gfx10
- Disable DPP functions on gfx10++
### Known issues
- Benchmark builds are disabled due to compiler bug.

## [rocPRIM-2.10.4 for ROCm 3.8.0]
### Fixed
- Fix for rocPRIM texture cache iterator
### Known issues
- None

## [rocPRIM-2.10.3 for ROCm 3.7.0]
### Fixed
- Package dependency correct to hip-rocclr
### Known issues
- rocPRIM texture cache iterator functionality is broken in the runtime. It will be fixed in the next release. Please use the prior release if calling this function.

## [rocPRIM-2.10.2 for ROCm 3.6.0]
### Added
- No new features

## [rocPRIM-2.10.1 for ROCm 3.5.1]
### Fixed
- Point release with compilation fix.

## [rocPRIM-2.10.1 for ROCm 3.5.0]
### Added
- Improved tests with fixed and random seeds for test data
- Network interface improvements with API v3
### Changed
- Switched to hip-clang as default compiler
- CMake searches for rocPRIM locally first; downloads from github if local search fails
