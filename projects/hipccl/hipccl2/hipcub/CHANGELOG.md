# Changelog for hipCUB

Full documentation for hipCUB is available at [https://rocm.docs.amd.com/projects/hipCUB/en/latest/](https://rocm.docs.amd.com/projects/hipCUB/en/latest/).

## hipCUB-3.5.0 for ROCm 6.5.0

### Added

* Added single pass operators in `agent/single_pass_scan_operators.hpp` which contains the following API:
  * `BlockScanRunningPrefixOp`
  * `ScanTileStatus`
  * `ScanTileState`
  * `ReduceByKeyScanTileState`
  * `TilePrefixCallbackOp`
* Added gfx950 support.
* Added an overload of `BlockScan::InclusiveScan` that accepts an initial value to seed the scan.
* Added an overload of `WarpScan::InclusiveScan` that accepts an initial value to seed the scan.
* `UnrolledThreadLoad`, `UnrolledCopy`, and `ThreadLoadVolatilePointer` were added to align hipCUB with CUB.
* `ThreadStoreVolatilePtr` and the `IterateThreadStore` struct were added to align hipCUB with CUB.

### Changed

* The NVIDIA backend now requires CUB, Thrust, and libcu++ 2.6.0. If they aren't found, they will be downloaded from the NVIDIA CCCL repository.
* Updated `thread_load` and `thread_store` to align hipCUB with CUB.
* All kernels now have hidden symbol visibility. All symbols now have inline namespaces that include the library version, (for example, hipcub::HIPCUB_300400_NS::symbol instead of hipcub::symbol), letting the user link multiple libraries built with different versions of hipCUB.

### Known issues

* `BlockAdjacentDifference::FlagHeads`, `BlockAdjacentDifference::FlagTails` and `BlockAdjacentDifference::FlagHeadsAndTails` have been removed from hipCUB's CUB backend. They were already deprecated as of version 2.12.0 of hipCUB and they were removed from CCCL (CUB) as of CCCL's 2.6.0 release.
* `BlockScan::InclusiveScan` for the NVIDIA backend does not compute the block aggregate correctly when passing an initial value parameter. This behavior is not matched by the AMD backend.

### Removed
* This release removes support for custom builds on gfx940 and gfx941.

### Upcoming Changes

* `BlockAdjacentDifference::FlagHeads`, `BlockAdjacentDifference::FlagTails` and `BlockAdjacentDifference::FlagHeadsAndTails` were deprecated as of version 2.12.0 of hipCUB, and will be removed from the rocPRIM backend in a future release for the next ROCm major version (ROCm 7.0.0).

## hipCUB-3.4.0 for ROCm 6.4.0

### Added
* Added regression tests to `rtest.py`. These tests recreate scenarios that have caused hardware problems in past emulation environments. Use `python rtest.py [--emulation|-e|--test|-t]=regression` to run these tests.
* Added extended tests to `rtest.py`. These tests are extra tests that did not fit the criteria of smoke and regression tests. These tests will take much longer to run relative to smoke and regression tests. Use `python rtest.py [--emulation|-e|--test|-t]=extended` to run these tests.
* Added `ForEach`, `ForEachN`, `ForEachCopy`, `ForEachCopyN` and `Bulk` functions to have parity with CUB.
* Added the `hipcub::CubVector` type for CUB parity.
* Added `--emulation` option for `rtest.py`
  * Unit tests can be run with `[--emulation|-e|--test|-t]=<test_name>`
* Added `DeviceSelect::FlaggedIf` and its inplace overload.
* Added CUB macros missing from hipCUB: `HIPCUB_MAX`, `HIPCUB_MIN`, `HIPCUB_QUOTIENT_FLOOR`, `HIPCUB_QUOTIENT_CEILING`, `HIPCUB_ROUND_UP_NEAREST` and `HIPCUB_ROUND_DOWN_NEAREST`.
* Added `hipcub::AliasTemporaries` function for CUB parity.

### Changed
* Removed usage of `std::unary_function` and `std::binary_function` in `test_hipcub_device_adjacent_difference.cpp`
* Changed the subset of tests that are run for smoke tests such that the smoke test will complete with faster run-time and to never exceed 2GB of vram usage. Use `python rtest.py [--emulation|-e|--test|-t]=smoke` to run these tests.
* The `rtest.py` options have changed. `rtest.py` is now run with at least either `--test|-t` or `--emulation|-e`, but not both options.
* The NVIDIA backend now requires CUB, Thrust and libcu++ 2.5.0. If it is not found it will be downloaded from the NVIDIA CCCL repository.
* Changed the C++ version from 14 to 17. C++14 will be deprecated in the next major release.

### Known issues
* When building on Windows using HIP SDK for ROCm 6.4, ``hipMalloc`` returns ``hipSuccess`` even when the size passed to it is too large and the allocation fails. Because of this, limits have been set for the maximum test case sizes for some unit tests such as HipcubDeviceRadixSort's SortKeysLargeSizes .

## hipCUB 3.3.0 for ROCm 6.3.0

### Added

* Support for large indices in `hipcub::DeviceSegmentedReduce::*` has been added, with the exception of `DeviceSegmentedReduce::Arg*`. Although rocPRIM's backend provides support for all reduce variants, CUB does not support large indices in `DeviceSegmentedReduce::Arg*`. For this reason, large index support is not available for `hipcub::DeviceSegmentedReduce::Arg*`.

### Changed

* Changed the default value of `rmake.py -a` to `default_gpus`. This is equivalent to `gfx906:xnack-,gfx1030,gfx1100,gfx1101,gfx1102,gfx1151,gfx1200,gfx1201`.
* The NVIDIA backend now requires CUB, Thrust, and libcu++ 2.3.2.

### Resolved issues
* Fixed an issue in `rmake.py` where the list storing cmake options would contain individual characters instead of a full string of options.
* Fixed an issue where `config.hpp` was not included in all hipCUB headers, resulting in build errors.

## hipCUB-3.2.0 for ROCm 6.2.0

### Added
* Add `DeviceCopy` function to have parity with CUB.
* In the rocPRIM backend, added `enum WarpExchangeAlgorithm`, which is used as the new optional template argument for `WarpExchange`.
  * The potential values for the enum are `WARP_EXCHANGE_SMEM` and `WARP_EXCHANGE_SHUFFLE`.
  * `WARP_EXCHANGE_SMEM` stands for the previous algorithm, while `WARP_EXCHANGE_SHUFFLE` performs the exchange via shuffle operations.
  * `WARP_EXCHANGE_SHUFFLE` does not require any pre-allocated shared memory, but the `ItemsPerThread` must be a divisor of `WarpSize`.
* Added `tuple.hpp` which defines templates `hipcub::tuple`, `hipcub::tuple_element`, `hipcub::tuple_element_t` and `hipcub::tuple_size`.
* Added new overloaded member functions to `BlockRadixSort` and `DeviceRadixSort` that expose a `decomposer` argument. Keys of a custom
  type (`key_type`) can be sorted via these overloads, if an appropriate decomposer is passed. The decomposer has to implement
  `operator(const key_type&)` which returns a `hipcub::tuple` of references pointing to members of `key_type`.

* On AMD GPUs (using the HIP backend), it is possible to issue hipCUB API calls inside of
  hipGraphs, with several exceptions:
   * CachingDeviceAllocator
   * GridBarrierLifetime
   * DeviceSegmentedRadixSort
   * DeviceRunLengthEncode
  Currently, these classes rely on one or more synchronous calls to function correctly. Because of this, they cannot be used inside of hipGraphs.

### Changed

* The NVIDIA backend now requires CUB, Thrust and libcu++ 2.2.0. If it is not found it will be downloaded from the NVIDIA CCCL repository.

### Fixed

* Fixed the derivation for the accumulator type for device scan algorithms in the rocPRIM backend being different compared to CUB.
  It now derives the accumulator type as the result of the binary operator.
* `debug_synchronous` has been deprecated in hipCUB-2.13.2, and it no longer has any effect. With this release, passing `debug_synchronous`
  to the device functions results in a deprecation warning both at runtime and at compile time.
  * The synchronization that was previously achievable by passing `debug_synchronous=true` can now be achieved at compile time
    by setting the `CUB_DEBUG_SYNC` (or higher debug level) or the `HIPCUB_DEBUG_SYNC` preprocessor definition.
  * The compile time deprecation warnings can be disabled by defining the `HIPCUB_IGNORE_DEPRECATED_API` preprocessor definition.

## hipCUB-3.1.0 for ROCm 6.1.0

### Changes

* CUB backend references CUB and Thrust version 2.1.0
* Updated the `HIPCUB_HOST_WARP_THREADS` macro definition to match `host_warp_size` changes
  from rocPRIM 3.0
* Implemented `__int128_t` and `__uint128_t` support for `radix_sort`

### Fixes

* Build issues with `rmake.py` on Windows when using Visual Studio 2017 15.8 or later (due to a
  breaking fix with extended aligned storage)

### Additions

* Interface `DeviceMemcpy::Batched` for batched memcpy from rocPRIM and CUB

## hipCUB-3.0.0 for ROCm 6.0.0

### Changes

* Removed `DOWNLOAD_ROCPRIM`
  * You can force rocPRIM to download using `DEPENDENCIES_FORCE_DOWNLOAD`

## hipCUB-2.13.2 for ROCm 5.7.0

### Changes

* CUB backend references CUB and Thrust version 2.0.1.
* Fixed `DeviceSegmentedReduce::ArgMin` and `DeviceSegmentedReduce::ArgMax` by returning the
  segment-relative index instead of the absolute one
* Fixed `DeviceSegmentedReduce::ArgMin` for inputs where the segment minimum is smaller than the
  value returned for empty segments; an equivalent fix is applied to `DeviceSegmentedReduce::ArgMax`

### Known issues

* `debug_synchronous` no longer works on the CUDA platform; use `CUB_DEBUG_SYNC` instead
* `DeviceReduce::Sum` doesn't compile on the CUDA platform for mixed extended-floating-point or
  floating-point InputT and OutputT types
* `DeviceHistogram::HistogramEven` fails on CUDA platform for `[LevelT, SampleIteratorT] = [int, int]`.
* `DeviceHistogram::MultiHistogramEven` fails on CUDA platform for
  `[LevelT, SampleIteratorT] = [int, int/unsigned short/float/double]` and
  `[LevelT, SampleIteratorT] = [float, double]`

## hipCUB-2.13.1 for ROCm 5.5.0

### Additions

* Benchmarks for `BlockShuffle`, `BlockLoad`, and `BlockStore`

### Changes

* The CUB backend references CUB and Thrust version 1.17.2
* Improved benchmark coverage for:
  * `BlockScan` by adding `ExclusiveScan`
  * `BlockRadixSort` by adding `SortBlockedToStriped`
  * `WarpScan` by adding `Broadcast`
* Removed references to, and workarounds for, the deprecated hcc

### Known issues

* `BlockRadixRankMatch` is currently broken for the rocPRIM backend
* `BlockRadixRankMatch` with a warp size that does not divide exactly by the block size is broken for
  the CUB backend

## hipCUB-2.13.0 for ROCm 5.4.0

### Additions

* CMake functionality improves build parallelism for the test suite that splits compilation units by
function or parameters
* New overload for `BlockAdjacentDifference::SubtractLeftPartialTile`, which takes a predecessor item

### Changes

* Improved build parallelism of the test suite by splitting up large compilation units for `DeviceRadixSort`, `DeviceSegmentedRadixSort`, and `DeviceSegmentedSort`
* The CUB backend references CUB and Thrust version 1.17.1

### Known issues

* `BlockRadixRankMatch` is currently broken for the rocPRIM backend
* `BlockRadixRankMatch` with a warp size that does not divide exactly by the block size is broken for
  the CUB backend

## hipCUB-2.12.0 for ROCm 5.3.0

### Additions

* `UniqueByKey` device algorithm
* `SubtractLeft`, `SubtractLeftPartialTile`, `SubtractRight`, and `SubtractRightPartialTile` overload in
  `BlockAdjacentDifference`
  * The old overloads (`FlagHeads`, `FlagTails`, `FlagHeadsAndTails`) are deprecated
* `DeviceAdjacentDifference` algorithm
* Extended benchmark suite of `DeviceHistogram`, `DeviceScan`, `DevicePartition`, `DeviceReduce`,
`DeviceSegmentedReduce`, `DeviceSegmentedRadixSort`, `DeviceRadixSort`, `DeviceSpmv`,
`DeviceMergeSort`, and `DeviceSegmentedSort`

### Changes

* Obsolete type traits defined in `util_type.hpp`; use the standard library equivalents instead
* CUB backend references CUB and Thrust version 1.16.0
* `DeviceRadixSort` `num_items` parameter type is now templated instead of being an int
  * If an integral type with a maximum size of 4 bytes is passed (an int), the former logic applies;
    otherwise, the algorithm uses a larger indexing type that makes it possible to sort input data over
    $2^{32}$ elements

## hipCUB-2.11.1 for ROCm 5.2.0

### Additions

* Packages for tests and benchmark executables on all supported operating systems using CPack

## hipCUB-2.11.0 for ROCm 5.1.0

### Additions

* Device segmented sort
* `WarpMergeSort`, `WarpMask`, and thread sort from CUB 1.15.0 are supported in hipCUB
* Device three-way partition

### Changes

* `device_scan` and `device_segmented_scan`: `inclusive_scan` now uses the `input-type` as
  `accumulator-type`; `exclusive_scan` uses `initial-value-type`.
  * This changes the behavior of:
    * Small-size input types with large-size output types (e.g., short input, int output)
    * Low-res input with high-res output (e.g., float input, double output)
* Block merge sort no longer supports non-power of two block sizes

### Known issues

* Grid unit test hangs on HIP for Windows

## hipCUB-2.10.13 for ROCm 5.0.0

### Fixes

* Added missing includes to `hipcub.hpp`

### Additions

* Bfloat16 support to test cases (`device_reduce` and `device_radix_sort`)
* Device merge sort
* Block merge sort
* API update to CUB 1.14.0

### Changes

* The `SetupNVCC.cmake` automatic target selector selects all of the capabilities for all available cards
  with the NVIDIA backend

## hipCUB-2.10.12 for ROCm 4.5.0

### Additions

* Initial HIP on Windows support

### Changes

* Packaging changed to a development package (named `hipcub-dev` for `.deb` packages and
  `hipcub-devel` for `.rpm` packages). Because hipCUB is a header-only library, there is no runtime
  package. To aid in the transition, the development package sets the `provides` field to `hipcub`, so
  existing packages that are dependent on hipCUB can continue to work. This `provides` feature is
  introduced as a deprecated feature because it will be removed in a future ROCm release.

## hipCUB-2.10.11 for ROCm 4.4.0

### Additions

* gfx1030 support added
* AddressSanitizer build option

### Fixes

* `BlockRadixRank` unit test failure

## hipCUB-2.10.10 for ROCm 4.3.0

### Additions

* `DiscardOutputIterator` to backend header

## hipCUB-2.10.9 for ROCm 4.2.0

### Additions

* Support for `TexObjInputIterator` and `TexRefInputIterator`
* Support for `DevicePartition`

### Changes

* The minimum CMake version required is now 3.10.2
* The CUB backend has been updated to 1.11.0

### Fixes

* Benchmark build
* NVCC build

## hipCUB-2.10.8 for ROCm 4.1.0

### Added

* Support for `DiscardOutputIterator`

## hipCUB-2.10.7 for ROCm 4.0.0

* No changes

## hipCUB-2.10.6 for ROCm 3.10

* No changes

## hipCUB-2.10.5 for ROCm 3.9.0

* No changes

## hipCUB-2.10.4 for ROCm 3.8.0

* No changes

## hipCUB-2.10.3 for ROCm 3.7.0

* No changes

## hipCUB-2.10.2 for ROCm 3.6.0

* No changes

## hipCUB-2.10.1 for ROCm 3.5.0

### Additions

* Improved tests with fixed and random seeds for test data

### Changes

* Switched to hip-clang as default compiler
* CMake searches for rocPRIM locally first and, if not found, downloads it from GitHub

### Deprecations

* HCC build

### Known issues

* The following unit test failures (due to issues in ROCclr runtime) have been observed:
  * `BlockDiscontinuity`
  * `BlockExchange`
  * `BlockHistogram`
  * `BlockRadixSort`
  * `BlockReduce`
  * `BlockScan`
