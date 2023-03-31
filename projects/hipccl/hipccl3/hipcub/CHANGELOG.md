# Change Log for hipCUB

See README.md on how to build the hipCUB documentation using Doxygen.

## (Unreleased) hipCUB-2.13.1 for ROCm 5.5.0
### Added
- Benchmarks for `BlockShuffle`, `BlockLoad`, and `BlockStore`.
### Changed
- CUB backend references CUB and Thrust version 1.17.2.
- Improved benchmark coverage of `BlockScan` by adding `ExclusiveScan`, benchmark coverage of `BlockRadixSort` by adding `SortBlockedToStriped`, and benchmark coverage of `WarpScan` by adding `Broadcast`.
- Updated `docs` directory structure to match the standard of [rocm-docs-core](https://github.com/RadeonOpenCompute/rocm-docs-core).
### Known Issues
- `BlockRadixRankMatch` is currently broken under the rocPRIM backend.
- `BlockRadixRankMatch` with a warp size that does not exactly divide the block size is broken under the CUB backend.

## (Unreleased) hipCUB-2.13.0 for ROCm 5.4.0
### Added
- CMake functionality to improve build parallelism of the test suite that splits compilation units by
function or by parameters.
- New overload for `BlockAdjacentDifference::SubtractLeftPartialTile` that takes a predecessor item.
### Changed
- Improved build parallelism of the test suite by splitting up large compilation units for `DeviceRadixSort`, 
`DeviceSegmentedRadixSort` and `DeviceSegmentedSort`.
- CUB backend references CUB and Thrust version 1.17.1.

### Known Issues
- `BlockRadixRankMatch` is currently broken under the rocPRIM backend.
- `BlockRadixRankMatch` with a warp size that does not exactly divide the block size is broken under the CUB backend.

## hipCUB-2.12.0 for ROCm 5.3.0
### Added
- UniqueByKey device algorithm
- SubtractLeft, SubtractLeftPartialTile, SubtractRight, SubtractRightPartialTile overloads in BlockAdjacentDifference.
  - The old overloads (FlagHeads, FlagTails, FlagHeadsAndTails) are deprecated.
- DeviceAdjacentDifference algorithm.
- Extended benchmark suite of `DeviceHistogram`, `DeviceScan`, `DevicePartition`, `DeviceReduce`,
`DeviceSegmentedReduce`, `DeviceSegmentedRadixSort`, `DeviceRadixSort`, `DeviceSpmv`, `DeviceMergeSort`,
`DeviceSegmentedSort`
### Changed
- Obsolated type traits defined in util_type.hpp. Use the standard library equivalents instead.
- CUB backend references CUB and thrust version 1.16.0.
- DeviceRadixSort's num_items parameter's type is now templated instead of being an int.
  - If an integral type with a size at most 4 bytes is passed (i.e. an int), the former logic applies.
  - Otherwise the algorithm uses a larger indexing type that makes it possible to sort input data over 2**32 elements.

## hipCUB-2.11.1 for ROCm 5.2.0
### Added
- Packages for tests and benchmark executable on all supported OSes using CPack.

## hipCUB-2.11.0 for ROCm 5.1.0
### Added
- Device segmented sort
- Warp merge sort, WarpMask and thread sort from cub 1.15.0 supported in hipCUB
- Device three way partition
### Changed
- Device_scan and device_segmented_scan: inclusive_scan now uses the input-type as accumulator-type, exclusive_scan uses initial-value-type.
  - This particularly changes behaviour of small-size input types with large-size output types (e.g. short input, int output).
  - And low-res input with high-res output (e.g. float input, double output)
  - Block merge sort no longer supports non power of two blocksizes
### Known Issues
  - grid unit test hanging on HIP on Windows

## hipCUB-2.10.13 for ROCm 5.0.0
### Fixed
- Added missing includes to hipcub.hpp
### Added
- Bfloat16 support to test cases (device_reduce & device_radix_sort)
- Device merge sort
- Block merge sort
- API update to CUB 1.14.0
### Changed
- The SetupNVCC.cmake automatic target selector select all of the capabalities of all available card for NVIDIA backend.

## hipCUB-2.10.12 for ROCm 4.5.0
### Added
- Initial HIP on Windows support. See README for instructions on how to build and install.
### Changed
- Packaging changed to a development package (called hipcub-dev for `.deb` packages, and hipcub-devel for `.rpm` packages). As hipCUB is a header-only library, there is no runtime package. To aid in the transition, the development package sets the "provides" field to provide the package hipcub, so that existing packages depending on hipcub can continue to work. This provides feature is introduced as a deprecated feature and will be removed in a future ROCm release.

## [hipCUB-2.10.11 for ROCm 4.4.0]
### Added
- gfx1030 support added.
- Address Sanitizer build option
### Fixed
- BlockRadixRank unit test failure fixed.

## [hipCUB-2.10.10 for ROCm 4.3.0]
### Added
- DiscardOutputIterator to backend header

## [hipCUB-2.10.9 for ROCm 4.2.0]
### Added
- Support for TexObjInputIterator and TexRefInputIterator
- Support for DevicePartition
### Changed
- Minimum cmake version required is now 3.10.2
- CUB backend has been updated to 1.11.0
### Fixed
- Benchmark build fixed
- nvcc build fixed

## [hipCUB-2.10.8 for ROCm 4.1.0]
### Added
- Support for DiscardOutputIterator

## [hipCUB-2.10.7 for ROCm 4.0.0]
### Added
- No new features

## [hipCUB-2.10.6 for ROCm 3.10]
### Added
- No new features

## [hipCUB-2.10.5 for ROCm 3.9.0]
### Added
- No new features

## [hipCUB-2.10.4 for ROCm 3.8.0]
### Added
- No new features

## [hipCUB-2.10.3 for ROCm 3.7.0]
### Added
- No new features

## [hipCUB-2.10.2 for ROCm 3.6.0]
### Added
- No new features

## [hipCUB-2.10.1 for ROCm 3.5.0]
### Added
- Improved tests with fixed and random seeds for test data
### Changed
- Switched to hip-clang as default compiler
- CMake searches for rocPRIM locally first; downloads from github if local search fails
### Deprecated
- HCC build deprecated
### Known Issues
- The following unit test failures have been observed. These are due to issues in rocclr runtime.
    - BlockDiscontinuity
    - BlockExchange
    - BlockHistogram
    - BlockRadixSort
    - BlockReduce
    - BlockScan
