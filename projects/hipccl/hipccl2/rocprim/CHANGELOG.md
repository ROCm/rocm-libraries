# Change Log for rocPRIM

Full documentation for rocPRIM is available at [https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/)

## [Unreleased rocPRIM-2.10.10 for ROCm 4.3.0]
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
