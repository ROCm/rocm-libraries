# Change Log for rocPRIM

Full documentation for rocPRIM is available at [https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/](https://codedocs.xyz/ROCmSoftwarePlatform/rocPRIM/)

## [Unreleased rocPRIM-2.10.7 for ROCm 4.0.0]
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
