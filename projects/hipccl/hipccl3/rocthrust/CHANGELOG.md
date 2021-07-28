# Change Log for rocThrust

Full documentation for rocThrust is available at [https://rocthrust.readthedocs.io/en/latest/](https://rocthrust.readthedocs.io/en/latest/)

## (Unreleased) rocThrust-2.11.2 for ROCm 4.5.0
### Addded
- Initial HIP on Windows support. See README for instructions on how to build and install.
### Changed
- Packaging split into a runtime package called rocthrust and a development package called rocthrust-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.

## [Unreleased rocThrust-2.11.1 for ROCm 4.4.0]
### Added
- gfx1030 support
- Address Sanitizer build option
### Fixed
- async_transform unit test failure fixed.

## [Unreleased rocThrust-2.11.0 for ROCm 4.3.0]
### Added
- Updated to match upstream Thrust 1.11
- gfx90a support added
- gfx803 support re-enabled

## [rocThrust-2.10.9 for ROCm 4.2.0]
### Added
- Updated to match upstream Thrust 1.10
### Changed
- Minimum cmake version required for building rocThrust is now 3.10.2
### Fixed
- Size zero inputs are now properly handled with newer ROCm builds that no longer allow zero-size kernel grid/block dimensions
- Warning of unused results fixed.

## [rocThrust-2.10.8 for ROCm 4.1.0]
### Added
- No new features

## [rocThrust-2.10.7 for ROCm 4.0.0]
### Added
- Updated to upstream Thrust 1.10.0
- Implemented runtime error for unsupported algorithms and disabled respective tests.
- Updated CMake to use downloaded rocPRIM.

## [rocThrust-2.10.6 for ROCm 3.10]
### Added
- Added copy_if on device test case
### Known issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.5 for ROCm 3.9.0]
### Added
- Updated to upstream Thrust 1.9.8
- New test cases for device-side algorithms
### Fixes
- Bugfix for binary search
- Implemented workarounds for hipStreamDefault hang
### Known issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.4 for ROCm 3.8.0]
### Added
- No new features
### Known issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.3 for ROCm 3.7.0]
### Added
- Updated to upstream Thrust 1.9.4
### Changed
- Package dependecy change to rocprim only
### Known issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.2 for ROCm 3.6.0]
### Added
- No new features
### Known Issues
- ROCm support for device malloc has been disabled. As a result, rocThrust functionality dependent on device malloc does not work. Please avoid using device launched thrust::sort and thrust::sort_by_key. Host launched functionality is not impacted. A partial enablement of device malloc is possible by setting HIP_ENABLE_DEVICE_MALLOC to 1. Thrust::sort and thrust::sort_by_key may work on certain input sizes but is not recommended for production code.

## [rocThrust-2.10.1 for ROCm 3.5.0]
### Added
- Improved tests with fixed and random seeds for test data
### Changed
- CMake searches for rocThrust locally first; downloads from github if local search fails
### Deprecated
- HCC build deprecated
