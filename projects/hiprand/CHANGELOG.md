# Changelog for hipRAND

Documentation for hipRAND is available at
[https://rocm.docs.amd.com/projects/hipRAND/en/latest/](https://rocm.docs.amd.com/projects/hipRAND/en/latest/).

## hipRAND 3.1.0 for ROCm 7.1

### Resolved issues

* Updated error handling for several hipRAND unit tests to accomodate the new hipGetLastError behaviour that was introduced in ROCm 7.0.
As of ROCm 7.0, the internal error state is cleared on each call to `hipGetLastError` rather than on every HIP API call.

## hipRAND 3.0.0 for ROCm 7.0

### Added

* gfx950 support

### Changed

* Deprecated hipRAND's Fortran API in favor of hipfort.

### Removed

* Removed C++14 support, only C++17 is supported.

## hipRAND 2.12.0 for ROCm 6.4.0

### Changed

* When building hipRAND on Windows, use `HIP_PATH` (instead of the former `HIP_DIR`) to specify the path to the HIP SDK installation.
  * When building with the `rmake.py` script, if `HIP_PATH` is not set, it will default to `C:\hip`.

### Resolved issues

* Fixed an issue that was causing hipRAND build failures on Windows when the HIP SDK was installed to a location with a path that contains spaces.

## hipRAND-2.11.1 for ROCm 6.2.4

### Added

* GFX1151 Support

## hipRAND 2.11.0 for ROCm 6.2.0

### Added

* Added support for setting generator output ordering in C and C++ API
* `hiprandCreateGeneratorHost` dispatches to the host generator in the rocRAND backend instead of returning with `HIPRAND_STATUS_NOT_IMPLEMENTED`
* Added the option to create a host generator to the Fortran wrapper
* Added the option to create a host generator to the Python wrapper

### Changed

* Updated the default value for the `-a` argument from `rmake.py` to `gfx906:xnack-,gfx1030,gfx1100,gfx1101,gfx1102,gfx1151,gfx1200,gfx1201`.
* For internal testing with HMM the environment variable `ROCRAND_USE_HMM` was used in previous
  versions, it is now changed to `HIPRAND_USE_HMM`.
* The device API documentation is improved in this version.
* Static library: moved all internal symbols to namespaces to avoid potential symbol name collisions when linking.

### Removed

* Removed the option to build hipRAND as a submodule to rocRAND
* Removed references to, and workarounds for, the deprecated `hcc`
* Support for finding rocRAND based on the environment variable `ROCRAND_DIR` has been removed
  `ROCRAND_PATH` can be used instead.

### Resolved issues

* Fixed an issue in `rmake.py` where the list storing cmake options would contain individual characters instead of a full string of options.
* Build error when using Clang++ directly due to unsupported references to `amdgpu-target`

## hipRAND-2.10.17 for ROCm 5.6.0

### Fixes

* Fixed benchmark and unit test builds on Windows

## hipRAND-2.10.16 for ROCm 5.5.0

### Additions

* rocRAND backend support for Sobol 64, Scrambled Sobol 32 and 64, and MT19937
* `hiprandGenerateLongLong` for generating 64-bit uniformly distributed integers with Sobol 64 and
  Scrambled Sobol 64
* Accessor methods for Sobol 32 and 64 direction vectors and constants:
  * Enum `hiprandDirectionVectorSet_t` for direction vector set selection
  * `hiprandGetDirectionVectors32(...)`
  * `hiprandGetDirectionVectors32(...)`
  * `hiprandGetScrambleConstants32(...)`
  * `hiprandGetScrambleConstants32(...)`

### Changes

* Python 2.7 is no longer officially supported.

## hipRAND for ROCm 5.2.0

### Additions

* Backward compatibility for the deprecated `#include <hiprand.h>` using wrapper header files
* Packages for test and benchmark executables on all supported operating systems using CPack

## hipRAND for ROCm 5.0.0

### Additions

* Initial split from rocRAND
