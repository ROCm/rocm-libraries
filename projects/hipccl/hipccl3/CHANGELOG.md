# Changelog for hipCCL (hipCCL3)

## hipCCL 3.1.0 for ROCm 10.3.0

### Added

* A unified `hipccl` project that builds rocPRIM, hipCUB, rocThrust, and
  libhipcxx from one superbuild, selected with `HIPCCL_BUILD_ROCPRIM`,
  `HIPCCL_BUILD_HIPCUB`, `HIPCCL_BUILD_ROCTHRUST`, and
  `HIPCCL_BUILD_LIBHIPCXX`. `HIPCCL_BUILD_ALL` builds rocPRIM, hipCUB, and
  rocThrust; libhipcxx is not included.
* For hipCCL3 in ROCm, rocPRIM, hipCUB, and rocThrust are at version 5.1.0.
  hipCUB tracks CCCL 3.1.4.
* hipCUB and rocThrust require libhipcxx 3.4 or later on the ROCm platform.
  Build it from the submodule with `HIPCCL_BUILD_LIBHIPCXX`, or provide an
  installed copy.
* A single `hipccl` package and a `find_package(hipccl)` CMake config with one
  umbrella version (`HIPCCL_VERSION`).
* Headers install under `<prefix>/include/hipccl/<component>`.
* `HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS`, which keeps the pre-hipCCL include
  paths (for example `<prefix>/include/rocprim`) working. These are symbolic
  links on Linux and copies on Windows, where they are refreshed only on
  reinstall.
* `HIPCCL_ENABLE_COVERAGE`, which enables `ROCPRIM_ENABLE_COVERAGE`,
  `HIPCUB_ENABLE_COVERAGE`, and `ROCTHRUST_ENABLE_COVERAGE` together.
* `HIPCCL_USE_LEGACY_API` (default `ON`), which enables the rocPRIM and hipCUB
  APIs from before CCCL 3.0 parity that are kept for backward compatibility.
* Project documentation under `../docs/`, buildable with `BUILD_DOCS=ON`.

### Deprecated

* The legacy include paths created by `HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS`
  are temporary compatibility aids and will be removed in a future release.
  Include from `hipccl/<component>` instead.
* The old rocm-libraries project folders for rocPRIM, hipCUB, and rocThrust
  are deprecated and will be removed in a future release.

### Known issues

* libhipcxx built from the submodule with `HIPCCL_BUILD_LIBHIPCXX` is used for
  the build only and is not installed. The installed hipCUB and rocThrust CMake
  packages depend on a `libhipcxx` package, so projects that use
  `find_package(hipcub)`, `find_package(rocthrust)`, or `find_package(hipccl)`
  need libhipcxx 3.4 or later installed separately. Install libhipcxx into the
  same prefix, or into a prefix on `CMAKE_PREFIX_PATH`, and build hipCCL against
  that installation when the result is meant to be installed. See the hipCCL
  build documentation.

### Upcoming changes

* `BUILD_CODE_COVERAGE` is deprecated and will be removed in a future release.
  Use `HIPCCL_ENABLE_COVERAGE` instead.
