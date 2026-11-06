# Changelog for hipCCL (hipCCL3)

## hipCCL 3.1.0 for ROCm 10.3.0

### Added

* `HIPCCL_ENABLE_COVERAGE`, which enables `ROCPRIM_ENABLE_COVERAGE`, `HIPCUB_ENABLE_COVERAGE`, and `ROCTHRUST_ENABLE_COVERAGE` together.

### Removed

* Deprecated `BUILD_CODE_COVERAGE`, use `HIPCCL_ENABLE_COVERAGE` instead.

## hipCCL 3.0.0 for ROCM 10.2.0

### Added

* A unified `hipccl` project that builds rocPRIM, hipCUB, rocThrust, and
  (optionally) libhipcxx from one superbuild, selected with
  `HIPCCL_BUILD_ROCPRIM`, `HIPCCL_BUILD_HIPCUB`, `HIPCCL_BUILD_ROCTHRUST`, and
  `HIPCCL_BUILD_LIBHIPCXX`.
* For hipCCL3 in ROCm, rocPRIM, hipCUB, and rocThrust are at version 5.0.0.
* A single `hipccl` package and a `find_package(hipccl)` CMake config with one
  umbrella version (`HIPCCL_VERSION`).
* Headers install under `<prefix>/include/hipccl/<component>`.
* `HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS`, which keeps the pre-hipCCL include
  paths (for example `<prefix>/include/rocprim`) working. These are symbolic
  links on Linux and copies on Windows, where they are refreshed only on
  reinstall.
* Project documentation under `docs/`, buildable with `BUILD_DOCS=ON`.

### Deprecated

* The legacy include paths created by `HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS`
  are temporary compatibility aids and will be removed in a future release.
  Include from `hipccl/<component>` instead.
* The old rocm-libraries project folders for rocPRIM, hipCUB, and rocThrust
  are deprecated and will be removed in a future release.
