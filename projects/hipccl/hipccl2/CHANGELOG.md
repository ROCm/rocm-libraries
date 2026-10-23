# Changelog for hipCCL (hipCCL2)

## hipCCL 2.8.0 for ROCm 10.2.0

### Added

* A `hipccl` package wrapping the three components under one umbrella version
  (`HIPCCL_VERSION`), with a `find_package(hipccl)` CMake config.
* For hipCCL2, rocPRIM, hipCUB, and rocThrust are at version 4.7.0.
* Headers install under `<prefix>/include/hipccl/<component>`, with
  `HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS` keeping the pre-hipCCL include paths
  working (symbolic links on Linux, copies on Windows).
* Project documentation under `../docs/`, buildable with `BUILD_DOCS=ON`.

### Deprecated

* This hipCCL2 folder is meant as a temporary pre-CCCL 3.0 compatibility layer and 
  will be removed in a future release.  There are no planned future updates for hipCCL2.
* The legacy include paths are temporary compatibility aids and will be removed
  in a future release.
* The old rocm-libraries project folders for rocPRIM, hipCUB, and rocThrust 
  are deprecated and will be removed in a future release.  