# Changelog for hipFFT

Documentation for hipFFT is available at
[https://rocm.docs.amd.com/projects/hipFFT/en/latest/](https://rocm.docs.amd.com/projects/hipFFT/en/latest/).

## hipFFT 1.0.22 (unreleased)

## hipFFT 1.0.21 for ROCm 7.1.0

### Added

* Improved test coverage of multi-stream plans.
* Improved test coverage of user-specified work areas.
* Improved test coverage of default stride calculation.
* **[Experimental]** Introduced the hipFFTW library, interfacing rocFFT on AMD platforms using the same symbols as FFTW3 (with partial support).

## hipFFT 1.0.20 for ROCm 7.0.0

### Added

* Added gfx950 support.

### Removed

* Removed hipfft-rider legacy compatibility from clients
* Remove support for the gfx940 and gfx941 targets from the client programs.

## hipFFT 1.0.18 for ROCm 6.4.0

### Added

* Implemented the `hipfftMpAttachComm`, `hipfftXtSetDistribution`, and `hipfftXtSetSubformatDefault` APIs to allow
  computing FFTs that are distributed between multiple MPI (Message Passing Interface) processes.  These APIs can be enabled
  with the `HIPFFT_MPI_ENABLE` CMake option, which defaults to `OFF`.
* The backend FFT library called by hipFFT must support MPI for these APIs to work.

### Changed

* Building with the address sanitizer option sets xnack+ for the relevant GPU
  architectures.
* Use find_package CUDAToolkit instead of CUDA in cmake for modern-cmake compatibility.
* The `AMDGPU_TARGETS` build variable should be replaced with `GPU_TARGETS`. `AMDGPU_TARGETS` is deprecated.

### Resolved issues

* Fixed client packages to depend on hipRAND instead of rocRAND.

## hipFFT 1.0.17 for ROCm 6.3.0

### Added

* Support for the gfx1151, gfx1200, and gfx1201 architectures
* hipfft-test now includes a --smoketest option.

### Changed

* The AMD backend is now compiled using amdclang++ instead of hipcc. The NVIDIA CUDA backend still uses hipcc-nvcc.
* CLI11 replaces Boost Program Options as the command line parser for clients.

## hipFFT 1.0.16 for ROCm 6.2.4

### Changed
* Support gfx1151 architecture.

## hipFFT 1.0.15 for ROCm 6.2.0

### Fixes

* Added hip::host as a public link library, as hipfft.h includes HIP runtime headers.
* Prevent C++ exceptions leaking from public API functions.
* Make output of hipfftXt match cufftXt in geometry and alignment for 2D and 3D FFTs.

## hipFFT 1.0.14 for ROCm 6.1.0

### Changes

* When building hipFFT from source, rocFFT code no longer needs to be initialized as a git submodule.

### Fixes

* Fixed error when creating length-1 plans.

## hipFFT 1.0.13 for ROCm 6.0.0

### Changes

* `hipfft-rider` has been renamed to `hipfft-bench`; it is controlled by the `BUILD_CLIENTS_BENCH`
  CMake option (note that a link for the old file name is installed, and the old `BUILD_CLIENTS_RIDER`
  CMake option is accepted for backwards compatibility, but both will be removed in a future release)
* Binaries in debug builds no longer have a `-d` suffix
* The minimum rocFFT required version has been updated to 1.0.21

### Additions

* `hipfftXtSetGPUs`, `hipfftXtMalloc, hipfftXtMemcpy`, `hipfftXtFree`, and `hipfftXtExecDescriptor` APIs
  have been implemented to allow FFT computing on multiple devices in a single process

## hipFFT 1.0.12 for ROCm 5.6.0

### Additions

* `hipfftXtMakePlanMany`, `hipfftXtGetSizeMany`, and `hipfftXtExec` APIs have been implemented to
  allow half-precision transform requests

### Changes

* Added the `--precision` argument to benchmark and test clients (`--double` is still accepted, but has
  been deprecated as a method to request a double-precision transform)

## hipFFT 1.0.11 for ROCm 5.5.0

### Fixes

* Fixed old version ROCm include and lib folders that were not removed during upgrades

## hipFFT 1.0.10 for ROCm 5.4.0

### Additions

* Added the `hipfftExtPlanScaleFactor` API to efficiently multiply each output element of an FFT by a
  given scaling factor (result scaling must be supported in the backend FFT library)

### Changes

* rocFFT 1.0.19 or higher is now required for hipFFT builds on the rocFFT backend
* Data are initialized directly on GPUs using hipRAND
* Updated build files now use standard C++17

## hipFFT 1.0.9 for ROCm 5.3.0

### Changes

* Cleaned up build warnings
* GNUInstallDirs enhancements
* GoogleTest 1.11 is required

## hipFFT 1.0.8 for ROCm 5.2.0

### Additions

* Added file and folder reorganization changes with backward compatibility support when using
  rocm-cmake wrapper functions
* New packages for test and benchmark executables on all supported operating systems that use
  CPack
* Implemented `hipfftMakePlanMany64` and `hipfftGetSizeMany64`

## hipFFT 1.0.7 for ROCm 5.1.0

### Changes

* Use `fft_params` struct for accuracy and benchmark clients

## hipFFT 1.0.6 for ROCm 5.0.0

### Fixes

* Incorrect reporting of rocFFT version

### Changes

* Unconditionally enabled callback functionality: On the CUDA backend, callbacks only run
  correctly when hipFFT is built as a static library, and linked against the static cuFFT library

## hipFFT 1.0.5 for ROCm 4.5.0

### Additions

* Added support for Windows 10 as a build target

### Changes

* Packaging has been split into a runtime package (`hipfft`) and a development package
  (`hipfft-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.

## hipFFT 1.0.4 for ROCm 4.4.0

### Fixes

* Add calls to rocFFT setup and cleanup
* CMake fixes for clients and backend support

### Additions

* Added support for Windows 10 as a build target

## hipFFT 1.0.3 for ROCm 4.3.0

### Fixes

* CMake updates

### Additions

* New callback API in `hipfftXt.h` header

## hipFFT 1.0.2 for ROCm 4.2.0

* No changes

## hipFFT 1.0.1 for ROCm 4.1.0

### Fixes

* Batch support for `hipfftMakePlanMany`
* Work area handling during plan creation and `hipfftSetWorkArea`
* Honour `autoAllocate` flag

### Changes

* Testing infrastructure reuses code from [rocFFT](https://github.com/ROCmSoftwarePlatform/rocFFT)
