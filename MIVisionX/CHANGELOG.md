<p align="center"><img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/MIVisionX.png" /></p>

# Changelog for MIVisionX

The full documentation for MIVisionX is available at [https://rocm.docs.amd.com/projects/MIVisionX/en/latest/doxygen/html/index.html](https://rocm.docs.amd.com/projects/MIVisionX/en/latest/doxygen/html/index.html)

##  (unreleased) MIVisionX 3.2.0

### Changed

* Setup: OpenCV package install for Ubuntu
* AMD Clang is now the default CXX and C compiler
* The version of OpenMP included in the ROCm LLVM project is now used instead of `libomp-dev/devel`

### Known issues

* Installation on CentOS/RedHat/SLES requires the manual installation of the `FFMPEG` & `OpenCV` dev packages.
* Hardware decode requires the ROCm `graphics` usecase.

### Upcoming changes

* Optimized audio augmentations support for VX_RPP


##  MIVisionX 3.1.0 for ROCm 6.3.0

### Changed

* Setup: rocdecode install disabled
* Package: rocdecode dependency removed

### Optimizations

* Setup: only core dependency packages installed

### Known issues

* MIVisionX package installation on RedHat/SLES requires the manual installation of the `OpenCV` and `FFMPEG` development package.

### Upcoming changes

* Optimized audio augmentations support for VX_RPP

## MIVisionX 3.0.0 for ROCm 6.2.0

### Added

* Support has been added for the following:
  * Advanced GPUs
  * PreEmphasis Filter augmentation in openVX extensions
  * Spectrogram augmentation in openVX extensions
  * Downmix and ToDecibels augmentations in openVX extensions
  * Resample augmentation and Operator overloading nodes in openVX extensions
  * NonSilentRegion and Slice augmentations in openVX extensions
  * Mel-Filter bank and Normalize augmentations in openVX extensions
* CentOS 7 and SLES 15 SP5 support has been added to the setup script.
* The `FindPackage` modules have been updated with `FindMIVision`.
* New tests have been added for all modules.

### Changed

* rocAL is no longer installed with MIVisionX and must be installed separately.

### Resolved issues

* MIVisionX compatibility fix: Resample and pre-emphasis filter.
* Broken image links have been fixed in the documentation.

### Known issues

* Package install requires manually installing OpenCV
* Installation CentOS/RedHat/SLES requires manually installing the `FFMPEG Dev` package
* Hardware decode requires the ROCm `graphics` usecase.

### Upcoming changes

* Optimized audio augmentations support for VX_RPP.

## MIVisionX 2.5.0

### Additions

* CTest: Tests for install verification
* Hardware support updates
* Doxygen support for API documentation

### Optimizations

* CMakeList Cleanup
* Readme

### Changes

* rocAL: PyBind Link to prebuilt library
  * PyBind11
  * RapidJSON
* Setup Updates
* RPP - Use package install
* Dockerfiles: Updates & bugfix
* CuPy - No longer installed with setup.py

### Fixes

* rocAL bug fix and updates

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7` / `8`
  * RHEL - `8` / `9`
  * SLES - `15-SP4`
* ROCm: rocm-core - `5.7.0.50700-6`
* miopen-hip - `2.20.0.50700-63`
* MIGraphX - `2.7.0.50700-63`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [1.5.0]
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all preceding packages
* MIVisionX setup script - `V2.6.1`

### Known issues

* OpenCV 4.X support for some applications is missing
* MIVisionX package install requires manual prerequisites installation

## MIVisionX 2.4.0

### Additions

* OpenVX FP16 Support
* rocAL: CPU, HIP, & OCL backends
* AMD RPP: CPU, HIP, amd OCL backends
* MIVisionX Setup Support for RHEL
* Extended OS Support
* Docker Support for Ubuntu `22.04`
* Tests

### Optimizations

* CMakeList cleanup
* MIGraphX extension updates
* rocAL: Documentation
* CMakeList updates and cleanup

### Changes

* rocAL: Changing Python lib path
* Docker support: Ubuntu 18 support has been dropped
* RPP: Link to Version 1.0.0
* rocAL: Support updates
* Setup updates

### Fixes

* rocAL bug fix and updates
* AMD RPP bug fixes
* CMakeLists: Issues
* RPATH: Link issues

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7` / `8`
  * RHEL - `8` / `9`
  * SLES - `15-SP3`
* ROCm: rocm-core - `5.4.3.50403-121`
* miopen-hip - `2.19.0.50403-121`
* miopen-opencl - `2.18.0.50300-63`
* MIGraphX - `2.4.0.50403-121`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [1.0.0](https://github.com/ROCm/rpp/releases/tag/1.0.0)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all preceding packages
* MIVisionX setup script - `V2.4.2`

### Known issues

* OpenCV 4.X support for some applications is missing

## MIVisionX 2.3.0

### Additions

* Zen DNN samples
* OpenCV extension tests
* rocAL SPACK support
* AMD custom extension

### Optimizations

* MIGraphX updates
* Model compiler scripts

### Changes

* CMakeList: Find HIP updates

### Fixes

* rocAL issues
* MIGraphX issues

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `20.04` / `22.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-core - `5.3.0.50300-36`
* miopen-hip - `2.18.0.50300-36`
* miopen-opencl - `2.18.0.50300-36`
* MIGraphX - `2.3.0.50300-36`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.97](https://github.com/ROCm/rpp/releases/tag/0.97)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all preceding packages
* MIVisionX setup script - `V2.3.7`

### Known issues

* OpenCV 4.X support for some applications is missing

## MIVisionX 2.2.0

### Additions

* Docker support for ROCm `5.1.3`

### Optimizations

* MIGraphX: Implementation and samples

### Changes

* DockerFiles: Updates to install ROCm 5.1.1 Plus

### Fixes

* Minor bugs in setup and test scripts

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-core - `5.1.3.50103-66`
* miopen-hip - `2.16.0.50101-48`
* miopen-opencl - `2.16.0.50101-48`
* MIGraphX - `2.1.0.50101-48`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.93](https://github.com/ROCm/rpp/releases/tag/0.93)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* Dependencies for all preceding packages
* MIVisionX setup script - `V2.3.2`

### Known issues

* OpenCV 4.X support for some applications is missing

## MIVisionX 2.1.0

### Additions

* New Tests: `AMD_MEDIA`

### Optimizations

* Readme updates
* HIP buffer transfer: Eliminate `cupy` usage

### Changes

* **Backend**: Default backend has been set to `HIP`
* DockerFiles: Updates to install ROCm 5.0 Plus
* RPP: Upgraded to V0.93

### Fixes

* Minor bugs and warnings
* `AMD_MEDIA`: Bug Fixes

### Tested configurations

* Windows `10` / `11`
* Linux distribution
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-core - `5.0.0.50000-49`
* rocm-cmake - [rocm-5.1.1](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-5.1.1)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [4.5.5](https://github.com/opencv/opencv/releases/tag/4.5.5)
* RPP - [0.93](https://github.com/ROCm/rpp/releases/tag/0.93)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all preceding packages
* MIVisionX Setup Script - `V2.1.1`

### Known issues

* `CMakeList.txt` warnings: With ROCm CMake version MIVisionX will have CMake warnings

## MIVisionX 2.0.1

### Additions

* Support for CMake 3.22.X
* Support for OpenCV 4.X.X
* Support for mv_compile with the HIP GPU backend
* Support for tensor_compare node (`less`, `greater`, `less_than`, `greater_than`, and `equal` ONNX
  operators)

### Optimizations

* Code cleanup
* Readme updates

### Changes

* License updates

### Fixes

* Minor bugs and warnings
* Inference server application: OpenCL backend
* `vxCreateThreshold` fix: applications and sample

### Tested configurations

* Windows 10
* Linux distribution
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-dev - `4.5.2.40502-164`
* rocm-cmake - [rocm-4.2.0](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-4.2.0)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
* RPP - [0.92](https://github.com/ROCm/rpp/releases/tag/0.92)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all preceding packages
* MIVisionX Setup Script - `V2.0.0`

### Known issues

* Package install requires **OpenCV** `v3.4.X` to execute `AMD OpenCV extensions`

## MIVisionX 2.0.0

### Additions

* OpenVX 1.3: vision feature set
* Conformance test script
* HIP backend support for OpenVX and OpenVX extensions

### Optimizations

* Improved rocAL performance
* Improved the performance of OpenVX OpenCL backend functions

### Changes

* Docker build files

### Fixes

* `MIVisionX-setup.py` install on Linux
* Fixed out-of-bounds read for OpenVX OpenCL kernels
* OpenVX: Optical flow segfault fix

### Tested Configurations

* Windows 10
* Linux distribution
  * Ubuntu - `18.04` / `20.04`
  * CentOS - `7` / `8`
  * SLES - `15-SP2`
* ROCm: rocm-dev - `4.5.2.40502-164`
* rocm-cmake - [rocm-4.2.0](https://github.com/RadeonOpenCompute/rocm-cmake/releases/tag/rocm-4.2.0)
* MIOpenGEMM - [1.1.5](https://github.com/ROCmSoftwarePlatform/MIOpenGEMM/releases/tag/1.1.5)
* MIOpen - [2.14.0](https://github.com/ROCmSoftwarePlatform/MIOpen/releases/tag/2.14.0)
* Protobuf - [V3.12.0](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.0)
* OpenCV - [3.4.0](https://github.com/opencv/opencv/releases/tag/3.4.0)
* RPP - [0.92](https://github.com/ROCm/rpp/releases/tag/0.92)
* FFMPEG - [n4.0.4](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.0.4)
* Dependencies for all preceding packages
* MIVisionX Setup Script - `V2.0.0`

### Known issues

* Package install requires **OpenCV** `v3.4.X` to run `AMD OpenCV extensions`
