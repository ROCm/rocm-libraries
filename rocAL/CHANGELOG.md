# Changelog for rocAL

Full documentation for rocLibrary is available at [https://rocm.docs.amd.com/projects/rocAL/](https://rocm.docs.amd.com/projects/rocAL/en/latest/).

## rocAL 2.3.0 for ROCm 6.5.0

### Added
* Extended support to rocAL's video decoder to use rocDecode hardware decoder
* Setup - installs rocdecode dev packages for Ubuntu, RedHat, and SLES
* Setup - installs turbojpeg dev package for Ubuntu and Redhat
* rocAL's image decoder has been extended to support the rocJPEG hardware decoder
* Added numpy reader support for reading npy files in rocAL
* Added test case for numpy reader in C++ and python tests

### Resolved issues
* `TurboJPEG` no longer needs to be installed manually. It is now installed by the package installer.
* Hardware decode no longer requires that ROCm be installed with the `graphics` usecase

### Known issues
* Package installation on SLES requires manually installing `TurboJPEG`.
* Package installation on CentOS, RedHat, and SLES requires manually installing the `FFMPEG Dev` package.

### Upcoming changes
* rocJPEG support for JPEG decode

## rocAL 2.2.0 for ROCm 6.4.0

### Changed

* AMD Clang is now the default CXX and C compiler.

### Known issues

* The package installation requires manually installing `TurboJPEG`
* Package installation on CentOS, RedHat, SLES requires manually installing the `FFMPEG Dev` package.
* Hardware decode requires installing ROCm with the `graphics` usecase

## rocAL 2.1.0 for ROCm 6.3.0

### Added

* rocAL Pybind support for package installation has been added. To use the rocAL python module, set the `PYTHONPATH`: `export PYTHONPATH=/opt/rocm/lib:$PYTHONPATH`
* Last batch policy, pad last batch, stick to shard, and shard size support have been added for the coco, caffe, caffe2, mxnet, tf, and cifar10 image readers.

### Changed

* rocdecode installation disabled when running the setup script.

### Removed

* rocDecode dependencies for package install has been removed.

### Optimizations

* CTest has been updated.

### Resolved issues

* Test failures have been fixed

### Known issues

* The package installation requires the manual installation of `TurboJPEG` and `RapidJSON`.
* CentOS/RedHat/SLES requires additional the manual installation of the `FFMPEG Dev` package.
* Hardware decode requires installing ROCm with the `graphics` usecase.

### Upcoming changes

* Optimized audio augmentations support

## rocAL 2.0.0 for ROCm 6.2.1

### Changes

* The new version of rocAL introduces many new features, but does not modify any of the existing public API functions.However, the version number was incremented from 1.3 to 2.0.
  Applications linked to version 1.3 must be recompiled to link against version 2.0.
* Added development and test packages.
* Added C++ rocAL audio unit test and Python script to run and compare the outputs.
* Added Python support for audio decoders.
* Added Pytorch iterator for audio.
* Added Python audio unit test and support to verify outputs.
* Added rocDecode for HW decode.
* Added support for:
  * Audio loader and decoder, which uses libsndfile library to decode wav files
  * Audio augmentation - PreEmphasis filter, Spectrogram, ToDecibels, Resample, NonSilentRegionDetection. MelFilterBank
  * Generic augmentation - Slice, Normalize
  * Reading from file lists in file reader
  * Downmixing audio channels during decoding
  * TensorTensorAdd and TensorScalarMultiply operations
  * Uniform and Normal distribution nodes
* Image to tensor updates
* ROCm install - use case graphics removed

### Known issues

* Dependencies are not installed with the rocAL package installer. Dependencies must be installed with the prerequisite setup script provided. See the [rocAL README on GitHub](https://github.com/ROCm/rocAL/blob/docs/6.2.1/README.md#prerequisites-setup-script) for details.

## rocAL 1.0.0

### Added

* rocAL Tests

### Optimizations

* Image augmentations

### Changed

* Deps

### Fixed

* minor issues

### Tested Configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
* ROCm: rocm-core - `6.0.60002-1`
* Protobuf - [V3.12.4](https://github.com/protocolbuffers/protobuf/releases/tag/v3.12.4)
* OpenCV - [4.6.0](https://github.com/opencv/opencv/releases/tag/4.6.0)
* RPP - [1.4.0](https://github.com/ROCms/rpp/releases/tag/1.4.0)
* FFMPEG - [n4.4.2](https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2)
* MIVisionX - [master](https://github.com/ROCm/MIVisionX)
* Dependencies for all the above packages
* rocAL Setup Script - `V1.0.2`

### Known issues

* Requires custom version of libturbo-JPEG
