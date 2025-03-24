# Changelog for rocDecode

Full documentation for rocDecode is available at [https://rocm.docs.amd.com/projects/rocDecode/en/latest/](https://rocm.docs.amd.com/projects/rocDecode/en/latest/)

## rocDecode 0.12.0 for ROCm 6.5

### Added

* VP9 IVF container file parsing support in bitstream reader.
* CTest for VP9 decode on bitstream reader.
* HEVC stream syntax error handling.
* HEVC stream bit depth change handling and DPB buffer size change handling through decoder reconfiguration.
* rocDecode now uses the Cmake CMAKE_PREFIX_PATH directive.

### Optimized

* Decode session start latency reduction.
* Bitstream type detection optimization in bitstream reader.

### Resolved issues

* Fixed a bug in picture files sample "videoDecodePicFiles" that can results in incorrect output frame count.

### Removed

* GetStream() interface call from RocVideoDecoder utility class

## rocDecode 0.10.0 for ROCm 6.4

### Added

* The new bitstream reader feature. The bitstream reader contains built-in stream file parsers, including an elementary stream file parser and an IVF container file parser. It can parse AVC/HEVC/AV1 elementary stream files and AV1 IVF container files. Additional format support can be added in the future.
* VP9 decode support.
* More CTests: VP9 test and tests on video decode raw sample.
* Two new samples, videodecoderaw and videodecodepicfiles, have been added. videodecoderaw uses the bitstream reader instead of the FFMPEG demuxer to get picture data, and videodecodepicfiles shows how to decode an elementary video stream stored in multiple files with each file containing bitstream data of a coded picutre

### Changed

* AMD Clang++ is now the default CXX compiler.
* Moved MD5 code out of roc video decode utility.

### Removed

* FFMPEG executable requirement for the package

## rocDecode 0.8.0 for ROCm 6.3

### Added

* AV1 decode support

### Changed

* Clang is now the default CXX compiler.
* The new minimum supported version of va-api is 1.16.
* New build and runtime options have been added to the `rocDecode-setup.py` setup script.
* Added FFMpeg based software decoding into utils.
* Modified videodecode sample to allow FFMpeg based decoding

### Removed

* Make tests have been removed. CTEST is now used for both Make tests and package tests.
* `mesa-amdgpu-dri-drivers` has been removed as a dependency on RHEL and SLES.

### Resolved issues

* Fixed a bug in the size of output streams in the `videoDecodeBatch` sample.

## rocDecode 0.7.0

### Added

* Clang - Default CXX compiler
* Parser - Add new API rocDecParserMarkFrameForReuse()

### Optimized

* Setup Script - Build and runtime install options

### Changed

* CTest - Core tests for make test and package test

### Resolved issues

* Sample - Bugfix for videoDecodeBatch

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * RHEL - `8` / `9`
  * SLES - `15 SP5`
* ROCm:
  * rocm-core - `6.2.0.60200-66`
  * amdgpu-core - `1:6.2.60200-2009582`
* libva-dev - `2.7.0-2` / `2.14.0-1`
* mesa-amdgpu-va-drivers - `1:24.2.0.60200-2009582`
* FFmpeg - `4.2.7` / `4.4.2-0`
* rocDecode Setup Script - `V2.2.0`


## rocDecode 0.6.0

### Additions

* AVC decode support
* FFMPEG V5.X Support
* Mariner - Build Support

### Optimizations

* Setup Script - Error Check install

### Changes

* Dependencies - Updates to core dependencies
* LibVA Headers - Use public headers
* mesa-amdgpu-va-drivers - RPM Package available on RPM from ROCm 6.2

### Fixes

* Package deps
* RHEL/SLES - Additional required packages `mesa-amdgpu-dri-drivers libdrm-amdgpu`

### Tested configurations

* Linux
  * Ubuntu - `20.04` / `22.04`
  * RHEL - `8` / `9`
* ROCm:
  * rocm-core - `6.1.0.60100-64`
  * amdgpu-core - `1:6.1.60100-1741643`
* libva-dev - `2.7.0-2` / `2.14.0-1`
* mesa-amdgpu-va-drivers - `1:24.1.0`
* mesa-amdgpu-dri-drivers - `24.1.0.60200`
* FFmpeg - `4.2.7` / `4.4.2-0`
* rocDecode Setup Script - `V2.1.0`

## rocDecode 0.5.0

### Changes

* Added HEVC decode support
* Changed setup updates
* Added AMDGPU package support
* Optimized package dependencies
* Updated README

### Fixes

* Minor bug fix and updates

### Tested configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
* ROCm:
  * rocm-core - `6.1.0.60100-28`
  * amdgpu-core - `1:6.1.60100-1731559`
* FFMPEG - `4.2.7` / `4.4.2-0`
* rocDecode Setup Script - `V1.4`

## rocDecode 0.4.0

### Changes

* Added CTest - Tests for install verification
* Added Doxygen - Support for API documentation
* Changed setup updates
* Optimized CMakeList Cleanup
* Added README

### Fixes

* Minor bug fix and updates

### Tested configurations

* Linux distribution
  * Ubuntu - `20.04` / `22.04`
* ROCm:
  * rocm-core - `5.6.1.50601-93`
  * amdgpu-core - `1:5.6.50601-1649308`
* FFMPEG - `4.2.7` / `4.4.2-0`
* libva-dev - `2.7.0-2` / `2.14.0-1`
* rocDecode Setup Script - `V1.1`
