# Changelog for hipBLASLt

Full documentation for hipBLASLt is available at [rocm.docs.amd.com/projects/hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html).

## hipBLASLt 0.12.0 for ROCm 6.4.0

### Added

* Support roctx if `HIPBLASLT_ENABLE_MARKER=1` is set
* Output the profile logging if `HIPBLASLT_LOG_MASK=64` is set
* Support FP16 compute type
* Add memory bandwidth information in hipblaslt-bench output
* Support user offline tuning mechanism
* Add more samples

### Changed

* Output the bench command along with solution index if `HIPBLASLT_LOG_MASK=32` is set

### Optimized

* Improve the overall performance of XF32/FP16/BF16/FP8/BF8 data type
* Reduce library size

### Resolved issues

* Fix multi-threads bug
* Fix multi-streams bug

## hipBLASLt 0.10.0 for ROCm 6.3.0

### Added

* Support the V2 CPP extension API for backward compatibility
* Support for data type Int8 in with Int8 out
* Support for data type FP32/FP64 for gfx110x
* Add the Extension API `hipblaslt_ext::matmulIsTuned`
* Output atol and rtol for hipblaslt-bench validation
* Output the bench command for hipblaslt CPP ext API path if `HIPBLASLT_LOG_MASK=32` is set
* Support odd sizes for FP8/BF8 GEMM

### Changed

* Reorganize and add more sample code
* Add a dependency with the hipblas-common package and remove the dependency with the hipblas package

### Optimized

* Support fused kernel for HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER for FP8/BF8 data type
* Improve the library loading time
* Improve the overall performance of first returned solution

### Upcoming changes

*  The V1 CPP extension API will be deprecated in a future release of hipBLASLt

## hipBLASLt 0.8.0

### Added

* Extension APIs:
  * `hipblasltExtAMaxWithScale`
* `GemmTuning` extension parameter to set wgm by user
* Support HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER for the FP8/BF8 data types
* Support for FP8/BF8 input, FP32/FP16/BF16/F8/BF8 output (only for the gfx94x architectures)
* Support HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT and HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT for FP16 input data type to use FP8/BF8 mfma
* Support for the gfx110x architecture

### Optimized

* Improve the library loading time

## hipBLASLt 0.7.0

### Additions

* Extension APIs:
  * `hipblasltExtSoftmax`
  * `hipblasltExtLayerNorm`
  * `hipblasltExtAMax`
* `GemmTuning` extension parameter to set split-k by user
* Support for mixed-precision datatype: FP16/FP8 in with FP16 out
* Add CMake support for documentation

### Deprecations

* algoGetHeuristic() ext API for GroupGemm will be deprecated in a future release of hipBLASLt

## hipBLASLt 0.6.0

### Additions

* New `UserArguments` variable for `GroupedGemm`
* Support for datatype: FP16 in with FP32 out
* Support for datatype: Int8 in Int32 out
* Support for gfx94x platform
* Support for FP8/BF8 datatype (only for gfx94x platform)
* Support scalar A,B,C,D for FP8/BF8 datatype
* Added samples

### Changes

* Replaced `hipblasDatatype_t` with `hipDataType`
* Replaced `hipblasLtComputeType_t` with `hipblasComputeType_t`

### Removals

* Deprecated `HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER`

## hipBLASLt 0.3.0

### Additions

* Added `getAllAlgos` extension APIs
* TensileLite support for new epilogues: gradient gelu, gradient D, gradient A/B, aux
* Added a sample package that includes three sample apps
* Added a new C++ GEMM class in the hipBLASLt extension

### Changes

* Refactored GroupGemm APIs as C++ class in the hipBLASLt extension
* Changed the scaleD vector enum to `HIPBLASLT_MATMUL_DESC_D_SCALE_VECTOR_POINTER`

### Fixes

* Enabled norm check validation for CI

### Optimizations

* GSU kernel: wider memory, PGR N
* Updated logic yaml to improve some FP16 NN sizes
* GroupGemm support for GSU kernel
* Added grouped GEMM tuning for aldebaran

## hipBLASLt 0.2.0

### Additions

* Added CI tests for TensileLite
* Initialized extension group GEMM APIs (FP16 only)
* Added a group GEMM sample app: `example_hipblaslt_groupedgemm`

### Fixes

* Fixed incorrect results for the ScaleD kernel

### Optimizations

* Tuned equality sizes for the HHS data type
* Reduced host-side overhead for `hipblasLtMatmul()`
* Removed unused kernel arguments
* Schedule values setup before first `s_waitcnt`
* Refactored TensileLite host codes
* Optimized build time

## hipBLASLt 0.1.0

### Additions

* Enabled hipBLASLt APIs
* Support for gfx90a
* Support for problem type: FP32, FP16, BF16
* Support activation: relu, gelu
* Support for bias vectors
* Integrated with TensileLite kernel generator
* Added Gtest: `hipblaslt-test`
* Added the full function tool `hipblaslt-bench`
* Added the sample app `example_hipblaslt_preference`

### Optimizations

* gridBase solution search algorithm for untuned size
* Tuned 10k sizes for each problem type
