# Change Log for hipSPARSELt

Full documentation for hipSPARSELt is available at [rocm.docs.amd.com/projects/hipSPARSELt](https://rocm.docs.amd.com/projects/hipSPARSELt/en/latest/index.html).

## hipSPARSELt 0.2.5 for ROCm 7.1.0

### Changed

* Changed the behavior of the Relu activation.

### Optimized

* Provided more kernels for the `FP16` and `BF16` data types.

## hipSPARSELt 0.2.4 for ROCm 7.0.0

### Added

* Support for the LLVM target gfx950.
* Support for the following data type combinations for the LLVM target gfx950:
  * FP8(E4M3) inputs, F32 output, and F32 Matrix Core accumulation.
  * BF8(E5M2) inputs, F32 output, and F32 Matrix Core accumulation.
* Support for ROC-TX if `HIPSPARSELT_ENABLE_MARKER=1` is set.
* Support for the cuSPARSELt v0.6.3 backend.

### Optimized

* Improved the library loading time.
* Provided more kernels for FP16 datatype.

### Removed

* Support for LLVM targets gfx940 and gfx941 has been removed.
* `hipsparseLtDatatype_t` has been removed.

## hipSPARSELt 0.2.3 for ROCm 6.4.0

### Added

* Support for alpha vector scaling.

### Changed

* Changed the check mechanism of the inputs when is using alpha vector scaling.

## hipSPARSELt 0.2.2 for ROCm 6.3.0

### Added

* Support for a new data type combination: INT8 inputs, BF16 output, and INT32 Matrix Core accumulation.
* Support for row-major memory order (HIPSPARSE_ORDER_ROW).

### Changed

* Changed the default compiler to amdclang++.

### Upcoming changes

* hipsparseLtDatatype_t is deprecated and will be removed in the next major release of ROCm. hipDataType should be used instead.

## hipSPARSELt 0.2.1 for ROCm 6.2

### Changed

* Refined test cases

## hipSPARSELt 0.2.0 for ROCm 6.1

### Changes

* Support Matrix B is a Structured Sparsity Matrix.

## hipSPARSELt 0.1.0 for ROCm 6.0

### Changes

* Enabled hipSPARSELt APIs
* Support for:
  * gfx940, gfx941, and gfx942 platforms
  * FP16, BF16, and INT8 problem types
  * ReLU, GELU, abs, sigmod, and tanh activation
  * GELU scaling
  * Bias vectors
  * cuSPARSELt v0.4 backend
* Integrated with Tensile Lite kernel generator
* Support for batched computation (single sparse x multiple dense and multiple sparse x single dense)
* GoogleTest: hipsparselt-test
* `hipsparselt-bench` benchmarking tool
* Sample apps: `example_spmm_strided_batched`, `example_prune`, `example_compress`
