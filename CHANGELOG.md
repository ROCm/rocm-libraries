# Changelog for rocWMMA

Documentation for rocWMMA is available at
[https://rocm.docs.amd.com/projects/rocWMMA/en/latest](https://rocm.docs.amd.com/projects/rocWMMA/en/latest).

## (Unreleased) rocWMMA 1.8.0 for ROCm 6.5.0

### Added

* Added internal flow control barriers to improve asm code layout and improve overall performance
* Added internal register layout transforms and tests to support interleaved MMA layouts
* Added support for the gfx950 target
* Added mixed input `bf8` / `fp8` types for MMA support

### Changed

* Removed support for the gfx940 and gfx941 targets
* Augmented load / store / MMA internals with static loop unrolling
* Switched to use interleaved layouts in MMA to improve overall performance
* rocWMMA mma_sync API now supports `wave tile` fragment sizes

### Resolved issues

* Fixed a validation issue for small precision compute types `< B32` on gfx9
* Fixed CMake validation of compiler support for `bf8` / `fp8` types
* Fixed linkage of rocwmma::synchronize_workgroup to inline

## rocWMMA 1.7.0 for ROCm 6.4.0

### Added

* Added interleaved layouts that enhance the performance of GEMM operations
* Added emulation test suites. These suites are lightweight and well-suited for execution on emulator platforms

### Changed

* Used GPU_TARGETS instead of AMDGPU_TARGETS in `cmakelists.txt`
* The binary sizes can be reduced on supported compilers by using the `--offload-compress` compiler flag

### Resolved issues

* For a CMake bug workaround, set `CMAKE_NO_BUILTIN_CHRPATH` when `BUILD_OFFLOAD_COMPRESS` is unset

## rocWMMA 1.6.0 for ROCm 6.3.0

### Added

* Added OCP f8/bf8 datatype support
* Added support for gfx12 architecture targets

### Changed

* Optimized some aos<->soa transforms with half-rotation offsets
* Refactored the rocBLAS reference entry point for validation and benchmarking
* ROCWMMA_* preprocessor configurations are now all assigned values
* Updated the default architecture targets for ASAN builds
* Updated the actor-critic implementation

### Resolved issues

* Fixed a bug in f64 validation due to faulty typecasting
* Fixed a bug causing runtime compilation errors with hipRTC
* Various documentation updates and fixes

### Upcoming changes

* rocWMMA 2.x plans to augment fragment API objects with additional meta-properties which will improve API expressiveness and configurability of parameters including multiple-wave cooperation. As part of this change, cooperative rocWMMA API functions `load_matrix_coop_sync` and `store_matrix_coop_sync` will become deprecated in a future ROCm release.

## rocWMMA 1.5.0 for ROCm 6.2.0

### Additions

* Added internal utilities for element-wise vector transforms
* Added internal utilities for cross-lane vector transforms
* Implemented internal aos<->soa transforms for block sizes of 16, 32, 64, 128 and 256 and vector widths of 2, 4, 8 and 16
* Added tests for new internal transforms

### Changes

* Improved loading layouts by increasing vector width for fragments with blockDim > 32
* API applyDataLayout transform now accepts WaveCount template argument for cooperative fragments
* API applyDataLayout transform now physically applies aos<->soa transform as necessary
* Refactored entry-point of std library usage to improve hipRTC support
* Documentation updates for installation, programmer's guide and API reference

### Fixes

* Fixed some header includes ordering to improve portability

## rocWMMA 1.4.0 for ROCm 6.1.0

### Additions

* Added bf16 support for hipRTC sample

### Changes

* Changed Clang C++ version to C++17
* Updated rocwmma_coop API
* Linked rocWMMA to hiprtc

### Fixes

* Fixed compile/runtime arch checks
* Built all test in large code model
* Removed inefficient branching in layout loop unrolling

## rocWMMA 1.3.0 for ROCm 6.0.0

### Additions

* Support for gfx940, gfx941, and gfx942 targets
* Support for f8, bf8, and xfloat32 data types
* support for `HIP_NO_HALF`, `__ HIP_NO_HALF_CONVERSIONS__`, and
    `__ HIP_NO_HALF_OPERATORS__` (e.g., PyTorch environment)

### Changes

* rocWMMA with hipRTC now supports `bfloat16_t` data type
* gfx11 WMMA now uses lane swap instead of broadcast for layout adjustment
* Updated samples GEMM parameter validation on host arch

### Fixes

* Disabled GoogleTest static library deployment
* Extended tests now build in large code model

## rocWMMA 1.2.0 for ROCm 5.7.0

### Changes

* Fixed a synchronization bug
* Updated rocWMMA CMake versioning

## rocWMMA 1.1.0 for ROCm 5.6.0

### Additions

* Cross-lane operation backends (Blend, Permute, Swizzle, and Dpp)
* GPU kernels for rocWMMA unit test pre-process and post-process operations (fill, validation)
* Performance GEMM samples for half, single, and double precision
* rocWMMA CMake versioning
* Vectorized support in coordinate transforms
* Included ROCm SMI for runtime clock rate detection
* Fragment transforms for transpose and change data layout

### Changes

* Default to GPU rocBLAS validation against rocWMMA
* Re-enabled int8 GEMM tests on gfx9
* Upgraded to C++17
* Restructured the unit test folder for consistency
* Consolidated rocWMMA samples common code

## rocWMMA 1.0 for ROCm 5.5.0

### Additions

* Support for Wave32 on gfx11+
* Infrastructure changes to support hipRTC
* Performance tracking system
* Library config to support multiple architectures
* Vector cross-lane operations support

### Changes

* Modified the assignment of hardware information
* Modified data access for unsigned data types
* Refactored vector backend to be compatible with `HIP_vector_type`

## rocWMMA 0.9 for ROCm 5.4.0

### Additions

* GEMM driver APIs for flow control built-ins
* benchmark logging systems
* Restructured tests to follow naming convention; added macros for test generation

### Changes

* Changed CMake to accommodate the modified test infrastructure
* Fine-tuned the multi-block kernels with and without lDs
* Adjusted maximum vector width to dWordx4
* Updated efficiencies to display as whole number percentages
* Updated throughput from GFlops/s to TFlops/s
* Reset the ad-hoc tests to use smaller sizes
* Modified the output validation to use CPU-based implementation against rocWMMA
* Modified the extended vector test to return error codes for memory allocation failures

## rocWMMA 0.8 for ROCm 5.3.0

### Additions

* Runtime checks to disable tests on non-target GPUS
* Workgroup-aware GEMM kernels
* Workgroup-aware validation and benchmark test suite
* Warm-up run to existing tests

### Changes

* Refactored `lds_mapping_util` into GEMM global, local mapping, GEMM driver, GEMM config, and
  scheduling classes
* Modified resource allocation and tracking of GEMM and DLRM buffers
* Improved low-level data loading patterns
* Reduced branching on cooperative load and store
* Updated GEMV sample
* Updated GEMM sample

## rocWMMA 0.7 for ROCm 5.2.0

### Additions

* Unit tests for DLRM kernels
* GEMM sample
* DLRM sample
* SGEMV sample
* Unit tests for cooperative WMMA load and stores
* Unit tests for `IOBarrier.h`
* WMMA load and store tests for different matrix types (A, B, and Accumulator)
* More block sizes (1, 2, 4, 8) to test `MmaSyncMultiTest`
* Block sizes 4, 8 to test `MmaSynMultiLdsTest`
* Support for WMMA load and store layouts with a block dimension greater than 64
* IOShape structure to define the attributes of mapping and layouts for all WMMA matrix types
* CI testing for rocWMMA

### Changes

* Renamed WMMA to rocWMMA in CMake, header files, and documentation
* Renamed library files
* Modified `Layout.h` to use different matrix offset calculations (base offset, incremental offset, and
  cumulative offset)
* Opaque load and store continue to use incremental offsets as they fill the entire block
* Cooperative load and store use cumulative offsets as they fill only small portions for the entire block
* Increased max split counts to 64 for cooperative load and store
* Moved all the WMMA definitions and API headers to the rocWMMA namespace
* Modified WMMA fill unit tests to validate all matrix types (A, B, Accumulator)

## rocWMMA 0.6

### Additions

* Unit tests for `MappingUtil.h`
* Unit tests for `Layout.h`
* Unit tests for non-native vector class in `Types.h`
* Unit tests for WMMA load and store contamination check
* Doxygen support for rocWMMA documentation
* MFMA barrier in `IOBarrier.h`
* A CMake flag to support WMMA kernel assembly code generation
* MMA sync test WMMA operation with LDS usage
* A script to generate the plots of different WMMA benchmarks
* Multi-block kernels with LDS usage
* Unit tests for multi-block WMMA kernels

### Changes

* Modified GLlops calculation to accommodate multiple devices
* Removed half-types packing quirk with col major output
* Moved HIP resource management to `HipResource` class
* Fixed NaN errors during output comparison

## rocWMMA 0.5

### Additions

* Templatization for the `amdgcn_convert` class
* WMMA load, store, and fill support for integral data types and float64
* MFMA support for i8
* Support for `bf16_1k` MFMA instructions
* Code to identify the card type and its support during runtime

### Changes

* Refactored and simplified `IOBroadcast.h`
* Modified the fragment interface compatible with NVIDIA's definition
* Modified CMake to create a lean build of the rocWMMA library

## rocWMMA 0.4

### Additions

* CMake support for the library and unit tests
* Integrated unit test with GoogleTest and OpenMP
* Host overload operators for `hfloat16_t`

### Fixes

* Relative error calculation for non-integral data comparison
* Assembly generation of cooperative load and store code
* Compiler issues with new versions of ROCm

### Changes

* Sped up compilation time by moving thread block sizes to function arguments instead of template
  parameters
* Moved all the existing unit tests to a `test` folder
* Moved all the header files to `library/include`
* Modified `Layout.h` to use RowNT/ColNT to eliminate LDS usage in `mma_sync`
* Deprecated buffer load/store and local load/store

## rocWMMA 0.3

### Additions

* support for the bfloat16 compute type

### Changes

* Renamed `__half` to `hfloat_16` for consistency
* Modified `Convert.h` to support native to bfloat16 conversion and vice versa
* Modified `IOBroadCast.h` to incorporate bfloat16 data packing
* Modified `IOTraits.h` to add bfloat16 packing traits
* Modified `MFMA.h` to add MFMA invocation calls to bfloat16 data
* Modified WMMA types to include `bfloat16_t`
* Modified the WMMA load, store, and MMA unit tests to validate bfloat16

## rocWMMA 0.2

### Additions

* Support for fp16 compute type
* Direct MFMA support for non-native `__half` data type

### Changes

* Adjusted the vector storage to accommodate non-native types
* Fixed data comparison operators for fp16 data types
* Modified `Convert.h` to support native to `__half` conversion and vice versa
* Modified `IOBroadCast.h` to incorporate `__half` data packing
* Modified `IOTraits.h` to add `__half` packing traits
* Modified `MFMA.h` to add MFMA invocation calls to `__half` data
* Modified WMMA Types to include `__half _t`
* Modified the WMMA load, store, and MMA unit tests to validate `__half`

## rocWMMA 0.1

### Additions

* Defined a WMMA namespace with the supported matrix types, memory, and layouts
* Defined a fragment datatype to control the data transfer between HIP and MFMA
* Implemented the rocWMMA functions : `load_matrix_sync`, `load_matrix_coop_sync`,
  `store_matrix_sync`, `fill_fragment`, and `mma_sync`
* Implemented `Types.h` to define the supported data types
* Implemented the class `IOTraits` to define packing traits for the defined types as WMMA works on the
  packed registers
* Buffer load, store to support LLVM data instructions
* Opaque load, store
* Cooperative load, store to optimize the memory overhead
* Local load, store to perform register packing
* Implemented `Convert.h` to perform non-native data type conversion to native types and vice versa
* `IOBroadcast` class to perform packing for all input data (multiple registers)
* Implemented `IOConfig` to set the optimal input/output configurations for rocWMMA matrix types
* Implemented `IOPack` and `IOUnpack` to convert the unpacked device memory into packed registers
  and vice versa
* `Layout` class to define the data layout in matrix space
* MFMA to call the low-level MFMA hardware instructions
* Implemented the `MappingUtil` class to map from workgroup configurations to functional wave units
* `Performance.h` to compute GFLOPS based on hardware configurations
* `Reference.h` to implement the CPU GEMM operation
* `Utils.h` to implement matrix data operations
* `rocBLASReference.h` to invoke the rocBLAS GEMM function
* Unit tests to validate WMMA APIs (`load`, `store`, `fill`, and `mma`)
* Makefile support to build library and tests
