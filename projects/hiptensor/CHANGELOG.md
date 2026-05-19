# Changelog for hipTensor

Full documentation for hipTensor is available at [rocm.docs.amd.com/projects/hiptensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html).

## Since last release ROCm 7.13

### Added

* Added support for new GPU target gfx1250.

## Since last release ROCm 7.12

### Added
* Added Windows support.
* Added contraction support with FP16 and BF16 data and compute types for gfx11 and gfx12 targets.
* Added support for the following new GPU targets:
  * gfx11: gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153.
  * gfx12: gfx1200, gfx1201.
* Added unary element-wise operators to contraction, including the new `BilinearUnary` class, dedicated instances, samples, and tests.
* Added Dockerfiles (prebuilt and full build) and documentation to streamline hipTensor build environment setup.
* Added the `CREATE_TEST_APP_LOCAL_DEPLOY` CMake option to stage required ROCm DLLs on Windows, and updated the Windows build documentation accordingly.

### Changed
* Replaced numeric UID-based actor-critic kernel lookup with platform-stable string-based kernel name comparison to enable cross-platform compatibility.
* Adopted FNV-1a string hashing in place of `std::hash` to ensure plan cache files are portable across platforms.
* Switched to ROCm-provided CMake install functions (`rocm_export_targets`) for consistency with other ROCm libraries.
* Adapted hipTensor to CK namespace changes for `host_tensor` functions.
* Cleaned up `rtest` script formatting and removed invalid run commands from `rtest.xml`.

### Removed
* Removed the legacy `.jenkins` folder since CI migration to rocJenkins.

### Optimized
* Improved column-major contraction performance by applying CK-style stride reordering for column-major inputs.
* Achieved 2x–3x speedup in contraction TFLOPS/s by using switch-case dispatch in `HiptensorUnaryOp` instead of static table lookup.

### Resolved issues
* Fixed use-after-free bug where `hiptensorCreatePlan` held dangling pointers to user-provided descriptors; the plan now deep-copies all descriptors.
* Fixed incorrect BF16 results in contraction with unary ops caused by silent `bhalf_t`-to-float integer promotion in cross-type overloads.

### Known issues 
* Unary operations in contraction are not currently supported with `HIPTENSOR_ALGO_ACTOR_CRITIC` for problems with both F16 datatypes and compute types, or both BF16 datatypes and compute types.

### Upcoming changes
* Add support for trinary contraction.

## hipTensor 2.2.0 for ROCm 7.2.0

### Added

* Added software-managed plan cache support.
* Added `hiptensorHandleWritePlanCacheToFile` to write the plan cache of a hipTensor handle to a file.
* Added `hiptensorHandleReadPlanCacheFromFile` to read a plan cache from a file into a hipTensor handle.
* Added `simple_contraction_plan_cache` to demonstrate plan cache usages.
* Added `plan_cache_test` to test the plan cache across various tensor ranks.
* Added C API headers to enable compatibility with C programs.
* Added a CMake function to allow projects to query architecture support.
* Added an option to configure the memory layout for tests and benchmarks.

### Changed

* hipTensor has been moved into the new rocm-libraries "monorepo" repository (https://github.com/ROCm/rocm-libraries). This repository consolidates a number of separate ROCm libraries and shared components.
  * The repository migration requires a few changes to the CMake configuration of hipTensor.
* Updated C++ standard from C++17 to C++20.
* Include files `hiptensor/hiptensor.hpp` and `hiptensor/hiptensor_types.hpp` are now deprecated. Use `hiptensor/hiptensor.h` and `hiptensor/hiptensor_types.h` instead.
* Converted include guards from #ifndef/#define/#endif to #pragma once.

### Resolved issues

* Removed large tensor sizes causing problem in benchmarks.

## hipTensor 2.1.0 for ROCm 7.1.0

### Added

* Added large tensor lengths in benchmark YAML files.
* Added a new "-l" option to tests for redirecting logs to a file.

### Changed

* Replaced `permutation` with `elementwise` or `elementwise_permute` across folder names, file names, function names, and variable names.

### Resolved issues

* Fixed an issue where `hiptensor-test` files are not removed after uninstallation.
* Fixed memory leaks in the test code and cleared Valgrind leak reports.

## hipTensor 2.0.0 for ROCm 7.0.0

### Added

* Added element-wise binary operation support.
* Added element-wise trinary operation support.
* Added support for new GPU target gfx950.
* Added dynamic unary and binary operator support for element-wise operations and permutation.
* Added a CMake check for `f8` datatype availability.
* Added `hiptensorDestroyOperationDescriptor` to free all resources related to the provided descriptor.
* Added `hiptensorOperationDescriptorSetAttribute` to set attribute of a `hiptensorOperationDescriptor_t` object.
* Added `hiptensorOperationDescriptorGetAttribute` to retrieve an attribute of the provided `hiptensorOperationDescriptor_t` object.
* Added `hiptensorCreatePlanPreference` to allocate the `hiptensorPlanPreference_t` and enabled users to limit the applicable kernels for a given plan or operation.
* Added `hiptensorDestroyPlanPreference` to free all resources related to the provided preference.
* Added `hiptensorPlanPreferenceSetAttribute` to set attribute of a `hiptensorPlanPreference_t` object.
* Added `hiptensorPlanGetAttribute` to retrieve information about an already-created plan.
* Added `hiptensorEstimateWorkspaceSize` to determine the required workspaceSize for the given operation.
* Added `hiptensorCreatePlan` to allocate a `hiptensorPlan_t` object, select an appropriate kernel for a given operation and prepare a plan that encodes the execution.
* Added `hiptensorDestroyPlan` to free all resources related to the provided plan.

### Changed

* Removed architecture support for gfx940 and gfx941.
* Generalized opaque buffer now for any descriptor.
* Replaced `hipDataType` with `hiptensorDataType_t` for all supported types, for example, `HIP_R_32F` to `HIPTENSOR_R_32F`.
* Replaced `hiptensorComputeType_t` with `hiptensorComputeDescriptor_t` for all supported types.
* Replaced `hiptensorInitTensorDescriptor` with `hiptensorCreateTensorDescriptor`.
* Changed handle type and API usage from `*handle` to `handle`.
* Replaced `hiptensorContractionDescriptor_t` with `hipTensorOperationDescriptor_t`.
* Replaced `hiptensorInitContractionDescriptor` with `hiptensorCreateContraction`.
* Replaced `hiptensorContractionFind_t` with `hiptensorPlanPreference_t`.
* Replaced `hiptensorInitContractionFind` with `hiptensorCreatePlanPreference`.
* Replaced `hiptensorContractionGetWorkspaceSize` with `hiptensorEstimateWorkspaceSize`.
* Replaced `HIPTENSOR_WORKSPACE_RECOMMENDED` with `HIPTENSOR_WORKSPACE_DEFAULT`.
* Replaced `hiptensorContractionPlan_t` with `hiptensorPlan_t`.
* Replaced `hiptensorInitContractionPlan` with `hiptensorCreatePlan`.
* Replaced `hiptensorContraction` with `hiptensorContract`.
* Replaced `hiptensorPermutation` with `hiptensorPermute`.
* Replaced `hiptensorReduction` with `hiptensorReduce`.
* Replaced `hiptensorElementwiseBinary` with `hiptensorElementwiseBinaryExecute`.
* Replaced `hiptensorElementwiseTrinary` with `hiptensorElementwiseTrinaryExecute`.
* Removed function `hiptensorReductionGetWorkspaceSize`.

## hipTensor 1.5.0 for ROCm 6.4.0

### Added

* Added benchmarking suites for contraction, permutation, and reduction. YAML files are categorized into bench and validation folders for organization
* Added emulation test suites for contraction, permutation, and reduction
* Support has been added for changing the default data layout using the `HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR` environment variable

### Changed

* Used `GPU_TARGETS` instead of `AMDGPU_TARGETS` in `cmakelists.txt`
* Binary sizes can be reduced on supported compilers by using the `--offload-compress` compiler flag

### Optimized

* Optimized the hyper-parameter selection algorithm for permutation

### Resolved issues

* For CMake bug workaround, set `CMAKE_NO_BUILTIN_CHRPATH` when `BUILD_OFFLOAD_COMPRESS` is unset

## hipTensor 1.4.0 for ROCm 6.3.0

### Added

* Added support for tensor reduction, including APIs, CPU reference, unit tests, and documentation

### Changed

* ASAN builds only support xnack+ targets.
* ASAN builds use `-mcmodel=large` to accommodate library sizes greater than 2GB.
* Updated the permute backend to accommodate changes to element-wise operations.
* Updated the actor-critic implementation.

### Optimized

* Split kernel instances to improve build times

### Resolved issues

* Fixed a bug in randomized tensor input data generation.
* Fixed the default strides calculation to be in column major order.
* Fixed a small memory leak by properly destroying HIP event objects in tests.
* Default strides calculations now follow column-major convention.
* Various documentation formatting updates and fixes.

## hipTensor 1.3.0 for ROCm 6.2.0

### Additions

* Added support for tensor permutation of ranks of 2, 3, 4, 5 and 6
* Added tests for tensor permutation of ranks of 2, 3, 4, 5 and 6
* Added support for tensor contraction of M6N6K6: M, N, K up to rank 6
* Added tests for tensor contraction of M6N6K6: M, N, K up to rank 6
* Added new test YAML parsing to support sequential parameters ordering

### Changes

* Documentation updates for installation, programmer's guide and API reference
* Prefer amd-llvm-devel package before system LLVM library
* Preferred compilers changed to CC=amdclang CXX=amdclang++
* Updated actor-critic selection for new contraction kernel additions

### Fixes

* Fixed LLVM parsing crash
* Fixed memory consumption issue in complex kernels
* Work-around implemented for compiler crash during debug build
* Allow random modes ordering for tensor contractions

## hipTensor 1.2.0 for ROCm 6.1.0

### Additions

* API support for permutation of rank 4 tensors: f16 and f32
* New datatype support in contractions of rank 4: f16, bf16, complex f32, complex f64
* Added scale and bilinear contraction samples and tests for new supported data types
* Added permutation samples and tests for f16, f32 types

### Fixes

* Fixed bug in contraction calculation with data type f32

## hipTensor 1.1.0 for ROCm 6.0.0

### Additions

* Architecture support for gfx940, gfx941, and gfx942
* Client tests configuration parameters now support YAML file input format

### Changes

* Doxygen now treats warnings as errors

### Fixes

* Client tests output redirections now behave accordingly
* Removed dependency static library deployment
* Security issues for documentation
* Compile issues in debug mode
* Corrected soft link for ROCm deployment

## hipTensor 1.0.0 for ROCm 5.7.0

### Additions

* Initial prototype enablement of hipTensor library that supports tensor operations
* Kernel selection support for Default and Actor-Critic algorithms
* API support for:
  * Definition and contraction of rank 4 tensors
  * Contextual logging and output redirection
  * Kernel selection caching
* Data type support for f32 and f64
* Architecture support for gfx908 and gfx90a
