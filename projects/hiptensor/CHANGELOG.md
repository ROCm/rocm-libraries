# Changelog for hipTensor

Full documentation for hipTensor is available at [rocm.docs.amd.com/projects/hiptensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html).

## hipTensor 2.6.0 for ROCm 10.3

### Added
* Added support for the following new GPU targets:
  * gfx11-7: gfx1170, gfx1171.

## hipTensor 2.5.0 for ROCm 10.2

### Added
* Added fp64 support to hiptensorPermute for ranks 2-6.
* Added broadcast support to the elementwise operations. An input tensor of `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, and `hiptensorCreateElementwiseTrinary` may now carry a subset of the output tensor's modes, in which case it's broadcast along the modes it lacks. For example, `A` with modes `{n}` can be combined with `B`, `C`, and `D` with modes `{m,n}`.
* Added `gfx1250-strict` as a supported offload target and classified its device pass as `gfx1250`.

### Changed
* Changed `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, and `hiptensorCreateElementwiseTrinary` to return `HIPTENSOR_STATUS_INVALID_VALUE` when a mode shared by an input tensor and the output tensor has a different extent in each, and `HIPTENSOR_STATUS_NOT_SUPPORTED` when a mode is repeated within one tensor. An input tensor that carries a mode the output tensor doesn't carry is still rejected with `HIPTENSOR_STATUS_NOT_SUPPORTED`, since that would have to be reduced away rather than broadcast.
* Changed `libhiptensor.so` to export symbols with hidden visibility by default, so only the documented public C API and the internal `hiptensor::Logger` and `hiptensor::HiptensorOptions` singletons, which the hipTensor tests share with the library, are exposed. Other internal C++ classes and Composable Kernel template instantiations that were previously visible are now local to the library. Code that relied on those undocumented internal symbols will no longer find them.
* Changed the plan cache file written by `hiptensorHandleWritePlanCacheToFile` to a versioned format that records the hipTensor version and the device's architecture and compute unit count. `hiptensorHandleReadPlanCacheFromFile` now returns `HIPTENSOR_STATUS_INVALID_VALUE` for a file of another hipTensor version, including every file written by earlier versions, or of another device; `HIPTENSOR_STATUS_IO_ERROR` for a damaged file; and `HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE`, with the number of cache lines the file needs in `numCachelinesRead`, when the file holds more lines than the plan cache. The plan cache is left unchanged in all these cases, and reading a file no longer changes the size of the plan cache, so resize the plan cache with `hiptensorHandleResizePlanCache` before reading a file with more lines. `hiptensorHandleWritePlanCacheToFile` and `hiptensorHandleReadPlanCacheFromFile` now return `HIPTENSOR_STATUS_NOT_INITIALIZED` for a null `handle` and `HIPTENSOR_STATUS_INVALID_VALUE` for a null `fileName` (or `numCachelinesRead`), also when the plan cache is disabled. With the plan cache disabled, `hiptensorHandleReadPlanCacheFromFile` sets `numCachelinesRead` to 0.
* Changed `hiptensorCreatePlan` to return `HIPTENSOR_STATUS_NOT_SUPPORTED` for any tensor whose elements span 2^31 bytes or more (for complex types, per real or imaginary part), measured from its lengths and strides, and, for `HIPTENSOR_R_64F` permutations and elementwise binary and trinary operations with an input laid out differently from the output, 2^29 bytes or more. The limits are deliberately conservative, so they also refuse some sizes that previously gave correct results, for example some 16-bit elementwise operations above 2 GiB and reductions of exactly 2^31 bytes.

### Resolved issues
* Fixed `hiptensorPermute` and the element-wise binary/trinary execute paths ignoring user-supplied output tensor strides, which caused the output to always be written contiguously regardless of the strides set on the output descriptor.
* Fixed elementwise operations reading the wrong elements when two input tensors ordered the same modes differently. Every tensor's strides are now permuted into the output tensor's mode order before reaching the kernel.
* Fixed scalars passed with a BF16 compute descriptor being read as zero, which made BF16 element-wise binary operations and BF16 reductions return zeros instead of the scaled result.
* Fixed `hiptensorElementwiseTrinaryExecute` applying `opAB` and `opABC` in swapped positions. It now computes `opABC(opAB(alpha * A, beta * B), gamma * C)` as documented. Results change whenever `opAB` and `opABC` differ.
* Fixed `hiptensorCreateElementwiseBinary` and `hiptensorCreateElementwiseTrinary` accepting an operator other than `HIPTENSOR_OP_ADD`, `HIPTENSOR_OP_MUL`, `HIPTENSOR_OP_MAX`, or `HIPTENSOR_OP_MIN` for `opAC`, `opAB`, or `opABC`, which left the output undefined. They now return `HIPTENSOR_STATUS_INVALID_VALUE`.
* Fixed elementwise kernel selection reading past the end of its tuning table for extents above the largest tuned size (rank 2 above 33,554,432, rank 3 above 524,288, and rank 4 above 131,072). Such extents now use the largest tuned size.
* Fixed `hiptensorPermute` returning an unscaled result when a plan whose first execution used `alpha == 1` was executed again with a different `alpha` under `HIPTENSOR_CACHE_MODE_NONE`.
* Fixed `hiptensorContract` and `hiptensorContractTrinary` returning wrong results for packed column-major tensors, including descriptors created with `strides = NULL`, whose modes weren't stored as all output modes followed by all contracted modes in the same order in every tensor (for example A with modes `{m0,k0,m1,k1}`), and for packed column-major tensors combined with row-major or padded tensors. The column-major stride optimization is now applied only when it addresses the same elements as the tensors' own strides.
* Fixed contractions reading C (and, in trinary contractions, D) with the output tensor's strides, which gave wrong results whenever it was laid out differently from the output. It is now read through its own modes and strides. `hiptensorCreateContraction` and `hiptensorCreateContractionTrinary` now return `HIPTENSOR_STATUS_NOT_SUPPORTED` when this tensor doesn't carry the same modes as the output, or, for complex compute types, when its layout differs from the output's, and `HIPTENSOR_STATUS_INVALID_VALUE` when a mode has a different extent than in the output.
* Fixed `hiptensorCreatePlan` with `HIPTENSOR_ALGO_DEFAULT` or `HIPTENSOR_ALGO_DEFAULT_PATIENT` aborting with an illegal memory access for tensors with padded (non-packed) strides. The scratch buffers used to time candidate kernels are now sized from the strides.
* Fixed loading hipTensor ending the host process on a GPU whose architecture it doesn't support and on a host without a visible GPU. The library now loads on any host, and `hiptensorCreate` returns `HIPTENSOR_STATUS_ARCH_MISMATCH` for an unsupported architecture or `HIPTENSOR_STATUS_HIP_ERROR` when the device can't be queried.
* Fixed HIP errors inside the library, such as running out of device memory, ending the host process or escaping the C API as a C++ exception. The API call now returns `HIPTENSOR_STATUS_ALLOC_FAILED` for an allocation failure, `HIPTENSOR_STATUS_HIP_ERROR` for any other failed HIP call, and `HIPTENSOR_STATUS_INTERNAL_ERROR` for any other error raised by the kernel backend. `hiptensorCreatePlan` now returns `HIPTENSOR_STATUS_HIP_ERROR` rather than `HIPTENSOR_STATUS_ALLOC_FAILED` when a HIP call other than an out-of-memory allocation fails while it times candidate kernels.
* Fixed `hiptensorOperationDescriptorSetAttribute` and `hiptensorOperationDescriptorGetAttribute` copying `sizeInBytes` bytes without checking them, which overwrote neighboring fields of the operation descriptor (for example when `HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_LEFT` or `HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_RIGHT` was set with one `uint32_t` per output mode), and writing through a null pointer for `HIPTENSOR_OPERATION_DESCRIPTOR_PADDING_VALUE`. A null buffer or a size that doesn't match the attribute now returns `HIPTENSOR_STATUS_INVALID_VALUE`. `PADDING_LEFT` and `PADDING_RIGHT` take one `uint32_t` per mode of the output tensor, and `PADDING_VALUE` takes one element of the output tensor's type. No kernel pads its output, so non-zero padding returns `HIPTENSOR_STATUS_NOT_SUPPORTED`.
* Fixed `hiptensorCreatePlan` dereferencing a null `pref`, which is documented as allowed. A null preference now selects the defaults of `hiptensorCreatePlanPreference` with `HIPTENSOR_ALGO_DEFAULT` and `HIPTENSOR_JIT_MODE_NONE`. A null `handle`, `plan`, or `desc` returns `HIPTENSOR_STATUS_NOT_INITIALIZED`.
* Fixed `hiptensorCreatePlan` leaving an allocated plan in `*plan` when it failed after creating it, for example when no kernel could run a contraction, which leaked unless the caller destroyed the plan. It now sets `*plan` to `NULL` whenever it doesn't return `HIPTENSOR_STATUS_SUCCESS`.
* Fixed `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, `hiptensorCreateElementwiseTrinary`, `hiptensorCreateReduction`, `hiptensorCreateContraction`, and `hiptensorCreateContractionTrinary` reading an undefined scalar type for `HIPTENSOR_COMPUTE_DESC_NONE`, the integer compute descriptors, and unknown values. They now return `HIPTENSOR_STATUS_INVALID_VALUE`. The contraction functions previously returned `HIPTENSOR_STATUS_ARCH_MISMATCH` for these.
* Fixed `hiptensorContract` reading released kernel arguments to log the required workspace size after returning `HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE`.
* Fixed permutations, elementwise binary and trinary operations, and reductions returning wrong results without an error, and contractions failing during plan creation or execution or ending the process (complex types), when a tensor's elements span 2^31 bytes or more. `hiptensorCreatePlan` now returns `HIPTENSOR_STATUS_NOT_SUPPORTED` for such a tensor, measured from its lengths and strides (for complex types, per real or imaginary part), and for the intermediate `A * B` tensor of a trinary contraction. For `HIPTENSOR_R_64F` permutations and elementwise binary and trinary operations with an input laid out differently from the output, including a broadcast input, which wrote zeros for elementwise tensors of 1 GiB, the limit is 2^29 bytes.
* Fixed the plan cache growing without bound. Adding a line to a full plan cache now evicts the least recently used line, and updating a line that is already cached no longer evicts another one.
* Fixed heap corruption after `hiptensorHandleWritePlanCacheToFile`, which crashed later executions on the same handle or the destruction of the handle.
* Fixed plan cache lookups matching a different problem, which then reused that problem's kernel. The key now also covers the device, the output tensor of trinary contractions, the operators, and the rank of each tensor.
* Fixed `hiptensorCreatePlan` with `HIPTENSOR_ALGO_ACTOR_CRITIC` selecting a contraction kernel that can't run the problem, after which `hiptensorContract` and `hiptensorContractTrinary` failed with `HIPTENSOR_STATUS_INTERNAL_ERROR`, and failing with `HIPTENSOR_STATUS_EXECUTION_FAILED` when its table names no kernel for the problem or the kernel it names isn't available. The plan now falls back to timing the kernels that can run the problem, and returns `HIPTENSOR_STATUS_NOT_SUPPORTED` when none can.
* Fixed `hiptensorCreatePlan` for trinary contractions selecting, with any algorithm, a kernel that can run the second step (`(A * B) * C`) but not the first (`A * B`), after which `hiptensorContractTrinary` failed in the first step. Only kernels that can run both steps are now selected.
* Fixed `hiptensorContractTrinary` with a D tensor returning NaN wherever the workspace held NaN before the call. The intermediate `A * B` tensor in the workspace was computed by a kernel that also read its previous contents, scaled by zero. The intermediate tensor is now cleared first.
* Fixed `hiptensorContractTrinary` returning `HIPTENSOR_STATUS_INTERNAL_ERROR` from the second execution of a plan created with `HIPTENSOR_ALGO_ACTOR_CRITIC` under `HIPTENSOR_CACHE_MODE_PEDANTIC`. The plan's kernel was replaced by a plan cache lookup among candidates that the plan doesn't keep, which left the plan without a kernel.
* Fixed the hipTensor test binaries exiting with an "Unknown command line argument" error when passed GoogleTest flags such as `--gtest_filter`.

## hipTensor 2.4.0 for ROCm 10.1

### Added
* Added `ffm-quick` and `ffm-full` test categories for emulation tests.
* Added a Windows version resource so `hiptensor.dll` exposes file metadata (file description, version, product name, and copyright) in File Explorer properties.

### Changed
* Reorganized test configurations per tier and category.

### Optimized
* Reduced test execution duration.

### Resolved issues
* Enabled `-frtti` on Windows to fix RTTI-related build failures.
* Fixed batched contractions reporting success while producing incorrect results. `hiptensorCreateContraction` and `hiptensorCreateContractionTrinary` now return `HIPTENSOR_STATUS_NOT_SUPPORTED` when a mode is shared by both inputs and the output of a contraction.
* Changed `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, and `hiptensorCreateElementwiseTrinary` to return `HIPTENSOR_STATUS_NOT_SUPPORTED` when an input tensor doesn't carry the same modes as the output tensor. These configurations previously produced a valid descriptor and plan, then failed with `HIPTENSOR_STATUS_INTERNAL_ERROR` at execution.
* Removed the internal compiler flags `-amdgpu-early-inline-all=true` and `-amdgpu-function-calls=false` from the build, which caused excessive compile-time memory usage (OOM) with newer ROCm/LLVM toolchains (JIRA: LCOMPILER-2589).

## hipTensor 2.3.0 for ROCm 7.14

### Added
* Added Windows support.
* Added contraction support with FP16 and BF16 data and compute types for gfx11 and gfx12 targets.
* Added support for the following new GPU targets:
  * gfx11: gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152, gfx1153.
  * gfx12: gfx1200, gfx1201, gfx1250.
* Added unary element-wise operators to contraction, including the new `BilinearUnary` class, dedicated instances, samples, and tests.
* Added Dockerfiles (prebuilt and full build) and documentation to streamline hipTensor build environment setup.
* Added the `CREATE_TEST_APP_LOCAL_DEPLOY` CMake option to stage required ROCm DLLs on Windows, and updated the Windows build documentation accordingly.
* Added YAML-driven CTest test filter standardization, applying `quick`/`standard`/`comprehensive`/`full` tier labels to the installed test tree so tests can be run by tier with `ctest -L <tier>`.
* Added support for trinary contractions.
* Added hipTensor to the TheRock build system, enabling source builds, artifact distribution, and CI testing on both Linux and Windows.
* Added native Linux packages with HPC SDK metapackages: `amdrocm-hiptensor` (runtime), `amdrocm-hiptensor-devel` (headers and CMake config), and `amdrocm-hiptensor-test` (CTest binaries).

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
* Re-selected `HIPTENSOR_ALGO_ACTOR_CRITIC` winning kernels for all contraction type/rank/layout combinations to reflect current Composable Kernel tile parameters.

### Resolved issues
* Fixed use-after-free bug where `hiptensorCreatePlan` held dangling pointers to user-provided descriptors; the plan now deep-copies all descriptors.
* Fixed incorrect BF16 results in contraction with unary ops caused by silent `bhalf_t`-to-float integer promotion in cross-type overloads.

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
