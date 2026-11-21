# Changelog for hipTensor

Full documentation for hipTensor is available at [rocm.docs.amd.com/projects/hiptensor](https://rocm.docs.amd.com/projects/hipTensor/en/latest/index.html).

## hipTensor 2.6.0 for ROCm 10.3

### Added
* Added support for the following new GPU targets:
  * gfx11-7: gfx1170, gfx1171.
* Added a backend interface between the solution layer and the kernels: `hiptensorCreatePlan` selects and prepares a backend for every operation, and executions run on it. The Composable Kernel backend is the default, and the rocKE backend is built with `-DHIPTENSOR_ENABLE_ROCKE_BACKEND=ON`. The environment variables `HIPTENSOR_BACKEND` (`auto`, `ck`, `rocke`), `HIPTENSOR_BACKEND_FALLBACK`, and `HIPTENSOR_BACKEND_AB` select the backend, control the fallback, and compare two backends.
* Added rocKE kernels to the rocKE backend for contractions, permutations, elementwise binary and trinary operations, and reductions, with problem sizes and strides passed at run time. The kernels are compiled at build time into a kernel pack per GPU target (`share/hiptensor/rocke/<target>/rocke.htpk`) by a C++ host tool, so neither the build nor the library runs Python. `HIPTENSOR_ROCKE_PACK_PATH` overrides where the library looks for packs.
* Added a JIT and a kernel cache to the rocKE backend (`-DHIPTENSOR_ENABLE_ROCKE_JIT=ON`, the default with the rocKE backend; links COMGR). Plans created with `HIPTENSOR_JIT_MODE_DEFAULT` replace their elementwise launches with kernels compiled for the problem at plan creation. Compiled kernels go to a process-wide kernel cache that `hiptensorWriteKernelCacheToFile` and `hiptensorReadKernelCacheFromFile` write and read.
* Added kernel selection to the rocKE backend at plan creation: capability checks of every candidate, then rocKE's tuned route tables, then a selection model per problem class shipped in the kernel pack, then rocKE's rules. `HIPTENSOR_PLAN_PREFERENCE_KERNEL_RANK` selects the k-th candidate, and `HIPTENSOR_ALGO_DEFAULT_PATIENT` times the leading candidates and stores the winner in the plan cache. `hiptensorEstimateWorkspaceSize` reports the workspace of the leading candidates.
* The rocKE backend stores the kernel every plan selects in the handle's plan cache (`HIPTENSOR_CACHE_MODE_PEDANTIC`), not only the winner of `HIPTENSOR_ALGO_DEFAULT_PATIENT`.
* Added contractions read and written in place to the rocKE backend: the ring GETT (f16 and bf16 on gfx950) and the direct GETT (every data type and layout, on every target) run in one launch with no workspace. Contractions with a zero workspace limit, and contractions with a 16-bit compute type on gfx90a and gfx942, are now served. Permutations and elementwise operations whose inputs and output are fastest along different modes run on a tiled kernel.
* Added the contraction routes of rocKE's AOT route table to the rocKE backend's kernel pack (f16 and bf16 on gfx950), selected ahead of the ring GETT. Their kernels have the problem's sizes and strides built in, so they serve only contractions that exactly match a table problem. Other contractions run on the other routes or, with `HIPTENSOR_JIT_MODE_DEFAULT`, on the contraction JIT.
* Added the unary operators `HIPTENSOR_OP_SIN` to `HIPTENSOR_OP_FLOOR` to the rocKE backend for f16, bf16, f32, and f64 data, evaluated in f64 with the errors listed in the programmer's guide. The rocKE backend's `HIPTENSOR_OP_SIGMOID` now keeps the small results of large negative arguments, and `HIPTENSOR_OP_CONJ` of a real tensor is the identity.
* Added f32 recipes to the rocKE backend for f32 and complex f32 contractions with `HIPTENSOR_COMPUTE_DESC_32F` or `HIPTENSOR_COMPUTE_DESC_C32F`, selected with `HIPTENSOR_ROCKE_F32_RECIPE`: `native`, `x9`, `x6`, `x4`, `x3` (split into bf16 parts), and `f64`. The default is `x6` on gfx950 and `native` on gfx942 and gfx90a. The programmer's guide gives the accuracy of each recipe.
* Added the GEMM routes of the rocKE backend on gfx942 for contractions with a 16-bit compute type and for the `x9`, `x6`, `x4`, and `x3` f32 recipes. The gfx942 selection models were retrained to rank these routes, with new models for the `x3`, `x4`, and `x9` recipes.
* Added trinary contractions, contractions of mixed data types, and tensors of 2 GiB and more to the rocKE backend. A trinary contraction runs as two contractions with the intermediate tensor in the workspace. A, B, C, and D can each be f16, bf16, f32, or f64 (or each complex f32 or complex f64), and the result is rounded once to D's type.
* Added rocKE's tuned reduction kernels to the rocKE backend's kernel pack: strided reductions that read A in its own layout, with multi-workgroup splits for large reductions. The selection uses rocKE's reduction route table on gfx950 and the nearest measured reduction otherwise, and lowers the split to fit the workspace limit. `HIPTENSOR_ROCKE_DISABLE_TUNED` falls back to the generic reduction kernels.
* Added a flat elementwise kernel to the rocKE backend's kernel pack (gfx942 and gfx950) for f16 and bf16 permutations and elementwise operations whose tensors are one contiguous run in the same order. It runs without index arithmetic, with 16-byte vector loads and stores. `HIPTENSOR_ROCKE_DISABLE_TUNED` turns it off.
* Added rocKE's tuned tiled elementwise kernels to the rocKE backend's kernel pack (gfx942 and gfx950) for permutations and elementwise operations whose inputs aren't all laid out like the output. They transpose through an LDS tile with 16-byte loads and stores, using the kernel measured for each problem of rocKE's tuned elementwise route table (gfx950). `HIPTENSOR_ROCKE_DISABLE_TUNED` turns them off.
* Added a contraction JIT to the rocKE backend (gfx950): with `HIPTENSOR_JIT_MODE_DEFAULT`, f16 and bf16 contractions that rocKE's route tables don't hold are planned by rocKE's tensor-contraction planner at plan creation. The chosen kernels are compiled with COMGR and kept in the kernel cache (also `HIPTENSOR_KERNEL_CACHE_DIR`), and `HIPTENSOR_JIT_MAX_MS` limits the compile time of a plan. `HIPTENSOR_ROCKE_JIT=0` turns it off.
* Added the mixed elementwise data type combinations of cuTENSOR 2.x to the rocKE backend, for example f32 into f64 permutations and binary operations of an f32 A with an f16 C. The rocKE backend computes them in f32 (f64 when a tensor is f64) and rounds D once. The Composable Kernel backend has no kernels for them, so they need `HIPTENSOR_BACKEND=rocke`.
* Extended the rocKE backend's JIT to every elementwise launch of its kernel pack, including the launches of mixed-type contractions, launches with 64-bit offsets, and launches with the operators `HIPTENSOR_OP_SIN` to `HIPTENSOR_OP_FLOOR`. The flat elementwise, GETT, and reduction launches keep their kernel-pack kernels.
* Added fp64 support to hiptensorPermute for ranks 2-6.
* Added bf16 support to hiptensorPermute for ranks 2-6, with a bf16 or fp32 scalar type.
* Added bf16 support to `hiptensorElementwiseTrinaryExecute` for ranks 2-6, with a bf16 or fp32 scalar type, and fp32 scalar support for fp16 tensors. Trinary now accepts the same data type combinations as `hiptensorElementwiseBinaryExecute`.

### Changed
* Changed the default f32 recipe of the rocKE backend to `x6` on gfx950, whose results differ in the low bits from the native f32 GEMM within the f32 accuracy class. gfx942 and gfx90a keep `native`, and `HIPTENSOR_ROCKE_F32_RECIPE` overrides the default.
* Changed the rocKE backend's contraction selection so that its models rank every route, including the contraction JIT, the ring GETT, and the direct GETT, which previously went first by rule. The gfx950 and gfx942 models were retrained, and the new CPU test `rocke_select_model_test` checks every shipped model. `HIPTENSOR_ROCKE_SELECT_MODELS` and `HIPTENSOR_ROCKE_SELECT_DUMP` support training and evaluating the models.
* Changed the plan cache to keep its entries per backend: the hash key of an entry includes the backend name, so plans of one backend never take a kernel cached for another. The plan cache file format is unchanged.
* Changed `hiptensorWriteKernelCacheToFile` and `hiptensorReadKernelCacheFromFile` to return `HIPTENSOR_STATUS_NOT_INITIALIZED` for a null `handle` and `HIPTENSOR_STATUS_INVALID_VALUE` for a null `fileName`. With no backend that keeps a kernel cache, both still succeed without touching the file.
* Extended the rocKE backend's tuned tiled elementwise kernels (gfx942 and gfx950) to layouts whose extents or strides aren't multiples of 16 bytes, which previously ran on the generic kernel. These ragged kernels use the widest vector the layout keeps aligned.
* Changed `hiptensorPermute` to return `HIPTENSOR_STATUS_NOT_SUPPORTED` on the Composable Kernel backend when the output data type differs from the input's, instead of failing with `HIPTENSOR_STATUS_INTERNAL_ERROR`. The rocKE backend serves the f32 into f64 permutation.
* Changed `hiptensorElementwiseTrinaryExecute` to return `HIPTENSOR_STATUS_NOT_SUPPORTED` on the Composable Kernel backend when the data type of D differs from that of C, matching `hiptensorElementwiseBinaryExecute`, instead of failing with `HIPTENSOR_STATUS_INTERNAL_ERROR`. The rocKE backend serves the combinations its kernel pack converts to.
* Changed the `hiptensor::hiptensor` CMake target to propagate `hip::host` instead of `hip::device`. Code that links hipTensor is no longer compiled as HIP for every GPU target by default. Projects that compile their own HIP kernels must link `hip::device` themselves.
* Changed the build to require only the Composable Kernel headers. hipTensor uses the `headers` component of the Composable Kernel package when available, so Composable Kernel can be built with `-DCK_HEADERS_ONLY=ON` without compiling any instance library, and otherwise falls back to the include directories of `device_other_operations`.

### Optimized
* Reduced build time by compiling library, test, and sample sources that define no GPU kernels for the host only, and by removing the unused CPU reference implementations from `libhiptensor.so`. They are now built only into the test suite's host reference library.

### Resolved issues
* Fixed concurrent calls to `hiptensorContract`, `hiptensorContractTrinary`, `hiptensorPermute`, `hiptensorElementwiseBinaryExecute`, `hiptensorElementwiseTrinaryExecute`, and `hiptensorReduce` that select the same kernel crashing, returning another call's result, or returning `HIPTENSOR_STATUS_CK_ERROR`. Plan creation, workspace estimation, and plan cache autotuning are also now safe to call concurrently, and a plan can be executed by several threads at once.
* Fixed `hiptensorContractTrinary` returning `HIPTENSOR_STATUS_INTERNAL_ERROR` on the first call of a plan created after the same problem was already in the plan cache.
* Fixed `hiptensorContract` and the other execute functions searching every candidate kernel on each call when the plan cache already holds the plan's kernel.
* Fixed `HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL` caching the wrong kernel when calls on different problems of the same operation were interleaved. Autotuning counted calls and kept the fastest kernel per operation type, so timings from one problem could select the kernel cached for another. Autotuning is now tracked per problem and handle. A failed `hiptensorContract` or `hiptensorContractTrinary` call is no longer cached as the fastest kernel; autotuning skips the failed candidate and moves on to the next one. Calls that share a plan each run the candidate they were given, and a round caches its kernel only after all of its candidates finish.
* Fixed `hiptensorPermute` ignoring alpha when a plan first executed with alpha equal to 1 was executed again with another alpha. The plan kept the copy-only kernel of its first call; a call that needs scaling now selects a kernel that scales.
* Fixed `hiptensorReduce` reporting 0 TFlops/s and 0 GB/s in `HIPTENSOR_LOG_LEVEL_PERF_TRACE` logs.
* Fixed `hiptensorDestroy` aborting with a double free, or corrupting the heap, after `hiptensorHandleWritePlanCacheToFile` was called on a handle whose plan cache held entries.
* Fixed a full plan cache growing past its size limit when a new problem was added, and evicting its least recently used entry when an existing problem was updated instead. A new entry now evicts the least recently used one, and updating an existing entry evicts nothing.
* Fixed `hiptensorElementwiseBinaryExecute` and `hiptensorReduce` treating the alpha, beta, and gamma scalars as 0 when the compute descriptor is `HIPTENSOR_COMPUTE_DESC_16BF`.
* Fixed hipTensor terminating the application while the library was loading, before `main()`, when the current GPU was unsupported or a HIP device query failed. The device is now checked when `hiptensorCreate` is called on the calling thread's current device: it returns `HIPTENSOR_STATUS_ARCH_MISMATCH` for an unsupported GPU and `HIPTENSOR_STATUS_HIP_ERROR` when HIP initialization or the device query fails, and leaves the handle set to `nullptr`. The error log says whether the architecture or the warp size was rejected. `hiptensorCreate` returns `HIPTENSOR_STATUS_INVALID_VALUE` when `handle` is `nullptr`. hipTensor no longer calls `exit()` when a HIP call fails.
* Fixed complex contractions reporting a device out-of-memory condition as an unsupported problem. `hiptensorCreatePlan` and `hiptensorContract` now return `HIPTENSOR_STATUS_ALLOC_FAILED` when the scratch buffers for the real and imaginary parts can't be allocated.
* Fixed `hiptensorReduce`, `hiptensorPermute`, `hiptensorElementwiseBinaryExecute`, and `hiptensorElementwiseTrinaryExecute` launching kernels on the wrong GPU when the current device differs from the one the handle was created on. Like the contraction APIs, they now return `HIPTENSOR_STATUS_ARCH_MISMATCH` in that case.

## hipTensor 2.5.0 for ROCm 10.2

### Added
* Added broadcast support to the elementwise operations. An input tensor of `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, and `hiptensorCreateElementwiseTrinary` may now carry a subset of the output tensor's modes, in which case it's broadcast along the modes it lacks. For example, `A` with modes `{n}` can be combined with `B`, `C`, and `D` with modes `{m,n}`.
* Added `gfx1250-strict` as a supported offload target and classified its device pass as `gfx1250`.

### Changed
* Changed `hiptensorCreatePermutation`, `hiptensorCreateElementwiseBinary`, and `hiptensorCreateElementwiseTrinary` to return `HIPTENSOR_STATUS_INVALID_VALUE` when a mode shared by an input tensor and the output tensor has a different extent in each, and `HIPTENSOR_STATUS_NOT_SUPPORTED` when a mode is repeated within one tensor. An input tensor that carries a mode the output tensor doesn't carry is still rejected with `HIPTENSOR_STATUS_NOT_SUPPORTED`, since that would have to be reduced away rather than broadcast.
* Changed `libhiptensor.so` to export symbols with hidden visibility by default, so only the documented public C API is exposed. Internal C++ classes and Composable Kernel template instantiations that were previously visible are now local to the library. Code that relied on those undocumented internal symbols will no longer find them.

### Resolved issues
* Fixed `hiptensorPermute` and the element-wise binary/trinary execute paths ignoring user-supplied output tensor strides, which caused the output to always be written contiguously regardless of the strides set on the output descriptor.
* Fixed elementwise operations reading the wrong elements when two input tensors ordered the same modes differently. Every tensor's strides are now permuted into the output tensor's mode order before reaching the kernel.

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
