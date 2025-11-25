# Changelog for rocBLAS

rocBLAS documentation is available at
[https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html).

## rocBLAS 5.1.0 for ROCm 7.1

### Added
* Sample for clients using OpenMP threads calling rocBLAS functions.
* gfx1103, gfx1150, and gfx1151 enabled.

### Changed
* By default, the Tensile build is no longer based on `tensile_tag.txt` but uses the same commit from shared/tensile in the rocm-libraries repository. The rmake or install `-t` option can build from another local path with a different commit.

### Optimized

* Improved the performance of Level 2 gemv transposed (`TransA != N`) for the problem sizes where `m` is small and `n` is large on gfx90a and gfx942.

## rocBLAS 5.0.0 for ROCm 7.0

### Added

* gfx950 support
* `ROCBLAS_LAYER = 8` internal API logging for `gemm` debugging
* Support for AOCL 5.0 gcc build as a client reference library
* Allow `PkgConfig` for client reference library fallback detection

### Changed

* `CMAKE_CXX_COMPILER` is now passed on during compilation for a Tensile build
* Change default atomics mode from `allowed` to `not allowed`

### Removed

* Support code for non-production gfx targets
* `rocblas_hgemm_kernel_name`, `rocblas_sgemm_kernel_name`, and `rocblas_dgemm_kernel_name` API functions
* Use of `warpSize` as a constexpr
* Use of deprecated behavior of `hipPeekLastError`
* `rocblas_is_user_managing_device_memory` and `rocblas_set_device_memory_size` API functions
* `rocblas_float8.h` and `rocblas_hip_f8_impl.h` files
* `rocblas_gemm_ex3`, `rocblas_gemm_batched_ex3`, `rocblas_gemm_strided_batched_ex3` API functions

### Optimized

* Optimized `gemm` by using `gemv` kernels when applicable
* Optimized `gemv` for small `m` and `n` with a large batch count on gfx942
* Improved the performance of Level 1 `dot` for all precisions and variants when `N > 100000000` on gfx942
* Improved the performance of Level 1 `asum` and `nrm2` for all precisions and variants on gfx942
* Improved the performance of Level 2 `sger` (single precision) on gfx942
* Improved the performance of Level 3 `dgmm` for all precisions and variants on gfx942

### Resolved issues

* Fixed environment variable path-based logging to append multiple handle output to the same file
* Support numerics when `trsm` is running with `rocblas_status_perf_degraded`
* Fixed the build dependency installation of `joblib` on some operating systems
* Return `rocblas_status_internal_error` when `rocblas_[set,get]_ [matrix,vector]` is called with a host pointer in place of a device pointer
* Reduced the default verbosity level for internal GEMM backend information
* Updated from the deprecated rocm-cmake to ROCmCMakeBuildTools
* Corrected AlmaLinux gfortran package dependencies
* Included upgrade of `packaging` in addition to `pip` in Tensile Virtual Environment to stabilize `setuptools` from failures

### Upcoming changes

* Deprecated the use of negative indices to indicate the default solution is being used for `gemm_ex` with `rocblas_gemm_algo_solution_index`

## rocBLAS 4.4.0 for ROCm 6.4

### Added

* rocTX support in rocBLAS (not available on Windows or in the static library version on Linux)
* On gfx12, all functions now support full `rocblas_int` dynamic range for `batch_count`
* `--ninja` build option
* Support for GPU_TARGETS cmake variable

### Changed

* rocblas-test client removes the stress tests unless YAML-based testing or `gtest_filter` adds them
* rocblas clients OpenMP default threading is reduced to be less than the logical core count
* `gemm_ex` testing and timing reuses device memory
* `gemm_ex` timing initializes matrices on device

### Optimized

* Significantly reduced workspace memory requirements for Level 1 ILP64: `iamax` and `iamin`
* Reduced workspace memory requirements for Level 1 ILP64: `dot`, `asum`, `nrm2`
* Improved the performance of Level 2 gemv for the problem sizes (`TransA == N && m > 2*n`) and (`TransA == T`)
* Improved the performance of Level 3 syrk and herk for the problem size (`k > 500 && n < 4000`)

### Resolved issues

* gfx12: `ger`, `geam`, `geam_ex`, `dgmm`, `trmm`, `symm`, `hemm`, ILP64 `gemm`, and larger data support
* Added a `gfortran` package dependency for Azure Linux OS
* Outdated SLES OS package dependencies (`cxxtools` and `joblib`) in `install.sh -d`
* Code object stripping for RPM packages

### Upcoming changes

* Deprecated the cmake variable `AMDGPU_TARGETS`. Use `GPU_TARGETS` instead.

## rocBLAS 4.3.0 for ROCm 6.3

### Added

* Level 3 and EX functions have an additional ILP64 API for both C and FORTRAN (_64 name suffix) with int64_t function arguments

### Changed

* amdclang is used as the default compiler instead of hipcc
* Internal performance scripts use amd-smi instead of the deprecated rocm-smi

### Optimized

* Improved performance of Level 2 gbmv
* Improved performance of Level 2 gemv for float and double precisions for problem sizes (TransA == N && m==n && m % 128 == 0) measured on a gfx942 GPU

### Resolved issues

* Fixed stbsv_strided_batched_64 Fortran binding

### Upcoming changes

* rocblas_Xgemm_kernel_name APIs are deprecated

## rocBLAS 4.2.1 for ROCm 6.2.1

### Removed

* Removed Device_Memory_Allocation.pdf link in documentation.

### Resolved issues

* Fixed error/warning message during `rocblas_set_stream()` call.

## rocBLAS 4.2.0 for ROCm 6.2

### Added

* Level 2 functions and level 3 trsm have additional ILP64 API for both C and FORTRAN (_64 name suffix) with int64_t function arguments
* Cache flush timing for gemm_batched_ex, gemm_strided_batched_ex, axpy
* Benchmark class for common timing code
* An environment variable "ROCBLAS_DEFAULT_ATOMICS_MODE" to set default atomics mode during creation of 'rocblas_handle'
* Extended dot_ex to support single-precision (fp32_r) input and double-precision (fp64_r) output and compute types

### Optimized

* Improved performance of Level 1 dot_batched and dot_strided_batched for all precisions. Performance enhanced by 6 times for bigger problem sizes measured on MI210 GPU

### Changed

* Linux AOCL dependency updated to release 4.2 gcc build
* Windows vcpkg dependencies updated to release 2024.02.14
* Increased default device workspace from 32 to 128 MiB for architecture gfx9xx with xx >= 40

### Upcoming changes

* rocblas_gemm_ex3, gemm_batched_ex3 and gemm_strided_batched_ex3 are deprecated and will be removed in the next major release of rocBLAS. Please refer to hipBLASLt for future 8 bit float usage https://github.com/ROCm/hipBLASLt

## rocBLAS 4.1.0 for ROCm 6.1

### Added

* Level 1 and Level 1 Extension functions have additional ILP64 API for both C and Fortran (`_64` name
  suffix) with int64_t function arguments
* Cache flush timing for `gemm_ex`

### Changed

* Some Level 2 function argument names have changed `m` to `n` to match legacy BLAS; there is no
  change in implementation
* Standardized the use of non-blocking streams for copying results from device to host

### Resolved issues

* Fixed host-pointer mode reductions for non-blocking streams

## rocBLAS 4.0.0 for ROCm 6.0

### Added

* Beta API `rocblas_gemm_batched_ex3` and `rocblas_gemm_strided_batched_ex3`
* Input/output type f16_r/bf16_r and execution type f32_r support for Level 2 gemv_batched and
  gemv_strided_batched: `rocblas_hshgemv_batched/strided_batched`, `rocblas_hssgemv_batched/strided_batched`,
  `rocblas_tstgemv_batched/strided_batched` and `rocblas_tssgemv_batched/strided_batched`
* Use of `rocblas_status_excluded_from_build` when calling functions that require Tensile (when using
  rocBLAS built without Tensile)
* System for asynchronous kernel launches that set a `rocblas_status` failure based on a
  `hipPeekAtLastError` discrepancy

### Optimized

* TRSM performance for small sizes (m < 32 && n < 32)

### Upcoming changes

* Atomic operations will be disabled by default in a future release of rocBLAS

### Removed

* `rocblas_gemm_ext2` API function
* In-place trmm API from Legacy BLAS is replaced by an API that supports both in-place and
  out-of-place trmm
* int8x4 support is removed (int8 support is unchanged)
* `#define __STDC_WANT_IEC_60559_TYPES_EXT__` is removed from `rocblas-types.h` (if you want
  ISO/IEC TS 18661-3:2015 functionality, you must define `__STDC_WANT_IEC_60559_TYPES_EXT__`
  before including `float.h`, `math.h`, and `rocblas.h`)
* The default build removes device code for gfx803 architecture from the fat binary

### Resolved issues

* Made offset calculations for 64-bit rocBLAS functions safe
  * Fixes for very large leading dimension or increment potentially causing overflow:
    * Level2: `gbmv`, `gemv`, `hbmv`, `sbmv`, `spmv`, `tbmv`, `tpmv`, `tbsv`, and `tpsv`
* Lazy loading supports heterogeneous architecture setup and load-appropriate tensile library files,
  based on device architecture
* Guards against no-op kernel launches that result in a potential `hipGetLastError`

### Changed

* Reduced the default verbosity of `rocblas-test` (you can see all tests by setting the
  `GTEST_LISTENER=PASS_LINE_IN_LOG` environment variable)

## rocBLAS 3.1.0 for ROCm 5.7

### Added

* YAML lock step argument scans for `rocblas-bench` and `rocblas-test` clients
* `rocblas-gemm-tune` is used to find the best performing GEMM kernel for each set of GEMM
  problems

### Resolved issues

* Made offset calculations for 64-bit rocBLAS functions safe
  * Fixes for very large leading dimensions or increments potentially causing overflow:
    * Level 1: `axpy`, `copy`, `rot`, `rotm`, `scal`, `swap`, `asum`, `dot`, `iamax, `iamin`,and `nrm2`
    * Level 2: `gemv`, `symv`, `hemv`, `trmv`, `ger`, `syr`, `her`, `syr2`, `her2`, and `trsv`
    * Level 3: `gemm`, `symm`, `hemm`, `trmm`, `syrk`, `herk`, `syr2k`, `her2k`, `syrkx`, `herkx`, `trsm`, `trtri`,
      `dgmm`, and `geam`
    * General: `set_vector`, `get_vector`, `set_matrix`, and `get_matrix`
    * Related fixes: internal scalar loads with > 32-bit offsets
    * In-place functionality for all `trtri` sizes

### Changed

* Dot when using `rocblas_pointer_mode_host` is now synchronous in order to match legacy BLAS as it
  stores results in host memory
* Enhanced reporting of installation issues caused by runtime libraries (Tensile)
* Standardized internal rocBLAS C++ interface across most functions

### Upcoming changes

* `__STDC_WANT_IEC_60559_TYPES_EXT__` define will be removed in a future release

### Dependencies

* Optional use of AOCL BLIS 4.0 on Linux for clients
* Optional build tool-only dependency on Python `psutil`

## rocBLAS 3.0.0 for ROCm 5.6

### Optimized

* Level 2 rocBLAS GEMV performance on gfx90a GPU for non-transposed problems that have small
  matrices (`m` and `n` <= 32) and large batch counts (`batch_count` >= 256)
* rocBLAS syr2k performance for single, double, and double-complex precision
* rocBLAS her2k performance for double-complex precision
* Improved performance for general sizes on gfx90a

### Added

* bf16 inputs and f32 compute support to Level 1 rocBLAS extension functions: `axpy_ex`, `scal_ex`, and
  `nrm2_ex`

### Upcoming changes

* In-place trmm has been replaced with trmm that has in-place and out-of-place functionality
* `rocblas_query_int8_layout_flag()`
* `rocblas_gemm_flags_pack_int8x4`
* `rocblas_set_device_memory_size()` will be replaced with `rocblas_increase_device_memory_size()`
* `rocblas_is_user_managing_device_memory()`

### Removed

* `is_complex` helper: use `rocblas_is_complex` instead
* The enum `truncate_t`: use `rocblas_truncate_t` instead
* The value `truncate`: use `rocblas_truncate` instead
* `rocblas_set_int8_type_for_hipblas`
* `rocblas_get_int8_type_for_hipblas`

### Dependencies

* Python `joblib` build-only dependency (used in Tensile builds)

### Resolved issues

* Made 64-bit trsm offset calculations safe
* CMake install fixed on some operating systems when using `install.sh -d --cmake_install`

### Changed

* Refactored ROTG test code

## rocBLAS 2.47.0 for ROCm 5.5.0

### Added

* `rocblas_geam_ex` functionality for matrix-matrix minimum operations
* HIP Graph support (beta feature) for rocBLAS Level 1, Level 2, and Level 3 (pointer mode host) functions
* Beta features API, exposed using compiler define `ROCBLAS_BETA_FEATURES_API`
* Support for vector initialization in the rocBLAS test framework with negative increments
* Windows build documentation for HIP SDK support
* Scripts for plotting the performance of multiple functions

### Optimized

* Performance improvements for Level 2 rocBLAS GEMV for float and double precision (150-200%
  improvement for certain problem sizes when (m==n) measured on a gfx90a GPU)
* Performance improvements for Level 2 rocBLAS GER for float, double, and complex float precisions
  (5-7% improvement for certain problem sizes when measured on a gfx90a GPU)
* Performance improvements for Level 2 rocBLAS SYMV for float and double precisions (120-150%
  improvement for certain problem sizes measured on both gfx908 and gfx90a GPUs)

### Resolved issues

* Executable mode setting on `rocblas_gentest.py` client script to avoid potential permission errors with
  clients `rocblas-test` and `rocblas-bench`
* Deprecated API compatibility with Visual Studio compiler
* Test framework memory exception handling for Level 2 functions when the host memory allocation
  exceeds the available memory

### Changed

* `install.sh` internally runs `rmake.py` (also used on Windows) and `rmake.py` can be used directly on
  Linux (use `--help`)
* rocBLAS client executables all now begin with the `rocblas-` prefix

### Removed

* `install.sh` no longer has the options `-o --cov` because Tensile will now use the default COV format,
  which is set by `cmake define Tensile_CODE_OBJECT_VERSION=default`

## rocBLAS 2.46.0 for ROCm 5.4.0

### Added

- client smoke test dataset added for quick validation using command `rocblas-test --yaml rocblas_smoke.yaml`
- Added stream order device memory allocation as a non-default beta option.

### Optimized

- Improved trsm performance for small sizes by using a substitution method technique
- Improved syr2k and her2k performance significantly by using a block-recursive algorithm

### Changed

- Level 2, Level 1, and Extension functions: argument checking when the handle is set to `rocblas_pointer_mode_host` now returns the status of `rocblas_status_invalid_pointer` only for pointers that must be dereferenced based on the alpha and beta argument values.  With handle mode `rocblas_pointer_mode_device` only pointers that are always dereferenced regardless of alpha and beta values are checked and so may lead to a return status of `rocblas_status_invalid_pointer`.   This improves consistency with legacy BLAS behavior.
- Add variable to turn on/off ieee16/ieee32 tests for mixed precision gemm
- Allow hipBLAS to select int8 datatype
- Disallow B == C && ldb != ldc in `rocblas_xtrmm_outofplace`

### Resolved issues

- Fortran interfaces generalized for Fortran compilers other than GFortran
- fix for `trsm_strided_batched rocblas-bench` performance gathering
- Fix for rocm-smi path in `commandrunner.py` script to match ROCm 5.2 and above

## rocBLAS 2.45.0 for ROCm 5.3.0

### Added

- `install.sh` option `--upgrade_tensile_venv_pip` to upgrade Pip in Tensile Virtual Environment. The corresponding CMake option is TENSILE_VENV_UPGRADE_PIP
- `install.sh` option `--relocatable` or `-r` adds rpath and removes ldconf entry on rocBLAS build
- `install.sh` option `--lazy-library-loading` to enable on-demand loading of tensile library files at runtime to speedup rocBLAS initialization
- Support for RHEL9 and CS9
- Added Numerical checking routine for symmetric, Hermitian, and triangular matrices, so that they could be checked for any numerical abnormalities such as NaN, Zero, infinity and denormal value

### Optimized

- `trmm_outofplace` performance improvements for all sizes and data types using block-recursive algorithm
- herkx performance improvements for all sizes and data types using block-recursive algorithm
- syrk/herk performance improvements by utilising optimised syrkx/herkx code
- symm/hemm performance improvements for all sizes and datatypes using block-recursive algorithm

### Changed

- Unifying library logic file names: affects HBH (->HHS_BH), BBH (->BBS_BH), 4xi8BH (->4xi8II_BH). All HPA types are using the new naming convention now.
- Level 3 function argument checking when the handle is set to rocblas_pointer_mode_host now returns the status of rocblas_status_invalid_pointer only for pointers that must be dereferenced based on the alpha and beta argument values. With handle mode rocblas_pointer_mode_device only pointers that are always dereferenced regardless of alpha and beta values are checked and so may lead to a return status of rocblas_status_invalid_pointer. This improves consistency with legacy BLAS behaviour
- Level 1, 2, and 3 function argument checking for enums is now more rigorously matching legacy BLAS so returns rocblas_status_invalid_value if arguments do not match the accepted subset
- Add quick-return for internal trmm and gemm template functions
- Moved function block sizes to a shared header file
- Level 1, 2, and 3 functions use rocblas_stride datatype for offset
- Modified the matrix and vector memory allocation in our test infrastructure for all Level 1, 2, 3 and BLAS_EX functions
- Added specific initialization for symmetric, Hermitian, and triangular matrix types in our test infrastructure
- Added NaN tests to the test infrastructure for the rest of Level 3, BLAS_EX functions

### Resolved issues

- Improved logic to #include <filesystem> vs <experimental/filesystem>
- `install.sh -s` option to build rocblas as a static library.
- dot function now sets the device results asynchronously for N <= 0

### Upcoming changes

- is_complex helper is now deprecated.  Use `rocblas_is_complex` instead
- The enum `truncate_t` and the value truncate is now deprecated and will removed from the ROCm release 6.0. It is replaced by `rocblas_truncate_t` and `rocblas_truncate`, respectively. The new enum `rocblas_truncate_t` and the value `rocblas_truncate` could be used from this ROCm release for an easy transition

### Removed

- `install.sh` options  `--hip-clang`, `--no-hip-clang`, `--merge-files`, `--no-merge-files` are removed

## rocBLAS 2.44.0 for ROCm 5.2.0

### Added

- Packages for test and benchmark executables on all supported operating systems using CPack
- Added denormal number detection to the Numerical checking helper function to detect denormal/subnormal numbers in the input and the output vectors of rocBLAS level 1 and 2 functions
- Added denormal number detection to the numerical checking helper function to detect denormal/subnormal numbers in the input and the output general matrices of rocBLAS level 2 and 3 functions
- Added NaN initialization tests to the YAML files of Level 2 rocBLAS batched and strided-batched functions for testing purposes
- Added memory allocation check to avoid disk swapping during rocblas-test runs by skipping tests

### Optimized

- Improved performance of non-batched and batched her2 for all sizes and data types
- Improved performance of non-batched and batched amin for all data types using shuffle reductions
- Improved performance of non-batched and batched amax for all data types using shuffle reductions
- Improved performance of trsv for all sizes and data types

### Changed

- Modifying `gemm_ex` for HBH (high-precision F16). The alpha/beta data type remains as F32 without narrowing to F16 and expanding back to F32 in the kernel. This change prevents rounding errors due to alpha/beta conversion in situations where alpha/beta are not exactly represented as an F16
- Modified non-batched and batched asum, nrm2 functions to use shuffle instruction based reductions
- For `gemm`, `gemm_ex`, `gemm_ex2` internal API use `rocblas_stride` datatype for offset
- For symm, hemm, syrk, herk, dgmm, geam internal API use `rocblas_stride` datatype for offset
-  AMD copyright year for all rocBLAS files
- For `gemv` (transpose-case), typecasted the 'lda'(offset) datatype to `size_t` during offset calculation to avoid overflow and remove duplicate template functions

### Resolved issues

- For function her2 avoid overflow in offset calculation
- For trsm when alpha == 0 and on host, allow A to be nullptr
- Fixed memory access issue in trsv
- Fixed git pre-commit script to update only AMD copyright year
- Fixed dgmm, geam test functions to set correct stride values
- For functions ssyr2k and dsyr2k allow trans == `rocblas_operation_conjugate_transpose`
- Fixed compilation error for clients-only build

### Removed

- Remove Navi12 (gfx1011) from fat binary

## rocBLAS 2.43.0 for ROCm 5.1.0

### Added

- Option to install script for number of jobs to use for rocBLAS and Tensile compilation (`-j`, `--jobs`)
- Option to install script to build clients without using any Fortran (`--clients_no_fortran`)
- `rocblas_client_initialize function`, to perform rocBLAS initialize for clients(benchmark/test) and report the execution time.
- Added tests for output of reduction functions when given bad input
- Added user specified initialization (`rand_int`/`trig_float`/`hpl`) for initializing matrices and vectors in `rocblas-bench`

### Optimized

- Improved performance of trsm with side == left and n == 1
- Improved performance of trsm with side == left and m <= 32 along with side == right and n <= 32

### Changed

- For syrkx and trmm internal API use `rocblas_stride` datatype for offset
- For non-batched and batched gemm_ex functions if the C matrix pointer equals the D matrix pointer (aliased) their respective type and leading dimension arguments must now match
- Test client dependencies updated to GTest 1.11
- non-global false positives reported by cppcheck from file based suppression to inline suppression. File based suppression will only be used for global false positives
- Help menu messages in `install.sh`
- For ger function, typecast the 'lda'(offset) datatype to `size_t` during offset calculation to avoid overflow and remove duplicate template functions
- Modified default initialization from `rand_int` to hpl for initializing matrices and vectors in `rocblas-bench`

### Resolved issues

- For function trmv (non-transposed cases) avoid overflow in offset calculation
- Fixed cppcheck errors/warnings
- Fixed Doxygen warnings

## rocBLAS 2.42.0 for ROCm 5.0.0

### Added

- Added `rocblas_get_version_string_size` convenience function
- Added `rocblas_xtrmm_outofplace`, an out-of-place version of `rocblas_xtrmm`
- Added hpl and trig initialization for `gemm_ex` to `rocblas-bench`
- Added source code gemm. It can be used as an alternative to Tensile for debugging and development
- Added option `ROCM_MATHLIBS_API_USE_HIP_COMPLEX` to opt-in to use `hipFloatComplex` and `hipDoubleComplex`

### Optimized

- Improved performance of non-batched and batched single-precision GER for size m > 1024. Performance enhanced by 5-10% measured on a MI100 (gfx908) GPU.
- Improved performance of non-batched and batched HER for all sizes and data types. Performance enhanced by 2-17% measured on a MI100 (gfx908) GPU.

### Changed

- Instantiate templated rocBLAS functions to reduce size of `librocblas.so`
- Removed static library dependency on msgpack
- Removed boost dependencies for clients

### Resolved issues

- Option to install script to build only rocBLAS clients with a pre-built rocBLAS library
- Correctly set output of `nrm2_batched_ex` and `nrm2_strided_batched_ex` when given bad input
- Fix for dgmm with side == `rocblas_side_left` and a negative incx
- Fixed out-of-bounds read for small trsm
- Fixed numerical checking for `tbmv_strided_batched`

## rocBLAS 2.41.0 for ROCm 4.5.0

### Optimized

- Improved performance of non-batched and batched syr for all sizes and data types
- Improved performance of non-batched and batched hemv for all sizes and data types
- Improved performance of non-batched and batched symv for all sizes and data types
- Improved memory utilization in `rocblas-bench`, `rocblas-test` gemm functions, increasing possible runtime sizes.
- Improved performance of non-batched and batched dot, dotc, and dot_ex for small n. e.g. sdot n <= 31000.
- Improved performance of non-batched and batched trmv for all sizes and matrix types.
- Improved performance of non-batched and batched gemv transpose case for all sizes and datatypes.
- Improved performance of sger and dger for all sizes, in particular the larger dger sizes.
- Improved performance of syrkx for for large size including those in rocBLAS Issue #1184.

### Changed

- Update from C++14 to C++17.
- Packaging split into a runtime package (called rocblas) and a development package (called rocblas-dev for `.deb` packages, and rocblas-devel for `.rpm` packages). The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.

### Resolved issues

- For function geam avoid overflow in offset calculation.
- For function syr avoid overflow in offset calculation.
- For function gemv (Transpose-case) avoid overflow in offset calculation.
- For functions ssyrk and dsyrk, allow conjugate-transpose case to match legacy BLAS. Behavior is the same as the transpose case.

## rocBLAS 2.39.0 for ROCm 4.3.0

### Optimized

- Improved performance of non-batched and batched `rocblas_Xgemv` for gfx908 when m <= 15000 and n <= 15000
- Improved performance of non-batched and batched `rocblas_sgemv` and `rocblas_dgemv` for gfx906 when m <= 6000 and n <= 6000
- Improved the overall performance of non-batched and `batched rocblas_cgemv` for gfx906
- Improved the overall performance of `rocblas_Xtrsv`

### Changed

- Internal use only APIs prefixed with `rocblas_internal_` and deprecated to discourage use

## rocBLAS 2.38.0 for ROCm 4.2.0

### Added

- Added option to install script to build only rocBLAS clients with a pre-built rocBLAS library
- Supported gemm ext for unpacked int8 input layout on gfx908 GPUs
  - Added new flags `rocblas_gemm_flags::rocblas_gemm_flags_pack_int8x4` to specify if using the packed layout
    - Set the `rocblas_gemm_flags_pack_int8x4` when using packed int8x4, this should be always set on GPUs before gfx908.
    - For gfx908 GPUs, unpacked int8 is supported so no need to set this flag.
    - Notice the default flags 0 uses unpacked int8, this somehow changes the behaviour of int8 gemm from ROCm 4.1.0
- Added a query function `rocblas_query_int8_layout_flag` to get the preferable layout of int8 for gemm by device

### Optimized

- Improved performance of single precision copy, swap, and scal when `incx` == 1 and `incy` == 1
- Improved performance of single precision axpy when `incx` == 1, `incy` == 1 and `batch_count` =< 8192
- Improved performance of trmm

### Changed

- Change `cmake_minimum_required` to VERSION 3.16.8

## rocBLAS 2.36.0 for ROCm 4.1.0

### Added

- Added Numerical checking helper function to detect zero/NaN/Inf in the input and the output vectors of rocBLAS level 1 and 2 functions
- Added Numerical checking helper function to detect zero/NaN/Inf in the input and the output general matrices of rocBLAS level 2 and 3 functions

### Resolved issues

- Fixed complex unit test bug caused by incorrect caxpy and zaxpy function signatures.
- Make functions compliant with Legacy Blas for special values `alpha` == 0, `k` == 0, `beta` == 1, `beta` == 0

### Optimized

- Improved performance of single precision `axpy_batched` and `axpy_strided_batched`: `batch_count` >= 8192
- Improved performance of trmm.

## rocBLAS 2.34.0 for ROCm 4.0.0

### Added

- Add changelog.
- Improved performance of gemm_batched for small m, n, k and NT, NC, TN, TT, TC, CN, CT, CC
- Improved performance of gemv, gemv_batched, gemv_strided_batched: small n large m
- Removed support for legacy hcc compiler
- Add `rot_ex`, `rot_batched_ex`, and `rot_strided_batched_ex`

### Resolved issues

- Removed `-DUSE_TENSILE_HOST` from `roc::rocblas` CMake usage requirements. This
  is a rocblas internal variable, and does not need to be defined in user code

## rocBLAS 2.32.0 for ROCm 3.10.0

### Added

- Improved performance of `gemm_batched` for NN, general m, n, k, small m, n, k

## rocBLAS 2.30.0 for ROCm 3.9.0

### Added

- Slight improvements to FP16 Megatron BERT performance on MI50
- Improvements to FP16 Transformer performance on MI50
- Slight improvements to FP32 Transformer performance on MI50
- Improvements to FP32 DLRM Terabyte performance on gfx908

## rocBLAS 2.28.0 for ROCm 3.8.0

### Added

- added two functions:
  - `rocblas_status rocblas_set_atomics_mode(rocblas_atomics_mode mode)`
  - `rocblas_status rocblas_get_atomics_mode(rocblas_atomics_mode mode)`
- added enum `rocblas_atomics_mode`. It can have two values
  - `rocblas_atomics_allowed`
  - `rocblas_atomics_not_allowed`
  The default is `rocblas_atomics_not_allowed`
- function `rocblas_Xdgmm` algorithm corrected and `incx`=0 support added
- dependencies:
  - `rocblas-tensile` internal component requires msgpack instead of LLVM
- Moved the following files from /opt/rocm/include to /opt/rocm/include/internal:
  - `rocblas-auxillary.h`
  - `rocblas-complex-types.h`
  - `rocblas-functions.h`
  - `rocblas-types.h`
  - `rocblas-version.h`
  - `rocblas_bfloat16.h`

  These files should NOT be included directly as this may lead to errors. Instead, `/opt/rocm/include/rocblas.h` should be included directly. `/opt/rocm/include/rocblas_module.f90` can also be directly used

## rocBLAS 2.26.0 for ROCm 3.7.0

### Added

- Improvements to `rocblas_Xgemm_batched` performance for small m, n, k
- Improvements to `rocblas_Xgemv_batched`  and `rocblas_Xgemv_strided_batched` performance for small m (QMCPACK use)
- Improvements to `rocblas_Xdot` (batched and non-batched) performance when both incx and incy are 1
- Improvements to FP32 ONNX BERT performance for MI50
- Significant improvements to FP32 Resnext, Inception Convolution performance for gfx908
- Slight improvements to FP32 DLRM Terabyte performance for gfx908
- Significant improvements to FP32 BDAS performance for gfx908
- Significant improvements to FP32 BDAS performance for MI50 and MI60
- Added substitution method for small trsm sizes with m <= 64 && n <= 64. Increases performance drastically for small batched trsm

## rocBLAS 2.24.0 for ROCm 3.6.0

### Added

- Improvements to User Guide and Design Document
- L1 dot function optimized to utilize shuffle instructions (improvements on bf16, f16, f32 data types)
- L1 dot function added x dot x optimized kernel
- Standardization of L1 rocblas-bench to use device pointer mode to focus on GPU memory bandwidth
- Adjustments for hipcc (hip-clang) compiler as standard build compiler and Centos8 support
- Added Fortran interface for all rocBLAS functions

## rocBLAS 2.22.0 for ROCm 3.5.0

### Added

- add `geam complex`, `geam_batched`, and `geam_strided_batched`
- add `dgmm`, `dgmm_batched`, and `dgmm_strided_batched`
- Optimized performance
  - ger
    - `rocblas_sger`, `rocblas_dger`
    - `rocblas_sger_batched`, `rocblas_dger_batched`
    - `rocblas_sger_strided_batched`, `rocblas_dger_strided_batched`
  - geru
    - `rocblas_cgeru`, `rocblas_zgeru`
    - `rocblas_cgeru_batched`, `rocblas_zgeru_batched`
    - `rocblas_cgeru_strided_batched`, `rocblas_zgeru_strided_batched`
  - gerc
    - `rocblas_cgerc`, `rocblas_zgerc`
    - `rocblas_cgerc_batched`, `rocblas_zgerc_batched`
    - `rocblas_cgerc_strided_batched`, `rocblas_zgerc_strided_batched`
  - symv
    - `rocblas_ssymv`, `rocblas_dsymv`, `rocblas_csymv`, `rocblas_zsymv`
    - `rocblas_ssymv_batched`, `rocblas_dsymv_batched`, `rocblas_csymv_batched`, `rocblas_zsymv_batched`
    - `rocblas_ssymv_strided_batched`, `rocblas_dsymv_strided_batched`, `rocblas_csymv_strided_batched`, `rocblas_zsymv_strided_batched`
  - sbmv
    - `rocblas_ssbmv`, `rocblas_dsbmv`
    - `rocblas_ssbmv_batched`, `rocblas_dsbmv_batched`
    - `rocblas_ssbmv_strided_batched`, `rocblas_dsbmv_strided_batched`
  - spmv
    - `rocblas_sspmv`, `rocblas_dspmv`
    - `rocblas_sspmv_batched`, `rocblas_dspmv_batched`
    - `rocblas_sspmv_strided_batched`, `rocblas_dspmv_strided_batched`
- improved documentation.
- Fix argument checking in functions to match legacy BLAS.
- Fixed conjugate-transpose version of geam.

### Known issues

- Compilation for GPU Targets:
When using the install.sh script for "all" GPU Targets, which is the default, you must first set an environment variable `HCC_AMDGPU_TARGET` listing the GPU targets, e.g.  `HCC_AMDGPU_TARGET=gfx803,gfx900,gfx906,gfx908`
If building for a specific architecture(s) using the  `-a` | --architecture flag, you should also set the environment variable `HCC_AMDGPU_TARGET` to match.
Mismatching the environment variable to the `-a` flag architectures creates builds that may result in `SEGFAULTS` when running on GPUs which weren't specified.
