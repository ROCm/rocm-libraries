# Changelog for rocSPARSE

Documentation for rocSPARSE is available at
[https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/).

## (Unreleased) rocSPARSE 4.2.0

### Added

* Added `--clients-only` option to the `install.sh` and `rmake.py` scripts to allow building only the clients while using an already installed version of rocSPARSE.

## rocSPARSE 4.1.0 for ROCm 7.1.0

### Added

* Added brain half float mixed precision to `rocsparse_axpby` where X and Y use bfloat16 and result and the compute type use float.
* Added brain half float mixed precision to `rocsparse_spvv` where X and Y use bfloat16 and result and the compute type use float.
* Added brain half float mixed precision to `rocsparse_spmv` where A and X use bfloat16 and Y and the compute type use float.
* Added brain half float mixed precision to `rocsparse_spmm` where A and B use bfloat16 and C and the compute type use float.
* Added brain half float mixed precision to `rocsparse_sddmm` where A and B use bfloat16 and C and the compute type use float.
* Added brain half float mixed precision to `rocsparse_sddmm` where A and B and C use bfloat16 and the compute type use float.
* Added half float mixed precision to `rocsparse_sddmm` where A and B and C use float16 and the compute type use float.
* Added brain half float uniform precision to `rocsparse_scatter` and `rocsparse_gather` routines.

### Optimized

* Improved the user documentation.

### Upcoming changes

* Deprecate trace, debug, and bench logging using environment variable `ROCSPARSE_LAYER`.

## rocSPARSE 4.0.2 for ROCm 7.0.0

### Added

* Adds `SpGEAM` generic routine for computing sparse matrix addition in CSR format
* Adds `v2_SpMV` generic routine for computing sparse matrix vector multiplication. As opposed to the deprecated `rocsparse_spmv` routine, this routine does not use a fallback algorithm if a non-implemented configuration is encountered and will return an error in such a case. For the deprecated routine `rocsparse_spmv`, the user can enable warning messages in situations where a fallback algorithm is used by either calling upfront the routine `rocsparse_enable_debug` or exporting the variable `ROCSPARSE_DEBUG` (with the shell command `export ROCSPARSE_DEBUG=1`).
* Adds half float mixed precision to `rocsparse_axpby` where X and Y use float16 and result and the compute type use float
* Adds half float mixed precision to `rocsparse_spvv` where X and Y use float16 and result and the compute type use float
* Adds half float mixed precision to `rocsparse_spmv` where A and X use float16 and Y and the compute type use float
* Adds half float mixed precision to `rocsparse_spmm` where A and B use float16 and C and the compute type use float
* Adds half float mixed precision to `rocsparse_sddmm` where A and B use float16 and C and the compute type use float
* Adds half float uniform precision to `rocsparse_scatter` and `rocsparse_gather` routines
* Adds half float uniform precision to `rocsparse_sddmm` routine
* Added `rocsparse_spmv_alg_csr_rowsplit` algorithm.
* Added support for gfx950
* Add ROC-TX instrumentation support in rocSPARSE (not available on Windows or in the static library version on Linux).
* Added the `almalinux` OS name to correct the gfortran dependency

### Changed

* Switch to defaulting to C++17 when building rocSPARSE from source. Previously rocSPARSE was using C++14 by default.

### Optimized

* Reduced the number of template instantiations in the library to further reduce the shared library binary size and improve compile times
* Allow SpGEMM routines to use more shared memory when available. This can speed up performance for matrices with a large number of intermediate products.
* Use of the `rocsparse_spmv_alg_csr_adaptive` or `rocsparse_spmv_alg_csr_default` algorithms in `rocsparse_spmv` to perform transposed sparse matrix multiplication (`C=alpha*A^T*x+beta*y`) resulted in unnecessary analysis on A and needless slowdown during the analysis phase. This has been fixed by skipping the analysis when performing the transposed sparse matrix multiplication.
* Improved the user documentation

### Resolved issues

* Fixed an issue in the public headers where `extern "C"` was not wrapped by `#ifdef __cplusplus`, which caused failures when building C programs with rocSPARSE.
* Fixed a memory access fault in the `rocsparse_Xbsrilu0` routines.
* Fixed failures that could occur in `rocsparse_Xbsrsm_solve` or `rocsparse_spsm` with BSR format when using host pointer mode.
* Fixed ASAN compilation failures
* Fixed failure that occurred when using const descriptor `rocsparse_create_const_csr_descr` with the generic routine `rocsparse_sparse_to_sparse`. Issue was not observed when using non-const descriptor `rocsparse_create_csr_descr` with `rocsparse_sparse_to_sparse`.
* Fixed a memory leak in the rocsparse handle

### Removed

* The deprecated `rocsparse_spmv_ex` routine
* The deprecated `rocsparse_sbsrmv_ex`, `rocsparse_dbsrmv_ex`, `rocsparse_cbsrmv_ex`, and `rocsparse_zbsrmv_ex`  routines
* The deprecated `rocsparse_sbsrmv_ex_analysis`, `rocsparse_dbsrmv_ex_analysis`, `rocsparse_cbsrmv_ex_analysis`, and `rocsparse_zbsrmv_ex_analysis`  routines

### Upcoming changes

* Deprecated the `rocsparse_spmv` routine. Users should use the `rocsparse_v2_spmv` routine going forward.
* Deprecated `rocsparse_spmv_alg_csr_stream` algorithm. Users should use the `rocsparse_spmv_alg_csr_rowsplit` algorithm going forward.
* Deprecated the `rocsparse_itilu0_alg_sync_split_fusion` algorithm. Users should use one of `rocsparse_itilu0_alg_async_inplace`, `rocsparse_itilu0_alg_async_split`, or `rocsparse_itilu0_alg_sync_split` going forward.

## rocSPARSE 3.4.0 for ROCm 6.4.0

### Added

* Added support for `rocsparse_matrix_type_triangular` in `rocsparse_spsv`
* Added test filters `smoke`, `regression`, and `extended` for emulation tests.
* Added `rocsparse_[s|d|c|z]csritilu0_compute_ex` routines for iterative ILU
* Added `rocsparse_[s|d|c|z]csritsv_solve_ex` routines for iterative triangular solve
* Added `GPU_TARGETS` to replace the now deprecated `AMDGPU_TARGETS` in cmake files
* Added BSR format to the SpMM generic routine `rocsparse_spmm`

### Changed

* By default, build rocsparse shared library using `--offload-compress` compiler option which compresses the fat binary. This significantly reduces the shared library binary size.

### Optimized

* Improved the performance of `rocsparse_spmm` when used with row order for `B` and `C` dense matrices and the row split algorithm, `rocsparse_spmm_alg_csr_row_split`.
* Improved the adaptive CSR sparse matrix-vector multiplication algorithm when the sparse matrix has many empty rows at the beginning or at the end of the matrix. This improves the routines `rocsparse_spmv` and `rocsparse_spmv_ex` when the adaptive algorithm `rocsparse_spmv_alg_csr_adaptive` is used.
* Improved stream CSR sparse matrix-vector multiplication algorithm when the sparse matrix size (number of rows) decreases. This improves the routines `rocsparse_spmv` and `rocsparse_spmv_ex` when the stream algorithm `rocsparse_spmv_alg_csr_stream` is used.
* Compared to `rocsparse_[s|d|c|z]csritilu0_compute`, the routines `rocsparse_[s|d|c|z]csritilu0_compute_ex` introduce a number of free iterations. A free iteration is an iteration that does not compute the evaluation of the stopping criteria, if enabled. This allows the user to tune the algorithm for performance improvements.
* Compared to `rocsparse_[s|d|c|z]csritsv_solve`, the routines `rocsparse_[s|d|c|z]csritsv_solve_ex` introduce a number of free iterations. A free iteration is an iteration that does not compute the evaluation of the stopping criteria. This allows the user to tune the algorithm for performance improvements.
* Improved user documentation

### Resolved issues
* Fixed an issue in `rocsparse_spgemm`, `rocsparse_[s|d|c|z]csrgemm`, and `rocsparse_[s|d|c|z]bsrgemm` where incorrect results could be produced when rocSPARSE was built with optimization level `O0`. This was caused by a bug in the hash tables that could allow keys to be inserted twice.
* Fixed an issue in the routine `rocsparse_spgemm` when using `rocsparse_spgemm_stage_symbolic` and `rocsparse_spgemm_stage_numeric`, where the routine would crash when `alpha` and `beta` were passed as host pointers and where `beta != 0`.
* Fixed compilation error resulting from incorrectly using `reinterpret_cast` to cast away a const qualifier in the `rocsparse_complex_num` constructor. See https://github.com/ROCm/rocSPARSE/issues/434 for more information.

### Upcoming changes

* Deprecated `rocsparse_[s|d|c|z]csritilu0_compute` routines. Users should use the newly added `rocsparse_[s|d|c|z]csritilu0_compute_ex` routines going forward.
* Deprecated `rocsparse_[s|d|c|z]csritsv_solve` routines. Users should use the newly added `rocsparse_[s|d|c|z]csritsv_solve_ex` routines going forward.
* Deprecated `AMDGPU_TARGETS` using in cmake files. Users should use `GPU_TARGETS` going forward.

### Known issues

* Under certain conditions, rocSPARSE might fail to compile with ASAN on Ubuntu 22.04.

## rocSPARSE 3.3.0 for ROCm 6.3.0

### Added

* Added the `azurelinux` OS name to correct the gfortran dependency
* Add `rocsparse_create_extract_descr`, `rocsparse_destroy_extract_descr`, `rocsparse_extract_buffer_size`, `rocsparse_extract_nnz`, and `rocsparse_extract` APIs to allow extraction of the upper or lower part of sparse CSR or CSC matrices.
* Support for the gfx1151, gfx1200, and gfx1201 architectures.

### Changed

* Change the default compiler from hipcc to amdclang in install script and cmake files.
* Change address sanitizer build targets so that only gfx908:xnack+, gfx90a:xnack+, gfx940:xnack+, gfx941:xnack+, and gfx942:xnack+ are built when `BUILD_ADDRESS_SANITIZER=ON` is configured.

### Optimized

* Improved user documentation

### Resolved issues

* Fixed the `csrmm` merge path algorithm so that diagonal is clamped to the correct range.
* Fixed a race condition in `bsrgemm` that could on rare occasions cause incorrect results.
* Fixed an issue in `hyb2csr` where the CSR row pointer array was not being properly filled when `n=0`, `coo_nnz=0`, or `ell_nnz=0`.
* Fixed scaling in `rocsparse_Xhybmv` when only performing `y=beta*y`, for example, where `alpha==0` in `y=alpha*Ax+beta*y`.
* Fixed `rocsparse_Xgemmi` failures when the y grid dimension is too large. This occured when n >= 65536.

## rocSPARSE 3.2.0 for ROCm 6.2.0

### Additions

* New Merge-Path algorithm to SpMM, supporting CSR format
* SpSM now supports row order
* rocsparseio I/O functionality has been added to the library
* `rocsparse_set_identity_permutation` has been added

### Changes

* Adjusted rocSPARSE dependencies to related HIP packages
* Binary size has been reduced
* A namespace has been wrapped around internal rocSPARSE functions and kernels
* `rocsparse_csr_set_pointers`, `rocsparse_csc_set_pointers`, and `rocsparse_bsr_set_pointers` do now allow the column indices and values arrays to be nullptr if `nnz` is 0
* gfx803 target has been removed from address sanitizer builds

### Optimizations

* Improved user manual
* Improved contribution guidelines
* SpMV adaptive and LRB algorithms have been further optimized on CSR format
* Improved performance of SpMV adaptive with symmetrically stored matrices on CSR format

### Fixes

* Compilation errors with `BUILD_ROCSPARSE_ILP64=ON` have been resolved

## rocSPARSE 3.1.1 for ROCm 6.1.0

### Additions

* New LRB algorithm to SpMV, supporting CSR format
* rocBLAS as now an optional dependency for SDDMM algorithms
* Additional verbose output for `csrgemm` and `bsrgemm`
* CMake support for documentation

### Optimizations

* Triangular solve with multiple rhs (SpSM, csrsm, ...) now calls SpSV, csrsv, etcetera when nrhs equals 1
* Improved user manual section *Installation and Building for Linux and Windows*

## rocSPARSE 3.0.2 for ROCm 6.0.0

### Additions

* `rocsparse_inverse_permutation`
* Mixed-precisions for SpVV
* Uniform int8 precision for gather and scatter

### Changes

* Added new `rocsparse_spmv` routine
* Added new `rocsparse_xbsrmv` routines
* When using host pointer mode, you must now call `hipStreamSynchronize` following `doti`, `dotci`,
  `spvv`, and `csr2ell`

### Optimizations

* `doti` routine
* Improved spin-looping algorithms
* Improved documentation
* Improved verbose output during argument checking on API function calls

### Deprecations

* `rocsparse_spmv_ex`
* `rocsparse_xbsrmv_ex`

### Removals

* Auto stages from `spmv`, `spmm`, `spgemm`, `spsv`, `spsm`, and `spitsv`
* Formerly deprecated `rocsparse_spmv` routines
* Formerly deprecated `rocsparse_xbsrmv` routines
* Formerly deprecated `rocsparse_spmm_ex` routine

### Fixes

* Bug in `rocsparse-bench` where the SpMV algorithm was not taken into account in CSR format
* BSR and GEBSR routines (`bsrmv`, `bsrsv`, `bsrmm`, `bsrgeam`, `gebsrmv`, `gebsrmm`) didn't always
  show `block_dim==0` as an invalid size
* Passing `nnz = 0` to `doti` or `dotci` wasn't always returning a dot product of 0
* `gpsv` minimum size is now `m >= 3`

## rocSPARSE 2.5.4 for ROCm 5.7.0

### Additions

* More mixed-precisions for SpMV, (`matrix: float`, `vectors: double`, `calculation: double`) and
  (`matrix: rocsparse_float_complex`, `vectors: rocsparse_double_complex`,
  `calculation: rocsparse_double_complex`)
* Support for gfx940, gfx941, and gfx942

### Fixes

* Bug in `csrsm` and `bsrsm`

### Known issues

* In `csritlu0`, the algorithm `rocsparse_itilu0_alg_sync_split_fusion` has some accuracy issues when
  XNACK is enabled (you can use `rocsparse_itilu0_alg_sync_split` as an alternative)

## rocSPARSE 2.5.2 for ROCm 5.6.0

### Fixes

* Memory leak in `csritsv`
* Bug in `csrsm` and `bsrsm`

## rocSPARSE 2.5.1 for ROCm 5.5.0

### Additions

* `bsrgemm` and `spgemm` for BSR format
* `bsrgeam`
* Build support for Navi32
* Experimental hipGraph support for some rocSPARSE routines
* `csritsv`, `spitsv` csr iterative triangular solve
* Mixed-precisions for SpMV
* Batched SpMM for transpose A in COO format with atomic algorithm

### Optimizations

* `csr2bsr`
* `csr2csr_compress`
* `csr2coo`
* `gebsr2csr`
* `csr2gebsr`

### Fixes

* Documentation
* Bug in COO SpMV grid size
* Bug in SpMM grid size when using very large matrices

### Known issues

* In `csritlu0`, the algorithm `rocsparse_itilu0_alg_sync_split_fusion` has some accuracy issues when
  XNACK is enabled (you can use `rocsparse_itilu0_alg_sync_split` as an alternative)

## rocSPARSE 2.4.0 for ROCm 5.4.0

### Additions

* `rocsparse_spmv_ex` routine
* `rocsparse_bsrmv_ex_analysis` and `rocsparse_bsrmv_ex` routines
* `csritilu0` routine
* Build support for Navi31 and Navi 33

### Optimizations

* Segmented algorithm for COO SpMV by performing analysis
* Improved performance when generating random matrices
* `bsr2csr` routine

### Fixes

* Integer overflow bugs
* Bug in `ellmv`

## rocSPARSE 2.3.2 for ROCm 5.3.0

### Additions

* Transpose A for SpMM COO format
* Matrix checker routines for verifying matrix data
* Atomic algorithm for COO SpMV
* `bsrpad` routine

### Fixes

* Bug in `csrilu0` that could cause a deadlock
* Bug where asynchronous `memcpy` would use wrong stream
* Potential size overflows

## rocSPARSE 2.2.0 for ROCm 5.2.0

### Additions

* Batched SpMM for CSR, CSC, and COO formats
* Packages for test and benchmark executables on all supported operating systems using CPack
* Clients file importers and exporters

### Optimizations

* Clients code size reduction
* Clients error handling
* Clients benchmarking for performance tracking

### Changes

* Test adjustments due to round-off errors
* Fixing API call compatibility with rocPRIM

## rocSPARSE 2.1.0 for ROCm 5.1.0

### Additions

* `gtsv_interleaved_batch`
* `gpsv_interleaved_batch`
* `SpGEMM_reuse`
* Allow copying of mat info struct

### Optimizations

* Optimization for SDDMM
* Allow unsorted matrices in `csrgemm` multipass algorithm

## rocSPARSE 2.0.0 for ROCm 5.0.0

### Additions

* `csrmv`, `coomv`, `ellmv`, and `hybmv` for (conjugate) transposed matrices
* `csrmv` for symmetric matrices
* Packages for test and benchmark executables on all supported operating systems using CPack

### Changes

* `spmm_ex` has been deprecated and will be removed in the next major release

### Optimizations

* Optimization for `gtsv`

## rocSPARSE 1.22.2 for ROCm 4.5.0

### Additions

* Triangular solve for multiple right-hand sides using BSR format
* SpMV for BSRX format
* SpMM in CSR format enhanced to work with transposed A
* Matrix coloring for CSR matrices
* Added batched tridiagonal solve (`gtsv_strided_batch`)
* SpMM for BLOCKED ELL format
* Generic routines for SpSV and SpSM
* Beta support for Windows 10
* Additional atomic-based algorithms for SpMM in COO format
* Extended version of SpMM
* Additional algorithm for SpMM in CSR format
* Added (conjugate) transpose support for CsrMV and SpMV (CSR) routines

### Changes

* Packaging has been split into a runtime package (`rocsparse`) and a development package
  (`rocsparse-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.

### Fixes

* Bug with `gemvi` on Navi21
* Bug with adaptive CsrMV

### Optimizations

* Optimization for pivot-based `gtsv`

## rocSPARSE 1.20.2 for ROCm 4.3.0

### Additions

* (batched) Tridiagonal solver with and without pivoting
* Dense matrix sparse vector multiplication (gemvi)
* Support for gfx90a
* Sampled dense-dense matrix multiplication (SDDMM)

### Optimizations

* client matrix download mechanism
* removed boost dependency in clients

## rocSPARSE 1.19.5 for ROCm 4.2.0

### Additions

* SpMM (CSR, COO)
* Code coverage analysis

### Optimizations

* Install script
* Level 2/3 unit tests
* `rocsparse-bench` no longer depends on boost

## rocSPARSE 1.19.4 for ROCm 4.1.0

### Additions

* `gebsrmm`
* `gebsrmv`
* `gebsrsv`
* `coo2dense` and `dense2coo`
* Generic APIs, including `axpby`, `gather`, `scatter`, `rot`, `spvv`, `spmv`, `spgemm`, `sparsetodense`, `densetosparse`
* Support for mixed indexing types in matrix formats

## rocSPARSE 1.18.4 for ROCm 4.0.0

### Additions

* Changelog
* `csr2gebsr`
* `gebsr2gebsc`
* `gebsr2gebsr`
* Treating filename as regular expression for YAML-based testing generation
* Documentation for `gebsr2csr`

### Optimizations

* `bsric0`

### Changes

* gfx1030 has been adjusted to the latest compiler
* Replace old XNACK 'off' compiler flag with new version
* Updated Debian package name

## rocSPARSE 1.17.6 for ROCm 3.9

### Additions

* `prune_csr2csr`, `prune_dense2csr_percentage` and `prune_csr2csr_percentage` added
* `bsrilu0 added`
* `csrilu0_numeric_boost` functionality added

## rocSPARSE 1.16.1 for ROCm 3.8

### Additions

* `bsric0`

## rocSPARSE 1.14.3 for ROCm 3.7

* No changes for this ROCm release

## rocSPARSE 1.14.3 for ROCm 3.6

### Additions

* Fortran bindings
* CentOS 6 support

### Optimizations

* `bsrmv`

## rocSPARSE 1.12.10 for ROCm 3.5

### Additions

* Default compiler switched to HIP-Clang
* `csr2dense`, `csc2dense`, `csr2csr_compress`, `nnz_compress`, `bsr2csr`, `csr2bsr`, `bsrmv`, and
  `csrgeam`
* Triangular solve for BSR format (`bsrsv`)
* Options for static build
* Examples

### Optimizations

* `dense2csr` and `dense2csc`
* Installation process
