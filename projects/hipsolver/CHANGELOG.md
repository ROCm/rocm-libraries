# Change Log for hipSOLVER

Full documentation for hipSOLVER is available at the [hipSOLVER Documentation](https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/index.html).

## (Unreleased) hipSOLVER

### Added
### Changed
### Removed
### Optimized
### Resolved issues

* Corrected the value of `lwork` returned by various `bufferSize` functions to be consistent with NVIDIA cuSOLVER. The following functions will
  now return `lwork` such that the workspace size (in bytes) is `sizeof(T) * lwork`, rather than `lwork`:
  * hipsolverXorgbr_bufferSize, hipsolverXorgqr_bufferSize, hipsolverXorgtr_bufferSize, hipsolverXormqr_bufferSize, hipsolverXormtr_bufferSize,
    hipsolverXgesvd_bufferSize, hipsolverXgesvdj_bufferSize, hipsolverXgesvdBatched_bufferSize, hipsolverXgesvdaStridedBatched_bufferSize,
    hipsolverXsyevd_bufferSize, hipsolverXsyevdx_bufferSize, hipsolverXsyevj_bufferSize, hipsolverXsyevjBatched_bufferSize,
    hipsolverXsygvd_bufferSize, hipsolverXsygvdx_bufferSize, hipsolverXsygvj_bufferSize, hipsolverXsytrd_bufferSize, hipsolverXsytrf_bufferSize

### Known issues
### Upcoming changes


## hipSOLVER 2.5.0 for ROCm 6.5.0

### Upcoming changes

* With the rocSOLVER backend, the bufferSize methods are currently outputting lwork such that the required workspace 
size (in bytes) is lwork. In ROCm 7.0 this will change to make the rocSOLVER backend consistent with cuSOLVER. The 
changed bufferSize methods will then return lwork such that the required workspace size (in bytes) is sizeof(T) * lwork,
where T is the used precision. This change will break ABI backward compatibility.


## hipSOLVER 2.4.0 for ROCm 6.4.0

### Added
* Added compatibility-only functions
  * csrlsvqr
    * hipsolverSpScsrlsvqr, hipsolverSpDcsrlsvqr, hipsolverSpCcsrlsvqr, hipsolverSpZcsrlsvqr


## hipSOLVER 2.3.0 for ROCm 6.3.0

### Added

* Added functions:
  * auxiliary
    * hipsolverSetDeterministicMode, hipsolverGetDeterministicMode
* Added compatibility-only functions:
  * potrf
    * hipsolverDnXpotrf_bufferSize
    * hipsolverDnXpotrf
  * potrs
    * hipsolverDnXpotrs
  * geqrf
    * hipsolverDnXgeqrf_bufferSize
    * hipsolverDnXgeqrf

### Changed

* Binaries in debug builds no longer have a "-d" suffix.
* Changed rocSPARSE and SuiteSparse to be run-time dependencies by default. The `BUILD_WITH_SPARSE` CMake option can still be used
  to convert them into build-time dependencies (now off by default).
* The `--no-sparse` option of the install script now only affects the hipSOLVER clients and their dependency on hipSPARSE. Use the
  `BUILD_HIPSPARSE_TESTS` CMake option to enable tests for the hipsolverSp API (on by default).

### Upcoming changes

* The Fortran bindings provided in `hipsolver_module.f90` have been deprecated.
  The Fortran bindings provided by the hipfort project are recommended instead.


## hipSOLVER 2.2.0 for ROCm 6.2.0

### Added

* Added functions
  * syevdx/heevdx
    * hipsolverSsyevdx_bufferSize, hipsolverDsyevdx_bufferSize, hipsolverCheevdx_bufferSize, hipsolverZheevdx_bufferSize
    * hipsolverSsyevdx, hipsolverDsyevdx, hipsolverCheevdx, hipsolverZheevdx
  * sygvdx/hegvdx
    * hipsolverSsygvdx_bufferSize, hipsolverDsygvdx_bufferSize, hipsolverChegvdx_bufferSize, hipsolverZhegvdx_bufferSize
    * hipsolverSsygvdx, hipsolverDsygvdx, hipsolverChegvdx, hipsolverZhegvdx
* Added compatibility-only functions
  * auxiliary
    * hipsolverDnCreateParams, hipsolverDnDestroyParams, hipsolverDnSetAdvOptions
  * getrf
    * hipsolverDnXgetrf_bufferSize
    * hipsolverDnXgetrf
  * getrs
    * hipsolverDnXgetrs
* Added support for building on Ubuntu 24.04 and CBL-Mariner.
* Added hip::host to roc::hipsolver usage requirements.

### Changed

* The numerical factorization in csrlsvchol will now be performed on the GPU. (The symbolic factorization is still performed on the CPU.)
* Renamed hipsolver-compat.h to hipsolver-dense.h.

### Removed

* Removed dependency on cblas from the hipsolver test and benchmark clients.


## hipSOLVER 2.1.1 for ROCm 6.1.1

### Changed

* `BUILD_WITH_SPARSE` now defaults to OFF on Windows.

### Resolved issues

* Fixed benchmark client build when `BUILD_WITH_SPARSE` is OFF.


## hipSOLVER 2.1.0 for ROCm 6.1.0

### Added

* Added compatibility API with hipsolverSp prefix
* Added compatibility-only functions
  * csrlsvchol
    * hipsolverSpScsrlsvcholHost, hipsolverSpDcsrlsvcholHost
    * hipsolverSpScsrlsvchol, hipsolverSpDcsrlsvchol
* Added rocSPARSE and SuiteSparse as optional dependencies to hipSOLVER (rocSOLVER backend only). Use the `BUILD_WITH_SPARSE` CMake option to enable
  functionality for the hipsolverSp API (on by default).
* Added hipSPARSE as an optional dependency to hipsolver-test. Use the `BUILD_WITH_SPARSE` CMake option to enable tests of the hipsolverSp API (on by default).

### Changed

* Relax array length requirements for GESVDA.

### Resolved issues

* Fixed incorrect singular vectors returned from GESVDA.


## hipSOLVER 2.0.0 for ROCm 6.0.0
### Added
- Added hipBLAS as an optional dependency to hipsolver-test. Use the `BUILD_HIPBLAS_TESTS` CMake option to test compatibility between hipSOLVER and hipBLAS.

### Changed
- Types hipsolverOperation_t, hipsolverFillMode_t, and hipsolverSideMode_t are now aliases of hipblasOperation_t, hipblasFillMode_t, and hipblasSideMode_t.

### Fixed
- Fixed tests for hipsolver info updates in ORGBR/UNGBR, ORGQR/UNGQR,
  ORGTR/UNGTR, ORMQR/UNMQR, and ORMTR/UNMTR.


## hipSOLVER 1.8.2 for ROCm 5.7.1
### Fixed
- Fixed conflicts between the hipsolver-dev and -asan packages by excluding
  hipsolver_module.f90 from the latter


## hipSOLVER 1.8.1 for ROCm 5.7.0
### Changed
- Changed hipsolver-test sparse input data search paths to be relative to the test executable


## hipSOLVER 1.8.0 for ROCm 5.6.0
### Added
- Added compatibility API with hipsolverRf prefix


## hipSOLVER 1.7.0 for ROCm 5.5.0
### Added
- Added functions
  - gesvdj
    - hipsolverSgesvdj_bufferSize, hipsolverDgesvdj_bufferSize, hipsolverCgesvdj_bufferSize, hipsolverZgesvdj_bufferSize
    - hipsolverSgesvdj, hipsolverDgesvdj, hipsolverCgesvdj, hipsolverZgesvdj
  - gesvdjBatched
    - hipsolverSgesvdjBatched_bufferSize, hipsolverDgesvdjBatched_bufferSize, hipsolverCgesvdjBatched_bufferSize, hipsolverZgesvdjBatched_bufferSize
    - hipsolverSgesvdjBatched, hipsolverDgesvdjBatched, hipsolverCgesvdjBatched, hipsolverZgesvdjBatched


## hipSOLVER 1.6.0 for ROCm 5.4.0
### Added
- Added compatibility-only functions
  - gesvdaStridedBatched
    - hipsolverDnSgesvdaStridedBatched_bufferSize, hipsolverDnDgesvdaStridedBatched_bufferSize, hipsolverDnCgesvdaStridedBatched_bufferSize, hipsolverDnZgesvdaStridedBatched_bufferSize
    - hipsolverDnSgesvdaStridedBatched, hipsolverDnDgesvdaStridedBatched, hipsolverDnCgesvdaStridedBatched, hipsolverDnZgesvdaStridedBatched


## hipSOLVER 1.5.0 for ROCm 5.3.0
### Added
- Added functions
  - syevj
    - hipsolverSsyevj_bufferSize, hipsolverDsyevj_bufferSize, hipsolverCheevj_bufferSize, hipsolverZheevj_bufferSize
    - hipsolverSsyevj, hipsolverDsyevj, hipsolverCheevj, hipsolverZheevj
  - syevjBatched
    - hipsolverSsyevjBatched_bufferSize, hipsolverDsyevjBatched_bufferSize, hipsolverCheevjBatched_bufferSize, hipsolverZheevjBatched_bufferSize
    - hipsolverSsyevjBatched, hipsolverDsyevjBatched, hipsolverCheevjBatched, hipsolverZheevjBatched
  - sygvj
    - hipsolverSsygvj_bufferSize, hipsolverDsygvj_bufferSize, hipsolverChegvj_bufferSize, hipsolverZhegvj_bufferSize
    - hipsolverSsygvj, hipsolverDsygvj, hipsolverChegvj, hipsolverZhegvj
- Added compatibility-only functions
  - syevdx/heevdx
    - hipsolverDnSsyevdx_bufferSize, hipsolverDnDsyevdx_bufferSize, hipsolverDnCheevdx_bufferSize, hipsolverDnZheevdx_bufferSize
    - hipsolverDnSsyevdx, hipsolverDnDsyevdx, hipsolverDnCheevdx, hipsolverDnZheevdx
  - sygvdx/hegvdx
    - hipsolverDnSsygvdx_bufferSize, hipsolverDnDsygvdx_bufferSize, hipsolverDnChegvdx_bufferSize, hipsolverDnZhegvdx_bufferSize
    - hipsolverDnSsygvdx, hipsolverDnDsygvdx, hipsolverDnChegvdx, hipsolverDnZhegvdx
- Added --mem_query option to hipsolver-bench, which will print the amount of device memory workspace required by the function.

### Changed
- The rocSOLVER backend will now set `info` to zero if rocSOLVER does not reference `info`. (Applies to orgbr/ungbr, orgqr/ungqr, orgtr/ungtr, ormqr/unmqr, ormtr/unmtr, gebrd, geqrf, getrs, potrs, and sytrd/hetrd).
- gesvdj will no longer require extra workspace to transpose `V` when `jobz` is `HIPSOLVER_EIG_MODE_VECTOR` and `econ` is 1.

### Fixed
- Fixed Fortran return value declarations within hipsolver_module.f90
- Fixed gesvdj_bufferSize returning `HIPSOLVER_STATUS_INVALID_VALUE` when `jobz` is `HIPSOLVER_EIG_MODE_NOVECTOR` and 1 <= `ldv` < `n`
- Fixed gesvdj returning `HIPSOLVER_STATUS_INVALID_VALUE` when `jobz` is `HIPSOLVER_EIG_MODE_VECTOR`, `econ` is 1, and `m` < `n`


## hipSOLVER 1.4.0 for ROCm 5.2.0
### Added
- Package generation for test and benchmark executables on all supported OSes using CPack.
- File/Folder Reorg
  - Added File/Folder Reorg Changes with backward compatibility support using ROCM-CMAKE wrapper functions.

### Fixed
- Fixed the ReadTheDocs documentation generation.


## hipSOLVER 1.3.0 for ROCm 5.1.0
### Added
- Added functions
  - gels
    - hipsolverSSgels_bufferSize, hipsolverDDgels_bufferSize, hipsolverCCgels_bufferSize, hipsolverZZgels_bufferSize
    - hipsolverSSgels, hipsolverDDgels, hipsolverCCgels, hipsolverZZgels
- Added library version and device information to hipsolver-test output.
- Added compatibility API with hipsolverDn prefix.
- Added compatibility-only functions
  - gesvdj
    - hipsolverDnSgesvdj_bufferSize, hipsolverDnDgesvdj_bufferSize, hipsolverDnCgesvdj_bufferSize, hipsolverDnZgesvdj_bufferSize
    - hipsolverDnSgesvdj, hipsolverDnDgesvdj, hipsolverDnCgesvdj, hipsolverDnZgesvdj
  - gesvdjBatched
    - hipsolverDnSgesvdjBatched_bufferSize, hipsolverDnDgesvdjBatched_bufferSize, hipsolverDnCgesvdjBatched_bufferSize, hipsolverDnZgesvdjBatched_bufferSize
    - hipsolverDnSgesvdjBatched, hipsolverDnDgesvdjBatched, hipsolverDnCgesvdjBatched, hipsolverDnZgesvdjBatched
  - syevj
    - hipsolverDnSsyevj_bufferSize, hipsolverDnDsyevj_bufferSize, hipsolverDnCheevj_bufferSize, hipsolverDnZheevj_bufferSize
    - hipsolverDnSsyevj, hipsolverDnDsyevj, hipsolverDnCheevj, hipsolverDnZheevj
  - syevjBatched
    - hipsolverDnSsyevjBatched_bufferSize, hipsolverDnDsyevjBatched_bufferSize, hipsolverDnCheevjBatched_bufferSize, hipsolverDnZheevjBatched_bufferSize
    - hipsolverDnSsyevjBatched, hipsolverDnDsyevjBatched, hipsolverDnCheevjBatched, hipsolverDnZheevjBatched
  - sygvj
    - hipsolverDnSsygvj_bufferSize, hipsolverDnDsygvj_bufferSize, hipsolverDnChegvj_bufferSize, hipsolverDnZhegvj_bufferSize
    - hipsolverDnSsygvj, hipsolverDnDsygvj, hipsolverDnChegvj, hipsolverDnZhegvj

### Changed
- The rocSOLVER backend now allows hipsolverXXgels and hipsolverXXgesv to be called in-place when B == X.
- The rocSOLVER backend now allows rwork to be passed as a null pointer to hipsolverXgesvd.

### Fixed
- bufferSize functions will now return HIPSOLVER_STATUS_NOT_INITIALIZED instead of HIPSOLVER_STATUS_INVALID_VALUE when both handle and lwork are null.
- Fixed rare memory allocation failure in syevd/heevd and sygvd/hegvd caused by improper workspace array allocation outside of rocSOLVER.


## hipSOLVER 1.2.0 for ROCm 5.0.0
### Added
- Added functions
  - sytrf
    - hipsolverSsytrf_bufferSize, hipsolverDsytrf_bufferSize, hipsolverCsytrf_bufferSize, hipsolverZsytrf_bufferSize
    - hipsolverSsytrf, hipsolverDsytrf, hipsolverCsytrf, hipsolverZsytrf

### Fixed
- Fixed use of incorrect `HIP_PATH` when building from source (#40).
  Thanks [@jakub329homola](https://github.com/jakub329homola)!


## hipSOLVER 1.1.0 for ROCm 4.5.0
### Added
- Added functions
  - gesv
    - hipsolverSSgesv_bufferSize, hipsolverDDgesv_bufferSize, hipsolverCCgesv_bufferSize, hipsolverZZgesv_bufferSize
    - hipsolverSSgesv, hipsolverDDgesv, hipsolverCCgesv, hipsolverZZgesv
  - potrs
    - hipsolverSpotrs_bufferSize, hipsolverDpotrs_bufferSize, hipsolverCpotrs_bufferSize, hipsolverZpotrs_bufferSize
    - hipsolverSpotrs, hipsolverDpotrs, hipsolverCpotrs, hipsolverZpotrs
  - potrsBatched
    - hipsolverSpotrsBatched_bufferSize, hipsolverDpotrsBatched_bufferSize, hipsolverCpotrsBatched_bufferSize, hipsolverZpotrsBatched_bufferSize
    - hipsolverSpotrsBatched, hipsolverDpotrsBatched, hipsolverCpotrsBatched, hipsolverZpotrsBatched
  - potri
    - hipsolverSpotri_bufferSize, hipsolverDpotri_bufferSize, hipsolverCpotri_bufferSize, hipsolverZpotri_bufferSize
    - hipsolverSpotri, hipsolverDpotri, hipsolverCpotri, hipsolverZpotri
  - orgbr/ungbr
    - hipsolverSorgbr_bufferSize, hipsolverDorgbr_bufferSize, hipsolverCungbr_bufferSize, hipsolverZungbr_bufferSize
    - hipsolverSorgbr, hipsolverDorgbr, hipsolverCungbr, hipsolverZungbr
  - orgqr/ungqr
    - hipsolverSorgqr_bufferSize, hipsolverDorgqr_bufferSize, hipsolverCungqr_bufferSize, hipsolverZungqr_bufferSize
    - hipsolverSorgqr, hipsolverDorgqr, hipsolverCungqr, hipsolverZungqr
  - orgtr/ungtr
    - hipsolverSorgtr_bufferSize, hipsolverDorgtr_bufferSize, hipsolverCungtr_bufferSize, hipsolverZungtr_bufferSize
    - hipsolverSorgtr, hipsolverDorgtr, hipsolverCungtr, hipsolverZungtr
  - ormqr/unmqr
    - hipsolverSormqr_bufferSize, hipsolverDormqr_bufferSize, hipsolverCunmqr_bufferSize, hipsolverZunmqr_bufferSize
    - hipsolverSormqr, hipsolverDormqr, hipsolverCunmqr, hipsolverZunmqr
  - ormtr/unmtr
    - hipsolverSormtr_bufferSize, hipsolverDormtr_bufferSize, hipsolverCunmtr_bufferSize, hipsolverZunmtr_bufferSize
    - hipsolverSormtr, hipsolverDormtr, hipsolverCunmtr, hipsolverZunmtr
  - gebrd
    - hipsolverSgebrd_bufferSize, hipsolverDgebrd_bufferSize, hipsolverCgebrd_bufferSize, hipsolverZgebrd_bufferSize
    - hipsolverSgebrd, hipsolverDgebrd, hipsolverCgebrd, hipsolverZgebrd
  - geqrf
    - hipsolverSgeqrf_bufferSize, hipsolverDgeqrf_bufferSize, hipsolverCgeqrf_bufferSize, hipsolverZgeqrf_bufferSize
    - hipsolverSgeqrf, hipsolverDgeqrf, hipsolverCgeqrf, hipsolverZgeqrf
  - gesvd
    - hipsolverSgesvd_bufferSize, hipsolverDgesvd_bufferSize, hipsolverCgesvd_bufferSize, hipsolverZgesvd_bufferSize
    - hipsolverSgesvd, hipsolverDgesvd, hipsolverCgesvd, hipsolverZgesvd
  - getrs
    - hipsolverSgetrs_bufferSize, hipsolverDgetrs_bufferSize, hipsolverCgetrs_bufferSize, hipsolverZgetrs_bufferSize
    - hipsolverSgetrs, hipsolverDgetrs, hipsolverCgetrs, hipsolverZgetrs
  - potrf
    - hipsolverSpotrf_bufferSize, hipsolverDpotrf_bufferSize, hipsolverCpotrf_bufferSize, hipsolverZpotrf_bufferSize
    - hipsolverSpotrf, hipsolverDpotrf, hipsolverCpotrf, hipsolverZpotrf
  - potrfBatched
    - hipsolverSpotrfBatched_bufferSize, hipsolverDpotrfBatched_bufferSize, hipsolverCpotrfBatched_bufferSize, hipsolverZpotrfBatched_bufferSize
    - hipsolverSpotrfBatched, hipsolverDpotrfBatched, hipsolverCpotrfBatched, hipsolverZpotrfBatched
  - syevd/heevd
    - hipsolverSsyevd_bufferSize, hipsolverDsyevd_bufferSize, hipsolverCheevd_bufferSize, hipsolverZheevd_bufferSize
    - hipsolverSsyevd, hipsolverDsyevd, hipsolverCheevd, hipsolverZheevd
  - sygvd/hegvd
    - hipsolverSsygvd_bufferSize, hipsolverDsygvd_bufferSize, hipsolverChegvd_bufferSize, hipsolverZhegvd_bufferSize
    - hipsolverSsygvd, hipsolverDsygvd, hipsolverChegvd, hipsolverZhegvd
  - sytrd/hetrd
    - hipsolverSsytrd_bufferSize, hipsolverDsytrd_bufferSize, hipsolverChetrd_bufferSize, hipsolverZhetrd_bufferSize
    - hipsolverSsytrd, hipsolverDsytrd, hipsolverChetrd, hipsolverZhetrd
  - getrf
    - hipsolverSgetrf_bufferSize, hipsolverDgetrf_bufferSize, hipsolverCgetrf_bufferSize, hipsolverZgetrf_bufferSize
    - hipsolverSgetrf, hipsolverDgetrf, hipsolverCgetrf, hipsolverZgetrf
  - auxiliary
    - hipsolverCreate, hipsolverDestroy
    - hipsolverSetStream, hipsolverGetStream

### Changed
- hipSOLVER functions will now return HIPSOLVER_STATUS_INVALID_ENUM or HIPSOLVER_STATUS_UNKNOWN status codes rather than throw exceptions.
- hipsolverXgetrf functions now take lwork as an argument.

### Removed
- Removed unused HIPSOLVER_FILL_MODE_FULL enum value.
- Removed hipsolverComplex and hipsolverDoubleComplex from the library. Use hipFloatComplex and hipDoubleComplex instead.
