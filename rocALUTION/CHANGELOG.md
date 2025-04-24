# Changelog for rocALUTION

Full documentation forrocALUTION is available at [https://rocm.docs.amd.com/projects/rocALUTION/en/latest/](https://rocm.docs.amd.com/projects/rocALUTION/en/latest/).

## (Unreleased) rocALUTION 3.2.4

## rocALUTION 3.2.4 for ROCm 6.5.0

### Added
* Added support for gfx950.

### Optimized
* Improved the user documentation

### Resolved issues
* Fix for GPU hashing algorithm when not compiling with -O2/O3

## rocALUTION 3.2.3 for ROCm 6.4.1

### Added
* The `-a` option has been added to the `rmake.py` build script. This option provides a way to select specific architectures when building on Windows.

### Resolved issues
* Fixed an issue where the `HIP_PATH` environment variable was being ignored when compiling on Windows.

## rocALUTION 3.2.2 for ROCm 6.4.0

### Changed
* Improved documentation

## rocALUTION 3.2.1 for ROCm 6.3.0

### Added

* Support for gfx1200, gfx1201, and gfx1151.

### Changed

* Changed the default compiler from `hipcc` to `amdclang` in the installation script and cmake files.
* Changed the address sanitizer build targets. Now only `gfx908:xnack+`, `gfx90a:xnack+`, `gfx940:xnack+`, `gfx941:xnack+`, and `gfx942:xnack+` are built with `BUILD_ADDRESS_SANITIZER=ON`.

### Resolved issues

* Fix hang in `RS-AMG` for Navi on some specific matrix sparsity patterns.
* Fix wrong results in `Apply` on multi-GPU setups.

## rocALUTION 3.2.0 for ROCm 6.2.0

### Additions
* New file I/O based on rocsparse I/O format
* `GetConvergenceHistory` for ItILU0 preconditioner

### Deprecations
* `LocalMatrix::ReadFileCSR`
* `LocalMatrix::WriteFileCSR`
* `GlobalMatrix::ReadFileCSR`
* `GlobalMatrix::WriteFileCSR`

## rocALUTION 3.1.1 for ROCm 6.1.0

### Additions

* `TripleMatrixProduct` functionality for `GlobalMatrix`
* Multi-Node/GPU support for `UA-AMG`, `SA-AMG` and `RS-AMG`
* Iterative ILU0 preconditioner `ItILU0`
* Iterative triangular solve, selectable via `SolverDecr` class

### Deprecations

* `LocalMatrix::AMGConnect`
* `LocalMatrix::AMGAggregate`
* `LocalMatrix::AMGPMISAggregate`
* `LocalMatrix::AMGSmoothedAggregation`
* `LocalMatrix::AMGAggregation`
* `PairwiseAMG`

### Known Issues
* `PairwiseAMG` does currently not support matrix sizes that exceed int32 range
* `PairwiseAMG` might fail building the hierarchy on certain input matrices

## rocALUTION 3.0.3 for ROCm 6.0.0

### Additions

* Support for 64bit integer vectors
* Inclusive and exclusive sum functionality for vector classes
* Transpose functionality for `GlobalMatrix` and `LocalMatrix`
* `TripleMatrixProduct` functionality for `LocalMatrix`
* `Sort()` function for `LocalVector` class
* Multiple stream support to the HIP backend

### Optimizations

* `GlobalMatrix::Apply()` now uses multiple streams to better hide communication

### Changes

* Matrix dimensions and number of non-zeros are now stored using 64-bit integers
* Improved the ILUT preconditioner

### Deprecations

* `LocalVector::GetIndexValues(ValueType*)`
* `LocalVector::SetIndexValues(const ValueType*)`
* `LocalMatrix::RSDirectInterpolation(const LocalVector&, const LocalVector&, LocalMatrix*, LocalMatrix*)`
* `LocalMatrix::RSExtPIInterpolation(const LocalVector&, const LocalVector&, bool, float, LocalMatrix*, LocalMatrix*)`
* `LocalMatrix::RugeStueben()`
* `LocalMatrix::AMGSmoothedAggregation(ValueType, const LocalVector&, const LocalVector&, LocalMatrix*, LocalMatrix*, int)`
* `LocalMatrix::AMGAggregation(const LocalVector&, LocalMatrix*, LocalMatrix*)`

### Fixes

* Unit tests no longer ignore BCSR block dimension
* Fixed documentation typos
* Bug in multi-coloring for non-symmetric matrix patterns

## rocALUTION 2.1.11 for ROCm 5.7.0

### Additions

* Support for gfx940, gfx941, and gfx942

### Fixes

* OpenMP runtime issue with Windows toolchain

## rocALUTION 2.1.9 for ROCm 5.6.0

### Fixes

* Synchronization issues in level 1 routines

## rocALUTION 2.1.8 for ROCm 5.5.0

### Additions

* Build support for Navi32

### Fixes

* Typo in MPI backend
* Bug with the backend when HIP support is disabled
* Bug in SAAMG hierarchy building on the HIP backend
* Improved SAAMG hierarchy build performance on the HIP backend

### Deprecations

* `LocalVector::GetIndexValues(ValueType*)`: use
  `LocalVector::GetIndexValues(const LocalVector&, LocalVector*)` instead
* `LocalVector::SetIndexValues(const ValueType*)`: use
  `LocalVector::SetIndexValues(const LocalVector&, const LocalVector&)` instead
* `LocalMatrix::RSDirectInterpolation(const LocalVector&, const LocalVector&, LocalMatrix*, LocalMatrix*)`:
  use `LocalMatrix::RSDirectInterpolation(const LocalVector&, const LocalVector&, LocalMatrix*)`
  instead
* `LocalMatrix::RSExtPIInterpolation(const LocalVector&, const LocalVector&, bool, float, LocalMatrix*, LocalMatrix*)`:
  use `LocalMatrix::RSExtPIInterpolation(const LocalVector&, const LocalVector&, bool, LocalMatrix*)`
  instead
* `LocalMatrix::RugeStueben()`
* `LocalMatrix::AMGSmoothedAggregation(ValueType, const LocalVector&, const LocalVector&, LocalMatrix*, LocalMatrix*, int)`:
  use `LocalMatrix::AMGAggregation(ValueType, const LocalVector&, const LocalVector&, LocalMatrix*, int)`
  instead
* `LocalMatrix::AMGAggregation(const LocalVector&, LocalMatrix*, LocalMatrix*)`: use
  `LocalMatrix::AMGAggregation(const LocalVector&, LocalMatrix*)` instead

## rocALUTION 2.1.3 for ROCm 5.4.0

### Additions

* Build support for Navi31 and Navi33
* Support for non-squared global matrices

### Fixes

* Memory leak in MatrixMult on HIP backend
* Global structures can now be used with a single process

### Changes

* Switched GTest death test style to 'threadsafe'
* Removed the native compiler option that was used during default library compilation

### Deprecations

* `GlobalVector::GetGhostSize()`
* `ParallelManager::GetGlobalSize(), ParallelManager::GetLocalSize()`, `ParallelManager::SetGlobalSize()`,
  and `ParallelManager::SetLocalSize()`
* `Vector::GetGhostSize()`
* `Multigrid::SetOperatorFormat(unsigned int)`: use `Multigrid::SetOperatorFormat(unsigned int, int)`
  instead
* `RugeStuebenAMG::SetCouplingStrength(ValueType)`: use `SetStrengthThreshold(float)` instead

## rocALUTION 2.1.0 for ROCm 5.3.0

### Additions

* Benchmarking tool
* Ext+I Interpolation with sparsify strategies added for RS-AMG

### Optimizations

* ParallelManager

## rocALUTION 2.0.3 for ROCm 5.2.0

### Additions

* New packages for test and benchmark executables on all supported operating systems using CPack

## rocALUTION 2.0.2 for ROCm 5.1.0

### Additions

* Added out-of-place matrix transpose functionality
* Added LocalVector<bool>

## rocALUTION 2.0.1 for ROCm 5.0.0

### Changes

* Changed to C++ 14 Standard
* Added sanitizer option
* Improved documentation

### Deprecations

* `GlobalPairwiseAMG` class: use `PairwiseAMG` instead

## rocALUTION 1.13.2 for ROCm 4.5.0

### Additions

* AddressSanitizer build option
* Enabled beta support for Windows 10

### Changes

* Packaging has been split into a runtime package (`rocalution`) and a development package
  (`rocalution-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.

### Deprecations

* `GlobalPairwiseAMG`: use `PairwiseAMG` instead

### Optimizations

* Improved (A)MG smoothing and convergence rate
* Improved IDR Gram-Schmidt process
* Optimized (A)MG solving phase

## rocALUTION 1.12.1 for ROCm 4.3.0

### Additions

* Support for gfx90a target
* Support for gfx1030 target

### Optimizations

* Install script

## rocALUTION 1.11.5 for ROCm 4.0.0

### Additions

* Changelog
* Block compressed sparse row (BCSRR) format support

### Changes

* Update to the Debian package name
* CMake file adjustments

### Fixes

* NaN issues

## rocALUTION 1.10 for ROCm 3.9

### Additions

* rocRAND support for GPU sampling of random data

## rocALUTION 1.9.3 for ROCm 3.8

### Additions

* `csr2dense` and `dense2csr` to HIP backend

## rocALUTION 1.9.1 for ROCm 3.5

### Additions

* Static build
* BCSR matrix format for SpMV

### Fixes

* Bug in conversion from CSR to HYB format
