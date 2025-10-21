# Changelog for rocFFT

Documentation for rocFFT is available at
[https://rocm.docs.amd.com/projects/rocFFT/en/latest/](https://rocm.docs.amd.com/projects/rocFFT/en/latest/).

## rocFFT 1.0.36 (unreleased)

### Optimized

* Removed a potential unnecessary global transpose operation from MPI 3D multi-GPU pencil decompositions.
* Enabled optimization of 3D pencil decompositions for single-process multi-GPU transforms.

### Resolved issues

* Fixed potential division by zero when constructing plans using dimensions of length 1.

## rocFFT 1.0.35 for ROCM 7.1.0

### Optimized

* Implemented single-kernel plans for some 2D problem sizes, on devices with at least 160KiB of LDS.
* Improved performance of unit-strided, complex-interleaved, forward/inverse FFTs for lengths:
  - (64,64,128)
  - (64,64,52)
  - (60,60,60)
  - (32,32,128)
  - (32,32,64)
  - (64,32,128)
* Improved performance of 3D MPI pencil decompositions by using sub-communicators for global transpose operations.

## rocFFT 1.0.34 for ROCm 7.0.0

### Added

* Added gfx950 support.

### Removed

* Removed rocfft-rider legacy compatibility from clients
* Removed support for the gfx940 and gfx941 targets from the client programs.

### Optimized

* Removed unnecessary HIP event/stream allocation and synchronization during MPI transforms.
* Implemented single-precision 1D kernels for lengths:
  - 4704
  - 5488
  - 6144
  - 6561
  - 8192
* Implemented single-kernel plans for some large 1D problem sizes, on devices with at least 160KiB of LDS.

### Resolved issues

* Fixed kernel faults on multi-device transforms that gather to a single device, when the input/output bricks are not 
  contiguous.

## rocFFT 1.0.32 for ROCm 6.4.0

### Changed

* Building with the address sanitizer option sets xnack+ on relevant GPU
  architectures and adds address-sanitizer support to runtime-compiled
  kernels.
* The `AMDGPU_TARGETS` build variable should be replaced with `GPU_TARGETS`. `AMDGPU_TARGETS` is deprecated.

### Removed

* Removed ahead-of-time compiled kernels for the gfx906, gfx940, and gfx941 architectures. These architectures still
  function the same, but kernels for them are now compiled at runtime.
* Removed consumer GPU architectures from the precompiled kernel cache that ships with
  rocFFT. rocFFT continues to ship with a cache of precompiled RTC kernels for data-center
  and workstation architectures. As before, user-level caches can be enabled by setting the
  environment variable ROCFFT_RTC_CACHE_PATH to a writeable file location.

### Optimized

* Improved MPI transform performance by using all-to-all communication for global transpose operations.  
  Point-to-point communications are still used when all-to-all is not possible.
* Improved the performance of unit-strided, complex interleaved, forward and inverse, length (64,64,64) FFTs.

### Resolved issues

* Fixed incorrect results from 2-kernel 3D FFT plans that used non-default output strides. For more information, see the [rocFFT GitHub issue](https://github.com/ROCm/rocFFT/issues/507).
* Plan descriptions can be reused with different strides for different plans. For more information, see the [rocFFT GitHub issue](https://github.com/ROCm/rocFFT/issues/504).
* Fixed client packages to depend on hipRAND instead of rocRAND.
* Fixed potential integer overflows during large MPI transforms.

## rocFFT 1.0.31 for ROCm 6.3.0

### Added

* rocfft-test now includes a --smoketest option.
* Support for the gfx1151, gfx1200, and gfx1201 architectures.
* Implemented experimental APIs to allow computing FFTs on data
  distributed across multiple MPI ranks. These APIs can be enabled with the
  `ROCFFT_MPI_ENABLE` CMake option. This option defaults to `OFF`.

  When `ROCFFT_MPI_ENABLE` is set to `ON`:

  * `rocfft_plan_description_set_comm` can be called to provide an
    MPI communicator to a plan description, which can then be passed
    to `rocfft_plan_create`. Each rank calls
    `rocfft_field_add_brick` to specify the layout of data bricks on
    that rank.

  * An MPI library with ROCm acceleration enabled is required at
    build time and at runtime.

### Changed

* Compilation uses amdclang++ instead of hipcc.
* CLI11 replaces Boost Program Options as the command line parser for clients and samples.

## rocFFT 1.0.30 for ROCm 6.2.4

### Optimizations

* Implemented 1D kernels for factorizable sizes > 1024 and < 2048.

### Fixes

* Fixed plan creation failure on some even-length real-complex transforms that use Bluestein's algorithm.

### Additions

* GFX1151 Support

## rocFFT 1.0.29 for ROCm 6.2.1

### Optimizations

* Implemented 1D kernels for factorizable sizes < 1024

## rocFFT 1.0.28 for ROCm 6.2.0

### Optimizations

* Implemented multi-device transform for 3D pencil decomposition.  Contiguous dimensions on input and output bricks
  are transformed locally, with global transposes to make remaining dimensions contiguous.

### Changes

* Add option in dyna-bench to load the libs in forward and then reverse order for benchmark tests.
* Randomly generated accuracy tests are now disabled by default; these can be enabled using
  the --nrand <int> option (which defaults to 0).
* Use Bonferroni multi-hypothesis testing framework by default for benchmark tests.

## rocFFT 1.0.27 for ROCm 6.1.1

### Fixes

* Fixed kernel launch failure on execute of very large odd-length real-complex transforms.

### Additions

* Enable multi-gpu testing on systems without direct GPU-interconnects

## rocFFT 1.0.26 for ROCm 6.1.0

### Changes

* Multi-device FFTs now allow batch greater than 1
* Multi-device, real-complex FFTs are now supported
* rocFFT now statically links libstdc++ when only `std::experimental::filesystem` is available (to guard
  against ABI incompatibilities with newer libstdc++ libraries that include `std::filesystem`)

## rocFFT 1.0.25 for ROCm 6.0.0

### Additions

* Implemented experimental APIs to allow computing FFTs on data distributed across multiple devices
  in a single process

  * `rocfft_field` is a new type that can be added to a plan description to describe the layout of FFT
    input or output
  * `rocfft_field_add_brick` can be called to describe the brick decomposition of an FFT field, where each
    brick can be assigned a different device

  These interfaces are still experimental and subject to change. We are interested in getting feedback.
  You can raise questions and concerns by opening issues in the
  [rocFFT issue tracker](https://github.com/ROCmSoftwarePlatform/rocFFT/issues).

  Note that multi-device FFTs currently have several limitations (we plan to address these in future
  releases):

  * Real-complex (forward or inverse) FFTs are not supported
  * Planar format fields are not supported
  * Batch (the `number_of_transforms` provided to `rocfft_plan_create`) must be 1
  * FFT input is gathered to the current device at run time, so all FFT data must fit on that device

### Optimizations

* Improved the performance of several 2D/3D real FFTs supported by `2D_SINGLE` kernel. Offline
  tuning provides more optimization for fx90a
* Removed an extra kernel launch from even-length, real-complex FFTs that use callbacks

### Changes

* Built kernels in a solution map to the library kernel cache
* Real forward transforms (real-to-complex) no longer overwrite input; rocFFT may still overwrite real
  inverse (complex-to-real) input, as this allows for faster performance

* `rocfft-rider` and `dyna-rocfft-rider` have been renamed to `rocfft-bench` and `dyna-rocfft-bench`;
  these are controlled by the `BUILD_CLIENTS_BENCH` CMake option
  * Links for the former file names are installed, and the former `BUILD_CLIENTS_RIDER` CMake option
    is accepted for compatibility, but both will be removed in a future release
* Binaries in debug builds no longer have a `-d` suffix

### Fixes

* rocFFT now correctly handles load callbacks that convert data from a smaller data type (e.g., 16-bit
  integers -> 32-bit float)

## rocFFT 1.0.24 for ROCm 5.7.0

### Optimizations

* Improved the performance of complex forward/inverse 1D FFTs (2049 <= length <= 131071) that use
  Bluestein's algorithm

### Additions

* Implemented a solution map version converter and finished the first conversion from ver.0 to ver.1
  * Version 1 removes some incorrect kernels (sbrc/sbcr using `half_lds`)

### Changes

* Moved `rocfft_rtc_helper` executable to the `lib/rocFFT` directory on Linux
* Moved library kernel cache to the `lib/rocFFT` directory

## rocFFT 1.0.23 for ROCm 5.6.0

### Additions

* Implemented half-precision transforms; these can be requested by passing `rocfft_precision_half` to
  `rocfft_plan_create`
* Implemented a hierarchical solution map that saves information on how to decompose a problem
  and the kernels that are used
* Implemented a first version of offline-tuner to support tuning kernels for C2C and Z2Z problems

### Changes

* Replaced `std::complex` with hipComplex data types for the data generator
* FFT plan dimensions are now sorted to be row-major internally where possible, which produces
  better plans if the dimensions were accidentally specified in a different order (column-major, for
  example)
* Added the `--precision` argument to benchmark and test clients (`--double` is still accepted but is
  deprecated as a method to request a double-precision transform)
* Improved performance test suite statistical framework

### Fixes

* Fixed over-allocation of LDS in some real-complex kernels, which was resulting in kernel launch
  failure

## rocFFT 1.0.22 for ROCm 5.5.0

### Optimizations

* Improved the performance of 1D lengths < 2048 that use Bluestein's algorithm
* Reduced code generation time during plan creation
* Optimized 3D R2C and C2R lengths 32, 84, 128
* Optimized batched small 1D R2C and C2R cases

### Additions

* Added gfx1101 to default `AMDGPU_TARGETS`

### Changes

* Moved client programs to C++17
* Moved planar kernels and infrequently used Stockham kernels to be runtime-compiled
* Moved transpose, real-complex, Bluestein, and Stockham kernels to the library kernel cache

### Fixes

* Removed zero-length twiddle table allocations, which fixes errors from `hipMallocManaged`
* Fixed incorrect freeing of HIP stream handles during twiddle computation when multiple devices are
  present

## rocFFT 1.0.21 for ROCm 5.4.3

### Fixes

* Removed the source directory from `rocm_install_targets` to prevent the installation of `rocfft.h` in an
  unintended location

## rocFFT 1.0.20 for ROCm 5.4.1

### Fixes

* Fixed incorrect results on strided large 1D FFTs where batch size does not equal the stride

## rocFFT 1.0.19 for ROCm 5.4.0

### Optimizations

* Optimized some strided large 1D plans

### Additions

* Added the `rocfft_plan_description_set_scale_factor` API to efficiently multiply each output element of
  an FFT by a given scaling factor
* Created a `rocfft_kernel_cache.db` file next to the installed library; SBCC, CR, and RC kernels are
  moved to this file when built with the library, and are runtime-compiled for new GPU architectures
* Added gfx1100 and gfx1102 to default `AMDGPU_TARGETS`

### Changes

* Moved the runtime compilation cache to in-memory by default
  * A default on-disk cache can encounter contention problems on multi-node clusters with a shared
    filesystem
  * rocFFT can still use an on-disk cache by setting the `ROCFFT_RTC_CACHE_PATH` environment
    variable

## rocFFT 1.0.18 for ROCm 5.3.0

### Changes

* The runtime compilation cache now looks for environment variables `XDG_CACHE_HOME` (on Linux)
  and `LOCALAPPDATA` (on Windows) before falling back to `HOME`
* Moved computation of the twiddle table from the host to the device

### Optimizations

* Optimized 2D R2C and C2R to use 2-kernel plans where possible
* Improved performance of the Bluestein algorithm
* Optimized sbcc-168 and 100 by using half-LDS
* Optimized length-280 2D and 3D transforms
* Added kernels for factorizable 1D lengths < 128

### Fixes

* Fixed occasional failures to parallelize runtime compilation of kernels (failures would be retried
  serially and ultimately succeed, but this would take extra time)
* Fixed failures of some R2C 3D transforms that use the unsupported `TILE_UNALGNED` SBRC kernels
  (an example is 98^3 R2C out-of-place)
* Fixed bugs in the `SBRC_ERC` type

## rocFFT 1.0.17 for ROCm 5.2.0

### Additions

* Packages for test and benchmark executables on all supported operating systems using CPack
* Added file and folder reorganization changes, with backward compatibility support, using
  `rocm-cmake` wrapper functions

### Changes

* Improved reuse of twiddle memory between plans
* Set a default load/store callback when only one callback type is set via the API (for improved
  performance)
* Updated the GoogleTest dependency to version 1.11

### Optimizations

* Introduced a new access pattern of LDS (non-linear) and applied it on sbcc kernels len 64 and 81 for a
  performance improvement
* Applied `lds-non-linear`, `direct-load-to-register`, and `direct-store-from-register` on sbcr kernels for
  a performance improvement

### Fixes

* Correctness of certain transforms with unusual strides
* Incorrect handling of user-specified stream for runtime-compiled kernels
* Incorrect buffer allocation in `rocfft-test` on in-place transforms with different input and output sizes

## rocFFT 1.0.16 for ROCm 5.1.0

### Changes

* Supported unaligned tile dimension for `SBRC_2D` kernels
* Improved test and benchmark infrastructure by adding RAII
* Enabled runtime compilation of length-2304 FFT kernel during plan creation
* Added tokenizer for test suite
* Reduce twiddle memory requirements for even-length, real-complex transforms
* Clients can now be built separately from the main library

### Optimizations

* Optimized more large 1D cases by using `L1D_CC` plan
* Optimized the 3D 200^3 C2R case
* Optimized the 1D 2^30 double precision on MI200
* Added padding to work buffer sizes to improve performance in many cases

### Fixes

* Fixed the correctness of some R2C transforms with unusual strides

### Removals

* The hipFFT API (header) has been removed; use the
  [hipFFT](https://github.com/ROCmSoftwarePlatform/hipFFT) package or repository to obtain the API

## rocFFT 1.0.15 for ROCm 5.0.0

### Changes

* Enabled runtime compilation of single FFT kernels > length 1024
* Re-aligned the split device library into four roughly equal libraries
* Implemented the FuseShim framework to replace the original OptimizePlan
* Implemented the generic buffer-assignment framework
  * The buffer assignment is no longer performed by each node--we designed a generic algorithm to
    test and pick the best assignment path
  * With the help of FuseShim, we can achieve the most kernel-fusions possible
* Don't read the imaginary part of the DC and Nyquist modes for even-length complex-to-real
  transforms

### Optimizations

* Optimized twiddle conjugation; complex-to-complex inverse transforms should now have similar
  performance to forward transforms
* Improved performance of single-kernel, small 2D transforms

## rocFFT 1.0.14 for ROCm 4.5.0

### Optimizations

* Optimized SBCC kernels of lengths 52, 60, 72, 80, 84, 96, 104, 108, 112, 160, 168, 208, 216, 224, and
  240 with a new kernel generator

### Additions

* Added support for Windows 10 as a build target

### Changes

* Packaging has been split into a runtime package (`rocfft`) and a development package
  (`rocfft-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.

### Fixes

* Fixed validation failures for even-length R2C inplace 2D and 3D cubics sizes, such as 100^2 (or ^3),
  200^2 (or ^3), and 256^2 (or ^3)
  * We combine two kernels (`r2c-transpose`) instead of combining the three kernels
    (`stockham-r2c-transpose`)

### Changes

* Split 2D device code into separate libraries

## rocFFT 1.0.13 for ROCm 4.4.0

### Optimizations

* Improved plans by removing unnecessary transpose steps
* Optimized scheme selection for 3D problems
  * Imposed fewer restrictions on `3D_BLOCK_RC` selection (more problems can use `3D_BLOCK_RC` and
    have performance gains)
  * Enabled `3D_RC`; some 3D problems with SBCC-supported z-dim can use fewer kernels to get
    benefits
  * Forced `--length` 336 336 56 (dp) to use faster `3D_RC` to prevent it from being skipped by a
    conservative threshold test
* Optimized some even-length R2C/C2R cases by doing more in-place operations and combining
  pre- and post-processing into Stockham kernels
* Added radix-17

### Additions

* Added a new kernel generator for select fused 2D transforms

### Fixes

* Improved large 1D transform decompositions

## rocFFT 1.0.12 for ROCm 4.3.0

### Changes

* Re-split device code into single-precision, double-precision, and miscellaneous kernels

### Fixes

* Fixed potential crashes in double-precision planar->planar transpose
* Fixed potential crashes in 3D transforms with unusual strides for SBCC-optimized sizes
* Improved buffer placement logic

### Additions

* Added a new kernel generator for select lengths; new kernels have improved performance
* Added public `rocfft_execution_info_set_load_callback` and`rocfft_execution_info_set_store_callback`
  API functions to allow running extra logic when loading data from and storing data to global
  memory during a transform

### Removals

* Removed R2C pair schemes and kernels

### Optimizations

* Optimized 2D and 3D R2C 100 and 1D Z2Z 2500
* Reduced number of kernels for 2D/3D sizes where higher dimension is 64, 128, 256

### Fixes

* Fixed potential crashes in 3D transforms with unusual strides, for SBCC-optimized sizes

## rocFFT 1.0.11 for ROCm 4.2.0

### Changes

* Move device code into the main library

### Optimizations

* Improved performance for single-precision kernels exercising all except radix-2/7 butterfly ops
* Minor optimization for C2R 3D 100 and 200 cube sizes
* Optimized some C2C and R2C 3D 64, 81, 100, 128, 200, and 256 rectangular sizes
* When factoring, test to see if the remaining length is explicitly supported
* Explicitly added radix-7 lengths 14, 21, and 224 to list of supported lengths
* Optimized R2C 2D and 3D 128, 200, and 256 cube sizes

### Known issues

* Fixed potential crashes in small 3D transforms with unusual strides
  ([issue 311](https://github.com/ROCmSoftwarePlatform/rocFFT/issues/311))
* Fixed potential crashes when running transforms on multiple devices
  ([issue 310](https://github.com/ROCmSoftwarePlatform/rocFFT/issues/310))

## rocFFT 1.0.10 for ROCm 4.1.0

### Additions

* Explicitly specify `MAX_THREADS_PER_BLOCK` through `__launch_bounds_` for all kernels
* Switched to a new syntax for specifying AMD GPU architecture names and features

### Optimizations

* Optimized C2C and R2C 3D 64, 81, 100, 128, 200, and 256 cube sizes
* Improved the performance of the standalone out-of-place transpose kernel
* Optimized the 1D length 40000 C2C case
* Enabled radix-7 for size 336
* New radix-11 and radix-13 kernels; used in length 11 and 13 (and some of their multiples)
  transforms

### Changes

* rocFFT now automatically allocates a work buffer if the plan requires one and none is provided
* An explicit `rocfft_status_invalid_work_buffer` error is now returned when a work buffer of insufficient
  size is provided
* Updated online documentation
* Updated Debian package name version with separated underscore ( _ )
* Adjusted accuracy test tolerances and how they are compared

### Fixes

* Fixed a 4x4x8192 accuracy failure

## rocFFT 1.0.8 for ROCm 3.10.0

### Optimizations

* Optimized the 1D length 10000 C2C case

### Changes

* Added the `BUILD_CLIENTS_ALL` CMake option

### Fixes

* Fixed the correctness of SBCC and SBRC kernels with non-unit strides
* Fixed fused C2R kernel when a Bluestein transform follows it

## rocFFT 1.0.7 for ROCm 3.9.0

### Optimizations

* New R2C and C2R fused kernels to combine pre- and post-processing steps with transpose
* Enabled diagonal transpose for 1D and 2D power-of-2 cases
* New single kernels for small power-of-2, 3, and 5 sizes
* Added more radix-7 kernels

### Changes

* Explicitly disabled XNACK and SRAM-ECC features on AMDGPU hardware

### Fixes

* Fixed 2D C2R transform with length 1 on one dimension
* Fixed a potential thread unsafety in logging

## rocFFT 1.0.6 for ROCm 3.8.0

### Optimizations

* Improved the performance of 1D batch-paired R2C transforms of odd length
* Added some radix-7 kernels
* Improved the performance for 1D length 6561 and 10000
* Improved the performance for certain 2D transform sizes

### Changes

* Allowed a static library build with `BUILD_SHARED_LIBS=OFF` CMake option
* Updated GoogleTest dependency to version 1.10

### Fixes

* Correctness of certain large 2D sizes

## rocFFT 1.0.5 for ROCM 3.7.0

### Optimizations

* Optimized C2C power-of-2 middle sizes

### Changes

* Parallelized work in unit tests and eliminated duplicate cases

### Fixes

* Correctness of certain large 1D, and 2D power-of-3 and 5 sizes
* Incorrect buffer assignment for some even-length R2C transforms
* `<cstddef>` inclusion on C compilers
* Incorrect results on non-unit strides with SBCC/SBRC kernels
