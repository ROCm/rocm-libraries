# Changelog for rocThrust

Documentation for rocThrust available at
[https://rocm.docs.amd.com/projects/rocThrust/en/latest/](https://rocm.docs.amd.com/projects/rocThrust/en/latest/).

## (Unreleased) rocThrust 3.x.x for ROCm 6.x

### Changes

* Changed the C++ version from 14 to 17. C++14 will be deprecated in the next major release.

## (Unreleased) rocThrust 3.3.0 for ROCm 6.4

### Added
* Added extended tests to `rtest.py`. These tests are extra tests that did not fit the criteria of smoke and regression tests. These tests will take much longer to run relative to smoke and regression tests. Use `python rtest.py [--emulation|-e|--test|-t]=extended` to run these tests.
* Added regression tests to `rtest.py`. These tests recreate scenarios that have caused hardware problems in past emulation environments. Use `python rtest.py [--emulation|-e|--test|-t]=regression` to run these tests.
* Added smoke test options, which runs a subset of the unit tests and ensures that less than 2gb of VRAM will be used. Use `python rtest.py [--emulation|-e|--test|-t]=smoke` to run these tests.
* Added `--emulation` option for `rtest.py`
* Merged changes from upstream CCCL/thrust 2.4.0
* Merged changes from upstream CCCL/thrust 2.5.0
* Added `find_first_of` to HIPSTDPAR
* Added `search` and `find_end` to HIPSTDPAR
* Added `search_n` to HIPSTDPAR
* Updated HIPSTDPAR's `adjacent_find` to use rocPRIM's implementation

### Changed
* `--test|-t` is no longer a required flag for `rtest.py`. Instead, the user can use either `--emulation|-e` or `--test|-t`, but not both.
* Split the contents of HIPSTDPAR's forwarding header into several implementation headers.
* Fixed `copy_if` to work with large data types (512 bytes)

## (Unreleased) rocThrust 3.2.0 for ROCm 6.3

### Added

* Merged changes from upstream CCCL/thrust 2.3.2
  * Only the NVIDIA backend uses `tuple` and `pair` types from libcu++, other backends continue to
    use the original Thrust implementations and hence do not require libcu++ (CCCL) as a dependency.
* Added the `thrust::hip::par_det` execution policy to enable bitwise reproducibility on algorithms that are not bitwise reproducible by default.
* Fix tests failing when compiling with `-D_GLIBCXX_ASSERTIONS=ON`.

### Changed

* Enabled the upstream (thrust) test suite for execution by default. It can still be disabled by CMake option `-DENABLE_UPSTREAM_TESTS=OFF`.

### Resolved issues

* Fixed the HIP backend not passing `TestCopyIfNonTrivial` from the upstream (thrust) test suite.

## (Unreleased) rocThrust 3.1.0 for ROCm 6.2

### Additions

* Merged changes from upstream CCCL/thrust 2.2.0
  * Updated the contents of `system/hip` and `test` with the upstream changes to `system/cuda` and `testing`
* Added HIPSTDPAR library as part of rocThrust.

### Changes

* Updated internal calls to `rocprim::detail::invoke_result` to use the public API `rocprim::invoke_result`.
* Use `rocprim::device_adjacent_difference` for `adjacent_difference` API call.
* Updated internal use of custom iterator in `thrust::detail::unique_by_key` to use rocPRIM's `rocprim::unique_by_key`.
* Updated `adjecent_difference` to make use of `rocprim:adjecent_difference` when iterators are comparable and not equal otherwise use `rocprim:adjacent_difference_inplace`.

### Fixes
* Fixed incorrect implementation of `thrust::optional<T&>::emplace()`.

### Known issues
* `thrust::reduce_by_key` outputs are not bit-wise reproducible, as run-to-run results for pseudo-associative reduction operators (e.g. floating-point arithmetic operators) are not deterministic on the same device.
* Note that currently, rocThrust memory allocation is performed in such a way that most algorithmic API functions cannot be called from within hipGraphs.

## rocThrust 3.0.0 for ROCm 6.0

### Additions

* Updated to match upstream Thrust 2.0.1
* NV_IF_TARGET macro from libcu++ for NVIDIA backend and HIP implementation for HIP backend.

### Changes

* The cmake build system now additionally accepts `GPU_TARGETS` in addition to `AMDGPU_TARGETS` for
  setting the targeted gpu architectures. `GPU_TARGETS=all` will compile for all supported architectures.
  `AMDGPU_TARGETS` is only provided for backwards compatibility, `GPU_TARGETS` should be preferred.
* Removed cub symlink from the root of the repository.
* Removed support for deprecated macros (THRUST_DEVICE_BACKEND and THRUST_HOST_BACKEND).

### Fixes
* Fixed a segmentation fault when binary search / upper bound / lower bound / equal range was invoked with `hip_rocprim::execute_on_stream_base` policy.

### Known issues

* The `THRUST_HAS_CUDART` macro, which is no longer used in Thrust (it's provided only for legacy
  support) is replaced with `NV_IF_TARGET` and `THRUST_RDC_ENABLED` in the NVIDIA backend. The
  HIP backend doesn't have a `THRUST_RDC_ENABLED` macro, so some branches in Thrust code may
  be unreachable in the HIP backend.

## rocThrust 2.18.0 for ROCm 5.7

### Fixes
* `lower_bound`, `upper_bound`, and `binary_search` failed to compile for certain types.
* Fixed issue where `transform_iterator` would not compile with `__device__`-only operators.

### Changes
* Updated `docs` directory structure to match the standard of [rocm-docs-core](https://github.com/RadeonOpenCompute/rocm-docs-core).
* Removed references to and workarounds for deprecated hcc


## rocThrust 2.17.0 for ROCm 5.5

### Additions

* Updates to match upstream Thrust 1.17.2

### Changes

* `partition_copy` now uses `rocprim::partition_two_way` for increased performance

### Fixes

* `set_difference` and `set_intersection` no longer hang if the number of items is above `UINT_MAX`
  (the unit tests for `set_difference` and `set_intersection` used to fail the
  `TestSetDifferenceWithBigIndexes`)

## rocThrust 2.16.0 for ROCm 5.3

### Additions

* Updates to match upstream Thrust 1.16.0

### Changes

* rocThrust functionality dependent on device malloc is functional (ROCm 5.2 reenabled device malloc); you can now use device launched `thrust::sort` and `thrust::sort_by_key`

## rocThrust 2.15.0 for ROCm 5.2

### Additions

* Packages for tests and benchmark executables on all supported operating systems using CPack

### Known issues

* `async_copy`, `partition`, and `stable_sort_by_key` unit tests are failing for HIP on Windows

## rocThrust 2.14.0 for ROCm 5.1

### Additions

* Updates to match upstream Thrust 1.15.0

### Known issues

* `async_copy`, `partition`, and `stable_sort_by_key` unit tests are failing for HIP on Windows

## rocThrust 2.13.0 for ROCm 5.0

### Changes

* Updates to match upstream Thrust 1.13.0
* Updates to match upstream Thrust 1.14.0
* Added async scan
* Scan algorithms: `inclusive_scan` now uses the `input-type` as `accumulator-type`; `exclusive_scan`
  uses `initial-value-type`
  * This changes the behavior of small-size input types with large-size output types (e.g. `short` input,
    `int` output) and low-res input with high-res output (e.g. `float` input, `double` output)

## rocThrust-2.11.2 for ROCm 4.5.0

### Additions

* Initial HIP on Windows support

### Changes

* Packaging has changed to a development package (called `rocthrust-dev` for `.deb` packages and
  `rocthrust-devel` for `.rpm` packages). Because rocThrust is a header-only library, there is no runtime package. To aid in the transition, the development package sets the `provides` field to `rocthrust`, so that existing packages that are dependent on rocThrust can continue to work. This `provides` feature is introduced as a deprecated feature because it will be removed in a future ROCm release.

### Known issues

* `async_copy`, `partition`, and `stable_sort_by_key` unit tests are failing for HIP on Windows
* Mixed-type exclusive scan algorithm is not using the initial value type for the results type

## [rocThrust-2.11.1 for ROCm 4.4.0]

### Additions

* gfx1030 support
* AddressSanitizer build option

### Fixes

* async_transform unit test failure

## [rocThrust-2.11.0 for ROCm 4.3.0]

### Additions

* Updates to match upstream Thrust 1.11
* gfx90a support
* gfx803 support re-enabled

## [rocThrust-2.10.9 for ROCm 4.2.0]

### Additions

* Updates to match upstream Thrust 1.10

### Changes

* rocThrust now requires CMake version 3.10.2 or greater

### Fixes

* Size zero inputs are now properly handled with newer ROCm builds, which no longer allow zero-size
  kernel grid/block dimensions
* Warning of unused results

## [rocThrust-2.10.8 for ROCm 4.1.0]

* There are no changes with this release

## [rocThrust-2.10.7 for ROCm 4.0.0]

### Additions

* Updated to upstream Thrust 1.10.0
* Implemented runtime error for unsupported algorithms and disabled respective tests
* Updated CMake to use downloaded rocPRIM

## [rocThrust-2.10.6 for ROCm 3.10]

### Additions

* `copy_if` on device test case

### Known issues

* We've disabled ROCm support for device malloc. As a result, rocThrust functionality dependent on
  device malloc does not work--avoid using device launched `thrust::sort` and `thrust::sort_by_key`. Note
  that Host launched functionality is not impacted.
  * A partial enablement of device malloc is possible by setting `HIP_ENABLE_DEVICE_MALLOC` to 1.
  * `thrust::sort` and `thrust::sort_by_key` may work on certain input sizes but we don't recommended
    this for production code.

## [rocThrust-2.10.5 for ROCm 3.9.0]

### Additions

* Updated to upstream Thrust 1.9.8
* New test cases for device-side algorithms

### Fixes

* Bug for binary search
* Implemented workarounds for `hipStreamDefault` hang

### Known issues

* We've disabled ROCm support for device malloc. As a result, rocThrust functionality dependent on
  device malloc does not work--avoid using device launched `thrust::sort` and `thrust::sort_by_key`. Note
  that Host launched functionality is not impacted.
  * A partial enablement of device malloc is possible by setting `HIP_ENABLE_DEVICE_MALLOC` to 1.
  * `thrust::sort` and `thrust::sort_by_key` may work on certain input sizes but we don't recommended
    this for production code.

## [rocThrust-2.10.4 for ROCm 3.8.0]

### Known issues

* We've disabled ROCm support for device malloc. As a result, rocThrust functionality dependent on
  device malloc does not work--avoid using device launched `thrust::sort` and `thrust::sort_by_key`. Note
  that Host launched functionality is not impacted.
  * A partial enablement of device malloc is possible by setting `HIP_ENABLE_DEVICE_MALLOC` to 1.
  * `thrust::sort` and `thrust::sort_by_key` may work on certain input sizes but we don't recommended
    this for production code.

## [rocThrust-2.10.3 for ROCm 3.7.0]

### Additions

* Updated to upstream Thrust 1.9.4

### Changes

* Package dependency has changed to rocPRIM only

### Known issues

* We've disabled ROCm support for device malloc. As a result, rocThrust functionality dependent on
  device malloc does not work--avoid using device launched `thrust::sort` and `thrust::sort_by_key`. Note
  that Host launched functionality is not impacted.
  * A partial enablement of device malloc is possible by setting `HIP_ENABLE_DEVICE_MALLOC` to 1.
  * `thrust::sort` and `thrust::sort_by_key` may work on certain input sizes but we don't recommended
    this for production code.

## [rocThrust-2.10.2 for ROCm 3.6.0]

### Known issues

* We've disabled ROCm support for device malloc. As a result, rocThrust functionality dependent on
  device malloc does not work--avoid using device launched `thrust::sort` and `thrust::sort_by_key`. Note
  that Host launched functionality is not impacted.
  * A partial enablement of device malloc is possible by setting `HIP_ENABLE_DEVICE_MALLOC` to 1.
  * `thrust::sort` and `thrust::sort_by_key` may work on certain input sizes but we don't recommended
    this for production code.

## [rocThrust-2.10.1 for ROCm 3.5.0]

### Additions

* Improved tests with fixed and random seeds for test data

### Changes

* CMake searches for rocThrust locally first; if it isn't found, CMake downloads it from GitHub

### Deprecations

* HCC build has been deprecated
