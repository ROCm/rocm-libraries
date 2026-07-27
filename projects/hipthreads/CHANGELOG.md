# Changelog for hipThreads

Full documentation for hipThreads is available at [https://rocm.docs.amd.com/projects/hipThreads/en/latest/](https://rocm.docs.amd.com/projects/hipThreads/en/latest/).

The format is based on recording the noticeable changes for each release under the categories **Added**, **Changed**, **Optimized**, **Deprecated**, **Removed**, and **Resolved issues**.
When opening a pull request, add your meaningful changes to the appropriate section under "Since last release".

## Since last release

### Added

### Changed

### Optimized

### Deprecated

### Removed

### Resolved issues

## hipThreads 0.1.1 for ROCm 7.12

### Added

* Performance testing in CI, including all hipThreads examples (#86).
* Email notifications for CI workflow runs (#61).

### Changed

* Install public headers under a `hipthreads/` subdirectory (#107).
* Use `ROCM_PATH` in the top-level CMake build, and in the examples with an `/opt/rocm` fallback (#107).
* Updated `README.md`, `docs/setup.md`, and `docs/index.md` for ROCm 7.12+ build and install instructions (#107).
* Replace `__libcpp_thread_sleep_for` with `__cccl_thread_sleep_for` for CCCL 3.0 and newer (#87).
* Updated the CPU InOneWeekend Raytracer samples to avoid virtual functions for better performance (#96).

### Resolved issues

* Fixed duplicate `hipthreads` linking in `test/CMakeLists.txt` that caused device-side duplicate symbol errors (#105).
* Applied a placement-new workaround for a ROCm 7.12 compiler bug in the examples (#107).
* Fixed the libhipcxx include path for the CI build on TheRock 7.12 (#76).

## hipThreads 0.1.0 for ROCm 7.0.2

Initial early-access technology preview of hipThreads — a C++-style concurrency library that brings `std::thread`-like primitives to AMD GPUs.
This first release establishes the core library and its supporting infrastructure:

* `std::thread`-style concurrency primitives that run inside GPU kernels: `hip::wthread`, `hip::mutex`, `hip::lock_guard`, `hip::condition_variable`, and the cooperative `pseudo_*` variants.
* Persistent scheduler kernel with host- and device-side work submission, and multi-fiber (SIMD `width`) execution support.
* CMake build (native HIP language support), a `lit`-based test suite, and a CI/CD workflow.
* Example projects demonstrating incremental CPU to GPU ports (saxpy, InOneWeekend Raytracer, sparse matrix multiplication, and llama3.c).
* Initial documentation (`README.md`, `docs/`, and Doxygen configuration).
