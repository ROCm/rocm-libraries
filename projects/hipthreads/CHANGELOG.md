# Changelog for hipThreads

Full documentation for hipThreads is available at [https://rocm.docs.amd.com/projects/hipThreads/en/latest/](https://rocm.docs.amd.com/projects/hipThreads/en/latest/).

## hipThreads 1.0 for ROCm 10.0.0

We are pleased to introduce hipThreads, a C++-style concurrency library for AMD GPUs. hipThreads brings `std::thread`-like primitives inside GPU kernels, so existing `std::thread` CPU code can be ported to run on AMD GPUs with minimal changes. It is supported on both Linux and Windows.

Highlights of this release:

* **`std::thread`-style concurrency primitives that run inside GPU kernels.** `hip::wthread`, `hip::mutex`, `hip::lock_guard`, and `hip::condition_variable`, along with cooperative `pseudo_*` variants, mirror the familiar C++ standard-library concurrency API.
* **A persistent scheduler kernel** that accepts work from both the host and the device, with multi-fiber (SIMD `width`) execution so a single unit of work can run across multiple GPU lanes.
* **Runtime-tunable scheduling.** Scheduler concurrency can be configured at runtime through the `HIPTHREADS_VCORES_PER_WGP` environment variable to match your GPU and workload.
* **Cross-platform build and tooling.** A CMake build with native HIP language support, a `lit`-based test suite, and example projects — all supported on Linux and Windows.
* **Example ports** demonstrating incremental CPU to GPU migration: saxpy, the *Ray Tracing in One Weekend* renderer, sparse matrix multiplication, and llama3.c.
* **Full documentation** hosted at [https://rocm.docs.amd.com/projects/hipThreads/en/latest/](https://rocm.docs.amd.com/projects/hipThreads/en/latest/).
