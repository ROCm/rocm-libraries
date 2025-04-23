.. meta::
  :description: ROCm Performance Primitives (RPP) prerequisites
  :keywords: RPP, ROCm, Performance Primitives, prerequisites

********************************************************************
ROCm Performance Primitives prerequisites
********************************************************************

ROCm Performance Primitives (RPP) is supported on the following operating systems:

* Ubuntu version 22.04 or 24.04
* RedHat version 8 or 9
* SLES 15-SP5

The following compilers and libraries are required to build and install RPP:

* half, the half-precision floating-point library, version 1.12.0 or later
* libstdc++-12-dev for Ubuntu 22.04 only
* Clang version 5.0.1 or later for CPU-only backends
* AMD Clang++ Version 18.0.0 or later for HIP and OpenCL backends

With the following compiler support:

* C++17 or later
* OpenMP
* Threads

On OpenCL and HIP backends, RPP requires ROCm installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ and the ``rocm`` usecase running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

On CPU-only backends, also referred to as HOST backends, RPP requires CPUs that support PCIe™ atomics.

The `test suite prerequisites <https://github.com/ROCm/rpp/blob/develop/utilities/test_suite/README.md>`_ are required to build the RPP test suite.
