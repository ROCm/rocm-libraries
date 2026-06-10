.. meta::
  :description: ROCm Performance Primitives (RPP) prerequisites
  :keywords: RPP, ROCm, Performance Primitives, prerequisites

********************************************************************
ROCm Performance Primitives prerequisites
********************************************************************

ROCm Performance Primitives (RPP) has been tested on the following Linux environments:

* Ubuntu 22.04 and 24.04

* RHEL 8 and 9
* SLES 15 SP7


See `Supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_ for the complete list of ROCm supported Linux environments.

The following compilers and libraries are required to build and install RPP:

* HIP
* OpenMP
* half, the half-precision floating-point library, version 1.12.0 or later
* libstdc++-12-dev for Ubuntu 22.04 only
* Clang version 5.0.1 or later for CPU-only backends
* AMD Clang++ Version 18.0.0 or later for HIP and OpenCL backends

With the following compiler support:

* C++17 or later
* OpenMP
* Threads

On OpenCL and HIP backends, RPP requires ROCm installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/install/install-methods/amdgpu-installer-index.html>`_ and the ``rocm`` usecase running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

On CPU-only backends, also referred to as HOST backends, RPP requires CPUs that support PCIe™ atomics.

The `test suite prerequisites <https://github.com/ROCm/rpp/blob/develop/utilities/test_suite/README.md>`_ are required to build the RPP test suite.

