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
* AMD Clang++ Version 18.0.0 or later for the HIP backend

With the following compiler support:

* C++17 or later
* OpenMP
* Threads

The HIP backend requires a working ROCm installation running on `ROCm-supported AMD GPUs and accelerators <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_. A ``gfx908`` or later GPU is required.

The CPU-only backend, also referred to as the HOST backend, has no GPU requirement.

The `test suite prerequisites <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rpp/utilities/test_suite/README.md>`_ are required to build the RPP test suite.
