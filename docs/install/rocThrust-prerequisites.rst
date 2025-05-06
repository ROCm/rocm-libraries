.. meta::
  :description: rocThrust Installation Prerequisites
  :keywords: installation, rocThrust, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocThrust prerequisites
********************************************************************

rocThrust has the following prerequisites on all platforms:

* ROCm 6.1 or later.
* CMake 3.10.2 or later.
* hipcc or amdclag++. See the `ROCm LLVM compiler infrastructure <https://rocm.docs.amd.com/projects/llvm-project/en/latest/index.html>`_ for more information. 
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_. rocPRIM is automatically downloaded and installed when rocThrust is built and installed.

rocThrust has the following HIP on Windows prerequisites:

* Python 3.6 or later
* Visual Studio 2019 with Clang support
* Strawberry Perl
