.. meta:: 
  :description: hipCUB Installation Prerequisites
  :keywords: install, hipCUB, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
hipCUB prerequisites
********************************************************************

hipCUB has the following prerequisites on Linux:

* `CMake <https://cmake.org/>`_ version 3.18 or higher
* `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html>`_ 
* `amdclang++ <https://rocm.docs.amd.com/projects/llvm-project/en/latest/index.html>`_ 
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ 

amdclang++ is installed with ROCm. rocPRIM is automatically downloaded and installed by the CMake script.


hipCUB has the following prerequisites on Microsoft Windows:

* `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_
* Python version 3.6 or later
* Visual Studio 2019 with Clang support
* Strawberry Perl
