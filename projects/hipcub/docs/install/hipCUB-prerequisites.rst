.. meta:: 
  :description: hipCUB Installation Prerequisites
  :keywords: install, hipCUB, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
hipCUB prerequisites
********************************************************************

hipCUB has the following prerequisites on all platforms:

* `CMake <https://cmake.org/>`_ version 3.16 or higher

On AMD GPUs:

* `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html>`_ 
* `amdclang++ <https://rocm.docs.amd.com/projects/llvm-project/en/latest/index.html>`_ 
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ 

amdclang++ is installed with ROCm. rocPRIM is automatically downloaded and installed by the CMake script.

On NVIDIA GPUs:

* The CUDA Toolkit
* CCCL library version 2.3.2 or later
* CUB and Thrust
* libcu++ version 2.2.0

The CCCL library is automatically downloaded and built by the CMake script. If libcu++ isn't found on the system, it will be downloaded from the CCCL repository.

On Microsoft Windows:


* Python verion 3.6 or later
* Visual Studio 2019 with Clang support
* Strawberry Perl
