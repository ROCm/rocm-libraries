.. meta::
  :description: rocThrust Installation Prerequisites
  :keywords: installation, rocThrust, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocThrust prerequisites
********************************************************************

On Linux, `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/index.html>`_ must be installed before rocThrust is installed.

rocThrust has the following prerequisites on Linux and Microsoft Windows:

* `CMake <https://cmake.org/>`_ version 3.10.2 or higher
* `hipcc <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_ 
* `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`_ 

rocPRIM can be automatically downloaded and installed when rocThrust is built.

rocThrust has these additional prerequisites on Windows:

* `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_
* `Python version 3.6 or later <https://www.python.org/>`_
* `Visual Studio 2019 with Clang support <https://visualstudio.microsoft.com/>`_
* `Strawberry Perl <https://strawberryperl.com/>`_

