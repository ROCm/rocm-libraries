.. meta::
  :description: rocPRIM prerequisites
  :keywords: install, rocPRIM, AMD, ROCm, prerequisites


***********************
rocPRIM prerequisites 
***********************

rocPRIM on Linux requires `ROCm <https://rocm.docs.amd.com/en/latest/>`_. rocPRIM on Windows requires `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_.

rocPRIM uses `HIPCC <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_ to build and run examples, tests, and benchmarks.

`CMake version 3.16 or later <https://cmake.org/>`_ and C++17 are required on both Linux and Windows.

The following additional prerequisites are needed on Windows only:

* `Python version 3.6 or higher <https://www.python.org/downloads/>`_
* `Microsoft Visual Studio 2019 with Clang support <https://visualstudio.microsoft.com/>`_
* `Strawberry Perl <https://www.strawberryperl.com/>`_
