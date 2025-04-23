.. meta::
   :description: rocALUTION prerequisites
   :keywords: rocALUTION, ROCm, library, API, tool, Windows, Linux, installation, building, HIP SDK, prerequisites


**************************
rocALUTION prerequisites
**************************

rocALUTION can be installed along with `ROCm <https://rocm.docs.amd.com/en/latest/>`_ or `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_, or it can be built from source code. 

HIP SDK for Windows must be installed before building rocALUTION on Microsoft Windows, and ROCm must be installed before building rocALUTION on Linux.


Building rocALUTION from source requires the following prerequisites on both Linux and Windows:

* `CMake <https://cmake.org/>`_
* `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/index.html>`_
* `rocSPARSE <https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/index.html>`_
* `rocRAND <https://rocm.docs.amd.com/projects/rocRAND/en/latest/index.html>`_
* `rocPRIM <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/index.html>`_

On Windows, the following prerequisites are also required:

* `Python 3 <https://www.python.org/downloads/>`_
* `Ninja <https://ninja-build.org/>`_
* `Strawberry Perl <https://strawberryperl.com/>`_
