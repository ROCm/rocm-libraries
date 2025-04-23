.. meta::
  :description: rocAL prerequisites
  :keywords: rocAL, ROCm, API, installation, prerequisites

.. _install:

********************************************************************
rocAL prerequisites
********************************************************************

rocAL requires ROCm running on `accelerators based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_ installed with the `AMDGPU installer <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_.

rocAL can be installed on the following Linux environments:
  
* Ubuntu 22.04 or 24.04
* RedHat 8 or 9
* SLES 15-SP5

:doc:`Building rocAL from source <./rocAL-build-and-install>` requires CMake Version 3.10 or later, AMD Clang++ Version 18.0.0 or later, and the following compiler support:

* C++17
* OpenMP
* Threads

Most prerequisites are installed with the :doc:`package installer <./rocAL-package-install>`. 

When building rocAL from source, the |setup| setup script can be used to install prerequisites:

.. code:: shell
  
  rocAL-setup.py [-h] [--directory DIRECTORY; default ~/] \
                      [--rocm_path ROCM_PATH; default /opt/rocm] \
                      [--backend HIP|OCL; default HIP] \
                      [--ffmpeg ON|OFF; default OFF] \
                      [--reinstall ON|OFF; default OFF]

The following prerequisites are required and are installed with both the package installer and the setup script:

* `MIVisionX <https://rocm.docs.amd.com/projects/MIVisionX/en/latest/index.html>`_ with |openvx|_ and the VX_RPP and AMD Media extensions
* `The half-precision floating-point library <https://half.sourceforge.net>`_ version 1.12.0 or later
* `Google Protobuf <https://developers.google.com/protocol-buffers>`_ version 3.12.4 or later
* `LMBD Library <http://www.lmdb.tech/doc/>`_
* `TurboJPEG <https://libjpeg-turbo.org/>`_
* `PyBind11 <https://github.com/pybind/pybind11/releases/tag/v2.11.1>`_ version 2.11.1
* `RapidJSON <https://github.com/Tencent/rapidjson>`_
* Python3, Python3 pip, and  Python3 wheel

libstdc++-12-dev is required on Ubuntu 22.04 only and must be installed manually.

`rocJPEG <https://rocm.docs.amd.com/projects/rocJPEG/en/latest/index.html>`_ is required and must be installed manually.

`FFMPEG <https://www.ffmpeg.org>`_ is not required, but is installed by the package installer. It can also be installed with the setup script using the ``--ffmpeg`` option. 


`rocDecode <https://rocm.docs.amd.com/projects/rocDecode/en/latest/index.html>`_ and `OpenCV <https://docs.opencv.org/4.6.0/index.html>`_ are not required, but are installed by the package installer and the setup script.

.. note::
  
  | TurboJPEG must be installed manually on SLES. 
  | To use FFMPeg on SLES and RedHat, the ``FFMPeg-dev`` package must be installed manually.


.. |trade| raw:: html

    &trade;

.. |openvx| replace:: AMD OpenVX\ |trade|
.. _openvx: https://rocm.docs.amd.com/projects/MIVisionX/en/latest/install/amd_openvx-install.html#amd-openvx-install

.. |setup| replace:: ``rocAL-setup.py``
.. _setup: https://github.com/ROCm/rocAL/blob/develop/rocAL-setup.py
