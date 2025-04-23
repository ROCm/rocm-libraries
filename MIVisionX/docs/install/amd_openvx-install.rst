.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _amd-openvx-install:

******************************************
AMD OpenVX installation
******************************************

Pre-requisites
==============

* **CPU**: AMD64
* **GPU**: AMD Radeon Graphics [optional]

  + Windows: install the latest drivers and OpenCL SDK `download <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases>`_
  + Linux: install `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_


Build Instructions
==================

Build this project to generate AMD OpenVX library 

* Refer to `openvx/include/VX <https://github.com/ROCm/MIVisionX/tree/master/amd_openvx/openvx/include>`_ for Khronos OpenVX standard header files.
* Refer to `openvx/include/vx_ext_amd.h <https://github.com/ROCm/MIVisionX/tree/master/amd_openvx/openvx/include/vx_ext_amd.h>`_ for vendor extensions in AMD OpenVX library

Build using Microsoft Visual Studio
---------------------------

* Optionally download and install `OpenCV <https://github.com/opencv/opencv/releases>`_ with or without `opencv_contrib modules <https://github.com/opencv/opencv_contrib>`_ to enable the ``RunVX`` tool to support camera capture and image display

  + ``OpenCV_DIR`` environment variable should point to ``OpenCV/build`` folder

* Use ``amd_openvx/amd_openvx.sln`` to build for ``x64`` platform
* If AMD GPU (or OpenCL) is not available, set build flag ``ENABLE_OPENCL=0``in ``openvx/openvx.vcxproj`` and ``runvx/runvx.vcxproj``

.. note:: 
  AMD GPU ``HIP`` backend is not supported on Windows 

Build using CMake
-----------------

* Install CMake 3.10 or later
* Use CMake to configure and generate Makefile
