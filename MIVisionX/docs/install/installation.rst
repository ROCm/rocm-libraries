.. meta::
  :description: MIVisionX API
  :keywords: MIVisionX, ROCm, API, reference, data type, support

.. _installation:

******************************************
Installation
******************************************

This topic provides instructions for installing MIVisionX and related packages.

Prerequisites
=======================

The following are the hardware and OS requirements of the MIVisionX library. Refer to `System Requirements <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_ for additional information. 

Hardware
---------------------

* **CPU**: AMD64 
* **GPU**: AMD Radeon Graphics 
* **APU**: AMD Radeon Mobile/Embedded 

.. note::
    Some modules in MIVisionX can be built for CPU only. To take advantage of advanced features and modules we recommend using AMD GPUs or APUs.

Operating Systems
------------------

* Linux

    * Ubuntu - 22.04 or 24.04
    * RedHat - 8 or 9
    * SLES - 15-SP5

* Windows 10 or 11
* macOS Ventura 13 or Sonoma 14

Compiler
------------------
* AMD Clang++

Libraries
------------------
* CMake - Version 3.10 and above
* Half-precision floating-point(half) library - Version `1.12.0`
* MIOpen
* MIGraphX
* RPP
* pkg-config
* OpenCV - Version 3.X or 4.X
* FFmpeg - Version 4.4.2 and above

Linux installation
===========================

The installation process uses the following steps:

* `ROCm-supported hardware <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_ install verification
* Install ROCm 6.0.0 or later with `amdgpu-install <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/amdgpu-install.html>`_ with ``--usecase=rocm``
* Install **either** from existing Packages or by building Source files as described below

Installation from packages
------------------------------

Install MIVisionX runtime, development, and test packages. 

* Runtime package - ``mivisionx`` only provides the dynamic libraries and executables
* Development package - ``mivisionx-dev`` / ``mivisionx-devel`` provides the libraries, executables, header files, and samples
* Test package - ``mivisionx-test`` provides ``ctest`` to verify installation

Ubuntu
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    sudo apt-get install mivisionx mivisionx-dev mivisionx-test


CentOS / RedHat
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    sudo yum install mivisionx mivisionx-devel mivisionx-test


SLES
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    sudo zypper install mivisionx mivisionx-devel mivisionx-test


* Package install supports HIP backend
* CentOS/RedHat/SLES requires FFMPEG & OpenCV Dev packages manual install


Installation from source files
-------------------------------------

For your convenience a setup script is provided, ``MIVisionX-setup.py``, which installs all required dependencies:

.. code-block:: shell

  python MIVisionX-setup.py --directory [setup directory - optional (default:~/)]
                            --opencv    [OpenCV Version - optional (default:4.6.0)]
                            --ffmpeg    [FFMPEG V4.4.2 Installation - optional (default:ON) [options:ON/OFF]]
                            --amd_rpp   [MIVisionX VX RPP Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --neural_net[MIVisionX Neural Net Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --inference [MIVisionX Neural Net Inference Dependency Install - optional (default:ON) [options:ON/OFF]]
                            --developer [Setup Developer Options - optional (default:OFF) [options:ON/OFF]]
                            --reinstall [Remove previous setup and reinstall (default:OFF)[options:ON/OFF]]
                            --backend   [MIVisionX Dependency Backend - optional (default:HIP) [options:HIP/OCL/CPU]]
                            --rocm_path [ROCm Installation Path - optional (default:/opt/rocm ROCm Installation Required)]


* Install ROCm before running the setup script
* This script only needs to be executed once
* ROCm upgrade requires the setup script to be rerun

Using MIVisionX-setup.py 
--------------------------------

* Clone MIVisionX git repository

.. code-block:: shell

  git clone https://github.com/ROCm/MIVisionX.git

.. note::
    
    MIVisionX supports two GPU backends: HIP and OPENCL. 
    Refer to the following instructions for installing with HIP backend. 
    Refer to `OPENCL GPU backend <https://github.com/ROCm/MIVisionX/wiki/OpenCL-Backend>`_ 
    for instructions on installing with OpenCL backend. 

Instructions for building MIVisionX with the **HIP** GPU backend (default backend)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Run the setup script to install all the dependencies required by the **HIP** GPU backend:
  
    .. code-block:: shell

        cd MIVisionX
        python MIVisionX-setup.py


#. Run the following commands to build MIVisionX with the **HIP** GPU backend:

    .. code-block:: shell

        mkdir build-hip
        cd build-hip
        cmake ../
        make -j8
        sudo make install

#. Run tests - `test option instructions <https://github.com/ROCm/MIVisionX/wiki/CTest>`_

    .. code-block:: shell

        make test


Windows
------------------

* Windows SDK
* Visual Studio 2019 or later
* Install the latest `AMD drivers <https://www.amd.com/en/support>`_
* Install `OpenCL SDK <https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases/tag/1.0>`_
* Install `OpenCV 4.6.0 <https://github.com/opencv/opencv/releases/tag/4.6.0>`_

  * Set ``OpenCV_DIR`` environment variable to ``OpenCV/build`` folder
  * Add ``%OpenCV_DIR%\x64\vc14\bin`` or ``%OpenCV_DIR%\x64\vc15\bin`` to your ``$PATH``


Using Visual Studio
^^^^^^^^^^^^^^^^^^^^^^^

Use ``MIVisionX.sln`` to build for x64 platform

.. important::

    Some modules in MIVisionX are only supported on Linux

macOS
------------------

Refer to `macOS build instructions <https://github.com/ROCm/MIVisionX/wiki/macOS#macos-build-instructions>`_

.. important::

    macOS only supports MIVisionX CPU backend

Verify installation
=========================

Linux / macOS
-------------------------

The installer will copy: 

  + Executables into ``/opt/rocm/bin``
  + Libraries into ``/opt/rocm/lib``
  + Header files into ``/opt/rocm/include/mivisionx``
  + Apps, & Samples folder into ``/opt/rocm/share/mivisionx``
  + Documents folder into ``/opt/rocm/share/doc/mivisionx``
  + Model Compiler, and Toolkit folder into ``/opt/rocm/libexec/mivisionx``


Verify with sample application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  **Canny Edge Detection**

.. image:: ../../samples/images/canny_image.PNG
   :alt: Canny Image

.. code-block:: shell

    export PATH=$PATH:/opt/rocm/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib
    runvx /opt/rocm/share/mivisionx/samples/gdf/canny.gdf

.. note::

    * More samples are available at ``../samples/README.md#samples``
    * For macOS use ``export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/rocm/lib``


Verify with mivisionx-test package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Test package will install ``ctest`` module to test MIVisionX. Use the following steps to test package install:

.. code-block:: shell

    mkdir mivisionx-test && cd mivisionx-test
    cmake /opt/rocm/share/mivisionx/test/
    ctest -VV


Windows
---------------------

* ``MIVisionX.sln`` builds the libraries & executables in the folder ``MIVisionX/x64``
* Use ``RunVX`` to test the build

.. code-block:: shell

    ./runvx.exe ADD_PATH_TO/MIVisionX/samples/gdf/skintonedetect.gdf


Docker
=====================

MIVisionX provides developers with docker images for Ubuntu 20.04 / 22.04. Using docker images developers can quickly prototype and build applications without having to be locked into a single system setup or lose valuable time figuring out the dependencies of the underlying software.

For more information about the Docker files used to build MIVisionX containers and a suggested workflow, see `MIVisionX Docker documentation <../how-to/mivisionx-docker.html>`_.

MIVisionX docker
---------------------------

* `Ubuntu 22.04 <https://cloud.docker.com/repository/docker/mivisionx/ubuntu-22.04>`_
* `Ubuntu 20.04 <https://cloud.docker.com/repository/docker/mivisionx/ubuntu-20.04>`_

Tested configurations
--------------------------------

* Windows 10 or 11
* Linux distribution

  + Ubuntu - 22.04 or 24.04
  + RHEL - 8 or 9
  + SLES - 15-SP5

* ROCm: 6.3.0
* RPP - 1.9.0
* miopen-hip - 3.1.0.60100
* migraphx - 2.9.0.60100
* OpenCV - `4.6.0 <https://github.com/opencv/opencv/releases/tag/4.6.0>`_
* FFMPEG - `n4.4.2 <https://github.com/FFmpeg/FFmpeg/releases/tag/n4.4.2>`_
* Dependencies for all the above packages
* MIVisionX Setup Script - V3.7.0

Known issues
-------------------

* OpenCV 4.X support for some apps missing
* MIVisionX Package install requires manual prerequisites installation
    * FFMPEG
    * OpenCV

MIVisionX dependency map
====================================

.. # COMMENT: The following lines define objects for use in the tabel below. 
.. |br| raw:: html 

    <br />

.. |green-sq| image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/green_square.png
    :alt: Green Square
.. |blue-sq| image:: https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/blue_square.png
    :alt: Blue Square
.. |ub-lvl1| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-1?style=flat-square
    :alt: Ubuntu 18.04 Level 1
.. |ub-lvl2| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-2?style=flat-square
    :alt: Ubuntu 18.04 Level 1
.. |ub-lvl3| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-3?style=flat-square
    :alt: Ubuntu 18.04 Level 1
.. |ub-lvl4| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-4?style=flat-square
    :alt: Ubuntu 18.04 Level 1
.. |ub-lvl5| image:: https://img.shields.io/docker/v/kiritigowda/ubuntu-18.04/mivisionx-level-5?style=flat-square
    :alt: Ubuntu 18.04 Level 1


**Docker Image:** |br|
``sudo docker build -f docker/ubuntu20/{DOCKER_LEVEL_FILE_NAME}.dockerfile -t {mivisionx-level-NUMBER} .``

* |green-sq| New component added to the level
* |blue-sq| Existing component from the previous level

.. csv-table::
  :widths: 5, 5, 8, 16, 5

    **Build Level**, **MIVisionX Dependencies**, **Modules**, **Libraries and Executables**, **Docker Tag**
    Level_1, cmake |br| gcc |br| g++, amd_openvx  |br| utilities, |green-sq| ``libopenvx.so`` - OpenVX Lib - CPU |br| |green-sq| ``libvxu.so`` - OpenVX immediate node Lib - CPU |br| |green-sq| ``runvx`` - OpenVX Graph Executor - CPU with Display OFF, |ub-lvl1|
    Level_2, ROCm HIP |br| +Level 1, amd_openvx |br| amd_openvx_extensions |br| utilities, |green-sq| ``libopenvx.so``  - OpenVX Lib - CPU/GPU |br| |green-sq| ``libvxu.so`` - OpenVX immediate node Lib - CPU/GPU |br| |green-sq| ``runvx`` - OpenVX Graph Executor - Display OFF, |ub-lvl2|
    Level_3, OpenCV |br| FFMPEG |br| +Level 2, amd_openvx |br| amd_openvx_extensions |br| utilities, |blue-sq| ``libopenvx.so`` - OpenVX Lib |br| |blue-sq| ``libvxu.so`` - OpenVX immediate node Lib |br| |green-sq| ``libvx_amd_media.so`` - OpenVX Media Extension |br| |green-sq| ``libvx_opencv.so`` - OpenVX OpenCV InterOp Extension |br| |green-sq| ``mv_compile`` - Neural Net Model Compile |br| |green-sq| ``runvx`` - OpenVX Graph Executor - Display ON, |ub-lvl3|
    Level_4, MIOpen |br| MIGraphX |br| +Level 3, amd_openvx |br| amd_openvx_extensions |br| utilities, |blue-sq| ``libopenvx.so`` - OpenVX Lib |br| |blue-sq| ``libvxu.so`` - OpenVX immediate node Lib |br| |blue-sq| ``libvx_amd_media.so`` - OpenVX Media Extension |br| |blue-sq| ``libvx_opencv.so`` - OpenVX OpenCV InterOp Extension |br| |blue-sq| ``mv_compile`` - Neural Net Model Compile |br| |blue-sq| ``runvx`` - OpenVX Graph Executor - Display ON |br| |green-sq| ``libvx_nn.so`` - OpenVX Neural Net Extension |br| |green-sq| ``libvx_amd_migraphx.so`` - OpenVX MIGraphX Extension, |ub-lvl4|
    Level_5, AMD RPP |br| +Level 4, amd_openvx |br| amd_openvx_extensions |br| utilities, |blue-sq| ``libopenvx.so``  - OpenVX Lib |br| |blue-sq| ``libvxu.so`` - OpenVX immediate node Lib |br| |blue-sq| ``libvx_amd_media.so`` - OpenVX Media Extension |br| |blue-sq| ``libvx_opencv.so`` - OpenVX OpenCV InterOp Extension |br| |blue-sq| ``mv_compile`` - Neural Net Model Compile |br| |blue-sq| ``runvx`` - OpenVX Graph Executor - Display ON |br| |blue-sq| ``libvx_nn.so`` - OpenVX Neural Net Extension |br| |green-sq| ``libvx_rpp.so`` - OpenVX RPP Extension, |ub-lvl5|


.. note::
    OpenVX and the OpenVX logo are trademarks of the Khronos Group Inc.
