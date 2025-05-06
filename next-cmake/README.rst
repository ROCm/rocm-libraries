.. highlight:: rst
.. |project_name| replace:: hipBLASLt

==============
|project_name|
==============

-----------------
Quick Start Guide
-----------------

This section describes how to configure and build the |project_name| project. We assume the user has a
ROCm installation, Python 3.8 or newer and CMake 3.25.0 or newer.

The |project_name| project consists of three components:

1. host library
2. device libraries
3. client applications

Each component has a corresponding subdirectory. The host and device libraries are independently
configurable and buildable but the client applications require the host library build time and the
device libraries at runtime.

^^^^^^^^^^^^^^^^^^^
Configure and build
^^^^^^^^^^^^^^^^^^^

|project_name| provides modern CMake support and relies on native CMake fnuctionality with exception of
some project specific options. As such, users are advised to refer to the CMake documentation for
general usage questions. Below are usage examples to get started. For details on all configuration
options see the options section.

Full build of |project_name| 
-----------------------

   .. code-block:: cmake
      :linenos:

      cd |project_name|/next-cmake
      # configure
      CC=/opt/rocm/bin/amdclang++          \
      CXX=/opt/rocm/bin/amdclang++         \
      cmake -B build                       \
            -S .                           \
            -D CMAKE_BUILD_TYPE=Release    \
            -D CMAKE_PREFIX_PATH=/opt/rocm \
            -D GPU_TARGETS=gfx950          \
            -D HIPBLASLT_ENABLE_DEVICE=ON  \
            -D HIPBLASLT_ENABLE_HOST=OFF   \
            -D HIPBLASLT_ENABLE_CLIENT=OFF 
      # build
      cmake --build build --parallel 32

Building device libraries
-------------------------
   .. code-block:: cmake
      :linenos:
      :emphasize-lines: 8,9

      cd |project_name|/next-cmake
      # configure
      CC=/opt/rocm/bin/amdclang++          \
      CXX=/opt/rocm/bin/amdclang++         \
      cmake -B build                       \
            -S .                           \
            -D CMAKE_BUILD_TYPE=Release    \
            -D CMAKE_PREFIX_PATH=/opt/rocm \
            -D GPU_TARGETS=gfx950          \
            -D HIPBLASLT_ENABLE_DEVICE=ON  \
            -D HIPBLASLT_ENABLE_HOST=OFF   \
            -D HIPBLASLT_ENABLE_CLIENT=OFF 
      # build
      cmake --build build --parallel 32

.. tip::
      **For Developers**

      View debugging info by adding ``--log-level=VERBOSE`` to the configure command.


Options
-------

*CMake options*:
* `CMAKE_BUILD_TYPE`: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* `CMAKE_INSTALL_PREFIX`: Base installation directory
* `CMAKE_PREFIX_PATH`: Find package search path (consider setting to ``$ROCM_PATH``)

*Project wide options*:

* `HIPBLASLT_ENABLE_HOST`: Enables generation of host library (default: `ON`)
* `HIPBLASLT_ENABLE_DEVICE`: Enables generation of device libraries (default: `ON`)
* `HIPBLASLT_ENABLE_CLIENT`: Enables generation of client applications (default: `ON`)
* `HIPBLASLT_ENABLE_LAZY_LOAD` Enable lazy loading of runtime code oject files to reduce init costs (default: `ON`)
* `GPU_TARGETS:` Semicolon separated list of gfx targets to build


*Host library options:*

* `HIPBLASLT_ENABLE_BLIS`: Enable BLIS support (default `ON`)
* `HIPBLASLT_ENABLE_HIP`: Use the HIP runtime (default `ON`)
* `HIPBLASLT_ENABLE_LLVM`: Use msgpack for parsing configuration files (default `OFF`)
* `HIBLASLT_ENABLE_MSGPACK`` Use msgpack for parsing configuration files (default `ON`)
* `HIPBLASLT_ENABLE_OPENMP`: "Use OpenMP to improve performance (default `ON`)
* `HIPBLASLT_ENABLE_ROCROLLER:` Use RocRoller library (default `OFF`)

*Device libraries options:*

* `HIPBLASLT_DEVICE_JOBS:` Allow N jobs generating device code libraries (default empty, uses nproc jobs)
* `HIPBLASLT_DEVICE_KEEP_TMP:` Keep temporary build files (default `OFF`)
* `HIPBLASLT_DEVICE_LIBLOGIC_PATH:` Custom path to library logic files (default empty, uses path to 'library')

*Client options:*

* `HIPBLASLT_BUILD_TESTING:` Build hipblaslt client tests (default `ON`)
* `HIPBLASLT_ENABLE_SAMPLES:` Build client samples (default `ON`)


CMake Targets
-------------

* `roc::hipblaslt`
* `rocisa::rocisa-cpp`

---------------
Physical Design
---------------

|project_name| consists of three components:

1. host library
2. device libraries
3. client applications

Each component has a corresponding directory. The host
and device libraries are independently configurable and
buildable but the client applications require the host
library to build and the device libraries to run.
