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
      cmake -B build                                       \
            -S .                                           \
            -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
            -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
            -D CMAKE_BUILD_TYPE=Release                    \
            -D CMAKE_PREFIX_PATH=/opt/rocm                 \
            -D GPU_TARGETS=gfx950
      # build
      cmake --build build --parallel 32

Building device libraries
-------------------------
   .. code-block:: cmake
      :linenos:
      :emphasize-lines: 8,9

      cd |project_name|/next-cmake
      # configure
      cmake -B build                                       \
            -S .                                           \
            -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
            -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
            -D CMAKE_BUILD_TYPE=Release                    \
            -D CMAKE_PREFIX_PATH=/opt/rocm                 \
            -D GPU_TARGETS=gfx950
            -D HIPBLASLT_ENABLE_DEVICE=ON  \
            -D HIPBLASLT_ENABLE_HOST=OFF   \
            -D HIPBLASLT_ENABLE_CLIENT=OFF 
      # build
      cmake --build build --parallel 32

.. tip::
      **For Developers**

      View debugging info by adding ``--log-level=VERBOSE`` to the configure command.


----------------------------
Building and Running Tests
----------------------------

While full test suites can be run with a single ``tox`` command, developers may wish to
build the hipBLASLt tensilelite client executable (``tensilelite-client``) and run individual tests separately. 
This is useful for debugging specific problems or isolating issues in a specific test.

**1. Run Full Test Suite with Tox**

The standard workflow for running the entire test suite is to use `tox`. This command will build ``tensilelite-client`` and execute all 
tests.

   .. code-block:: bash
      :linenos:

      tox -e py3 -- Tensile/Tests -m common

**2. Build with invoke and Run a Test (Default Path)**

This workflow uses ``invoke`` to build the client into the default ``build/`` directory. 
The executable will automatically be found when using the default build directory.

   .. code-block:: bash
      :linenos:

      # install invoke and rocisa if you haven't already
      pip3 install invoke rocisa/

      # build the client to the default location with the default option
      invoke build-client

      # run an individual test
      Tensile/bin/Tensile Tensile/Tests/common/exception/<test>.yaml tensile-out

**3. Build with CMake (Custom Location) and Run Test with Path Flag**

This workflow is for when you need to build the client in a location other
than the default ``build/`` directory by using CMake. The ``--prebuilt-client`` flag is then 
used to specify this custom path when running a test. Make sure to do this in the ``hipblaslt/tensilelite`` directory.

   .. code-block:: cmake
      :linenos:

    # configure in a custom directory (e.g., my-custom-build)
    cmake -S ../next-cmake -B my-custom-build   \
          -DCMAKE_PREFIX_PATH=/opt/rocm         \
          -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
          -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang \
          -DHIPBLASLT_ENABLE_CLIENT=OFF         \
          -DHIPBLASLT_ENABLE_HOST=OFF           \
          -DHIPBLASLT_ENABLE_DEVICE=OFF         \
          -DTENSILELITE_ENABLE_HOST=ON          \
          -DTENSILELITE_ENABLE_CLIENT=ON        \
          -DHIPBLASLT_ENABLE_LLVM=ON
      # build
      cmake --build my-custom-build --parallel

    # run a test, specifying the custom client path with --prebuilt-client
    Tensile/bin/Tensile Tensile/Tests/pre_checkin/<test>.yaml tensile-out \
      --prebuilt-client=my-custom-build/tensilelite-client/tensilelite-client

**4. Build with tox (Custom Build Args)**

This workflow uses ``tox`` with custom CMake arguments, which is useful for creating
specialized builds (e.g., Debug builds) and setting the architecture.

   .. code-block:: bash
      :linenos:

      # build the client using tox with custom CMake flags 
      TENSILELITE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox -e py3 -- Tensile/Tests -m common


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
* `TENSILELITE_ENABLE_HOST`: Enables generation of tensilelite host (default: `ON`)
* `TENSILELITE_ENABLE_CLIENT`: Enables generation of tensilelite client application (default: `OFF`)
* `HIPBLASLT_ENABLE_LAZY_LOAD` Enable lazy loading of runtime code oject files to reduce init costs (default: `ON`)
* `GPU_TARGETS:` Semicolon separated list of gfx targets to build


*Host library options:*

* `HIPBLASLT_ENABLE_BLIS`: Enable BLIS support (default `ON`)
* `HIPBLASLT_ENABLE_HIP`: Use the HIP runtime (default `ON`)
* `HIPBLASLT_ENABLE_LLVM`: Use msgpack for parsing configuration files (default `OFF`)
* `HIPBLASLT_ENABLE_MSGPACK`` Use msgpack for parsing configuration files (default `ON`)
* `HIPBLASLT_ENABLE_OPENMP`: "Use OpenMP to improve performance (default `ON`)
* `HIPBLASLT_ENABLE_ROCROLLER:` Use RocRoller library (default `OFF`)

*Device libraries options:*

* `HIPBLASLT_DEVICE_JOBS:` Allow N jobs generating device code libraries (default empty, uses nproc jobs)
* `HIPBLASLT_KEEP_TMP:` Keep temporary build files (default `OFF`)
* `HIPBLASLT_LIBLOGIC_PATH:` Custom path to library logic files (default empty, uses path to 'library')

*Client options:*

* `HIPBLASLT_BUILD_TESTING:` Build hipblaslt client tests (default `ON`)
* `HIPBLASLT_ENABLE_SAMPLES:` Build client samples (default `ON`)


CMake Targets
-------------

* `roc::hipblaslt`
* `rocisa::rocisa-cpp`
* `tensilelite::tensilelite-client`

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
