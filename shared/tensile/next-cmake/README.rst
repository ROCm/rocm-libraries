.. highlight:: rst
.. |project_name| replace:: Tensile

==============
|project_name|
==============

-----------------
Quick Start Guide
-----------------

This section describes how to configure and build the |project_name| project. We assume the user has a
ROCm installation, Python 3.8 or newer and CMake 3.21.2 or newer.

The |project_name| project consists of three components:

1. host library
2. device libraries
3. client applications

Each component has a corresponding subdirectory. The host and device libraries are independently
configurable and buildable but the client applications require the host library at build time and the
device libraries at runtime.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Integration via `add_subdirectory`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use Tensile in your project, add it as an out of source subdirectory as follows:

.. code-block:: cmake
   :linenos:
   :emphasize-lines: 8

   add_subdirectory(/path/to/Tensile/next-cmake ${CMAKE_CURRENT_BINARY_DIR}/Tensile)

^^^^^^^^^^^^^^^^^^^
Configure and build
^^^^^^^^^^^^^^^^^^^

|project_name| provides modern CMake support and relies on native CMake functionality with exception of
some project specific options. As such, users are advised to refer to the CMake documentation for
general usage questions. For details on all configuration options see the options section.

.. note::
      Tensile is designed to be consumed via `add_subdirectory` and therefore isn't intended to be consumed via standalone builds.
      Standalone builds are only supported for development and testing purposes.

Full build of |project_name|
-----------------------

   .. code-block:: cmake
      :linenos:

      cd Tensile/next-cmake
      # configure
      cmake -B build                                       \
            -S .                                           \
            -D CMAKE_BUILD_TYPE=Release                    \
            -D CMAKE_PREFIX_PATH=/opt/rocm                 \
            -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
            -D GPU_TARGETS=gfx90a
      # build
      cmake --build build --parallel 32

----------------------------
Building and Running Tests
----------------------------

While full test suites can be run with a single ``tox`` command, developers may wish to
build the Tensile client executable (``tensile-client``) and run individual tests separately.
This is useful for debugging specific problems or isolating issues in a specific test.

**1. Run Full Test Suite with Tox**

The standard workflow for running an entire test suite (e.g., `pre_checkin`, `unit`, `extended`,
`integration`) is to use `tox`. This command will build ``tensile-client`` and execute all
tests within the specified suite.

   .. code-block:: bash
      :linenos:

    tox run -e ci -- -m {pre_checkin|extended|integration|unit}

**2. Build with invoke and Run a Test (Default Path)**

This workflow uses ``invoke`` to build the client into the default ``build/`` directory.
The executable will automatically be found when using the default build directory.

   .. code-block:: bash
      :linenos:

      # install invoke if you haven't already
      pip3 install invoke

      # build the client to the default location with the default option
      invoke build-client

      # run an individual test
      Tensile/bin/Tensile Tensile/Tests/pre_checkin/<test>.yaml tensile-out

**3. Build with CMake (Custom Location) and Run Test with Path Flag**

This workflow is for when you need to build the client in a location other
than the default ``build/`` directory by using CMake. The ``--prebuilt-client`` flag is then
used to specify this custom path when running a test.

   .. code-block:: cmake
      :linenos:

    # configure in a custom directory (e.g., my-custom-build)
    cmake -S next-cmake -B my-custom-build      \
          -DCMAKE_PREFIX_PATH=/opt/rocm         \
          -DTENSILE_ENABLE_CLIENT=ON            \
          -DTENSILE_ENABLE_HOST=ON              \
          -DTENSILE_ENABLE_DEVICE=OFF
      # build
      cmake --build my-custom-build --parallel

    # run a test, specifying the custom client path with --prebuilt-client
    Tensile/bin/Tensile Tensile/Tests/pre_checkin/<test>.yaml tensile-out \
      --prebuilt-client=my-custom-build/client/tensile-client

**4. Build with tox (Custom Build Args)**

This workflow uses ``tox`` with custom CMake arguments, which is useful for creating
specialized builds (e.g., Debug builds) and setting the architecture.

   .. code-block:: bash
      :linenos:

      # build the client using tox with custom CMake flags
      TENSILE_CLIENT_ARGS="--build-type Debug --gpu-targets gfx90a --clean" tox run -e build-client

Options
-------

*CMake options*:

* ``CMAKE_BUILD_TYPE``: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* ``CMAKE_INSTALL_PREFIX``: Base installation directory (defaults to /opt/rocm on Linux, C:/hipSDK on Windows)
* ``CMAKE_PREFIX_PATH``: Find package search path (consider setting to ``$ROCM_PATH``)
* ``CMAKE_EXPORT_COMPILE_COMMANDS``: Export compile_commands.json for clang tooling support (default: ``ON``)

*Project wide options*:

* ``GPU_TARGETS``: List of GPU targets to build for, if unset or set to "all", all supported targets will be built (default: unset)
* ``TENSILE_ENABLE_HOST``: Enables generation of host library (default: ``ON``)
* ``TENSILE_ENABLE_DEVICE``: Enables generation of device libraries (default: ``ON``)
* ``TENSILE_ENABLE_CLIENT``: Enables generation of client applications (default: ``ON``)
* ``TENSILE_BUILD_TESTING``: Build host library tests (default: ``ON``)
* ``TENSILE_CXX_STANDARD``: C++ standard version to use (14, 17, 20) (default: ``17``)

*Host library options*:

* ``TENSILE_BUILD_SHARED_LIBS``: Build the |project_name| shared or static library (default: ``OFF``)
* ``TENSILE_ENABLE_MSGPACK``: Enable MessagePack support (default: ``ON``)
* ``TENSILE_ENABLE_LLVM``: Use LLVM YAML library; this should only be used by project developers and for testing (default: ``OFF``)

*Client options*:

* ``TENSILE_ENABLE_ROCM_SMI``: Enable rocm_smi support (default: ``ON`` on Linux, ``OFF`` on Windows)

CMake Targets
-------------

* ``roc::tensile-host``: Main Tensile host library target
