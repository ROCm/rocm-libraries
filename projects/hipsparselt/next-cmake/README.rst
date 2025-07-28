.. highlight:: rst
.. |project_name| replace:: hipSPARSELt

==============
|project_name|
==============

----------------
Key Improvements
----------------

++++++++++++
Fixed Issues
++++++++++++

- ✅ **Removed Tensile-tag consumption mechanism** - Now uses `add_subdirectory` for `hipblaslt/tensilelite`
- ✅ **Fixed Python invocations** - Uses `Python3_EXECUTABLE` properly
- ✅ **Updated TensileCreateLibrary** - Uses modern `TensileCreateLibrary` instead of legacy `TensileCreateLibraryFiles`
- ✅ **Target-level compiler features** - No global `CMAKE_CXX_STANDARD` settings
- ✅ **Target-level flags** - No global `CMAKE_CXX_FLAGS` modifications
- ✅ **Fixed shared/static handling** - Uses `HIPSPARSELT_BUILD_SHARED_LIBS` without modifying `BUILD_SHARED_LIBS`
- ✅ **Removed hardcoded install prefix** - Uses standard CMake install prefix handling
- ✅ **Eliminated legacy commands** - No `include_directories()` or `add_definitions()`
- ✅ **Fixed cuSPARSELt detection** - Proper library finding and consumption

+++++++++++++++++++++
Modern CMake Features
+++++++++++++++++++++

- Target-level property management
- Proper generator expression usage
- Modern dependency management
- Component-based installation
- Proper imported target usage

-----------------
Quick Start Guide
-----------------

This section describes how to configure and build the |project_name| project. We assume the user has a
ROCm installation, Python 3.8 or newer and CMake 3.25.2 or newer.

^^^^^^^^^^^^^^^^^^^
Configure and build
^^^^^^^^^^^^^^^^^^^

|project_name| provides modern CMake support and relies on native CMake functionality with exception of
some project specific options. As such, users are advised to refer to the CMake documentation for
general usage questions. Below are usage examples to get started. For details on all configuration
options see the options section.

Full build of |project_name|
-----------------------

   .. code-block:: cmake
      :linenos:

      cd hipsparselt/next-cmake
      # configure
      cmake -B build                                       \
            -S .                                           \
            -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \
            -D CMAKE_C_COMPILER=/opt/rocm/bin/amdclang     \
            -D CMAKE_BUILD_TYPE=Release                    \
            -D CMAKE_PREFIX_PATH=/opt/rocm                 \
            -D GPU_TARGETS=gfx1201
      # build
      cmake --build build --parallel 32

.. tip::
      **For Developers**

      View debugging info by adding ``--log-level=VERBOSE`` to the configure command.

List available presets
----------------------

   .. code-block:: cmake

      cd projects/hipsparselt/next-cmake
      # View all configure presets
      cmake --list-presets=configure

Release build
-------------

   .. code-block:: cmake

      # Configure with default release preset
      cmake --preset default-release

      # Build using the fast preset (32 parallel jobs)
      cmake --build _build --parallel 32 --verbose

Debug build for development
---------------------------

   .. code-block:: cmake

      cmake --preset debug
      cmake --build _build
      ./_build/staging/hipsparselt-test

Build specific GPU targets
--------------------------

   .. code-block:: cmake

      cmake --preset default-release -D GPU_TARGETS="gfx1201"
      cmake --build _build --parallel 32 --verbose

Build with CUDA support
-----------------------

   .. code-block:: cmake

      cmake --preset cuda
      cmake --build _build --parallel 32 --verbose

.. tip::

      Make sure that `HIP_PLATFORM="nvidia"` is set in the environment when building with CUDA.

Options
-------

*CMake options*:

* ``CMAKE_BUILD_TYPE``: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* ``CMAKE_INSTALL_PREFIX``: Base installation directory (defaults to ``/opt/rocm`` on Linux, ``C:/hipSDK`` on Windows)
* ``CMAKE_PREFIX_PATH``: Find package search path (consider setting to ``$ROCM_PATH``)
* ``CMAKE_EXPORT_COMPILE_COMMANDS``: Export compile_commands.json for clang tooling support (default: ``ON``)

*Superbuild options*:

* ``BUILD_SHARED_LIBS``: Build the |project_name| shared or static library (default: ``ON``)
* ``BUILD_TESTING``: Build test client (default: ``ON``)
* ``BUILD_CODE_COVERAGE``: Build tests with coverage support (default: ``OFF``)

*Project wide options*:

* ``HIPSPARSELT_ENABLE_TENSILE``: Enables generation of device libraries (default: ``ON``)
* ``HIPSPARSELT_ENABLE_CLIENT``: Enables generation of client applications (default: ``ON``)
* ``HIPSPARSELT_ENABLE_ASAN``: Build with address sanitizer enabled (default: ``OFF``)
* ``HIPSPARSELT_BUILD_COVERAGE``: Build tests with coverage support (default: ``OFF``)

*|project_name| library options*:

* ``HIPSPARSELT_BUILD_SHARED_LIBS``: Build the |project_name| shared or static library (default: same as
* ``HIPSPARSELT_ENABLE_TENSILE``: Build |project_name| library with Tensile backend (default: ``ON``)
* ``HIPSPARSELT_ENABLE_HIPBLASLT``: Build |project_name| library with hipBLASLt backend (default: ``ON``)
* ``HIPSPARSELT_ENABLE_BLIS``: Enable BLIS support (default: ``ON``)
* ``HIPSPARSELT_ENABLE_MARKER``: Enable rocTracer marker support (default: ``OFF``)
* ``HIPSPARSELT_CONFIG_DIR``: Path placed into ldconfig file (default: ``${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}``)

*Tensile device libraries options*:

* ``HIPSPARSELT_TENSILE_SUBDIR_PATH``: Path to Tensile subdirectory (default: ``${CMAKE_CURRENT_SOURCE_DIR}/../../Tensile``)
* ``HIPSPARSELT_TENSILE_INSTALL_DIR``: Path to tensile library (default: ``${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/hipsparselt`` on Linux, ``${CPACK_PACKAGING_INSTALL_PREFIX}hipsparselt/bin`` on Windows)

*Client options*:

* ``HIPSPARSELT_BUILD_TESTING``: Build test client (default: same as ``BUILD_TESTING``)
* ``HIPSPARSELT_ENABLE_BENCHMARKS``: Build benchmark client (default: ``ON``)
* ``HIPSPARSELT_ENABLE_SAMPLES``: Build client samples (default: ``ON``)
* ``HIPSPARSELT_ENABLE_FORTRAN``: Build Fortran clients (default: ``OFF``)

CMake Targets
-------------

* ``roc::rocblas``

