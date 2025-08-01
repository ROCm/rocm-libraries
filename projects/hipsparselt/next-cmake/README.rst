.. highlight:: rst
.. |project_name| replace:: hipSPARSELt

==============
|project_name|
==============

-----------------
Quick Start Guide
-----------------

This section describes how to configure and build the |project_name| project. We assume the user has a
ROCm installation, Python 3.8 or newer, and CMake 3.25.2 or newer.

^^^^^^^^^^^^^^^^^^^
Configure and build
^^^^^^^^^^^^^^^^^^^

|project_name| provides modern CMake support and relies on native CMake functionality with exception of
some project specific options. As such, users are advised to refer to the CMake documentation for
general usage questions. Below are some examples to get started. For details on all configuration
options see the options section.

Build on fresh clone of `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_
--------------------------------------------------------------------------------

   .. code-block:: cmake
      :linenos:

      cd projects/hipsparselt/next-cmake

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

Release build
-------------

   .. code-block:: cmake

      # configure with default release preset
      cmake --preset default-release

      # build
      cmake --build _build --parallel 32 --verbose

.. note::

      Built executables are placed in the ``_build`` directory by default when using presets.

Debug build for development
---------------------------

   .. code-block:: cmake

      cmake --preset debug
      cmake --build _build

Build with coverage
-------------------

   .. code-block:: cmake

      cmake --preset coverage
      cmake --build _build --target coverage

Build specific GPU targets
--------------------------

   .. code-block:: cmake

      cmake --preset default-release -D GPU_TARGETS="gfx1201"
      cmake --build _build

Build with CUDA support
-----------------------

   .. code-block:: cmake

      cmake --preset cuda
      cmake --build _build

.. tip::

      Make sure that `HIP_PLATFORM="nvidia"` is set in the environment when building with CUDA.

List available presets
----------------------

   .. code-block:: cmake

      cd projects/hipsparselt/next-cmake
      cmake --list-presets

Options
-------

*CMake options*:

* ``CMAKE_BUILD_TYPE``: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* ``CMAKE_INSTALL_PREFIX``: Base installation directory (defaults to ``/opt/rocm`` on Linux, ``C:/hipSDK`` on Windows)
* ``CMAKE_PREFIX_PATH``: Find package search path (consider setting to ``$ROCM_PATH``)
* ``CMAKE_EXPORT_COMPILE_COMMANDS``: Export compile_commands.json for clang tooling support (default: ``ON``)

*Build control options*:

* ``GPU_TARGETS``: AMD GFX targets to cross-compile for (default: ``all``)
* ``HIPSPARSELT_BUILD_SHARED_LIBS``: Build the |project_name| shared or static library (default: ``ON``)
* ``HIPSPARSELT_BUILD_TESTING``: Build test client (default: ``ON``)
* ``HIPSPARSELT_BUILD_COVERAGE``: Build tests with coverage support (default: ``OFF``)

*Backend options*:

* ``HIPSPARSELT_ENABLE_HIP``: Build hipSPARSELt with HIP backend (default: ``ON``)
* ``HIPSPARSELT_ENABLE_CUDA``: Build hipSPARSELt with CUDA backend (default: ``OFF``)

*Client options*:

* ``HIPSPARSELT_ENABLE_CLIENT``: Build hipSPARSELt clients (default: ``ON``)
* ``HIPSPARSELT_ENABLE_BENCHMARKS``: Build benchmark client (default: ``ON``)
* ``HIPSPARSELT_ENABLE_SAMPLES``: Build client samples (default: ``ON``)
* ``HIPSPARSELT_ENABLE_FORTRAN``: Build Fortran clients (default: ``OFF``)
* ``HIPSPARSELT_ENABLE_BLIS``: Enable BLIS support for reference implementations (default: ``ON``)

*Advanced options*:

* ``HIPSPARSELT_ENABLE_MARKER``: Enable rocTracer marker support (default: ``OFF``)
* ``HIPSPARSELT_ENABLE_ASAN``: Build with address sanitizer enabled (default: ``OFF``)
* ``HIPSPARSELT_ENABLE_VERBOSE``: Output additional build information (default: ``OFF``)
* ``HIPSPARSELT_HIPBLASLT_PATH``: Path to hipblaslt directory (default: ``${CMAKE_CURRENT_SOURCE_DIR}/../../hipblaslt/next-cmake``)
* ``HIPSPARSELT_COVERAGE_GTEST_FILTER``: GTest filter for coverage tests (default: ``*pre_checkin*``)

^^^^^^^^^^^^^
CMake Targets
^^^^^^^^^^^^^

*Libraries*:

* ``roc::hipsparselt`` - Main library target

*Executables*:

* ``hipsparselt-test`` - Test executable (when HIPSPARSELT_BUILD_TESTING=ON)
* ``hipsparselt-bench`` - Benchmark executable (when HIPSPARSELT_ENABLE_BENCHMARKS=ON)
* ``example_spmm_strided_batched`` - Sample executable (when HIPSPARSELT_ENABLE_SAMPLES=ON)
* ``example_prune_strip`` - Sample executable (when HIPSPARSELT_ENABLE_SAMPLES=ON)
* ``example_compress`` - Sample executable (when HIPSPARSELT_ENABLE_SAMPLES=ON)
* ``coverage`` - Code coverage target (when HIPSPARSELT_BUILD_COVERAGE=ON)
