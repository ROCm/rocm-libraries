.. meta::
  :description: Build and install hipCCL from source
  :keywords: install, building, hipCCL, AMD, ROCm, source code, cmake, Linux, Windows

.. _build-from-source:

****************************
Build hipCCL from source
****************************

To build hipCCL as part of the ROCm Core SDK, use the `TheRock build instructions <https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

You can also build hipCCL on its own.

.. _hipccl-prerequisites:

Prerequisites
=============

hipCCL on Linux requires `ROCm <https://rocm.docs.amd.com/en/latest/>`_.
hipCCL uses `HIPCC <https://rocm.docs.amd.com/projects/HIPCC/en/latest/index.html>`_ to build and run examples, tests, and benchmarks.

`CMake 3.25.2 or later <https://cmake.org/>`_ and C++17 are required.

In ``hipccl3``, hipCUB and rocThrust also require libhipcxx 3.4 or later on the ROCm platform.
See :ref:`hipccl-libhipcxx`.

.. _hipccl-get-source:

Get the hipCCL source code
==========================

The hipCCL source code is in the `ROCm libraries GitHub repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipccl>`_.
Use a sparse checkout when cloning hipCCL.
The project is self-contained and doesn't need any other directory under ``projects/`` alongside it.

.. code:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipccl

Then run ``git checkout`` for the branch you need.

The develop branch is for previewing new features or contributing to hipCCL.
If you aren't contributing and aren't previewing features, use a branch that matches the ROCm version installed on your system.

.. _hipccl-build:

Build on Linux
==============

hipCCL is built on Linux with CMake.
There are two entry points.

- The layout selector lets one ``cmake`` invocation choose the ``hipccl3`` layout or the ``hipccl2`` layout.
  ``hipccl3`` is the default.

  .. code:: shell

     mkdir build
     cd build
     export CXX=hipcc
     cmake -S ../projects/hipccl -B .
     make install

  Pass ``-DHIPCCL_BUILD_LEGACY=ON`` to CMake to build the ``hipccl2`` layout instead.

- You can also build a layout directly.

  .. code:: shell

     cd projects/hipccl/hipccl3
     mkdir build
     cd build
     export CXX=hipcc
     cmake ../.
     make install

  The difference between the two layouts is documented in :doc:`../conceptual/hipccl-layout`.

Both layouts and the selector share these CMake options.

``HIPCCL_BUILD_LEGACY``
    Set to ``ON`` to build the ``hipccl2`` layout instead of the default ``hipccl3`` layout.
    This option applies only when building through the ``projects/hipccl`` selector.
    The default is ``OFF``.

``HIPCCL_BUILD_ROCPRIM``
    Set to ``OFF`` to exclude rocPRIM from the build.
    The default is ``ON``.

``HIPCCL_BUILD_HIPCUB``
    Set to ``OFF`` to exclude hipCUB from the build.
    The default is ``ON``.

``HIPCCL_BUILD_ROCTHRUST``
    Set to ``OFF`` to exclude rocThrust from the build.
    The default is ``ON``.

``HIPCCL_BUILD_ALL``
    Set to ``ON`` to force ``HIPCCL_BUILD_ROCPRIM``, ``HIPCCL_BUILD_HIPCUB``, and ``HIPCCL_BUILD_ROCTHRUST`` on, including any that were set to ``OFF``.
    It doesn't build libhipcxx; see ``HIPCCL_BUILD_LIBHIPCXX``.
    The default is ``OFF``.

``HIPCCL_BUILD_LIBHIPCXX``
    This option exists only for ``hipccl3``.
    Set to ``ON`` to also build the ``libhipcxx`` git submodule as part of the build.
    Initialize the submodule first with ``git submodule update --init projects/hipccl/hipccl3/libhipcxx``.
    The default is ``OFF``.
    The submodule build isn't installed; see :ref:`hipccl-libhipcxx`.

``HIPCCL_USE_LEGACY_API``
    This option exists only for ``hipccl3``.
    Set to ``ON`` to enable the rocPRIM and hipCUB APIs from before CCCL 3.0 parity that are kept for backward compatibility, for example with older PyTorch releases.
    The value is recorded in the installed ``rocprim_version.hpp`` and ``hipcub_version.hpp`` as ``ROCPRIM_USE_LEGACY_API`` and ``HIPCUB_USE_LEGACY_API``.
    The default is ``ON``.

``HIPCCL_ENABLE_COVERAGE``
    Set to ``ON`` to build rocPRIM, hipCUB, and rocThrust with code coverage enabled.
    This sets ``ROCPRIM_ENABLE_COVERAGE``, ``HIPCUB_ENABLE_COVERAGE``, and ``ROCTHRUST_ENABLE_COVERAGE`` together.
    The default is ``OFF``.
    The older ``BUILD_CODE_COVERAGE`` option is deprecated.
    It still works but prints a deprecation message. Use ``HIPCCL_ENABLE_COVERAGE`` instead.

``HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS``
    Set to ``ON`` to install the temporary compatibility symlinks described in :doc:`../conceptual/hipccl-layout`.
    Set to ``OFF`` to skip them.
    The default is ``ON``.

``BUILD_DOCS``
    Set to ``ON`` to build a local copy of the hipCCL documentation.
    The default is ``OFF``.

rocPRIM, hipCUB, and rocThrust also accept their own ``BUILD_TEST``, ``BUILD_BENCHMARK``, and ``BUILD_EXAMPLE`` CMake options.
Those options are forwarded unchanged.
Each component's build documentation lists them.

Run ``make`` after ``cmake`` to build, then run ``make install``.
For example, to build only rocPRIM and hipCUB, and to skip rocThrust:

.. code:: shell

   export CXX=hipcc
   cmake -DHIPCCL_BUILD_ROCTHRUST=OFF ../.
   make
   sudo make install

In ``hipccl3``, hipCUB also needs libhipcxx; see :ref:`hipccl-libhipcxx`.

Run the tests
-------------

Configure with ``-DBUILD_TEST=ON``, build, and run ``ctest`` from the build directory.
hipCCL doesn't add tests of its own.
The test suites are those of rocPRIM, hipCUB, and rocThrust, built through the same configuration.
Each component also builds a ``generate_resource_spec`` helper that produces the resource file CTest needs to schedule tests across several GPUs.
For the test options, how to run in parallel, and how to pick GPUs, see each component's ``testing.md``:
`rocPRIM <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipccl/hipccl3/rocprim/testing.md>`_,
`hipCUB <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipccl/hipccl3/hipcub/testing.md>`_, and
`rocThrust <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipccl/hipccl3/rocthrust/testing.md>`_.

.. _hipccl-libhipcxx:

libhipcxx requirement
---------------------

In ``hipccl3``, hipCUB and rocThrust require libhipcxx 3.4 or later on the ROCm platform.
Configuration fails if neither of the following is true:

- A libhipcxx installation that meets the minimum version is found by ``find_package(libhipcxx)``.
  Set ``CMAKE_PREFIX_PATH`` or ``ROCM_ROOT`` so it can be located.
  The source is at `ROCm/libhipcxx <https://github.com/ROCm/libhipcxx>`_.
  This is the recommended way.
- The ``hipccl3`` superbuild builds the submodule, when ``-DHIPCCL_BUILD_LIBHIPCXX=ON`` is passed.
  Initialize the submodule first.

For example, to build the submodule:

.. code:: shell

   git submodule update --init projects/hipccl/hipccl3/libhipcxx
   cmake -S projects/hipccl/hipccl3 -B build -DHIPCCL_BUILD_LIBHIPCXX=ON

``HIPCCL_BUILD_ALL`` doesn't build libhipcxx.
The ``hipccl2`` layout has no such requirement.

.. important::

   **Known issue:** libhipcxx built from the submodule is used for the build only and isn't installed.
   The installed hipCUB and rocThrust CMake packages declare a dependency on a ``libhipcxx`` package.
   A project that calls ``find_package(hipcub)``, ``find_package(rocthrust)``, or ``find_package(hipccl)`` against that install needs libhipcxx 3.4 or later installed separately.

   To avoid the problem:

   - Install libhipcxx 3.4 or later into the same prefix as hipCCL, or into a prefix on your project's ``CMAKE_PREFIX_PATH``, before you configure the project that uses hipCCL.
   - When you plan to install hipCCL for other projects to use, build hipCCL against that installed libhipcxx instead of using ``HIPCCL_BUILD_LIBHIPCXX=ON``.
   - If you use ``HIPCCL_BUILD_LIBHIPCXX=ON`` only to build and test hipCCL in place, the issue doesn't affect you.

CMake presets
-------------

``projects/hipccl/hipccl3/CMakePresets.json`` provides the ``hipccl-all``, ``rocprim``, ``hipcub``, and ``rocthrust`` presets.
They configure a release build into ``build`` with an install prefix of ``/opt/rocm``.
None of the presets build libhipcxx.
Install libhipcxx first, or add ``-DHIPCCL_BUILD_LIBHIPCXX=ON`` to a preset after initializing the submodule.
See :ref:`hipccl-libhipcxx` for the install limitation that applies to the second option.

.. code:: shell

   cd projects/hipccl/hipccl3
   cmake --preset hipccl-all
   cmake --build build
