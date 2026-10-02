.. meta::
  :description: hipCCL project layout - the hipccl3 and hipccl2 build layouts, unified include directory, and versioning
  :keywords: hipCCL, ROCm, layout, hipccl2, hipccl3, CMake

.. _hipccl-layout:

*********************
hipCCL project layout
*********************

hipCCL lives at ``projects/hipccl`` in the `rocm-libraries
<https://github.com/ROCm/rocm-libraries>`_ repository, and contains two build
layouts side by side, plus a small router that selects between them.

Two layouts
===========

``hipccl3`` (default)
  The unified, actively-developed layout. Its own copies of rocPRIM, hipCUB,
  and rocThrust are kept in sync with each project's own upstream
  development, the same way `NVIDIA's CCCL <https://github.com/NVIDIA/cccl>`_
  keeps CUB, Thrust, and libcudacxx in sync under one repository. New
  algorithm work and bug fixes land here.

``hipccl2``
  A frozen, pre-unification compatibility snapshot of rocPRIM, hipCUB, and
  rocThrust. It exists so that consumers who depended on the exact,
  pre-hipCCL behavior of these three libraries have a stable fallback during
  the migration - it is not kept in sync with upstream the way ``hipccl3``
  is.

Both layouts expose the same CMake option surface
(``HIPCCL_BUILD_ROCPRIM``/``HIPCUB``/``ROCTHRUST``/``ALL``, the same unified
``hipccl`` CPack package, and the same ``find_package(hipccl)`` entry point)
so that switching between them is a one-flag decision, not a different build
system to learn. See :doc:`../install/build` for the actual CMake invocations.

The layout selector
====================

``projects/hipccl/CMakeLists.txt`` is a small router that redirects a single
``cmake`` invocation to one layout or the other, without changing how either
behaves when built directly:

.. code-block:: shell

  # Default (HIPCCL_BUILD_LEGACY=OFF): build the unified hipccl3 project.
  cmake -S projects/hipccl -B build

  # Build the legacy hipccl2 layout instead.
  cmake -S projects/hipccl -B build -DHIPCCL_BUILD_LEGACY=ON

Building either layout directly (``cd hipccl3 && cmake ..``, or ``cd
hipccl2 && cmake ..``) continues to work exactly the same way - the router is
a purely additional entry point, not a replacement for either.

Unified include layout
=======================

All hipCCL headers install under ``<prefix>/include/hipccl/<component>``, for
example ``/opt/rocm/include/hipccl/rocprim`` and
``/opt/rocm/include/hipccl/thrust``, instead of each component claiming a
flat, top-level ``<prefix>/include/<component>`` path of its own. This avoids
collisions between hipCCL's bundled copies and any separately-installed,
standalone copy of rocPRIM/hipCUB/rocThrust sharing the same install prefix.

During the deprecation window, ``HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS``
(default ``ON``) additionally installs a compatibility symlink at the old,
flat location for each built component, so that existing code doing e.g.
``#include <rocprim/...>`` against a plain ``-I<prefix>/include`` keeps
working without changes. This is a temporary migration aid, not a permanent
feature - it will eventually be turned off and removed once downstream
consumers have migrated to the ``include/hipccl/<component>`` paths.

Versioning
==========

hipCCL versions its own umbrella package (the CPack package and
``find_package(hipccl)``) independently of the individual components.
rocPRIM, hipCUB, and rocThrust each keep their own, pre-existing version line
and their own ``find_package(rocprim)`` etc. version - hipCCL does not force
them to match the umbrella version. This mirrors how hipCUB already
distinguishes "my own version" from "the CCCL version I correspond to."
