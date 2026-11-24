.. meta::
  :description: Use hipCCL from CMake and C++ code, and migrate from standalone rocPRIM, hipCUB, and rocThrust
  :keywords: hipCCL, find_package, CMake, include paths, migration, rocPRIM, hipCUB, rocThrust

.. _use-hipccl:

*************************
Use hipCCL in a project
*************************

This page describes how to consume an installed hipCCL from CMake and C++, and how to move a project that used standalone rocPRIM, hipCUB, or rocThrust.

Find hipCCL in CMake
====================

The CMake package name is ``hipccl``.
The supported components are ``rocprim``, ``hipcub``, and ``rocthrust``.

.. code:: cmake

   # All three components
   find_package(hipccl REQUIRED)
   target_link_libraries(my_target PRIVATE hipccl::hipccl)

   # One component
   find_package(hipccl REQUIRED COMPONENTS hipcub)
   target_link_libraries(my_target PRIVATE hip::hipcub)

``find_package(hipccl)`` finds each requested component's own CMake package and doesn't define the component targets itself.
The targets are:

.. list-table::
   :header-rows: 1

   * - Component
     - Target
   * - rocPRIM
     - ``roc::rocprim``
   * - hipCUB
     - ``hip::hipcub``
   * - rocThrust
     - ``roc::rocthrust``
   * - All three
     - ``hipccl::hipccl``, an interface target that links the three targets above.
       It's defined only when all three components are found.

A component that is requested under ``OPTIONAL_COMPONENTS`` and isn't found doesn't cause an error, and neither does an unknown name under ``OPTIONAL_COMPONENTS``.
Requesting a component other than the three above under ``COMPONENTS`` fails with an unsupported component message.

``find_package(rocprim)``, ``find_package(hipcub)``, and ``find_package(rocthrust)`` also continue to work and use the same targets.

A hipCCL built from source installs to ``/opt/rocm`` by default.
A packaged ROCm install might use a different prefix.
If ROCm isn't on your ``CMAKE_PREFIX_PATH``, add it, or point ``hipccl_DIR`` at ``install_prefix/lib/cmake/hipccl``.

Include paths
=============

hipCCL installs headers under ``install_prefix/include/hipccl/component_name``.
Linking a hipCCL target adds the right include directories, so the includes in your code stay the same:

.. code:: cpp

   #include <rocprim/rocprim.hpp>
   #include <hipcub/hipcub.hpp>
   #include <thrust/device_vector.h>

If you don't use the CMake targets and add include paths yourself, use ``-Iinstall_prefix/include/hipccl``.

``-Iinstall_prefix/include`` with ``#include <rocprim/...>`` only works while the temporary compatibility symlinks are installed.
See :ref:`install-legacy-symlinks`.

Migrate from standalone packages
================================

1. Remove the standalone ``rocprim``, ``hipcub``, and ``rocthrust`` development packages and install the ``amdrocm-hipccl-dev`` (Debian-based) or ``amdrocm-hipccl-devel`` (RPM-based) package.
   See :ref:`install-upgrading`.
2. Keep your existing ``find_package(rocprim)``, ``find_package(hipcub)``, and ``find_package(rocthrust)`` calls, or replace them with a single ``find_package(hipccl)``.
3. Remove any hard-coded ``-Iinstall_prefix/include`` that relied on the old flat header locations.
   Use the CMake targets, or ``-Iinstall_prefix/include/hipccl``.

.. note::

   The compatibility symlinks are temporary and will be removed in a future release.
   Don't depend on them.
