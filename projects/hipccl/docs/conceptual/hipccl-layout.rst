.. meta::
  :description: hipCCL project layout, the hipccl3 and hipccl2 build layouts, the unified include directory, and versioning
  :keywords: hipCCL, ROCm, layout, hipccl2, hipccl3, CMake

.. _hipccl-layout:

**********************
hipCCL project layout
**********************

The hipCCL sources are under ``projects/hipccl`` in the `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_ repository.
The directory contains two build layouts, ``hipccl3`` and ``hipccl2``, and a router that selects between them.

* ``hipccl3``: The default layout, and the layout under active development. Its copies of rocPRIM, hipCUB, and rocThrust stay in sync with each project's upstream development.

* ``hipccl2``: A frozen compatibility snapshot of rocPRIM, hipCUB, and rocThrust from before unification. It provides a stable fallback during migration. It's not kept in sync with upstream the way ``hipccl3`` is.

Both layouts expose the same CMake options: ``HIPCCL_BUILD_ROCPRIM``, ``HIPCCL_BUILD_HIPCUB``, ``HIPCCL_BUILD_ROCTHRUST``, and ``HIPCCL_BUILD_ALL``.
Both use the same ``hipccl`` CPack package and the same ``find_package(hipccl)`` entry point.

The layout selector
===================

``projects/hipccl/CMakeLists.txt`` redirects one ``cmake`` invocation to one layout or the other.
A layout built from its own directory behaves the same way.

.. code:: shell

   # Default, HIPCCL_BUILD_LEGACY=OFF: build the unified hipccl3 project.
   cmake -S projects/hipccl -B build

   # Build the legacy hipccl2 layout instead.
   cmake -S projects/hipccl -B build -DHIPCCL_BUILD_LEGACY=ON

The selector commands configure either layout from ``projects/hipccl``.
``cmake ..`` from inside ``hipccl3`` or ``hipccl2`` configures that layout on its own.

Unified include layout
======================

All hipCCL headers install under ``install_prefix/include/hipccl/component_name``.
For example, rocPRIM headers install under ``/opt/rocm/include/hipccl/rocprim``, and Thrust headers install under ``/opt/rocm/include/hipccl/thrust``.
A component doesn't install a ``install_prefix/include/component_name`` path of its own.
That avoids a collision between hipCCL's bundled copy and a separately installed copy of rocPRIM, hipCUB, or rocThrust that uses the same install prefix.

During the deprecation window, ``HIPCCL_INSTALL_LEGACY_HEADER_SYMLINKS`` also installs a compatibility symlink at the old flat location for each built component.
The default is ``ON``.
Code that uses ``#include <rocprim/...>`` with ``-Iinstall_prefix/include`` keeps working without changes.
The symlink is a temporary migration aid.
It will be turned off and removed once consumers use the ``include/hipccl/component_name`` paths.

Versioning
==========

hipCCL is versioned independently of its individual components.
rocPRIM, hipCUB, and rocThrust each keep their own version, and each keeps its own ``find_package`` version.
