.. meta::
  :description: hipCCL installation
  :keywords: install, hipCCL, AMD, ROCm, installation, prerequisites, dependencies, requirements

.. _installation:

******************
Install hipCCL
******************

Before you begin, make sure your system is supported.
For more information, see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see :doc:`./build`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

hipCCL is included with the ROCm Core SDK on Linux and Windows.
On Linux, the ``amdrocm-core-sdk`` meta package installs hipCCL with the ROCm Core SDK.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`.
Use the selector panel on that page to view instructions for your system.

.. _install-base:

Install hipCCL on Linux
=======================

To install hipCCL without additional ROCm libraries and tools, install the ``amdrocm-hipccl`` package.
The package bundles together rocPRIM, hipCUB, and rocThrust.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to install dependencies and configure GPU access permissions.

2. Install the hipCCL package that matches your ROCm version, development package needs, and AMD GPU architecture.

   Package names start with ``amdrocm-hipccl``.
   Optional suffixes select the development files, the ROCm version, and one AMD GPU architecture.

   - ``dev_suffix`` selects the library files and headers.
     Use ``-dev`` on Debian-based distributions, including Ubuntu.
     Use ``-devel`` on RPM-based distributions, including RHEL and SLES.
     Omit ``dev_suffix`` to install runtime packages only.
   - ``rocm_version`` is the ROCm Core SDK version to install.
     Omit ``rocm_version`` to install the latest available version.
   - ``llvm_target`` is one AMD GPU architecture.
     The value starts with ``gfx``.
     Omit ``llvm_target`` to install every supported architecture.
     That install uses more disk space.

   For example, to install the latest hipCCL development package for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code:: shell

            sudo apt install amdrocm-hipccl-dev

      .. tab-item:: RHEL-based distros

         .. code:: shell

            sudo dnf install amdrocm-hipccl-devel

      .. tab-item:: SLES

         .. code:: shell

            sudo zypper install amdrocm-hipccl-devel

.. _install-legacy-symlinks:

Header path compatibility
=========================

hipCCL installs headers under ``install_prefix/include/hipccl/component_name``.
For example, rocPRIM headers install under ``install_prefix/include/hipccl/rocprim``.
Older standalone rocPRIM, hipCUB, and rocThrust installs used ``install_prefix/include/component_name``.

During the deprecation window, the ``amdrocm-hipccl`` package also installs a compatibility symlink at each legacy location.
Existing code that uses ``#include <rocprim/...>`` with ``-Iinstall_prefix/include`` continues to work unchanged.
The symlink is temporary and will be removed.
Details are in :doc:`../conceptual/hipccl-layout`.
Update include paths to ``-Iinstall_prefix/include/hipccl``, or use ``find_package(hipccl)`` or ``find_package(rocprim)``, before the symlink is removed.

.. note::

   On Windows the compatibility paths are always installed as copies of the headers, never as symlinks.
   The copies are only refreshed when hipCCL is installed again.
   The compatibility paths on Windows might be absent from a given binary distribution and shouldn't be relied upon.

.. _install-upgrading:

Upgrading from standalone rocPRIM, hipCUB, or rocThrust
=======================================================

hipCCL supersedes the standalone ``rocprim``, ``hipcub``, and ``rocthrust`` development packages.
The standalone packages aren't removed automatically. They must be removed manually to avoid conflicts.


.. tab-set::

   .. tab-item:: Debian-based distros

      .. code:: shell

         sudo apt remove rocprim-dev hipcub-dev rocthrust-dev
         sudo apt install amdrocm-hipccl-dev

      ``sudo apt full-upgrade`` permits package removals and resolves the conflict in one step.

   .. tab-item:: RHEL-based distros

      .. code:: shell

         sudo dnf remove rocprim-devel hipcub-devel rocthrust-devel
         sudo dnf install amdrocm-hipccl-devel

   .. tab-item:: SLES

      .. code:: shell

         sudo zypper remove rocprim-devel hipcub-devel rocthrust-devel
         sudo zypper install amdrocm-hipccl-devel


hipCCL ships each component's CMake package configuration, so ``find_package(rocprim)``, ``find_package(hipcub)``, and ``find_package(rocthrust)`` continue to resolve.

.. _install-nightly:

Install a nightly build
=======================

`TheRock <https://github.com/ROCm/TheRock>`__ also publishes nightly builds for the ROCm Core SDK and its components, including hipCCL.
Details are in `Nightly release status <https://github.com/ROCm/TheRock#nightly-release-status>`__.
