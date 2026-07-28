.. meta::
  :description: hipThreads installation and prerequisites
  :keywords: install, hipThreads, AMD, ROCm, prerequisites, dependencies, requirements

.. _installation:

**********************
Install hipThreads
**********************

Before you begin, verify that your system is supported.
For more information, see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see :doc:`./build-from-source`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

hipThreads is included with the ROCm Core SDK on Linux and Windows.
For the most complete installation on Linux, use the ``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`.
Use the selector panel on that page to view instructions appropriate for your system environment.

.. _install-base:

Install hipThreads as a standalone package on Linux
===================================================

Alternatively, if you want to install hipThreads without the full set of ROCm libraries and tools, install the ``amdrocm-threads`` package.
This is a granular subset of the ROCm Core SDK ``amdrocm-core-sdk`` that provides hipThreads on its own.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to install dependencies and configure GPU access permissions.

2. Install the ``amdrocm-threads`` package that matches the desired ROCm version, development package needs, and AMD GPU architecture.
   Package names use the following format:

   .. code:: shell

      amdrocm-threads-package_suffix

   Where ``package_suffix`` is built from optional parts:

   - ``-dev`` on Debian-based distributions, including Ubuntu, or ``-devel`` on RPM-based distributions, including RHEL and SLES, adds library files and headers.
     Omit this part to install runtime packages only.
   - ``-rocm_version`` selects a specific ROCm Core SDK version.
     Omit it to install the latest available version.
   - ``-gfx_target`` limits the package to one AMD GPU architecture.
     Omit it to install for all supported architectures at the cost of disk space.

   For example, to install the latest hipThreads development package for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code:: shell

            sudo apt install amdrocm-threads-dev

      .. tab-item:: RHEL-based distros

         .. code:: shell

            sudo dnf install amdrocm-threads-devel

      .. tab-item:: SLES

         .. code:: shell

            sudo zypper install amdrocm-threads-devel

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes nightly builds for the ROCm Core SDK and its components, including hipThreads.
See `Nightly release status <https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
