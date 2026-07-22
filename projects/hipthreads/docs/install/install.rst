.. meta::
  :description: hipThreads installation and prerequisites
  :keywords: install, hipThreads, AMD, ROCm, prerequisites, dependencies, requirements

.. _installation:

******************
Install hipThreads
******************

Before you begin, verify that your system is supported.
For more information, see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see :doc:`./source-build`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

hipThreads is included with the ROCm Core SDK on Linux and Windows.
For the most complete installation on Linux, we recommend that developers use the ``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`.
Use the selector panel on that page to view instructions appropriate for your system environment.

.. _install-base:

Install hipThreads as a standalone package on Linux
===================================================

Alternatively, if you want to install hipThreads without the full set of ROCm libraries and tools, install the ``amdrocm-threads`` package.
This is a granular subset of the ROCm Core SDK ``amdrocm-core-sdk`` that provides hipThreads on its own.

#. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to install dependencies and configure GPU access permissions.

#. Install the ``amdrocm-threads`` package that matches your desired ROCm version, development package needs, and AMD GPU architecture.
   Package names use the following format:

   .. code-block:: shell-session

      amdrocm-threads<dev/devel><rocm_version>-<llvm_target>

   Where:

   * ``<rocm_version>`` is the ROCm Core SDK version to install.
     Omit this suffix to install the latest available version.

   * ``<dev/devel>`` specifies whether to install the library files and headers.
     Omit this suffix to only install runtime packages.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<llvm_target>`` (starting with ``gfx``) is used if you are installing for a single AMD GPU architecture.
     Omit this to install for all architectures at the cost of disk space.

   For example, to install the latest hipThreads development package for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-threads-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-threads-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-threads-devel

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes nightly builds for the ROCm Core SDK and its components, including hipThreads.
See `Nightly release status <https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
