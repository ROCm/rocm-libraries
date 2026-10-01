.. meta::
  :description: hipCCL installation
  :keywords: install, hipCCL, AMD, ROCm, installation, prerequisites, dependencies, requirements

.. _installation:

**************
Install hipCCL
**************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./build`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

hipCCL is included with the ROCm Core SDK on Linux and Windows. For the most
complete installation on Linux, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install hipCCL on Linux
=======================

Alternatively, if you want to install hipCCL (rocPRIM, hipCUB, and rocThrust
bundled together as a single package) without additional ROCm libraries and
tools, install the ``amdrocm-ccl`` package.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm CCL package that matches your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-ccl<dev/devel><rocm_version>-<llvm_target>

   Where:

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   * ``<dev/devel>`` specifies whether to install the library files and
     headers. Omit this suffix to only install runtime packages.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<llvm_target>`` (starting with ``gfx``) is used if you are installing
     for a single AMD GPU architecture. Omit this to install for all
     architectures at the cost of disk space.

   For example, to install the latest ROCm CCL development package release for
   supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-ccl-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-ccl-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-ccl-devel

.. _install-legacy-symlinks:

Header path compatibility
==========================

hipCCL installs headers under ``<prefix>/include/hipccl/<component>`` (for
example ``include/hipccl/rocprim``), rather than the flat
``<prefix>/include/<component>`` paths older, standalone rocPRIM/hipCUB/
rocThrust installs used. During the deprecation window, the ``amdrocm-ccl``
package also installs a compatibility symlink at each old, flat location, so
existing code doing e.g. ``#include <rocprim/...>`` against a plain
``-I<prefix>/include`` continues to work unchanged. This is temporary and
will eventually be removed - see :doc:`../conceptual/hipccl-layout` for
details, and plan to update include paths to
``-I<prefix>/include/hipccl`` (or the equivalent CMake
``find_package(hipccl)``/``find_package(rocprim)`` usage) ahead of removal.

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including hipCCL.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
