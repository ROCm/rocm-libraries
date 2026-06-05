.. meta::
   :description: Installation instructions for hipSPARSELt
   :keywords: lib, hipsparselt, blas, algorithm, install, sdk, rocm, math

.. _installation:

*******************
Install hipSPARSELt
*******************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./build`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

hipSPARSELt is included with the ROCm Core SDK on Linux. For the most complete
installation, we recommend that developers use the ``amdrocm-core-sdk`` meta
package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm BLAS libraries on Linux
====================================

Alternatively, if you want to install hipSPARSELt as part of the ROCm
BLAS package (a subset of the ROCm Core SDK ``amdrocm-core-sdk``) without
additional ROCm libraries and tools, install the ``amdrocm-blas`` package.
This includes hipSPARSELt, other BLAS libraries, and runtime components.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm BLAS package that matches your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-blas<-dev/-devel><rocm_version><-llvm_target>

   Where:

   * ``<-dev/-devel>`` specifies whether to install the library files and
     headers. Omit this suffix to only install runtime packages.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   * ``<-llvm_target>`` (starting with ``gfx``) is used if you are installing
     for a single AMD GPU architecture. Omit this to install for all
     architectures at the cost of disk space.

   For example: ``amdrocm-blas-dev7.13-gfx950``

   Use the following command to install the latest BLAS development package
   release for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-blas-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-blas-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-blas-devel

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including hipSPARSELt.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
