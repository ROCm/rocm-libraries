.. meta::
   :description: Installation instructions for hipRAND
   :keywords: lib, hiprand, rand, rng, random, algorithm, install, sdk, rocm

.. _installation:

***************
Install hipRAND
***************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./build`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

hipRAND is included with the ROCm Core SDK on Linux and Windows. For the most
complete installation on Linux, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm rand libraries on Linux
====================================

Alternatively, if you want to install hipRAND as part of the ROCm
rand package (a subset of the ROCm Core SDK ``amdrocm-core-sdk``) without
additional ROCm libraries and tools, install the ``amdrocm-rand`` package.
This includes hipRAND and rocRAND.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm rand package that matches your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-rand<-dev/-devel><rocm_version><-llvm_target>

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

   For example: ``amdrocm-rand-dev7.13-gfx950``

   Use the following command to install the latest development package release
   for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-rand-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-rand-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-rand-devel

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including hipRAND.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
