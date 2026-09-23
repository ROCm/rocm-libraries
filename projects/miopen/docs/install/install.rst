.. meta::
  :description: Installing MIOpen from a package
  :keywords: MIOpen, ROCm, API, documentation, install, package

************************************
Install MIOpen
************************************

Install MIOpen from a pre-built package for your Linux distribution or from a
pre-compiled kernels package.

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./build-source`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

MIOpen is included with the ROCm Core SDK on Linux and Windows. For the most
complete installation on Linux, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm DNN libraries on Linux
===================================

Alternatively, if you want to install MIOpen as part of the ROCm
DNN package (a subset of the ROCm Core SDK ``amdrocm-core-sdk``) without
additional ROCm libraries and tools, install the ``amdrocm-dnn`` package.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm DNN package that matches your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-dnn<-dev/-devel><rocm_version><-llvm_target>

   Where:

   * ``<-dev/-devel>`` specifies whether to install library files and
     headers. Omit this suffix to only install runtime packages.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   * ``<-llvm_target>`` (starting with ``gfx``) is used if you are installing
     for a single AMD GPU architecture. Omit this suffix to install for all
     architectures at the cost of disk space.

   For example: ``amdrocm-dnn-dev7.13-gfx950``

   Use the following command to install the latest DNN development package
   release for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-dnn-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-dnn-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-dnn-devel

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including MIOpen.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.

Installing using a kernels package
==================================

MIOpen provides an optional pre-compiled kernels package to reduce startup latency. These
precompiled kernels consist of a select set of popular input configurations. This collection of kernels
will continue to expand to include additional coverage.

.. note::

   All compiled kernels are locally cached in the ``$HOME/.cache/miopen/`` folder, so these
   pre-compiled kernels only reduce the startup latency for the first run of a neural network. Pre-compiled
   kernels don't reduce the startup time on subsequent runs.

To install the kernels package for your GPU architecture, use the following command:

.. code:: shell

   apt-get install miopen-hip-<arch>kdb

Where ``<arch>`` is the GPU architecture, for example, ``gfx900``, ``gfx906``, or ``gfx1030``.

.. note::

   If you don't install these packages, it doesn't impact the functioning of MIOpen. This is because MIOpen compiles
   them on the target machine after you run the kernel. However, the compilation step might significantly
   increase the startup time for certain operations.

The ``utils/install_precompiled_kernels.sh`` script provided as part of MIOpen automates the preceding
process. It queries the user machine for the GPU architecture and then installs the appropriate
package. To run it, use the following command:

.. code:: shell

   ./utils/install_precompiled_kernels.sh

The preceding script depends on the ``rocminfo`` package to query the GPU architecture.
