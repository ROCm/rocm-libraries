.. meta::
   :description: Install rocALUTION
   :keywords: rocALUTION, ROCm, library, API, install, linux, HPC SDK

.. _install-rocalution:

******************
Install rocALUTION
******************

Before you begin, verify that your GPU, operating system, and ROCm version are
supported. For the authoritative list of supported GPUs and their LLVM targets,
see the :doc:`ROCm compatibility matrix
<rocm:compatibility/compatibility-matrix>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`Build rocALUTION from source <./rocALUTION-build-from-source>`.

.. _rocalution-supported-gpus:

Supported GPUs
==============

rocALUTION packages are published for the following LLVM targets. To find the
LLVM target for an installed GPU, run ``rocminfo | grep gfx``.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - GPU series
     - Architecture
     - LLVM target
   * - AMD Instinct MI350 series
     - CDNA 4
     - ``gfx950``
   * - AMD Instinct MI300 series
     - CDNA 3
     - ``gfx942``
   * - AMD Instinct MI200 series
     - CDNA 2
     - ``gfx90a``
   * - AMD Instinct MI100 series
     - CDNA
     - ``gfx908``
   * - AMD Radeon AI PRO R9000 / RX 9000 series
     - RDNA 4
     - ``gfx1200``, ``gfx1201``
   * - AMD Radeon PRO W7000 / RX 7000 / PRO V series
     - RDNA 3
     - ``gfx1100``, ``gfx1101``, ``gfx1102``
   * - AMD Radeon PRO W6000 series
     - RDNA 2
     - ``gfx1030``
   * - AMD Ryzen integrated graphics
     - RDNA 3, RDNA 3.5
     - ``gfx1103``, ``gfx1150``, ``gfx1151``, ``gfx1152``, ``gfx1153``

.. _install-rocalution-hpc-sdk:

Install the ROCm HPC SDK
========================

rocALUTION is part of the ROCm HPC SDK on Linux. It is not included in the
ROCm Core SDK, and it is not installed by the default ``amdrocm`` metapackage.
For the most complete installation, install the ``amdrocm-hpc-sdk`` metapackage.

For instructions on installing the ROCm HPC SDK, see
:doc:`Install ROCm HPC SDK <rocm:components/hpc-sdk/install>`. For general
ROCm installation, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on the ROCm install page to view instructions appropriate for
your system environment.

.. _install-rocalution-linux:

Install rocALUTION on Linux
===========================

Alternatively, you can use the ``amdrocm-rocalution`` packages to install
rocALUTION without the full ROCm HPC SDK.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the rocALUTION packages that match your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-rocalution<rocm_version><-llvm_target>
      amdrocm-rocalution<-dev/-devel><rocm_version>

   Where:

   * ``<-dev/-devel>`` is the development package suffix (headers and CMake
     configuration files):

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<rocm_version>`` is the ROCm version to install. Omit this suffix to
     install the latest available version (required when ``<-llvm_target>`` is
     specified).

   * ``<-llvm_target>`` (starting with ``gfx``) is used on the runtime package
     if you are installing for a single AMD GPU architecture. Omit this to
     install for all architectures at the cost of disk space. See
     :ref:`Supported GPUs <rocalution-supported-gpus>` for the available
     targets.

   For example: ``amdrocm-rocalution10.2-gfx950`` and ``amdrocm-rocalution-dev10.2``

   The following packages are available for each ROCm version:

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - Package
        - Contents
      * - ``amdrocm-rocalution``
        - Metapackage that installs the runtime for every supported GPU
          architecture. Depends on the versioned runtime package, which in
          turn depends on all per-architecture packages.
      * - ``amdrocm-rocalution-dev``, ``amdrocm-rocalution-devel``
        - Headers, CMake configuration files, and development symlinks.
          Use ``-dev`` on Debian-based distributions and ``-devel`` on
          RPM-based distributions.
      * - ``amdrocm-rocalution<rocm_version>-<llvm_target>``
        - Device code for a single GPU architecture, such as
          ``amdrocm-rocalution10.2-gfx950``. Also installs the matching
          rocBLAS, rocSPARSE, and rocRAND packages for that architecture.
      * - ``amdrocm-rocalution-host<rocm_version>``
        - Architecture-independent host library. Installed as a dependency of
          every per-architecture package.

   Use the following commands to install the latest rocALUTION runtime and
   development packages for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-rocalution amdrocm-rocalution-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-rocalution amdrocm-rocalution-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-rocalution amdrocm-rocalution-devel

.. _uninstall-rocalution:

Uninstall rocALUTION
====================

If you installed versioned or architecture-specific packages, substitute the
exact package names used during installation (for example,
``amdrocm-rocalution10.2-gfx950`` and ``amdrocm-rocalution-dev10.2`` or
``amdrocm-rocalution-devel10.2``).

.. tab-set::

   .. tab-item:: Debian-based distros

      .. code-block:: bash

         sudo apt autoremove amdrocm-rocalution amdrocm-rocalution-dev

   .. tab-item:: RHEL-based distros

      .. code-block:: bash

         sudo dnf remove amdrocm-rocalution amdrocm-rocalution-devel

   .. tab-item:: SLES

      .. code-block:: bash

         sudo zypper remove 'amdrocm-rocalution*'

.. _install-rocalution-tarball:

Install from a tarball
======================

The standard ROCm tarball installation includes rocALUTION. No additional steps
are required. For details, see :doc:`Install AMD ROCm <rocm:install/rocm>` and
select the tarball installation method.

The standard ROCm uninstallation process removes rocALUTION. No additional
steps are required.

.. _install-rocalution-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for ROCm and its components, including rocALUTION. See
`Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
