.. meta::
   :description: installation instructions for the hipTensor library
   :keywords: hipTensor, ROCm, library, API, tool, installation

.. _installation:

*****************
Install hipTensor
*****************

You can install hipTensor on Linux from prebuilt packages either as part of
the ROCm HPC SDK, as standalone packages, or from a ROCm Core SDK tarball
release. Before you begin, verify that your operating system and environment
are supported using the installation selector on :doc:`Install ROCm HPC SDK
<rocm:components/hpc-sdk/install>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./source-build`.

.. _install-hiptensor-hpc-sdk:

Install the ROCm HPC SDK
========================

hipTensor is part of the ROCm HPC SDK on Linux. It is not included in the
ROCm Core SDK through the standard ``amdrocm`` metapackage; it is installed
through the opt-in ``amdrocm-hpc-sdk`` metapackage. This is the recommended
installation method for the most complete installation.

For instructions, see :doc:`Install ROCm HPC SDK
<rocm:components/hpc-sdk/install>` and use the selector panel to view
instructions for your system environment. For ROCm core installation, see
:doc:`Install AMD ROCm <rocm:install/rocm>`.

.. _install-hiptensor-linux:

Install hipTensor standalone on Linux
=====================================

Alternatively, you can use the ``amdrocm-hiptensor`` packages to install
hipTensor without the full ROCm HPC SDK.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the hipTensor packages that match your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-hiptensor<rocm_version><-llvm_target>
      amdrocm-hiptensor<-dev/-devel><rocm_version>

   Where:

   * ``<-dev/-devel>`` is the development package suffix (headers and CMake
     configuration files):

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<rocm_version>`` is the ROCm version to install (for example, ``10.2``).
     Omit this suffix to install the latest available version. If you include
     ``<-llvm_target>``, also include ``<rocm_version>`` (for example,
     ``amdrocm-hiptensor10.2-gfx950``).

   * ``<-llvm_target>`` (starting with ``gfx``) is used on the runtime package
     if you are installing for a single AMD GPU architecture. Omit this to
     install for all supported architectures at the cost of disk space. To find
     the LLVM target for an installed GPU, run ``rocminfo | grep gfx``.

   .. dropdown:: Supported GPUs and GFX targets
      :animate: fade-in-slide-down
      :color: success
      :icon: note
      :chevron: down-up
      :open:

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
         * - AMD Ryzen integrated graphics
           - RDNA 3, RDNA 3.5
           - ``gfx1103``, ``gfx1150``, ``gfx1151``, ``gfx1152``, ``gfx1153``

        hipTensor requires matrix core hardware support (AMD CDNA and RDNA 3 or
        later architectures).  Packages for earlier architectures such as ``gfx103x``
        aren't available.

   For example: ``amdrocm-hiptensor10.2-gfx950`` and ``amdrocm-hiptensor-dev10.2``

   Use the following commands to install the latest hipTensor runtime and
   development packages for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-hiptensor amdrocm-hiptensor-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-hiptensor amdrocm-hiptensor-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-hiptensor amdrocm-hiptensor-devel

.. _install-hiptensor-tarball:

Install from a tarball
======================

The standard ROCm Core SDK tarball installation includes hipTensor. No additional steps
are required. For details, see :doc:`Install AMD ROCm <rocm:install/rocm>` and
select the tarball installation method.

The standard ROCm uninstallation process removes hipTensor. No additional
steps are required.

.. _install-hiptensor-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for ROCm and its components, including hipTensor. See
`Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.

.. _verify-hiptensor-install:

Verify your installation
========================

After installing the runtime and development packages (or the ROCm HPC SDK), you
can verify your installation by compiling and running a minimal smoke test.

1. Create a file named ``test_hiptensor.cpp``:

   .. code-block:: cpp

      #include <cstdio>
      #include <hiptensor/hiptensor.h>

      int main() {
          hiptensorHandle_t handle;
          CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
          std::printf("hipTensor version: %zu (HIP runtime: %d)\n",
                      hiptensorGetVersion(), hiptensorGetHiprtVersion());
          CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));
          return 0;
      }

2. Compile and run the program:

   .. code-block:: bash

      hipcc test_hiptensor.cpp -lhiptensor -o test_hiptensor
      ./test_hiptensor

   If hipTensor is installed properly, the program initializes a hipTensor
   handle on the active GPU and prints the installed hipTensor and HIP runtime
   versions.

.. _uninstall-hiptensor:

Uninstall hipTensor
===================

If you installed versioned or architecture-specific packages, substitute the
exact package names used during installation (for example,
``amdrocm-hiptensor10.2-gfx950`` and ``amdrocm-hiptensor-dev10.2`` or
``amdrocm-hiptensor-devel10.2``).

.. tab-set::

   .. tab-item:: Debian-based distros

      .. code-block:: bash

         sudo apt autoremove amdrocm-hiptensor amdrocm-hiptensor-dev

   .. tab-item:: RHEL-based distros

      .. code-block:: bash

         sudo dnf remove amdrocm-hiptensor amdrocm-hiptensor-devel

   .. tab-item:: SLES

      .. code-block:: bash

         sudo zypper remove 'amdrocm-hiptensor*'
