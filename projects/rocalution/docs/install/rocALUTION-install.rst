.. meta::
   :description: Install rocALUTION
   :keywords: rocALUTION, ROCm, library, API, install, linux, HPC SDK

.. _install-rocalution:

******************
Install rocALUTION
******************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`Build rocALUTION from source <./rocALUTION-build-from-source>`.

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

1. Complete the `ROCm installation prerequisites <https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=pkgman>`__ to
   install dependencies and configure GPU access permissions.

2. Install the rocALUTION packages that match your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-rocalution<-dev/-devel><rocm_version><-llvm_target>

   Where:

   * ``<-dev/-devel>`` specifies whether to install the library files and
     headers. Omit this suffix to only install runtime packages.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<rocm_version>`` is the ROCm version to install. Omit this suffix to
     install the latest available version.

   * ``<-llvm_target>`` (starting with ``gfx``) is used if you are installing
     for a single AMD GPU architecture. Omit this to install for all
     architectures at the cost of disk space.

   For example: ``amdrocm-rocalution7.14-gfx950``

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
