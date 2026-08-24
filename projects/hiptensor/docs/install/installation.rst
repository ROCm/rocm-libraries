.. meta::
   :description: installation instructions for the hipTensor library
   :keywords: hipTensor, ROCm, library, API, tool, installation

.. _installation:

*****************
Install hipTensor
*****************

hipTensor is supported on AMD Instinct and Radeon GPUs by ROCm. See the `ROCm compatibility matrix
<https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html>`__ for support information.

Before installing hipTensor, make sure your system meets the ROCm hardware,
software, and driver requirements.

Install the ROCm HPC SDK
------------------------

hipTensor is part of the ROCm HPC SDK on Linux. It is not included in the ROCm Core SDK. For the most complete installation, we recommend that developers use the ``amdrocm-hpc-sdk`` meta package.

For instructions, see `Install ROCm HPC SDK <https://rocm.docs.amd.com/en/latest/components/hpc-sdk/install.html>`_. Use the selector panel on that page to view instructions appropriate for your system environment.

Install hipTensor on Linux
--------------------------

Alternatively, if you want to install hipTensor without additional ROCm libraries and tools, install the ``amdrocm-hiptensor`` package.

1. Complete the `ROCm installation prerequisites
   <https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=pkgman#prerequisites>`_ to install dependencies and configure GPU access permissions.

2. Install the hipTensor package that matches your desired ROCm version,
   development package needs, and AMD GPU architecture. Package names use the
   following format:

   .. code-block:: shell-session

      amdrocm-hiptensor<rocm_version><-llvm_target> amdrocm-hiptensor<-dev/-devel><rocm_version>

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

   For example: ``amdrocm-hiptensor7.14-gfx950 amdrocm-hiptensor-dev7.14``

   Use the following command to install the desired hipTensor development package
   release for supported GPU architectures:  

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-hiptensor<rocm_version><-llvm_target> amdrocm-hiptensor-dev<rocm_version>

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-hiptensor<rocm_version><-llvm_target> amdrocm-hiptensor-devel<rocm_version>

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-hiptensor<rocm_version><-llvm_target> amdrocm-hiptensor-devel<rocm_version>

Uninstall hipTensor
-------------------

Replace ``<rocm_version>`` with the ROCm version used during installation and
``<-llvm_target>`` with the GPU architecture used during installation.

.. tab-set::

   .. tab-item:: Debian-based distros

      .. code-block:: bash

         sudo apt autoremove amdrocm-hiptensor<rocm_version><-llvm_target> amdrocm-hiptensor-dev<rocm_version>

   .. tab-item:: RHEL-based distros

      .. code-block:: bash

         sudo dnf remove amdrocm-hiptensor<rocm_version><-llvm_target> amdrocm-hiptensor-devel<rocm_version>

   .. tab-item:: SLES

      .. code-block:: bash

         sudo zypper remove amdrocm-hiptensor<rocm_version><-llvm_target> amdrocm-hiptensor-devel<rocm_version>

Install from Tarball
--------------------

Install
^^^^^^^

The standard ROCm tarball installation includes hipTensor. No additional steps
are required. For details on ROCm tarball installation, refer to `Install AMD
ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=tar>`_ and select the
Tarball installation method.

Uninstall
^^^^^^^^^

The standard ROCm uninstallation process removes hipTensor. No additional steps
are required. Refer to the `Uninstalling
<https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=tar#uninstalling>`_ section
and select the Tarball installation method.

Install a nightly build
-----------------------

The `TheRock <https://github.com/ROCm/TheRock>`_ build system also publishes nightly builds for ROCm and its components, including hipTensor. See `Nightly <https://github.com/ROCm/TheRock#nightly-release-status>`_ release status for details.
