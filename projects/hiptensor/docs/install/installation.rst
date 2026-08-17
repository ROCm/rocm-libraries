:selector-toc2: Installation environment
:selector-toc2-icon: fa-solid fa-computer

.. meta::
   :description: installation instructions for the hipTensor library
   :keywords: hipTensor, ROCm, library, API, tool, installation

.. _installation:

*****************
Install hipTensor
*****************

hipTensor is supported on AMD Instinct and Radeon GPUs by ROCm. See the `ROCm compatibility matrix
<https://rocm.docs.amd.com/en/docs-7.14.0/compatibility/compatibility-matrix.html>`__ for support information.

.. selector:: Device family
   :key: fam

   .. selector-option:: All
      :value: all w=compute
      :width: 6

   .. selector-option:: AMD Instinct™
      :value: instinct w=compute
      :width: 6
      :toc-label: AMD Instinct

.. include:: /install/include/gpu-selector.rst

.. selected:: fam=instinct

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi355x gpu=mi350x gpu=mi325x

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 20%

      .. selector-option:: Debian
         :value: debian
         :width: 20%

      .. selector-option:: RHEL
         :value: rhel
         :width: 20%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 20%

      .. selector-option:: SLES
         :value: sles
         :width: 20%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi350p

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 25%

      .. selector-option:: Debian
         :value: debian
         :width: 25%

      .. selector-option:: RHEL
         :value: rhel
         :width: 25%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 25%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi300x

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: Debian
         :value: debian
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 4

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 4

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi300a

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 20%

      .. selector-option:: Debian
         :value: debian
         :width: 20%

      .. selector-option:: RHEL
         :value: rhel
         :width: 20%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 20%

      .. selector-option:: SLES
         :value: sles
         :width: 20%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi250x gpu=mi250

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 25%

      .. selector-option:: Debian
         :value: debian
         :width: 25%

      .. selector-option:: RHEL
         :value: rhel
         :width: 25%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 25%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi210

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi100

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

.. selector:: Linux distribution
   :key: os
   :show-cond: fam=radeon

   .. selector-option:: Ubuntu
      :value: ubuntu
      :width: 6

   .. selector-option:: RHEL
      :value: rhel
      :width: 6
      :toc-label: Red Hat Enterprise Linux

.. selector:: Operating system
   :key: os
   :show-cond: fam=ryzen

   .. selector-option:: Ubuntu
      :value: ubuntu
      :width: 12

.. selected:: fam=all

   .. selector:: Operating system
      :key: os

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: Debian
         :value: debian
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 4

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 4

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

.. selector:: Installation method
   :show-cond: os=ubuntu os=debian
   :key: i

   .. selector-option:: apt
      :value: pkgman
      :width: 6

   .. selector-option:: Tarball
      :value: tar
      :width: 6

.. selector:: Installation method
   :show-cond: os=rhel os=oracle-linux os=rocky-linux
   :key: i

   .. selector-option:: dnf
      :value: pkgman
      :width: 6

   .. selector-option:: Tarball
      :value: tar
      :width: 6

.. selector:: Installation method
   :show-cond: os=sles
   :key: i

   .. selector-option:: zypper
      :value: pkgman
      :width: 6

   .. selector-option:: Tarball
      :value: tar
      :width: 6

----

Before installing hipTensor, make sure your system meets the ROCm hardware,
software, and driver requirements. For instructions, see `Install AMD ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_. Use the selector panel on that page to view instructions appropriate for your system
environment.

Install hipTensor
-----------------

.. selected:: i=tar

   The standard ROCm tarball installation includes hipTensor. No
   additional steps are required. For details on ROCm tarball installation,
   refer to `Install AMD ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ and select Tarball
   installation method from installation environment selector.

.. selected:: i=pkgman

   1. `Install AMD ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_. Remember to complete the `ROCm
      installation prerequisites <https://rocm.docs.amd.com/en/latest/install/rocm.html#prerequisites>`_ to install dependencies
      and configure GPU access permissions.

   2. Use the following command to install hipTensor:

      .. selected:: fam=all

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx950

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx950 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx950 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx950 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx942

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx942 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx942 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx942 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx90a

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx90a amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx90a amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx90a amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx908

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx908 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx908 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx908 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1200

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1200 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1200 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1200 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1201

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1201 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1201 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1201 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1100

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1100 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1100 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1100 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1101

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1101 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1101 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1101 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1102

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1102 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1102 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1102 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1103

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1103 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1103 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1103 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1030

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1030 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1030 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1030 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1151

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1151 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1151 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1151 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1150

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1150 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1150 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1150 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1152

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1152 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1152 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1152 amdrocm-hiptensor-devel7.14

      .. selected:: gfx=gfx1153

         .. selected:: os=ubuntu os=debian

            .. code-block:: bash

               sudo apt install amdrocm-hiptensor7.14-gfx1153 amdrocm-hiptensor-dev7.14

         .. selected:: os=rhel os=rocky-linux os=oracle-linux

            .. code-block:: bash

               sudo dnf install amdrocm-hiptensor7.14-gfx1153 amdrocm-hiptensor-devel7.14

         .. selected:: os=sles

            .. code-block:: bash

               sudo zypper install amdrocm-hiptensor7.14-gfx1153 amdrocm-hiptensor-devel7.14

Uninstall hipTensor
-------------------

.. selected:: i=tar

   The standard ROCm uninstallation process can be followed to uninstall hipTensor. No additional
   steps are required to remove hipTensor separately. Refer to the `Uninstalling <https://rocm.docs.amd.com/en/latest/install/rocm.html#uninstalling>`_
   section and select Tarball from the installation environment selector.

.. selected:: i=pkgman

   .. selected:: fam=all

      Use the following command to uninstall hipTensor for all GPU architectures:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx950

      Use the following command to uninstall hipTensor for your ``gfx950`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx950 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx950 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx950 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx942

      Use the following command to uninstall hipTensor for your ``gfx942`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx942 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx942 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx942 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx90a

      Use the following command to uninstall hipTensor for your ``gfx90a`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx90a amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx90a amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx90a amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx908

      Use the following command to uninstall hipTensor for your ``gfx908`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx908 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx908 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx908 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1200

      Use the following command to uninstall hipTensor for your ``gfx1200`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1200 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1200 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1200 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1201

      Use the following command to uninstall hipTensor for your ``gfx1201`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1201 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1201 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1201 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1100

      Use the following command to uninstall hipTensor for your ``gfx1100`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1100 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1100 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1100 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1101

      Use the following command to uninstall hipTensor for your ``gfx1101`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1101 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1101 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1101 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1102

      Use the following command to uninstall hipTensor for your ``gfx1102`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1102 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1102 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1102 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1103

      Use the following command to uninstall hipTensor for your ``gfx1103`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1103 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1103 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1103 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1030

      Use the following command to uninstall hipTensor for your ``gfx1030`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1030 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1030 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1030 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1151

      Use the following command to uninstall hipTensor for your ``gfx1151`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1151 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1151 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1151 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1150

      Use the following command to uninstall hipTensor for your ``gfx1150`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1150 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1150 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1150 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1152

      Use the following command to uninstall hipTensor for your ``gfx1152`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1152 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1152 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1152 amdrocm-hiptensor-devel7.14

   .. selected:: gfx=gfx1153

      Use the following command to uninstall hipTensor for your ``gfx1153`` GPU:

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

             sudo apt autoremove amdrocm-hiptensor7.14-gfx1153 amdrocm-hiptensor-dev7.14

      .. selected:: os=rhel os=rocky-linux os=oracle-linux

         .. code-block:: bash

             sudo dnf remove amdrocm-hiptensor7.14-gfx1153 amdrocm-hiptensor-devel7.14

      .. selected:: os=sles

         .. code-block:: bash

             sudo zypper remove amdrocm-hiptensor7.14-gfx1153 amdrocm-hiptensor-devel7.14
