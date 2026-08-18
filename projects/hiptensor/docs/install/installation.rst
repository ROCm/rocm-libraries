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
      :width: 3

   .. selector-option:: AMD Instinct™
      :value: instinct w=compute
      :width: 3
      :toc-label: AMD Instinct

   .. selector-option:: AMD Radeon™
      :value: radeon w=compute
      :width: 3
      :toc-label: AMD Radeon

   .. selector-option:: AMD Ryzen™
      :value: ryzen w=compute
      :width: 3
      :toc-label: AMD Ryzen

.. selected:: w=compute

   .. selected:: fam=instinct fam=radeon fam=ryzen

      .. selector-dropdown:: Instinct GPU
         :key: gpu
         :show-cond: fam=instinct
         :sort: desc

         .. selector-option:: AMD Instinct MI355X (gfx950)
            :value: mi355x gfx=gfx950

         .. selector-option:: AMD Instinct MI350X (gfx950)
            :value: mi350x gfx=gfx950

         .. selector-option:: AMD Instinct MI350P (gfx950)
            :value: mi350p gfx=gfx950

         .. selector-option:: AMD Instinct MI325X (gfx942)
            :value: mi325x gfx=gfx942

         .. selector-option:: AMD Instinct MI300X (gfx942)
            :value: mi300x gfx=gfx942

         .. selector-option:: AMD Instinct MI300A (gfx942)
            :value: mi300a gfx=gfx942

         .. selector-option:: AMD Instinct MI250X (gfx90a)
            :value: mi250x gfx=gfx90a

         .. selector-option:: AMD Instinct MI250 (gfx90a)
            :value: mi250 gfx=gfx90a

         .. selector-option:: AMD Instinct MI210 (gfx90a)
            :value: mi210 gfx=gfx90a

         .. selector-option:: AMD Instinct MI100 (gfx908)
            :value: mi100 gfx=gfx908

      .. selector-dropdown:: Radeon GPU
         :key: gpu
         :show-cond: fam=radeon
         :sort: desc

         .. selector-option:: AMD Radeon AI PRO R9700S (gfx1201)
            :value: ai-r9700s gfx=gfx1201

         .. selector-option:: AMD Radeon AI PRO R9700 (gfx1201)
            :value: ai-r9700 gfx=gfx1201

         .. selector-option:: AMD Radeon AI PRO R9600D (gfx1201)
            :value: ai-r9600d gfx=gfx1201

         .. selector-option:: AMD Radeon RX 9070 XT (gfx1201)
            :value: rx-9070-xt gfx=gfx1201

         .. selector-option:: AMD Radeon RX 9070 GRE (gfx1201)
            :value: rx-9070-gre gfx=gfx1201

         .. selector-option:: AMD Radeon RX 9070 (gfx1201)
            :value: rx-9070 gfx=gfx1201

         .. selector-option:: AMD Radeon RX 9060 XT LP (gfx1200)
            :value: rx-9060-xt-lp gfx=gfx1200

         .. selector-option:: AMD Radeon RX 9060 XT (gfx1200)
            :value: rx-9060-xt gfx=gfx1200

         .. selector-option:: AMD Radeon RX 9060 (gfx1200)
            :value: rx-9060 gfx=gfx1200

         .. selector-option:: AMD Radeon PRO W7900 Dual Slot (gfx1100)
            :value: w7900-dual-slot gfx=gfx1100

         .. selector-option:: AMD Radeon PRO W7900 (gfx1100)
            :value: w7900 gfx=gfx1100

         .. selector-option:: AMD Radeon PRO W7800 48GB (gfx1100)
            :value: w7800-48gb gfx=gfx1100

         .. selector-option:: AMD Radeon PRO W7800 (gfx1100)
            :value: w7800 gfx=gfx1100

         .. selector-option:: AMD Radeon RX 7900 XTX (gfx1100)
            :value: rx-7900-xtx gfx=gfx1100

         .. selector-option:: AMD Radeon RX 7900 XT (gfx1100)
            :value: rx-7900-xt gfx=gfx1100

         .. selector-option:: AMD Radeon RX 7900 GRE (gfx1100)
            :value: rx-7900-gre gfx=gfx1100

         .. selector-option:: AMD Radeon PRO W7700 (gfx1101)
            :value: w7700 gfx=gfx1101

         .. selector-option:: AMD Radeon RX 7800 XT (gfx1101)
            :value: rx-7800-xt gfx=gfx1101

         .. selector-option:: AMD Radeon RX 7700 XT (gfx1101)
            :value: rx-7700-xt gfx=gfx1101

         .. selector-option:: AMD Radeon RX 7700 (gfx1101)
            :value: rx-7700 gfx=gfx1101

         .. selector-option:: AMD Radeon PRO V710 (gfx1101)
            :value: v710 gfx=gfx1101

         .. selector-option:: AMD Radeon RX 7600 (gfx1102)
            :value: rx-7600 gfx=gfx1102

         .. selector-option:: AMD Radeon PRO W6800 (gfx1030)
            :value: w6800 gfx=gfx1030

         .. selector-option:: AMD Radeon PRO V620 (gfx1030)
            :value: v620 gfx=gfx1030

      .. selector-dropdown:: Ryzen APU
         :key: gpu
         :show-cond: fam=ryzen
         :sort: desc

         .. selector-option:: AMD Ryzen AI Max+ PRO 495 (gfx1151)
            :value: max-plus-pro-495 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max PRO 490 (gfx1151)
            :value: max-pro-490 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max PRO 485 (gfx1151)
            :value: max-pro-485 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max+ PRO 395 (gfx1151)
            :value: max-pro-395 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max PRO 390 (gfx1151)
            :value: max-pro-390 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max PRO 385 (gfx1151)
            :value: max-pro-385 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max PRO 380 (gfx1151)
            :value: max-pro-380 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max+ 395 (gfx1151)
            :value: max-395 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max+ 392 (gfx1151)
            :value: max-392 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max+ 388 (gfx1151)
            :value: max-388 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max 390 (gfx1151)
            :value: max-390 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI Max 385 (gfx1151)
            :value: max-385 gfx=gfx1151

         .. selector-option:: AMD Ryzen AI 9 HX PRO 475 (gfx1150)
            :value: ai-9-hx-pro-475 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 9 HX PRO 470 (gfx1150)
            :value: ai-9-hx-pro-470 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 9 PRO 465 (gfx1150)
            :value: ai-9-pro-465 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 7 PRO 450 (gfx1152)
            :value: ai-7-pro-450 gfx=gfx1152

         .. selector-option:: AMD Ryzen AI 5 PRO 440 (gfx1152)
            :value: ai-5-pro-440 gfx=gfx1152

         .. selector-option:: AMD Ryzen AI 5 PRO 435 (gfx1153)
            :value: ai-5-pro-435 gfx=gfx1153

         .. selector-option:: AMD Ryzen AI 9 HX 475 (gfx1150)
            :value: ai-9-hx-475 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 9 HX 470 (gfx1150)
            :value: ai-9-hx-470 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 9 465 (gfx1150)
            :value: ai-9-465 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 7 450 (gfx1152)
            :value: ai-7-450 gfx=gfx1152

         .. selector-option:: AMD Ryzen AI 9 HX PRO 375 (gfx1150)
            :value: 9-hx-pro-375 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 9 HX PRO 370 (gfx1150)
            :value: 9-hx-pro-370 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 7 PRO 350 (gfx1152)
            :value: ai-7-pro-350 gfx=gfx1152

         .. selector-option:: AMD Ryzen AI 5 PRO 340 (gfx1152)
            :value: ai-5-pro-340 gfx=gfx1152

         .. selector-option:: AMD Ryzen AI 9 HX 375 (gfx1150)
            :value: 9-hx-375 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 9 HX 370 (gfx1150)
            :value: 9-hx-370 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 9 365 (gfx1150)
            :value: 9-365 gfx=gfx1150

         .. selector-option:: AMD Ryzen AI 7 350 (gfx1152)
            :value: ai-7-350 gfx=gfx1152

         .. selector-option:: AMD Ryzen AI 7 345 (gfx1152)
            :value: ai-7-345 gfx=gfx1152

         .. selector-option:: AMD Ryzen AI 5 435 (gfx1153)
            :value: ai-5-435 gfx=gfx1153

         .. selector-option:: AMD Ryzen AI 5 430 (gfx1153)
            :value: ai-5-430 gfx=gfx1153

         .. selector-option:: AMD Ryzen AI 7 445 (gfx1153)
            :value: ai-7-445 gfx=gfx1153

         .. selector-option:: AMD Ryzen AI 5 340 (gfx1152)
            :value: ai-5-340 gfx=gfx1152

         .. selector-option:: AMD Ryzen AI 5 330 (gfx1152)
            :value: ai-5-330 gfx=gfx1152

         .. selector-option:: AMD Ryzen 7 PRO 250 (gfx1103)
            :value: 7-pro-250 gfx=gfx1103

         .. selector-option:: AMD Ryzen 5 PRO 230 (gfx1103)
            :value: 5-pro-230 gfx=gfx1103

         .. selector-option:: AMD Ryzen 5 PRO 220 (gfx1103)
            :value: 5-pro-220 gfx=gfx1103

         .. selector-option:: AMD Ryzen 5 PRO 215 (gfx1103)
            :value: 5-pro-215 gfx=gfx1103

         .. selector-option:: AMD Ryzen 3 PRO 210 (gfx1103)
            :value: 3-pro-210 gfx=gfx1103

         .. selector-option:: AMD Ryzen 9 270 (gfx1103)
            :value: 9-270 gfx=gfx1103

         .. selector-option:: AMD Ryzen 7 260 (gfx1103)
            :value: 7-260 gfx=gfx1103

         .. selector-option:: AMD Ryzen 7 250 (gfx1103)
            :value: 7-250 gfx=gfx1103

         .. selector-option:: AMD Ryzen 5 240 (gfx1103)
            :value: 5-240 gfx=gfx1103

         .. selector-option:: AMD Ryzen 5 230 (gfx1103)
            :value: 5-230 gfx=gfx1103

         .. selector-option:: AMD Ryzen 5 220 (gfx1103)
            :value: 5-220 gfx=gfx1103

         .. selector-option:: AMD Ryzen 3 210 (gfx1103)
            :value: 3-210 gfx=gfx1103

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

Install the ROCm HPC SDK
------------------------

hipTensor is part of the ROCm HPC SDK on Linux. It is not included in the ROCm Core SDK. For the most complete installation, we recommend that developers use the ``amdrocm-hpc-sdk`` meta package.

For instructions, see `Install ROCm HPC SDK <https://rocm.docs.amd.com/en/latest/components/hpc-sdk/install.html>`_. Use the selector panel on that page to view instructions appropriate for your system environment.

Install hipTensor on Linux
--------------------------

.. selected:: i=tar

   The standard ROCm tarball installation includes hipTensor. No
   additional steps are required. For details on ROCm tarball installation,
   refer to `Install AMD ROCm <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ and select Tarball
   installation method from installation environment selector.

.. selected:: i=pkgman

   If you want to install hipTensor without additional ROCm libraries and tools, install the ``amdrocm-hiptensor`` package.

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

   The standard ROCm uninstallation process removes hipTensor. No additional steps
   are required. Refer to the `Uninstalling
   <https://rocm.docs.amd.com/en/latest/install/rocm.html?fam=all&w=compute&os=ubuntu&ubuntu-ver=26.04&i=tar#uninstalling>`_ section
   and select the Tarball installation method.

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

Install a nightly build
-----------------------

The `TheRock <https://github.com/ROCm/TheRock>`_ build system also publishes nightly builds for ROCm and its components, including hipTensor. See `Nightly <https://github.com/ROCm/TheRock#nightly-release-status>`_ release status for details.
