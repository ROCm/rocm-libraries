.. meta::
  :description: Installing ROCm Performance Primitives  with the package installer
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing, package installer

********************************************************************
Installing ROCm Performance Primitives with the package installer
********************************************************************

There are three ROCm Performance Primitives (RPP) packages available:

``amdrocm-rpp``: The RPP runtime package. This is the basic package that only installs the ``librpp.so`` library.

``amdrocm-rpp-dev`` (Debian) or ``amdrocm-rpp-devel`` (RPM): The RPP development package. This package installs the ``librpp.so`` library, the RPP header files, and the RPP CMake package configuration.

``amdrocm-rpp-test``: A test package that provides CTest to verify the installation.

All the required dependencies are installed when the package installation method is used.

.. note::

    RPP is included in the ``amdrocm-core`` and ``amdrocm-core-sdk`` meta packages, so a standard ROCm installation already provides the RPP runtime and development packages. Only ``amdrocm-rpp-test`` has to be installed explicitly.

Set up the ROCm package repository for your distribution before running these commands. See `TheRock releases guide <https://github.com/ROCm/TheRock/blob/main/RELEASES.md>`_ for repository setup instructions.

Use the following commands to install only the RPP runtime package:

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

        sudo apt install amdrocm-rpp


  .. tab-item:: RHEL

    .. code:: shell

        sudo dnf install amdrocm-rpp


  .. tab-item:: SLES

    .. code:: shell

        sudo zypper install amdrocm-rpp


Use the following commands to install all three RPP packages:

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

        sudo apt install amdrocm-rpp amdrocm-rpp-dev amdrocm-rpp-test


  .. tab-item:: RHEL

    .. code:: shell

        sudo dnf install amdrocm-rpp amdrocm-rpp-devel amdrocm-rpp-test


  .. tab-item:: SLES

    .. code:: shell

        sudo zypper install amdrocm-rpp amdrocm-rpp-devel amdrocm-rpp-test
