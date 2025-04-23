.. meta::
  :description: Installing rocAL using the package installer
  :keywords: rocAL, ROCm, API, install, installation, package installer

.. _install:

********************************************************************
Installing rocAL with the package installer
********************************************************************

Three rocAL packages are available:

* ``rocal``: The rocAL runtime package. This is the basic rocAL package that only provides dynamic libraries. It must always be installed.

* ``rocal-dev``: The rocAL development package. This package installs a full suite of libraries, header files, and samples. This package needs to be installed to use samples.
* ``rocal-test``: A test package that provides a CTest to verify the installation. 

All the required prerequisites are installed when the package installation method is used.

.. note::
  
  | TurboJPEG must be installed manually on SLES. 
  | To use FFMPeg on SLES and RedHat, the ``FFMPeg-dev`` package must be installed manually.


Basic installation
========================================

Use the following commands to install only the rocAL runtime package:

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install rocal

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocal

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install rocal


Complete installation
========================================

Use the following commands to install ``rocal``, ``rocal-dev``, and ``rocal-test``:

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt-get install rocal rocal-dev rocal-test

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocal rocal-devel rocal-test

  .. tab-item:: SLES

    .. code:: shell

    sudo zypper install rocal rocal-devel rocal-test


The rocAL test package will install a CTest module. Use the following steps to test the installation:

.. code-block:: shell

  mkdir rocAL-test
  cd rocAL-test
  cmake /opt/rocm/share/rocal/test/
  ctest -VV

