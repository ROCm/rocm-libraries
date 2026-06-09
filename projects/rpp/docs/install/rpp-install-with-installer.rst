.. meta::
  :description: Installing ROCm Performance Primitives  with the package installer
  :keywords: rpp, ROCm Performance Primitives, ROCm, documentation, installing, package installer

********************************************************************
Installing ROCm Performance Primitives with the package installer
********************************************************************

There are three ROCm Performance Primitives (RPP) packages available:

``rpp``: The RPP runtime package. This is the basic package that only installs the ``librpp.so`` library.

``rpp-dev``: The RPP development package. This package installs the ``librpp.so`` library, the RPP header files, and the RPP samples.

``rpp-test``: A test package that provides CTest to verify the installation.

All the required dependencies are installed when the package installation method is used.

Use the following commands to install only the RPP runtime package:

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

        sudo apt install rpp

  
  .. tab-item:: RHEL

    .. code:: shell

        sudo yum install rpp 


  .. tab-item:: SLES

    .. code:: shell
  
        sudo zypper install rpp


Use the following commands to install all three RPP packages:

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

        sudo apt install rpp rpp-dev rpp-test

  
  .. tab-item:: RHEL

    .. code:: shell

        sudo yum install rpp rpp-devel rpp-test


  .. tab-item:: SLES

    .. code:: shell
  
        sudo zypper install rpp rpp-devel rpp-test

