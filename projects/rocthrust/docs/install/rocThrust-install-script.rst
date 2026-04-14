.. meta::
  :description: Build and install rocThrust with the installation script
  :keywords: install, building, rocThrust, AMD, ROCm, source code, installation script, Linux

********************************************************************
Building and installing rocThrust on Linux with the install script
********************************************************************

You can use the ``install`` script to build and install rocThrust on Linux. You can also use `CMake <./rocThrust-install-with-cmake.html>`_ if you want more build and installation options.

:doc:`Clone the rocThrust project <./rocThrust-install-overview>`. The ``install`` script will be located in the ``rocthrust`` root directory.

To build and install rocThrust with the ``install`` script, run:

.. code-block:: shell

  ./install --install

This command will also download and install rocPRIM.

To build rocThrust and generate tar, zip, and debian packages, run:

.. code-block:: shell

  ./install --package

To see a complete list of options, run:

.. code-block:: shell

  ./install --help
