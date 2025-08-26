.. meta::
  :description: hipSPARSE installation guide for Linux
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, install

.. _hipsparse_building:

*******************************************
Installing and building hipSPARSE for Linux
*******************************************

This topic explains how to install and build the hipSPARSE library on Linux by using prebuilt packages or building from source.
For information on installing and building hipSPARSE on Microsoft Windows, see :doc:`hipSPARSE for Windows <./install-windows>`.

Prerequisites
=============

hipSPARSE requires a ROCm enabled platform.

Installing prebuilt packages
=============================

hipSPARSE can be installed from the AMD ROCm repository.
For detailed instructions on installing ROCm, see :doc:`ROCm installation <rocm-install-on-linux:index>`.

To install hipSPARSE on Ubuntu, run these commands:

.. code-block:: shell

   sudo apt-get update
   sudo apt-get install hipsparse

After hipSPARSE is installed, it can be used just like any other library with a C API.
To call hipSPARSE, the header file must be included in the user code.
This means the hipSPARSE shared library becomes a link-time and run-time dependency for the user application.

Building hipSPARSE from source
==============================

It isn't necessary to build hipSPARSE from source because it's ready to use after installing
the prebuilt packages, as described above.
To build hipSPARSE from source, follow the instructions in this section.

To compile and run hipSPARSE, the `AMD ROCm Platform <https://github.com/ROCm/ROCm>`_ is required.
The build also requires the following compile-time dependencies:

*  `rocSPARSE <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse>`_
*  `git <https://git-scm.com/>`_
*  `CMake <https://cmake.org/>`_ (Version 3.5 or later)
*  `GoogleTest <https://github.com/google/googletest>`_ (Optional: only required to build the clients)

Downloading hipSPARSE
-------------------------

The hipSPARSE source code is available from the `hipSPARSE folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.
Download the develop branch using either a sparse checkout or a full clone of the rocm-libraries repository.

To limit your local checkout to only the hipSPARSE project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

.. note::

   To include the rocSPARSE dependencies, set the projects for the sparse checkout using
   ``git sparse-checkout set projects/hipsparse projects/rocsparse``.

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipsparse # add projects/rocsparse to include dependencies
   git checkout develop # or use the branch you want to work with


To download the develop branch for all projects in rocm-libraries, use these commands. This process takes
longer but is recommended for those working with a large number of libraries.

.. code-block:: shell

   git clone -b develop https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/hipsparse


.. note::

   To build ROCm 6.4.3 and earlier, use the hipSPARSE repository at `<https://github.com/ROCm/hipSPARSE>`_.
   For more information, see the documentation associated with the release you want to build.

Building hipSPARSE using the install script
-------------------------------------------

It's recommended to use the ``install.sh`` script to install hipSPARSE.
Here are the steps required to build different packages of the library, including the dependencies and clients.

.. note::

   You can run the ``install.sh`` script from the ``projects/hipsparse`` directory.

Using install.sh to build hipSPARSE with dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table lists the common ways to use ``install.sh`` to build the hipSPARSE dependencies and library.

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``./install.sh -h``", "Print the help information."
   "``./install.sh -d``", "Build the dependencies and library in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of the script, it isn't necessary to rebuild the dependencies."
   "``./install.sh``", "Build the library in your local directory. The script assumes the dependencies are available."
   "``./install.sh -i``", "Build the library, then build and install the hipSPARSE package in ``/opt/rocm/hipsparse``. The script prompts you for ``sudo`` access. This installs hipSPARSE for all users."

Using install.sh to build hipSPARSE with dependencies and clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The clients contains example code and unit tests. Common use cases of ``install.sh`` to build them are listed in the table below.

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``./install.sh -h``", "Print the help information."
   "``./install.sh -dc``", "Build the dependencies, library, and client in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of the script, it isn't necessary to rebuild the dependencies."
   "``./install.sh -c``", "Build the library and client in your local directory. The script assumes the dependencies are available."
   "``./install.sh -idc``", "Build the library, dependencies, and client, then build and install the hipSPARSE package in ``/opt/rocm/hipsparse``. The script prompts you for ``sudo`` access. This installs hipSPARSE and the clients for all users."
   "``./install.sh -ic``", "Build the library and client, then build and install the hipSPARSE package in ``opt/rocm/hipsparse``. The script prompts you for ``sudo`` access. This installs hipSPARSE and the clients for all users"

Building hipSPARSE using individual make commands
--------------------------------------------------

You can build hipSPARSE using the following commands:

.. note::

   Run these commands from the ``projects/hipsparse`` directory.

.. note::

   CMake 3.5 or later is required to build hipSPARSE.

.. code-block:: bash

   # Create and change to build directory
   mkdir -p build/release ; cd build/release

   # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
   cmake ../..

   # Compile hipSPARSE library
   make -j$(nproc)

   # Install hipSPARSE to /opt/rocm
   make install

You can build hipSPARSE with the dependencies and clients using the following commands:

.. note::

   GoogleTest is required to build the hipSPARSE clients.

.. code-block:: shell

   # Install GoogleTest
   mkdir -p build/release/deps ; cd build/release/deps
   cmake ../../../deps
   make -j$(nproc) install

   # Change to build directory
   cd ..

   # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
   cmake ../.. -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_SAMPLES=ON

   # Compile hipSPARSE library
   make -j$(nproc)

   # Install hipSPARSE to /opt/rocm
   make install

Testing hipSPARSE
==============================

You can test the installation by running one of the hipSPARSE examples after successfully compiling the library with the clients.

.. code-block:: shell

      # Navigate to clients binary directory
      cd build/release/clients/staging

      # Execute hipSPARSE example
      ./example_csrmv 1000

Supported targets
==============================

For a list of the currently supported operating systems, see the :doc:`ROCm compatibility matrix <rocm:compatibility/compatibility-matrix>`.
