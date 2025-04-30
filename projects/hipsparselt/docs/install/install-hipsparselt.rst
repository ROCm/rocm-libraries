.. meta::
   :description: Installing hipSPARSELt on Linux
   :keywords: hipSPARSELt, ROCm, install, Linux

.. _install-linux:

**************************************************************************
Installing and building hipSPARSELt
**************************************************************************

This topic explains how to install and build the hipSPARSELt library.

Prerequisites
====================================

hipSPARSELt requires a :doc:`ROCm-enabled platform <rocm:index>` and the
:doc:`hipSPARSE library <hipsparse:index>` for the header file.

Installing prebuilt packages
==================================


hipSPARSELt can be installed from the AMD ROCm repository.
For detailed instructions on installing ROCm, see :doc:`ROCm installation <rocm-install-on-linux:index>`.

To install hipSPARSELt on Ubuntu, run these commands:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install hipsparselt


After hipSPARSELt is installed, it can be used just like any other library with a C API.
To call hipSPARSELt, the header file must be included in the user code.
This means the hipSPARSELt shared library becomes a link-time and run-time dependency for the user application.

Building hipSPARSELt from source
======================================================

It isn't necessary to build hipSPARSELt from source because it's ready to use after installing
the prebuilt packages, as described above.
To build hipSPARSELt from source, follow the instructions in this section.

To compile and run hipSPARSELt, the `ROCm platform <https://github.com/ROCm/ROCm>`_ is required.
The build also requires the following compile-time dependencies:

*  `hipSPARSE <https://github.com/ROCm/hipSPARSE>`_
*  `git <https://git-scm.com/>`_
*  `CMake <https://cmake.org/>`_ (Version 3.5 or later)
*  `GoogleTest <https://github.com/google/googletest>`_ (Optional: only required to build the clients)

Downloading hipSPARSELt
--------------------------------------------------------------------------------------

The hipSPARSELt source code is available from the
`hipSPARSELt GitHub <https://github.com/ROCm/hipSPARSELt>`_.

Download the develop branch using these commands:

.. code-block:: bash

   git clone -b develop https://github.com/ROCm/hipSPARSELt.git
   cd hipSPARSELt


Building hipSPARSELt using the install script
----------------------------------------------

It's recommended to use the ``install.sh`` script to install hipSPARSELt.
Here are the steps required to build different packages of the library, including the dependencies and clients.

Using install.sh to build hipSPARSELt with dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following table lists the common ways to use ``install.sh`` to build the hipSPARSELt dependencies and library.

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``./install.sh -h``", "Print the help information."
   "``./install.sh -d``", "Build the dependencies and library in your local directory.  The ``-d`` flag only needs to be used once. For subsequent invocations of the script, it isn't necessary to rebuild the dependencies."
   "``./install.sh``", "Build the library in your local directory. The script assumes the dependencies are available."
   "``./install.sh -i``", "Build the library, then build and install the hipSPARSELt package in ``/opt/rocm/hipsparselt``. The script prompts you for ``sudo`` access. This installs hipSPARSELt for all users."

Using install.sh to build hipSPARSELt with dependencies and clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The clients contains example code and unit tests. Common uses of ``install.sh`` to build them are listed in the table below.

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``./install.sh -h``", "Print the help information."
   "``./install.sh -dc``", "Build the dependencies, library, and client in your local directory. The ``-d`` flag only needs to be used once. For subsequent invocations of the script, it isn't necessary to rebuild the dependencies."
   "``./install.sh -c``", "Build the library and client in your local directory. The script assumes the dependencies are available."
   "``./install.sh -idc``", "Build the library, dependencies, and client, then build and install the hipSPARSELt package in ``/opt/rocm/hipsparselt``. The script prompts you for ``sudo`` access. This installs hipSPARSELt for all users."
   "``./install.sh -ic``", "Build the library and client, then build and install the hipSPARSELt package in ``opt/rocm/hipsparselt``. The script prompts you for ``sudo`` access. This installs hipSPARSELt for all users."

Building hipSPARSELt using individual make commands
----------------------------------------------------

You can build hipSPARSELt using the following commands:

.. note::

   CMake 3.16.8 or later is required to build hipSPARSELt.

.. code-block:: bash

   # Create and change the build directory
   mkdir -p build/release ; cd build/release

   # Change default install path (/opt/rocm); use -DCMAKE_INSTALL_PREFIX=<path> to adjust the path
   cmake ../..

   # Compile the hipSPARSELt library
   make -j$(nproc)

   # Install hipSPARSELt to `/opt/rocm`
   make install


You can build hipSPARSELt with the dependencies and clients using the following commands:

.. note::

   GoogleTest is required to build the hipSPARSELt clients.


.. code-block:: bash

   # Install GoogleTest
   mkdir -p build/release/deps ; cd build/release/deps
   cmake ../../../deps
   make -j$(nproc) install

   # Change to build directory
   cd ..

   # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
   cmake ../.. -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_SAMPLES=ON

   # Compile hipSPARSELt library
   make -j$(nproc)

   # Install hipSPARSELt to /opt/rocm
   make install

Testing the hipSPARSELt installation
==========================================

After successfully compiling the library with the clients, you can test the hipSPARSELt installation by running an example:

.. code-block:: bash

   # Navigate to clients binary directory
   cd hipSPARSELt/build/release/clients/staging

   # Execute hipSPARSELt example
   ./example_spmm_strided_batched -m 32 -n 32 -k 32 --batch_count 1

Running the benchmarks and unit tests
----------------------------------------------------------------------------

To run the benchmarks, build hipSPARSELt with the ``-DBUILD_CLIENTS_BENCHMARKS=ON`` option (or use ``./install.sh -c``).

.. code-block:: bash

   # Go to hipSPARSELt build directory
   cd hipSPARSELt/build/release

   # Run benchmark, e.g.
   ./clients/staging/hipsparselt-bench -f spmm -i 200 -m 256 -n 256 -k 256

To run the unit tests, build hipSPARSELt with the ``-DBUILD_CLIENTS_TESTS=ON`` option (or use ``./install.sh -c``).

.. code-block:: bash

   # Go to hipSPARSELt build directory
   cd hipSPARSELt; cd build/release

   # Run all tests
   ./clients/staging/hipsparselt-test
