.. meta::
   :description: Installing hipSPARSELt on Linux
   :keywords: hipSPARSELt, ROCm, install, Linux

.. _install-linux:

*****************************************
Build and install hipSPARSELt from source
*****************************************

To build hipSPARSELt as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build hipSPARSELt standalone using the following
instructions.

Prerequisites
=============

hipSPARSELt requires a :doc:`ROCm-enabled platform <rocm:index>` and the
:doc:`hipSPARSE library <hipsparse:index>` for the header file.

Quick start using install.sh
============================

The root directory of the `hipSPARSELt project <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparselt>`_
within the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_ contains the
helper bash script ``install.sh`` for building and installing hipSPARSELt on Ubuntu with a single command. The
script only accepts a few options and hardcodes configuration that can be specified by invoking
CMake directly. However, it's a great way to get started quickly and can serve as an example of how to build
and install hipSPARSELt. Some commands require ``sudo`` access, so the script might prompt you for a password.

.. code-block:: bash

   # Run install.sh script
   # Command line options:
   #   -h|--help            - prints help message
   #   -i|--install         - install after build
   #   -d|--dependencies    - install build dependencies
   #   -c|--clients         - build library clients too (combines with -i & -d)
   #   -g|--debug           - build with debug flag
   #   -k|--relwithdebinfo  - build with RelWithDebInfo

   ./install.sh -dc

For more detailed installation instructions, see :ref:`install-linux`.

Building hipSPARSELt from source
================================

To build hipSPARSELt from source, follow the instructions in this section.

To compile and run hipSPARSELt, the `ROCm platform <https://github.com/ROCm/ROCm>`_ is required.
The build also requires the following compile-time dependencies:

*  `hipSPARSE <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse>`_
*  `git <https://git-scm.com/>`_
*  `CMake <https://cmake.org/>`_ (Version 3.5 or later)
*  `GoogleTest <https://github.com/google/googletest>`_ (Optional: only required to build the clients)

Downloading hipSPARSELt
--------------------------------------------------------------------------------------

The hipSPARSELt source code is available from the
the `hipSPARSELt <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparselt>`_ directory
of the `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_ GitHub.

Download the develop branch for hipSPARSELt and all projects in the rocm-libraries repository
using these commands:

.. code-block:: bash

   git clone -b develop https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries/projects/hipsparselt

To limit your local checkout to the hipSPARSELt project, configure ``sparse-checkout`` before cloning.
This uses the Git partial clone feature (``--filter=blob:none``) to reduce how much data is downloaded.
Use the following commands for a sparse checkout:

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipsparselt
   git checkout develop # or use the branch you want to work with

.. note::

   To build ROCm 6.4.3 and earlier, use the hipSPARSELt repository at `<https://github.com/ROCm/hipSPARSELt>`_.
   For more information, see the documentation associated with the release you want to build.

Building hipSPARSELt using the install script
----------------------------------------------

It's recommended to use the ``install.sh`` script to install hipSPARSELt.
Here are the steps required to build different packages of the library, including the dependencies and clients.

.. note::

   You can run the ``install.sh`` script from the ``projects/hipsparselt`` directory.

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

   Run these commands from the ``projects/hipsparselt`` directory.

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

.. note::

   Run these commands from the ``projects/hipsparselt`` directory.

.. code-block:: bash

   # Navigate to clients binary directory
   cd build/release/clients/staging

   # Execute hipSPARSELt example
   ./example_spmm_strided_batched -m 32 -n 32 -k 32 --batch_count 1

Running the benchmarks and unit tests
----------------------------------------------------------------------------

To run the benchmarks, build hipSPARSELt with the ``-DBUILD_CLIENTS_BENCHMARKS=ON`` option (or use ``./install.sh -c``).

.. code-block:: bash

   # Go to hipSPARSELt build directory
   cd build/release

   # Run benchmark, e.g.
   ./clients/staging/hipsparselt-bench -f spmm -i 200 -m 256 -n 256 -k 256

To run the unit tests, build hipSPARSELt with the ``-DBUILD_CLIENTS_TESTS=ON`` option (or use ``./install.sh -c``).

.. code-block:: bash

   # Go to hipSPARSELt build directory
   cd build/release

   # Run all tests
   ./clients/staging/hipsparselt-test
