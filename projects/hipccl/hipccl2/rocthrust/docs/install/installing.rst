.. meta::
  :description: rocThrust API data type support
  :keywords: rocThrust, ROCm, API, reference, data type, support

.. _install:

******************************************
Installation
******************************************

There are two main methods for installing rocThrust: 

* Install the prebuilt packages from the ROCm repositories
* Build and install rocThrust from source files

Installing prebuilt packages
=============================

Installing the prebuilt rocThrust packages requires a ROCm-enabled platform. See the `ROCm documentation <https://docs.amd.com/>`_ for more information. After installing ROCm or enabling the ROCm repositories, rocThrust can be obtained using the system package manager.

For Ubuntu and Debian:

.. code-block:: shell

    sudo apt-get install rocthrust

For CentOS:

.. code-block:: shell

    sudo yum install rocthrust

For SLES:

.. code-block:: shell

    sudo dnf install rocthrust

This will install rocThrust into the ``/opt/rocm`` directory.

Building rocThrust from source files
=====================================

Obtaining sources
-----------------

The rocThrust sources are available from the `rocThrust GitHub Repository <https://github.com/ROCmSoftwarePlatform/rocThrust>`_. Use the branch that matches the system-installed version of ROCm. For example on a system that has ROCm 6.0 installed, use the following command to obtain rocThrust sources:

.. code-block:: shell

    git clone -b release/rocm-rel-6.0 https://github.com/ROCm/rocThrust.git

Building the library
--------------------

After obtaining the sources, rocThrust can be built using the installation script. Note that this method only works on Linux, see the following sections for how to build rocThrust on Windows:

.. code-block:: shell

    cd rocThrust
    ./install --install

This automatically builds all required dependencies, excluding HIP and Git, and installs the project to ``/opt/rocm`` if everything went well. See ``./install --help`` for further information.

Building with rmake.py
----------------------

Alternatively, the ``rmake.py`` script can be used to install rocThrust. This is the recommended method to install rocThrust from source on Windows. After obtaining the sources, rocThrust can be installed this way as follows:

.. code-block:: shell

    cd rocThrust
    # The -i option will install rocThrust into /opt/rocm on Linux, and C:\hipSDK on Windows.
    python rmake.py -i
    # The -c option will build all clients, including unit tests
    python rmake.py -c

Building with CMake
-------------------

For a more elaborate installation process, rocThrust can be built manually using CMake. This enables certain configuration options that are not exposed to the ``./install`` and ``rmake.py`` scripts. Note that building rocThrust using CMake works on both Linux and Windows. In general, rocThrust can be built and installed using CMake as follows:

.. code-block:: shell

    cd rocThrust;
    # Configure the project
    CXX=hipcc cmake -S . -B build [options]
    # Build
    cmake --build build
    # Optionally, run the tests
    ctest --output-on-failure
    # Install
    cmake --install build

.. note::
    ``CXX`` must be set to ``hipcc`` to build for the ROCm platform. 
    
The following configuration options are available, in addition to the built-in CMake options:

* ``DISABLE_WERROR`` disables passing ``-Werror`` to the compiler during the build. ``ON`` by default.
* ``BUILD_TEST`` controls whether to build the rocThrust and hipstdpar tests. ``OFF`` by default.
* ``BUILD_HIPSTDPAR_TEST`` controls whether to build the hipstdpar tests. Enabling this option is only necessary when BUILD_TEST is OFF. ``OFF`` by default.
* ``BUILD_BENCHMARK`` controls whether to build the rocThrust benchmarks. ``OFF`` by default.
* ``BUILD_EXAMPLES`` controls whether to build rocThrust examples. ``OFF`` by default.
* ``DOWNLOAD_ROCPRIM`` controls whether to force downloading rocPRIM, regardless of whether rocPRIM is currently installed. Defaults to ``OFF``.
* ``RNG_SEED_COUNT`` sets the non-repeatable random dataset count. Defaults to ``0``.
* ``PRNG_SEEDS`` sets RNG seeds, to ensure reproducible random data generation. Defaults to ``1``.
