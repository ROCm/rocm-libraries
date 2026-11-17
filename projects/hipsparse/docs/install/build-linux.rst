.. meta::
   :description: hipSPARSE installation guide for Linux
   :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, install

.. _hipsparse_building:

*************************************
Build and install hipSPARSE for Linux
*************************************

For information on installing and building hipSPARSE on Microsoft Windows, see :doc:`hipSPARSE for Windows <./build-windows>`.

To build hipSPARSE as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build hipSPARSE standalone using the following
instructions.

Prerequisites
=============

hipSPARSE requires a :doc:`ROCm-enabled platform <rocm:compatibility/compatibility-matrix>`.

Building hipSPARSE from source
==============================

It isn't necessary to build hipSPARSE from source because it's ready to use after installing
the prebuilt packages, as described in :doc:`Install hipSPARSE <./install>`.
To build hipSPARSE from source, follow the instructions in this section.

To compile and run hipSPARSE, the `AMD ROCm Platform <https://github.com/ROCm/ROCm>`_ is required.
The build also requires the following compile-time dependencies:

*  `rocSPARSE <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse>`_
*  `git <https://git-scm.com/>`_
*  `CMake <https://cmake.org/>`_ (Version 3.25.2 or later)
*  `ROCm CMake <https://github.com/ROCm/rocm-cmake>`_ (Version 0.11.0 or later)
*  `GoogleTest <https://github.com/google/googletest>`_ (Optional: only required to build the clients)

By default, the CMake build doesn't download ROCm CMake or GoogleTest. If CMake can't find one of them,
the configuration step stops with an error naming it.
Add the install prefix of the dependency to ``CMAKE_PREFIX_PATH`` if it isn't installed in a default location.
To download ROCm CMake and GoogleTest when they aren't installed, configure with ``-DALLOW_FETCH_DEPS=ON``.
This requires network access during configuration.
If GoogleTest isn't installed, you can instead build it from a local source tree without network access
by setting ``FETCHCONTENT_SOURCE_DIR_GOOGLETEST`` to that directory.

The tests need 19 SuiteSparse matrices, which are stored with `DVC <https://dvc.org/>`_.
Before building the tests, run ``dvc pull -R projects/hipsparse/clients/matrices`` from the root of the rocm-libraries repository.
Reading from the DVC storage doesn't require credentials.
In a sparse checkout, DVC needs the ``.dvc`` directory at the repository root. If your checkout omits it, run
``git sparse-checkout add .dvc`` first.
Alternatively, point ``CMAKE_MATRICES_DIR`` at a directory with the converted ``.bin`` files,
or configure with ``-DALLOW_FETCH_DEPS=ON`` to download the matrices from SuiteSparse.

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
   ``git sparse-checkout set projects/hipsparse projects/rocsparse .dvc``.
   The ``.dvc`` directory is needed to download the test matrices with ``dvc pull``.

.. code-block:: shell

   git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout init --cone
   git sparse-checkout set projects/hipsparse .dvc # add projects/rocsparse to include dependencies
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
   "``./install.sh -d``", "Install the build dependencies using the distribution package manager, then build the library in your local directory. The ``-d`` flag only needs to be used once."
   "``./install.sh``", "Build the library in your local directory. The script assumes the dependencies are available."
   "``./install.sh -i``", "Build the library, then build and install the hipSPARSE package in ``/opt/rocm/hipsparse``. The script prompts you for ``sudo`` access. This installs hipSPARSE for all users."

Using install.sh to build hipSPARSE with dependencies and clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The clients contains example code and unit tests. Common use cases of ``install.sh`` to build them are listed in the table below.

.. note::

   The tests need the SuiteSparse test matrices. Before running these commands, pull them with
   ``dvc pull -R clients/matrices`` from the ``projects/hipsparse`` directory.
   ``install.sh`` downloads missing matrices only when you pass ``--matrices-dir``.
   The clients also need GoogleTest, which ``-d`` doesn't install. Install it first, for example the
   ``libgtest-dev`` or ``gtest-devel`` package.

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``./install.sh -h``", "Print the help information."
   "``./install.sh -dc``", "Install the build dependencies using the distribution package manager, then build the library and client in your local directory. The ``-d`` flag doesn't install GoogleTest, which the client requires. The ``-d`` flag only needs to be used once."
   "``./install.sh -c``", "Build the library and client in your local directory. The script assumes the dependencies are available."
   "``./install.sh -idc``", "Install the build dependencies, build the library and client, then build and install the hipSPARSE package in ``/opt/rocm/hipsparse``. The ``-d`` flag doesn't install GoogleTest, which the client requires. The script prompts you for ``sudo`` access. This installs hipSPARSE and the clients for all users."
   "``./install.sh -ic``", "Build the library and client, then build and install the hipSPARSE package in ``opt/rocm/hipsparse``. The script prompts you for ``sudo`` access. This installs hipSPARSE and the clients for all users"

Building hipSPARSE using individual make commands
--------------------------------------------------

You can build hipSPARSE using the following commands:

.. note::

   Run these commands from the ``projects/hipsparse`` directory.

.. note::

   CMake 3.25.2 or later is required to build hipSPARSE.

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

   GoogleTest is required to build the hipSPARSE clients. Install it, for example the ``libgtest-dev`` or
   ``gtest-devel`` package, or configure with ``-DALLOW_FETCH_DEPS=ON`` to download it.
   ``-DALLOW_FETCH_DEPS=ON`` also downloads the test matrices from SuiteSparse instead of the ``dvc pull`` step.

.. code-block:: shell

   # Pull the test matrices
   dvc pull -R clients/matrices

   # Create and change to build directory
   mkdir -p build/release ; cd build/release

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

