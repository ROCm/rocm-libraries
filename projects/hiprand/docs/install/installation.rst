.. meta::
   :description: hipRAND installation guide
   :keywords: hipRAND, ROCm, library, API, tool, installation, build, Python wrapper

.. _installation:

*******************************************************************
Installing and building hipRAND
*******************************************************************

To install hipRAND, choose between the following two methods:

-   :ref:`Using prebuilt packages from the ROCm repositories <prebuilt-packages>`
-   :ref:`Building from source <build-from-source>`   

.. _prebuilt-packages:

Install using prebuilt packages
===============================

The prebuilt hipRAND packages require a ROCm-enabled platform.
For information on installing ROCm, see the `ROCm installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_.
After installing ROCm or enabling the ROCm repositories, use the system package manager to install hipRAND.

For Ubuntu and Debian:

.. code-block:: shell

   sudo apt-get install hiprand

For CentOS-based systems:

.. code-block:: shell

   sudo yum install hiprand

For SLES:

.. code-block:: shell

   sudo dnf install hiprand

These commands install hipRAND in the ``/opt/rocm`` directory.

.. _build-from-source:

Build hipRAND from source
===============================

This section provides the information required to build hipRAND from source.

Requirements
----------------------------

To build hipRAND for the ROCm platform, CMake version 3.16 or later is required,
along with the following components:

* `ROCm Software <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_ (version 5.0.0 or later)
* `rocRAND <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocrand>`_

Downloading the source code
----------------------------

The hipRAND source code is available from the `hiprand folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiprand>`_ of
the `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_ GitHub repository.
Use the branch that matches the ROCm version installed on the system.
The hipRAND source code can be cloned in two different ways.

.. note::

   For both methods, replace all occurrences of "x.y" in the commands with the version number matching your ROCm installation.
   For example, if you have ROCm 7.0 installed, clone the ``release/rocm-rel-7.0`` branch.

*  Clone the entire `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_ repository.
   This is the default method and is the recommended option if you need to install other
   ROCm libraries alongside hipRAND. However, due to the download size, ``git clone``
   might take a significant amount of time to complete.

   On a system with ROCm x.y installed, use the following command to obtain the source code
   for hipRAND version x.y. Replace x.y with the actual version:

   .. code-block:: shell

      git clone -b release/rocm-rel-7.0 https://github.com/ROCm/rocm-libraries.git

*  Clone the individual hipRAND project folder. This option only fetches the hipRAND source code,
   without any additional ROCm libraries. This significantly reduces the amount of time required
   to complete the clone operation. However, it requires Git 2.25 or later.
   To use this method to obtain the source code for hipRAND version x.y, run the following commands.
   Replace x.y with the actual version:

   .. code-block:: shell

      git clone -b release/rocm-rel-x.y --no-checkout --depth=1 --filter=tree:0 https://github.com/ROCm/rocm-libraries.git
      cd rocm-libraries
      git sparse-checkout set --cone projects/hiprand
      git checkout release/rocm-rel-x.y

.. note::

   To build ROCm 6.4 and earlier, use the hipRAND repository at `<https://github.com/ROCm/hipRAND>`_.
   For more information, see the documentation associated with the release you want to build.

Building the library
----------------------------

After obtaining the sources and dependencies, build hipRAND for ROCm software using the installation script:

.. code-block:: shell

    cd rocm-libraries/projects/hiprand
    ./install --install

This automatically builds all required dependencies, excluding Git and the requirements listed above,
and installs the project to ``/opt/rocm``. For further information, run the ``./install --help`` command.

The clients, which are enabled using the ``--clients`` option, consist of the hipRAND tests and add the additional dependency
of `GoogleTest <https://github.com/google/googletest>`_.

Building with CMake
----------------------------

For a more detailed installation process, build hipRAND manually using CMake.
This enables certain configuration options that are not available through the ``install`` script.
To build hipRAND, use CMake with the following configuration:

.. code-block:: shell

    cd rocm-libraries/projects/hiprand; mkdir build; cd build
    # Configure the project
    CXX=<compiler> cmake [options] ..
    # Build
    make -j$(nproc)
    # Optionally, run the tests
    ctest --output-on-failure
    # Install
    [sudo] make install

Where ``<compiler>`` should be set to ``hipcc`` or ``amdclang`` for ROCm.
The default build configuration is ``Release``.

Here are the CMake options:

* ``BUILD_FORTRAN_WRAPPER``: Builds the Fortran wrapper when set to ``ON``. Defaults to ``OFF``.
* ``BUILD_TEST``: Builds the hipRAND tests when set to ``ON``. Defaults to ``OFF``.
* ``BUILD_BENCHMARK``: Builds the hipRAND benchmarks when set to ``ON``. Defaults to ``OFF``.
* ``BUILD_ADDRESS_SANITIZER``: Builds with address sanitization enabled when set to ``ON``. Defaults to ``OFF``.
* ``ROCRAND_PATH``: Specifies a rocRAND install other than the default system installed version.
* ``DOWNLOAD_ROCRAND``: Downloads and installs rocRAND in the build directory when set to ``ON``. Defaults to ``OFF``.
* ``DEPENDENCIES_FORCE_DOWNLOAD``: Downloads and builds the dependencies instead of using the system-installed dependencies when set to ``ON``. Defaults to ``OFF``.

If you are using ``ROCRAND_PATH`` or ``DOWNLOAD_ROCRAND`` when rocRAND is already installed in the default location,
you must use the ``CMAKE_NO_SYSTEM_FROM_IMPORTED=ON`` option to configure the project.
Failing to do so might process the rocRAND headers from the system-installed version instead of the specified version,
leading to errors or missing functionality.

Common build errors
^^^^^^^^^^^^^^^^^^^

Use the following tips to troubleshoot build problems.

*  ``rocrand`` package configuration file not found:

   .. code-block:: shell

      Could not find a package configuration file provided by "rocrand" with any of the following names:

      rocrandConfig.cmake
      rocrand-config.cmake

   **Solution**: Install `rocRAND <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocrand>`_.

*  ``ROCM`` package configuration file not found:

   .. code-block:: shell

      Could not find a package configuration file provided by "ROCM" with any of the following names:

      ROCMConfig.cmake
      rocm-config.cmake

   **Solution**: Install the `ROCm CMake modules <https://github.com/ROCm/rocm-cmake>`_.

Building the Python API wrapper
===============================

This section provides the information required to build the hipRAND Python API wrapper.

Requirements
----------------------------

The hipRAND Python API Wrapper requires the following dependencies:

* hipRAND
* Python 3.5
* NumPy (This is installed automatically as a dependency, if necessary.)

.. note::
    
   If hipRAND is built from source but is either not installed or installed in a
   non-standard directory, then set the ``ROCRAND_PATH`` or ``HIPRAND_PATH`` environment variable to the
   path containing ``libhiprand.so`` as shown below:

   .. code-block:: shell

      export HIPRAND_PATH=~/rocm-libraries/projects/hiprand/build/library/

Installation
----------------------------

To install the Python hipRAND module using ``pip``, run these commands:

.. code-block:: shell

   cd rocm-libraries/projects/hiprand/python/hiprand
   pip install .

Use these commands to run the tests:

.. code-block:: shell

   cd rocm-libraries/projects/hiprand/python/hiprand
   python tests/hiprand_test.py
