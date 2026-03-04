.. meta::
  :description: hipDNN installation on Linux 
  :keywords: hipDNN, ROCm, install

.. _install:

****************************
hipDNN installation on Linux
****************************

This guide shows you how to install hipDNN on Linux. Before you begin, ensure you've installed the required dependencies outlined in :ref:`prerequisites`. 

Build hipDNN from TheROCK
=========================

You can install hipDNN and all the dependencies from `TheROCK <https://github.com/ROCm/TheRock/blob/main/RELEASES.md>`_. 
Follow the installation instructions for the latest release to install hipDNN and other ROCm release artifacts.

Build and install hipDNN from source
====================================

Alternatively, you can build hipDNN from the GitHub repository instead:

1. Clone the rocm-libraries repository with ``git sparse-checkout``:

   .. code:: bash

      git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-libraries.git
      cd rocm-libraries
      git sparse-checkout init --cone
      git sparse-checkout set projects/hipdnn dnn-providers/miopen-provider
      git checkout develop # or the branch you are starting from

2. Build hipDNN:

   .. code:: bash

      cd rocm-libraries/projects/hipdnn
      mkdir build && cd build

      # Configure with Ninja (recommended)
      cmake -GNinja ..

      # Build and run all tests
      # Note that some tests may take several minutes to complete
      ninja check

   Refer to the :ref:`target` section below for information on additional build targets.

3. Install hipDNN.

   Set the install path with a build configuration. Refer to the :ref:`configure` section to determine which build configuration suits your workflow. 

   .. code:: bash

      sudo ninja install

.. _configure:

Build configurations
====================

Use the code in these sections to configure specific hipDNN builds for your workflow.

Release build (default)
-----------------------

.. code:: bash

   cmake -GNinja ..

Debug build
-----------

.. code:: bash

   cmake -GNinja -DCMAKE_BUILD_TYPE=Debug ..

Code coverage build
-------------------

.. code:: bash

   cmake -GNinja -DHIPDNN_ENABLE_COVERAGE=ON ..
   ninja coverage
   # Unit tests will be run and coverage reports will be generated in build/coverage_report/


Address sanitizer build
-----------------------

.. code:: bash

   cmake -GNinja -DBUILD_ADDRESS_SANITIZER=ON ..
   ninja check
   # Note: Some HIP-related tests may be skipped due to AddressSanitizer incompatibility


Build-specific components
-------------------------

.. code:: bash

   # Build without plugins
   cmake -GNinja -DHIP_DNN_BUILD_PLUGINS=OFF ..

   # Build without frontend
   cmake -GNinja -DHIP_DNN_BUILD_FRONTEND=OFF ..

   # Build without backend
   cmake -GNinja -DHIP_DNN_BUILD_BACKEND=OFF ..

.. _rocm-path:

Configure CMake variables
=========================

If the ROCm ``bin`` folder is included in your system path, then the AMD toolchain should be detected automatically. 
If not, these CMake variables can be used to assist CMake in the tool discovery.

- ``ROCM_PATH``: Specifies the root ROCm folder location. The toolchain folders are hard-coded using that path, skipping auto-detection of the toolchain (does not have a default value). 

  .. warning::

    Do *not* set ``ROCM_PATH`` in your environment. Setting ``ROCM_PATH`` in the environment will cause the compiler check to fail. Instead, use the ``-D`` option to cmake. For example, ``-DROCM_PATH=/path/to/rocm``.

- ``ROCM_CMAKE_PATH`` (preferred): Similar to ``ROCM_PATH`` but relies on CMake's built-in detection to locate the toolchain. (Default: ``/opt/rocm`` (Linux) / ``C:/dist/therock`` (Windows)). This can be set in your system environment. This can be set automatically if the ROCm ``bin`` folder is in your system path.

.. note::

   If ``ROCM_PATH`` is set using the ``-D`` option in CMake, then it'll take precedence over ``ROCM_CMAKE_PATH``.

The HIP compiler is required to build some integration tests, but isn't required for the hipDNN library itself.

Use this CMake variable to control where the hipDNN library files will be installed when the ``install`` target is run: ``CMAKE_INSTALL_PREFIX``. This specifies where hipDNN will be installed (defaults to ``ROCM_PATH`` if ``ROCM_PAth`` is set, then ``ROCM_CMAKE_PATH`` if set, otherwise uses the CMake system default).

These variables can all be set independently:

.. code:: bash

   # Default: Use system path to locate ROCm folder, install path is unset.
   cmake -GNinja ..

   # Install hipDNN to custom location, find ROCm dependencies in the default location
   cmake -GNinja -DCMAKE_INSTALL_PREFIX=/custom/install/path ..

   # Both custom
   cmake -GNinja -DROCM_CMAKE_PATH=/custom/rocm -DCMAKE_INSTALL_PREFIX=/another/path ..


Clang tools
===========

Different versions of Clang tools are required. For example, clang-format version 18 and clang-tidy version 20. The hipDNN project tool discovery provides two mechanisms to assist with finding the needed version of each tool.

Version suffix
--------------

Before searching for the tool using it's standard name, a search will be made for a tool that has the version appended as a suffix. For example, before looking for ``clang-format`` a search for a file named ``clang-format-18`` will be run first, and if that fails then a search will be made for ``clang-format``. 
Similarly, ``clang-tidy-20`` will be searched for first, and then ``clang-tidy``. This approach can be used if it's possible to modify the Clang toolchain folder(s) on your system to give the tools the corresponding names.

.. _llvm:

``LLVM_TOOLS_SEARCH_PREFIX``
----------------------------

As an alternative, ``LLVM_TOOLS_SEARCH_PREFIX`` can be set as a prefix for the folder path where the Clang tools are installed, such that ``${LLVM_TOOLS_SEARCH_PREFIX}18/bin`` is where the Clang version 18 tools are located, and ``${LLVM_TOOLS_SEARCH_PREFIX}20/bin`` is where the Clang version 20 tools are located. 
The CMake configuration step will automatically select the required version for each tool from these folders. 
For example with ``-DLLVM_TOOLS_SEARCH_PREFIX=c:\tools\clang`` these folders will be searched for Clang tools (depending on the version of each tool that is needed):

- ``c:\tools\clang18\bin``
- ``c:\tools\clang20\bin``
- ``c:\tools\clang\bin``

.. _target:

Build targets
=============

.. note::

   Make is supported for all targets. Configure with ``cmake -G "Unix Makefiles" ..`` if it is not the default generator in your environment. For parallel builds, use ``make -j$(nproc)`` on Linux. Unlike ``ninja``, ``make`` doesn't build in parallel by default.

All targets support parallel builds with ``ninja``.

.. list-table::
   :widths: 3 5
   :header-rows: 1

   * - Target
     - Description
   * - ``\<no target\>``
     - Builds all components 
   * - ``check`` / ``check-verbose``
     - Build and run all tests (see `hipDNN testing <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/Testing.md>`_ for more information)
   * - ``unit-check`` / ``unit-check-verbose``
     - Build and run the unit tests exclusively and the API tests (minimal version of ``check``)
   * - ``integration-check`` / ``integration-check-verbose``
     - Build and run the E2E integration tests exclusively (this is the bulk of the testing time)
   * - ``install``
     - Install libraries and headers
   * - ``format``
     - Auto-format all C++ source files
   * - ``check_format``
     - Check code formatting compliance
   * - ``coverage``
     - Run ``check`` and generate test coverage reports (requires ``-DHIPDNN_ENABLE_COVERAGE=ON``)
   * - ``\<no target\>``
     - Builds all components 
   * - ``unit-coverage`` / ``integration-coverage``
     - Run ``unit-check`` or ``integration-check`` (respectively) and generate test coverage reports (requires ``-DHIPDNN_ENABLE_COVERAGE=ON``)
   * - ``current-coverage``
     - Generate test coverage reports using coverage data already on disk (does not automatically run ``check``; it requires ``-DHIPDNN_ENABLE_COVERAGE=ON``)
   * - ``clean``
     - Clean build artifacts
   * - ``validate_test_names``
     - Validates test names conform to naming rules
   * - ``generate_hipdnn_data_sdk_headers``
     - Generate C++ headers from schema (``.fbs``) files

Build commands
--------------

These example build commands are equivalent (depending on which generator was used) and will build the ``check`` target to build and run all tests.

Use ``cmake`` to invoke build (regardless of which generator was used):

.. code:: bash

   projects/hipdnn/build> cmake --build . --target check

If ``Ninja`` was used as the generator:

.. code:: bash

   projects/hipdnn/build> ninja check

If a Makefile-type generator was used (not recommended):

.. code:: bash
   
   projects/hipdnn/build> make check

Troubleshooting
===============

Common build issues
-------------------

Out of memory during build
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run this code to resolve this issue:

.. code:: bash

   # Reduce parallel jobs
   ninja -j4  # or even -j2 for systems with limited RAM

Docker GPU access issues
~~~~~~~~~~~~~~~~~~~~~~~~

- Ensure ROCm is installed on the host system.
- Verify the GPU is visible using ``rocm-smi`` or ``rocminfo``.
- Ensure the user is in ``video`` and ``render`` groups:
 
  .. code:: bash

    sudo usermod -a -G video,render $USER
    # Log out and back in for changes to take effect

Verify the installation
=======================

See `hipDNN Samples <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/README.md>`_ for detailed instructions on building test sample programs using hipDNN.
