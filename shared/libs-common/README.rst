.. highlight:: rst
.. |project_name| replace:: libs-common

==============
|project_name|
==============

-----------------
Quick Start Guide
-----------------

|project_name| is a header-only library that provides common files for ROCm and HIP libraries.

This section describes how to configure and build the |project_name| project. It assumes the user has a
ROCm installation, Python 3.8 or later, and CMake 3.11 or later.

^^^^^^^^^^^^^^^^^^^
Configure and build
^^^^^^^^^^^^^^^^^^^

|project_name| provides modern CMake support and relies on native CMake functionality, with the exception of
some project specific options. As such, users are advised to consult the CMake documentation for
general usage questions. Below are usage examples to get started. For details on all configuration
options, see the options section.

Build on fresh clone of `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_
--------------------------------------------------------------------------------

   .. code-block:: cmake
      :linenos:

      cd shared/libs-common

      # configure
      cmake --preset default-release

      # build
      cmake --build _build

      # install (optional)
      cmake --install _build

Options
-------

*CMake options*:

* ``CMAKE_BUILD_TYPE``: Any of Release, Debug, RelWithDebInfo, MinSizeRel
* ``CMAKE_INSTALL_PREFIX``: Base installation directory (defaults to ``/opt/rocm`` on Linux, ``C:/hipSDK`` on Windows)
* ``CMAKE_PREFIX_PATH``: Find package search path (consider setting to ``$ROCM_PATH``)

*Build control options*:

* ``LIBS_COMMON_ENABLE_HIPBLAS``: Enable hipBLAS common library (default: ``ON``)
* ``LIBS_COMMON_ENABLE_HIPSPARSE``: Enable hipSPARSE common library (default: ``ON``)


^^^^^^^^^^^^^^^^^^^^^^^
Using in CMake projects
^^^^^^^^^^^^^^^^^^^^^^^

To use |project_name| in your `rocm-libraries <https://github.com/ROCm/rocm-libraries>`_ CMake project,
add the following to your ``CMakeLists.txt`` file:

.. code-block:: cmake
   :linenos:

   # Add `libs-common` as a subdirectory
   # This will build `libs-common` and add it to your build-tree
   add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/<path-to>/shared/libs-common libs-common)

   # Link your target to the desired library
   # This will propagate required include paths to your target
   target_link_libraries(my_target PRIVATE roc::hipsparse-common)

^^^^^^^^^^^^^
CMake targets
^^^^^^^^^^^^^

*Libraries*:

* ``roc::hipblas-common``
* ``roc::hipsparse-common``
