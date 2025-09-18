.. highlight:: rst
.. |project_name| replace:: hipBLAS-common

==============
|project_name|
==============

-----------------
Quick Start Guide
-----------------

|project_name| is a header-only library that provides common files for hipBLAS and hipBLASLt.

This section describes how to configure and build the |project_name| project. It assumes the user has a
ROCm installation, Python 3.8 or later, and CMake 3.25.0 or later.

^^^^^^^^^^^^^^^^^^^
Configure and build
^^^^^^^^^^^^^^^^^^^

|project_name| provides modern CMake support and relies on native CMake functionality, with the exception of
some project specific options. As such, users are advised to consult the CMake documentation for
general usage questions. Below are usage examples to get started. For details on all configuration
options, see the options section.

Build and install |project_name|
--------------------------------

   .. code-block:: bash
      :linenos:

      cd hipBLAS-common
      CXX=/opt/rocm/bin/amdclang++             \
      cmake -B build                           \
            -S .                               \
            -D CMAKE_INSTALL_PREFIX=<preferred installation path>
      cmake --build build --target install

CMake targets
-------------

* ``roc::hipblas-common``

Using rmake script
------------------

   .. code-block:: bash
      :linenos:

      python3 ./rmake.py --install
