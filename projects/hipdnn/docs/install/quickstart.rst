.. meta::
  :description: This quick start guide demonstrates how to include hipDNN in a CMake project.
  :keywords: hipDNN, ROCm, install, Windows

******************
hipDNN quick start
******************

This guide demonstrates how to include hipDNN in a CMake project. This guide requires hipDNN and ROCM already installed on your system.

.. important::

  This guide is appropriate for integrating hipDNN into applications for running deep learning worlkloads. Plugin developers who want to build from source should refer to :ref:`plugin-install`.

Prerequisites
==============

- ROCm and hipDNN installed with dependencies. See :ref:`prerequisites`.

Add the hipdnn_frontend package to your project files
=====================================================

1. Add ``hipdnn_frontend`` in your ``CMakeLists.txt`` file:

   .. code:: cmake

     cmake_minimum_required(VERSION <your_minimum>)
     project(my_app LANGUAGES CXX)
     set(CMAKE_CXX_STANDARD 17)

     find_package(hipdnn_frontend CONFIG REQUIRED)

     add_executable(my_app main.cpp)
     target_link_libraries(my_app PRIVATE hipdnn_frontend)

   .. note::

     - ``find_package(hipdnn_frontend)`` transitively brings in ``hipdnn_backend``, ``hipdnn_data_sdk``, and ``hip`` — only one ``find_package`` call is needed. HIP runtime APIs (``hipMalloc``, ``hipFree``, etc.) are available through the transitive ``hip::host`` link dependency.
     - If CMake can't find the packages, set ``CMAKE_PREFIX_PATH`` to the ROCm install location. hipDNN CMake files are installed to ``<ROCM Install Path>/lib/cmake`` by default, which CMake may already search automatically depending on your system configuration. For example: ``-DCMAKE_PREFIX_PATH=/opt/rocm/lib/cmake``.


2. Include the frontend header to access the hipDNN graph API:

   .. code:: cpp

     #include <hipdnn_frontend.hpp>

   See `Samples <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/README.md>`_ for complete working examples of building and executing operation graphs.
