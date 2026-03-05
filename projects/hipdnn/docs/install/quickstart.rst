.. meta::
  :description: 
  :keywords: hipDNN, ROCm, install, Windows

******************
hipDNN quick start
******************

This guide shows how to consume an already-installed hipDNN in a CMake project. 

.. important::

  This guide is appropriate for most general-purpose users. Plugin developers who want to build from source should refer to :ref:`plugin-install`.

Prerequisites
==============

- **ROCm** / **TheRock** installed with hipDNN packages (see `TheRock Releases <https://github.com/ROCm/TheRock/blob/main/RELEASES.md>`_)
- **CMake**
- **Ninja** (recommended)
- A **C++17** compatible compiler

Configure project files
=======================

1. Configure your ``CMakeLists.txt`` file:

   .. code:: cmake

     cmake_minimum_required(VERSION <your_minimum>)
     project(my_app LANGUAGES CXX)
     set(CMAKE_CXX_STANDARD 17)

     find_package(hipdnn_frontend CONFIG REQUIRED)

     add_executable(my_app main.cpp)
     target_link_libraries(my_app PRIVATE hipdnn_frontend)

   .. note::

     - ``find_package(hipdnn_frontend)`` transitively brings in ``hipdnn_backend``, ``hipdnn_data_sdk``, and ``hip`` — only one ``find_package`` call is needed. HIP runtime APIs (``hipMalloc``, ``hipFree``, etc.) are available through the transitive ``hip::host`` link dependency.
     - If CMake can't find the packages, set ``CMAKE_PREFIX_PATH`` to the install location. hipDNN CMake files are installed to ``/opt/rocm/lib/cmake`` by default, which CMake may already search automatically depending on your system configuration.


2. Include the frontend header to access the hipDNN graph API:

   .. code:: cpp

     #include <hipdnn_frontend.hpp>

   See `Samples <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/README.md>`_ for complete working examples of building and executing operation graphs.

Build the project
=================

Use this code to build the project:

.. code:: bash

  mkdir build && cd build
  cmake -GNinja -DCMAKE_PREFIX_PATH=/opt/rocm ..
  ninja