.. meta::
  :description: Learn how to include hipDNN in a CMake project.
  :keywords: hipDNN, ROCm, install, CMake

.. _add-project:

**************************
Add hipDNN to your project
**************************

This topic demonstrates how to include hipDNN in a CMake project on a system with ROCm installed.

Prerequisites
==============

ROCm and hipDNN must be installed along with the dependencies. See :ref:`prerequisites`.

.. _add-hipdnn-steps:

Steps
=====

1. Add ``hipdnn_frontend`` in your ``CMakeLists.txt`` file:

   .. code:: cmake

     project(my_app LANGUAGES CXX)
     set(CMAKE_CXX_STANDARD 17)

     find_package(hipdnn_frontend CONFIG REQUIRED)

     add_executable(my_app main.cpp)
     target_link_libraries(my_app PRIVATE hipdnn_frontend)

   - ``find_package(hipdnn_frontend)`` transitively brings in ``hipdnn_backend``, ``hipdnn_data_sdk``, and ``hip``, so only one ``find_package`` call is needed. HIP runtime APIs (``hipMalloc``, ``hipFree``, etc.) are available through the transitive ``hip::host`` link dependency.
   - If CMake can't find the packages, set ``CMAKE_PREFIX_PATH`` to the ROCm install location. hipDNN CMake files are installed to ``<ROCM Install Path>/lib/cmake`` by default, which CMake might already be searching automatically depending on your system configuration. For example: ``-DCMAKE_PREFIX_PATH=/opt/rocm/lib/cmake``.

2. Include the frontend header to access the hipDNN graph API:

   .. code:: cpp

     #include <hipdnn_frontend.hpp>

Next steps
==========

- :ref:`architecture`
- :ref:`build-execute`
- See `Samples <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/samples/README.md>`_ for complete working examples of building and executing operation graphs.
