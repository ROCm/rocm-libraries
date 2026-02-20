.. meta::
  :description: Learn how to migrate your frontend code from cudNN to hipDNN.
  :keywords: hipDNN, ROCm, API, how-to 

.. _migrate-cudnn:

*********************************
Migrate a cudNN project to hipDNN
*********************************

This guide demonstrates how to migrate a cuDNN code project to hipDNN. 

Key differences cuDNN vs. hipDNN
================================

Review this table to see a high-level overview of the differences between cuDNN vs. hipDNN:

.. list-table::
   :widths: 3 3 3 
   :header-rows: 1

   * - Aspect
     - cuDNN frontend
     - hipDNN frontend
   * - Namespace
     - ``cudnn_frontend``
     - ``hipdnn_frontend``
   * - Handle creation
     - ``cudnnCreate(&handle)``
     - ``hipdnnCreate(&handle``
   * - Handle destruction
     - ``cudnnDestroy(handle)``
     - ``hipdnnDestroy(handle)``
   * - Heuristics modes
     - All cuDNN heuristic modes
     - Currently only ``HeurMode_t::FALLBACK``
   * - Operation support
     - All cuDNN operations
     - See `Operation support <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/OperationSupport.md>`_ for more information.
   * - Device memory utility
     - ``Surface<type>``
     - ``MigratableMemory<type>``
   * - Device memory access
     - ``Surface<type>::devPtr``
     - ``MigratableMemory<type>::deviceData()``


Migrate your cuDNN project
==========================

Before you begin, you should have hipDNN built and installed.
See :ref:`install` for more information. Here's an MVP (minimum viable product) example of a hipDNN project in ``CMakeLists.txt``:

.. code:: cmake

  cmake_minimum_required(VERSION <your_minimum>)


  project(my_hipdnn_project VERSION 1.0.0 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 17)

  # CRITICAL: Enable PIC for plugin system compatibility
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  # Find required packages
  find_package(hipdnn_frontend CONFIG REQUIRED)

  # Create executable
  add_executable(my_app main.cpp)

  # Link libraries
  target_link_libraries(my_app PRIVATE
      hipdnn_frontend
  )

Build commands:

.. code:: bash

  mkdir build && cd build
  cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm -GNinja
  ninja


.. tip::

  See `Working examples in the Porting Guide <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/PortingGuide.md#working-examples>`_ for ported code samples.


Troubleshooting
===============

Error: ``TLS model mismatch, plugin load failures``
---------------------------------------------------

hipDNN uses a plugin system that requires position-independent code. Use this code to fix the error:

.. code:: cmake

  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

Error: ``Missing Heuristic modes A and B``
------------------------------------------

The heuristic implementation in hipDNN has yet to be implemented

TO fix the problem, use a combination of ``graph::get_ranked_engine_ids()`` and ``graph::set_preferred_engine_id_ext()`` if you need more detailed control over engine selection.

Error: ``Different memory utilities for allocating device memory.``
-------------------------------------------------------------------

The memory utilities are typically consumer dependent, and written on an as-needed basis.  
cuDNN provides a surface utility for their samples, for example.

To fix the issue, use ``MigratableMemory<type>``; its a utility that can automatically migrate data between the host and device (its also works as a stand-in).  
If you want to manage dims/strides more carefully, use the ``Tensor`` utility class ̶̶  both of these classes can be found in the ``hipdnn_data_sdk::utilities`` namespace.