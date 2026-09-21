.. meta::
  :description: Learn how to migrate your frontend code from cuDNN to hipDNN.
  :keywords: hipDNN, ROCm, migrate, cuDNN

.. _migrate-cudnn:

********************************************
Migrate NVIDIA CUDA cuDNN projects to hipDNN
********************************************

This topic demonstrates how to migrate a NVIDIA CUDA cuDNN project to hipDNN.

Before you begin, ensure hipDNN (ROCm) is installed. See :ref:`prerequisites` for more information.

Here's a minimal example of a hipDNN project in ``CMakeLists.txt``:

.. code:: cmake

    project(my_app LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 17)

    find_package(hipdnn_frontend CONFIG REQUIRED)

    add_executable(my_app main.cpp)
    target_link_libraries(my_app PRIVATE hipdnn_frontend)

.. tip::

  See `Working examples in the Porting Guide <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/PortingGuide.md#working-examples>`_ for ported code samples.

Key differences between cuDNN and hipDNN
========================================

This table provides a high-level overview of the differences between cuDNN and hipDNN:

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
     - See :ref:`plugin-support` for more information.
   * - Device memory utility
     - ``Surface<type>``
     - ``MigratableMemory<type>``
   * - Device memory access
     - ``Surface<type>::devPtr``
     - ``MigratableMemory<type>::deviceData()``

See the :ref:`dimension-layouts` section of the Operation Support doccument for operation-specifc hipDNN tensor layout details.

Troubleshooting
===============

Error: Missing Heuristic modes A and B
--------------------------------------

The heuristic implementation in hipDNN is not implemented.

To fix the problem, use a combination of ``graph::get_ranked_engine_ids()`` and ``graph::set_preferred_engine_id_ext()`` if you need more detailed control over engine selection.

Error: Different memory utilities for allocating device memory
--------------------------------------------------------------

The memory utilities are typically consumer dependent and written on an as-needed basis.
cuDNN provides a surface utility for their samples, for example.

To fix the issue, use ``MigratableMemory<type>``, a utility that can automatically migrate data between the host and device (it also works as a stand-in).
If you want to manage dims/strides more carefully, use the ``Tensor`` utility class. Both of these classes can be found in the ``hipdnn_data_sdk::utilities`` namespace.
