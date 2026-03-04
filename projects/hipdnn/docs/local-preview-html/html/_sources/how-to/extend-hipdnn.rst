.. meta::
  :description: Extend hipDNN functionality with new functionality.
  :keywords: hipDNN, ROCm, API, how-to 

.. _extend:

***************************
Extend hipDNN functionality
***************************

This section covers how to extend hipDNN with new functionality.

.. important::

   This page is for advanced users such as senior developers, engineers, and system administrators who are looking to add new functionality to hipDNN. Most users should use the default frontend functionality described in :ref:`build-execute`.

Development workflow
====================

Follow these steps and best practices when engaging in any development for plugins and operations for hipDNN.

Typical development flow
-------------------------

- For new operations: ``Data SDK Schema → Frontend Classes → Plugin Implementation → Tests``
- For existing operations in new plugins: ``Plugin Implementation → Integration Tests``

Building and testing
--------------------

- Rebuild hipDNN: After changing hipDNN, you'll need to rebuild. See :ref:`install`, or rebuild the specific targets.
- Test your implementation:
  
  - Use unit tests for individual components.
  - Use integration tests for new and untested end-to-end functionality.

Important considerations
------------------------

- **Backward Compatibility**: Ensure schema changes don't break existing operations.
- **Plugin Discovery**: Ensure discoverability of your plugins through logical placement. For example, engine plugins are loaded from ``hipdnn_plugins/engines/`` relative to the backend library.
- **Error Handling**: Implement proper error reporting through the plugin API.
- **Performance**: Optimization is critical for facilitating plugin adoption.

Debugging tips
---------------

- Enable logging with environment variables. See :ref:`variables` for more info.
- Use integration tests to verify operation behavior.
- Check plugin loading with ``HIPDNN_LOG_LEVEL=info``.
- For plugin issues, check the default plugin path or use custom paths with ``hipdnnSetEnginePluginPaths_ext``.

Add a new plugin
================

Plugins extend hipDNN to support new or additional implementations of kernel engines, benchmarking, and heuristics. 
For comprehensive guidance on plugin development, including architecture details, implementation steps, and examples, see :ref:`develop-plugins`.

.. _add-operation:

Add a new operation
===================

Adding a new operation requires coordinated changes across multiple components.

When adding a completely new operation type (not currently supported in hipDNN), you'll need to:

1. Define the operation in the Data SDK schemas.
2. Create frontend classes.
3. Implement the operation in target plugins.

Data SDK schema changes
-----------------------

If the operation is new to hipDNN, start by defining its data structures:

1. Create the Attribute schema:
   
   1. Add a new ``.fbs`` file in `data_sdk/schemas/ <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/data_sdk/schemas>`_.
   2. Define the operation's attributes (parameters, configurations).
  
   Example: `data_sdk/schemas/batchnorm_attributes.fbs <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/data_sdk/schemas/batchnorm_attributes.fbs>`_

2. Update the Graph schema:

   1. Modify `data_sdk/schemas/graph.fbs <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/data_sdk/schemas/graph.fbs>`_.
   2. Add your new attributes to the ``NodeAttributes`` union.
   3. Include your schema file.

      Example:

      .. code:: flatbuffers
   
        include "your_operation_attributes.fbs";

        union NodeAttributes {
           BatchnormInferenceAttributes,
           PointwiseAttributes,
           ...
           YourOperationAttributes  // Add your new operation
        }

      After updating FlatBuffer schemas, regenerate the C++ headers:

      .. code:: bash

        ninja generate_hipdnn_data_sdk_headers

Frontend implementation
-----------------------

Create C++ classes to expose the operation to users:

1. Create the Node Class:
   
   1. Add the header file in `frontend/include/hipdnn_frontend/node/ <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/frontend/include/hipdnn_frontend/node>`_.
   2. Inherit from the base ``Node`` class.
   
   Example: `frontend/include/hipdnn_frontend/node/BatchnormNode.hpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/frontend/include/hipdnn_frontend/node/BatchnormNode.hpp>`_

2. Create Attribute Classes:

   - Add the corresponding attribute classes in `frontend/include/hipdnn_frontend/attributes/ <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/frontend/include/hipdnn_frontend/attributes>`_.
   - These wrap the FlatBuffer-generated structures.

3. Update frontend tests:

   - Add tests for your new node and attributes.
   - See examples in `frontend/tests/ <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/frontend/tests>`_.

Plugin integration
-------------------

See :ref:`develop-plugins` to implement the operation execution in target plugins.

Troubleshooting
===============

Segmentation faults during graph execution plan build
------------------------------------------------------

If you're seeing segfaults when building execution plans for graphs, this might be caused by Thread Local Storage (TLS) allocation issues (such as static TLS exhaustion) between the executable and dynamically loaded backend plugins.

To resolve this, enable PIC/PIE to ensure compatibility with the plugin loader system (dlopen). This setting instructs CMake to emit position-independent code (for example, via ``-fPIC``  or ``-fPIE``), which is necessary for creating shared libraries or executables that load plugins dynamically.

.. code:: cmake

   set(CMAKE_POSITION_INDEPENDENT_CODE ON)