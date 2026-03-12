.. meta::
  :description: hipDNN has a plugin-based architecture in order to allow contributors and users to extend hipDNN without modifying the core library.
  :keywords: hipDNN, ROCm, API,

.. _architecture:

******************************
hipDNN high-level architecture
******************************

hipDNN has a plugin-based architecture in order to allow contributors and users to extend hipDNN without modifying the core library.
hipDNN has support for engine plugins, which provide the kernels to solve graphs.

.. note::

  See :ref:`backend-architecture` for a more granular breakdown of the system architecture and the backend API.

The Hipdnn library is structured as three primary components:

- **Frontend**: The hipDNN frontend is a header-only C++ library that provides an industry standard API for interacting with hipDNN. The frontend wraps the backend C API to provide a more user-friendly C++ interface, encapsulating much of the detail required when creating graphs using the backend API.
- **Backend**: The hipDNN backend is a shared library which provides a C API for hipDNN. The backend is the core component of hipDNN which acts as a plugin loader and manager, connecting problems to engines optimized to solve them.
- **Engine plugins**: The hipDNN engine provider plugins are shared libraries responsible for matching the operations they provide to graphs, and executing the operations on the supported hardware. Plugins will continue to be added over time to provide additional operational support or performance improvements. See :ref:`plugin-support` for more information.

The frontend API defines tensors and attaches them to operational nodes on a graph. The graph is then lowered through the backend APIs where it's examined by each plugin's engine to determine a match, at which a point an execution plan is established with the preferred matched engine.
The frontend then allocates memory and populates the data used by the graph, which is then dispatched to the selected plugin's engine to run.

Header-only SDKs provide shared utilities and interfaces.
hipDNN provides three SDKs:

- :ref:`data` (graph schemas and data structures)
- :ref:`plugin-sdk` (plugin API and utilities)
- :ref:`test-sdk` (testing utilities and CPU reference implementations)

Architecture overview
=====================

.. image:: ../images/hipDNN_Architecture.png

Frontend
--------

The frontend provides a user-friendly C++ interface to hipDNN, wrapping the lower-level C API provided by the backend.

Key characteristics:

- **Header-only C++ library**: There are no compiled libraries, which simplifies integration.
- **Dependencies**: The frontend is dependent on the :ref:`backend` and :ref:`data`.
- **Purpose**: It provides the API for users consuming hipDNN.
- **Expected usage**: The frontend should be consumed as a header-only dependency in user projects.

Frontend architecture
~~~~~~~~~~~~~~~~~~~~~

.. _graph:

Graph class
^^^^^^^^^^^

The central abstraction in the frontend is the ``Graph`` class, which:

- Manages the construction of operation graphs.
- Handles the creation and configuration of nodes.
- Orchestrates the execution workflow.

.. _nodes:

Nodes
^^^^^

Nodes represent individual operations within a graph (for example, ``BatchnormNode``, ``PointwiseNode``).

- Nodes encapsulate their specific attributes and tensor connections.
- The Frontend uses the Backend API to convert the graph to backend descriptors for engine consumption.

See :ref:`plugin-support` for a detailed list of the supported operations.

.. _attributes:

Attributes
^^^^^^^^^^

Attributes configure the behavior of nodes.

- Each node type has corresponding attribute classes (for example, ``Batchnorm_attributes``).
- Attributes include operation-specific parameters like epsilon, momentum, etc.

.. _tensor-attributes:

TensorAttributes
^^^^^^^^^^^^^^^^

Tensors are attributes that define the shape of the data processed by operations in the graph.
Tensors are attached to graph operation nodes and determine:

- The dimensions of the data.
- How the data is packed in memory.
- The data's type.


Simplified workflow example
^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref: `build-execute` for a simplified workflow example.

SDKs and plugin architecture
============================

This section is relevant primarily for plugin developers who want a breakdown of the system architecture for the SDK and plugin components.

SDKs
----

hipDNN provides three header-only SDK libraries that serve as the foundation for communication between different components.

.. _data:

Data SDK (``data_sdk``)
~~~~~~~~~~~~~~~~~~~~~~~

The Data SDK contains ``FlatBuffers`` schemas and data structures for graph representation.
The serialized structures allow data to be marshalled and passed between the backend and plugins in a type-safe and highly version-compatible fashion.

- **Dependencies**: ``FlatBuffers`` and ``spdlog``.
- **Purpose**: Provides data structures and serialization for graphs, tensors, and configurations.
- **Expected usage**: Consumed by the backend and plugins for graph data handling.
- **Core functionality**:

  - ``FlatBuffer`` schema definitions for graphs, nodes, and attributes.
  - Data structures for deserializing serialized graphs.
  - Logging utilities and type helpers (``half``, ``bfloat16``, etc.).

.. _plugin-sdk:

Plugin SDK (``plugin_sdk``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Plugin SDK contains the plugin API and utilities for creating engine plugins.

- **Dependencies**: :ref:`data`.
- **Purpose**: Provides the interface and utilities for plugin development.
- **Expected usage**: Consumed by plugin projects.
- **Core functionality**:

  - Plugin API definitions (for example, ``hipdnnEnginePluginCreate`` and ``hipdnnEnginePluginExecuteOpGraph``).
  - Base classes for engine implementation.
  - Utilities for plugin development.

.. _test-sdk:

Test SDK (``test_sdk``)
~~~~~~~~~~~~~~~~~~~~~~~

The Test SDK provides utilities for testing plugins.

- **Dependencies**: :ref:`data` and :ref:`plugin-sdk`.
- **Purpose**: Provides testing infrastructure for plugin validation.
- **Expected usage**: Consumed by plugin test suites.
- **Core functionality**:

  - CPU reference implementations for validation (convolution, batchnorm, etc.).
  - Test utilities (tolerances, seeds, logging).
  - Mock objects for unit testing.


Plugin engines
--------------

Engines, provided by each plugin, provide the actual computational implementations for hipDNN graphs.

Key characteristics:

- **Separate installable projects**: Independent development and deployment.
- **Dependencies**: :ref:`data` and :ref:`plugin-sdk` (and plugin-specific dependencies as needed).
- **Purpose**: Provides engines which are capable of solving graphs.
- **Expected usage**: Loaded at runtime by the backend.

Engine types
------------

Static kernel engines
~~~~~~~~~~~~~~~~~~~~~

- Provides pre-compiled kernels for specific operations.
- Only handles specific configurations.
- For example, the MIOpen Provider plugin.
- **Advantages**:

  - Highly optimized for supported cases.
  - Predictable performance.
  - Lower compilation overhead.

Dynamic kernel engines
~~~~~~~~~~~~~~~~~~~~~~

- Generate kernels at runtime based on graph structure.
- Broad support: Handles general graph patterns.
- For example, future JIT-compilation plugins
- **Advantages**:

  - Flexible operation fusion.
  - Support for novel graph patterns.
  - Adaptable to hardware capabilities.

See :ref:`develop-plugins` for information on developing and using plugins.

Plugin architecture
-------------------

Plugin loading
~~~~~~~~~~~~~~~

- The backend discovers plugins at runtime via the default plugin path, environment variables, or by using ``hipdnnSetEnginePluginPaths_ext`` to provide additional paths from which to load the plugins.
- Each plugin exports standard entry points defined in the Plugin SDK.

Engine management
~~~~~~~~~~~~~~~~~

- Each plugin can provide multiple engines.
- Engines must have fixed globally-unique IDs that remain constant for each run.
- Plugins determine which engines are applicable for a given graph.

Key plugin functions
~~~~~~~~~~~~~~~~~~~~

.. code:: c

  // Get all available engine IDs
  hipdnnEnginePluginGetAllEngineIds(engine_ids, max_engines, num_engines);

  // Check which engines can solve a graph
  hipdnnEnginePluginGetApplicableEngineIds(handle, graph, engine_ids, max, num);

  // Create execution context for a specific engine
  hipdnnEnginePluginCreateExecutionContext(handle, config, graph, context);

  // Execute the graph
  hipdnnEnginePluginExecuteOpGraph(handle, context, workspace, buffers, num_buffers);

.. _backend:

Backend
-------

The backend is the core engine of hipDNN responsible for managing plugins and orchestrating graph execution.

Key characteristics
~~~~~~~~~~~~~~~~~~~

- **Installable library**: C API with ABI for language interoperability, which is dynamically loadable.
- **Dependencies**: :ref:`data`.
- **Purpose**: Provides a stable C API for interacting with the hipDNN kernel providers.
- **Expected usage**: Library linked to the frontend API and expert user projects that provides access to the backend API.

Workflow
~~~~~~~~

1. **Create a Graph**: Build an operation graph using the frontend.
2. **Create heuristic descriptor**: Initialize with the graph and desired heuristic mode.
3. **Get engine configs**: Query available engine configurations from the heuristic.
4. **Create execution plan**: Combine selected engine config with the graph.
5. **Run execution plan**: Execute with variant pack containing tensor data.

.. code:: c

  // Simplified Backend workflow
  hipdnnBackendDescriptor_t graph_desc, heuristic_desc, config_desc, plan_desc, variant_desc;

  // 1. Create graph (from serialized data)
  hipdnnBackendCreateAndDeserializeGraph_ext(&graph_desc, serialized_graph, size);

  // 2. Create and configure heuristic
  hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristic_desc);
  hipdnnBackendSetAttribute(heuristic_desc, HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, ...);
  hipdnnBackendFinalize(heuristic_desc);

  // 3. Get engine configurations
  hipdnnBackendGetAttribute(heuristic_desc, HIPDNN_ATTR_ENGINEHEUR_RESULTS, ...);

  // 4. Create execution plan
  hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan_desc);
  hipdnnBackendSetAttribute(plan_desc, HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, ...);
  hipdnnBackendFinalize(plan_desc);

  // 5. Execute
  hipdnnBackendExecute(handle, plan_desc, variant_desc);

Error handling strategy
-----------------------

hipDNN uses a layered error handling approach designed to be robust across C/C++ boundaries:

-  **Plugins**: Plugin entry points return ``hipdnnPluginStatus_t`` codes. Internal exceptions are caught at the plugin boundary and converted to status codes. Error strings are stored in thread-local storage via ``PluginLastErrorManager``.
-  **Frontend (C++ API)**: The C++ frontend checks ``hipdnnStatus_t`` codes from the backend. On failure, it retrieves the detailed error message via ``hipdnnGetLastErrorString`` and returns an ``Error`` object containing the error code and description. The frontend utilizes *value-based error handling* rather than throwing exceptions.
-  **Backend (C API)**: All public API functions return ``hipdnnStatus_t`` codes. The backend catches any internal C++ exceptions, converts them to the appropriate status code, and stores the exception message. Users can retrieve descriptive error messages using ``hipdnnGetLastErrorString``.

Memory management
=================

hipDNN adopts a caller-owned memory model:

-  **Tensor data**: The user is responsible for allocating and managing device memory for input and output tensors. These pointers are passed to the backend via the *Variant Pack*.
-  **Workspace memory**: Some graph executions require temporary scratch memory. The backend calculates the required size during the Execution Plan phase (``HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE``). The user must allocate this memory and pass the pointer during execution.
-  **Host memory**: API descriptors and graph structures manage their own host resources. Backend API users must explicitly destroy descriptors using ``hipdnnBackendDestroyDescriptor``.

Thread safety
=============

- **Library Handle** (``hipdnnHandle_t``): This handle is *not* thread-safe. Users should create a unique handle for each thread or use external synchronization locks when sharing a handle across threads.
- **Descriptors**: Read-only access to finalized descriptors is thread-safe. Modifying a descriptor while it is being used in another thread is undefined behavior.
