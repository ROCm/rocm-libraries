.. meta::
  :description: Learn how to build and execute operation graphs in hipDNN.
  :keywords: hipDNN, ROCm, API, how-to 

.. _build-execute:

********************************************
Build and execute operation graphs in hipDNN
********************************************

This section covers how to use the frontend API to build and execute graph operations in hipDNN.

The hipDNN frontend provides a C++ header-only API for building and executing operation graphs. 

.. note::

  The MIOpen Provider plugin serves as the kernel provider. It employs a modular C++ architecture, largely decoupled from the API layer. See :ref:`miopen` for more info.

Frontend file structure
=======================

Here's the basic frontend file structure with links to the GitHub repository:

- `Library includes <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/frontend/include>`_
- `Unit tests <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/frontend/tests>`_
- `Samples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples>`_

Frontend architecture
=====================

The frontend architecture consists of the ``Graph`` class, tensors, nodes, and attributes.

.. _graph:

Graph class
-----------

The central abstraction in the frontend is the ``Graph`` class, which:

- Manages the construction of operation graphs.
- Handles the creation and configuration of nodes.
- Orchestrates the execution workflow.

.. _nodes:

Tensors
-------

Tensors are defined by Data Type, Dimensions, and Layout.

See :ref:`operation-support` for a detailed list of the supported operations.

Nodes
-----

Nodes represent individual operations within a graph:

- Each node type (for example, ``BatchnormNode``, ``PointwiseNode``) inherits from ``INode``.
- Nodes encapsulate their specific attributes and tensor connections.
- Support serialization to Flatbuffer format for backend consumption.

.. _attributes:

Attributes
----------

Attributes configure the behavior of nodes:

- Each node type has corresponding attribute classes (for example, ``Batchnorm_attributes``).
- Attributes include operation-specific parameters like epsilon, momentum, etc.
- Support builder pattern for easy configuration.

Simplified workflow
===================

This example demonstrates sample code that creates a graph, creates tensors, and adds operations before building and executing them. Here's a simplified workflow example:

.. code:: cpp

  // Create a graph
  Graph graph;
  graph.set_compute_data_type(DataType_t::FLOAT);

  // Create tensors
  auto x = Graph::tensor(/* tensor attributes */);
  auto scale = Graph::tensor(/* tensor attributes */);
  auto bias = Graph::tensor(/* tensor attributes */);

  // Add operations
  auto [y, mean, inv_var, _, _] = graph.batchnorm(x, scale, bias, bn_attributes);

  // Build and execute
  graph.build_operation_graph(handle);
  graph.create_execution_plans();
  graph.build_plans();
  graph.execute(handle, variant_pack, workspace);

This is the basic frontend workflow:

1. Instantiate a :ref:`graph` that houses tensors and operations.
2. Create input tensors for the operations within the graph.
3. Add operations which become :ref:`nodes`. Any :ref:`attributes` you add configure the behaviour of these nodes.

For complete working examples, see the official `samples on GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples>`_.

