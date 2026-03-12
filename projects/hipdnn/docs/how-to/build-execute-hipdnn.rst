.. meta::
  :description: Learn how to build and execute operation graphs in hipDNN.
  :keywords: hipDNN, ROCm, API, how-to

.. _build-execute:

********************************************
Build and execute operation graphs in hipDNN
********************************************

This section covers how to use the frontend API to build and execute graph operations in hipDNN.

The hipDNN frontend provides a C++ header-only API for building and executing operation graphs.

Frontend file structure
=======================

Here's the basic frontend file structure with links to the GitHub repository:

- `Library includes <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/frontend/include>`_
- `Samples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples>`_
- `Unit tests <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/frontend/tests>`_


Frontend architecture
=====================

Refer to :ref: `architecture` for a conceptual description of the hipDNN graph, tensors, nodes, and attributes.

See :ref:`plugin-support` for a detailed list of the supported operations.

Simplified workflow example
===========================

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
3. Add operations which become :ref:`nodes`, attaching the input Tensors to the nodes and creating output tensors from the node's operation. Any :ref:`attributes` you add configure the behavior of these nodes.
4. Continue adding operations and attributes using the output Tensors from prior nodes as input Tensors for new nodes.

The graph is then processed to find a matching engine, the configuration knobs are applied and execution plans are built, memory is allocated, tensor data is supplied, and the resulting plan is executed on the GPU hardware.

For complete working examples, see the official `samples on GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples>`_.
