.. meta::
  :description: Learn how to build and execute operation graphs in hipDNN.
  :keywords: hipDNN, ROCm, graphs

.. _build-execute:

********************************************
Build and execute operation graphs in hipDNN
********************************************

This topic covers how to use the frontend API to build and execute graph operations in hipDNN.

The hipDNN frontend provides a C++ header-only API for building and executing operation graphs.

Frontend file structure
=======================

Here's the basic frontend file structure with links to the GitHub repository:

- `Library includes <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/frontend/include>`_
- `Samples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples>`_
- `Unit tests <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/frontend/tests>`_


Frontend architecture
=====================

See :ref:`architecture` for a conceptual description of the hipDNN graph, tensors, nodes, and attributes.

See :ref:`plugin-support` for a detailed list of the supported operations.

Simplified workflow example
===========================

Add ``hipdnn_frontend`` to your CMake project. See :ref:`add-hipdnn-steps`.

The following simplified sample code creates a graph, creates tensors, and adds operations before building and executing them:

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

  // Build (finalize) internal operational graph representation.
  graph.build_operation_graph(handle);

  // [Optional: apply knob settings]

  graph.create_execution_plans();
  graph.build_plans();

  // Define memory for each tensor.
  utilities::Tensor<DataType_t::FLOAT> xTensor(x->get_dim(), TensorLayout::NCHW);
  utilities::Tensor<DataType_t::FLOAT> scaleTensor(scale->get_dim());
  utilities::Tensor<DataType_t::FLOAT> biasTensor(bias->get_dim());
  utilities::Tensor<DataType_t::FLOAT> meanTensor(mean->get_dim());
  utilities::Tensor<DataType_t::FLOAT> invVarianceTensor(inv_var->get_dim());
  utilities::Tensor<DataType_t::FLOAT> yTensor(y->get_dim(), TensorLayout::NCHW);

  // [... populate Tensor input data ...]

  // Allocate GPU memory for each tensor.
  std::unordered_map<int64_t, void*> variantPack;
  variantPack[x->get_uid()] = xTensor.memory().deviceData();
  variantPack[scale->get_uid()] = scaleTensor.memory().deviceData();
  variantPack[bias->get_uid()] = biasTensor.memory().deviceData();
  variantPack[mean->get_uid()] = meanTensor.memory().deviceData();
  variantPack[inv_var->get_uid()] = invVarianceTensor.memory().deviceData();
  variantPack[y->get_uid()] = yTensor.memory().deviceData();

  // Execute the graph
  graph.execute(handle, variant_pack, workspace);

  // Make graph output avaialble on host.
  yTensor.memory().markDeviceModified();
  auto yHostPtr = yTensor.memory().hostData();
  // Results available via yHostPtr[].

This is the basic frontend workflow:

1. Instantiate a :ref:`graph` that houses tensors and operations.
2. Create input tensors for the operations within the graph.
3. Add operations, which become :ref:`nodes`, attaching the input tensors to the nodes and creating output tensors from the node's operation. Any :ref:`attributes` you add configure the behavior of these nodes.
4. Continue adding operations and attributes using the output tensors from prior nodes as input tensors to new nodes.
5. The graph is processed to find a matching engine.
6. (Optional) Any non-default engine-specific configuration knobs are applied.
7. Execution plans are built, memory is allocated, tensor data is initialized.
8. The resulting plan is executed on the GPU hardware.

For complete working examples, see the official `samples on GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples>`_.
