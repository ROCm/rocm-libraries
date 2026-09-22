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

Add ``hipdnn_frontend`` to your CMake project. See :ref:`add-hipdnn-steps` in :ref:`add-project` for more information.

This simplified sample code creates a graph, creates tensors, and adds operations before building and executing them:

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
  // The following code to manage host and device memory may need to be customized or
  // replace with code more appropriate to your environment. The utilities::Tensor
  // class is provided for assistance. This class ensures sufficient memory is
  // allocated on the host and GPU for each tensor and uses the MigratableMemory
  // class to facilitate synchronizing data between host and device.
  utilities::Tensor<DataType_t::FLOAT> xTensor(x->get_dim(), TensorLayout::NCHW);
  utilities::Tensor<DataType_t::FLOAT> scaleTensor(scale->get_dim());
  utilities::Tensor<DataType_t::FLOAT> biasTensor(bias->get_dim());
  utilities::Tensor<DataType_t::FLOAT> meanTensor(mean->get_dim());
  utilities::Tensor<DataType_t::FLOAT> invVarianceTensor(inv_var->get_dim());
  utilities::Tensor<DataType_t::FLOAT> yTensor(y->get_dim(), TensorLayout::NCHW);

  // [... populate Tensor input data on host ...]

  // Allocate GPU memory for each tensor and pass to the engine via the execute()
  // function's variantPack parameter. The code below to allocate and manage GPU
  // memory may need to be customized for your environment or application. The
  // MigratableMemory::deviceData() function will sync the tensor data in host
  // memory to GPU memory and return the GPU memory address of the data.
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

Variant pack
------------

The variant pack specifies which GPU memory address correspond to which tensors.
The tensors are added to the variant pack using the tensor UID as the key and the GPU memory address as the value.
It's the responsibility of the application to ensure sufficient GPU memory is allocated for each tensor, and to ensure data is synchronized between the GPU and host memory.
The Data SDK ``utilities::Tensor`` class assists with tensor host and GPU memory management, though it may not be suitable for all applications.

Measure execution time
======================

Use ``Graph::execute_timed_ext()`` to execute the active plan once and obtain an
``ExecutionTiming`` result. Allocate buffers and workspace before the call. The call
waits for timing to complete, but it does not perform hidden warmups or retries.

Check the returned ``Error`` before using the timing:

* ``DEVICE_ONLY`` means that a stall gate excluded host submission gaps.
* ``UNSTALLED`` means that the measurement did not use the gate. Depending on the
  runtime and engine, the event interval may include host submission time.
* ``INVALID`` means that no usable elapsed value is available.

Do not rank ``DEVICE_ONLY`` and ``UNSTALLED`` samples together. Valid elapsed values
are finite and non-negative; zero is valid for work below the event timer's resolution.

If the host does not release the gate within its watchdog budget, the watchdog writes
the release signal. Execution can complete successfully, but ``timedOut`` is true,
``quality`` is ``INVALID``, and ``elapsedMs`` is empty. This does not disable later
timed executions. A bad ``Error`` also invalidates the timing; it does not imply
that no device work ran.

Python returns ``(Error, ExecutionTiming)`` from ``graph.execute_timed_ext()``. The
corresponding result fields are ``quality``, ``timed_out``, and ``elapsed_ms``.

Autotune and ingestor benchmarking apply a comparison-local recovery policy. If a
device-only pass times out or obtains an unstalled sample, they stop that pass,
discard its measurements, and rerun all candidates unstalled once. A later
independent comparison can attempt device-only timing again.

Reuse a backend profiling context
--------------------------------

Backend API callers can reuse ``HIPDNN_BACKEND_PROFILING_CONTROL_EXT``:

1. Create the descriptor and set ``HIPDNN_ATTR_PROFILING_HANDLE_EXT``.
2. Optionally set ``HIPDNN_ATTR_PROFILING_STALL_ARM_EXT`` before recording start.
3. Set ``HIPDNN_ATTR_PROFILING_START_EXT``, execute on the bound stream, and set
   ``HIPDNN_ATTR_PROFILING_STOP_EXT``.
4. Release the stall and finalize the descriptor. Finalization also releases a
   pending stall before synchronizing the stop event.
5. Read ``HIPDNN_ATTR_PROFILING_STALL_TIMED_OUT_EXT`` and
   ``HIPDNN_ATTR_PROFILING_STALL_USED_EXT`` before using
   ``HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT``. No timeout alone does not prove
   device-only timing.
6. Set ``HIPDNN_ATTR_PROFILING_RESET_EXT`` before the next measurement.

``RESET_EXT`` is a scalar ``HIPDNN_TYPE_BOOLEAN`` trigger; its value is ignored.
It is also valid on a fresh or incomplete descriptor. It releases and drains
unfinished work, clears the old result, and retains the events and gate.
Result getters remain unavailable until the next successful finalization.

Keep the handle, stream, and current HIP device unchanged while using the context.
Rebinding requires a new descriptor. Do not reset a descriptor while another thread
reads it. Direct ``StallGate`` and Python ``HipStallGate`` users must synchronize the
previously armed stream before re-arming; ``release()`` alone does not guarantee
that its wait packet has retired.
