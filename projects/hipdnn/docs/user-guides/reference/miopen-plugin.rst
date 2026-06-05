.. meta::
  :description: The MIOpen provider plugin serves as the kernel provider. It employs a modular C++ architecture, largely decoupled from the API layer.
  :keywords: hipDNN, ROCm, API, MIOpen

.. _miopen-provider:

*********************************
MIOpen provider operation support
*********************************

The MIOpen provider adds a number of MIOpen operations to hipDNN.
`MIOpen <https://rocm.docs.amd.com/projects/MIOpen/en/latest/index.html>`_ is an AMD ROCm deep-learning primitives library for GPUs.
It implements fusion to optimize memory bandwidth and GPU launch overheads.
It also implements different algorithms to optimize convolutions for different filter and input sizes.

.. _operation-support:

Operation support
=================

These are the supported datatypes:

- ``FP16``: Half-precision floating point (16-bit)
- ``BFP16``: Brain floating point (16-bit)
- ``FP32``: Single-precision floating point (32-bit)

These are the supported layouts:

- **NCHW**: Batch, Channels, Height, Width (2D, channel-first)
- **NHWC**: Batch, Height, Width, Channels (2D, channel-last)
- **NCDHW**: Batch, Channels, Depth, Height, Width (3D, channel-first)
- **NDHWC**: Batch, Depth, Height, Width, Channels (3D, channel-last)

This table lists all operations supported in hipDNN:

.. list-table::
   :widths: 3 3 3 5
   :header-rows: 1

   * - Operation
     - Datatypes
     - Layouts
     - Notes
   * - Batchnorm Inference with Variance
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹
   * - Batchnorm Inference + DRelu + Backward
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph³
   * - Batchnorm Training
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹, no running stats⁴
   * - Batchnorm Training + Activation
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph³ ⁴
   * - Batchnorm Backward
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹
   * - Convolution Dgrad
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²
   * - Convolution Forward
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²
   * - Convolution Forward + (Bias) + Activation⁵
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph²³
   * - Convolution Wgrad
     - ``FP16``, ``BFP16``, ``FP32``
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²

.. note::

  - For annotations ¹ through ⁴, see :ref:`operations`.
  - For annotation ⁵, see :ref:`detailed` for more information.

.. _detailed:

Detailed requirements
=====================

Convolution Forward + (Bias) + Activation
-----------------------------------------

Convolution forward node
~~~~~~~~~~~~~~~~~~~~~~~~

- Compute data type: ``FP32``

- Y tensor

    - Virtual
    - Data type: ``FP32`` or the input data type (the latter only if bias is used)

Bias node (optional)
~~~~~~~~~~~~~~~~~~~~

- Compute data type: input data type
- Output tensor
    - Virtual
    - Data type: ``FP32`` or the input data type

Activation node
~~~~~~~~~~~~~~~

- Compute data type: ``FP32``
- Activation mode: RELU_FORWARD
- Supports
    - No clipping
    - ``relu_lower_clip`` set
    - ``relu_lower_clip`` and ``relu_upper_clip`` set

.. _operations:

Operation notes
================

- ¹ **Batchnorm Operations**: Only spatial batchnorm mode is supported. Spatial mode computes statistics over the batch (N) and spatial dimensions (H, W, or D, H, W) for each channel.
- ² **Convolution Operations**: Only cross-correlation convolutions are supported. True mathematical convolution (with kernel flipping) is not yet implemented. In practice, cross-correlation is the standard operation used in modern deep learning frameworks.
- ³ **Fused Operations**: Fused graph patterns combine multiple operations.

  - **Batchnorm Inference + DReLU + Backward**: Combines batchnorm inference, activation backward (DReLU), and batchnorm backward.
  - **Batchnorm Training + Activation**: Combines batchnorm training with forward activation.
  - **Convolution Forward + (Bias) + Activation**: Combines convolution forward, optional bias addition, and forward activation.

- ⁴ **Batchnorm Training Running Statistics**: Batchnorm training only supports computing batch statistics (mean and inverse variance) without updating running statistics.

- **Activation Functions**: Supports ReLU, Clipped ReLU (with configurable upper clip), and CLAMP (with configurable lower/upper clips).
- **Sparse Support**: All operations only work with dense tensors.

Knobs
=====

The MIOpen provider plugin supports :ref:`knobs` that control kernel selection, performance tuning, and memory usage. These knobs allow you to optimize MIOpen's behavior for your specific workload and hardware configuration.

The MIOpen provider plugin supports two types of knobs:

- **Global Knobs**: Standard knobs available for all engines (namespace: ``global.*``)
- **Custom Knobs**: Operation-specific knobs provided dynamically based on the graph (there are no custom knobs yet)
.. Is custom knob support not operational yet?

This table lists all configuration knobs supported by the MIOpen Provider plugin:

.. |br| raw:: html

    <br />

.. list-table::
   :widths: 3 3 3 3 3 5
   :header-rows: 1

   * - Knob
     - Type
     - Scope
     - Default
     - Valid Range
     - Description
   * - ``global.benchmarking``
     - Integer (int64)
     - All operations
     - 0 (disabled)
     - 0-1
     - Enable benchmarking mode for kernel selection
   * - ``global.workspace`` |br| ``_size_limit``
     - Integer (int64)
     - Convolution operations only
     - Maximum
     - Dynamic (solver-dependent)
     - Maximum workspace memory in bytes

.. note::

  The ``global.workspace_size_limit`` knob is only available for convolution operations (Forward, Backward Data, Backward Weights). It's not supported for batchnorm or other operations.

Knob benchmarking
-----------------

The MIOpen Provider plugin uses the ``global.benchmarking`` knob to control whether MIOpen performs kernel benchmarking to find the optimal solver for a given operation. It's an int64 integer-type knob and has these values:

- ``0`` (Benchmarking disabled (default)): MIOpen uses heuristics to select a kernel. The first execution is relatively fast, but it may not use the optimal kernel for your specific configuration.
- ``1`` (Benchmarking enabled): MIOpen benchmarks multiple solver candidates. The first execution is slower, but subsequent executions use the cached optimal solver which provides the best performance for production workloads. It has minimal overhead and typically results in a 10-50% performance improvement over the default heuristic selection.

Caching
~~~~~~~

Benchmark results are cached in MIOpen's performance database. The default location is ``~/.config/miopen/`` on Linux.
The cache persists across application runs. It's specific to the GPU model, operation parameters, tensor dimensions, and data types.

Code sample
~~~~~~~~~~~

.. code:: cpp

  // Enable benchmarking
  std::vector<KnobSetting> settings;
  settings.emplace_back("global.benchmarking", 1);
  graph.create_execution_plan_ext(engineId, settings);

Workspace size limit
--------------------

The MIOpen provider plugin uses the ``global.workspace_size_limit`` knob (Integer (int64)) to limit the maximum workspace memory that MIOpen solvers can use for convolution operations (Forward, Backward Data, and Backward Weights).

The valid range for the knob is dynamic, that is, determined at runtime based on the available MIOpen solvers with this workflow:

1. MIOpen queries all available solvers for the specific operation and tensor configuration.
2. Each solver reports its workspace memory requirement.
3. The minimum workspace is the smallest requirement across all solvers.
4. The maximum workspace is the largest requirement across all solvers.
5. Default is set to the maximum for optimal performance.

The range is operation-specific and depends on the:

- Convolution type
- Tensor dimensions and data types
- Available MIOpen solvers for the configuration
- GPU memory constraints

Here's an example range for a specific convolution forward operation:

- **Minimum**: 512 KB (lightweight kernel with minimal workspace)
- **Maximum**: 128 MB (high-performance kernel with large workspace)
- **Default**: 128 MB (use maximum for best performance)

Setting ``global.workspace_size_limit`` to a value lower than the maximum range:

- Constrains solver selection to only those requiring less than the specified workspace.
- Can reduce performance if optimal solvers require more workspace.
- Is useful for memory-constrained systems where total GPU memory is limited.

Setting ``global.workspace_size_limit`` to the maximum range (or not setting it):

- Allows all solvers to be considered.
- Provides the best performance.
- Uses more GPU memory.

.. important::

  The ``global.workspace_size_limit`` knob is dynamically provided *only* when applicable. It won't appear in the knobs list for non-convolution operations (for example, batchnorm, pointwise).

.. warning::

  Setting a workspace limit below the minimum required by all solvers will result in an error when creating the execution plan.

Code sample
~~~~~~~~~~~

.. code:: cpp

  // Query knobs to find valid range
  std::vector<Knob> knobs;
  graph.get_knobs_for_engine(engineId, knobs);

  for (const auto& knob : knobs) {
      if (knob.knobId() == "global.workspace_size_limit") {
          // Check constraints to see valid range
          const auto* constraint = knob.constraint();
          std::cout << "Workspace range: " << constraint->toString() << "\n";
      }
  }

  // Limit workspace to 32 MB
  std::vector<KnobSetting> settings;
  settings.emplace_back("global.workspace_size_limit", 32LL * 1024 * 1024);
  graph.create_execution_plan_ext(engineId, settings);
