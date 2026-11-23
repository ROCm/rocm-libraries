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

- **NCL**: Batch, Channels, Length (1D, channel-first)
- **NLC**: Batch, Length, Channels (1D, channel-last)
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
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹
   * - Batchnorm Inference + DRelu + Backward
     - ``FP16``, ``BFP16``, ``FP32``
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Fused graph³
   * - Batchnorm Training
     - ``FP16``, ``BFP16``, ``FP32``
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹, no running stats⁴
   * - Batchnorm Training + Activation
     - ``FP16``, ``BFP16``, ``FP32``
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Fused graph³ ⁴
   * - Batchnorm Backward
     - ``FP16``, ``BFP16``, ``FP32``
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹
   * - Convolution Dgrad
     - ``FP16``, ``BFP16``, ``FP32``
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only², explicit ``dx`` dimensions⁶, applicability limits⁹
   * - Convolution Forward
     - ``FP16``, ``BFP16``, ``FP32``
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only², applicability limits⁹
   * - Convolution Forward + (Bias) + Activation⁵
     - ``FP16``, ``BFP16``, ``FP32``
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Fused graph²³, applicability limits⁹
   * - Convolution Wgrad
     - ``FP16``, ``BFP16``, ``FP32``
     - NCL, NLC, NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only², explicit ``dw`` dimensions⁶, applicability limits⁹
   * - Pointwise Activation (standalone)
     - ``FP16``, ``FP32``
     - NCHW, NHWC
     - Single-node graph
   * - Pointwise Binary Elementwise (standalone)
     - ``FP16``, ``FP32``
     - NCHW, NCDHW
     - Single-node graph⁷
   * - Reduction (standalone)
     - ``FP16``, ``FP32``
     - Any, rank 1-5
     - Single-node graph⁸

.. note::

  - For annotations ¹ through ⁴ and ⁶ through ⁹, see :ref:`operations`.
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
- Bias operand (the ADD operand that is not the convolution output)

    - Rank: at most the convolution output rank; a lower-rank bias is right-aligned against the output, so its missing leading dimensions are 1
    - Dims, once aligned: ``[1, K, 1, ..]``, where K is the convolution output channel count
    - Channel stride: 1 when K > 1
    - Data type: the input data type

Activation node
~~~~~~~~~~~~~~~

- Compute data type: ``FP32``
- Activation mode: RELU_FORWARD
- Supports
    - Standard ReLU: no clips set, or ``relu_lower_clip`` = 0
    - Clipped ReLU: ``relu_upper_clip`` set
    - CLAMP: ``relu_lower_clip`` and ``relu_upper_clip`` set
- ``relu_lower_clip_slope`` must be unset or 0 (leaky ReLU is not supported when fused)

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
- ⁶ **Convolution Backward Output Dimensions**: Convolution Dgrad requires explicit ``dx`` dimensions matching the original forward input tensor. Convolution Wgrad requires explicit ``dw`` dimensions matching the original forward weights tensor. These dimensions aren't inferred from ``dy`` and the other convolution tensors because floor division in the forward spatial formula can make multiple input or kernel shapes map to the same output-gradient shape. Output strides may still be inferred once explicit dimensions are present.

- **Activation Functions (fused)**: Supports ReLU, Clipped ReLU (with configurable upper clip), and CLAMP (with configurable lower/upper clips). Leaky ReLU (non-zero ``relu_lower_clip_slope``), Sigmoid and Tanh are not supported when fused onto a producer op; use a standalone activation node for those.
- **Standalone Activation**: A single-node pointwise graph supports ReLU, Clipped ReLU (configurable upper clip), CLAMP (configurable lower/upper clips), Leaky ReLU (configurable negative slope), Sigmoid and Tanh.
- **ReLU Parameters**: An unset ``relu_lower_clip`` is treated as 0, an unset ``relu_upper_clip`` as no upper bound, and an unset ``relu_lower_clip_slope`` or a slope of exactly 0 as no leaky slope, so equivalent encodings select the same MIOpen activation. A non-zero ``relu_lower_clip`` without an upper clip is not supported.
- ⁷ **Standalone Binary Elementwise**: A single-node pointwise graph supports ADD, SUB, MUL, MAX and MIN across ranks 3-5, with packed channels-first strides and ``FP16``/``FP32`` datatypes only. ``in_0`` must be the full-shape operand; only ``in_1`` may broadcast per-axis at matching rank, and all three tensors must share one datatype. In-place execution (``out_0`` aliasing an input) isn't supported.
- ⁸ **Standalone Reduction**: A single-node reduction graph supports ADD, MUL, MIN, MAX, AMAX, AVG, NORM1 and NORM2 with an ``FP32`` compute type. Input and output share one datatype and one rank (1-5), and each output dimension is either 1 (reduced) or equal to the input dimension, with at least one dimension reduced. MUL_NO_ZEROS, ``BFP16``, deterministic reduction (``is_deterministic``) and in-place execution aren't supported; reduction is only available through the default (non-deterministic) engine.
- ⁹ **Convolution Applicability**: The plugin declines the convolution graphs below. A declined graph falls to another engine, or no engine is found for it. These are limits of the plugin's MIOpen-backed convolution engines, not errors in the graph: for graphs of these kinds MIOpen can return wrong output or fault. The rules are conservative and also decline some graphs MIOpen computes correctly. The data type, layout, packing and size rules apply to the plain Convolution Forward, Dgrad and Wgrad rows; the fused row is subject to the last rule only.

  - **Data type** (plain Convolution Forward, Dgrad and Wgrad): the three tensors (``x``, ``w``, ``y``; ``dy``, ``w``, ``dx``; ``x``, ``dy``, ``dw``) must share one data type, ``FP32``, ``FP16`` or ``BFP16``.
  - **Layout** (plain Convolution Forward, Dgrad and Wgrad): the three tensors must share one layout: all channels-first (NCL, NCHW, NCDHW) or all channels-last (NLC, NHWC, NDHWC). A packed tensor is judged by its exact canonical strides, including those of extent-1 dimensions; a non-packed tensor is judged by its stride order. Forward also accepts three packed tensors whose strides, read in one layout's order, never increase, extent-1 dimensions included: for example a 1x1 convolution on an ``[N, C, 1, 1]`` channels-first input with a channels-last output, or a batch-1 input sliced from a wider channel dimension.
  - **Packing, plain Convolution Forward**: a non-packed tensor (the strides do not address exactly the element count) is declined in a grouped convolution and in a 3D channels-last convolution. Any tensor, packed or not, is declined when its strides may let two elements share an address (a zero, negative or overlapping stride; the check is conservative). Other non-packed tensors, such as padded rows or a channel slice, are accepted.
  - **Packing, plain Convolution Dgrad and Wgrad**: a channels-first non-packed tensor is accepted; a channels-last non-packed tensor is declined.
  - **Size, plain Convolution Dgrad**: ``dy`` and ``dx`` must each have at most 2^31 - 1 elements.
  - **Fused Convolution Forward + (Bias) + Activation**: none of the rules above is applied to its ``x``, ``w`` and ``y``. The graph is declined when the bias operand does not meet the requirements under :ref:`detailed` (dims, channel stride and data type), or when the final activation output does not have the data type of ``x`` and the dims of the convolution output, packed in the same layout as ``x``. An ADD with any other operand, for example a residual of the output's shape, is not treated as a bias and the graph is declined.
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

The ``HIPDNN_FORCE_BENCHMARKING`` environment variable overrides this knob for the whole process. See :ref:`engine-selection-variables`.

The MIOpen provider does not report ``SUPPORTS_GRAPH_CAPTURE``. Benchmarking inside HIP stream capture fails. For a conditional workaround for convolutions, see :ref:`capture-miopen`.

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
