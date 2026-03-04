.. meta::
  :description: The MIOpen Provider plugin serves as the kernel provider. It employs a modular C++ architecture, largely decoupled from the API layer.
  :keywords: hipDNN, ROCm, API, 

.. _miopen:

**********************
MIOpen Provider plugin
**********************

The MIOpen Provider plugin serves as the kernel provider. It employs a modular C++ architecture, largely decoupled from the API layer.

The plugin consists of these components:

- Dependency injection container (``MiopenContainer``): This is the root object that manages the lifecycle and dependencies of all other components. It initializes the ``EngineManager`` and ensures that all necessary services are correctly injected.
- Engine manager (``EngineManager``): The central registry for execution engines. It orchestrates the selection of the appropriate engine for a given operation graph by querying its registered engines.
- Plan builders (``IPlanBuilder``): Each engine is associated with a set of plan builders. These components are responsible for:

  - Applicability: Inspecting an operation graph to determine if the engine can execute it.
  - Resource estimation: Calculating the required workspace size.
  - Plan construction: Creating an executable IPlan object if the graph is supported.

- Plans (IPlan): An IPlan represents a strategy for executing a specific operation graph. It encapsulates all the necessary logic and state to run the routine, abstracting the details from the higher-level engine management.
- C-API Interface: A thin translation layer that exposes these internal C++ components to the backend via the required engine plugin C-API.

.. _operation-support:

Operation support
=================

These are the supported datatypes:

- **FP16**: Half-precision floating point (16-bit)
- **BFP16**: Brain floating point (16-bit)
- **FP32**: Single-precision floating point (32-bit)

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
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹
   * - Batchnorm Inference + DRelu + Backward 
     - FP16, BFP16, FP32 
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph³
   * - Batchnorm Training
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹, No running stats⁴
   * - Batchnorm Training + Activation
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph³ ⁴
   * - Batchnorm Backward 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Spatial mode only¹
   * - Convolution Dgrad 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²
   * - Convolution Forward
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²
   * - Convolution Forward + (Bias) + Activation⁵ 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Fused graph²³
   * - Convolution Wgrad 
     - FP16, BFP16, FP32
     - NCHW, NHWC, NCDHW, NDHWC
     - Cross-correlation only²

.. note::

  - For annotations ¹ through ⁴, refer to :ref:`operations`.
  - For annotation ⁵, see :ref:`detailed` for more information.

.. _detailed:

Detailed requirements
=====================

Convolution Forward + (Bias) + Activation
-----------------------------------------

Convolution forward node
~~~~~~~~~~~~~~~~~~~~~~~~

- Compute data type: FP32
- Y tensor
    - Virtual
    - Data type: FP32 or the input data type (the latter only if bias is used)

Bias node (optional)
~~~~~~~~~~~~~~~~~~~~

- Compute data type: input data type
- Output tensor
    - Virtual
    - Data type: FP32 or the input data type

Activation node
~~~~~~~~~~~~~~~

- Compute data type: FP32
- Activation mode: RELU_FORWARD
- Supports
    - no clipping
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

The MIOpen Provider plugin supports :ref:`knobs` that control kernel selection, performance tuning, and memory usage. These knobs allow you to optimize MIOpen's behavior for your specific workload and hardware configuration.

The MIOpen Provider plugin supports two types of knobs:

- **Global Knobs**: Standard knobs available for all engines (namespace: ``global.*``)
- **Custom Knobs**: Operation-specific knobs provided dynamically based on the graph (no custom knobs currently) 
.. Is custom knob support not operational yet?

This table lists all configuration knobs supported by the MIOpen Provider plugin:

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
     - All Operations
     - 0 (disabled)
     - 0-1
     - Enable benchmarking mode for kernel selection
   * - ``global.workspace_size_limit``
     - Integer (int64)
     - Convolution operations only
     - Maximum
     - Dynamic (solver-dependent)
     - Maximum workspace memory in bytes

.. note::

  The ``global.workspace_size_limit`` knob is *only available* for convolution operations (Forward, Backward Data, Backward Weights). It's not supported for batchnorm or other operations.

Knob benchmarking
-----------------

The MIOpen Provider plugin uses the ``global.benchmarking`` knob to control whether MIOpen performs kernel benchmarking to find the optimal solver for a given operation. It's an Integer (int64) type Knob, and it has these values:

- ``0`` (Benchmarking disabled (default)): MIOpen uses heuristics to select a kernel. The first execution is relatively fast, but it may not use the optimal kernel for your specific configuration.
- ``1`` (Benchmarking enabled): MIOpen benchmarks multiple solver candidates. The first execution is slower, but subsequent executions use the cached optimal solver which provides the best performance for production workloads. It has minimal overhead and typically results in 10-50% performance improvement over the default heuristic selection.

Caching
~~~~~~~

Benchmark results are cached in MIOpen's performance database. The default location is ``~/.config/miopen/`` on Linux.
Cache persists across application runs. It's specific to the GPU model, operation parameters, tensor dimensions, and data types.

Code sample
~~~~~~~~~~~

.. code:: cpp

  // Enable benchmarking
  std::vector<KnobSetting> settings;
  settings.emplace_back("global.benchmarking", 1);
  graph.create_execution_plan_ext(engineId, settings);

Workspace size limit
--------------------

The MIOpen Provider plugin uses the ``global.workspace_size_limit`` knob (Integer (int64)) to limit the maximum workspace memory that MIOpen solvers can use for convolution operations (Forward, Backward Data, Backward Weights).

The valid range for the knob is dynamic, i.e., determined at runtime based on the available MIOpen solvers with this workflow:

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

Here's an example range for a specific convultion forward operation:

- **Minimum**: 512 KB (lightweight kernel with minimal workspace)
- **Maximum**: 128 MB (high-performance kernel with large workspace)
- **Default**: 128 MB (use maximum for best performance)

Setting ``global.workspace_size_limit`` to a value lower than the maximum range:

- Constrains solver selection to only those requiring less than the specified workspace.
- May reduce performance if optimal solvers require more workspace.
- Is useful for memory-constrained systems where total GPU memory is limited.

Setting ``global.workspace_size_limit`` to the maximum range (or not setting it):

- Allows all solvers to be considered.
- Provides the best performance.
- Uses more GPU memory.

.. important::

  The ``global.workspace_size_limit`` knob is dynamically provided *only* when applicable. It won't appear in the knobs list for non-convolution operations (e.g., batchnorm, pointwise).

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

Usage examples
--------------

Query available knobs
~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

  #include <hipdnn_frontend.hpp>

  using namespace hipdnn_frontend;

  // After building the graph
  Graph graph;
  // ... setup and build graph ...

  // Query knobs for MIOpen engine
  std::vector<Knob> knobs;
  auto error = graph.get_knobs_for_engine(MIOPEN_ENGINE_ID, knobs);

  if (error.is_good()) {
      std::cout << "Available knobs for MIOpen engine:\n";
      for (const auto& knob : knobs) {
          std::cout << "  " << knob.knobId() << ": " << knob.description() << "\n";
          std::cout << "  Default: ";

          const auto& defaultVal = knob.defaultValue();
          if (std::holds_alternative<int64_t>(defaultVal)) {
              std::cout << std::get<int64_t>(defaultVal);
          }
          std::cout << "\n";

          if (const auto* constraint = knob.constraint()) {
              std::cout << "  Constraint: " << constraint->toString() << "\n";
          }
          std::cout << "\n";
      }
  }

Combined knob settings
~~~~~~~~~~~~~~~~~~~~~~

.. code:: cpp

  #include <hipdnn_frontend.hpp>

  using namespace hipdnn_frontend;

  Graph graph;
  // ... setup convolution graph ...
  graph.build_operation_graph(handle);

  // Enable benchmarking and set workspace limit
  std::vector<KnobSetting> settings;
  settings.emplace_back("global.benchmarking", 1);
  settings.emplace_back("global.workspace_size_limit", 128LL * 1024 * 1024);

  auto error = graph.create_execution_plan_ext(MIOPEN_ENGINE_ID, settings);

  if (error.is_good()) {
      std::cout << "Execution plan created with custom knob settings\n";
  }

Best practices
==============

For development
---------------

- **Start with defaults**: Use default knob values during initial development.
- **Profile first**: Measure baseline performance before tuning knobs.
- **Query knobs**: Always check available knobs and their constraints using ``get_knobs_for_engine()``.
- **Test incremental changes**: Modify one knob at a time to understand impact.

For production
--------------

- **Enable benchmarking during warm-up**:

  .. code:: cpp

    // Warm-up phase
    std::vector<KnobSetting> warmupSettings;
    warmupSettings.emplace_back("global.benchmarking", 1);
    graph.create_execution_plan_ext(engineId, warmupSettings);

    // Execute a few times to populate cache
    for (int i = 0; i < 5; i++) {
        graph.execute(handle, variantPack, workspace);
    }

- **Use cached results in production**: After warm-up, benchmarking can be disabled as results are cached.
- **Document knob settings**: Keep a record of knob configurations used in production for reproducibility.

For memory-constrained environments
-----------------------------------

- **Query workspace ranges**:

  .. code:: cpp

    // Find minimum and maximum workspace for the operation
    std::vector<Knob> knobs;
    graph.get_knobs_for_engine(engineId, knobs);

    for (const auto& knob : knobs) {
        if (knob.knobId() == "global.workspace_size_limit") {
            // Log constraint to understand valid range
        }
    }

- **Set conservative limits**: Start with a lower workspace limit and increase if performance is insufficient.
- **Balance batch size and workspace**: Reducing workspace allows larger batch sizes, which may offset performance loss.

Error handling
--------------

.. code:: cpp

  auto error = graph.create_execution_plan_ext(engineId, settings);
  if (!error.is_good()) {
      std::cerr << "Failed to create execution plan: " << error.get_message() << "\n";

      // Common errors:
      // - Workspace limit below minimum
      // - Invalid knob ID
      // - Value outside valid range
  }
