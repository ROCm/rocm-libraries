.. meta::
  :description: Learn how to migrate your frontend code from cuDNN to hipDNN.
  :keywords: hipDNN, ROCm, migrate, cuDNN

.. _migrate-cudnn:

********************************************
Migrate NVIDIA CUDA cuDNN projects to hipDNN
********************************************

This topic demonstrates how to migrate a NVIDIA CUDA cuDNN project to hipDNN.

Before you begin, ensure hipDNN (ROCm) is installed. See :ref:`prerequisites` for more information.

.. note::

  NVIDIA, CUDA, and cuDNN are trademarks and/or registered trademarks of NVIDIA Corporation.
  hipDNN is not affiliated with or endorsed by NVIDIA.

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
     - ``hipdnnCreate(&handle)``
   * - Handle destruction
     - ``cudnnDestroy(handle)``
     - ``hipdnnDestroy(handle)``
   * - Heuristics modes
     - All cuDNN heuristic modes
     - ``HeuristicMode::A``, ``HeuristicMode::B``, and ``HeuristicMode::FALLBACK``
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

Compile cuDNN frontend source with the compatibility shim
=========================================================

Instead of porting to the ``hipdnn_frontend`` API, you can compile existing cuDNN frontend source
against hipDNN's header-only cuDNN-compatibility shim. Its cuDNN-spelled headers live under
``frontend/include/hipdnn_compatibility/cudnn/`` (for example,
``#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>``) and are installed with the
``hipdnn_frontend`` headers only when hipDNN is built with ``-DHIPDNN_ENABLE_CUDNN_COMPATIBILITY=ON``.
Request them as a package component, so that configuration fails against a hipDNN built without
them:

.. code:: cmake

    find_package(hipdnn_frontend CONFIG REQUIRED COMPONENTS cudnn_compatibility)
    target_link_libraries(my_app PRIVATE hipdnn_frontend)

The shim is source compatible only: rebuild your code against it. Its C entry points take HIP types
(for example, ``cudnnSetStream`` takes a ``hipStream_t``), so CUDA runtime calls in your source still
need porting to HIP.

Replace the cuDNN include and alias the namespace:

.. code:: cpp

    #include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

    namespace cudnn_frontend = hipdnn_frontend::compatibility::cudnn_frontend;

Upstream declares a global ``json`` alias for ``nlohmann::json``; the shim declares it as
``cudnn_frontend::json``. Code that names a bare ``json`` adds ``using json = cudnn_frontend::json;``.

See the samples in `samples/cudnn_shim <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn/samples/cudnn_shim>`_.
The shim's behavior differs from cuDNN in places. See
`Known divergences <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md>`__
for the full list.

cuDNN v9 graph API
------------------

Source written against the v9 graph API (``cudnn_frontend::graph::Graph``) compiles after the include
change. ``HeurMode_t::A`` and ``HeurMode_t::B`` select hipDNN's prediction policies (see
:ref:`Troubleshooting <migrate-cudnn-troubleshooting>`); ``HeurMode_t::OPENSOURCE`` logs a warning
once per process and runs hipDNN's fallback engine selection.

cuDNN v8 API: v0.x builders
---------------------------

The same include provides the cuDNN v8 surface:

* The v0.x builders: ``TensorBuilder``, ``ConvDescBuilder``, ``PointWiseDescBuilder``,
  ``MatMulDescBuilder``, ``ReductionDescBuilder``, ``ResampleDescBuilder``, ``RngDescBuilder``,
  ``OperationBuilder``, ``OperationGraphBuilder``, ``EngineHeuristicsBuilder``, ``EngineBuilder``,
  ``EngineConfigBuilder``, ``EngineFallbackListBuilder``, ``ExecutionPlanBuilder`` and
  ``VariantPackBuilder``, plus filters, errata, ``ExecutionPlanCache`` and ``get_heuristics_list``.
  As upstream, a failed ``build()`` throws ``cudnnException`` unless ``NV_CUDNN_DISABLE_EXCEPTION``
  is defined.
* From the backend C API, only ``cudnnBackendExecute`` and ``cudnnGetLastErrorString``, both passing
  straight through to hipDNN. ``cudnnBackendDescriptor_t`` is ``hipdnnBackendDescriptor_t``, and
  ``get_raw_desc()`` of a plan or variant pack is the hipDNN descriptor. The descriptor calls
  (``cudnnBackendCreateDescriptor``, ``cudnnBackendSetAttribute``, ``cudnnBackendGetAttribute``,
  ``cudnnBackendFinalize``, ``cudnnBackendDestroyDescriptor``) and the ``CUDNN_ATTR_*`` and
  ``CUDNN_TYPE_*`` enums are not provided, so code that calls them directly does not compile; build
  the objects with the builders instead
  (`Backend descriptor C API <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#backend-descriptor-c-api>`__).
* Plan search: include ``<hipdnn_compatibility/cudnn/cudnn_frontend_get_plan.h>`` for
  ``EngineConfigGenerator::cudnnGetPlan`` and ``<hipdnn_compatibility/cudnn/cudnn_frontend_find_plan.h>``
  for ``cudnnFindPlan``, ``cudnnFindPlanAndCache`` and ``time_sorted_plan``, as with upstream.

A convolution forward through the builders:

.. code:: cpp

    cudnn_frontend::Tensor makeTensor(int64_t uid,
                                      const std::vector<int64_t>& dims,
                                      const std::vector<int64_t>& strides)
    {
        return cudnn_frontend::TensorBuilder()
            .setDataType(CUDNN_DATA_FLOAT)
            .setDim(static_cast<int64_t>(dims.size()), dims.data())
            .setStride(static_cast<int64_t>(strides.size()), strides.data())
            .setId(uid)
            .setAlignment(16)
            .build();
    }

    // x, w and y are device pointers.
    cudnnStatus_t convFprop(cudnnHandle_t handle, void* x, void* w, void* y)
    {
        // Packed NCHW: x and y are 16x16x16x16, w is 16x16x3x3; padding 1, stride 1.
        const std::vector<int64_t> xDims = {16, 16, 16, 16}, xStrides = {4096, 256, 16, 1};
        const std::vector<int64_t> wDims = {16, 16, 3, 3}, wStrides = {144, 9, 3, 1};
        const std::vector<int64_t> yDims = xDims, yStrides = xStrides;
        const std::array<int64_t, 2> pad = {1, 1}, stride = {1, 1}, dilation = {1, 1};
        cudnn_frontend::ConvDesc conv = cudnn_frontend::ConvDescBuilder()
                                            .setComputeType(CUDNN_DATA_FLOAT)
                                            .setMathMode(CUDNN_CROSS_CORRELATION)
                                            .setSpatialDimCount(2)
                                            .setPrePadding(2, pad.data())
                                            .setPostPadding(2, pad.data())
                                            .setSpatialStride(2, stride.data())
                                            .setDilation(2, dilation.data())
                                            .build();
        const cudnn_frontend::Operation op
            = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                  .setxDesc(makeTensor('x', xDims, xStrides))
                  .setwDesc(makeTensor('w', wDims, wStrides))
                  .setyDesc(makeTensor('y', yDims, yStrides))
                  .setcDesc(conv)
                  .build();

        std::array<const cudnn_frontend::Operation*, 1> ops = {&op};
        cudnn_frontend::OperationGraph graph = cudnn_frontend::OperationGraphBuilder()
                                                   .setHandle(handle)
                                                   .setOperationGraph(1, ops.data())
                                                   .build();

        cudnn_frontend::EngineHeuristics heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                                                          .setOperationGraph(graph)
                                                          .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                                          .build();
        const int64_t count = heuristics.getEngineConfigCount();
        if(count == 0)
        {
            return CUDNN_STATUS_NOT_SUPPORTED; // No hipDNN engine runs this graph on this device.
        }
        auto& configs = heuristics.getEngineConfig(count);
        cudnn_frontend::ExecutionPlan plan = cudnn_frontend::ExecutionPlanBuilder()
                                                 .setHandle(handle)
                                                 .setEngineConfig(configs[0], graph.getTag())
                                                 .build();

        // Size the workspace from this plan: the size can differ between builds of one configuration.
        void* workspace = nullptr;
        const auto workspaceSize = static_cast<size_t>(plan.getWorkspaceSize());
        if(workspaceSize > 0 && hipMalloc(&workspace, workspaceSize) != hipSuccess)
        {
            return CUDNN_STATUS_ALLOC_FAILED;
        }

        std::array<void*, 3> pointers = {x, w, y};
        const std::array<int64_t, 3> uids = {'x', 'w', 'y'};
        const cudnn_frontend::VariantPack pack = cudnn_frontend::VariantPackBuilder()
                                                     .setWorkspacePointer(workspace)
                                                     .setDataPointers(3, pointers.data())
                                                     .setUids(3, uids.data())
                                                     .build();
        const cudnnStatus_t status
            = cudnnBackendExecute(handle, plan.get_raw_desc(), pack.get_raw_desc());
        (void)hipFree(workspace); // hipFree waits for the device to finish with the workspace.
        return status;
    }

``samples/cudnn_shim/ConvFpropV8.cpp`` in the hipDNN source tree runs this graph end to end.

The main differences from cuDNN on the v8 API follow, each with a link to its known-divergences
section.
`Known divergences: cuDNN v8 API <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#cudnn-v8-api>`__
has the full list.

* **Heuristics:** every ``CUDNN_HEUR_MODE_*`` returns the same list, in hipDNN's fallback order
  (`Heuristic modes <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#heuristic-modes>`__).
* **Handles:** ``OperationGraphBuilder`` needs ``setHandle``; without it ``build()`` returns
  ``CUDNN_STATUS_BAD_PARAM``. ``cudnnBackendExecute`` runs a plan on whichever handle it is given,
  provided the handle is on the plan's device
  (`Operation graphs <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#operation-graphs>`__,
  `Engines, plans and execution <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#engines-plans-and-execution>`__).
* **Convolution mode:** call ``setMathMode(CUDNN_CROSS_CORRELATION)``. A graph that keeps the
  builder's default, ``CUDNN_CONVOLUTION``, has no engine
  (`Builders <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#builders>`__).
* **Scaling:** convolution, pointwise or resample alpha other than 1, or beta other than 0, returns
  ``CUDNN_STATUS_NOT_SUPPORTED``
  (`Operations <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#operations>`__).
* **Tensor alignment:** a tensor aligned below 16 bytes leaves the graph with no engine
  (`Tensors <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#tensors>`__).
* **Unsupported operations:** operations hipDNN does not run, such as ``GEN_STATS``, ``RNG`` and
  ``RESHAPE``, fail at ``build()`` with ``CUDNN_STATUS_NOT_SUPPORTED``
  (`Operations <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#operations>`__).
  A graph that translates but that no engine supports can still build; heuristics then return an
  empty list.
* **Engine numbers:** engine indices order hipDNN's engines by engine ID. NVIDIA engine numbers in
  errata files or hard-coded in your source select unrelated engines, or none
  (`Engines, plans and execution <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#engines-plans-and-execution>`__).
* **Enum values:** the C enums have cuDNN's names but the shim's own values. Don't store or compare
  raw values produced by cuDNN
  (`Enum values <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/KNOWN_DIVERGENCES.md#enum-values>`__).

.. _migrate-cudnn-troubleshooting:

Troubleshooting
===============

Prediction modes return no ranking
----------------------------------

Mode A ranks engines by their graph-level UHD prediction. Mode B prefers a
calibrated configuration prediction and falls back to the graph-level prediction
for each engine. Neither mode benchmarks the graph. Install compatible trained
models for the selected device architecture.

``HeuristicMode::A`` and ``HeuristicMode::B`` are not backend heuristic modes: the
frontend turns them into the ``SelectionHeuristic::ModeA`` and
``SelectionHeuristic::ModeB`` entries of the heuristic descriptor's ordered policy
list, always preceded by ``SelectionHeuristic::Config`` and followed by
``SelectionHeuristic::StaticOrdering``. If no applicable engine has a usable
prediction, the prediction policy declines and ordinary static engine selection
still runs, so ``graph.create_execution_plans({HeuristicMode::B})`` never fails for
want of a model. Setting ``HIPDNN_HEUR_POLICY_ORDER`` overrides the whole list.

Both modes rank by one *ranking metric*, ``tflops`` by default: calibrated
throughput, highest first. ``graph.set_ranking_metric("time")`` ranks by predicted
milliseconds, lowest first, and makes the chosen engine pick its kernel by predicted
time too; ``graph.get_ranking_metric()`` reads the choice back. An unregistered name
returns an ``Error`` and leaves the metric unchanged. An engine whose models do not
cover the requested metric is ranked after those that do, never by another metric,
so if no engine has a model for it the prediction policy declines exactly as above.
``HIPDNN_HEUR_RANKING_METRIC`` overrides the Graph's choice for heuristic selection.

Error: Different memory utilities for allocating device memory
--------------------------------------------------------------

The memory utilities are typically consumer dependent and written on an as-needed basis.
cuDNN provides a surface utility for their samples, for example.

To fix the issue, use ``MigratableMemory<type>``, a utility that can automatically migrate data between the host and device (it also works as a stand-in).
If you want to manage dims/strides more carefully, use the ``Tensor`` utility class. Both of these classes can be found in the ``hipdnn_data_sdk::utilities`` namespace.
