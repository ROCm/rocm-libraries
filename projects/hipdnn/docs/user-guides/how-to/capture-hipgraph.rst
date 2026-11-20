.. meta::
  :description: How to run hipDNN graphs inside HIP stream capture (hipGraph) and which engines support it.
  :keywords: hipDNN, ROCm, hipGraph, stream capture, hipStreamBeginCapture, SUPPORTS_GRAPH_CAPTURE

.. _capture-hipgraph:

*************************************
Capture hipDNN graphs in a hipGraph
*************************************

HIP stream capture records the work that you submit to a stream and turns it into a hipGraph.
You can then instantiate the hipGraph once and replay it many times. This topic describes how to
capture ``Graph::execute()`` and which hipDNN engines you can capture.

hipDNN has no equivalent of the cuDNN ``populate_cuda_graph`` and ``update_cuda_graph`` calls.
In the cuDNN compatibility shim, both calls return ``GRAPH_NOT_SUPPORTED``. Stream capture is
the only supported way to put hipDNN work in a hipGraph.

Supported usage
===============

Do all setup before you call ``hipStreamBeginCapture()``:

1. Build the graph and the execution plans.
2. Get the workspace size and allocate the workspace.
3. Allocate every tensor buffer.
4. Set the stream of the hipDNN handle to the stream that you capture.

Inside the capture, call only ``graph.execute(handle, variantPack, workspace)``. Then end the
capture, instantiate the hipGraph, and replay it.

.. code-block:: cpp

   #include <hipdnn_frontend.hpp>

   // Setup, before capture: build the graph, allocate the workspace and all tensor buffers.
   graph->build(*handle);
   int64_t workspaceSize = 0;
   graph->get_workspace_size(workspaceSize);
   void* workspace = nullptr;
   hipMalloc(&workspace, static_cast<size_t>(workspaceSize));
   // ... allocate the tensor buffers and fill the variant pack ...

   hipStream_t stream;
   hipStreamCreate(&stream);
   hipdnn_frontend::setHipdnnHandleStream(handle, stream);

   // Capture only execute().
   hipGraph_t hipGraph;
   hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
   graph->execute(*handle, variantPack, workspace);
   hipStreamEndCapture(stream, &hipGraph);

   hipGraphExec_t hipGraphExec;
   hipGraphInstantiateWithFlags(&hipGraphExec, hipGraph, 0);

   // Replay. The hipGraph holds the device pointers from capture time, so
   // update the input data in place and keep the same pointers.
   hipGraphLaunch(hipGraphExec, stream);
   hipStreamSynchronize(stream);

Replays read and write the device pointers that were in the variant pack at capture time. To
change the input of a replay, write new data to the same buffers. Do not expect a new pointer in
the variant pack to take effect.

A replay also keeps the workspace pointer and the engine's internal state from capture time. For
example, with ``HIPBLASLT_ENGINE`` a replay keeps the pointer to hipBLASLt's synchronization
flags, which belong to the hipBLASLt handle of the hipDNN handle, whichever stream the graph was
captured on. Graphs captured from the same handle, or that share a workspace, must not run at the
same time as each other or as other work on that handle. Order them on one stream, or give each
graph that you replay concurrently its own handle and workspace.

The capture works in ``hipStreamCaptureModeGlobal`` (the PyTorch default),
``hipStreamCaptureModeThreadLocal``, and ``hipStreamCaptureModeRelaxed``.

Check whether an engine supports capture
========================================

An engine that supports capture reports the ``SUPPORTS_GRAPH_CAPTURE`` behavior note. The note
means that the engine is safe to capture with default settings, that is, with benchmarking off.
The first execution of a plan can be captured.

Get the engine of the active plan, and then read its behavior notes. Use the graph after
``build()``:

.. code-block:: cpp

   int64_t engineId = 0;
   graph->get_execution_plan_engine_id(engineId);

   std::vector<hipdnn_frontend::BehaviorNote> notes;
   auto err = graph->get_behavior_notes_for_engine(engineId, notes);

   const bool capturable
       = err.is_good()
         && std::find(notes.begin(),
                      notes.end(),
                      hipdnn_frontend::BehaviorNote::SUPPORTS_GRAPH_CAPTURE)
                != notes.end();

The Python frontend has the same ``Graph.get_execution_plan_engine_id()`` and
``Graph.get_behavior_notes_for_engine(engine_id)`` methods.

An engine that does not report the note is not tested for capture. It can still work in some
cases. See :ref:`capture-miopen`.

Engines that support capture
----------------------------

These engines report ``SUPPORTS_GRAPH_CAPTURE``. They were verified on gfx950, unless noted:

- Kernel ingestor engines: ``hipkernel:Gfx950AttentionDense``, ``hipkernel:flydsl_rmsnorm``,
  ``hipkernel:flydsl_rmsnorm_bwd``, ``hipkernel:flydsl_sdpa_cdna``,
  ``hipkernel:flydsl_sdpa_cdna_dualwave``, and ``hipkernel:flydsl_sdpa_cdna_decode``.
- RDNA kernel ingestor engines, verified on gfx1201 and, except ``hipkernel:flydsl_sdpa_aiw``, on
  gfx1152 on Windows: ``hipkernel:flydsl_sdpa_rdna``, ``hipkernel:flydsl_sdpa_rdna_decode``,
  ``hipkernel:flydsl_sdpa_aiw``, ``hipkernel:Gfx1151WmmaAttention``, and
  ``hipkernel:Gfx1151WmmaAttentionDecode``.
- SDPA backward kernel ingestor engines: ``hipkernel:flydsl_sdpa_bwd_merged`` (gfx950) and
  ``hipkernel:flydsl_sdpa_bwd_split`` (gfx950 and gfx1101).
- ``HIP_MLOPS_ENGINE``.
- ``ASM_SDPA_ENGINE``. The forward pass was verified on gfx950 and the backward pass on gfx942.
- ``HIPBLASLT_ENGINE``.
- ``HIP_FLASH2_ENGINE`` (optional, off by default), verified on gfx942.

The ingestor engines declare the note with the ``supports_graph_capture`` tag in the
``behavior_notes`` field of their engine descriptor.

Engines that are not in this list are not covered by this topic.

What does not work in a capture
===============================

Engine benchmarking
-------------------

When benchmarking is on (the ``global.benchmarking`` knob or ``HIPDNN_FORCE_BENCHMARKING``), the
plan times its candidates on the first ``execute()``. This uses stream and event
synchronization, which is not allowed in a capture. The capture is invalidated, and
``hipStreamEndCapture()`` returns ``hipErrorStreamCaptureInvalidated``.

To use benchmarking with capture, run the plan one time before the capture. The plan keeps the
candidate it chose, and later executions can be captured.

Autotune
--------

``Graph::autotune()`` synchronizes the device, so it fails inside a capture. Call ``autotune()``
before the capture. Then capture ``execute()`` of the plan that autotune chose.

Runtime pass-by-value scalars
-----------------------------

hipDNN reads a runtime pass-by-value scalar from host memory when the plan executes. A captured
graph keeps the value that the scalar had at capture time. If you change the host value later, the
replays do not change. This applies, for example, to a runtime attention scale or a runtime
epsilon. The SDPA backward engines (``hipkernel:flydsl_sdpa_bwd_merged`` and
``hipkernel:flydsl_sdpa_bwd_split``) apply this to a by-value dropout seed and offset: hipDNN reads
them at execute and freezes them into the captured graph, while a seed and offset passed as device
tensors are read by the kernel at replay, so updating the tensors changes the mask.

To avoid a frozen value, use a compile-time constant, or capture the graph again after you change
the value. See :ref:`pass-by-value-tensors`.

After a failed capture
----------------------

When a capture fails, ``hipStreamEndCapture()`` returns ``hipErrorStreamCaptureInvalidated`` and
ends the capture. On Linux, you can use the same stream and the same handle afterward. On
Windows, the stream can stay in an error state. If this happens, create a new stream and set it
on the handle.

.. _capture-miopen:

MIOpen provider
===============

The MIOpen provider engines (``MIOPEN_ENGINE`` and ``MIOPEN_ENGINE_DETERMINISTIC``) do not report
``SUPPORTS_GRAPH_CAPTURE``, and the capture tests skip them. Capture can work in some cases. Treat
these cases as a conditional workaround, not as support.

A convolution plan chooses its algorithm with MIOpen Find on the first ``execute()``. With the
default find mode (``DYNAMIC_HYBRID``) and an empty find database, Find benchmarks the solvers.
When this happens inside a capture, the capture fails. Capture works when one of these is true:

- You set ``MIOPEN_FIND_MODE=FAST`` before the process starts.
- The user find database already has the entry, for example from an earlier process.
- You ran the plan before the capture.

For some convolutions, MIOpen frees a temporary Find buffer later, on a following call. This was
seen on weight-gradient convolutions. One run before the capture is then not enough, because the
free can fall inside the capture. Run the plan more than once before the capture, and wait
briefly or synchronize the stream between the runs. Or use ``MIOPEN_FIND_MODE=FAST``.

The fused convolution, bias, and activation plans (the fusion API) and the batchnorm inference
plans were captured on the first execution with no extra setup.

Test an engine with the integration tests
=========================================

The provider integration tests have an opt-in ``--graph-capture`` flag. With it, each verified
test captures one more execution of the plan into a hipGraph, replays it, and compares the result
with the reference. The flag needs a GPU. Each engine in the list above that has a provider test
lane also has a graph-capture lane, which runs a few quick test cases with the flag in the
standard, comprehensive and full test tiers. For the full description and the other options,
see ``dnn-providers/integration-tests/docs/running-tests.md`` in the repository.
