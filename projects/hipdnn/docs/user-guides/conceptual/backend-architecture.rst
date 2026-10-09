.. meta::
  :description: A granular breakdown of the hipDNN system architecture and the backend API.
  :keywords: hipDNN, ROCm, framework, C, frontend, backend

.. _backend-architecture:

***************************
hipDNN backend architecture
***************************

The hipDNN framework consists of a frontend (C++ Graph API), a backend (core runtime), and a plugin system. The backend prepares and dispatches execution to dynamically loaded plugins via a C-API interface.

.. important::

  This topic is for advanced users who want a more granular breakdown of the system architecture and the backend API. See :ref:`architecture` for a high-level overview of the system architecture.

The C-API plugin separates the plugin implementation and engine details from the hipDNN backend.

SDKs are provided to assist plugin developers, but the implementation details are otherwise at the discretion of the developer.
The goal of the Plugin SDK is to standardize and provide these as reusable components for plugin development, so developers can focus on implementing the underlying kernels and libraries.

Functions and call flow
=======================

hipDNN uses a three-layer architecture:

.. code::

   User Application
         |
      1. Frontend (C++ API) -- hipdnn_frontend namespace
         |
      2. Backend (C API) -- hipdnn_backend.h
         |
      3. Plugin SDK (C API) -- PluginApi.h + EnginePluginApi.h
         |
         Engine Plugin (.so) -- e.g., MIOpen plugin

The frontend provides a high-level graph-based C++ API. It translates user operations into backend descriptor calls. The backend manages descriptors, handles, and the plugin system. Plugins implement the actual GPU computation.

Frontend API control flow
=========================

``setEnginePluginPaths`` (frontend -> backend)
----------------------------------------------

.. code::

   Frontend: setEnginePluginPaths(paths, mode)
      -> Backend: hipdnnSetEnginePluginPaths_ext(count, cPaths, backendMode)
         -> EnginePluginResourceManager::setPluginPaths() [stores paths in static config]
         [No plugin calls -- paths stored for later use during hipdnnCreate]


``createHipdnnHandle`` (frontend -> backend -> plugin)
------------------------------------------------------

.. code::

   Frontend: createHipdnnHandle(handle, stream)
      -> Backend: hipdnnCreate(handlePtr)
         -> HandleFactory::createHandle()
            -> EnginePluginResourceManager::create()
               -> EnginePluginManager::loadPlugins(paths)
                  -> For each plugin .so:
                     1. SharedLibrary::load() -- dlopen
                     2. Resolve symbols: PluginGetName, PluginGetVersion, PluginGetApiVersion, PluginGetType
                     3. Plugin: hipdnnPluginGetName()
                     4. Plugin: hipdnnPluginGetVersion()
                     5. Plugin: hipdnnPluginGetApiVersion()
                     6. Plugin: hipdnnPluginGetType()
                     7. Plugin: hipdnnPluginSetLoggingCallback()
               -> For each EnginePlugin:
                  1. Plugin: hipdnnEnginePluginCreate(handle*)
                  2. Plugin: hipdnnEnginePluginGetAllEngineIds()
                  3.  Map engine IDs to plugin handles
      -> Backend: hipdnnSetStream(handle, stream) [if stream != nullptr]
         -> For each plugin handle:
            1.  Plugin: hipdnnEnginePluginSetStream(handle, stream)

``setHipdnnHandleStream`` (frontend -> backend -> plugin)
---------------------------------------------------------

.. code::

   Frontend: setHipdnnHandleStream(handle, stream)
      -> Backend: hipdnnSetStream(handle, stream)
         -> EnginePluginResourceManager::setStream(stream)
            -> For each (pluginHandle, plugin):
               Plugin: hipdnnEnginePluginSetStream(pluginHandle, stream)


``graph.build()`` (full compilation pipeline)
---------------------------------------------

.. code::

   Frontend: graph.build(handle, modes, policy)
     1. graph.validate()              [Frontend-only: DAG validation, topological sort]
     2. graph.build_operation_graph(handle)
           a. For each node: node->create_operation() [creates backend operation descriptors]
           b. Backend: hipdnnBackendCreateDescriptor(GRAPH)
           c. Backend: hipdnnBackendSetAttribute(graphDesc, HANDLE, ...)
           d. Backend: hipdnnBackendSetAttribute(graphDesc, OPERATIONS, ...)
           e. Backend: hipdnnBackendSetAttribute(graphDesc, COMPUTE_DATA_TYPE, ...)
           f. Backend: hipdnnBackendSetAttribute(graphDesc, INTERMEDIATE_DATA_TYPE, ...)
           g. Backend: hipdnnBackendSetAttribute(graphDesc, IO_DATA_TYPE, ...)
           h. Backend: hipdnnBackendSetAttribute(graphDesc, PREFERRED_ENGINE_ID, ...) [if set]
           i. Backend: hipdnnBackendFinalize(graphDesc)
     3. graph.create_execution_plans(modes)
        a. Backend: hipdnnBackendCreateDescriptor(HEURISTIC)
        b. Backend: hipdnnBackendSetAttribute(heuristic, GRAPH, graphDesc)
        c. Backend: hipdnnBackendSetAttribute(heuristic, HEURISTIC_MODE, modes)
        d. Backend: hipdnnBackendFinalize(heuristic)
           -> EngineHeuristicDescriptor::finalize()
              -> Plugin: hipdnnEnginePluginGetApplicableEngineIds()  [one pass across plugins]
              -> Heuristic policies rank the applicable engine IDs  [no GetEngineDetails]
        e. initializeEngineConfig(heuristic)
           -> detail::getEngineConfigs(configs, ids, heuristic, getAll=true)
              -> Backend: hipdnnBackendGetAttribute(heuristic, ENGINEHEUR_RESULTS, ...)
                 -> EngineHeuristicDescriptor::getEngineConfigs()
                    -> For each ranked engine ID:
                       EngineDescriptor::createForApplicableEngine(graph, engineId)
                          -> Plugin: hipdnnEnginePluginGetEngineDetails()  [no applicability re-probe]
              -> For each engine config:
                 Backend: hipdnnBackendFinalize(engineConfig)
                    -> EngineConfigDescriptor::finalize()
                       -> Plugin: hipdnnEnginePluginGetWorkspaceSize()
           -> Select the preferred engine if it is ranked, otherwise the top-ranked engine
        f. Backend: hipdnnBackendCreateDescriptor(EXECUTION_PLAN)  [per engine config, unfinalized]
     4. graph.check_support()         [validates descriptors exist]
     5. graph.build_plans()           [active plan, or every plan with BuildPlanPolicy::ALL]
        -> finalizePlanDescriptor(plan)
           a. Backend: hipdnnBackendSetAttribute(execPlan, ENGINE_CONFIG, engineConfig)
           b. Backend: hipdnnBackendFinalize(execPlan)
              -> ExecutionPlanDescriptor::finalize()
                 -> Plugin: hipdnnEnginePluginCreateExecutionContext()
                 -> Plugin: hipdnnEnginePluginGetWorkspaceSize()
           c. Backend: hipdnnBackendGetAttribute(execPlan, WORKSPACE_SIZE, ...)


``graph.get_workspace_size()``
------------------------------

.. code::

   Frontend: graph.get_workspace_size(workspaceSize)
      -> Backend: hipdnnBackendGetAttribute(execPlanDesc, WORKSPACE_SIZE, ...)
         [No plugin calls -- workspace size was cached during build_plans]


``graph.execute()``
-------------------

.. code::

   Frontend: graph.execute(handle, variantPack, workspace)
     1. Backend: hipdnnBackendCreateDescriptor(VARIANT_PACK)
     2. Backend: hipdnnBackendSetAttribute(variantPack, DATA_POINTERS, ...)
     3. Backend: hipdnnBackendSetAttribute(variantPack, UNIQUE_IDS, ...)
     4. Backend: hipdnnBackendSetAttribute(variantPack, WORKSPACE, ...)
     5. Backend: hipdnnBackendFinalize(variantPackDesc)
     6. Backend: hipdnnBackendExecute(handle, execPlanDesc, variantPackDesc)
        -> EnginePluginResourceManager::executeOpGraph()
           -> Extract engine ID from execution plan chain
           -> Look up plugin handle for engine ID
           -> Plugin: hipdnnEnginePluginExecuteOpGraph(handle, ctx, workspace, buffers, count)
              [Actual GPU kernel execution happens here]


Handle destruction (implicit via RAII)
--------------------------------------

.. code::

   Frontend: ~HipdnnHandlePtr()
      -> Backend: hipdnnDestroy(handle)
         -> ~EnginePluginResourceManager()
            -> For each execution context:
               Plugin: hipdnnEnginePluginDestroyExecutionContext(handle, ctx)
            -> For each engine details:
               Plugin: hipdnnEnginePluginDestroyEngineDetails(handle, details*)
            -> For each plugin handle:
               Plugin: hipdnnEnginePluginDestroy(handle)

``get_knobs_for_engine()``
--------------------------

.. code::

   Frontend: graph.get_knobs_for_engine(engineId, knobs)
      -> detail::createEngineDescriptorForGraph()
         -> Backend: hipdnnBackendCreateDescriptor(ENGINE)
         -> Backend: hipdnnBackendSetAttribute(engine, GRAPH, graphDesc)
         -> Backend: hipdnnBackendSetAttribute(engine, ENGINE_ID, engineId)
         -> Backend: hipdnnBackendFinalize(engine)
            -> EngineDescriptor::finalize()
               -> Plugin: hipdnnEnginePluginGetApplicableEngineIds()  [one pass across plugins]
               -> Plugin: hipdnnEnginePluginGetEngineDetails(handle, engineId, graph, details*)
      -> detail::unpackKnobsFromDescriptors(engineDesc, knobs)
         -> Backend: hipdnnBackendGetAttribute(engine, KNOB_INFO, ...)

``get_ranked_engine_ids()``
---------------------------

.. code::

   Frontend: graph.get_ranked_engine_ids(ids, modes)
      -> detail::createEngineHeuristicDescriptorForGraph()
         -> Backend: hipdnnBackendCreateDescriptor(HEURISTIC)
         -> Backend: hipdnnBackendSetAttribute(heuristic, GRAPH, graphDesc)
         -> Backend: hipdnnBackendSetAttribute(heuristic, HEURISTIC_MODE, modes)
         -> Backend: hipdnnBackendFinalize(heuristic)
            -> EngineHeuristicDescriptor::finalize()
               -> Plugin: hipdnnEnginePluginGetApplicableEngineIds()  [one pass across plugins]
      -> detail::getEngineConfigs(configs, ids, heuristicDesc, getAll=true)
         -> Backend: hipdnnBackendGetAttribute(heuristic, ENGINEHEUR_RESULTS, ...)
            -> EngineDescriptor::createForApplicableEngine()  [per ranked engine]
               -> Plugin: hipdnnEnginePluginGetEngineDetails()
         -> Backend: hipdnnBackendFinalize(engineConfig)  [per ranked engine]
            -> Plugin: hipdnnEnginePluginGetWorkspaceSize()


``create_execution_plan_ext()``
-------------------------------

.. code::

   Frontend: graph.create_execution_plan_ext(engineId, knobSettings)
     -> compilePlanFromSpec(engineId, knobSettings, plan)
        1. detail::createEngineDescriptorForGraph(engineDesc, graphDesc, engineId)
           -> Backend: hipdnnBackendCreateDescriptor(ENGINE)
           -> Backend: hipdnnBackendSetAttribute(engine, GRAPH, graphDesc)
           -> Backend: hipdnnBackendSetAttribute(engine, ENGINE_ID, engineId)
           -> Backend: hipdnnBackendFinalize(engine)
              -> EngineDescriptor::finalize()
                 -> Plugin: hipdnnEnginePluginGetApplicableEngineIds()  [one pass across plugins]
                 -> Plugin: hipdnnEnginePluginGetEngineDetails()        [this engine only]
        2. Backend: hipdnnBackendCreateDescriptor(ENGINECFG)
           -> Backend: hipdnnBackendSetAttribute(engineConfig, ENGINE, engineDesc)
        3. knobLookupFromEngineDescriptor(engineDesc, knobs)  [reads this engine's knobs]
           -> detail::unpackKnobsFromDescriptors(engineDesc, knobs)
              -> Backend: hipdnnBackendGetAttribute(engine, KNOB_INFO, ...)
        4. validateAndFilterKnobSettings(knobSettings, knobs, validatedSettings)
        5. detail::applyKnobSettingsViaDescriptors(engineConfig, validatedSettings)
           -> Backend: hipdnnBackendSetAttribute(engineConfig, KNOB_CHOICES, ...)
        6. Backend: hipdnnBackendFinalize(engineConfig)
           -> EngineConfigDescriptor::finalize()
              -> Plugin: hipdnnEnginePluginGetWorkspaceSize()
        7. Backend: hipdnnBackendCreateDescriptor(EXECUTION_PLAN)  [unfinalized]

   Frontend: graph.build_plans()
     -> finalizePlanDescriptor(plan)
        -> Backend: hipdnnBackendSetAttribute(execPlan, ENGINE_CONFIG, ...)
        -> Backend: hipdnnBackendFinalize(execPlan)
           -> ExecutionPlanDescriptor::finalize()
              -> Plugin: hipdnnEnginePluginCreateExecutionContext()
              -> Plugin: hipdnnEnginePluginGetWorkspaceSize()
        -> Backend: hipdnnBackendGetAttribute(execPlan, WORKSPACE_SIZE, ...)


Plugin lifecycle
=================

1. **Discovery**: ``Plugin.so`` loaded, metadata queried (name, version, type)
2. **Instance creation**: ``PluginCreate`` and ``GetAllEngineIds`` during handle creation
3. **Engine selection**: ``GetApplicableEngineIds`` and ``GetEngineDetails`` during graph build
4. **Context creation**: ``CreateExecutionContext`` and ``GetWorkspaceSize`` during plan finalization
5. **Execution**: ``ExecuteOpGraph``
6. **Cleanup**: ``DestroyExecutionContext``, ``DestroyEngineDetails``, and ``PluginDestroy``

By default, hipDNN keeps plugins loaded after all hipDNN handles are closed.
This reduces the overhead of reloading plugins when subsequent hipDNN handles are created.
The ``hipdnnSetPluginUnloadMode_ext()`` backend API function can be used to change this behaviour so that plugins unload immediately once all hipDNN handles are closed.

Backend descriptor types
========================

The backend uses descriptors as opaque handles to manage different aspects of graph execution:

Operation graph descriptor (``HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR``)
-------------------------------------------------------------------------

- Represents the computational graph to be executed.
- Contains nodes, tensors, and their connections.

Engine heuristic descriptor (``HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR``)
----------------------------------------------------------------------

- Manages the selection of appropriate engines for a graph.
- Queries plugins for applicable engines.
- Extensible plugin design to control engine selection.

Engine config descriptor (``HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR``)
------------------------------------------------------------------

- Represents a specific engine configuration.
- Contains engine ID and configuration parameters.
- Retrieved from heuristic results.

Engine descriptor (``HIPDNN_BACKEND_ENGINE_DESCRIPTOR``)
--------------------------------------------------------

- Represents a backend engine.
- Contains engine ID, and a set of behavioral notes and configurable settings.
- Retrieved from engine config descriptor.
- Exposes the engine name through the read-only ``HIPDNN_ATTR_ENGINE_NAME_EXT`` attribute (``HIPDNN_TYPE_CHAR``).

Execution plan descriptor (``HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR``)
------------------------------------------------------------------------

- Combines an engine configuration with a graph.
- Manages workspace requirements.
- Prepares for actual execution.

Variant pack descriptor (``HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR``)
--------------------------------------------------------------------

- Contains runtime data for execution.
- Maps tensor UIDs to device memory pointers.
- Includes workspace device memory pointer.
