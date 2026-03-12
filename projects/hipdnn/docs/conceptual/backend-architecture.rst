.. meta::
  :description: hipDNN high-level architecture
  :keywords: hipDNN, ROCm, API,

.. _backend-architecture:

***************************
hipDNN backend architecture
***************************

.. important::

  This page is for advanced users who want a more granular breakdown of the system architecture and the backend API. See :ref:`architecture` for a high-level overview of the system architecture.

The hipDNN framework consists of a frontend (C++ Graph API), a backend (core runtime), and a plugin system. The backend prepares and dispatches execution to dynamically loaded plugins via a C-API interface.

The plugin C-API separates the plugin implementation from the hipDNN backend and effectively separates the plugin engine implementation details from the hipDNN backend.

SDKs are provided to assist plugin developers, but the implementation details are otherwise at the discretion of the developer. The goal of the Plugin SDK is to standardize and provide these as reusable components for plugin development, so developers can focus on the implementations of the underlying kernels and libraries.


## Functions and call flow

hipDNN uses a three-layer architecture:

```
User Application
    |
1. Frontend (C++ API) -- hipdnn_frontend namespace
    |
2. Backend (C API) -- hipdnn_backend.h
    |
3. Plugin SDK (C API) -- PluginApi.h + EnginePluginApi.h
    |
   Engine Plugin (.so) -- e.g., MIOpen plugin
```

The frontend provides a high-level graph-based C++ API. It translates user operations into backend descriptor calls. The backend manages descriptors, handles, and the plugin system. Plugins implement the actual GPU computation.

---

## 1. Frontend API Functions

### 1.1 Plugin Configuration

| Function | Signature |
|----------|-----------|
| `setEnginePluginPaths()` | `Error setEnginePluginPaths(Container& paths, PluginLoadingMode mode)` |
| `getLoadedEnginePluginPaths()` | `Error getLoadedEnginePluginPaths(hipdnnHandle_t, vector<path>&)` |

### 1.2 Handle Management

| Function | Signature |
|----------|-----------|
| `createHipdnnHandle()` | `Error createHipdnnHandle(HipdnnHandlePtr&, hipStream_t)` |
| `createHipdnnHandle()` | `pair<HipdnnHandlePtr, Error> createHipdnnHandle(hipStream_t)` |
| `setHipdnnHandleStream()` | `Error setHipdnnHandleStream(HipdnnHandlePtr&, hipStream_t)` |
| `getHipdnnHandleStream()` | `Error getHipdnnHandleStream(HipdnnHandlePtr&, hipStream_t*)` |
| `~HipdnnHandleDeleter` | Implicit via RAII unique_ptr destruction |

### 1.3 Graph Configuration

| Function | Returns |
|----------|---------|
| `Graph::set_io_data_type(DataType)` | `Graph&` |
| `Graph::set_compute_data_type(DataType)` | `Graph&` |
| `Graph::set_intermediate_data_type(DataType)` | `Graph&` |
| `Graph::set_name(string)` | `Graph&` |
| `Graph::set_preferred_engine_id_ext(int64_t)` | `Graph&` |

### 1.4 Tensor Creation

| Function | Returns |
|----------|---------|
| `Graph::tensor(TensorAttributes)` | `shared_ptr<TensorAttributes>` |
| `Graph::tensor_like(shared_ptr<TensorAttributes>)` | `shared_ptr<TensorAttributes>` |

### 1.5 Operation Methods (adds nodes to graph DAG)

All operations create a node in the graph's internal DAG. Backend calls are deferred until `build()`.

| Function | Node Type | Returns |
|----------|-----------|---------|
| `graph.conv_fprop(x, w, attrs)` | ConvolutionFpropNode | `y` |
| `graph.conv_dgrad(dy, w, attrs)` | ConvolutionDgradNode | `dx` |
| `graph.conv_wgrad(dy, x, attrs)` | ConvolutionWgradNode | `dw` |
| `graph.batchnorm(x, scale, bias, attrs)` | BatchnormNode | `[y, mean, invVar, runMean, runVar]` |
| `graph.batchnorm_backward(dy, x, scale, attrs)` | BatchnormBackwardNode | `[dx, dscale, dbias]` |
| `graph.batchnorm_inference(x, mean, invVar, scale, bias, attrs)` | BatchnormInferenceNode | `y` |
| `graph.batchnorm_inference_variance_ext(...)` | BatchnormInferenceNodeVarianceExt | `y` |
| `graph.layernorm(x, scale, bias, attrs)` | LayerNormNode | `[y, mean, invVar]` |
| `graph.rmsnorm(x, scale, attrs)` | RMSNormNode | `[y, invRms]` |
| `graph.block_scale_dequantize(x, scale, attrs)` | BlockScaleDequantizeNode | `y` |
| `graph.block_scale_quantize(x, attrs)` | BlockScaleQuantizeNode | `[y, scale]` |
| `graph.pointwise(in0, attrs)` | PointwiseNode (unary) | `out0` |
| `graph.pointwise(in0, in1, attrs)` | PointwiseNode (binary) | `out0` |
| `graph.pointwise(in0, in1, in2, attrs)` | PointwiseNode (ternary) | `out0` |
| `graph.matmul(a, b, attrs)` | MatmulNode | `c` |
| `graph.sdpa(q, k, v, attrs)` | SdpaFpropNode | `[o, stats]` |

### 1.6 Graph Compilation

| Function | Purpose |
|----------|---------|
| `graph.validate()` | Frontend-only DAG validation (cycles, UIDs) |
| `graph.build(handle, modes, policy)` | Full compilation pipeline |
| `graph.build_operation_graph(handle)` | Translate graph to backend descriptors |
| `graph.create_execution_plans(modes)` | Engine heuristic selection |
| `graph.create_execution_plan_ext(engineId, settings)` | Explicit engine + knob selection |
| `graph.check_support()` | Verify plan creation succeeded |
| `graph.build_plans()` | Finalize execution plan |
| `graph.get_workspace_size(int64_t&)` | Query workspace requirement |

### 1.7 Graph Execution

| Function | Purpose |
|----------|---------|
| `graph.execute(handle, tensorLookup, workspace)` | Execute with TensorAttributes map |
| `graph.execute(handle, variantPack, workspace)` | Execute with UID-to-pointer map |

### 1.8 Engine/Knob Query

| Function | Purpose |
|----------|---------|
| `graph.get_knobs_for_engine(engineId, knobs)` | Query available knobs |
| `graph.get_knob_lookup_for_engine(engineId, knobs)` | Query knobs as map |
| `graph.get_ranked_engine_ids(ids, modes)` | Get engines ranked by heuristic |

### 1.9 Serialization

| Function | Format |
|----------|--------|
| `graph.toBinary() / serialize(vector<uint8_t>&)` | Raw binary |
| `graph.toJson() / serialize(json&)` | JSON |

---

## 2. Backend API Functions

### 2.1 Handle Management

| Function | Plugin Calls |
|----------|-------------|
| `hipdnnCreate(handle*)` | `PluginCreate`, `PluginGetAllEngineIds`, `PluginSetStream` (if stream) |
| `hipdnnDestroy(handle)` | `PluginDestroy` (each handle) |
| `hipdnnSetStream(handle, stream)` | `PluginSetStream` (each handle) |
| `hipdnnGetStream(handle, stream*)` | None |

### 2.2 Descriptor Management

| Function | Plugin Calls |
|----------|-------------|
| `hipdnnBackendCreateDescriptor(type, desc*)` | None |
| `hipdnnBackendDestroyDescriptor(desc)` | None |
| `hipdnnBackendSetAttribute(desc, name, type, count, data)` | None |
| `hipdnnBackendGetAttribute(desc, name, type, reqCount, count*, data)` | None |
| `hipdnnBackendFinalize(desc)` | **Varies by descriptor type** (see below) |

**hipdnnBackendFinalize plugin calls by descriptor type:**

| Descriptor Type | Plugin Calls |
|----------------|-------------|
| Graph | None |
| EngineHeuristic | `PluginGetApplicableEngineIds`, `PluginGetEngineDetails` |
| Engine | `PluginGetEngineDetails` (if not cached) |
| EngineConfig | None |
| ExecutionPlan | `PluginGetEngineDetails`, `PluginCreateExecutionContext`, `PluginGetWorkspaceSize` |
| Tensor, Operations, VariantPack, KnobSetting | None |

### 2.3 Execution

| Function | Plugin Calls |
|----------|-------------|
| `hipdnnBackendExecute(handle, plan, variantPack)` | `PluginExecuteOpGraph` |

### 2.4 Graph Serialization (Backend)

| Function | Plugin Calls |
|----------|-------------|
| `hipdnnBackendCreateAndDeserializeGraph_ext(desc*, data, size)` | None |
| `hipdnnBackendGetSerializedBinaryGraph_ext(desc, reqSize, size*, data)` | None |

### 2.5 Plugin Configuration

| Function | Plugin Calls |
|----------|-------------|
| `hipdnnSetEnginePluginPaths_ext(count, paths, mode)` | None (config only) |
| `hipdnnSetPluginUnloadMode_ext(mode)` | None (config only) |
| `hipdnnGetLoadedEnginePluginPaths_ext(handle, count*, paths, maxLen*)` | None |
| `hipdnnGetEngineCount_ext(handle, count*)` | None (uses cached data) |
| `hipdnnGetEngineInfo_ext(handle, idx, ...)` | None (uses cached data) |

### 2.6 Logging & Error

| Function | Plugin Calls |
|----------|-------------|
| `hipdnnGetErrorString(status)` | None |
| `hipdnnGetLastErrorString(msg, maxSize)` | None |
| `hipdnnPeekLastErrorString_ext(msg, maxSize)` | None |
| `hipdnnLoggingCallback_ext(severity, msg)` | None |
| `hipdnnSetUserLogCallback_ext(callback, level, mode, handle)` | None |
| `hipdnnBackendSetGlobalLogLevel_ext(level)` | None |
| `hipdnnBackendGetGlobalLogLevel_ext(level*)` | None |

### 2.7 Version

| Function | Plugin Calls |
|----------|-------------|
| `hipdnnGetVersion_ext(version**)` | None |
| `hipdnnVersionString_ext()` | None |

---

## 3. Plugin API Functions

### 3.1 Common Plugin API (PluginApi.h)

| Function | Phase | Purpose |
|----------|-------|---------|
| `hipdnnPluginGetName(name**)` | Discovery | Return plugin name |
| `hipdnnPluginGetVersion(version**)` | Discovery | Return plugin version |
| `hipdnnPluginGetApiVersion(version**)` | Discovery | Return API version supported |
| `hipdnnPluginGetType(type*)` | Discovery | Return plugin type (ENGINE) |
| `hipdnnPluginGetLastErrorString(str**)` | Error | Return thread-local error |
| `hipdnnPluginSetLoggingCallback(callback)` | Setup | Inject backend logging callback |

### 3.2 Engine Plugin API (EnginePluginApi.h)

| Function | Phase | Purpose |
|----------|-------|---------|
| `hipdnnEnginePluginGetAllEngineIds(ids*, max, count*)` | Discovery | List all engines in plugin |
| `hipdnnEnginePluginCreate(handle*)` | Setup | Create plugin instance |
| `hipdnnEnginePluginDestroy(handle)` | Cleanup | Destroy plugin instance |
| `hipdnnEnginePluginSetStream(handle, stream)` | Setup | Bind HIP stream |
| `hipdnnEnginePluginGetApplicableEngineIds(handle, graph, ids*, max, count*)` | Compilation | Find compatible engines |
| `hipdnnEnginePluginGetEngineDetails(handle, id, graph, details*)` | Compilation | Get engine capabilities/knobs |
| `hipdnnEnginePluginDestroyEngineDetails(handle, details*)` | Cleanup | Free engine details |
| `hipdnnEnginePluginGetWorkspaceSize(handle, config, graph, size*)` | Compilation | Get workspace requirement |
| `hipdnnEnginePluginCreateExecutionContext(handle, config, graph, ctx*)` | Compilation | Build executable plan |
| `hipdnnEnginePluginDestroyExecutionContext(handle, ctx)` | Cleanup | Free execution context |
| `hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(handle, ctx, size*)` | Compilation | Get workspace from context |
| `hipdnnEnginePluginExecuteOpGraph(handle, ctx, workspace, buffers, count)` | Execution | Run GPU computation |

---

## 4. Detailed Control Flow Traces

### 4.1 setEnginePluginPaths (Frontend -> Backend)

```
Frontend: setEnginePluginPaths(paths, mode)
  -> Backend: hipdnnSetEnginePluginPaths_ext(count, cPaths, backendMode)
     -> EnginePluginResourceManager::setPluginPaths() [stores paths in static config]
     [No plugin calls -- paths stored for later use during hipdnnCreate]
```

### 4.2 createHipdnnHandle (Frontend -> Backend -> Plugin)

```
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
              8. Plugin: hipdnnEnginePluginCreate(handle*)
              9. Plugin: hipdnnEnginePluginGetAllEngineIds()
              10. Map engine IDs to plugin handles
  -> Backend: hipdnnSetStream(handle, stream) [if stream != nullptr]
     -> For each plugin handle:
        11. Plugin: hipdnnEnginePluginSetStream(handle, stream)
```

### 4.3 setHipdnnHandleStream (Frontend -> Backend -> Plugin)

```
Frontend: setHipdnnHandleStream(handle, stream)
  -> Backend: hipdnnSetStream(handle, stream)
     -> EnginePluginResourceManager::setStream(stream)
        -> For each (pluginHandle, plugin):
           Plugin: hipdnnEnginePluginSetStream(pluginHandle, stream)
```

### 4.4 graph.build() (Full Compilation Pipeline)

```
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
           -> Plugin: hipdnnEnginePluginGetApplicableEngineIds()  [per plugin]
           -> Plugin: hipdnnEnginePluginGetEngineDetails()        [per applicable engine]
     e. initializeEngineConfig() [selects best engine config from heuristic results]
  4. graph.check_support()         [validates descriptors exist]
  5. graph.build_plans()
     a. Backend: hipdnnBackendSetAttribute(execPlan, ENGINE_CONFIG, engineConfig)
     b. Backend: hipdnnBackendFinalize(execPlan)
        -> ExecutionPlanDescriptor::finalize()
           -> Plugin: hipdnnEnginePluginGetEngineDetails()          [if not cached]
           -> Plugin: hipdnnEnginePluginCreateExecutionContext()
           -> Plugin: hipdnnEnginePluginGetWorkspaceSize()
```

### 4.5 graph.get_workspace_size()

```
Frontend: graph.get_workspace_size(workspaceSize)
  -> Backend: hipdnnBackendGetAttribute(execPlanDesc, WORKSPACE_SIZE, ...)
     [No plugin calls -- workspace size was cached during build_plans]
```

### 4.6 graph.execute()

```
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
```

### 4.7 Handle Destruction (Implicit via RAII)

```
Frontend: ~HipdnnHandlePtr()
  -> Backend: hipdnnDestroy(handle)
     -> ~EnginePluginResourceManager()
        -> For each execution context:
           Plugin: hipdnnEnginePluginDestroyExecutionContext(handle, ctx)
        -> For each engine details:
           Plugin: hipdnnEnginePluginDestroyEngineDetails(handle, details*)
        -> For each plugin handle:
           Plugin: hipdnnEnginePluginDestroy(handle)
```

### 4.8 get_knobs_for_engine()

```
Frontend: graph.get_knobs_for_engine(engineId, knobs)
  -> detail::createEngineDescriptorForGraph()
     -> Backend: hipdnnBackendCreateDescriptor(ENGINE)
     -> Backend: hipdnnBackendSetAttribute(engine, GRAPH, graphDesc)
     -> Backend: hipdnnBackendSetAttribute(engine, ENGINE_ID, engineId)
     -> Backend: hipdnnBackendFinalize(engine)
        -> Plugin: hipdnnEnginePluginGetEngineDetails(handle, engineId, graph, details*)
  -> detail::getKnobsForEngine(knobs, engineDesc)
     -> Backend: hipdnnBackendGetAttribute(engine, KNOB_INFO, ...)
```

### 4.9 get_ranked_engine_ids()

```
Frontend: graph.get_ranked_engine_ids(ids, modes)
  -> detail::createEngineHeuristicDescriptorForGraph()
     -> Backend: hipdnnBackendCreateDescriptor(HEURISTIC)
     -> Backend: hipdnnBackendSetAttribute(heuristic, GRAPH, graphDesc)
     -> Backend: hipdnnBackendSetAttribute(heuristic, HEURISTIC_MODE, modes)
     -> Backend: hipdnnBackendFinalize(heuristic)
        -> Plugin: hipdnnEnginePluginGetApplicableEngineIds()
        -> Plugin: hipdnnEnginePluginGetEngineDetails() [per engine]
  -> detail::getEngineConfigs(configs, ids, heuristicDesc)
```

### 4.10 create_execution_plan_ext()

```
Frontend: graph.create_execution_plan_ext(engineId, knobSettings)
  1. get_knob_lookup_for_engine(engineId) [validate knob settings]
  2. initializeEngineConfig(engineId)
     -> detail::createEngineDescriptorForGraph(engineId)
        -> Backend: hipdnnBackendCreateDescriptor(ENGINE)
        -> Backend: hipdnnBackendSetAttribute(engine, GRAPH, graphDesc)
        -> Backend: hipdnnBackendSetAttribute(engine, ENGINE_ID, engineId)
        -> Backend: hipdnnBackendFinalize(engine)
           -> Plugin: hipdnnEnginePluginGetEngineDetails()
  3. applyKnobSettingsToEngineConfig(settings)
     -> Backend: hipdnnBackendSetAttribute(engineConfig, KNOB_CHOICE, ...)
  4. Backend: hipdnnBackendFinalize(engineConfig)
  5. Create executionPlanDesc
     -> Backend: hipdnnBackendCreateDescriptor(EXECUTION_PLAN)
     -> Backend: hipdnnBackendSetAttribute(execPlan, ENGINE_CONFIG, ...)
     -> Backend: hipdnnBackendFinalize(execPlan)
        -> Plugin: hipdnnEnginePluginCreateExecutionContext()
        -> Plugin: hipdnnEnginePluginGetWorkspaceSize()
```

---

## 5. Plugin Lifecycle

1. **Discovery**: Plugin .so loaded, metadata queried (name, version, type)
2. **Instance creation**: `PluginCreate` + `GetAllEngineIds` during handle creation
3. **Engine selection**: `GetApplicableEngineIds` + `GetEngineDetails` during graph build
4. **Context creation**: `CreateExecutionContext` + `GetWorkspaceSize` during plan finalization
5. **Execution**: `ExecuteOpGraph`
6. **Cleanup**: `DestroyExecutionContext` + `DestroyEngineDetails` + `PluginDestroy`


Backend descriptor types
========================

The backend uses descriptors as opaque handles to manage different aspects of graph execution:

Operation Graph Descriptor (``HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR``)
-------------------------------------------------------------------------

- Represents the computational graph to be executed.
- Contains nodes, tensors, and their connections.

Engine Heuristic Descriptor (``HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR``)
----------------------------------------------------------------------

- Manages the selection of appropriate engines for a graph.
- Queries plugins for applicable engines.
- Extensible plugin design to control engine selection.

Engine Config Descriptor (``HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR``)
------------------------------------------------------------------

- Represents a specific engine configuration.
- Contains engine ID and configuration parameters.
- Retrieved from heuristic results.

Engine Descriptor (``HIPDNN_BACKEND_ENGINE_DESCRIPTOR``)
--------------------------------------------------------

- Represents a backend engine.
- Contains engine ID, and a set of behavioral notes + configurable settings.
- Retrieved from engine config Descriptor.

Execution Plan Descriptor (``HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR``)
------------------------------------------------------------------------

- Combines an engine configuration with a graph.
- Manages workspace requirements.
- Prepares for actual execution.

Variant Pack Descriptor (``HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR``)
--------------------------------------------------------------------

- Contains runtime data for execution.
- Maps tensor UIDs to device memory pointers.
- Includes workspace device memory pointer.
