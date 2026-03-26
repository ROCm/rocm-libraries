// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file is the main entry point for fusilli-plugin, implementations for all
// required hipDNN engine plugin API functions live here.
//
//===----------------------------------------------------------------------===//

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/vector.h>
#include <fusilli.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_plugin_sdk/EnginePluginApi.h>
#include <hipdnn_plugin_sdk/PluginApi.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginHelpers.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <iree/hal/buffer.h>
#include <iree/hal/buffer_view.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "graph_import.h"
#include "hipdnn_engine_plugin_execution_context.h"
#include "hipdnn_engine_plugin_handle.h"
#include "utils.h"

using namespace hipdnn_plugin_sdk;
using namespace fusilli_plugin;

// TODO(#2317): ensure single source of truth for plugin version
static const char *fusilliPluginVersion = "0.0.1";

// s_lastError is thread_local static so can't be initialized in the header file
// as the header file is included in many context. Clear the string here.
thread_local char
    PluginLastErrorManager::s_lastError[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] =
        "";

extern "C" {

// ----------------------------------------------------------------------
// Implementations for the basic plugin API defined in
// hipDNN/sdk/include/hipdnn_sdk/plugin/PluginApi.h
// ----------------------------------------------------------------------

hipdnnPluginStatus_t hipdnnPluginGetName(const char **name) {
  LOG_API_ENTRY("name_ptr=" << static_cast<void *>(name));
  FUSILLI_PLUGIN_CHECK_NULL(name);

  *name = hipdnn_data_sdk::utilities::FUSILLI_ENGINE_NAME;

  LOG_API_SUCCESS_AUTO("pluginName=" << *name);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char **version) {
  LOG_API_ENTRY("version_ptr=" << static_cast<void *>(version));
  FUSILLI_PLUGIN_CHECK_NULL(version);

  *version = fusilliPluginVersion;

  LOG_API_SUCCESS_AUTO("version=" << *version);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t *type) {
  LOG_API_ENTRY("type_ptr=" << static_cast<void *>(type));
  FUSILLI_PLUGIN_CHECK_NULL(type);

  *type = HIPDNN_PLUGIN_TYPE_ENGINE;

  LOG_API_SUCCESS_AUTO("type=" << *type);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void hipdnnPluginGetLastErrorString(const char **error_str) {
  if (error_str) {
    *error_str = hipdnn_plugin_sdk::PluginLastErrorManager::getLastError();
  }
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback) {
  // No LOG_API_ENTRY as logging won't be wired up yet.
  FUSILLI_PLUGIN_CHECK_NULL(callback);

  hipdnn_plugin_sdk::logging::initializeCallbackLogging(
      hipdnn_data_sdk::utilities::FUSILLI_ENGINE_NAME, callback);

  LOG_API_SUCCESS_AUTO("logging callback initialized");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------
// Implementations for engine plugin API defined in
// hipDNN/sdk/include/hipdnn_sdk/plugin/EnginePluginApi.h
// ----------------------------------------------------------------------

hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(int64_t *engineIds,
                                                       uint32_t maxEngines,
                                                       uint32_t *numEngines) {
  LOG_API_ENTRY("engineIds=" << static_cast<void *>(engineIds)
                             << ", maxEngines=" << maxEngines << ", numEngines="
                             << static_cast<void *>(numEngines));
  FUSILLI_PLUGIN_CHECK_NULL(numEngines);
  if (maxEngines != 0) {
    FUSILLI_PLUGIN_CHECK_NULL(engineIds);
  }

  // Set `numEngines` regardless of how many engines are actually returned.
  // The backend queries this function twice:
  // - First call: engineIds=NULL, maxEngines=0 to get the count
  // - Second call: engineIds allocated based on numEngines from first pass
  *numEngines = 1;

  if (maxEngines >= 1) {
    engineIds[0] = hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID;
  }

  LOG_API_SUCCESS_AUTO("numEngines=" << *numEngines);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t *handle) {
  LOG_API_ENTRY("handle_ptr=" << static_cast<void *>(handle));
  FUSILLI_PLUGIN_CHECK_NULL(handle);

  // Get device id.
  int deviceId;
  FUSILLI_PLUGIN_CHECK_ERROR(hipGetDevice(&deviceId));

  // Create handle.
  *handle = new HipdnnEnginePluginHandle(deviceId);

  LOG_API_SUCCESS_AUTO("createdHandle=" << static_cast<void *>(*handle));
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle) {
  LOG_API_ENTRY("handle=" << static_cast<void *>(handle));
  FUSILLI_PLUGIN_CHECK_NULL(handle);

  delete handle;

  LOG_API_SUCCESS_AUTO("");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                            hipStream_t stream) {
  LOG_API_ENTRY("handle=" << static_cast<void *>(handle)
                          << ", stream_id=" << static_cast<void *>(stream));
  FUSILLI_PLUGIN_CHECK_NULL(handle);

  // Get device associated with stream.
  hipDevice_t deviceId;
  FUSILLI_PLUGIN_CHECK_ERROR(hipStreamGetDevice(stream, &deviceId));

  // This should never happen, check so that when it does we get a nice error
  // message.
  if (deviceId != handle->deviceId) {
    return hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "Stream is associated with different device. Device reported "
        "through `hipStreamGetDevice` does not match active "
        "device reported through `hipGetDevice`.");
  }

  // Set stream, it will be used to create fusilli::Handle later.
  handle->setStream(stream);

  LOG_API_SUCCESS_AUTO("");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginGetApplicableEngineIds(
    hipdnnEnginePluginHandle_t handle, const hipdnnPluginConstData_t *opGraph,
    int64_t *engineIds, uint32_t maxEngines, uint32_t *numEngines) {
  LOG_API_ENTRY("handle=" << static_cast<void *>(handle)
                          << ", opGraph=" << static_cast<const void *>(opGraph)
                          << ", engineIds=" << static_cast<void *>(engineIds)
                          << ", maxEngines=" << maxEngines << ", numEngines="
                          << static_cast<void *>(numEngines));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(opGraph);
  if (maxEngines != 0) {
    FUSILLI_PLUGIN_CHECK_NULL(engineIds);
  }
  FUSILLI_PLUGIN_CHECK_NULL(numEngines);

  *numEngines = 0;
  if (maxEngines < 1) {
    HIPDNN_PLUGIN_LOG_INFO(
        "Maximum number of engines reached ("
        << maxEngines
        << "), ignoring additional engines, numEngines count: " << *numEngines);
    LOG_API_SUCCESS_AUTO("numEngines=" << *numEngines);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
  }

  // Use the graph import translation layer to determine if this graph is
  // supported. If import succeeds, the graph is composed of ops fusilli can
  // handle.
  //
  // NOTE: If a translatable graph should not be claimed (e.g. numerical
  // issues), one can gate particular ops in the translation layer
  // (graph_import.h), or filter out very specific graph types here - gates
  // should check environment variables so it's easy to run the problematic
  // graphs during development.
  auto result = importGraph(opGraph);
  if (fusilli::isError(static_cast<fusilli::ErrorObject>(result))) {
    HIPDNN_PLUGIN_LOG_INFO(
        "Graph not supported: "
        << static_cast<fusilli::ErrorObject>(result).getMessage());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
  }

  // Graph passes all checks, the fusilli engine is applicable.
  engineIds[0] = hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID;
  *numEngines = 1;

  LOG_API_SUCCESS_AUTO("numEngines=" << *numEngines);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                   int64_t engineId,
                                   const hipdnnPluginConstData_t *opGraph,
                                   hipdnnPluginConstData_t *engineDetails) {
  // ----------------------------------------------------------------------
  // Plugin API call flow for engine configuration and execution.
  //
  // hipDNN                                       Plugin
  // ======================================================================
  // hipdnnEnginePluginGetEngineDetails        -> populates engineDetails
  //                                              (flatbuffer object) with
  //                                              behavioral notes + knob
  //                                              definitions that are available
  //                                              to the higher level API.
  //                                              Return populated engineDetails
  //                                           <- (hipdnnPluginConstData_t).
  //
  // Decides final configuration, populating   ~~
  // engineConfig flatbuffer
  // (hipdnnPluginConstData_t) based on info
  // provided in engineDetails.
  //
  // hipdnnEnginePluginCreateExecutionContext  -> Creates execution context
  //                                              (hipdnnEnginePluginExecutionContext_t)
  //                                           <- based on engineConfig.
  //
  // Uses returned execution context to        ~~
  // invoke kernels
  //
  // hipdnnEnginePluginDestroyEngineDetails    -> cleans up engine details.
  //
  // hipdnnEnginePluginDestroyExecutionContext -> cleans up execution context.
  // ----------------------------------------------------------------------

  LOG_API_ENTRY(
      "handle=" << static_cast<void *>(handle) << ", engineId=" << engineId
                << ", opGraph=" << static_cast<const void *>(opGraph)
                << ", engineDetails=" << static_cast<void *>(engineDetails));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(opGraph);
  FUSILLI_PLUGIN_CHECK_NULL(engineDetails);

  if (engineId != hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID) {
    return hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "unexpected engine id");
  }

  // Build engine details object, we're only storing the engine id for the time
  // being.
  flatbuffers::FlatBufferBuilder builder;
  auto engineDetailsObj =
      hipdnn_data_sdk::data_objects::CreateEngineDetails(builder, engineId);
  builder.Finish(engineDetailsObj);

  // Populate out parameter.
  auto detachedBuffer =
      std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
  engineDetails->ptr = detachedBuffer->data();
  engineDetails->size = detachedBuffer->size();

  // Store owning pointer in handle, hipdnnEnginePluginDestroyEngineDetails will
  // inform us when it's safe to clean this up.
  handle->storeEngineDetailsBuffer(engineDetails->ptr,
                                   std::move(detachedBuffer));

  LOG_API_SUCCESS_AUTO("engineDetails->ptr=" << engineDetails->ptr);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       hipdnnPluginConstData_t *engineDetails) {
  // See comment in hipdnnEnginePluginGetEngineDetails for more about how this
  // function fits into the flow.

  LOG_API_ENTRY("handle=" << static_cast<void *>(handle) << ", engineDetails="
                          << static_cast<void *>(engineDetails));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(engineDetails);
  FUSILLI_PLUGIN_CHECK_NULL(engineDetails->ptr);

  // Deallocate engine details.
  handle->eraseEngineDetailsBuffer(engineDetails->ptr);
  engineDetails->ptr = nullptr;
  engineDetails->size = 0;

  LOG_API_SUCCESS_AUTO("engineDetails->ptr=" << engineDetails->ptr);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                   const hipdnnPluginConstData_t *engineConfig,
                                   const hipdnnPluginConstData_t *opGraph,
                                   size_t *workspaceSize) {
  LOG_API_ENTRY(
      "handle=" << static_cast<void *>(handle)
                << ", engineConfig=" << static_cast<const void *>(engineConfig)
                << ", opGraph=" << static_cast<const void *>(opGraph)
                << ", workspaceSize=" << static_cast<void *>(workspaceSize));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(engineConfig);
  FUSILLI_PLUGIN_CHECK_NULL(opGraph);
  FUSILLI_PLUGIN_CHECK_NULL(workspaceSize);

  // TODO(#197): Create a heuristic to estimate workspace size from the op graph
  // without requiring full compilation. For now, return 0 — the actual
  // workspace size will be reported by GetWorkspaceSizeFromExecutionContext
  // after the graph is compiled.
  *workspaceSize = 0;

  LOG_API_SUCCESS_AUTO("workspaceSize=" << *workspaceSize);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t *engineConfig,
    const hipdnnPluginConstData_t *opGraph,
    hipdnnEnginePluginExecutionContext_t *executionContext) {
  // See comment in hipdnnEnginePluginGetEngineDetails for more about how this
  // function fits into the flow.

  LOG_API_ENTRY("handle=" << static_cast<void *>(handle) << ", engineConfig="
                          << static_cast<const void *>(engineConfig)
                          << ", opGraph=" << static_cast<const void *>(opGraph)
                          << ", executionContext="
                          << static_cast<void *>(executionContext));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(engineConfig);
  FUSILLI_PLUGIN_CHECK_NULL(opGraph);
  FUSILLI_PLUGIN_CHECK_NULL(executionContext);

  // Ensure that config contains expected engine id.
  hipdnn_plugin_sdk::EngineConfigWrapper engineConfigWrapper(
      engineConfig->ptr, engineConfig->size);
  if (engineConfigWrapper.engineId() !=
      hipdnn_data_sdk::utilities::FUSILLI_ENGINE_ID) {
    return hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "unexpected engine id");
  }

  auto importAndCompile = [&handle](const hipdnnPluginConstData_t *opGraph)
      -> fusilli::ErrorOr<HipdnnEnginePluginExecutionContext> {
    // Import fusilli::Graph and compute UID -> fusilli::TensorAttr map for
    // graph boundary tensors.
    FUSILLI_ASSIGN_OR_RETURN(HipdnnEnginePluginExecutionContext graphImport,
                             importGraph(opGraph));

    // Compile graph
    FUSILLI_ASSIGN_OR_RETURN(auto fusilliHandle, handle->getFusilliHandle());
    FUSILLI_CHECK_ERROR(graphImport.graph.compile(fusilliHandle));

    return fusilli::ok(std::move(graphImport));
  };

  FUSILLI_PLUGIN_ASSIGN_OR_RETURN(auto importedGraph,
                                  importAndCompile(opGraph));
  *executionContext =
      new HipdnnEnginePluginExecutionContext(std::move(importedGraph));

  LOG_API_SUCCESS_AUTO(
      "created_execution_context=" << static_cast<void *>(*executionContext));
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext) {
  LOG_API_ENTRY("handle=" << static_cast<void *>(handle)
                          << ", executionContext="
                          << static_cast<void *>(executionContext));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(executionContext);

  delete executionContext;

  LOG_API_SUCCESS_AUTO("destroyed executionContext");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext,
    size_t *workspaceSize) {
  LOG_API_ENTRY(
      "handle=" << static_cast<void *>(handle) << ", executionContext="
                << static_cast<void *>(executionContext)
                << ", workspaceSize=" << static_cast<void *>(workspaceSize));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(executionContext);
  FUSILLI_PLUGIN_CHECK_NULL(workspaceSize);

  // This should never happen. When it does we'll at least get a sane error
  // message.
  std::optional<size_t> maybeSize = executionContext->graph.getWorkspaceSize();
  if (!maybeSize.has_value()) {
    return hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "Workspace size not available — graph may not be compiled");
  }
  *workspaceSize = *maybeSize;

  LOG_API_SUCCESS_AUTO("workspaceSize=" << *workspaceSize);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginExecuteOpGraph(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext, void *workspacePtr,
    const hipdnnPluginDeviceBuffer_t *deviceBuffers,
    uint32_t numDeviceBuffers) {
  // See comment in hipdnnEnginePluginGetEngineDetails for more about how this
  // function fits into the flow.

  LOG_API_ENTRY(
      "handle=" << static_cast<void *>(handle) << ", executionContext="
                << static_cast<void *>(executionContext)
                << ", workspace=" << workspacePtr << ", deviceBuffers="
                << static_cast<const void *>(deviceBuffers)
                << ", numDeviceBuffers=" << numDeviceBuffers);
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(executionContext);
  FUSILLI_PLUGIN_CHECK_NULL(deviceBuffers);

  // Get device allocator for buffer imports below.
  FUSILLI_PLUGIN_ASSIGN_OR_RETURN(auto fusilliHandle,
                                  handle->getFusilliHandle());
  iree_hal_allocator_t *deviceAllocator =
      iree_hal_device_allocator(fusilliHandle.get());

  // Fill variant pack for graph execution. Fusilli expects a variant pack to
  // map from fusilli::TensorAttr -> fusilli::Buffer for all boundary tensors.
  //
  // The execution context (created by hipdnnEnginePluginCreateExecutionContext)
  // holds a UID -> fusilli::TensorAttr mapping for all boundary tensors
  // already. To build the mapping we need to:
  //   1. Find the external HIP-allocated device buffer in `deviceBuffers`
  //      associated with UID.
  //   2. Import buffer from 1) into IREE runtime and create fusilli::Buffer.
  //
  // We may want to cache all of this in the future. As long as the device
  // pointers + UIDs haven't changed it should be possible to re-use an already
  // imported buffer + buffer view + the call that fusilli::Graph::execute
  // builds internally.
  std::unordered_map<std::shared_ptr<fusilli::TensorAttr>,
                     std::shared_ptr<fusilli::Buffer>>
      variantPack;
  for (auto &[uid, tensorAttr] : executionContext->uidToFusilliTensorAttr) {
    // 1. Find associated buffer.
    FUSILLI_PLUGIN_ASSIGN_OR_RETURN(
        hipdnnPluginDeviceBuffer_t hipMallocedBuffer,
        findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers));

    // 2. Import external buffer into IREE runtime and create fusilli::Buffer.
    FUSILLI_PLUGIN_ASSIGN_OR_RETURN(
        auto elementType,
        fusilliDataTypeToIreeHalDataType(tensorAttr->getDataType()));
    size_t sizeBytes = iree_hal_element_dense_byte_count(elementType) *
                       static_cast<size_t>(tensorAttr->getVolume());
    std::vector<int64_t> dims = tensorAttr->getPhysicalDim();
    std::vector<iree_hal_dim_t> shape(dims.begin(), dims.end());
    FUSILLI_PLUGIN_ASSIGN_OR_RETURN(
        auto fusilliBuffer,
        importDevicePointer(/*deviceAllocator=*/deviceAllocator,
                            /*devicePtr=*/hipMallocedBuffer.ptr,
                            /*sizeBytes=*/sizeBytes,
                            /*shape=*/shape,
                            /*elementType=*/elementType));
    variantPack[tensorAttr] = std::move(fusilliBuffer);
  }

  // Import workspace buffer if the compiled graph requires transient storage.
  std::shared_ptr<fusilli::Buffer> workspace = nullptr;
  std::optional<size_t> maybeWorkspaceSize =
      executionContext->graph.getWorkspaceSize();
  if (!maybeWorkspaceSize.has_value()) {
    return hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        "Workspace size not available — graph may not be compiled");
  }
  size_t workspaceSize = *maybeWorkspaceSize;
  if (workspaceSize > 0) {
    if (workspacePtr == nullptr) {
      return hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
          HIPDNN_PLUGIN_STATUS_BAD_PARAM,
          "Workspace of size " + std::to_string(workspaceSize) +
              " bytes required but workspace pointer is null");
    }
    // Workspace is opaque 1D array of bytes.
    iree_hal_dim_t workspaceShape[1] = {workspaceSize};
    FUSILLI_PLUGIN_ASSIGN_OR_RETURN(
        workspace,
        importDevicePointer(/*deviceAllocator=*/deviceAllocator,
                            /*devicePtr=*/workspacePtr,
                            /*sizeBytes=*/workspaceSize,
                            /*shape=*/workspaceShape,
                            /*elementType=*/IREE_HAL_ELEMENT_TYPE_UINT_8));
  }

  FUSILLI_PLUGIN_CHECK_ERROR(
      executionContext->graph.execute(fusilliHandle, variantPack, workspace));

  LOG_API_SUCCESS_AUTO("executed graph");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // extern "C"
