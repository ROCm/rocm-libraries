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

// hipDNN logging expects COMPONENT_NAME to be defined
#define COMPONENT_NAME FUSILLI_PLUGIN_NAME

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/vector.h>
#include <fusilli.h>
#include <hip/hip_runtime.h>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/logging/Logger.hpp>
#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>
#include <hipdnn_sdk/plugin/PluginHelpers.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <iree/hal/buffer.h>
#include <iree/hal/buffer_view.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "graph_import.h"
#include "hipdnn_engine_plugin_execution_context.h"
#include "hipdnn_engine_plugin_handle.h"
#include "utils.h"

using namespace hipdnn_plugin;
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
  LOG_API_ENTRY("name_ptr={:p}", static_cast<void *>(name));
  FUSILLI_PLUGIN_CHECK_NULL(name);

  *name = FUSILLI_PLUGIN_NAME;

  LOG_API_SUCCESS_AUTO("pluginName={}", *name);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char **version) {
  LOG_API_ENTRY("version_ptr={:p}", static_cast<void *>(version));
  FUSILLI_PLUGIN_CHECK_NULL(version);

  *version = fusilliPluginVersion;

  LOG_API_SUCCESS_AUTO("version={}", *version);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t *type) {
  LOG_API_ENTRY("type_ptr={:p}", static_cast<void *>(type));
  FUSILLI_PLUGIN_CHECK_NULL(type);

  *type = HIPDNN_PLUGIN_TYPE_ENGINE;

  LOG_API_SUCCESS_AUTO("type={}", *type);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void hipdnnPluginGetLastErrorString(const char **error_str) {
  if (error_str) {
    *error_str = hipdnn_plugin::PluginLastErrorManager::getLastError();
  }
}

// Once plugins are loaded via plugin manager then logging will work for them
hipdnnPluginStatus_t hipdnnPluginSetLoggingCallback(hipdnnCallback_t callback) {
  // No LOG_API_ENTRY as logging won't be wired up yet.
  FUSILLI_PLUGIN_CHECK_NULL(callback);

  hipdnn::logging::initializeCallbackLogging(FUSILLI_PLUGIN_NAME, callback);

  LOG_API_SUCCESS_AUTO("{}", "logging callback initialized");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ----------------------------------------------------------------------
// Implementations for engine plugin API defined in
// hipDNN/sdk/include/hipdnn_sdk/plugin/EnginePluginApi.h
// ----------------------------------------------------------------------

hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(int64_t *engineIds,
                                                       uint32_t maxEngines,
                                                       uint32_t *numEngines) {
  LOG_API_ENTRY("engineIds={:p}, maxEngines={}, numEngines={:p}",
                static_cast<void *>(engineIds), maxEngines,
                static_cast<void *>(numEngines));
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
    engineIds[0] = FUSILLI_PLUGIN_ENGINE_ID;
  }

  LOG_API_SUCCESS_AUTO("numEngines={}", *numEngines);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t *handle) {
  LOG_API_ENTRY("handle_ptr={:p}", static_cast<void *>(handle));
  FUSILLI_PLUGIN_CHECK_NULL(handle);

  // Get device id.
  int deviceId;
  FUSILLI_PLUGIN_CHECK_ERROR(hipGetDevice(&deviceId));

  // Create handle.
  *handle = new HipdnnEnginePluginHandle(deviceId);

  LOG_API_SUCCESS_AUTO("createdHandle={:p}", static_cast<void *>(*handle));
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle) {
  LOG_API_ENTRY("handle={:p}", static_cast<void *>(handle));
  FUSILLI_PLUGIN_CHECK_NULL(handle);

  delete handle;

  LOG_API_SUCCESS_AUTO("", "");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                            hipStream_t stream) {
  LOG_API_ENTRY("handle={:p}, stream_id={:p}", static_cast<void *>(handle),
                static_cast<void *>(stream));
  FUSILLI_PLUGIN_CHECK_NULL(handle);

  // Get device associated with stream.
  hipDevice_t deviceId;
  FUSILLI_PLUGIN_CHECK_ERROR(hipStreamGetDevice(stream, &deviceId));

  // This should never happen, check so that when it does we get a nice error
  // message.
  if (deviceId != handle->deviceId) {
    return hipdnn_plugin::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "Stream is associated with different device. Device reported "
        "through `hipStreamGetDevice` does not match active "
        "device reported through `hipGetDevice`.");
  }

  // Set stream, it will be used to create fusilli::Handle later.
  handle->setStream(stream);

  LOG_API_SUCCESS_AUTO("", "");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginGetApplicableEngineIds(
    hipdnnEnginePluginHandle_t handle, const hipdnnPluginConstData_t *opGraph,
    int64_t *engineIds, uint32_t maxEngines, uint32_t *numEngines) {
  LOG_API_ENTRY("handle={:p}, opGraph={:p}, engineIds={:p}, maxEngines={}, "
                "numEngines={:p}",
                static_cast<void *>(handle), static_cast<const void *>(opGraph),
                static_cast<void *>(engineIds), maxEngines,
                static_cast<void *>(numEngines));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(opGraph);
  if (maxEngines != 0) {
    FUSILLI_PLUGIN_CHECK_NULL(engineIds);
  }
  FUSILLI_PLUGIN_CHECK_NULL(numEngines);

  *numEngines = 0;
  if (maxEngines < 1) {
    HIPDNN_LOG_INFO(
        "Maximum number of engines reached ({}), ignoring additional "
        "engines, numEngines count: {}",
        maxEngines, *numEngines);
    LOG_API_SUCCESS_AUTO("numEngines={}", *numEngines);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
  }

  // Check for single conv_fprop node graph
  GraphWrapper opGraphWrapper(opGraph->ptr, opGraph->size);
  if (opGraphWrapper.nodeCount() != 1) {
    HIPDNN_LOG_INFO("Fusilli plan builder is (currently) only applicable only "
                    "for single node conv_fprop graphs.",
                    opGraphWrapper.nodeCount());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
  }
  if (!opGraphWrapper.hasOnlySupportedAttributes(
          std::set<hipdnn_sdk::data_objects::NodeAttributes>{
              hipdnn_sdk::data_objects::NodeAttributes::
                  ConvolutionFwdAttributes})) {
    HIPDNN_LOG_INFO("Fusilli plan builder is (currently) only applicable only "
                    "for single node conv_fprop graphs.",
                    opGraphWrapper.nodeCount());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
  }

  // Check single conv_fprop node for symmetric padding
  const hipdnn_sdk::data_objects::ConvolutionFwdAttributes *convFwdAttrs =
      opGraphWrapper.getNode(0).attributes_as_ConvolutionFwdAttributes();
  // pre/post_padding are flatbuffer::vectors (not std::vectors) and don't
  // override ==, so we use std::ranges::equal for structural vs referential
  // equality.
  if (!std::ranges::equal(*convFwdAttrs->pre_padding(),
                          *convFwdAttrs->post_padding())) { // C++ 20
    HIPDNN_LOG_INFO("Fusilli plan builder is (currently) requires symmetric "
                    "padding for conv_fprop nodes.",
                    opGraphWrapper.nodeCount());
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
  }

  // We have a single conv_fprop node with symmetric padding, the fusilli engine
  // is applicable.
  engineIds[0] = FUSILLI_PLUGIN_ENGINE_ID;
  *numEngines = 1;

  LOG_API_SUCCESS_AUTO("numEngines={}", *numEngines);
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

  LOG_API_ENTRY("handle={:p}, engineId={}, opGraph={:p}, engineDetails={:p}",
                static_cast<void *>(handle), engineId,
                static_cast<const void *>(opGraph),
                static_cast<void *>(engineDetails));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(opGraph);
  FUSILLI_PLUGIN_CHECK_NULL(engineDetails);

  if (engineId != FUSILLI_PLUGIN_ENGINE_ID) {
    return hipdnn_plugin::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "unexpected engine id");
  }

  // Build engine details object, we're only storing the engine id for the time
  // being.
  flatbuffers::FlatBufferBuilder builder;
  auto engineDetailsObj =
      hipdnn_sdk::data_objects::CreateEngineDetails(builder, engineId);
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

  LOG_API_SUCCESS_AUTO("engineDetails->ptr={:p}", engineDetails->ptr);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       hipdnnPluginConstData_t *engineDetails) {
  // See comment in hipdnnEnginePluginGetEngineDetails for more about how this
  // function fits into the flow.

  LOG_API_ENTRY("handle={:p}, engineDetails={:p}", static_cast<void *>(handle),
                static_cast<void *>(engineDetails));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(engineDetails);
  FUSILLI_PLUGIN_CHECK_NULL(engineDetails->ptr);

  // Deallocate engine details.
  handle->eraseEngineDetailsBuffer(engineDetails->ptr);
  engineDetails->ptr = nullptr;
  engineDetails->size = 0;

  LOG_API_SUCCESS_AUTO("engineDetails->ptr={:p}", engineDetails->ptr);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                   const hipdnnPluginConstData_t *engineConfig,
                                   const hipdnnPluginConstData_t *opGraph,
                                   size_t *workspaceSize) {
  LOG_API_ENTRY(
      "handle={:p}, engineConfig={:p}, opGraph={:p}, workspaceSize={:p}",
      static_cast<void *>(handle), static_cast<const void *>(engineConfig),
      static_cast<const void *>(opGraph), static_cast<void *>(workspaceSize));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(engineConfig);
  FUSILLI_PLUGIN_CHECK_NULL(opGraph);
  FUSILLI_PLUGIN_CHECK_NULL(workspaceSize);

  // TODO(#2309): for now we're focusing on kernels that don't require scratch
  // buffer space. Eventually we will need to teach IREE to report what scratch
  // buffer space required, and how to use a passed in pre-allocated scratch
  // space rather than a runtime allocated scratch space.
  *workspaceSize = 0;

  LOG_API_SUCCESS_AUTO("workspaceSize={}", *workspaceSize);
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t *engineConfig,
    const hipdnnPluginConstData_t *opGraph,
    hipdnnEnginePluginExecutionContext_t *executionContext) {
  // See comment in hipdnnEnginePluginGetEngineDetails for more about how this
  // function fits into the flow.

  LOG_API_ENTRY(
      "handle={:p}, engineConfig={:p}, opGraph={:p}, executionContext={:p}",
      static_cast<void *>(handle), static_cast<const void *>(engineConfig),
      static_cast<const void *>(opGraph),
      static_cast<void *>(executionContext));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(engineConfig);
  FUSILLI_PLUGIN_CHECK_NULL(opGraph);
  FUSILLI_PLUGIN_CHECK_NULL(executionContext);

  // Ensure that config contains expected engine id.
  hipdnn_plugin::EngineConfigWrapper engineConfigWrapper(engineConfig->ptr,
                                                         engineConfig->size);
  if (engineConfigWrapper.engineId() != FUSILLI_PLUGIN_ENGINE_ID) {
    return hipdnn_plugin::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM, "unexpected engine id");
  }

  auto importAndCompile = [&handle](const hipdnnPluginConstData_t *opGraph)
      -> fusilli::ErrorOr<HipdnnEnginePluginExecutionContext> {
    // Import fusilli::Graph and compute UID -> fusilli::TensorAttr map for
    // graph boundary tensors.
    HipdnnEnginePluginExecutionContext graphImport =
        FUSILLI_TRY(importGraph(opGraph));

    // Compile graph
    FUSILLI_CHECK_ERROR(graphImport.graph.validate());
    FUSILLI_CHECK_ERROR(
        graphImport.graph.compile(FUSILLI_TRY(handle->getFusilliHandle())));

    return fusilli::ok(std::move(graphImport));
  };

  *executionContext = new HipdnnEnginePluginExecutionContext(
      FUSILLI_PLUGIN_TRY(importAndCompile(opGraph)));

  LOG_API_SUCCESS_AUTO("created_execution_context={:p}",
                       static_cast<void *>(*executionContext));
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext) {
  LOG_API_ENTRY("handle={:p}, executionContext={:p}",
                static_cast<void *>(handle),
                static_cast<void *>(executionContext));
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(executionContext);

  delete executionContext;

  LOG_API_SUCCESS_AUTO("", "destroyed executionContext");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnEnginePluginExecuteOpGraph(
    hipdnnEnginePluginHandle_t handle,
    hipdnnEnginePluginExecutionContext_t executionContext, void *workspace,
    const hipdnnPluginDeviceBuffer_t *deviceBuffers,
    uint32_t numDeviceBuffers) {
  // See comment in hipdnnEnginePluginGetEngineDetails for more about how this
  // function fits into the flow.

  LOG_API_ENTRY(
      "handle={:p}, executionContext={:p}, workspace={:p}, deviceBuffers={:p}, "
      "numDeviceBuffers={}",
      static_cast<void *>(handle), static_cast<void *>(executionContext),
      workspace, static_cast<const void *>(deviceBuffers), numDeviceBuffers);
  FUSILLI_PLUGIN_CHECK_NULL(handle);
  FUSILLI_PLUGIN_CHECK_NULL(executionContext);
  FUSILLI_PLUGIN_CHECK_NULL(deviceBuffers);

  // Params and allocators hoisted out of loop below.
  iree_hal_allocator_t *deviceAllocator = iree_hal_device_allocator(
      FUSILLI_PLUGIN_TRY(handle->getFusilliHandle()).get());
  iree_allocator_t ireeHostAllocator = iree_allocator_system();
  iree_hal_buffer_params_t bufferParams = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

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
    hipdnnPluginDeviceBuffer_t hipMallocedBuffer = FUSILLI_PLUGIN_TRY(
        findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers));

    // 2.1. Import external buffer into IREE runtime. This isn't allocating a
    // buffer, it's making an existing allocation available to the IREE runtime.
    iree_hal_external_buffer_t externalBuffer = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
        .flags = 0,
        .size = static_cast<iree_device_size_t>(sizeof(float) *
                                                tensorAttr->getVolume()),
        .handle =
            {
                .device_allocation =
                    {
                        .ptr = (uint64_t)hipMallocedBuffer.ptr,
                    },
            },
    };
    iree_hal_buffer_t *importedBuffer = nullptr;
    FUSILLI_PLUGIN_CHECK_ERROR(iree_hal_allocator_import_buffer(
        deviceAllocator, bufferParams, &externalBuffer,
        iree_hal_buffer_release_callback_null(), &importedBuffer));

    // 2.2. Create a buffer view for external buffer.
    iree_hal_buffer_view_t *outBufferView = nullptr;
    FUSILLI_PLUGIN_CHECK_ERROR(iree_hal_buffer_view_create(
        /*buffer=*/importedBuffer, /*shape_rank=*/tensorAttr->getDim().size(),
        /*shape=*/(const iree_hal_dim_t *)tensorAttr->getDim().data(),
        /*element_type=*/
        FUSILLI_PLUGIN_TRY(
            fusilliDataTypeToIreeHalDataType(tensorAttr->getDataType())),
        /*encoding_type=*/IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        /*host_allocator=*/ireeHostAllocator,
        /*out_buffer_view=*/&outBufferView));

    // Release our reference to buffer. The buffer view holds a reference to
    // buffer and will handle release + possible destruction when it's
    // destroyed.
    iree_hal_buffer_release(importedBuffer);

    // 2.3. Create fusilli::Buffer from buffer view. Buffer::import is a RAII
    // type that retains the buffer view, incrementing its reference count, on
    // construction and releases the buffer view on destruction.
    variantPack[tensorAttr] = std::make_shared<fusilli::Buffer>(
        FUSILLI_PLUGIN_TRY(fusilli::Buffer::import(outBufferView)));

    // Release our reference to buffer view. The buffer view and buffer will
    // (now) be tied to fusilli::Buffer's lifetime as it holds the only
    // reference to the buffer view.
    iree_hal_buffer_view_release(outBufferView);
  }

  FUSILLI_PLUGIN_CHECK_ERROR(executionContext->graph.execute(
      FUSILLI_PLUGIN_TRY(handle->getFusilliHandle()), variantPack));

  LOG_API_SUCCESS_AUTO("{}", "executed graph");
  return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // extern "C"
