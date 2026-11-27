// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "MiopenPluginDefines.hpp"
#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>

// Engine plugin API 1.1.0 (RFC 0008) adds an optional override-shape execute entry point
// that EnginePluginImpl.inl does not generate. MIOpen plans bake the graph's dimensions into
// their descriptors when they are built, so every plan builder declines override-shape graphs
// during applicability and no execution context of this provider is built for one. This
// entry therefore rejects every call.
extern "C" HIPDNN_PLUGIN_NODISCARD HIPDNN_PLUGIN_EXPORT hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraphWithOverrides(
        hipdnnEnginePluginHandle_t handle,
        hipdnnEnginePluginExecutionContext_t executionContext,
        [[maybe_unused]] void* workspace,
        [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
        [[maybe_unused]] uint32_t numDeviceBuffers,
        uint32_t numOverrides,
        [[maybe_unused]] const int64_t* overrideUniqueIds,
        [[maybe_unused]] const uint32_t* overrideLengths,
        [[maybe_unused]] const int64_t* const* overrideShapes,
        [[maybe_unused]] const int64_t* const* overrideStrides)
{
    LOG_API_ENTRY("handle=" << static_cast<void*>(handle)
                            << ", executionContext=" << static_cast<void*>(executionContext)
                            << ", numOverrides=" << numOverrides);

    return hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "The MIOpen provider does not support override-shape execution; its plan builders "
        "decline override-shape graphs during applicability");
}
