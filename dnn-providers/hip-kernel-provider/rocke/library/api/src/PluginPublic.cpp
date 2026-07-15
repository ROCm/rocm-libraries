// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "PluginDefines.hpp"
#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>

// Engine plugin C ABI 1.1.0 (RFC 0008 section 4.5) adds the optional
// override-shape execute entry point. EnginePluginImpl.inl generates every
// required symbol but not this optional one, so the rocKE client exports it here
// to be a fully wired 1.1 plugin (see HIPDNN_PLUGIN_API_VERSION in
// PluginDefines.hpp).
//
// rocKE does not serve override-shape graphs today: SdpaGraphAdapter declines any
// graph with is_override_shape_enabled during applicability, so the host never
// selects this engine for override execution and this entry is unreachable in
// normal operation. It is implemented as an explicit, clean decline rather than
// an empty stub so a contract violation surfaces loudly instead of silently
// mis-executing with runtime shapes the selected kernel was not built for.
extern "C" hipdnnPluginStatus_t hipdnnEnginePluginExecuteOpGraphWithOverrides(
    [[maybe_unused]] hipdnnEnginePluginHandle_t handle,
    [[maybe_unused]] hipdnnEnginePluginExecutionContext_t executionContext,
    [[maybe_unused]] void* workspace,
    [[maybe_unused]] const hipdnnPluginDeviceBuffer_t* deviceBuffers,
    [[maybe_unused]] uint32_t numDeviceBuffers,
    [[maybe_unused]] uint32_t numOverrides,
    [[maybe_unused]] const int64_t* overrideUniqueIds,
    [[maybe_unused]] const uint32_t* overrideLengths,
    [[maybe_unused]] const int64_t* const* overrideShapes,
    [[maybe_unused]] const int64_t* const* overrideStrides)
{
    return hipdnn_plugin_sdk::PluginLastErrorManager::setLastError(
        HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE,
        "rocke-client does not support override-shape execution; such graphs are declined "
        "during applicability");
}
