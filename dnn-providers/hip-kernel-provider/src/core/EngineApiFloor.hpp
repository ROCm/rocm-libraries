// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string_view>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

namespace hip_kernel_provider::core
{

/// The engine plugin API the compiled-in engines (HIP_MLOPS, HIP_FLASH2, ASM_SDPA) are
/// written against. The plugin advertises a newer API so that ingestor engines whose
/// descriptors declare a newer sdk_version receive ragged, custom-alignment and
/// token-unit ragged graphs. The compiled-in engines predate those features and keep
/// declining such graphs, exactly as when the whole plugin advertised this version.
inline constexpr std::string_view COMPILED_IN_ENGINE_API_VERSION
    = hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION;

/// True when @p opGraph needs a newer engine plugin API than the compiled-in engines
/// understand. An unstamped graph reads as the baseline.
inline bool needsNewerThanCompiledInEngineApi(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    static const hipdnn_data_sdk::utilities::Version s_understood{COMPILED_IN_ENGINE_API_VERSION};
    return s_understood < hipdnn_plugin_sdk::fromEngineApiVersion(
               opGraph.getGraph().min_required_engine_api_version());
}

} // namespace hip_kernel_provider::core
