// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <string_view>

namespace hipdnn_plugin_sdk
{

// Baseline engine plugin C ABI version for plugins that do not export
// `hipdnnPluginGetApiVersion`. This preserves compatibility with existing
// plugins across the first explicit engine-plugin API versioning rollout.
inline constexpr std::string_view K_ENGINE_PLUGIN_API_VERSION_BASELINE = "1.0.0";

// Minimum engine plugin C ABI version that advertises support for the
// override-execute entry point (RFC 0008 §4.5). Override-execute is the
// additive minor feature introduced in engine plugin API 1.1.0; see
// `engine_api_version.h` for the canonical MAJOR.MINOR.PATCH macros. The
// host's applicability filter rejects any plugin reporting an API version
// strictly less than this when the graph opts into overridable shapes.
inline constexpr std::string_view K_OVERRIDE_EXECUTE_MIN_API_VERSION = "1.1.0";

// Minimum engine plugin C ABI version that advertises support for runtime
// pass-by-value scalar tensors (RFC 0016). This is the additive minor feature
// introduced in engine plugin API 1.2.0; see `engine_api_version.h` for the
// canonical MAJOR.MINOR.PATCH macros. The host's applicability filter rejects
// any plugin reporting an API version strictly less than this when the graph
// contains any runtime pass-by-value tensor.
inline constexpr std::string_view K_PASS_BY_VALUE_MIN_API_VERSION = "1.2.0";

/// @brief Computes the minimum engine plugin API version a graph requires,
/// given the graph-level feature flags that gate additive plugin ABI surface.
///
/// This is the single source of truth for the graph -> required-API-version
/// mapping: GraphDescriptor stamps the result into the serialized graph's
/// `min_required_engine_api_version` field (as an EngineApiVersion struct, see
/// graph.fbs) at build/deserialize time, and EnginePluginResourceManager's
/// applicability filter calls it directly to decide which loaded plugins can
/// serve a graph. Keeping both call sites on this one function means the
/// deserialize-time reader-version guard and the plugin-version floor can
/// never drift apart.
///
/// Runtime pass-by-value (1.2.0) dominates the override-execute floor (1.1.0)
/// and the baseline (1.0.0); each is an additive minor feature layered on the
/// last, so the highest applicable floor wins.
inline const hipdnn_data_sdk::utilities::Version&
    computeMinimumEnginePluginApiVersion(bool isOverrideShapeEnabled, bool isRuntimePassByValue)
{
    static const hipdnn_data_sdk::utilities::Version s_baselineVersion{
        K_ENGINE_PLUGIN_API_VERSION_BASELINE};
    static const hipdnn_data_sdk::utilities::Version s_overrideExecuteMinVersion{
        K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    static const hipdnn_data_sdk::utilities::Version s_passByValueMinVersion{
        K_PASS_BY_VALUE_MIN_API_VERSION};

    if(isRuntimePassByValue)
    {
        return s_passByValueMinVersion;
    }
    if(isOverrideShapeEnabled)
    {
        return s_overrideExecuteMinVersion;
    }
    return s_baselineVersion;
}

/// @brief Converts a Version to the flatbuffer EngineApiVersion struct for
/// stamping into a serialized Graph.
inline hipdnn_flatbuffers_sdk::data_objects::EngineApiVersion
    toEngineApiVersion(const hipdnn_data_sdk::utilities::Version& version)
{
    return {static_cast<uint32_t>(version.major),
            static_cast<uint32_t>(version.minor),
            static_cast<uint32_t>(version.patch)};
}

/// @brief Converts a serialized graph's EngineApiVersion struct back to a Version
/// for comparison against the engine-plugin version constants above.
inline hipdnn_data_sdk::utilities::Version
    fromEngineApiVersion(const hipdnn_flatbuffers_sdk::data_objects::EngineApiVersion& version)
{
    return {static_cast<int>(version.major()),
            static_cast<int>(version.minor()),
            static_cast<int>(version.patch())};
}

/// @brief Same as above, but tolerates a graph a writer never stamped (e.g. a
/// hand-built test fixture or a graph deserialized from JSON): a null pointer
/// reads as the baseline "1.0.0" floor, mirroring the pre-EngineApiVersion
/// `min_reader_version`'s implicit `0` default for unstamped graphs.
inline hipdnn_data_sdk::utilities::Version
    fromEngineApiVersion(const hipdnn_flatbuffers_sdk::data_objects::EngineApiVersion* version)
{
    if(version == nullptr)
    {
        return hipdnn_data_sdk::utilities::Version{K_ENGINE_PLUGIN_API_VERSION_BASELINE};
    }
    return fromEngineApiVersion(*version);
}

} // namespace hipdnn_plugin_sdk
