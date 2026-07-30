// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <string_view>

namespace hipdnn_plugin_sdk
{

// Baseline engine plugin C ABI version for plugins that don't export
// `hipdnnPluginGetApiVersion`; preserves compatibility predating explicit
// engine-plugin API versioning.
inline constexpr std::string_view K_ENGINE_PLUGIN_API_VERSION_BASELINE = "1.0.0";

// Minimum engine plugin API version for the override-execute entry point
// (RFC 0008 §4.5), an additive minor feature. The applicability filter
// rejects plugins below this when the graph opts into overridable shapes.
inline constexpr std::string_view K_OVERRIDE_EXECUTE_MIN_API_VERSION = "1.1.0";

// Minimum engine plugin API version for runtime pass-by-value scalar
// tensors (RFC 0016), an additive minor feature. The applicability filter
// rejects plugins below this when the graph contains any such tensor.
inline constexpr std::string_view K_PASS_BY_VALUE_MIN_API_VERSION = "1.2.0";

// Minimum engine plugin C ABI version that advertises support for ragged
// tensors (RFC 0014). Introduced in engine plugin API 1.2.0. The host's
// applicability filter rejects any plugin reporting an API version strictly
// less than this when the graph contains ragged tensors.
inline constexpr std::string_view K_RAGGED_TENSOR_MIN_API_VERSION = "1.3.0";

// Minimum engine plugin API version for tensors carrying a non-default byte
// alignment. Plugins predating this version are assumed to require the default
// (16-byte) alignment, so the applicability filter rejects any plugin reporting
// an API version strictly less than this when the graph sets a custom alignment.
inline constexpr std::string_view K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION
    = K_RAGGED_TENSOR_MIN_API_VERSION;

// Deserialize ceiling: a graph whose min_required_engine_api_version exceeds this
// is rejected. Must equal the highest feature-gated version constant above.
inline constexpr std::string_view K_MAX_SUPPORTED_API_VERSION = K_RAGGED_TENSOR_MIN_API_VERSION;

/// @brief Computes the minimum engine plugin API version a graph requires,
/// from the graph-level feature flags gating additive plugin ABI surface.
///
/// Single source of truth for graph -> required-API-version: GraphDescriptor
/// stamps the result into `min_required_engine_api_version` (graph.fbs) at
/// build/deserialize time, and EnginePluginResourceManager's applicability
/// filter calls it to pick loaded plugins. One function keeps the
/// deserialize-time reader-version guard and the plugin-version floor from
/// drifting apart.
///
/// Runtime pass-by-value (1.2.0) dominates override-execute (1.1.0) and the
/// baseline (1.0.0): the highest applicable floor wins.
inline const hipdnn_data_sdk::utilities::Version&
    computeMinimumEnginePluginApiVersion(bool isOverrideShapeEnabled,
                                         bool isRuntimePassByValue,
                                         bool isRaggedTensorEnabled,
                                         bool hasNonDefaultTensorAlignment)
{
    static const hipdnn_data_sdk::utilities::Version s_baselineVersion{
        K_ENGINE_PLUGIN_API_VERSION_BASELINE};
    static const hipdnn_data_sdk::utilities::Version s_overrideExecuteMinVersion{
        K_OVERRIDE_EXECUTE_MIN_API_VERSION};
    static const hipdnn_data_sdk::utilities::Version s_passByValueMinVersion{
        K_PASS_BY_VALUE_MIN_API_VERSION};
    static const hipdnn_data_sdk::utilities::Version s_raggedTensorMinVersion{
        K_RAGGED_TENSOR_MIN_API_VERSION};
    static const hipdnn_data_sdk::utilities::Version s_tensorAlignmentMinVersion{
        K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION};

    // NOTE: MUST be ordered by highest version to lowest
    if(isRaggedTensorEnabled)
    {
        return s_raggedTensorMinVersion;
    }
    if(hasNonDefaultTensorAlignment)
    {
        return s_tensorAlignmentMinVersion;
    }
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
