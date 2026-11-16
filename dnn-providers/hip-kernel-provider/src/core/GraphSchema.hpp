// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

/**
 * @file GraphSchema.hpp
 * @brief The graph schema each of the provider's engines understands, and the one a graph needs.
 *
 * The provider declares hipDNN's newest engine plugin API (PluginPublic.cpp), so hipDNN's
 * backend routes it graphs that use any schema field up to it: ragged tensors and declared
 * alignments (1.3.0), ragged offset multipliers (1.4.0). An engine reads those fields only once
 * it says so: each engine declares the newest graph schema its matchers were reviewed against
 * (Container's engine table), PROVIDER_BASELINE_GRAPH_SCHEMA unless it opts in, and the
 * ApplicabilityManager declines a graph for every engine below the graph's floor before the
 * engine runs. Descriptor-driven (ingestor) engines declare their UED's `sdk_version`, the claim
 * their own plan builder enforces too.
 */
namespace hip_kernel_provider::core
{

/// The schema every engine understood before the provider declared more: runtime pass-by-value
/// tensors (1.2.0), the engine plugin API the provider declared until then.
inline hipdnn_data_sdk::utilities::Version providerBaselineGraphSchema()
{
    return hipdnn_data_sdk::utilities::Version{hipdnn_plugin_sdk::K_PASS_BY_VALUE_MIN_API_VERSION};
}

/// The newest graph schema @p opGraph uses: the floor the writer stamped on it, or, for a graph
/// no writer stamped (a test fixture, a graph built from JSON), the one its fields imply, as
/// hipDNN's backend computes it; the larger of the two.
inline hipdnn_data_sdk::utilities::Version
    graphSchemaFloor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    const auto& graph = opGraph.getGraph();
    bool runtimePassByValue = false;
    bool ragged = false;
    bool aligned = false;
    bool multiplier = false;
    if(const auto* tensors = graph.tensors())
    {
        for(const auto* tensor : *tensors)
        {
            runtimePassByValue = runtimePassByValue || tensor->is_runtime_pass_by_value();
            ragged = ragged || tensor->ragged_offset_tensor_uid().has_value();
            aligned = aligned || tensor->alignment() != 16;
            multiplier = multiplier || tensor->ragged_offset_multiplier() != 1;
        }
    }
    const auto implied = hipdnn_plugin_sdk::computeMinimumEnginePluginApiVersion(
        graph.is_override_shape_enabled(), runtimePassByValue, ragged, aligned, multiplier);
    const auto stamped
        = hipdnn_plugin_sdk::fromEngineApiVersion(graph.min_required_engine_api_version());
    return std::max(implied, stamped, [](const auto& lhs, const auto& rhs) { return lhs < rhs; });
}

} // namespace hip_kernel_provider::core
