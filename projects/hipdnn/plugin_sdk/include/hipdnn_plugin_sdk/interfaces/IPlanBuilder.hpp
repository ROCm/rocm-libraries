// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Interface for a plan builder.
 *
 * An IPlanBuilder handles specific graphs of operations. It is responsible for:
 * - Determining if a graph is applicable (can be handled by this builder)
 * - Calculating the required workspace size
 * - Building an executable plan for the graph
 * - Optionally providing custom knobs for engine configuration
 *
 * Plan builders are typically owned by an engine and have the same lifecycle
 * as the engine that contains them.
 *
 * @note Implementations should be stateless or thread-safe, as plan builders
 *       may be accessed from multiple threads concurrently.
 */
class IPlanBuilder
{
public:
    virtual ~IPlanBuilder() = default;

    /**
     * @brief Checks if this plan builder can handle the given operation graph.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph to check.
     * @return true if this plan builder can handle the graph, false otherwise.
     */
    virtual bool isApplicable(const HipdnnEnginePluginHandle& handle, const IGraph& opGraph) const
        = 0;

    /**
     * @brief Returns the maximum workspace size required for the given graph.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @return The maximum workspace size in bytes.
     */
    virtual size_t getMaxWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                       const IGraph& opGraph) const
        = 0;

    /**
     * @brief Builds an executable plan for the given graph.
     *
     * Creates an IPlan instance and stores it on the execution context.
     * The plan should be ready for execution after this call.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph to build a plan for.
     * @param engineConfig The engine configuration containing knob settings.
     *                     May be unused if the plan builder has no custom knobs.
     * @param executionContext The execution context to store the plan on.
     * @throws HipdnnPluginException if the plan cannot be built.
     */
    virtual void buildPlan(const HipdnnEnginePluginHandle& handle,
                           const IGraph& opGraph,
                           [[maybe_unused]] const IEngineConfig& engineConfig,
                           HipdnnEnginePluginExecutionContext& executionContext) const
        = 0;

    /**
     * @brief Checks if this plan builder has custom knobs.
     *
     * Custom knobs allow plan builders to expose configuration options
     * that can be set by the user when creating an engine configuration.
     *
     * @return true if this plan builder has custom knobs, false otherwise.
     */
    virtual bool hasCustomKnobs() const = 0;

    /**
     * @brief Gets the custom knobs for this plan builder.
     *
     * This method returns the knob definitions that this plan builder supports.
     * The caller is responsible for converting to FlatBuffers if needed.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @return A vector of KnobT objects representing the custom knobs.
     */
    virtual std::vector<hipdnn_data_sdk::data_objects::KnobT>
        getCustomKnobs(const HipdnnEnginePluginHandle& handle, const IGraph& opGraph) const = 0;
};

} // namespace hipdnn_plugin_sdk
