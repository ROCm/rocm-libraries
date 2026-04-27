// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
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
 * @tparam THandle The plugin-specific handle type (e.g., HipdnnMiopenHandle).
 * @tparam TSettings The plugin-specific settings type (e.g., HipdnnMiopenSettings).
 * @tparam TContext The plugin-specific context type (e.g., HipdnnMiopenContext).
 *
 * @note Implementations should be stateless or thread-safe, as plan builders
 *       may be accessed from multiple threads concurrently.
 */
template <typename THandle, typename TSettings, typename TContext>
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
    virtual bool
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        isApplicable(const THandle& handle,
                     const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;

    /**
     * @brief Returns the maximum workspace size required for the given graph.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @param executionSettings The plugin-specific execution settings.
     * @return The maximum workspace size in bytes.
     */
    virtual size_t
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getMaxWorkspaceSize(const THandle& handle,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                            const TSettings& executionSettings) const
        = 0;

    /**
     * @brief Initializes execution settings from an engine configuration.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @param engineConfig The engine configuration containing knob settings.
     * @param executionSettings The settings to initialize.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual void initializeExecutionSettings(
        const THandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        TSettings& executionSettings) const
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
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual void buildPlan(
        const THandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig&
            engineConfig,
        TContext& executionContext) const
        = 0;

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
    virtual std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT>
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getCustomKnobs(const THandle& handle,
                       const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;
};

} // namespace hipdnn_plugin_sdk
