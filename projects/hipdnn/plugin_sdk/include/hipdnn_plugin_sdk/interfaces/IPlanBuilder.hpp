// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <memory>

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
    virtual bool isApplicable(const HipdnnEnginePluginHandle& handle,
                              const IGraph& opGraph) const = 0;

    /**
     * @brief Returns the maximum workspace size required for the given graph.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @return The maximum workspace size in bytes.
     */
    virtual size_t getMaxWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                       const IGraph& opGraph) const = 0;

    /**
     * @brief Builds an executable plan for the given graph.
     *
     * Creates an IPlan instance that can execute the operation graph.
     * The returned plan should be ready for execution.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph to build a plan for.
     * @return A unique pointer to the created plan.
     * @throws HipdnnPluginException if the plan cannot be built.
     */
    virtual std::unique_ptr<IPlan> buildPlan(const HipdnnEnginePluginHandle& handle,
                                             const IGraph& opGraph) const = 0;
};

} // namespace hipdnn_plugin_sdk
