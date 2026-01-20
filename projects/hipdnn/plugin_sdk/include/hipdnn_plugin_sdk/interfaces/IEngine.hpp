// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hipdnn_data_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Interface for an engine.
 *
 * An IEngine represents an engine that can handle one or more graphs of operations.
 * For example, a BatchNormEngine might be able to handle single-op and simple fused
 * graphs that contain batchnorm operations.
 *
 * Engines are responsible for:
 * - Providing a unique identifier
 * - Determining if they can handle a given graph
 * - Providing engine details (capabilities, behavioral notes)
 * - Calculating workspace requirements
 * - Creating executable plans via their plan builders
 *
 * @note Engines typically have the same lifecycle as the EngineManager that contains
 *       them. Implementations should be stateless or thread-safe.
 *
 * @note You probably don't want to have a massive number of engines in your plugin.
 *       The plugin is responsible for choosing its solution given a graph, and without
 *       sampling. The plan should be finalized and ready to go by execution time.
 */
class IEngine
{
public:
    virtual ~IEngine() = default;

    /**
     * @brief Returns the unique identifier for this engine.
     *
     * @return The engine's unique ID.
     */
    virtual int64_t id() const = 0;

    /**
     * @brief Checks if this engine can handle the given operation graph.
     *
     * Typically this is implemented by checking all the engine's plan builders.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph to check.
     * @return true if this engine can handle the graph, false otherwise.
     *
     * @note Only a single plan builder should be applicable per engine for a given graph.
     *       If multiple plan builders have overlapping graph support, it's up to the
     *       plugin implementor to decide how to handle this selection.
     */
    virtual bool isApplicable(HipdnnEnginePluginHandle& handle, const IGraph& opGraph) const = 0;

    /**
     * @brief Gets the details of this engine.
     *
     * Engine details include information about the engine's capabilities,
     * behavioral notes, and supported configurations.
     *
     * @param handle The engine plugin handle.
     * @param detailsOut Output parameter for the engine details data.
     *                   The caller is responsible for freeing this data.
     */
    virtual void getDetails(HipdnnEnginePluginHandle& handle,
                            hipdnnPluginConstData_t& detailsOut) const = 0;

    /**
     * @brief Returns the maximum workspace size required for the given graph.
     *
     * This is typically a pass-through to the applicable plan builders, taking
     * the maximum of all workspaces queried.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @return The maximum workspace size in bytes.
     */
    virtual size_t getMaxWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                       const IGraph& opGraph) const = 0;

    /**
     * @brief Creates an executable plan for the given graph and configuration.
     *
     * This is a pass-through to the appropriate plan builder. It's expected that
     * only one plan builder will be applicable for a given graph.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @param engineConfig The engine configuration settings.
     * @return A unique pointer to the created plan.
     * @throws HipdnnPluginException if no applicable plan builder is found or
     *         if plan creation fails.
     */
    virtual std::unique_ptr<IPlan> createPlan(const HipdnnEnginePluginHandle& handle,
                                              const IGraph& opGraph,
                                              const IEngineConfig& engineConfig) const = 0;
};

} // namespace hipdnn_plugin_sdk
