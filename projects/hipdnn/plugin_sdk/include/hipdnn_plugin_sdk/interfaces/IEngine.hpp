// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
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
 * @tparam THandle The plugin-specific handle type (e.g., HipdnnMiopenHandle).
 * @tparam TSettings The plugin-specific settings type (e.g., HipdnnMiopenSettings).
 * @tparam TContext The plugin-specific context type (e.g., HipdnnMiopenContext).
 *
 * @note Engines should be stateless. Any state required for execution should be stored on
 *  an execution context.
 */
template <typename THandle, typename TSettings, typename TContext>
class IEngine
{
public:
    virtual ~IEngine() = default;

    /**
     * @brief Returns the unique identifier for this engine.
     *
     * @return The engine's unique ID.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
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
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual bool isApplicable( // NOLINT(portability-template-virtual-member-function)
        THandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
        = 0;

    /**
     * @brief Gets the details of this engine.
     *
     * Engine details include information about the engine's capabilities,
     * behavioral notes, and supported configurations. The operation graph
     * is provided so that the engine can query applicable plan builders
     * for custom knobs.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @param detailsOut Output parameter for the engine details data.
     *                   The caller is responsible for freeing this data.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual void getDetails(THandle& handle,
                            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                            hipdnnPluginConstData_t& detailsOut) const
        = 0;

    /**
     * @brief Returns the maximum workspace size required for the given graph.
     *
     * This is typically a pass-through to the applicable plan builders, taking
     * the maximum of all workspaces queried.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @param engineConfig The engine configuration.
     * @return The maximum workspace size in bytes.
     */
    virtual size_t
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        getMaxWorkspaceSize(
            const THandle& handle,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
        = 0;

    /**
     * @brief Initializes the execution context with a plan for the given graph.
     *
     * This is a pass-through to the appropriate plan builder. It's expected that
     * only one plan builder will be applicable for a given graph. The built plan
     * is set on the execution context.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @param engineConfig The engine configuration settings.
     * @param executionContext The execution context to initialize with a plan.
     * @throws HipdnnPluginException if no applicable plan builder is found or
     *         if plan creation fails.
     */
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual void initializeExecutionContext(
        const THandle& handle,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
        TContext& executionContext) const
        = 0;
};

} // namespace hipdnn_plugin_sdk
