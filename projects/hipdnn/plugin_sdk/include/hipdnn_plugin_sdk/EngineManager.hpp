// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Manages a collection of engines within a plugin.
 *
 * The EngineManager is responsible for:
 * - Maintaining a registry of all engines in the plugin
 * - Finding applicable engines for a given operation graph
 * - Delegating operations to the appropriate engine based on engine ID
 *
 * This class is typically owned by an EnginePluginContainer and shares
 * its lifespan with it.
 *
 * @tparam THandle The plugin-specific handle type (e.g., HipdnnMiopenHandle).
 * @tparam TSettings The plugin-specific settings type (e.g., HipdnnMiopenSettings).
 * @tparam TContext The plugin-specific context type (e.g., HipdnnMiopenContext).
 *
 * @note Implementations should be stateless or thread-safe, as the engine
 *       manager may be accessed from multiple threads concurrently.
 */
template <typename THandle, typename TSettings, typename TContext>
class EngineManager
{
public:
    using Engine = IEngine<THandle, TSettings, TContext>;
    using IGraph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph;
    using IEngineConfig = hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig;

    EngineManager() = default;
    virtual ~EngineManager() = default;

    // Disallow copy and assignment
    EngineManager(const EngineManager&) = delete;
    EngineManager& operator=(const EngineManager&) = delete;

    // Allow move
    EngineManager(EngineManager&&) = default;
    EngineManager& operator=(EngineManager&&) = default;

    /**
     * @brief Adds an engine to this manager.
     *
     * @param engine The engine to add. Ownership is transferred.
     */
    void addEngine(std::unique_ptr<Engine> engine)
    {
        auto id = engine->id();
        _engines.emplace(id, std::move(engine));
    }

    /**
     * @brief Returns the IDs of all engines managed by this manager.
     *
     * @return Vector of all engine IDs.
     */
    std::vector<int64_t> getAllEngineIds() const
    {
        std::vector<int64_t> ids;
        ids.reserve(_engines.size());
        for(const auto& [id, engine] : _engines)
        {
            ids.push_back(id);
        }
        return ids;
    }

    /**
     * @brief Returns the IDs of engines that can handle the given graph.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph to check.
     * @return Vector of applicable engine IDs.
     */
    std::vector<int64_t> getApplicableEngineIds(THandle& handle, const IGraph& opGraph)
    {
        std::vector<int64_t> applicable;
        for(const auto& [id, engine] : _engines)
        {
            if(engine->isApplicable(handle, opGraph))
            {
                applicable.push_back(id);
            }
        }
        return applicable;
    }

    /**
     * @brief Gets the details of a specific engine.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph (may be used for context).
     * @param engineId The ID of the engine.
     * @param engineDetailsOut Output parameter for the engine details.
     */
    void getEngineDetails(THandle& handle,
                          const IGraph& opGraph,
                          int64_t engineId,
                          hipdnnPluginConstData_t& engineDetailsOut)
    {
        auto& engine = getEngine(engineId);
        engine.getDetails(handle, opGraph, engineDetailsOut);
    }

    /**
     * @brief Gets the workspace size for a specific engine and graph.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @param engineConfig The engine configuration.
     * @return The required workspace size in bytes.
     */
    size_t getMaxWorkspaceSize(const THandle& handle,
                               const IGraph& opGraph,
                               const IEngineConfig& engineConfig) const
    {
        auto& engine = getEngine(engineConfig.engineId());
        return engine.getMaxWorkspaceSize(handle, opGraph, engineConfig);
    }

    /**
     * @brief Initializes an execution context with a plan for the given graph.
     *
     * Creates a plan using the appropriate engine and attaches it to the
     * execution context.
     *
     * @param handle The engine plugin handle.
     * @param opGraph The operation graph.
     * @param engineConfig The engine configuration.
     * @param executionContext The execution context to initialize.
     */
    void initializeExecutionContext(const THandle& handle,
                                    const IGraph& opGraph,
                                    const IEngineConfig& engineConfig,
                                    TContext& executionContext) const
    {
        auto& engine = getEngine(engineConfig.engineId());
        engine.initializeExecutionContext(handle, opGraph, engineConfig, executionContext);
    }

protected:
    /**
     * @brief Gets an engine by ID.
     *
     * @param engineId The ID of the engine to get.
     * @return Reference to the engine.
     * @throws HipdnnPluginException if the engine is not found.
     */
    Engine& getEngine(int64_t engineId) const
    {
        auto it = _engines.find(engineId);
        if(it == _engines.end())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                        "Engine with ID " + std::to_string(engineId)
                                            + " not found.");
        }
        return *it->second;
    }

private:
    std::unordered_map<int64_t, std::unique_ptr<Engine>> _engines;
};

} // namespace hipdnn_plugin_sdk
