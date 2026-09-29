// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/interfaces/IEngine.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Owns a plugin's engine registry: lookup, applicability checks, and dispatch by id.
 * Typically owned by an EnginePluginContainer and shares its lifespan.
 *
 * @tparam THandle Plugin handle type (e.g. HipdnnMiopenHandle).
 * @tparam TSettings Plugin settings type.
 * @tparam TContext Plugin execution context type.
 *
 * @note Thread-safe: may be accessed from multiple threads concurrently.
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

    EngineManager(const EngineManager&) = delete;
    EngineManager& operator=(const EngineManager&) = delete;

    EngineManager(EngineManager&&) = default;
    EngineManager& operator=(EngineManager&&) = default;

    /**
     * @brief Adds an engine, keyed by its id.
     *
     * An id already claimed keeps the incumbent and logs the loser rather than throwing:
     * two engines hashing onto one id (RFC 0003) is an authoring collision, and this is
     * the backstop for pairs the descriptor policy can't see -- RFC 0020 §10.2.1 drops
     * every UED in a name collision before it reaches this list, but two hand-written
     * engines have no such gate, and refusing both here would take a working engine down
     * to punish one duplicate.
     *
     * @param name Optional declared name, used to identify the loser in that log. Only the
     * caller can supply it for a runtime-discovered engine, which has no registry entry.
     */
    void addEngine(std::unique_ptr<Engine> engine, std::string_view name = {})
    {
        auto id = engine->id();
        if(!_engines.emplace(id, std::move(engine)).second)
        {
            HIPDNN_PLUGIN_LOG_ERROR("engine manager: an engine with id "
                                    << describeEngine(id, name)
                                    << " is already registered; the duplicate is discarded");
        }
    }

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

    void getEngineDetails(THandle& handle,
                          const IGraph& opGraph,
                          int64_t engineId,
                          hipdnnPluginConstData_t& engineDetailsOut)
    {
        auto& engine = getEngine(engineId);
        engine.getDetails(handle, opGraph, engineDetailsOut);
    }

    size_t getMaxWorkspaceSize(const THandle& handle,
                               const IGraph& opGraph,
                               const IEngineConfig& engineConfig) const
    {
        auto& engine = getEngine(engineConfig.engineId());
        return engine.getMaxWorkspaceSize(handle, opGraph, engineConfig);
    }

    /// Builds a plan via the resolved engine and attaches it to @p executionContext.
    void initializeExecutionContext(const THandle& handle,
                                    const IGraph& opGraph,
                                    const IEngineConfig& engineConfig,
                                    TContext& executionContext) const
    {
        auto& engine = getEngine(engineConfig.engineId());
        engine.initializeExecutionContext(handle, opGraph, engineConfig, executionContext);
    }

protected:
    /// @throws HipdnnPluginException if @p engineId is not registered.
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
    /// Renders an engine as its id, suffixed with a name when one can be had. The id is never
    /// dropped: it is what correlates the line with the admission and resolution logs.
    static std::string describeEngine(int64_t id, std::string_view name)
    {
        auto label = hipdnn_data_sdk::utilities::formatEngineIdHex(id);
        if(name.empty())
        {
            const auto& idToName = hipdnn_data_sdk::utilities::getEngineIdToNameMap();
            if(const auto it = idToName.find(id); it != idToName.end())
            {
                name = it->second;
            }
        }
        if(!name.empty())
        {
            label += " (";
            label += name;
            label += ")";
        }
        return label;
    }

    std::unordered_map<int64_t, std::unique_ptr<Engine>> _engines;
};

} // namespace hipdnn_plugin_sdk
