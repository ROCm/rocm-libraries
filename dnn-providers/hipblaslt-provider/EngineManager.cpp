// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>

#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "EngineManager.hpp"
#include "engines/PlanBuilderEngine.hpp"

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

namespace hipblaslt_plugin
{

void EngineManager::addEngine(std::unique_ptr<IEngine> engine)
{
    _engines.emplace(engine->id(), std::move(engine));
}

std::vector<int64_t> EngineManager::getApplicableEngineIds(
    HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    std::vector<int64_t> applicable;
    for(const auto& engine : _engines)
    {
        if(engine.second->isApplicable(handle, opGraph))
        {
            applicable.push_back(engine.second->id());
        }
    }
    return applicable;
}

void EngineManager::getEngineDetails(
    HipdnnEnginePluginHandle& handle,
    [[maybe_unused]] const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    int64_t engineId,
    hipdnnPluginConstData_t& engineDetailsOut)
{
    auto& engine = getEngine(engineId);
    engine.getDetails(handle, engineDetailsOut);
}

size_t EngineManager::getWorkspaceSize(
    const HipdnnEnginePluginHandle& handle,
    int64_t engineId,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    auto& engine = getEngine(engineId);
    return engine.getWorkspaceSize(handle, opGraph);
}

namespace
{

/// The engine configuration's `global.benchmarking` knob: absent or an invalid configuration
/// reads as off; a non-integer setting is an error, as for the kernel ingestor's engines.
bool benchmarkingKnob(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig)
{
    using hipdnn_flatbuffers_sdk::data_objects::IntValue;
    using hipdnn_flatbuffers_sdk::data_objects::KnobValue;
    if(!engineConfig.isValid()
       || !engineConfig.hasKnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME))
    {
        return false;
    }
    const auto& setting
        = engineConfig.getKnobSettingByName(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);
    if(setting.valueType() != KnobValue::IntValue)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                    std::string("knob '")
                                        + hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME
                                        + "' must be set to an integer value");
    }
    return setting.valueAs<IntValue>().value() != 0;
}

} // namespace

void EngineManager::initializeExecutionContext(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    auto& engine = getEngine(engineConfig.engineId());
    executionContext.setBenchmarking(
        hipdnn_plugin_sdk::benchmarkingOverrideFromEnv().value_or(benchmarkingKnob(engineConfig)));
    engine.initializeExecutionContext(handle, opGraph, executionContext);
}

IEngine& EngineManager::getEngine(int64_t engineId) const
{
    auto it = _engines.find(engineId);
    if(it == _engines.end())
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                    "Engine with ID " + std::to_string(engineId) + " not found.");
    }
    return *it->second;
}

} // namespace hipblaslt_plugin
