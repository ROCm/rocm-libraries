// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenEngine.hpp"
#include "plans/MiopenBatchnormPlanBuilder.hpp"

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace miopen_plugin
{

namespace
{

auto createBenchmarkingKnob(flatbuffers::FlatBufferBuilder& builder)
{
    return hipdnn_plugin_sdk::KnobFactory::createIntKnob(
        builder, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, "Enable benchmarking", 0, 0, 1, 1, {});
}

void handleBenchmarkingKnobSetting(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnMiopenSettings& executionSettings)
{
    if(!engineConfig.hasKnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME))
    {
        return;
    }

    const auto& knobSetting
        = engineConfig.getKnobSettingByName(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);

    if(knobSetting.valueType() != hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "Benchmarking knob setting value is not an integer. Type: "
                + std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameKnobValue(
                    knobSetting.valueType())));
    }

    auto value = knobSetting.valueAs<hipdnn_flatbuffers_sdk::data_objects::IntValue>().value();
    executionSettings.setBenchmarkingEnabled(value != 0);
}

void initializeMiopenSettings(
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnMiopenSettings& executionSettings)
{
    if(engineConfig.isValid())
    {
        handleBenchmarkingKnobSetting(engineConfig, executionSettings);
    }
    else
    {
        HIPDNN_PLUGIN_LOG_WARN("Engine config is invalid");
    }
}

} // namespace

MiopenEngine::MiopenEngine(int64_t id)
    : _id(id)
{
}

int64_t MiopenEngine::id() const
{
    return _id;
}

bool MiopenEngine::isApplicable(
    HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    // This is wrong if we ever have more than 1 plan builder thats applicable.
    // If this is the case, we should split plan builders accross multiple engines.
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            return true;
        }
    }
    return false;
}

void MiopenEngine::getDetails(HipdnnMiopenHandle& handle,
                              const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                              hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    auto benchmarkingKnob = createBenchmarkingKnob(builder);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Knob>> knobsVector;
    knobsVector.push_back(benchmarkingKnob);

    // Collect custom knobs from plan builders
    for(const auto& planBuilder : _planBuilders)
    {
        auto customKnobs = planBuilder->getCustomKnobs(handle, opGraph);

        if(customKnobs.empty())
        {
            continue;
        }

        for(const auto& knobT : customKnobs)
        {
            auto knobOffset = hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &knobT);
            knobsVector.push_back(knobOffset);
        }

        // Only one plan builder should be applicable for a given graph and return custom knobs.
        // Stop after finding the first one to avoid duplicates.
        break;
    }

    auto knobs = builder.CreateVector(knobsVector);

    auto engineDetails
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, _id, knobs);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t MiopenEngine::getMaxWorkspaceSize(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig) const
{
    HipdnnMiopenSettings baseExecutionSettings;
    initializeMiopenSettings(engineConfig, baseExecutionSettings);

    size_t workspaceSize = 0;

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            HipdnnMiopenSettings executionSettings = baseExecutionSettings;
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            workspaceSize
                = std::max(workspaceSize,
                           planBuilder->getMaxWorkspaceSize(handle, opGraph, executionSettings));
        }
    }

    return workspaceSize;
}

void MiopenEngine::initializeExecutionContext(
    const HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& engineConfig,
    HipdnnMiopenContext& executionContext) const
{
    HipdnnMiopenSettings executionSettings;
    initializeMiopenSettings(engineConfig, executionSettings);

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->initializeExecutionSettings(
                handle, opGraph, engineConfig, executionSettings);
            break;
        }
    }

    executionContext.setExecutionSettings(executionSettings);

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->buildPlan(handle, opGraph, engineConfig, executionContext);
            break;
        }
    }
}

void MiopenEngine::addPlanBuilder(
    std::unique_ptr<hipdnn_plugin_sdk::
                        IPlanBuilder<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>>
        planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

}
