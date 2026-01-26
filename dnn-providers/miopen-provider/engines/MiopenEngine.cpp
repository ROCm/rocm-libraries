// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenEngine.hpp"
#include "plans/MiopenBatchnormPlanBuilder.hpp"

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>

namespace miopen_legacy_plugin
{

MiopenEngine::MiopenEngine(int64_t id)
    : _id(id)
{
}

int64_t MiopenEngine::id() const
{
    return _id;
}

bool MiopenEngine::isApplicable(HipdnnEnginePluginHandle& handle,
                                const hipdnn_plugin_sdk::IGraph& opGraph) const
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

void MiopenEngine::getDetails(HipdnnEnginePluginHandle& handle,
                              hipdnnPluginConstData_t& detailsOut) const
{
    flatbuffers::FlatBufferBuilder builder;

    auto knob = hipdnn_plugin_sdk::KnobFactory::createIntKnob(
        builder, hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, "Enable benchmarking", 0, 0, 1, 1, {});

    std::vector<flatbuffers::Offset<hipdnn_data_sdk::data_objects::Knob>> knobsVector;
    knobsVector.push_back(knob);
    auto knobs = builder.CreateVector(knobsVector);

    auto engineDetails = hipdnn_data_sdk::data_objects::CreateEngineDetails(builder, _id, knobs);
    builder.Finish(engineDetails);
    auto detachedBuffer = std::make_unique<flatbuffers::DetachedBuffer>(builder.Release());
    detailsOut.ptr = detachedBuffer->data();
    detailsOut.size = detachedBuffer->size();

    handle.storeEngineDetailsDetachedBuffer(detailsOut.ptr, std::move(detachedBuffer));
}

size_t MiopenEngine::getWorkspaceSize(const HipdnnEnginePluginHandle& handle,
                                      const hipdnn_plugin_sdk::IGraph& opGraph) const
{
    size_t workspaceSize = 0;
    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            workspaceSize = std::max(workspaceSize, planBuilder->getWorkspaceSize(handle, opGraph));
        }
    }
    return workspaceSize;
}

void MiopenEngine::initializeExecutionContext(
    const HipdnnEnginePluginHandle& handle,
    const hipdnn_plugin_sdk::IGraph& opGraph,
    const hipdnn_plugin_sdk::IEngineConfig& engineConfig,
    HipdnnEnginePluginExecutionContext& executionContext) const
{
    if(engineConfig.isValid())
    {
        if(engineConfig.hasKnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME))
        {
            const auto& knobSetting
                = engineConfig.getKnobSettingByName(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME);
            if(knobSetting.valueType() == hipdnn_data_sdk::data_objects::KnobValue::IntValue)
            {
                auto value = knobSetting.valueAs<hipdnn_data_sdk::data_objects::IntValue>().value();
                executionContext.setBenchmarkingEnabled(value != 0);
            }
            else
            {
                HIPDNN_LOG_WARN(
                    "Benchmarking knob setting value is not an integer. Type: {}",
                    hipdnn_data_sdk::data_objects::EnumNameKnobValue(knobSetting.valueType()));
            }
        }
    }
    else
    {
        HIPDNN_LOG_WARN("Engine config is invalid");
    }

    for(const auto& planBuilder : _planBuilders)
    {
        if(planBuilder->isApplicable(handle, opGraph))
        {
            planBuilder->buildPlan(handle, opGraph, executionContext);
            break;
        }
    }
}

void MiopenEngine::addPlanBuilder(std::unique_ptr<IPlanBuilder> planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

}
