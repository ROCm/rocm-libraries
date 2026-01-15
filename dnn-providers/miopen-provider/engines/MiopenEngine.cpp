// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenEngine.hpp"
#include "plans/MiopenBatchnormPlanBuilder.hpp"

#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>

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

    auto knobIdStr = builder.CreateString("benchmarking");
    auto description = builder.CreateString("Enable benchmarking");
    auto defaultValue
        = hipdnn_data_sdk::data_objects::CreateIntValue(builder, static_cast<int64_t>(0));
    auto constraint = hipdnn_data_sdk::data_objects::CreateIntConstraint(builder, 0, 1, 1);

    auto knob = hipdnn_data_sdk::data_objects::CreateKnob(
        builder,
        static_cast<int64_t>(hipdnn_data_sdk::utilities::fnv1aHash("benchmarking")),
        knobIdStr,
        description,
        hipdnn_data_sdk::data_objects::KnobValue::IntValue,
        defaultValue.Union(),
        hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint,
        constraint.Union(),
        false);

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
        auto& config = engineConfig.getEngineConfig();
        if(config.knobs() != nullptr)
        {
            auto benchmarkingId = hipdnn_data_sdk::utilities::fnv1aHash("benchmarking");
            for(const auto* knobSetting : *config.knobs())
            {
                if(knobSetting->knob_id() == static_cast<int64_t>(benchmarkingId))
                {
                    if(knobSetting->value_type()
                       == hipdnn_data_sdk::data_objects::KnobValue::IntValue)
                    {
                        auto value = knobSetting->value_as_IntValue()->value();
                        executionContext.benchmarkingEnabled = (value != 0);
                    }
                }
            }
        }
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
