// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "MiopenEngine.hpp"
#include "plans/MiopenBatchnormPlanBuilder.hpp"

#include <algorithm>
#include <string>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/KnobFactory.hpp>
#include <hipdnn_plugin_sdk/NumericalNote.h>
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

    // Applied outside the isValid() branch and after the knob: an unset override leaves
    // the above untouched, =1 forces on even for an invalid config (the plain-execute
    // path), and =0 forces off a knob-enabled run.
    if(const auto forced = hipdnn_plugin_sdk::benchmarkingOverrideFromEnv())
    {
        executionSettings.setBenchmarkingEnabled(*forced);
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

namespace
{

// Bytes per element of a data type; sub-byte types are packed, so their buffers are byte-aligned.
int64_t elementBytes(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(dataType)
    {
    case DataType::DOUBLE:
    case DataType::INT64:
        return 8;
    case DataType::FLOAT:
    case DataType::INT32:
        return 4;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 2;
    default:
        return 1;
    }
}

// MIOpen accepts any pointer aligned to its element size, so the engines take any declared
// alignment of at least that; below it (an address inside an element) they decline. Ragged
// tensors have no MIOpen equivalent.
bool meetsMiopenTensorRules(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph)
{
    for(const auto& [uid, tensor] : opGraph.getTensorMap())
    {
        if(tensor == nullptr || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor))
        {
            continue;
        }
        if(tensor->ragged_offset_tensor_uid().has_value()
           || tensor->ragged_offset_multiplier() != 1)
        {
            return false;
        }
        if(tensor->alignment() < elementBytes(tensor->data_type()))
        {
            return false;
        }
    }
    return true;
}

} // namespace

bool MiopenEngine::isApplicable(
    HipdnnMiopenHandle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    if(!meetsMiopenTensorRules(opGraph))
    {
        return false;
    }
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

    // Determinism is the only numerical claim: MIOPEN_ENGINE_DETERMINISTIC reports an explicit
    // empty note list (the field is present), every other engine is NONDETERMINISTIC.
    const std::vector<int32_t> numericalNotesList
        = _id == hipdnn_data_sdk::utilities::MIOPEN_ENGINE_DETERMINISTIC_ID
              ? std::vector<int32_t>{}
              : std::vector<int32_t>{HIPDNN_NUMERICAL_NOTE_NONDETERMINISTIC};
    auto numericalNotes = builder.CreateVector(numericalNotesList);

    auto engineDetails = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(
        builder, _id, knobs, 0, 0, 0, numericalNotes);
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

    const auto applicableBuilder
        = std::find_if(_planBuilders.begin(), _planBuilders.end(), [&](const auto& planBuilder) {
              return planBuilder->isApplicable(handle, opGraph);
          });
    if(applicableBuilder == _planBuilders.end())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "No MIOpen plan builder is applicable to the graph for engine " + std::to_string(_id));
    }

    (*applicableBuilder)
        ->initializeExecutionSettings(handle, opGraph, engineConfig, executionSettings);
    executionContext.setExecutionSettings(executionSettings);
    (*applicableBuilder)->buildPlan(handle, opGraph, engineConfig, executionContext);
}

void MiopenEngine::addPlanBuilder(
    std::unique_ptr<hipdnn_plugin_sdk::
                        IPlanBuilder<HipdnnMiopenHandle, HipdnnMiopenSettings, HipdnnMiopenContext>>
        planBuilder)
{
    _planBuilders.push_back(std::move(planBuilder));
}

}
