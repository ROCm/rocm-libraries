// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EngineConfigDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "EngineDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnBackendFlatbufferData.h"
#include "HipdnnException.hpp"
#include "KnobSettingDescriptor.hpp"
#include "handle/Handle.hpp"

#include <cmath>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/KnobSettingWrapper.hpp>
#include <limits>
#include <unordered_set>

namespace hipdnn_backend
{

EngineConfigDescriptor::EngineConfigDescriptor()
{
    _engineConfigData = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT>();
}

void EngineConfigDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineConfigDescriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(_engine,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineConfigDescriptor::finalize() failed: Engine is not set.");

    if(!_deferWorkspace)
    {
        ensureWorkspaceSize();
    }
    HipdnnBackendDescriptorImpl<EngineConfigDescriptor>::finalize();
}

void EngineConfigDescriptor::ensureWorkspaceSize() const
{
    std::call_once(_workspaceOnce, [this] {
        const auto graph = _engine->getGraph();
        const auto manager = graph->getHandle()->getPluginResourceManager();
        const auto config = getSerializedEngineConfig();
        const auto workspace
            = manager->getWorkspaceSize(_engine->getEngineId(), &config, graph.get());
        THROW_IF_TRUE(workspace > static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "Engine workspace size exceeds int64");
        _maxWorkspaceSize = static_cast<int64_t>(workspace);
    });
}

void EngineConfigDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                          hipdnnBackendAttributeType_t attributeType,
                                          int64_t requestedElementCount,
                                          int64_t* elementCount,
                                          void* arrayOfElements) const
{
    // A prediction configuration carries constraints only: it is deliberately never
    // finalized, so no engine metadata, catalog, or workspace query is performed.
    THROW_IF_FALSE(isFinalized() || attributeName == HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "EngineConfigDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINECFG_ENGINE:
        getEngine(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
        getMaxWorkspaceSize(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT:
        getPrediction(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineConfigDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

const flatbuffers::DetachedBuffer& EngineConfigDescriptor::ensurePrediction() const
{
    // A lock rather than call_once: this descriptor stays mutable while the prediction is
    // readable, so the cache has to be rebuildable. Two concurrent readers must not both
    // pack and replace it, which would free bytes the other just returned.
    const std::lock_guard<std::mutex> guard(_predictionMutex);
    if(_predictionBuffer.size() == 0)
    {
        THROW_IF_NULL(_engine,
                      HIPDNN_STATUS_BAD_PARAM,
                      "Assign the engine before reading a configuration prediction");
        const auto graph = _engine->getGraph();
        const auto manager = graph->getHandle()->getPluginResourceManager();
        const auto request = getEngineConfigForPrediction(_engine->getEngineId(), graph.get());
        flatbuffers::FlatBufferBuilder configBuilder;
        configBuilder.Finish(
            hipdnn_flatbuffers_sdk::data_objects::EngineConfig::Pack(configBuilder, &request));
        const auto prediction = manager->getEnginePrediction(
            {configBuilder.GetBufferPointer(), configBuilder.GetSize()},
            graph->getSerializedGraph(),
            HIPDNN_ENGINE_PREDICTION_CONFIGURATION,
            _predictionEvaluate);
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(
            hipdnn_flatbuffers_sdk::data_objects::EnginePrediction::Pack(builder, &prediction));
        _predictionBuffer = builder.Release();
    }
    return _predictionBuffer;
}

void EngineConfigDescriptor::invalidatePrediction()
{
    const std::lock_guard<std::mutex> guard(_predictionMutex);
    if(_predictionBuffer.size() != 0)
    {
        // Retired, not freed: a caller may still hold these bytes, which the attribute
        // documents as valid for the descriptor's lifetime.
        _retiredPredictions.push_back(std::move(_predictionBuffer));
    }
    _predictionBuffer = flatbuffers::DetachedBuffer();
}

void EngineConfigDescriptor::getPrediction(hipdnnBackendAttributeType_t attributeType,
                                           int64_t requestedElementCount,
                                           int64_t* elementCount,
                                           void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine config prediction requires flatbuffer data type");
    THROW_IF_TRUE(requestedElementCount < 0 || requestedElementCount > 1,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine config prediction element count must be 0 or 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    if(requestedElementCount == 1)
    {
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "Engine config prediction output is null");
        const auto& buffer = ensurePrediction();
        *static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements)
            = {buffer.data(), buffer.size()};
    }
}

void EngineConfigDescriptor::getEngine(hipdnnBackendAttributeType_t attributeType,
                                       int64_t requestedElementCount,
                                       int64_t* elementCount,
                                       void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to get engine: "
                "Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to get engine: "
                "Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineConfigDescriptor failed to get engine: "
                  "Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    HipdnnBackendDescriptor::packDescriptor(_engine, arrayOfElements);
}

void EngineConfigDescriptor::getMaxWorkspaceSize(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to get max workspace size: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to get max workspace size: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineConfigDescriptor failed to get max workspace size: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    ensureWorkspaceSize();
    *static_cast<int64_t*>(arrayOfElements) = _maxWorkspaceSize;
}

void EngineConfigDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                          hipdnnBackendAttributeType_t attributeType,
                                          int64_t elementCount,
                                          const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "EngineConfigDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINECFG_ENGINE:
        setEngine(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_CHOICE_SERIALIZED_VALUE:
        setKnobChoice(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
        setKnobSettingDescriptor(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_PREDICTION_EVALUATE_EXT:
    {
        THROW_IF_NE(attributeType,
                    HIPDNN_TYPE_INT64,
                    HIPDNN_STATUS_BAD_PARAM,
                    "Prediction evaluate requires int64");
        THROW_IF_NE(
            elementCount, 1, HIPDNN_STATUS_BAD_PARAM, "Prediction evaluate requires one value");
        THROW_IF_NULL(
            arrayOfElements, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "Prediction evaluate is null");
        const auto value = *static_cast<const int64_t*>(arrayOfElements);
        THROW_IF_TRUE(value != 0 && value != 1,
                      HIPDNN_STATUS_BAD_PARAM,
                      "Prediction evaluate must be zero or one");
        _predictionEvaluate = value != 0;
        break;
    }
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineConfigDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }

    // reset the cached buffers when an attribute is set so they cannot go out of date.
    _engineConfigSerializedBuffer = flatbuffers::DetachedBuffer();
    invalidatePrediction();
}

void EngineConfigDescriptor::setEngine(hipdnnBackendAttributeType_t attributeType,
                                       int64_t elementCount,
                                       const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to set engine: "
                "Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to set engine: "
                "Invalid element count.");

    auto engine = HipdnnBackendDescriptor::unpackDescriptor<const EngineDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineConfigDescriptor failed to set engine: Engine is null.");

    THROW_IF_FALSE(engine->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineConfigDescriptor failed to set engine: "
                   "Engine is not finalized.");

    _engine = engine;
    _engineConfigData->engine_id = _engine->getEngineId();
}

std::shared_ptr<const EngineDescriptor> EngineConfigDescriptor::getEngine() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineConfigDescriptor::getEngine() failed: Not finalized.");
    return _engine;
}

hipdnnBackendDescriptorType_t EngineConfigDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR;
}

hipdnnPluginConstData_t EngineConfigDescriptor::getSerializedEngineConfig() const
{
    if(_engineConfigSerializedBuffer.size() == 0)
    {
        THROW_IF_NULL(_engine,
                      HIPDNN_STATUS_INTERNAL_ERROR,
                      "EngineConfigDescriptor::getSerializedEngineConfig: engine is null");

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(hipdnn_flatbuffers_sdk::data_objects::EngineConfig::Pack(
            builder, _engineConfigData.get()));
        _engineConfigSerializedBuffer = builder.Release();
    }

    return {_engineConfigSerializedBuffer.data(), _engineConfigSerializedBuffer.size()};
}

hipdnn_flatbuffers_sdk::data_objects::EngineConfigT
    EngineConfigDescriptor::getEngineConfigForPrediction(int64_t engineId,
                                                         const GraphDescriptor* graph) const
{
    if(_engine != nullptr)
    {
        THROW_IF_TRUE(_engine->getEngineId() != engineId || _engine->getGraph().get() != graph,
                      HIPDNN_STATUS_BAD_PARAM,
                      "Prediction config does not belong to the requested engine and graph");
    }
    auto config = *_engineConfigData;
    config.engine_id = engineId;
    validateEngineConfig(config, engineId);
    return config;
}

void EngineConfigDescriptor::validateEngineConfig(
    const hipdnn_flatbuffers_sdk::data_objects::EngineConfigT& config, int64_t engineId)
{
    THROW_IF_NE(config.engine_id,
                engineId,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine config identity does not match its engine descriptor");
    THROW_IF_TRUE(config.knobs.size() > static_cast<size_t>(MAX_KNOB_CHOICES),
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine config exceeds MAX_KNOB_CHOICES");
    std::unordered_set<std::string> knobIds;
    for(const auto& knob : config.knobs)
    {
        THROW_IF_TRUE(knob == nullptr || knob->knob_id.empty(),
                      HIPDNN_STATUS_BAD_PARAM,
                      "Engine config has a missing knob");
        THROW_IF_FALSE(knobIds.insert(knob->knob_id).second,
                       HIPDNN_STATUS_BAD_PARAM,
                       "Engine config has duplicate knobs");
        using hipdnn_flatbuffers_sdk::data_objects::KnobValue;
        THROW_IF_TRUE(knob->value.value == nullptr
                          || (knob->value.type != KnobValue::IntValue
                              && knob->value.type != KnobValue::FloatValue
                              && knob->value.type != KnobValue::StringValue),
                      HIPDNN_STATUS_BAD_PARAM,
                      "Engine config has an invalid knob value");
        if(const auto* value = knob->value.AsFloatValue())
        {
            THROW_IF_FALSE(std::isfinite(value->value),
                           HIPDNN_STATUS_BAD_PARAM,
                           "Engine config has a non-finite knob value");
        }
    }
}

void EngineConfigDescriptor::setEngineConfig(
    const hipdnn_flatbuffers_sdk::data_objects::EngineConfigT& config, bool deferWorkspace)
{
    THROW_IF_TRUE(
        isFinalized(), HIPDNN_STATUS_NOT_INITIALIZED, "Cannot replace a finalized engine config");
    THROW_IF_NULL(_engine, HIPDNN_STATUS_BAD_PARAM, "Assign the engine before its config");
    validateEngineConfig(config, _engine->getEngineId());
    _engineConfigData
        = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT>(config);
    _engineConfigSerializedBuffer = flatbuffers::DetachedBuffer();
    invalidatePrediction();
    _deferWorkspace = deferWorkspace;
}

void EngineConfigDescriptor::setKnobChoice(hipdnnBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to set knob choice: Invalid attribute type.");

    THROW_IF_LT(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to set knob choice: Element count must be > 0.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineConfigDescriptor failed to set knob choice: Null pointer.");

    auto* inputArray = static_cast<const hipdnnBackendFlatbufferData_t*>(arrayOfElements);

    for(int64_t i = 0; i < elementCount; ++i)
    {
        const auto& flatbufferData = inputArray[i];

        THROW_IF_NULL(flatbufferData.ptr,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineConfigDescriptor failed to set knob choice: "
                      "Flatbuffer data pointer is null.");

        THROW_IF_EQ(flatbufferData.size,
                    0UL,
                    HIPDNN_STATUS_BAD_PARAM,
                    "EngineConfigDescriptor failed to set knob choice: "
                    "Flatbuffer data size must be > 0.");

        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::KnobSettingWrapper wrapper(
            flatbufferData.ptr, flatbufferData.size);

        THROW_IF_FALSE(wrapper.isValid(),
                       HIPDNN_STATUS_BAD_PARAM,
                       "EngineConfigDescriptor failed to set knob choice: "
                       "Invalid knob setting flatbuffer.");

        // Convert to KnobSettingT and add to the engine config data
        auto knobSettingT = wrapper.toKnobSettingT();
        _engineConfigData->knobs.push_back(std::move(knobSettingT));
    }
}

void EngineConfigDescriptor::setKnobSettingDescriptor(hipdnnBackendAttributeType_t attributeType,
                                                      int64_t elementCount,
                                                      const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to set knob choices: Invalid attribute type.");

    THROW_IF_LT(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineConfigDescriptor failed to set knob choices: Element count must be > 0.");

    THROW_IF_TRUE(elementCount > MAX_KNOB_CHOICES,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineConfigDescriptor failed to set knob choices: "
                  "Element count exceeds MAX_KNOB_CHOICES ("
                      + std::to_string(MAX_KNOB_CHOICES) + ").");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineConfigDescriptor failed to set knob choices: Null pointer.");

    auto* descriptorArray = static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements);

    for(int64_t i = 0; i < elementCount; ++i)
    {
        auto knobDesc = HipdnnBackendDescriptor::unpackDescriptor<const KnobSettingDescriptor>(
            descriptorArray[i],
            HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
            "EngineConfigDescriptor failed to set knob choices: "
            "Knob setting descriptor at index "
                + std::to_string(i) + " is null.");

        THROW_IF_FALSE(knobDesc->isFinalized(),
                       HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                       "EngineConfigDescriptor failed to set knob choices: "
                       "Knob setting descriptor at index "
                           + std::to_string(i) + " is not finalized.");

        _engineConfigData->knobs.push_back(knobDesc->toKnobSettingT());
    }
}

std::string EngineConfigDescriptor::toString() const
{
    std::string str = "EngineConfigDescriptor: {engineId=";
    str += _engine ? std::to_string(_engine->getEngineId()) : "null";
    str += ", maxWorkspaceSize=" + std::to_string(_maxWorkspaceSize) + "}";
    return str;
}

} // namespace hipdnn_backend
