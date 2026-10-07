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

#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/KnobSettingWrapper.hpp>

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

    auto graph = _engine->getGraph();
    auto handle = graph->getHandle();
    auto pluginResourceManager = handle->getPluginResourceManager();

    auto engineId = _engine->getEngineId();

    auto engineConfigPluginData = getSerializedEngineConfig();
    auto workspaceSize = static_cast<int64_t>(
        pluginResourceManager->getWorkspaceSize(engineId, &engineConfigPluginData, graph.get()));

    THROW_IF_LT(workspaceSize,
                0,
                HIPDNN_STATUS_INTERNAL_ERROR,
                "EngineConfigDescriptor::setMaxWorkspaceSize() failed: "
                "Max workspace size cannot be negative.");

    _maxWorkspaceSize = workspaceSize;
    HipdnnBackendDescriptorImpl<EngineConfigDescriptor>::finalize();
}

void EngineConfigDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                          hipdnnBackendAttributeType_t attributeType,
                                          int64_t requestedElementCount,
                                          int64_t* elementCount,
                                          void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
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
    case HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES:
        getKnobSettingDescriptors(
            attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineConfigDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
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
    case HIPDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO:
    case HIPDNN_ATTR_ENGINECFG_WORKSPACE_SIZE:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineConfigDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }

    // reset the serialized buffer when an attribute is set to ensure it's not cached out of date.
    _engineConfigSerializedBuffer = flatbuffers::DetachedBuffer();
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

std::shared_ptr<EngineConfigDescriptor> EngineConfigDescriptor::fromEngineConfigT(
    const std::shared_ptr<const EngineDescriptor>& engine,
    const hipdnn_flatbuffers_sdk::data_objects::EngineConfigT& entry)
{
    THROW_IF_NULL(engine,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "EngineConfigDescriptor::fromEngineConfigT() failed: Engine is null.");

    auto config = std::make_shared<EngineConfigDescriptor>();
    config->_engine = engine;
    config->_engineConfigData
        = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT>(entry);
    config->_engineConfigData->engine_id = engine->getEngineId();
    config->finalize();
    return config;
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

void EngineConfigDescriptor::getKnobSettingDescriptors(hipdnnBackendAttributeType_t attributeType,
                                                       int64_t requestedElementCount,
                                                       int64_t* elementCount,
                                                       void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                 attributeType,
                 "EngineConfigDescriptor::getAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES)");

    const auto count = static_cast<int64_t>(_engineConfigData->knobs.size());

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineConfigDescriptor::getAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES): "
                      "elementCount is null");
        *elementCount = count;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= count,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineConfigDescriptor::getAttribute(HIPDNN_ATTR_ENGINECFG_KNOB_CHOICES): "
                   "requestedElementCount < knob choice count");

    std::vector<std::shared_ptr<KnobSettingDescriptor>> knobDescriptors;
    knobDescriptors.reserve(_engineConfigData->knobs.size());
    for(const auto& knob : _engineConfigData->knobs)
    {
        knobDescriptors.push_back(KnobSettingDescriptor::fromKnobSettingT(*knob));
    }

    if(elementCount != nullptr)
    {
        *elementCount = count;
    }

    HipdnnBackendDescriptor::packDescriptorArray(
        knobDescriptors, static_cast<HipdnnBackendDescriptor**>(arrayOfElements));
}

std::string EngineConfigDescriptor::toString() const
{
    std::string str = "EngineConfigDescriptor: {engineId=";
    str += _engine ? std::to_string(_engine->getEngineId()) : "null";
    str += ", maxWorkspaceSize=" + std::to_string(_maxWorkspaceSize) + "}";
    return str;
}

} // namespace hipdnn_backend
