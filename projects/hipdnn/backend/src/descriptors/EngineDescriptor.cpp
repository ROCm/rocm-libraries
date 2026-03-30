// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EngineDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnBackendFlatbufferData.h"
#include "HipdnnException.hpp"
#include "KnobDescriptor.hpp"
#include "handle/Handle.hpp"
#include "logging/Logging.hpp"
#include "plugin/EnginePluginResourceManager.hpp"

#include <hipdnn_data_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>

namespace hipdnn_backend
{

void EngineDescriptor::finalize()
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineDescriptor::finalize() failed: Already finalized.");

    THROW_IF_NULL(
        _graph, HIPDNN_STATUS_BAD_PARAM, "EngineDescriptor::finalize() failed: Graph is not set.");

    THROW_IF_FALSE(_engineIdSet,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineDescriptor::finalize() failed: Engine id is not set.");

    auto handle = _graph->getHandle();
    auto pluginResourceManager = handle->getPluginResourceManager();

    auto engineIds = pluginResourceManager->getApplicableEngineIds(_graph.get());
    if(std::find(engineIds.begin(), engineIds.end(), _engineId) == engineIds.end())
    {
        throw HipdnnException(HIPDNN_STATUS_BAD_PARAM,
                              "EngineDescriptor::finalize() failed: Engine id is not in a valid "
                              "range of engine IDs");
    }

    _engineDetails = plugin::EnginePluginResourceManager::getEngineDetails(
        pluginResourceManager, _engineId, _graph.get());

    auto engineDetailsPtr = _engineDetails->get();
    if(engineDetailsPtr != nullptr)
    {
        const hipdnn_data_sdk::flatbuffer_utilities::EngineDetailsWrapper detailsWrapper(
            engineDetailsPtr);
        auto knobCount = detailsWrapper.knobCount();

        if(knobCount > 0)
        {
            const auto& knobWrappers = detailsWrapper.knobWrappers();
            _knobSerializedBuffers.reserve(knobCount);

            for(const auto& knobWrapper : knobWrappers)
            {
                flatbuffers::FlatBufferBuilder builder;
                hipdnn_data_sdk::data_objects::KnobT knobNative;
                knobWrapper->getKnob().UnPackTo(&knobNative);
                auto knobOffset = hipdnn_data_sdk::data_objects::Knob::Pack(builder, &knobNative);
                builder.Finish(knobOffset);
                _knobSerializedBuffers.push_back(builder.Release());
            }
        }
    }

    HipdnnBackendDescriptorImpl<EngineDescriptor>::finalize();
}

void EngineDescriptor::getAttribute(hipdnnBackendAttributeName_t attributeName,
                                    hipdnnBackendAttributeType_t attributeType,
                                    int64_t requestedElementCount,
                                    int64_t* elementCount,
                                    void* arrayOfElements) const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_NOT_INITIALIZED,
                   "EngineDescriptor::getAttribute() failed: Not finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINE_OPERATION_GRAPH:
        getGraph(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_GLOBAL_INDEX:
        getGlobalId(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE_EXT:
        getKnobInfo(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
        getKnobInfoDescriptors(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET:
    case HIPDNN_ATTR_ENGINE_DEVICEPROP:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineDescriptor::getAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineDescriptor::getGraph(hipdnnBackendAttributeType_t attributeType,
                                int64_t requestedElementCount,
                                int64_t* elementCount,
                                void* arrayOfElements) const
{

    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get graph: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get graph: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineDescriptor failed to get graph: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    HipdnnBackendDescriptor::packDescriptor(_graph, arrayOfElements);
}

void EngineDescriptor::getGlobalId(hipdnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get global engine ID: Invalid attribute type.");

    THROW_IF_NE(requestedElementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get global engine ID: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineDescriptor failed to get global engine ID: Null pointer.");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }

    *static_cast<int64_t*>(arrayOfElements) = _engineId;
}

void EngineDescriptor::setAttribute(hipdnnBackendAttributeName_t attributeName,
                                    hipdnnBackendAttributeType_t attributeType,
                                    int64_t elementCount,
                                    const void* arrayOfElements)
{
    THROW_IF_TRUE(isFinalized(),
                  HIPDNN_STATUS_NOT_INITIALIZED,
                  "EngineDescriptor::setAttribute() failed: Already finalized.");

    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINE_OPERATION_GRAPH:
        setGraph(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_GLOBAL_INDEX:
        setGlobalId(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case HIPDNN_ATTR_ENGINE_SM_COUNT_TARGET:
    case HIPDNN_ATTR_ENGINE_DEVICEPROP:
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineDescriptor::setGraph(hipdnnBackendAttributeType_t attributeType,
                                int64_t elementCount,
                                const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to set graph: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to set graph: Invalid element count.");

    auto graph = HipdnnBackendDescriptor::unpackDescriptor<const GraphDescriptor>(
        arrayOfElements,
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
        "EngineDescriptor failed to set graph: Graph is null.");

    THROW_IF_FALSE(graph->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "EngineDescriptor failed to set graph: Graph is not finalized.");

    _graph = graph;
}

void EngineDescriptor::setGlobalId(hipdnnBackendAttributeType_t attributeType,
                                   int64_t elementCount,
                                   const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine failed to set engine id: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine failed to set engine id: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "Engine failed to set engine id: Null pointer.");

    _engineId = *static_cast<const int64_t*>(arrayOfElements);
    _engineIdSet = true;
}

std::shared_ptr<const GraphDescriptor> EngineDescriptor::getGraph() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineDescriptor::getGraph() failed: Not finalized.");

    return _graph;
}

int64_t EngineDescriptor::getEngineId() const
{
    THROW_IF_FALSE(isFinalized(),
                   HIPDNN_STATUS_INTERNAL_ERROR,
                   "EngineDescriptor::getEngineId() failed: Not finalized.");

    return _engineId;
}

hipdnnBackendDescriptorType_t EngineDescriptor::getStaticType()
{
    return HIPDNN_BACKEND_ENGINE_DESCRIPTOR;
}

void EngineDescriptor::getKnobInfo(hipdnnBackendAttributeType_t attributeType,
                                   int64_t requestedElementCount,
                                   int64_t* elementCount,
                                   void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get knob info: Invalid attribute type.");

    auto knobCount = static_cast<int64_t>(_knobSerializedBuffers.size());

    // If requestedElementCount is 0, just return the count
    if(requestedElementCount == 0)
    {
        if(elementCount != nullptr)
        {
            *elementCount = knobCount;
        }
        return;
    }

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineDescriptor failed to get knob info: Null pointer.");

    // Fill the output array with hipdnnBackendFlatbufferData_t structs
    auto* outputArray = static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements);
    auto elementsToReturn = std::min(requestedElementCount, knobCount);

    for(int64_t i = 0; i < elementsToReturn; ++i)
    {
        outputArray[i].ptr = _knobSerializedBuffers[static_cast<size_t>(i)].data();
        outputArray[i].size = _knobSerializedBuffers[static_cast<size_t>(i)].size();
    }

    if(elementCount != nullptr)
    {
        *elementCount = elementsToReturn;
    }
}

void EngineDescriptor::getKnobInfoDescriptors(hipdnnBackendAttributeType_t attributeType,
                                              int64_t requestedElementCount,
                                              int64_t* elementCount,
                                              void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                 attributeType,
                 "EngineDescriptor::getAttribute(HIPDNN_ATTR_ENGINE_KNOB_INFO)");

    // Lazily build KnobDescriptor objects from the serialized knob buffers.
    if(_knobDescriptors.empty() && !_knobSerializedBuffers.empty())
    {
        for(const auto& buffer : _knobSerializedBuffers)
        {
            auto knobFb = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Knob>(buffer.data());
            if(knobFb == nullptr)
            {
                continue;
            }
            hipdnn_data_sdk::data_objects::KnobT knobNative;
            knobFb->UnPackTo(&knobNative);
            auto knobDesc = std::make_shared<KnobDescriptor>();

            // Set knob ID
            knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_TYPE_EXT,
                                   HIPDNN_TYPE_CHAR,
                                   static_cast<int64_t>(knobNative.knob_id.size()),
                                   knobNative.knob_id.c_str());

            // Set description
            if(!knobNative.description.empty())
            {
                knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_DESCRIPTION_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(knobNative.description.size()),
                                       knobNative.description.c_str());
            }

            // Set deprecated flag
            knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_DEPRECATED_EXT,
                                   HIPDNN_TYPE_BOOLEAN,
                                   1,
                                   &knobNative.deprecated);

            // Set default value and matching constraint fields based on type
            switch(knobNative.default_value.type)
            {
            case hipdnn_data_sdk::data_objects::KnobValue::IntValue:
            {
                auto val = knobNative.default_value.AsIntValue()->value;
                knobDesc->setAttribute(
                    HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT, HIPDNN_TYPE_INT64, 1, &val);

                if(knobNative.constraint.type
                   == hipdnn_data_sdk::data_objects::KnobConstraint::IntConstraint)
                {
                    const auto* c = knobNative.constraint.AsIntConstraint();
                    // Only set range bounds when non-zero: {0,0} is the plugin SDK's
                    // sentinel meaning "no range constraint" (only valid_values applies).
                    if(c->min_value != 0 || c->max_value != 0)
                    {
                        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT,
                                               HIPDNN_TYPE_INT64,
                                               1,
                                               &c->min_value);
                        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT,
                                               HIPDNN_TYPE_INT64,
                                               1,
                                               &c->max_value);
                    }
                    if(c->step > 0)
                    {
                        knobDesc->setAttribute(
                            HIPDNN_ATTR_KNOB_INFO_STRIDE_EXT, HIPDNN_TYPE_INT64, 1, &c->step);
                    }
                    if(!c->valid_values.empty())
                    {
                        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_INT_EXT,
                                               HIPDNN_TYPE_INT64,
                                               static_cast<int64_t>(c->valid_values.size()),
                                               c->valid_values.data());
                    }
                }
                break;
            }
            case hipdnn_data_sdk::data_objects::KnobValue::FloatValue:
            {
                auto val = knobNative.default_value.AsFloatValue()->value;
                knobDesc->setAttribute(
                    HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT, HIPDNN_TYPE_DOUBLE, 1, &val);

                if(knobNative.constraint.type
                   == hipdnn_data_sdk::data_objects::KnobConstraint::FloatConstraint)
                {
                    const auto* c = knobNative.constraint.AsFloatConstraint();
                    // Only set range bounds when non-zero: {0.0,0.0} is the plugin SDK's
                    // sentinel meaning "no range constraint".
                    if(c->min_value != 0.0 || c->max_value != 0.0)
                    {
                        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_MINIMUM_VALUE_EXT,
                                               HIPDNN_TYPE_DOUBLE,
                                               1,
                                               &c->min_value);
                        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE_EXT,
                                               HIPDNN_TYPE_DOUBLE,
                                               1,
                                               &c->max_value);
                    }
                }
                break;
            }
            case hipdnn_data_sdk::data_objects::KnobValue::StringValue:
            {
                const auto& val = knobNative.default_value.AsStringValue()->value;
                knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_DEFAULT_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(val.size()),
                                       val.c_str());

                if(knobNative.constraint.type
                   == hipdnn_data_sdk::data_objects::KnobConstraint::StringConstraint)
                {
                    const auto* c = knobNative.constraint.AsStringConstraint();
                    if(c->max_length > 0)
                    {
                        auto maxLen = static_cast<int32_t>(c->max_length);
                        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_STRING_MAX_LENGTH_EXT,
                                               HIPDNN_TYPE_INT32,
                                               1,
                                               &maxLen);
                    }
                    if(!c->valid_values.empty())
                    {
                        // Build null-separated buffer: "str1\0str2\0str3\0"
                        std::string buf;
                        for(const auto& s : c->valid_values)
                        {
                            buf.append(s);
                            buf.push_back('\0');
                        }
                        knobDesc->setAttribute(HIPDNN_ATTR_KNOB_INFO_VALID_VALUES_STRING_EXT,
                                               HIPDNN_TYPE_CHAR,
                                               static_cast<int64_t>(buf.size()),
                                               buf.data());
                    }
                }
                break;
            }
            default:
                HIPDNN_BACKEND_LOG_WARN(
                    "EngineDescriptor::getKnobInfoDescriptors: skipping knob '{}' "
                    "with unknown default value type {}",
                    knobNative.knob_id,
                    static_cast<int>(knobNative.default_value.type));
                continue;
            }

            knobDesc->finalize();
            _knobDescriptors.push_back(std::move(knobDesc));
        }
    }

    auto count = static_cast<int64_t>(_knobDescriptors.size());

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineDescriptor::getAttribute(HIPDNN_ATTR_ENGINE_KNOB_INFO): "
                      "elementCount is null");
        *elementCount = count;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= count,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineDescriptor::getAttribute(HIPDNN_ATTR_ENGINE_KNOB_INFO): "
                   "requestedElementCount < knob count");

    if(elementCount != nullptr)
    {
        *elementCount = count;
    }

    auto outputArray = static_cast<HipdnnBackendDescriptor**>(arrayOfElements);

    std::vector<HipdnnBackendDescriptor*> packed;
    packed.reserve(_knobDescriptors.size());
    try
    {
        for(const auto& knobDesc : _knobDescriptors)
        {
            packed.push_back(HipdnnBackendDescriptor::packDescriptor(knobDesc));
        }
    }
    catch(...)
    {
        for(auto* p : packed)
        {
            delete p;
        }
        throw;
    }

    for(size_t i = 0; i < packed.size(); ++i)
    {
        outputArray[i] = packed[i];
    }
}

std::string EngineDescriptor::toString() const
{
    std::string str = "EngineDescriptor: {engineId=";
    str += _engineIdSet ? std::to_string(_engineId) : "unset";
    str += _graph ? ", graph=" + fmt::format("{:p}", static_cast<const void*>(_graph.get()))
                  : ", graph=null";
    str += "}";
    return str;
}

} // namespace hipdnn_backend
