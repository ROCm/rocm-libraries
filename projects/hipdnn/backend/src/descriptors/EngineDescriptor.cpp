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

#include <algorithm>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>

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
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper detailsWrapper(
            engineDetailsPtr);
        auto rawBehaviorNotes = detailsWrapper.behaviorNotes();
        std::vector<hipdnnBackendBehaviorNote_t> behaviorNotes;
        behaviorNotes.reserve(rawBehaviorNotes.size());
        for(auto rawNote : rawBehaviorNotes)
        {
            THROW_IF_TRUE(rawNote < 0,
                          HIPDNN_STATUS_BAD_PARAM,
                          "EngineDescriptor::finalize() failed: Invalid behavior note value.");
            behaviorNotes.push_back(rawNote);
        }
        _behaviorNotes = std::move(behaviorNotes);

        auto knobCount = detailsWrapper.knobCount();

        if(knobCount > 0)
        {
            const auto& knobWrappers = detailsWrapper.knobWrappers();
            _knobSerializedBuffers.reserve(knobCount);

            for(const auto& knobWrapper : knobWrappers)
            {
                hipdnn_flatbuffers_sdk::data_objects::KnobT knobNative;
                knobWrapper->getKnob().UnPackTo(&knobNative);

                // Serialize for the flatbuffer-based getAttribute path.
                flatbuffers::FlatBufferBuilder builder;
                auto knobOffset
                    = hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &knobNative);
                builder.Finish(knobOffset);
                _knobSerializedBuffers.push_back(builder.Release());

                // Build KnobDescriptor eagerly so the descriptor is fully
                // immutable after finalize() and safe to share across threads.
                auto knobDesc = KnobDescriptor::fromKnobT(knobNative);
                if(knobDesc)
                {
                    _knobDescriptors.push_back(std::move(knobDesc));
                }
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
    case HIPDNN_ATTR_KNOB_INFO_SERIALIZED_VALUE:
        getKnobInfo(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
        getKnobInfoDescriptors(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
        getBehaviorNotes(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_CU_COUNT_TARGET_EXT:
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
    case HIPDNN_ATTR_ENGINE_CU_COUNT_TARGET_EXT:
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

    HipdnnBackendDescriptor::packDescriptorArray(
        _knobDescriptors, static_cast<HipdnnBackendDescriptor**>(arrayOfElements));
}

void EngineDescriptor::getBehaviorNotes(hipdnnBackendAttributeType_t attributeType,
                                        int64_t requestedElementCount,
                                        int64_t* elementCount,
                                        void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BEHAVIOR_NOTE,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to get behavior notes: Invalid attribute type.");

    auto count = static_cast<int64_t>(_behaviorNotes.size());
    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineDescriptor failed to get behavior notes: elementCount is null.");
        *elementCount = count;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= count,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineDescriptor failed to get behavior notes: requested element count is "
                   "too small.");

    if(elementCount != nullptr)
    {
        *elementCount = count;
    }

    std::copy(_behaviorNotes.begin(),
              _behaviorNotes.end(),
              static_cast<hipdnnBackendBehaviorNote_t*>(arrayOfElements));
}

std::string EngineDescriptor::toString() const
{
    std::string str = "EngineDescriptor: {engineId=";
    str += _engineIdSet ? std::to_string(_engineId) : "unset";
    str += _graph ? ", graph=" + fmt::format("{:p}", static_cast<const void*>(_graph.get()))
                  : ", graph=null";
    str += '}';
    return str;
}

} // namespace hipdnn_backend
