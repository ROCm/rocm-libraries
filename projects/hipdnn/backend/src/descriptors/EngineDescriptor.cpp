// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "EngineDescriptor.hpp"
#include "BackendEnumStringUtils.hpp"
#include "DescriptorAttributeUtils.hpp"
#include "EngineConfigDescriptor.hpp"
#include "GraphDescriptor.hpp"
#include "HipdnnBackendDescriptorType.h"
#include "HipdnnBackendFlatbufferData.h"
#include "HipdnnException.hpp"
#include "KnobDescriptor.hpp"
#include "KnobSettingDescriptor.hpp"
#include "handle/Handle.hpp"
#include "logging/Logging.hpp"
#include "plugin/EnginePluginResourceManager.hpp"

#include <algorithm>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <optional>
#include <string>
#include <string_view>

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

    ensureDetailsLoaded(pluginResourceManager);

    HipdnnBackendDescriptorImpl<EngineDescriptor>::finalize();
}

void EngineDescriptor::initializeHeuristicResult(std::shared_ptr<const GraphDescriptor> graph,
                                                 int64_t engineId)
{
    THROW_IF_TRUE(isFinalized() || _engineIdSet || _graph != nullptr,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Heuristic engine descriptor must be newly created");
    THROW_IF_NULL(graph, HIPDNN_STATUS_BAD_PARAM, "Heuristic engine graph is not set");
    THROW_IF_FALSE(graph->isFinalized(),
                   HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                   "Heuristic engine requires a finalized graph");
    _graph = std::move(graph);
    _engineId = engineId;
    _engineIdSet = true;
    HipdnnBackendDescriptorImpl<EngineDescriptor>::finalize();
}

void EngineDescriptor::ensureDetailsLoaded(
    const std::shared_ptr<plugin::EnginePluginResourceManager>& manager) const
{
    std::call_once(_detailsOnce, [this, &manager]() {
        auto resourceManager = manager ? manager : _graph->getHandle()->getPluginResourceManager();
        auto details = plugin::EnginePluginResourceManager::getEngineDetails(
            resourceManager, _engineId, _graph.get());
        std::optional<std::string> detailsName;
        std::vector<hipdnnBackendBehaviorNote_t> behaviorNotes;
        std::vector<flatbuffers::DetachedBuffer> knobBuffers;
        std::vector<std::shared_ptr<KnobDescriptor>> knobDescriptors;
        if(const auto* engineDetails = details->get())
        {
            const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper wrapper(
                engineDetails);
            detailsName = wrapper.name();
            const auto rawNotes = wrapper.behaviorNotes();
            behaviorNotes.reserve(rawNotes.size());
            for(auto note : rawNotes)
            {
                THROW_IF_TRUE(note < 0,
                              HIPDNN_STATUS_BAD_PARAM,
                              "EngineDescriptor received an invalid behavior note value");
                behaviorNotes.push_back(note);
            }

            const auto knobCount = wrapper.knobCount();
            knobBuffers.reserve(knobCount);
            knobDescriptors.reserve(knobCount);
            for(const auto& knob : wrapper.knobWrappers())
            {
                hipdnn_flatbuffers_sdk::data_objects::KnobT native;
                knob->getKnob().UnPackTo(&native);
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(hipdnn_flatbuffers_sdk::data_objects::Knob::Pack(builder, &native));
                knobBuffers.push_back(builder.Release());
                if(auto descriptor = KnobDescriptor::fromKnobT(native))
                {
                    knobDescriptors.push_back(std::move(descriptor));
                }
            }
        }
        auto name = resourceManager->resolveEngineName(
            _engineId, detailsName ? std::optional<std::string_view>(*detailsName) : std::nullopt);

        // Publish only after every conversion succeeds. call_once retries exceptions
        // without exposing partial vectors or duplicating entries on the next query.
        _engineDetails = std::move(details);
        _behaviorNotes = std::move(behaviorNotes);
        _knobSerializedBuffers = std::move(knobBuffers);
        _knobDescriptors = std::move(knobDescriptors);
        _engineName = std::move(name);
        _detailsLoaded.store(true, std::memory_order_release);
    });
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
    case HIPDNN_ATTR_ENGINE_NAME_EXT:
        ensureDetailsLoaded();
        getString(_engineName,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "EngineDescriptor::getAttribute()");
        break;
    case HIPDNN_ATTR_ENGINE_CANDIDATES_EXT:
        getCandidates(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_PREDICTION_EXT:
        getPrediction(attributeType, requestedElementCount, elementCount, arrayOfElements);
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
    case HIPDNN_ATTR_ENGINE_PREDICTION_EVALUATE_EXT:
    case HIPDNN_ATTR_ENGINE_CANDIDATE_OFFSET_EXT:
    case HIPDNN_ATTR_ENGINE_CANDIDATE_LIMIT_EXT:
        setInspectionScalar(attributeName, attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_CANDIDATE_SCOPE_EXT:
        setCandidateScope(attributeType, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_KNOB_INFO:
    case HIPDNN_ATTR_ENGINE_NUMERICAL_NOTE:
    case HIPDNN_ATTR_ENGINE_LAYOUT_INFO:
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
    case HIPDNN_ATTR_ENGINE_CU_COUNT_TARGET_EXT:
    case HIPDNN_ATTR_ENGINE_DEVICEPROP:
    case HIPDNN_ATTR_ENGINE_NAME_EXT:
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
    ensureDetailsLoaded();

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
    ensureDetailsLoaded();

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
    ensureDetailsLoaded();

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

void EngineDescriptor::setInspectionScalar(hipdnnBackendAttributeName_t attributeName,
                                           hipdnnBackendAttributeType_t attributeType,
                                           int64_t elementCount,
                                           const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_INT64,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to set an inspection input: Invalid attribute type.");

    THROW_IF_NE(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to set an inspection input: Invalid element count.");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineDescriptor failed to set an inspection input: Null pointer.");

    const auto value = *static_cast<const int64_t*>(arrayOfElements);
    switch(attributeName)
    {
    case HIPDNN_ATTR_ENGINE_PREDICTION_EVALUATE_EXT:
        THROW_IF_TRUE(value != 0 && value != 1,
                      HIPDNN_STATUS_BAD_PARAM,
                      "Prediction evaluate must be zero or one");
        _predictionEvaluate = value != 0;
        break;
    case HIPDNN_ATTR_ENGINE_CANDIDATE_OFFSET_EXT:
        THROW_IF_TRUE(value < 0, HIPDNN_STATUS_BAD_PARAM, "Negative candidate offset");
        _candidateOffset = value;
        break;
    case HIPDNN_ATTR_ENGINE_CANDIDATE_LIMIT_EXT:
        THROW_IF_TRUE(value < 1 || value > MAX_CANDIDATE_LIMIT,
                      HIPDNN_STATUS_BAD_PARAM,
                      "Candidate page limit must be in [1, " + std::to_string(MAX_CANDIDATE_LIMIT)
                          + "]");
        _candidateLimit = value;
        break;
    default:
        throw HipdnnException(
            HIPDNN_STATUS_NOT_SUPPORTED,
            std::string("EngineDescriptor::setAttribute() is not supported for attribute ")
                + hipdnn_backend::hipdnnGetAttributeNameString(attributeName) + ".");
    }
}

void EngineDescriptor::setCandidateScope(hipdnnBackendAttributeType_t attributeType,
                                         int64_t elementCount,
                                         const void* arrayOfElements)
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to set the candidate scope: Invalid attribute type.");

    THROW_IF_LT(elementCount,
                1,
                HIPDNN_STATUS_BAD_PARAM,
                "EngineDescriptor failed to set the candidate scope: Element count must be > 0.");

    THROW_IF_TRUE(elementCount > EngineConfigDescriptor::MAX_KNOB_CHOICES,
                  HIPDNN_STATUS_BAD_PARAM,
                  "EngineDescriptor failed to set the candidate scope: Element count exceeds "
                  "MAX_KNOB_CHOICES ("
                      + std::to_string(EngineConfigDescriptor::MAX_KNOB_CHOICES) + ").");

    THROW_IF_NULL(arrayOfElements,
                  HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                  "EngineDescriptor failed to set the candidate scope: Null pointer.");

    const auto* descriptorArray = static_cast<HipdnnBackendDescriptor* const*>(arrayOfElements);
    std::vector<std::shared_ptr<const KnobSettingDescriptor>> scope;
    scope.reserve(static_cast<size_t>(elementCount));
    for(int64_t i = 0; i < elementCount; ++i)
    {
        auto knobDesc = HipdnnBackendDescriptor::unpackDescriptor<const KnobSettingDescriptor>(
            descriptorArray[i],
            HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
            "EngineDescriptor failed to set the candidate scope: Knob setting descriptor at index "
                + std::to_string(i) + " is null.");

        THROW_IF_FALSE(knobDesc->isFinalized(),
                       HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED,
                       "EngineDescriptor failed to set the candidate scope: Knob setting "
                       "descriptor at index "
                           + std::to_string(i) + " is not finalized.");

        scope.push_back(std::move(knobDesc));
    }
    _candidateScope = std::move(scope);
}

const std::vector<uint8_t>& EngineDescriptor::ensureCandidates() const
{
    std::call_once(_candidatesOnce, [this] {
        hipdnn_flatbuffers_sdk::data_objects::EngineConfigT scope;
        scope.engine_id = _engineId;
        scope.knobs.reserve(_candidateScope.size());
        for(const auto& knob : _candidateScope)
        {
            scope.knobs.push_back(knob->toKnobSettingT());
        }
        EngineConfigDescriptor::validateEngineConfig(scope, _engineId);

        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(hipdnn_flatbuffers_sdk::data_objects::EngineConfig::Pack(builder, &scope));
        const auto manager = _graph->getHandle()->getPluginResourceManager();
        _candidatePage
            = manager->enumerateCandidates(_engineId,
                                           {builder.GetBufferPointer(), builder.GetSize()},
                                           _graph.get(),
                                           static_cast<uint64_t>(_candidateOffset),
                                           static_cast<uint64_t>(_candidateLimit));
    });
    return _candidatePage;
}

void EngineDescriptor::getCandidates(hipdnnBackendAttributeType_t attributeType,
                                     int64_t requestedElementCount,
                                     int64_t* elementCount,
                                     void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                HIPDNN_STATUS_BAD_PARAM,
                "Candidate page requires flatbuffer data type");
    THROW_IF_TRUE(requestedElementCount < 0 || requestedElementCount > 1,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Candidate page element count must be 0 or 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    if(requestedElementCount == 1)
    {
        THROW_IF_NULL(
            arrayOfElements, HIPDNN_STATUS_BAD_PARAM_NULL_POINTER, "Candidate page output is null");
        const auto& page = ensureCandidates();
        *static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements) = {page.data(), page.size()};
    }
}

const flatbuffers::DetachedBuffer& EngineDescriptor::ensurePrediction() const
{
    std::call_once(_predictionOnce, [this] {
        hipdnn_flatbuffers_sdk::data_objects::EngineConfigT request;
        request.engine_id = _engineId;
        flatbuffers::FlatBufferBuilder configBuilder;
        configBuilder.Finish(
            hipdnn_flatbuffers_sdk::data_objects::EngineConfig::Pack(configBuilder, &request));
        const auto manager = _graph->getHandle()->getPluginResourceManager();
        const auto prediction = manager->getEnginePrediction(
            {configBuilder.GetBufferPointer(), configBuilder.GetSize()},
            _graph->getSerializedGraph(),
            HIPDNN_ENGINE_PREDICTION_ENGINE,
            _predictionEvaluate);
        flatbuffers::FlatBufferBuilder builder;
        builder.Finish(
            hipdnn_flatbuffers_sdk::data_objects::EnginePrediction::Pack(builder, &prediction));
        _prediction = builder.Release();
    });
    return _prediction;
}

void EngineDescriptor::getPrediction(hipdnnBackendAttributeType_t attributeType,
                                     int64_t requestedElementCount,
                                     int64_t* elementCount,
                                     void* arrayOfElements) const
{
    THROW_IF_NE(attributeType,
                HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine prediction requires flatbuffer data type");
    THROW_IF_TRUE(requestedElementCount < 0 || requestedElementCount > 1,
                  HIPDNN_STATUS_BAD_PARAM,
                  "Engine prediction element count must be 0 or 1");

    if(elementCount != nullptr)
    {
        *elementCount = 1;
    }
    if(requestedElementCount == 1)
    {
        THROW_IF_NULL(arrayOfElements,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "Engine prediction output is null");
        const auto& buffer = ensurePrediction();
        *static_cast<hipdnnBackendFlatbufferData_t*>(arrayOfElements)
            = {buffer.data(), buffer.size()};
    }
}

std::string EngineDescriptor::toString() const
{
    std::string str = "EngineDescriptor: {engineId=";
    str += _engineIdSet ? std::to_string(_engineId) : "unset";
    str += ", engineName="
           + (_detailsLoaded.load(std::memory_order_acquire) && !_engineName.empty()
                  ? _engineName
                  : std::string("unset"));
    str += _graph ? ", graph=" + fmt::format("{:p}", static_cast<const void*>(_graph.get()))
                  : ", graph=null";
    str += '}';
    return str;
}

} // namespace hipdnn_backend
