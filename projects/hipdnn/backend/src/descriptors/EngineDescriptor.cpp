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
#include "handle/Handle.hpp"
#include "logging/Logging.hpp"
#include "plugin/EnginePluginResourceManager.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_knob_catalog_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/knob_value_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineDetailsWrapper.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace hipdnn_backend
{

namespace
{

constexpr std::string_view GLOBAL_KNOB_PREFIX = "global.";

bool isGlobalKnob(const std::string& knobId)
{
    return knobId.compare(0, GLOBAL_KNOB_PREFIX.size(), GLOBAL_KNOB_PREFIX) == 0;
}

/// Order-independent identity of an entry's knob settings, used to detect equal entries.
std::string entryKey(const hipdnn_flatbuffers_sdk::data_objects::EngineConfigT& entry)
{
    std::vector<const hipdnn_flatbuffers_sdk::data_objects::KnobSettingT*> knobs;
    knobs.reserve(entry.knobs.size());
    for(const auto& knob : entry.knobs)
    {
        knobs.push_back(knob.get());
    }
    std::sort(knobs.begin(), knobs.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->knob_id < rhs->knob_id;
    });

    std::string key;
    for(const auto* knob : knobs)
    {
        key += std::to_string(knob->knob_id.size()) + ':' + knob->knob_id;
        switch(knob->value.type)
        {
        case hipdnn_flatbuffers_sdk::data_objects::KnobValue::IntValue:
            key += "i" + std::to_string(knob->value.AsIntValue()->value);
            break;
        case hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue:
        {
            std::array<char, 64> buffer{};
            std::snprintf(buffer.data(), buffer.size(), "f%a", knob->value.AsFloatValue()->value);
            key += buffer.data();
            break;
        }
        case hipdnn_flatbuffers_sdk::data_objects::KnobValue::StringValue:
        {
            const auto& value = knob->value.AsStringValue()->value;
            key += "s" + std::to_string(value.size()) + ':' + value;
            break;
        }
        default:
            key += "?";
            break;
        }
        key += ';';
    }
    return key;
}

/// Reports a catalog violation as HIPDNN_STATUS_INTERNAL_ERROR naming the engine and entry.
[[noreturn]] void failEntry(const std::string& engineLabel, size_t index, const std::string& reason)
{
    throw HipdnnException(HIPDNN_STATUS_INTERNAL_ERROR,
                          "Knob catalog for " + engineLabel + " entry " + std::to_string(index)
                              + " is invalid: " + reason);
}

/// Checks a plugin-supplied catalog against the five catalog rules. Throws
/// HIPDNN_STATUS_INTERNAL_ERROR naming the engine and the offending entry index.
void validateKnobCatalog(
    const std::string& engineLabel,
    int64_t engineId,
    const std::vector<std::shared_ptr<KnobDescriptor>>& advertisedKnobs,
    const std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT>>&
        entries)
{
    std::unordered_map<std::string, const KnobDescriptor*> advertised;
    for(const auto& knob : advertisedKnobs)
    {
        advertised.emplace(knob->knobId(), knob.get());
    }

    std::unordered_map<std::string, size_t> seenEntries;

    for(size_t index = 0; index < entries.size(); ++index)
    {
        const auto& entry = entries[index];
        if(entry == nullptr)
        {
            failEntry(engineLabel, index, "the entry is null.");
        }

        // Rule 5
        if(entry->engine_id != engineId)
        {
            failEntry(engineLabel,
                      index,
                      "it carries engine ID " + std::to_string(entry->engine_id)
                          + " instead of the queried engine ID " + std::to_string(engineId) + ".");
        }

        std::unordered_set<std::string> knobIds;
        for(const auto& setting : entry->knobs)
        {
            if(setting == nullptr)
            {
                failEntry(engineLabel, index, "it contains a null knob setting.");
            }

            const auto& knobId = setting->knob_id;

            // Rule 2
            if(isGlobalKnob(knobId))
            {
                failEntry(engineLabel,
                          index,
                          "knob '" + knobId
                              + "' is a global knob; catalogs contain engine-specific knobs only.");
            }

            // Rule 1
            const auto advertisedIt = advertised.find(knobId);
            if(advertisedIt == advertised.end())
            {
                failEntry(engineLabel,
                          index,
                          "knob '" + knobId + "' is not advertised by the engine for this graph.");
            }

            // Rule 4, within the entry
            if(!knobIds.insert(knobId).second)
            {
                failEntry(engineLabel,
                          index,
                          "knob '" + knobId + "' appears more than once in the entry.");
            }

            // Rule 3
            if(auto violation = advertisedIt->second->validateValue(setting->value))
            {
                failEntry(engineLabel,
                          index,
                          "knob '" + knobId + "' has an invalid value: " + *violation);
            }
        }

        // Rule 4, across entries
        const auto [seenIt, inserted] = seenEntries.emplace(entryKey(*entry), index);
        if(!inserted)
        {
            failEntry(engineLabel,
                      index,
                      "it sets the same knob values as entry " + std::to_string(seenIt->second)
                          + ".");
        }
    }
}

} // namespace

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

    // Declared here so its storage outlives the resolveEngineName() call below,
    // which takes a view of it.
    std::optional<std::string> detailsName;

    auto engineDetailsPtr = _engineDetails->get();
    if(engineDetailsPtr != nullptr)
    {
        const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineDetailsWrapper detailsWrapper(
            engineDetailsPtr);
        detailsName = detailsWrapper.name();
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

    _engineName = pluginResourceManager->resolveEngineName(
        _engineId, detailsName ? std::optional<std::string_view>(*detailsName) : std::nullopt);

    HipdnnBackendDescriptorImpl<EngineDescriptor>::finalize();
}

void EngineDescriptor::loadKnobCatalog() const
{
    auto pluginResourceManager = _graph->getHandle()->getPluginResourceManager();

    const auto engineLabel = "engine '" + _engineName + "' (ID " + std::to_string(_engineId) + ")";

    auto catalog = pluginResourceManager->getEngineKnobCatalog(_engineId, _graph.get());

    std::vector<std::unique_ptr<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT>> entries;
    if(catalog != nullptr)
    {
        entries = std::move(catalog->configs);
        validateKnobCatalog(engineLabel, _engineId, _knobDescriptors, entries);
    }
    else
    {
        // The plugin offers no catalog: an engine without engine-specific knobs has exactly
        // one configuration, the default one. With knobs, its configurations cannot be
        // derived from their metadata, so there is nothing valid to report.
        const auto engineSpecificKnobs = static_cast<size_t>(
            std::count_if(_knobDescriptors.begin(), _knobDescriptors.end(), [](const auto& knob) {
                return !isGlobalKnob(knob->knobId());
            }));
        THROW_IF_TRUE(engineSpecificKnobs > 0,
                      HIPDNN_STATUS_NOT_SUPPORTED,
                      "EngineDescriptor::getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT) failed: "
                          + engineLabel + " advertises " + std::to_string(engineSpecificKnobs)
                          + " engine-specific knob(s) but its plugin supplies no knob catalog.");

        auto defaultEntry = std::make_unique<hipdnn_flatbuffers_sdk::data_objects::EngineConfigT>();
        defaultEntry->engine_id = _engineId;
        entries.push_back(std::move(defaultEntry));
    }

    _knobCatalog = std::move(entries);
}

void EngineDescriptor::getKnobCatalogDescriptors(hipdnnBackendAttributeType_t attributeType,
                                                 int64_t requestedElementCount,
                                                 int64_t* elementCount,
                                                 void* arrayOfElements) const
{
    checkGetArgs(HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                 attributeType,
                 "EngineDescriptor::getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT)");

    // Fetched on first read rather than in finalize(): most callers never ask, and the
    // plugin may do real work to enumerate its configurations.
    std::call_once(_knobCatalogOnce, [this]() { loadKnobCatalog(); });

    const auto count = static_cast<int64_t>(_knobCatalog.size());

    if(arrayOfElements == nullptr || requestedElementCount == 0)
    {
        THROW_IF_NULL(elementCount,
                      HIPDNN_STATUS_BAD_PARAM_NULL_POINTER,
                      "EngineDescriptor::getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT): "
                      "elementCount is null");
        *elementCount = count;
        return;
    }

    THROW_IF_FALSE(requestedElementCount >= count,
                   HIPDNN_STATUS_BAD_PARAM,
                   "EngineDescriptor::getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT): "
                   "requestedElementCount < catalog entry count");

    auto self = weak_from_this().lock();
    THROW_IF_NULL(self,
                  HIPDNN_STATUS_INTERNAL_ERROR,
                  "EngineDescriptor::getAttribute(HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT) failed: "
                  "the descriptor is not owned by a shared_ptr.");

    std::vector<std::shared_ptr<EngineConfigDescriptor>> configs;
    configs.reserve(_knobCatalog.size());
    for(const auto& entry : _knobCatalog)
    {
        configs.push_back(EngineConfigDescriptor::fromEngineConfigT(self, *entry));
    }

    if(elementCount != nullptr)
    {
        *elementCount = count;
    }

    HipdnnBackendDescriptor::packDescriptorArray(
        configs, static_cast<HipdnnBackendDescriptor**>(arrayOfElements));
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
    case HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT:
        getKnobCatalogDescriptors(
            attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_BEHAVIOR_NOTE:
        getBehaviorNotes(attributeType, requestedElementCount, elementCount, arrayOfElements);
        break;
    case HIPDNN_ATTR_ENGINE_NAME_EXT:
        getString(_engineName,
                  attributeType,
                  requestedElementCount,
                  elementCount,
                  arrayOfElements,
                  "EngineDescriptor::getAttribute()");
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
    case HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT:
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
    str += ", engineName=" + (_engineName.empty() ? std::string("unset") : _engineName);
    str += _graph ? ", graph=" + fmt::format("{:p}", static_cast<const void*>(_graph.get()))
                  : ", graph=null";
    str += '}';
    return str;
}

} // namespace hipdnn_backend
