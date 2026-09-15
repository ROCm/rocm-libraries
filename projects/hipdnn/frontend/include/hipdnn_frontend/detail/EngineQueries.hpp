// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// @file
/// Generation-tool surfaces for engine inspection (RFC 0019 Open Question 12,
/// RFC 0017 §2). Catalog enumeration and calibrated predictions are read from the
/// ENGINE and ENGINECFG backend descriptors; they are deliberately not part of the
/// public Graph API, because a consumer selects engines through the heuristic
/// descriptor, never by walking a kernel catalog.

#include <HipdnnBackendFlatbufferData.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_frontend/EngineQueryTypes.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/CreateBackendDescriptor.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>
#include <hipdnn_frontend/detail/KnobPacker.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>

namespace hipdnn_frontend::detail
{

/// Creates an unfinalized engine descriptor bound to the graph. Inspection inputs
/// (paging, scope, evaluate) must be set before finalize, so the caller finalizes.
inline Error beginEngineDescriptor(ScopedHipdnnBackendDescriptor& engineDesc,
                                   hipdnnBackendDescriptor_t graphDesc,
                                   int64_t engineId)
{
    engineDesc = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    if(!engineDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Cannot create an engine descriptor"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(engineDesc.get(),
                                               HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                               graphDesc,
                                               "inspection operation graph"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(engineDesc.get(),
                                               HIPDNN_ATTR_ENGINE_GLOBAL_INDEX,
                                               HIPDNN_TYPE_INT64,
                                               engineId,
                                               "inspection engine id"));
    return {};
}

/// Reads a descriptor attribute that returns one owned flatbuffer. The bytes stay
/// valid until the descriptor is destroyed.
inline Error readFlatbufferAttribute(hipdnnBackendDescriptor_t desc,
                                     hipdnnBackendAttributeName_t attributeName,
                                     hipdnnBackendFlatbufferData_t& data,
                                     const char* context)
{
    data = {};
    const auto status = hipdnnBackend()->backendGetAttribute(
        desc, attributeName, HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT, 1, nullptr, &data);
    if(status == HIPDNN_STATUS_NOT_SUPPORTED)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, std::string(context) + " is not supported"};
    }
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, std::string("Cannot read ") + context);
    if(data.ptr == nullptr || data.size == 0)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, std::string("Empty ") + context};
    }
    return {};
}

/// Applies a knob scope to the engine descriptor being prepared for enumeration.
inline Error applyCandidateScope(hipdnnBackendDescriptor_t engineDesc,
                                 const std::vector<KnobSetting>& scope)
{
    if(scope.empty())
    {
        return {};
    }
    std::vector<ScopedHipdnnBackendDescriptor> knobDescs;
    knobDescs.reserve(scope.size());
    std::vector<hipdnnBackendDescriptor_t> knobDescPtrs;
    knobDescPtrs.reserve(scope.size());
    for(const auto& setting : scope)
    {
        ScopedHipdnnBackendDescriptor knobDesc;
        HIPDNN_CHECK_ERROR(createKnobSettingDescriptor(setting, knobDesc));
        knobDescs.push_back(std::move(knobDesc));
        knobDescPtrs.push_back(knobDescs.back().get());
    }
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendSetAttribute(engineDesc,
                                             HIPDNN_ATTR_ENGINE_CANDIDATE_SCOPE_EXT,
                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                             static_cast<int64_t>(knobDescPtrs.size()),
                                             static_cast<const void*>(knobDescPtrs.data())),
        "Cannot restrict the candidate scope");
    return {};
}

inline Error
    decodeEnginePrediction(const hipdnn_flatbuffers_sdk::data_objects::EnginePrediction& source,
                           int64_t engineId,
                           PredictionKind kind,
                           EnginePrediction& prediction)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;

    const auto expectedKind = kind == PredictionKind::ENGINE ? fb::PredictionKind::ENGINE
                                                             : fb::PredictionKind::CONFIGURATION;
    if(source.engine_id() != engineId || source.kind() != expectedKind)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Prediction identity does not match query"};
    }

    EnginePrediction decoded;
    decoded.engineId = engineId;
    decoded.kind = kind;
    switch(source.status())
    {
    case fb::PredictionStatus::AVAILABLE:
        decoded.status = PredictionStatus::AVAILABLE;
        decoded.tflops = source.tflops();
        if(!std::isfinite(*decoded.tflops) || *decoded.tflops < 0.0)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Invalid calibrated prediction"};
        }
        break;
    case fb::PredictionStatus::UNAVAILABLE:
        decoded.status = PredictionStatus::UNAVAILABLE;
        break;
    case fb::PredictionStatus::INVALID:
        decoded.status = PredictionStatus::INVALID;
        break;
    default:
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Unknown prediction status"};
    }
    if(source.uhd_id() != nullptr)
    {
        decoded.model = source.uhd_id()->str();
    }
    if(source.reason() != nullptr)
    {
        decoded.reason = source.reason()->str();
    }
    try
    {
        if(source.binding_json() != nullptr && source.binding_json()->size() != 0)
        {
            decoded.binding = nlohmann::json::parse(source.binding_json()->str());
        }
        if(source.features_json() != nullptr && source.features_json()->size() != 0)
        {
            decoded.features = nlohmann::json::parse(source.features_json()->str());
        }
    }
    catch(const nlohmann::json::exception& error)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                std::string("Invalid prediction metadata: ") + error.what()};
    }
    if(!decoded.binding.is_object() || !decoded.features.is_object())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Invalid prediction metadata"};
    }

    if(const auto* configuration = source.engine_config())
    {
        EngineVariant variant;
        variant.engineId = configuration->engine_id();
        if(variant.engineId != engineId)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Prediction configuration engine mismatch"};
        }
        if(const auto* settings = configuration->knobs())
        {
            for(const auto* setting : *settings)
            {
                if(setting->knob_id() == nullptr)
                {
                    return {ErrorCode::HIPDNN_BACKEND_ERROR, "Unnamed prediction knob"};
                }
                KnobValueVariant value;
                switch(setting->value_type())
                {
                case fb::KnobValue::IntValue:
                    value = setting->value_as_IntValue()->value();
                    break;
                case fb::KnobValue::FloatValue:
                    if(!std::isfinite(setting->value_as_FloatValue()->value()))
                    {
                        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Non-finite prediction knob"};
                    }
                    value = setting->value_as_FloatValue()->value();
                    break;
                case fb::KnobValue::StringValue:
                    if(setting->value_as_StringValue()->value() == nullptr)
                    {
                        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Null string knob value"};
                    }
                    value = setting->value_as_StringValue()->value()->str();
                    break;
                default:
                    return {ErrorCode::HIPDNN_BACKEND_ERROR, "Invalid prediction knob value"};
                }
                if(!variant.knobSettings.emplace(setting->knob_id()->str(), std::move(value))
                        .second)
                {
                    return {ErrorCode::HIPDNN_BACKEND_ERROR, "Duplicate prediction knob"};
                }
            }
        }
        decoded.configuration = std::move(variant);
    }
    if(decoded.status == PredictionStatus::AVAILABLE && kind == PredictionKind::CONFIGURATION
       && !decoded.configuration)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Available configuration prediction has no configuration"};
    }
    prediction = std::move(decoded);
    return {};
}

/// Generation-tool surface. Reads a calibrated prediction from the descriptor that
/// defines its kind: the engine descriptor for an engine-level estimate, an engine
/// config descriptor for the exact configuration its knobs describe. Missing models
/// report UNAVAILABLE; they never affect engine applicability.
inline Error getEnginePrediction(hipdnnBackendDescriptor_t graphDesc,
                                 int64_t engineId,
                                 EnginePrediction& prediction,
                                 PredictionKind kind = PredictionKind::ENGINE,
                                 bool evaluate = true,
                                 const std::vector<KnobSetting>& constraints = {})
{
    prediction = {};
    if(graphDesc == nullptr
       || (kind != PredictionKind::ENGINE && kind != PredictionKind::CONFIGURATION))
    {
        return {ErrorCode::INVALID_VALUE, "Prediction requires a built graph and valid kind"};
    }
    if(kind == PredictionKind::ENGINE && !constraints.empty())
    {
        return {ErrorCode::INVALID_VALUE,
                "Engine-level predictions take no knob constraints; query the configuration kind"};
    }

    const int64_t evaluateFlag = evaluate ? 1 : 0;
    ScopedHipdnnBackendDescriptor engineDesc;
    HIPDNN_CHECK_ERROR(beginEngineDescriptor(engineDesc, graphDesc, engineId));
    if(kind == PredictionKind::ENGINE)
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(engineDesc.get(),
                                                   HIPDNN_ATTR_ENGINE_PREDICTION_EVALUATE_EXT,
                                                   HIPDNN_TYPE_INT64,
                                                   evaluateFlag,
                                                   "prediction evaluate"));
    }
    HIPDNN_CHECK_ERROR(finalizeDescriptor(engineDesc.get(), "inspection engine descriptor"));

    hipdnnBackendFlatbufferData_t data{};
    ScopedHipdnnBackendDescriptor config;
    if(kind == PredictionKind::ENGINE)
    {
        HIPDNN_CHECK_ERROR(readFlatbufferAttribute(
            engineDesc.get(), HIPDNN_ATTR_ENGINE_PREDICTION_EXT, data, "engine prediction"));
    }
    else
    {
        config = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        if(!config.valid())
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Cannot create prediction constraints"};
        }
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(config.get(),
                                                   HIPDNN_ATTR_ENGINECFG_ENGINE,
                                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                   engineDesc.get(),
                                                   "prediction engine"));
        HIPDNN_CHECK_ERROR(applyKnobSettingsViaDescriptors(config.get(), constraints));
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(config.get(),
                                                   HIPDNN_ATTR_ENGINECFG_PREDICTION_EVALUATE_EXT,
                                                   HIPDNN_TYPE_INT64,
                                                   evaluateFlag,
                                                   "prediction evaluate"));
        // Knob-only and deliberately unfinalized: no engine metadata, catalog, or
        // workspace query is performed for a prediction configuration.
        HIPDNN_CHECK_ERROR(readFlatbufferAttribute(config.get(),
                                                   HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT,
                                                   data,
                                                   "engine configuration prediction"));
    }

    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(data.ptr), data.size);
    if(!verifier.VerifyBuffer<fb::EnginePrediction>())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Invalid engine prediction buffer"};
    }
    return decodeEnginePrediction(*fb::GetEnginePrediction(data.ptr), engineId, kind, prediction);
}

/// Generation-tool surface. Enumerates actual applicable catalog entries, not guessed
/// knob combinations. `scope` restricts the matched catalog; unset knobs are
/// unconstrained, and default knob values never restrict discovery. limit is in
/// [1, 10000]. Returned EngineCandidate::variant is accepted by add_engine_variants().
/// Engines that cannot enumerate return an error explicitly, not an empty catalog.
inline Error getEngineCandidates(hipdnnBackendDescriptor_t graphDesc,
                                 int64_t engineId,
                                 EngineCandidatePage& page,
                                 int64_t offset = 0,
                                 int64_t limit = 10000,
                                 const std::vector<KnobSetting>& scope = {})
{
    page = {};
    if(graphDesc == nullptr || offset < 0 || limit < 1 || limit > 10000)
    {
        return {ErrorCode::INVALID_VALUE,
                "Candidate enumeration requires a built graph, nonnegative offset, "
                "and limit in [1, 10000]"};
    }
    ScopedHipdnnBackendDescriptor engineDesc;
    HIPDNN_CHECK_ERROR(beginEngineDescriptor(engineDesc, graphDesc, engineId));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(engineDesc.get(),
                                               HIPDNN_ATTR_ENGINE_CANDIDATE_OFFSET_EXT,
                                               HIPDNN_TYPE_INT64,
                                               offset,
                                               "candidate offset"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(engineDesc.get(),
                                               HIPDNN_ATTR_ENGINE_CANDIDATE_LIMIT_EXT,
                                               HIPDNN_TYPE_INT64,
                                               limit,
                                               "candidate limit"));
    HIPDNN_CHECK_ERROR(applyCandidateScope(engineDesc.get(), scope));
    HIPDNN_CHECK_ERROR(finalizeDescriptor(engineDesc.get(), "inspection engine descriptor"));

    hipdnnBackendFlatbufferData_t data{};
    const auto status = hipdnnBackend()->backendGetAttribute(engineDesc.get(),
                                                             HIPDNN_ATTR_ENGINE_CANDIDATES_EXT,
                                                             HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT,
                                                             1,
                                                             nullptr,
                                                             &data);
    if(status == HIPDNN_STATUS_NOT_SUPPORTED)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Engine does not support matched-catalog enumeration"};
    }
    HIPDNN_RETURN_ON_BACKEND_FAILURE(status, "Cannot read candidate page");

    using namespace hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(data.ptr), data.size);
    if(data.ptr == nullptr || !verifier.VerifyBuffer<EngineDetails>())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Invalid candidate page buffer"};
    }
    const auto* details = GetEngineDetails(data.ptr);
    const auto* source = details->candidate_page();
    if(details->engine_id() != engineId || source == nullptr)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Invalid candidate page engine"};
    }
    hipdnn_frontend::EngineCandidatePage decoded;
    decoded.engineId = engineId;
    if(source->engine_name() != nullptr)
    {
        decoded.engineName = source->engine_name()->str();
    }
    if(source->engine_descriptor_id() != nullptr)
    {
        decoded.engineDescriptorId = source->engine_descriptor_id()->str();
    }
    decoded.graphId = source->graph_id()->str();
    decoded.deviceId = source->device_id()->str();
    decoded.deviceArch = source->device_arch()->str();
    decoded.offset = source->offset();
    decoded.totalCount = source->total_count();
    try
    {
        decoded.problemFeatures = nlohmann::json::parse(source->problem_features()->str());
        decoded.deviceFeatures = nlohmann::json::parse(source->device_features()->str());
        std::unordered_set<std::string> ids;
        std::set<std::map<KnobType_t, KnobValueVariant>> tuples;
        if(const auto* candidates = source->candidates())
        {
            decoded.candidates.reserve(candidates->size());
            for(const auto* entry : *candidates)
            {
                hipdnn_frontend::EngineCandidate candidate;
                candidate.id = entry->id()->str();
                if(candidate.id.empty() || !ids.insert(candidate.id).second
                   || (!decoded.candidates.empty() && decoded.candidates.back().id >= candidate.id))
                {
                    return {ErrorCode::HIPDNN_BACKEND_ERROR, "Ambiguous candidate identity"};
                }
                candidate.variant.engineId = engineId;
                candidate.kernelFeatures = nlohmann::json::parse(entry->kernel_features()->str());
                if(!candidate.kernelFeatures.is_object())
                {
                    return {ErrorCode::HIPDNN_BACKEND_ERROR, "Invalid kernel feature map"};
                }
                if(const auto* settings = entry->knob_settings())
                {
                    for(const auto* setting : *settings)
                    {
                        if(setting->knob_id() == nullptr)
                        {
                            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Unnamed candidate knob"};
                        }
                        KnobValueVariant value;
                        switch(setting->value_type())
                        {
                        case KnobValue::IntValue:
                            value = setting->value_as_IntValue()->value();
                            break;
                        case KnobValue::FloatValue:
                            value = setting->value_as_FloatValue()->value();
                            break;
                        case KnobValue::StringValue:
                            if(setting->value_as_StringValue()->value() == nullptr)
                            {
                                return {ErrorCode::HIPDNN_BACKEND_ERROR, "Null string knob value"};
                            }
                            value = setting->value_as_StringValue()->value()->str();
                            break;
                        default:
                            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                                    "Invalid candidate knob value"};
                        }
                        if(!candidate.variant.knobSettings
                                .emplace(setting->knob_id()->str(), std::move(value))
                                .second)
                        {
                            return {ErrorCode::HIPDNN_BACKEND_ERROR, "Duplicate candidate knob"};
                        }
                    }
                }
                if(!tuples.insert(candidate.variant.knobSettings).second)
                {
                    return {ErrorCode::HIPDNN_BACKEND_ERROR, "Ambiguous candidate knob tuple"};
                }
                for(const auto& pin : scope)
                {
                    const auto found = candidate.variant.knobSettings.find(pin.knobId());
                    if(found == candidate.variant.knobSettings.end()
                       || found->second != pin.value())
                    {
                        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Candidate violates knob scope"};
                    }
                }
                decoded.candidates.push_back(std::move(candidate));
            }
        }
    }
    catch(const nlohmann::json::exception& error)
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                std::string("Invalid candidate features: ") + error.what()};
    }
    if(!decoded.problemFeatures.is_object() || !decoded.deviceFeatures.is_object()
       || decoded.graphId.empty() || decoded.deviceId.empty() || decoded.deviceArch.empty()
       || decoded.offset != static_cast<uint64_t>(offset) || decoded.totalCount < decoded.offset
       || decoded.candidates.size()
              != std::min<uint64_t>(static_cast<uint64_t>(limit),
                                    decoded.totalCount - decoded.offset))
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Inconsistent candidate page"};
    }
    const auto next = decoded.offset + decoded.candidates.size();
    if(next < decoded.totalCount)
    {
        decoded.nextOffset = next;
    }
    page = std::move(decoded);
    return {};
}

} // namespace hipdnn_frontend::detail
