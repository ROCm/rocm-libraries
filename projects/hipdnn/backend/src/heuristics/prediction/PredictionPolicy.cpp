// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PredictionPolicy.hpp"

#include "heuristics/BuiltInLogging.hpp"

#include <hipdnn_data_sdk/utilities/EngineOrdering.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/device_properties_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_prediction_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace hipdnn_backend::heuristics::prediction
{
namespace
{
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::MODE_A_POLICY_NAME;
using hipdnn_data_sdk::utilities::MODE_B_POLICY_NAME;
using hipdnn_data_sdk::utilities::policyNameToId;

thread_local char lastError[1024]{};

// File-scope logging callback / level, set through the C-ABI-shaped
// SetLoggingCallback / SetLogLevel below. registerPlugin() installs a callback
// that forwards to the backend logger, so lines from this module reach the same
// sink as the rest of the backend. Same identity contract as the Config
// built-in: the last writer wins, and that is fine because every callback
// forwards to one process-wide sink.
hipdnnCallback_t g_loggingCallback = nullptr; // NOLINT(readability-identifier-naming)
hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO; // NOLINT(readability-identifier-naming)

#define PREDICTION_BUILTIN_LOG(severity, ...) \
    HIPDNN_BUILTIN_HEURISTIC_LOG(             \
        g_loggingCallback, g_logLevel, severity, "[BuiltInPrediction] ", __VA_ARGS__)

// The backend built-in adapter (PredictionBuiltIn) dispatches these functions.
// No engine catalogs, HIP calls, tuning caches, or model loaders belong here.
template <typename F>
hipdnnPluginStatus_t guarded(F&& f) noexcept
{
    try
    {
        return f();
    }
    catch(const std::exception& error)
    {
        std::snprintf(lastError, sizeof(lastError), "%s", error.what());
    }
    catch(...)
    {
        std::snprintf(lastError, sizeof(lastError), "Unknown prediction policy exception");
    }
    return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
}

struct Session
{
    bool devicePropertiesSet = false;
};

struct Handle
{
    std::shared_ptr<Session> session = std::make_shared<Session>();
};

struct RankedEngine
{
    int64_t id;
    bool available = false;
    double tflops = 0;
    flatbuffers::DetachedBuffer config;
};

struct Descriptor
{
    std::shared_ptr<Session> session;
    bool modeB = false;
    bool graphSet = false;
    bool finalized = false;
    std::vector<int64_t> engineIds;
    std::vector<RankedEngine> ranked;

    void invalidate()
    {
        finalized = false;
        ranked.clear();
    }
};

template <typename T>
bool verify(const hipdnnPluginConstData_t* data)
{
    if(data == nullptr || data->ptr == nullptr || data->size == 0)
    {
        return false;
    }
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(data->ptr), data->size);
    return verifier.VerifyBuffer<T>(nullptr);
}

bool usableConfig(const EngineConfig* config, int64_t engineId)
{
    if(config == nullptr || config->engine_id() != engineId)
    {
        return false;
    }
    std::unordered_set<std::string_view> knobIds;
    if(const auto* knobs = config->knobs())
    {
        for(const auto* knob : *knobs)
        {
            if(knob == nullptr || knob->knob_id() == nullptr || knob->knob_id()->size() == 0
               || !knobIds.insert(knob->knob_id()->string_view()).second
               || knob->value() == nullptr)
            {
                return false;
            }
            switch(knob->value_type())
            {
            case KnobValue::IntValue:
            case KnobValue::StringValue:
                break;
            case KnobValue::FloatValue:
                if(!std::isfinite(knob->value_as_FloatValue()->value()))
                {
                    return false;
                }
                break;
            default:
                return false;
            }
        }
    }
    return true;
}

const EnginePrediction* query(const hipdnnHeuristicHostCallbacks_t& host,
                              int64_t engineId,
                              hipdnnEnginePredictionKind_t kind)
{
    hipdnnPluginConstData_t data{};
    if(host.get_prediction(host.context, engineId, kind, &data) != HIPDNN_PLUGIN_STATUS_SUCCESS
       || !verify<EnginePrediction>(&data))
    {
        return nullptr;
    }
    const auto* prediction = flatbuffers::GetRoot<EnginePrediction>(data.ptr);
    const auto expectedKind = kind == HIPDNN_ENGINE_PREDICTION_ENGINE
                                  ? PredictionKind::ENGINE
                                  : PredictionKind::CONFIGURATION;
    if(prediction->engine_id() != engineId || prediction->kind() != expectedKind
       || prediction->status() != PredictionStatus::AVAILABLE
       || !std::isfinite(prediction->tflops()) || prediction->tflops() < 0)
    {
        return nullptr;
    }
    if(kind == HIPDNN_ENGINE_PREDICTION_CONFIGURATION
       && !usableConfig(prediction->engine_config(), engineId))
    {
        return nullptr;
    }
    return prediction;
}
} // namespace

hipdnnPluginStatus_t getName(const char** name)
{
    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = "BuiltInPredictionHeuristic";
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = "1.0.0";
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getApiVersion(const char** version)
{
    if(version == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *type = HIPDNN_PLUGIN_TYPE_HEURISTIC;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t setLoggingCallback(hipdnnCallback_t callback)
{
    g_loggingCallback = callback;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t setLogLevel(hipdnnSeverity_t level)
{
    g_logLevel = level;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void getLastErrorString(const char** error)
{
    if(error != nullptr)
    {
        *error = lastError;
    }
}

hipdnnPluginStatus_t getAllPolicyIds(int64_t* ids, uint32_t capacity, uint32_t* count)
{
    if(count == nullptr || (capacity != 0 && ids == nullptr))
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *count = 2;
    if(capacity == 0)
    {
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    if(capacity < 2)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    ids[0] = policyNameToId(MODE_A_POLICY_NAME);
    ids[1] = policyNameToId(MODE_B_POLICY_NAME);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getPolicyName(int64_t id, const char** name)
{
    if(name == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = nullptr;
    if(id == policyNameToId(MODE_A_POLICY_NAME))
    {
        *name = MODE_A_POLICY_NAME;
    }
    else if(id == policyNameToId(MODE_B_POLICY_NAME))
    {
        *name = MODE_B_POLICY_NAME;
    }
    return *name == nullptr ? HIPDNN_PLUGIN_STATUS_BAD_PARAM : HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t handleCreate(hipdnnHeuristicHandle_t* handle)
{
    if(handle == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *handle = nullptr;
    return guarded([&] {
        auto owned = std::make_unique<Handle>();
        *handle = reinterpret_cast<hipdnnHeuristicHandle_t>(owned.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    });
}

hipdnnPluginStatus_t handleDestroy(hipdnnHeuristicHandle_t handle)
{
    if(handle == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    delete reinterpret_cast<Handle*>(handle);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t handleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                               const hipdnnPluginConstData_t* properties)
{
    if(handle == nullptr || !verify<DeviceProperties>(properties))
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    reinterpret_cast<Handle*>(handle)->session->devicePropertiesSet = true;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t policyDescriptorCreate(hipdnnHeuristicHandle_t handle,
                                            int64_t policyId,
                                            hipdnnHeuristicPolicyDescriptor_t* descriptor)
{
    if(descriptor == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *descriptor = nullptr;
    if(handle == nullptr
       || (policyId != policyNameToId(MODE_A_POLICY_NAME)
           && policyId != policyNameToId(MODE_B_POLICY_NAME)))
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    return guarded([&] {
        auto owned = std::make_unique<Descriptor>();
        owned->session = reinterpret_cast<Handle*>(handle)->session;
        owned->modeB = policyId == policyNameToId(MODE_B_POLICY_NAME);
        *descriptor = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(owned.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    });
}

hipdnnPluginStatus_t policyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t descriptor)
{
    if(descriptor == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    delete reinterpret_cast<Descriptor*>(descriptor);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t policySetEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                        const int64_t* ids,
                                        size_t count)
{
    if(descriptor == nullptr || (count != 0 && ids == nullptr))
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    return guarded([&] {
        auto& desc = *reinterpret_cast<Descriptor*>(descriptor);
        desc.invalidate();
        desc.engineIds.clear();
        if(count != 0)
        {
            std::unordered_set<int64_t> unique(ids, ids + count);
            if(unique.size() != count)
            {
                return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
            }
            desc.engineIds.assign(ids, ids + count);
        }
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    });
}

hipdnnPluginStatus_t policySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                              const hipdnnPluginConstData_t* graph)
{
    if(descriptor == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    auto& desc = *reinterpret_cast<Descriptor*>(descriptor);
    desc.invalidate();
    desc.graphSet = verify<Graph>(graph);
    return desc.graphSet ? HIPDNN_PLUGIN_STATUS_SUCCESS : HIPDNN_PLUGIN_STATUS_BAD_PARAM;
}

hipdnnPluginStatus_t policyFinalize(hipdnnHeuristicPolicyDescriptor_t descriptor, int32_t* applied)
{
    return policyFinalizeWithHost(descriptor, nullptr, applied);
}

hipdnnPluginStatus_t policyFinalizeWithHost(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                            const hipdnnHeuristicHostCallbacks_t* host,
                                            int32_t* applied)
{
    if(applied == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *applied = 0;
    if(descriptor == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    return guarded([&] {
        auto& desc = *reinterpret_cast<Descriptor*>(descriptor);
        desc.invalidate();
        if(host == nullptr)
        {
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }
        if(host->version != 1 || host->struct_size < sizeof(hipdnnHeuristicHostCallbacks_t)
           || host->get_prediction == nullptr)
        {
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
        }
        if(!desc.graphSet || !desc.session->devicePropertiesSet)
        {
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }
        desc.ranked.reserve(desc.engineIds.size());
        bool available = false;
        for(const auto id : desc.engineIds)
        {
            RankedEngine row{id, false, 0, {}};
            const EnginePrediction* estimate = nullptr;
            if(desc.modeB)
            {
                estimate = query(*host, id, HIPDNN_ENGINE_PREDICTION_CONFIGURATION);
            }
            if(estimate == nullptr)
            {
                estimate = query(*host, id, HIPDNN_ENGINE_PREDICTION_ENGINE);
            }
            EngineConfigT config;
            config.engine_id = id;
            if(estimate != nullptr)
            {
                row.available = true;
                row.tflops = estimate->tflops();
                available = true;
                if(desc.modeB && estimate->kind() == PredictionKind::CONFIGURATION)
                {
                    estimate->engine_config()->UnPackTo(&config);
                }
            }
            // L1-only and unknown engines use their ordinary selector; only the quick
            // policy's winner is refined below (RFC 0019 §11.2).
            // L2 estimates retain the exact scored configuration and all its knobs.
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(EngineConfig::Pack(builder, &config));
            row.config = builder.Release();
            desc.ranked.push_back(std::move(row));
        }
        if(!available)
        {
            desc.ranked.clear();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }
        std::stable_sort(desc.ranked.begin(),
                         desc.ranked.end(),
                         [](const RankedEngine& left, const RankedEngine& right) {
                             if(left.available != right.available)
                             {
                                 return left.available;
                             }
                             return left.available && left.tflops > right.tflops;
                         });
        // RFC 0019 §11.2: the quick policy "ranks applicable engines by A; picks the
        // winner; if the winner has a config UHD (B), runs it to pick the kernel"
        // (§11.2 quick policy, table row "A and B"). Only the winner drills down —
        // losers are never scored at the kernel level — and the ranking is untouched
        // because §11.2 ranks by A alone. A winner that answers nothing usable keeps
        // its knob-less config and performs its own kernel selection (row "A only").
        if(!desc.modeB && desc.ranked.front().available)
        {
            auto& winner = desc.ranked.front();
            if(const auto* drilled
               = query(*host, winner.id, HIPDNN_ENGINE_PREDICTION_CONFIGURATION))
            {
                EngineConfigT config;
                drilled->engine_config()->UnPackTo(&config);
                flatbuffers::FlatBufferBuilder builder;
                builder.Finish(EngineConfig::Pack(builder, &config));
                winner.config = builder.Release();
            }
        }
        // RFC 0019 §11.2: an engine that supplies neither estimate "falls back to
        // static ordering; contributes no score" (table row "No declared model") and
        // "is ordered by the existing static rules". Emitting the unscored tail in
        // candidate-arrival order is neither the static rules nor deterministic, so
        // order it with the shared static ordering; scored rows keep their ranking.
        const auto tail = std::find_if(desc.ranked.begin(),
                                       desc.ranked.end(),
                                       [](const RankedEngine& row) { return !row.available; });
        if(tail != desc.ranked.end())
        {
            std::vector<int64_t> unscored;
            unscored.reserve(static_cast<size_t>(desc.ranked.end() - tail));
            for(auto row = tail; row != desc.ranked.end(); ++row)
            {
                unscored.push_back(row->id);
            }
            hipdnn_data_sdk::utilities::sortEngineIds(unscored);
            // Engine ids are unique (policySetEngineIds rejects duplicates), so each
            // target id selects exactly one row and the permutation is a swap chain.
            for(std::size_t i = 0; i < unscored.size(); ++i)
            {
                const auto slot = tail + static_cast<std::ptrdiff_t>(i);
                if(slot->id == unscored[i])
                {
                    continue;
                }
                std::iter_swap(slot,
                               std::find_if(slot + 1,
                                            desc.ranked.end(),
                                            [target = unscored[i]](const RankedEngine& row) {
                                                return row.id == target;
                                            }));
            }
            // Mode A ranks on the engine-level prediction alone and never asks an
            // engine for a configuration-level one, so an engine here may hold a
            // perfectly good `sort_kernel_catalog` model that this policy declined to
            // pay for. That is the deliberate cost of the quick policy (RFC 0019
            // §11.2), but it is not visible in the result, so say it out loud: the
            // ordering these engines received is vendor precedence, not merit, and
            // Mode B is the policy that would have scored them.
            if(!desc.modeB)
            {
                std::string names;
                for(const auto id : unscored)
                {
                    names += (names.empty() ? "" : ", ") + std::to_string(id);
                }
                PREDICTION_BUILTIN_LOG(HIPDNN_SEV_WARN,
                                       "ModeA ranked %zu of %zu engines on their engine-level "
                                       "prediction; engine(s) %s supplied none and were appended "
                                       "in static order without being scored. Select "
                                       "SelectionHeuristic::ModeB to rank on configuration-level "
                                       "predictions instead.",
                                       desc.ranked.size() - unscored.size(),
                                       desc.ranked.size(),
                                       names.c_str());
            }
        }
        desc.finalized = true;
        *applied = 1;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    });
}

hipdnnPluginStatus_t policyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                              int64_t* ids,
                                              size_t* count)
{
    if(descriptor == nullptr || count == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    const auto& desc = *reinterpret_cast<Descriptor*>(descriptor);
    if(!desc.finalized)
    {
        return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
    }
    const auto size = ids == nullptr ? desc.ranked.size() : std::min(*count, desc.ranked.size());
    if(ids != nullptr)
    {
        for(size_t i = 0; i < size; ++i)
        {
            ids[i] = desc.ranked[i].id;
        }
    }
    *count = size;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t policyGetEngineConfig(hipdnnHeuristicPolicyDescriptor_t descriptor,
                                           int64_t engineId,
                                           hipdnnPluginConstData_t* config)
{
    if(config == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *config = {};
    if(descriptor == nullptr)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    const auto& desc = *reinterpret_cast<Descriptor*>(descriptor);
    if(!desc.finalized)
    {
        return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
    }
    for(const auto& row : desc.ranked)
    {
        if(row.id == engineId)
        {
            *config = {row.config.data(), row.config.size()};
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }
    }
    return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
}

} // namespace hipdnn_backend::heuristics::prediction
