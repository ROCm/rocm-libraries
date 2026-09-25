// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "PredictionPolicy.hpp"

#include "heuristics/BuiltInLogging.hpp"

#include <hipdnn_data_sdk/utilities/EngineOrdering.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hipdnn_backend::heuristics::prediction
{
namespace
{
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC;
using hipdnn_data_sdk::utilities::findRankingMetric;
using hipdnn_data_sdk::utilities::isBetterMetricValue;
using hipdnn_data_sdk::utilities::isValidMetricValue;
using hipdnn_data_sdk::utilities::MODE_A_POLICY_NAME;
using hipdnn_data_sdk::utilities::MODE_B_POLICY_NAME;
using hipdnn_data_sdk::utilities::policyNameToId;
using hipdnn_data_sdk::utilities::RankingMetric;

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
    double value = 0;
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

// The metric this finalize ranks by. A version 1 host predates metrics and meant TFLOPS;
// a version 2 host names the metric, and a table too short to carry the field is read
// the same way as version 1 rather than past its end. nullptr: a name no registry row
// matches, which cannot be ranked because it has no direction.
const RankingMetric* requestedMetric(const hipdnnHeuristicHostCallbacks_t& host)
{
    constexpr auto METRIC_END
        = offsetof(hipdnnHeuristicHostCallbacks_t, ranking_metric) + sizeof(const char*);
    if(host.version < 2 || host.struct_size < METRIC_END)
    {
        return findRankingMetric(DEFAULT_RANKING_METRIC);
    }
    return host.ranking_metric == nullptr ? nullptr : findRankingMetric(host.ranking_metric);
}

const EnginePrediction* query(const hipdnnHeuristicHostCallbacks_t& host,
                              int64_t engineId,
                              hipdnnEnginePredictionKind_t kind,
                              const RankingMetric& metric)
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
    // The host already refuses an answer in another metric; checking again costs nothing
    // and keeps a value from being compared against numbers of a different quantity.
    if(prediction->engine_id() != engineId || prediction->kind() != expectedKind
       || prediction->status() != PredictionStatus::AVAILABLE || prediction->metric() == nullptr
       || prediction->metric()->string_view() != metric.name
       || !isValidMetricValue(metric, prediction->value()))
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

std::string joinIds(const std::vector<int64_t>& ids)
{
    std::string names;
    for(const auto id : ids)
    {
        names += (names.empty() ? "" : ", ") + std::to_string(id);
    }
    return names;
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
        if(host->version < 1
           || host->struct_size < offsetof(hipdnnHeuristicHostCallbacks_t, get_prediction)
                                      + sizeof(host->get_prediction)
           || host->get_prediction == nullptr)
        {
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
        }
        const auto* metric = requestedMetric(*host);
        if(metric == nullptr)
        {
            std::snprintf(lastError,
                          sizeof(lastError),
                          "Host names unregistered ranking metric '%s'",
                          host->ranking_metric == nullptr ? "(null)" : host->ranking_metric);
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
        }
        if(!desc.graphSet || !desc.session->devicePropertiesSet)
        {
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }
        // RFC 0019 §11.2: the quick policy (ModeA) ranks by L1 alone and never evaluates
        // L2 — not even to fill in the winner's kernel, which plan build chooses by the
        // same metric (RFC 0019 Open Question 21, option (b)). The thorough policy (ModeB)
        // asks each engine for L2 first and falls back to L1 where L2 has no answer.
        desc.ranked.reserve(desc.engineIds.size());
        bool available = false;
        for(const auto id : desc.engineIds)
        {
            RankedEngine row{id, false, 0, {}};
            const EnginePrediction* estimate = nullptr;
            if(desc.modeB)
            {
                estimate = query(*host, id, HIPDNN_ENGINE_PREDICTION_CONFIGURATION, *metric);
            }
            if(estimate == nullptr)
            {
                estimate = query(*host, id, HIPDNN_ENGINE_PREDICTION_ENGINE, *metric);
            }
            EngineConfigT config;
            config.engine_id = id;
            if(estimate != nullptr)
            {
                row.available = true;
                row.value = estimate->value();
                available = true;
                if(estimate->kind() == PredictionKind::CONFIGURATION)
                {
                    // L2 estimates retain the exact scored configuration and all its knobs.
                    estimate->engine_config()->UnPackTo(&config);
                }
            }
            // Engines without an L2 configuration keep a knob-less config, so their own
            // selector picks the kernel at plan build, ranked by the same metric.
            config.ranking_metric = std::string(metric->name);
            flatbuffers::FlatBufferBuilder builder;
            builder.Finish(EngineConfig::Pack(builder, &config));
            row.config = builder.Release();
            desc.ranked.push_back(std::move(row));
        }
        if(!available)
        {
            // A static order returned from here would read as a ranking by the metric it
            // is not, so decline and let the next policy (normally static ordering) say
            // what it is (RFC 0019 §11.2).
            PREDICTION_BUILTIN_LOG(HIPDNN_SEV_WARN,
                                   "%s declined: no engine serves '%.*s' at %s",
                                   desc.modeB ? MODE_B_POLICY_NAME : MODE_A_POLICY_NAME,
                                   static_cast<int>(metric->name.size()),
                                   metric->name.data(),
                                   desc.modeB ? "L1 or L2" : "L1");
            desc.ranked.clear();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }
        // Scored engines first, best-first in the metric's direction; ties and the
        // unscored tail follow the static rules (RFC 0019 §11.2), not candidate-arrival
        // order, which is neither those rules nor deterministic.
        std::vector<int64_t> staticOrder = desc.engineIds;
        hipdnn_data_sdk::utilities::sortEngineIds(staticOrder);
        std::unordered_map<int64_t, std::size_t> staticRank;
        staticRank.reserve(staticOrder.size());
        for(std::size_t rank = 0; rank < staticOrder.size(); ++rank)
        {
            staticRank.emplace(staticOrder[rank], rank);
        }
        std::sort(desc.ranked.begin(),
                  desc.ranked.end(),
                  [&](const RankedEngine& left, const RankedEngine& right) {
                      if(left.available != right.available)
                      {
                          return left.available;
                      }
                      if(left.available)
                      {
                          if(isBetterMetricValue(*metric, left.value, right.value))
                          {
                              return true;
                          }
                          if(isBetterMetricValue(*metric, right.value, left.value))
                          {
                              return false;
                          }
                      }
                      return staticRank.at(left.id) < staticRank.at(right.id);
                  });
        // An unscored engine may hold a perfectly good model for another metric, or (under
        // ModeA) an L2 model this policy declined to pay for; its place is vendor
        // precedence, not merit, and the result does not show that, so say it once.
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
            PREDICTION_BUILTIN_LOG(HIPDNN_SEV_WARN,
                                   "%s ranked %zu of %zu engines by '%.*s' at %s; engine(s) %s "
                                   "had no usable '%.*s' prediction and were appended in static "
                                   "order without being scored.",
                                   desc.modeB ? MODE_B_POLICY_NAME : MODE_A_POLICY_NAME,
                                   desc.ranked.size() - unscored.size(),
                                   desc.ranked.size(),
                                   static_cast<int>(metric->name.size()),
                                   metric->name.data(),
                                   desc.modeB ? "L1 or L2" : "L1",
                                   joinIds(unscored).c_str(),
                                   static_cast<int>(metric->name.size()),
                                   metric->name.data());
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
