// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file ConfigBuiltIn.cpp
 * @brief Backend-internal implementation of the
 *        SelectionHeuristic::Config policy.
 *
 * The policy reads HIPDNN_HEUR_CONFIG_PATH (a JSON file mapping operation
 * match criteria to engine names via EngineOverrideConfig), walks supported
 * primary nodes in the serialized graph, and on the first matching rule
 * reorders the candidate engine IDs so the chosen engine is first. When the
 * env var is unset, the file is missing/invalid, no rule matches, or the
 * matched engine is not among the candidates, the policy declines so the
 * outer policy loop can try the next plugin.
 *
 * Mechanics mirror StaticOrderingBuiltIn — a function-pointer table wrapped
 * by HeuristicPlugin::createBuiltIn so registration and validation flow
 * through the same paths as dlopen-loaded plugins.
 */

#include "ConfigBuiltIn.hpp"

#include "EngineOverrideConfig.hpp"
#include "heuristics/BuiltInLogging.hpp"
#include "logging/Logging.hpp"

#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/HeuristicValidation.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>
#include <hipdnn_plugin_sdk/heuristic_api_version.h>

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace hipdnn_backend::heuristics::config
{
namespace
{
namespace config_criterion = hipdnn_data_sdk::detail::autotune_config::criterion;
namespace config_op = hipdnn_data_sdk::detail::autotune_config::op;
namespace config_tensor = hipdnn_data_sdk::detail::autotune_config::tensor;

constexpr const char* PLUGIN_NAME = "BuiltInConfigHeuristic";
constexpr const char* PLUGIN_VERSION = "1.0.0";
constexpr const char* POLICY_NAME = "SelectionHeuristic::Config";

// File-scope logging callback / level, set via the C-ABI-shaped
// SetLoggingCallback / SetLogLevel below. The backend supplies its own
// callback when registering the built-in (see PluginManagerBase::registerPlugin)
// so log lines from this module flow through the backend logger.
//
// Identity contract: the built-in is statically linked into the backend, so
// these globals live in the same process image as the caller. The last writer
// wins — if multiple HeuristicPluginManager instances register the built-in
// they overwrite each other's callback. This is intentional: registerPlugin()
// hands in a callback that forwards to the backend logger, which is itself a
// process-wide sink, so the identity of the "current" callback does not matter
// as long as one is installed. Do not assume per-instance scoping here.
hipdnnCallback_t g_loggingCallback = nullptr; // NOLINT(readability-identifier-naming)
hipdnnSeverity_t g_logLevel = HIPDNN_SEV_INFO; // NOLINT(readability-identifier-naming)

#define CONFIG_BUILTIN_LOG(severity, ...) \
    HIPDNN_BUILTIN_HEURISTIC_LOG(         \
        g_loggingCallback, g_logLevel, severity, "[BuiltInConfig] ", __VA_ARGS__)

int64_t policyId()
{
    static const int64_t s_id = hipdnn_data_sdk::utilities::policyNameToId(POLICY_NAME);
    return s_id;
}

// ---- Graph-walk helpers (lifted from the former PreferredEngineResolver) ---

using hipdnn_flatbuffers_sdk::data_objects::Graph;

std::vector<int64_t> toVector(const flatbuffers::Vector<int64_t>* fb)
{
    if(fb == nullptr)
    {
        return {};
    }
    return {fb->begin(), fb->end()};
}

struct TensorDimsStrides
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
};

std::unordered_map<int64_t, TensorDimsStrides> indexTensorsByUid(const Graph* graph)
{
    std::unordered_map<int64_t, TensorDimsStrides> out;
    const auto* tensors = graph->tensors();
    if(tensors == nullptr)
    {
        return out;
    }
    out.reserve(tensors->size());
    for(const auto* t : *tensors)
    {
        if(t == nullptr)
        {
            continue;
        }
        out.emplace(t->uid(), TensorDimsStrides{toVector(t->dims()), toVector(t->strides())});
    }
    return out;
}

constexpr int K_POINTWISE_PRIORITY = 10;
constexpr int K_REDUCTION_PRIORITY = 20;
constexpr int K_RESAMPLE_PRIORITY = 20;
constexpr int K_NORM_PRIORITY = 30;
constexpr int K_BATCHNORM_PRIORITY = 40;
constexpr int K_MATMUL_PRIORITY = 50;
constexpr int K_SDPA_PRIORITY = 60;
constexpr int K_CONVOLUTION_PRIORITY = 70;

// Priority gaps are intentional: they preserve the current primary-op ordering while leaving
// room to insert new operation classes without renumbering every existing class.
struct BackendAutotuneConfigMatchKey
{
    BackendAutotuneConfigMatchKey(std::string_view opName, int opPriority)
        : op(opName)
        , priority(opPriority)
    {
    }

    void appendTensor(std::string_view tensorId, const TensorDimsStrides& tensor)
    {
        tensors.push_back(TensorView{tensorId, &tensor.dims, &tensor.strides});
    }

    void addCriterion(std::string_view criterionName, int64_t criterionValue)
    {
        criteria.push_back(Criterion{std::string(criterionName), criterionValue});
    }

    std::string op;
    std::vector<Criterion> criteria;
    std::vector<TensorView> tensors;
    int priority = 0;
};

class BackendAutotuneConfigMatchKeyBuilder
{
public:
    explicit BackendAutotuneConfigMatchKeyBuilder(
        const std::unordered_map<int64_t, TensorDimsStrides>& tensorIndex)
        : _tensorIndex(tensorIndex)
    {
    }

    std::optional<BackendAutotuneConfigMatchKey>
        buildKeyForNode(const hipdnn_flatbuffers_sdk::data_objects::Node* node) const
    {
        if(node == nullptr)
        {
            return std::nullopt;
        }

        if(const auto* fwd = node->attributes_as_ConvolutionFwdAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::CONV_FPROP, K_CONVOLUTION_PRIORITY);
            if(!appendUid(key, config_tensor::X, fwd->x_tensor_uid())
               || !appendUid(key, config_tensor::W, fwd->w_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* bwd = node->attributes_as_ConvolutionBwdAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::CONV_DGRAD, K_CONVOLUTION_PRIORITY);
            if(!appendUid(key, config_tensor::DY, bwd->dy_tensor_uid())
               || !appendUid(key, config_tensor::W, bwd->w_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* wrw = node->attributes_as_ConvolutionWrwAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::CONV_WGRAD, K_CONVOLUTION_PRIORITY);
            if(!appendUid(key, config_tensor::X, wrw->x_tensor_uid())
               || !appendUid(key, config_tensor::DY, wrw->dy_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* sdpa = node->attributes_as_SdpaAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::SDPA_FWD, K_SDPA_PRIORITY);
            if(!appendUid(key, config_tensor::Q, sdpa->q_tensor_uid())
               || !appendUid(key, config_tensor::K, sdpa->k_tensor_uid())
               || !appendUid(key, config_tensor::V, sdpa->v_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SCALE, sdpa->scale_tensor_uid())
               || !appendOptionalUid(key, config_tensor::ATTN_MASK, sdpa->attn_mask_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SEQ_LEN_Q, sdpa->seq_len_q_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SEQ_LEN_KV, sdpa->seq_len_kv_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SEED, sdpa->seed_tensor_uid())
               || !appendOptionalUid(key, config_tensor::OFFSET, sdpa->offset_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::DROPOUT_MASK, sdpa->dropout_mask_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::DROPOUT_SCALE, sdpa->dropout_scale_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::PAGE_TABLE_K, sdpa->page_table_k_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::PAGE_TABLE_V, sdpa->page_table_v_tensor_uid())
               || !appendOptionalUid(key, config_tensor::BLOCK_MASK, sdpa->block_mask_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SINK_TOKEN, sdpa->sink_token_tensor_uid())
               || !appendOptionalUid(key, config_tensor::DESCALE_Q, sdpa->descale_q_tensor_uid())
               || !appendOptionalUid(key, config_tensor::DESCALE_K, sdpa->descale_k_tensor_uid())
               || !appendOptionalUid(key, config_tensor::DESCALE_V, sdpa->descale_v_tensor_uid())
               || !appendOptionalUid(key, config_tensor::DESCALE_S, sdpa->descale_s_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SCALE_S, sdpa->scale_s_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SCALE_O, sdpa->scale_o_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* sdpa = node->attributes_as_SdpaBackwardAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::SDPA_BWD, K_SDPA_PRIORITY);
            if(!appendUid(key, config_tensor::Q, sdpa->q_tensor_uid())
               || !appendUid(key, config_tensor::K, sdpa->k_tensor_uid())
               || !appendUid(key, config_tensor::V, sdpa->v_tensor_uid())
               || !appendUid(key, config_tensor::O, sdpa->o_tensor_uid())
               || !appendUid(key, config_tensor::DO, sdpa->do_tensor_uid())
               || !appendUid(key, config_tensor::STATS, sdpa->stats_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SCALE, sdpa->scale_tensor_uid())
               || !appendOptionalUid(key, config_tensor::ATTN_MASK, sdpa->attn_mask_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SEQ_LEN_Q, sdpa->seq_len_q_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SEQ_LEN_KV, sdpa->seq_len_kv_tensor_uid())
               || !appendOptionalUid(key, config_tensor::SEED, sdpa->seed_tensor_uid())
               || !appendOptionalUid(key, config_tensor::OFFSET, sdpa->offset_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::DROPOUT_MASK, sdpa->dropout_mask_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::DROPOUT_SCALE, sdpa->dropout_scale_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::DROPOUT_SCALE_INV, sdpa->dropout_scale_inv_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* matmul = node->attributes_as_MatmulAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::MATMUL, K_MATMUL_PRIORITY);
            if(!appendUid(key, config_tensor::A, matmul->a_tensor_uid())
               || !appendUid(key, config_tensor::B, matmul->b_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* batchnorm = node->attributes_as_BatchnormAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::BATCHNORM_TRAINING, K_BATCHNORM_PRIORITY);
            if(!appendUid(key, config_tensor::X, batchnorm->x_tensor_uid())
               || !appendUid(key, config_tensor::SCALE, batchnorm->scale_tensor_uid())
               || !appendUid(key, config_tensor::BIAS, batchnorm->bias_tensor_uid())
               || !appendUid(key, config_tensor::EPSILON, batchnorm->epsilon_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::PREV_RUNNING_MEAN, batchnorm->prev_running_mean_tensor_uid())
               || !appendOptionalUid(key,
                                     config_tensor::PREV_RUNNING_VARIANCE,
                                     batchnorm->prev_running_variance_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::MOMENTUM, batchnorm->momentum_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* batchnorm = node->attributes_as_BatchnormInferenceAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::BATCHNORM_INFERENCE, K_BATCHNORM_PRIORITY);
            if(!appendUid(key, config_tensor::X, batchnorm->x_tensor_uid())
               || !appendUid(key, config_tensor::MEAN, batchnorm->mean_tensor_uid())
               || !appendUid(key, config_tensor::INV_VARIANCE, batchnorm->inv_variance_tensor_uid())
               || !appendUid(key, config_tensor::SCALE, batchnorm->scale_tensor_uid())
               || !appendUid(key, config_tensor::BIAS, batchnorm->bias_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* batchnorm = node->attributes_as_BatchnormInferenceAttributesVarianceExt())
        {
            BackendAutotuneConfigMatchKey key(config_op::BATCHNORM_INFERENCE_VARIANCE_EXT,
                                              K_BATCHNORM_PRIORITY);
            if(!appendUid(key, config_tensor::X, batchnorm->x_tensor_uid())
               || !appendUid(key, config_tensor::MEAN, batchnorm->mean_tensor_uid())
               || !appendUid(key, config_tensor::VARIANCE, batchnorm->variance_tensor_uid())
               || !appendUid(key, config_tensor::SCALE, batchnorm->scale_tensor_uid())
               || !appendUid(key, config_tensor::BIAS, batchnorm->bias_tensor_uid())
               || !appendUid(key, config_tensor::EPSILON, batchnorm->epsilon_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* batchnorm = node->attributes_as_BatchnormBackwardAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::BATCHNORM_BACKWARD, K_BATCHNORM_PRIORITY);
            if(!appendUid(key, config_tensor::DY, batchnorm->dy_tensor_uid())
               || !appendUid(key, config_tensor::X, batchnorm->x_tensor_uid())
               || !appendUid(key, config_tensor::SCALE, batchnorm->scale_tensor_uid())
               || !appendOptionalUid(key, config_tensor::MEAN, batchnorm->mean_tensor_uid())
               || !appendOptionalUid(
                   key, config_tensor::INV_VARIANCE, batchnorm->inv_variance_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* layernorm = node->attributes_as_LayernormAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::LAYERNORM, K_NORM_PRIORITY);
            key.addCriterion(config_criterion::NORM_FWD_PHASE,
                             static_cast<int64_t>(layernorm->forward_phase()));
            if(!appendUid(key, config_tensor::X, layernorm->x_tensor_uid())
               || !appendUid(key, config_tensor::SCALE, layernorm->scale_tensor_uid())
               || !appendUid(key, config_tensor::BIAS, layernorm->bias_tensor_uid())
               || !appendUid(key, config_tensor::EPSILON, layernorm->epsilon_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* rmsnorm = node->attributes_as_RMSNormAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::RMSNORM, K_NORM_PRIORITY);
            key.addCriterion(config_criterion::NORM_FWD_PHASE,
                             static_cast<int64_t>(rmsnorm->forward_phase()));
            if(!appendUid(key, config_tensor::X, rmsnorm->x_tensor_uid())
               || !appendUid(key, config_tensor::SCALE, rmsnorm->scale_tensor_uid())
               || !appendUid(key, config_tensor::EPSILON, rmsnorm->epsilon_tensor_uid())
               || !appendOptionalUid(key, config_tensor::BIAS, rmsnorm->bias_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* rmsnorm = node->attributes_as_RMSNormBackwardAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::RMSNORM_BACKWARD, K_NORM_PRIORITY);
            if(!appendUid(key, config_tensor::DY, rmsnorm->dy_tensor_uid())
               || !appendUid(key, config_tensor::X, rmsnorm->x_tensor_uid())
               || !appendUid(key, config_tensor::SCALE, rmsnorm->scale_tensor_uid())
               || !appendUid(key, config_tensor::INV_RMS, rmsnorm->inv_rms_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* reduction = node->attributes_as_ReductionAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::REDUCTION, K_REDUCTION_PRIORITY);
            key.addCriterion(config_criterion::REDUCTION_MODE,
                             static_cast<int64_t>(reduction->mode()));
            if(!appendUid(key, config_tensor::INPUT, reduction->in_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* resample = node->attributes_as_ResampleFwdAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::RESAMPLE_FWD, K_RESAMPLE_PRIORITY);
            key.addCriterion(config_criterion::RESAMPLE_MODE,
                             static_cast<int64_t>(resample->resample_mode()));
            key.addCriterion(config_criterion::PADDING_MODE,
                             static_cast<int64_t>(resample->padding_mode()));
            if(!appendUid(key, config_tensor::X, resample->x_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        if(const auto* pointwise = node->attributes_as_PointwiseAttributes())
        {
            BackendAutotuneConfigMatchKey key(config_op::POINTWISE, K_POINTWISE_PRIORITY);
            key.addCriterion(config_criterion::POINTWISE_MODE,
                             static_cast<int64_t>(pointwise->operation()));
            if(!appendUid(key, config_tensor::IN_0, pointwise->in_0_tensor_uid())
               || !appendOptionalUid(key, config_tensor::IN_1, pointwise->in_1_tensor_uid())
               || !appendOptionalUid(key, config_tensor::IN_2, pointwise->in_2_tensor_uid()))
            {
                return std::nullopt;
            }
            return key;
        }
        return std::nullopt;
    }

private:
    const TensorDimsStrides* tensorFor(int64_t uid) const
    {
        const auto it = _tensorIndex.find(uid);
        return it == _tensorIndex.end() ? nullptr : &it->second;
    }

    bool appendUid(BackendAutotuneConfigMatchKey& key, std::string_view tensorId, int64_t uid) const
    {
        const auto* tensor = tensorFor(uid);
        if(tensor == nullptr)
        {
            return false;
        }
        key.appendTensor(tensorId, *tensor);
        return true;
    }

    bool appendOptionalUid(BackendAutotuneConfigMatchKey& key,
                           std::string_view tensorId,
                           const std::optional<int64_t>& uid) const
    {
        if(!uid.has_value())
        {
            return true;
        }
        return appendUid(key, tensorId, *uid);
    }

    const std::unordered_map<int64_t, TensorDimsStrides>& _tensorIndex;
};

std::optional<int64_t>
    matchOverrideConfig(const EngineOverrideConfig& config,
                        const Graph* graph,
                        const std::unordered_map<int64_t, TensorDimsStrides>& tensorIndex)
{
    const auto* nodes = graph->nodes();
    if(nodes == nullptr)
    {
        return std::nullopt;
    }

    const BackendAutotuneConfigMatchKeyBuilder keyBuilder(tensorIndex);

    std::vector<BackendAutotuneConfigMatchKey> bestKeys;
    int bestPriority = 0;
    for(const auto* node : *nodes)
    {
        if(node == nullptr)
        {
            continue;
        }
        auto candidate = keyBuilder.buildKeyForNode(node);
        if(!candidate.has_value())
        {
            continue;
        }
        if(candidate->priority > bestPriority)
        {
            bestPriority = candidate->priority;
            bestKeys.clear();
        }
        if(candidate->priority == bestPriority)
        {
            bestKeys.push_back(std::move(*candidate));
        }
    }

    for(const auto& key : bestKeys)
    {
        auto match = config.matchOperation(key.op, key.criteria, key.tensors);
        if(match.has_value())
        {
            return match;
        }
    }
    return std::nullopt;
}

/// Validate the buffer and return the typed Graph root, or nullptr on failure.
const Graph* parseGraphBuffer(const std::vector<uint8_t>& buffer)
{
    if(buffer.empty())
    {
        return nullptr;
    }
    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if(!verifier.VerifyBuffer<Graph>())
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_WARN, "policyFinalize: invalid serialized graph buffer");
        return nullptr;
    }
    return hipdnn_flatbuffers_sdk::data_objects::GetGraph(buffer.data());
}

/// Reorder @p candidates so @p preferredEngineId comes first, preserving the
/// relative order of the rest. Returns nullopt if the preferred id is not in
/// the candidate list.
std::optional<std::vector<int64_t>>
    reorderWithPreferredFirst(const std::vector<int64_t>& candidates, int64_t preferredEngineId)
{
    auto it = std::find(candidates.begin(), candidates.end(), preferredEngineId);
    if(it == candidates.end())
    {
        return std::nullopt;
    }
    std::vector<int64_t> reordered;
    reordered.reserve(candidates.size());
    reordered.push_back(preferredEngineId);
    for(const int64_t engineId : candidates)
    {
        if(engineId != preferredEngineId)
        {
            reordered.push_back(engineId);
        }
    }
    return reordered;
}

// ---- Per-handle / per-descriptor state -------------------------------------

struct Handle
{
    std::vector<uint8_t> devicePropertiesBuffer;
    bool devicePropertiesSet = false;
};

struct PolicyDescriptor
{
    Handle* handle = nullptr;
    std::vector<int64_t> candidateEngineIds;
    std::vector<uint8_t> serializedGraph;
    std::vector<int64_t> sortedEngineIds;
    bool finalized = false;

    explicit PolicyDescriptor(Handle* h)
        : handle(h)
    {
    }
};

// ---- Base plugin metadata --------------------------------------------------

hipdnnPluginStatus_t getName(const char** name)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(name, CONFIG_BUILTIN_LOG, "getName: null output pointer");
    *name = PLUGIN_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getVersion(const char** version)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(version, CONFIG_BUILTIN_LOG, "getVersion: null output pointer");
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getApiVersion(const char** version)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        version, CONFIG_BUILTIN_LOG, "getApiVersion: null output pointer");
    *version = HIPDNN_HEURISTIC_API_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getType(hipdnnPluginType_t* type)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(type, CONFIG_BUILTIN_LOG, "getType: null output pointer");
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

void getLastErrorString(const char** errorStr)
{
    if(errorStr == nullptr)
    {
        return;
    }
    *errorStr = "No error information available";
}

// ---- Policy enumeration ----------------------------------------------------

hipdnnPluginStatus_t
    getAllPolicyIds(int64_t* policyIds, uint32_t maxPolicies, uint32_t* numPolicies)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numPolicies, CONFIG_BUILTIN_LOG, "getAllPolicyIds: null num_policies");

    constexpr uint32_t TOTAL_POLICIES = 1;
    *numPolicies = TOTAL_POLICIES;
    if(policyIds == nullptr || maxPolicies == 0)
    {
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    if(maxPolicies < TOTAL_POLICIES)
    {
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    policyIds[0] = policyId();
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t getPolicyName(int64_t id, const char** name)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(name, CONFIG_BUILTIN_LOG, "getPolicyName: null output pointer");
    if(id != policyId())
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "getPolicyName: unknown policy ID");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = POLICY_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Handle lifecycle ------------------------------------------------------

hipdnnPluginStatus_t handleCreate(hipdnnHeuristicHandle_t* outHandle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outHandle, CONFIG_BUILTIN_LOG, "handleCreate: null output pointer");
    try
    {
        auto h = std::make_unique<Handle>();
        *outHandle = reinterpret_cast<hipdnnHeuristicHandle_t>(h.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "handleCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t handleDestroy(hipdnnHeuristicHandle_t handle)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(handle, CONFIG_BUILTIN_LOG, "handleDestroy: null handle");
    delete reinterpret_cast<Handle*>(handle);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t handleSetDeviceProperties(hipdnnHeuristicHandle_t handle,
                                               const hipdnnPluginConstData_t* devicePropsSerialized)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        handle, CONFIG_BUILTIN_LOG, "handleSetDeviceProperties: null handle");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(devicePropsSerialized,
                                     true,
                                     CONFIG_BUILTIN_LOG,
                                     "handleSetDeviceProperties: invalid buffer");
    try
    {
        auto* h = reinterpret_cast<Handle*>(handle);
        const auto* data = reinterpret_cast<const uint8_t*>(devicePropsSerialized->ptr);
        h->devicePropertiesBuffer.assign(data, data + devicePropsSerialized->size);
        h->devicePropertiesSet = true;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "handleSetDeviceProperties failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Policy descriptor lifecycle ------------------------------------------

hipdnnPluginStatus_t policyDescriptorCreate(hipdnnHeuristicHandle_t pluginHandle,
                                            int64_t id,
                                            hipdnnHeuristicPolicyDescriptor_t* outDesc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        pluginHandle, CONFIG_BUILTIN_LOG, "policyDescriptorCreate: null handle");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outDesc, CONFIG_BUILTIN_LOG, "policyDescriptorCreate: null output pointer");
    if(id != policyId())
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policyDescriptorCreate: unknown policy ID");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    try
    {
        auto desc = std::make_unique<PolicyDescriptor>(reinterpret_cast<Handle*>(pluginHandle));
        *outDesc = reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(desc.release());
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policyDescriptorCreate failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policyDescriptorDestroy(hipdnnHeuristicPolicyDescriptor_t desc)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, CONFIG_BUILTIN_LOG, "policyDescriptorDestroy: null descriptor");
    delete reinterpret_cast<PolicyDescriptor*>(desc);
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

// ---- Policy inputs ---------------------------------------------------------

hipdnnPluginStatus_t policySetEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                        const int64_t* engineIds,
                                        size_t engineIdCount)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_BUILTIN_LOG, "policySetEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_ARRAY(engineIds,
                                engineIdCount,
                                CONFIG_BUILTIN_LOG,
                                "policySetEngineIds: null engine_ids with count > 0");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        d->candidateEngineIds.assign(engineIds, engineIds + engineIdCount);
        d->finalized = false;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policySetEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policySetSerializedGraph(hipdnnHeuristicPolicyDescriptor_t desc,
                                              const hipdnnPluginConstData_t* serializedGraph)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, CONFIG_BUILTIN_LOG, "policySetSerializedGraph: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_CONST_DATA(serializedGraph,
                                     false,
                                     CONFIG_BUILTIN_LOG,
                                     "policySetSerializedGraph: invalid graph buffer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        const auto* bytes = reinterpret_cast<const uint8_t*>(serializedGraph->ptr);
        if(bytes == nullptr || serializedGraph->size == 0)
        {
            d->serializedGraph.clear();
        }
        else
        {
            d->serializedGraph.assign(bytes, bytes + serializedGraph->size);
        }
        d->finalized = false;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policySetSerializedGraph failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

// ---- Selection -------------------------------------------------------------

hipdnnPluginStatus_t policyFinalize(hipdnnHeuristicPolicyDescriptor_t desc, int32_t* outApplied)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(desc, CONFIG_BUILTIN_LOG, "policyFinalize: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        outApplied, CONFIG_BUILTIN_LOG, "policyFinalize: null output pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        *outApplied = 0;

        if(d->candidateEngineIds.empty())
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO, "policyFinalize: no candidate engines; declining");
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        const auto config = EngineOverrideConfig::loadFromEnv();
        if(!config.has_value())
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO,
                               "policyFinalize: HIPDNN_HEUR_CONFIG_PATH unset or "
                               "unreadable; declining");
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        const Graph* graph = parseGraphBuffer(d->serializedGraph);
        if(graph == nullptr)
        {
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        const auto preferredEngineId
            = matchOverrideConfig(*config, graph, indexTensorsByUid(graph));
        if(!preferredEngineId.has_value())
        {
            CONFIG_BUILTIN_LOG(
                HIPDNN_SEV_INFO,
                "policyFinalize: no rule matched any supported primary node; declining");
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        auto reordered = reorderWithPreferredFirst(d->candidateEngineIds, *preferredEngineId);
        if(!reordered.has_value())
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO,
                               "policyFinalize: matched engine 0x%llx not in candidates; declining",
                               static_cast<unsigned long long>(*preferredEngineId));
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }

        CONFIG_BUILTIN_LOG(HIPDNN_SEV_INFO,
                           "policyFinalize: reordered %zu engines with preferred 0x%llx first",
                           reordered->size(),
                           static_cast<unsigned long long>(*preferredEngineId));
        d->sortedEngineIds = std::move(*reordered);
        d->finalized = true;
        *outApplied = 1;
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policyFinalize failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

hipdnnPluginStatus_t policyGetSortedEngineIds(hipdnnHeuristicPolicyDescriptor_t desc,
                                              int64_t* engineIds,
                                              size_t* numEngines)
{
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        desc, CONFIG_BUILTIN_LOG, "policyGetSortedEngineIds: null descriptor");
    HIPDNN_PLUGIN_REQUIRE_NOT_NULL(
        numEngines, CONFIG_BUILTIN_LOG, "policyGetSortedEngineIds: null num_engines pointer");
    try
    {
        auto* d = reinterpret_cast<PolicyDescriptor*>(desc);
        if(!d->finalized)
        {
            CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR,
                               "policyGetSortedEngineIds: descriptor not finalized");
            return HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED;
        }
        if(engineIds == nullptr)
        {
            *numEngines = d->sortedEngineIds.size();
            return HIPDNN_PLUGIN_STATUS_SUCCESS;
        }
        *numEngines = std::min(*numEngines, d->sortedEngineIds.size());
        std::copy_n(d->sortedEngineIds.begin(), *numEngines, engineIds);
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        CONFIG_BUILTIN_LOG(HIPDNN_SEV_ERROR, "policyGetSortedEngineIds failed: %s", e.what());
        return HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR;
    }
}

} // namespace

hipdnn_backend::plugin::HeuristicPluginFunctionTable populateFunctionTable()
{
    hipdnn_backend::plugin::HeuristicPluginFunctionTable funcs{};
    funcs.getName = &getName;
    funcs.getVersion = &getVersion;
    funcs.getApiVersion = &getApiVersion;
    funcs.getType = &getType;
    funcs.setLoggingCallback = &setLoggingCallback;
    funcs.setLogLevel = &setLogLevel;
    funcs.getLastErrorString = &getLastErrorString;
    funcs.getAllPolicyIds = &getAllPolicyIds;
    funcs.getPolicyName = &getPolicyName;
    funcs.handleCreate = &handleCreate;
    funcs.handleDestroy = &handleDestroy;
    funcs.handleSetDeviceProperties = &handleSetDeviceProperties;
    funcs.policyDescriptorCreate = &policyDescriptorCreate;
    funcs.policyDescriptorDestroy = &policyDescriptorDestroy;
    funcs.policySetEngineIds = &policySetEngineIds;
    funcs.policySetSerializedGraph = &policySetSerializedGraph;
    funcs.policyFinalize = &policyFinalize;
    funcs.policyGetSortedEngineIds = &policyGetSortedEngineIds;
    return funcs;
}

} // namespace hipdnn_backend::heuristics::config
