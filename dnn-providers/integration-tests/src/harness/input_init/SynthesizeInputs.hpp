// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input_init/SynthesisConfig.hpp"

namespace hipdnn_integration_tests
{

// Pre-allocated input tensors keyed by uid, handed to the fill function.
using InputTensorMap
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

struct SynthesisResult
{
    bool filled = false;
    std::string reason;

    static SynthesisResult ok()
    {
        return {true, {}};
    }
    static SynthesisResult unsupported(std::string why)
    {
        return {false, std::move(why)};
    }
};

// ── Fill dispatch ───────────────────────────────────────────────────────────

enum class FillStatus
{
    FILLED,
    UNHANDLED
};

inline FillStatus
    fill(hipdnn_data_sdk::utilities::ITensor& tensor, const TensorInit& init, unsigned int seed)
{
    switch(init.kind)
    {
    case TensorInit::Kind::FREE:
        tensor.fillTensorWithRandomValues(init.lo, init.hi, seed);
        return FillStatus::FILLED;
    case TensorInit::Kind::FIXED:
        tensor.fillTensorWithValue(init.value);
        return FillStatus::FILLED;
    case TensorInit::Kind::STRUCTURED:
    case TensorInit::Kind::DERIVED:
    default:
        return FillStatus::UNHANDLED;
    }
}

// ── Per-op init defaults ────────────────────────────────────────────────────
// Each function sets defaults for one node type via config.setDefault().
// setDefault uses try_emplace — if the test already set() a uid, the default
// is silently skipped.

// ── Batchnorm ────────────────────────────────────────────────────────────────

inline void
    setBatchnormInferenceInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                      SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->mean_tensor_uid(), TensorInit::free(-0.1f, 0.1f));
    config.setDefault(a->inv_variance_tensor_uid(), TensorInit::free(0.5f, 1.5f));
}

inline void setBatchnormInferenceVarianceInitDefaults(
    const hipdnn_flatbuffers_sdk::data_objects::Node& node, SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->mean_tensor_uid(), TensorInit::free(-0.1f, 0.1f));
    config.setDefault(a->variance_tensor_uid(), TensorInit::free(0.5f, 1.5f));
    config.setDefault(a->epsilon_tensor_uid(), TensorInit::fixed(1e-5f));
}

inline void setBatchnormTrainingInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                             SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BatchnormAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->epsilon_tensor_uid(), TensorInit::fixed(1e-5f));
    config.setDefault(a->prev_running_mean_tensor_uid(), TensorInit::free(-0.1f, 0.1f));
    config.setDefault(a->prev_running_variance_tensor_uid(), TensorInit::free(0.5f, 1.5f));
    config.setDefault(a->momentum_tensor_uid(), TensorInit::free(0.0f, 1.0f));

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            config.setDefault(uid, TensorInit::structured());
        }
    }
}

inline void setBatchnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                             SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BatchnormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->mean_tensor_uid(), TensorInit::free(-0.1f, 0.1f));
    config.setDefault(a->inv_variance_tensor_uid(), TensorInit::free(0.5f, 1.5f));

    if(a->peer_stats_tensor_uid() != nullptr)
    {
        for(const int64_t uid : *a->peer_stats_tensor_uid())
        {
            config.setDefault(uid, TensorInit::structured());
        }
    }
}

// ── LayerNorm ────────────────────────────────────────────────────────────────

inline void setLayernormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                     SynthesisConfig& config)
{
    const auto* a = node.attributes_as_LayernormAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->epsilon_tensor_uid(), TensorInit::fixed(1e-5f));
}

inline void setLayernormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                             SynthesisConfig& config)
{
    const auto* a = node.attributes_as_LayernormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->mean_tensor_uid(), TensorInit::derived());
    config.setDefault(a->inv_variance_tensor_uid(), TensorInit::derived());
    config.setDefault(a->epsilon_tensor_uid(), TensorInit::fixed(1e-5f));
}

// ── RMSNorm ──────────────────────────────────────────────────────────────────

inline void setRmsnormInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                   SynthesisConfig& config)
{
    const auto* a = node.attributes_as_RMSNormAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->epsilon_tensor_uid(), TensorInit::fixed(1e-5f));
}

inline void setRmsnormBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                           SynthesisConfig& config)
{
    const auto* a = node.attributes_as_RMSNormBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->inv_rms_tensor_uid(), TensorInit::derived());
}

// ── Block-scale quantization ─────────────────────────────────────────────────

inline void
    setBlockScaleDequantizeInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                        SynthesisConfig& config)
{
    const auto* a = node.attributes_as_BlockScaleDequantizeAttributes();
    if(a == nullptr)
    {
        return;
    }
    config.setDefault(a->scale_tensor_uid(), TensorInit::structured());
}

// ── SDPA ─────────────────────────────────────────────────────────────────────

inline void setSdpaForwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                       SynthesisConfig& config)
{
    const auto* a = node.attributes_as_SdpaAttributes();
    if(a == nullptr)
    {
        return;
    }

    config.setDefault(a->scale_tensor_uid(), TensorInit::free(0.1f, 1.0f));

    config.setDefault(a->descale_q_tensor_uid(), TensorInit::structured());
    config.setDefault(a->descale_k_tensor_uid(), TensorInit::structured());
    config.setDefault(a->descale_v_tensor_uid(), TensorInit::structured());
    config.setDefault(a->descale_s_tensor_uid(), TensorInit::structured());
    config.setDefault(a->scale_s_tensor_uid(), TensorInit::structured());
    config.setDefault(a->scale_o_tensor_uid(), TensorInit::structured());

    config.setDefault(a->seq_len_q_tensor_uid(), TensorInit::structured());
    config.setDefault(a->seq_len_kv_tensor_uid(), TensorInit::structured());
    config.setDefault(a->page_table_k_tensor_uid(), TensorInit::structured());
    config.setDefault(a->page_table_v_tensor_uid(), TensorInit::structured());
    config.setDefault(a->block_mask_tensor_uid(), TensorInit::structured());
    config.setDefault(a->seed_tensor_uid(), TensorInit::structured());
    config.setDefault(a->offset_tensor_uid(), TensorInit::structured());
}

inline void setSdpaBackwardInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                                        SynthesisConfig& config)
{
    const auto* a = node.attributes_as_SdpaBackwardAttributes();
    if(a == nullptr)
    {
        return;
    }

    config.setDefault(a->scale_tensor_uid(), TensorInit::free(0.1f, 1.0f));
    config.setDefault(a->dropout_scale_tensor_uid(), TensorInit::free(0.1f, 1.0f));
    config.setDefault(a->dropout_scale_inv_tensor_uid(), TensorInit::free(0.1f, 1.0f));

    config.setDefault(a->o_tensor_uid(), TensorInit::derived());
    config.setDefault(a->stats_tensor_uid(), TensorInit::derived());

    config.setDefault(a->seq_len_q_tensor_uid(), TensorInit::structured());
    config.setDefault(a->seq_len_kv_tensor_uid(), TensorInit::structured());
    config.setDefault(a->seed_tensor_uid(), TensorInit::structured());
    config.setDefault(a->offset_tensor_uid(), TensorInit::structured());
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

inline void setInitDefaults(const hipdnn_flatbuffers_sdk::data_objects::Node& node,
                            SynthesisConfig& config)
{
    using NA = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

    switch(node.attributes_type())
    {
    case NA::BatchnormInferenceAttributes:
        setBatchnormInferenceInitDefaults(node, config);
        break;
    case NA::BatchnormInferenceAttributesVarianceExt:
        setBatchnormInferenceVarianceInitDefaults(node, config);
        break;
    case NA::BatchnormAttributes:
        setBatchnormTrainingInitDefaults(node, config);
        break;
    case NA::BatchnormBackwardAttributes:
        setBatchnormBackwardInitDefaults(node, config);
        break;
    case NA::LayernormAttributes:
        setLayernormInitDefaults(node, config);
        break;
    case NA::LayernormBackwardAttributes:
        setLayernormBackwardInitDefaults(node, config);
        break;
    case NA::RMSNormAttributes:
        setRmsnormInitDefaults(node, config);
        break;
    case NA::RMSNormBackwardAttributes:
        setRmsnormBackwardInitDefaults(node, config);
        break;
    case NA::BlockScaleDequantizeAttributes:
        setBlockScaleDequantizeInitDefaults(node, config);
        break;
    case NA::SdpaAttributes:
        setSdpaForwardInitDefaults(node, config);
        break;
    case NA::SdpaBackwardAttributes:
        setSdpaBackwardInitDefaults(node, config);
        break;
    default:
        break;
    }
}

// ── Free function: set defaults + fill ───────────────────────────────────────

inline void synthesizeInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                             InputTensorMap& inputs,
                             const std::vector<int64_t>& ownedUids,
                             SynthesisConfig& config)
{
    for(flatbuffers::uoffset_t i = 0; i < graph.nodes()->size(); ++i)
    {
        setInitDefaults(*graph.nodes()->Get(i), config);
    }

    std::mt19937 rng(config.getSeedEntropy());
    for(const int64_t uid : ownedUids)
    {
        unsigned int seed = config.resolveSeed(uid).value_or(static_cast<unsigned int>(rng()));
        fill(*inputs.at(uid), config.get(uid), seed);
    }
}

} // namespace hipdnn_integration_tests
