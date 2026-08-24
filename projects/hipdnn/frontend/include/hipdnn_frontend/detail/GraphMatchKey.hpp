// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_data_sdk/detail/AutotuneConfigNames.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNodeVarianceExt.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>
#include <hipdnn_frontend/node/LayerNormNode.hpp>
#include <hipdnn_frontend/node/MatmulNode.hpp>
#include <hipdnn_frontend/node/Node.hpp>
#include <hipdnn_frontend/node/NodeType.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_frontend/node/RMSNormBackwardNode.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>
#include <hipdnn_frontend/node/ReductionNode.hpp>
#include <hipdnn_frontend/node/ResampleFwdNode.hpp>
#ifdef HIPDNN_ENABLE_SDPA
#include <hipdnn_frontend/node/SdpaBwdNode.hpp>
#include <hipdnn_frontend/node/SdpaFwdNode.hpp>
#endif

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hipdnn_frontend::detail
{
namespace config_criterion = hipdnn_data_sdk::detail::autotune_config::criterion;
namespace config_op = hipdnn_data_sdk::detail::autotune_config::op;
namespace config_tensor = hipdnn_data_sdk::detail::autotune_config::tensor;

using AutotuneConfigCriteria = std::vector<std::pair<std::string, int64_t>>;

struct AutotuneConfigMatchTensor
{
    std::string_view tensorId;
    std::shared_ptr<graph::TensorAttributes> tensor;

    graph::TensorAttributes* operator->() const
    {
        return tensor.get();
    }
};

struct AutotuneConfigMatchKey
{
    std::string opName;
    AutotuneConfigCriteria criteria;
    std::vector<AutotuneConfigMatchTensor> tensors;
};

struct PrioritizedAutotuneConfigMatchKey
{
    int priority = 0;
    AutotuneConfigMatchKey key;
};

// Per-operation-class priorities used to pick the most specific match key when a
// graph contains multiple ops. Higher wins.
namespace match_priority
{
inline constexpr int CONVOLUTION = 70;
inline constexpr int SDPA = 60;
inline constexpr int MATMUL = 50;
inline constexpr int BATCHNORM = 40;
inline constexpr int NORM = 30;
inline constexpr int REDUCTION = 20;
inline constexpr int RESAMPLE = 20;
inline constexpr int POINTWISE = 10;
} // namespace match_priority

inline bool appendRequiredMatchTensor(AutotuneConfigMatchKey& key,
                                      std::string_view tensorId,
                                      const std::shared_ptr<graph::TensorAttributes>& tensor)
{
    if(!tensor)
    {
        return false;
    }
    key.tensors.push_back(AutotuneConfigMatchTensor{tensorId, tensor});
    return true;
}

inline void appendOptionalMatchTensor(AutotuneConfigMatchKey& key,
                                      std::string_view tensorId,
                                      const std::shared_ptr<graph::TensorAttributes>& tensor)
{
    if(tensor)
    {
        key.tensors.push_back(AutotuneConfigMatchTensor{tensorId, tensor});
    }
}

template <typename T>
inline bool addCriterion(AutotuneConfigMatchKey& key,
                         std::string_view criterionName,
                         const std::optional<T>& criterionValue)
{
    if(!criterionValue.has_value())
    {
        return false;
    }
    key.criteria.emplace_back(std::string(criterionName), static_cast<int64_t>(*criterionValue));
    return true;
}

inline std::optional<PrioritizedAutotuneConfigMatchKey>
    getAutotuneConfigMatchKeyForNode(const graph::INode& node)
{
    AutotuneConfigMatchKey key;

    switch(node.getNodeType())
    {
    case graph::NodeType::CONVOLUTION_FPROP:
    {
        const auto& conv = static_cast<const graph::ConvolutionFpropNode&>(node);
        key.opName = config_op::CONV_FPROP;
        if(!appendRequiredMatchTensor(key, config_tensor::X, conv.attributes.get_x())
           || !appendRequiredMatchTensor(key, config_tensor::W, conv.attributes.get_w()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::CONVOLUTION, std::move(key)};
    }
    case graph::NodeType::CONVOLUTION_DGRAD:
    {
        const auto& conv = static_cast<const graph::ConvolutionDgradNode&>(node);
        key.opName = config_op::CONV_DGRAD;
        if(!appendRequiredMatchTensor(key, config_tensor::DY, conv.attributes.get_dy())
           || !appendRequiredMatchTensor(key, config_tensor::W, conv.attributes.get_w()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::CONVOLUTION, std::move(key)};
    }
    case graph::NodeType::CONVOLUTION_WGRAD:
    {
        const auto& conv = static_cast<const graph::ConvolutionWgradNode&>(node);
        key.opName = config_op::CONV_WGRAD;
        if(!appendRequiredMatchTensor(key, config_tensor::X, conv.attributes.get_x())
           || !appendRequiredMatchTensor(key, config_tensor::DY, conv.attributes.get_dy()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::CONVOLUTION, std::move(key)};
    }
#ifdef HIPDNN_ENABLE_SDPA
    case graph::NodeType::SDPA_FWD:
    {
        const auto& sdpa = static_cast<const graph::SdpaFwdNode&>(node);
        key.opName = config_op::SDPA_FWD;
        if(!appendRequiredMatchTensor(key, config_tensor::Q, sdpa.attributes.get_q())
           || !appendRequiredMatchTensor(key, config_tensor::K, sdpa.attributes.get_k())
           || !appendRequiredMatchTensor(key, config_tensor::V, sdpa.attributes.get_v()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, config_tensor::SCALE, sdpa.attributes.get_attn_scale());
        appendOptionalMatchTensor(key, config_tensor::ATTN_MASK, sdpa.attributes.get_bias());
        appendOptionalMatchTensor(key, config_tensor::SEQ_LEN_Q, sdpa.attributes.get_seq_len_q());
        appendOptionalMatchTensor(key, config_tensor::SEQ_LEN_KV, sdpa.attributes.get_seq_len_kv());
        appendOptionalMatchTensor(key, config_tensor::SEED, sdpa.attributes.get_seed());
        appendOptionalMatchTensor(key, config_tensor::OFFSET, sdpa.attributes.get_offset());
        appendOptionalMatchTensor(
            key, config_tensor::DROPOUT_MASK, sdpa.attributes.get_dropout_mask());
        appendOptionalMatchTensor(
            key, config_tensor::DROPOUT_SCALE, sdpa.attributes.get_dropout_scale());
        appendOptionalMatchTensor(
            key, config_tensor::PAGE_TABLE_K, sdpa.attributes.get_page_table_k());
        appendOptionalMatchTensor(
            key, config_tensor::PAGE_TABLE_V, sdpa.attributes.get_page_table_v());
        appendOptionalMatchTensor(key, config_tensor::BLOCK_MASK, sdpa.attributes.get_block_mask());
        appendOptionalMatchTensor(key, config_tensor::SINK_TOKEN, sdpa.attributes.get_sink_token());
        appendOptionalMatchTensor(key, config_tensor::DESCALE_Q, sdpa.attributes.get_descale_q());
        appendOptionalMatchTensor(key, config_tensor::DESCALE_K, sdpa.attributes.get_descale_k());
        appendOptionalMatchTensor(key, config_tensor::DESCALE_V, sdpa.attributes.get_descale_v());
        appendOptionalMatchTensor(key, config_tensor::DESCALE_S, sdpa.attributes.get_descale_s());
        appendOptionalMatchTensor(key, config_tensor::SCALE_S, sdpa.attributes.get_scale_s());
        appendOptionalMatchTensor(key, config_tensor::SCALE_O, sdpa.attributes.get_scale_o());
        return PrioritizedAutotuneConfigMatchKey{match_priority::SDPA, std::move(key)};
    }
    case graph::NodeType::SDPA_BWD:
    {
        const auto& sdpa = static_cast<const graph::SdpaBwdNode&>(node);
        key.opName = config_op::SDPA_BWD;
        if(!appendRequiredMatchTensor(key, config_tensor::Q, sdpa.attributes.get_q())
           || !appendRequiredMatchTensor(key, config_tensor::K, sdpa.attributes.get_k())
           || !appendRequiredMatchTensor(key, config_tensor::V, sdpa.attributes.get_v())
           || !appendRequiredMatchTensor(key, config_tensor::O, sdpa.attributes.get_o())
           || !appendRequiredMatchTensor(key, config_tensor::DO, sdpa.attributes.get_do())
           || !appendRequiredMatchTensor(key, config_tensor::STATS, sdpa.attributes.get_stats()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, config_tensor::SCALE, sdpa.attributes.get_attn_scale());
        appendOptionalMatchTensor(key, config_tensor::ATTN_MASK, sdpa.attributes.get_bias());
        appendOptionalMatchTensor(key, config_tensor::SEQ_LEN_Q, sdpa.attributes.get_seq_len_q());
        appendOptionalMatchTensor(key, config_tensor::SEQ_LEN_KV, sdpa.attributes.get_seq_len_kv());
        appendOptionalMatchTensor(key, config_tensor::SEED, sdpa.attributes.get_seed());
        appendOptionalMatchTensor(key, config_tensor::OFFSET, sdpa.attributes.get_offset());
        appendOptionalMatchTensor(
            key, config_tensor::DROPOUT_MASK, sdpa.attributes.get_dropout_mask());
        appendOptionalMatchTensor(
            key, config_tensor::DROPOUT_SCALE, sdpa.attributes.get_dropout_scale());
        appendOptionalMatchTensor(
            key, config_tensor::DROPOUT_SCALE_INV, sdpa.attributes.get_dropout_scale_inv());
        return PrioritizedAutotuneConfigMatchKey{match_priority::SDPA, std::move(key)};
    }
#endif
    case graph::NodeType::MATMUL:
    {
        const auto& matmul = static_cast<const graph::MatmulNode&>(node);
        key.opName = config_op::MATMUL;
        if(!appendRequiredMatchTensor(key, config_tensor::A, matmul.attributes.get_a())
           || !appendRequiredMatchTensor(key, config_tensor::B, matmul.attributes.get_b()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::MATMUL, std::move(key)};
    }
    case graph::NodeType::BATCHNORM:
    {
        const auto& batchnorm = static_cast<const graph::BatchnormNode&>(node);
        key.opName = config_op::BATCHNORM_TRAINING;
        if(!appendRequiredMatchTensor(key, config_tensor::X, batchnorm.attributes.get_x())
           || !appendRequiredMatchTensor(
               key, config_tensor::SCALE, batchnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(key, config_tensor::BIAS, batchnorm.attributes.get_bias())
           || !appendRequiredMatchTensor(
               key, config_tensor::EPSILON, batchnorm.attributes.get_epsilon()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(
            key, config_tensor::PREV_RUNNING_MEAN, batchnorm.attributes.get_prev_running_mean());
        appendOptionalMatchTensor(key,
                                  config_tensor::PREV_RUNNING_VARIANCE,
                                  batchnorm.attributes.get_prev_running_variance());
        appendOptionalMatchTensor(
            key, config_tensor::MOMENTUM, batchnorm.attributes.get_momentum());
        return PrioritizedAutotuneConfigMatchKey{match_priority::BATCHNORM, std::move(key)};
    }
    case graph::NodeType::BATCHNORM_INFERENCE:
    {
        const auto& batchnorm = static_cast<const graph::BatchnormInferenceNode&>(node);
        key.opName = config_op::BATCHNORM_INFERENCE;
        if(!appendRequiredMatchTensor(key, config_tensor::X, batchnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, config_tensor::MEAN, batchnorm.attributes.get_mean())
           || !appendRequiredMatchTensor(
               key, config_tensor::INV_VARIANCE, batchnorm.attributes.get_inv_variance())
           || !appendRequiredMatchTensor(
               key, config_tensor::SCALE, batchnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(key, config_tensor::BIAS, batchnorm.attributes.get_bias()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::BATCHNORM, std::move(key)};
    }
    case graph::NodeType::BATCHNORM_INFERENCE_VARIANCE_EXT:
    {
        const auto& batchnorm = static_cast<const graph::BatchnormInferenceNodeVarianceExt&>(node);
        key.opName = config_op::BATCHNORM_INFERENCE_VARIANCE_EXT;
        if(!appendRequiredMatchTensor(key, config_tensor::X, batchnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, config_tensor::MEAN, batchnorm.attributes.get_mean())
           || !appendRequiredMatchTensor(
               key, config_tensor::VARIANCE, batchnorm.attributes.get_variance())
           || !appendRequiredMatchTensor(
               key, config_tensor::SCALE, batchnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(key, config_tensor::BIAS, batchnorm.attributes.get_bias())
           || !appendRequiredMatchTensor(
               key, config_tensor::EPSILON, batchnorm.attributes.get_epsilon()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::BATCHNORM, std::move(key)};
    }
    case graph::NodeType::BATCHNORM_BACKWARD:
    {
        const auto& batchnorm = static_cast<const graph::BatchnormBackwardNode&>(node);
        key.opName = config_op::BATCHNORM_BACKWARD;
        if(!appendRequiredMatchTensor(key, config_tensor::DY, batchnorm.attributes.get_dy())
           || !appendRequiredMatchTensor(key, config_tensor::X, batchnorm.attributes.get_x())
           || !appendRequiredMatchTensor(
               key, config_tensor::SCALE, batchnorm.attributes.get_scale()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, config_tensor::MEAN, batchnorm.attributes.get_mean());
        appendOptionalMatchTensor(
            key, config_tensor::INV_VARIANCE, batchnorm.attributes.get_inv_variance());
        return PrioritizedAutotuneConfigMatchKey{match_priority::BATCHNORM, std::move(key)};
    }
    case graph::NodeType::LAYER_NORM:
    {
        const auto& layernorm = static_cast<const graph::LayerNormNode&>(node);
        key.opName = config_op::LAYERNORM;
        if(!addCriterion(key,
                         config_criterion::NORM_FWD_PHASE,
                         toBackendNormFwdPhase(layernorm.attributes.get_forward_phase()))
           || !appendRequiredMatchTensor(key, config_tensor::X, layernorm.attributes.get_x())
           || !appendRequiredMatchTensor(
               key, config_tensor::SCALE, layernorm.attributes.get_scale())
           || !appendRequiredMatchTensor(key, config_tensor::BIAS, layernorm.attributes.get_bias())
           || !appendRequiredMatchTensor(
               key, config_tensor::EPSILON, layernorm.attributes.get_epsilon()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::NORM, std::move(key)};
    }
    case graph::NodeType::RMS_NORM:
    {
        const auto& rmsnorm = static_cast<const graph::RMSNormNode&>(node);
        key.opName = config_op::RMSNORM;
        if(!addCriterion(key,
                         config_criterion::NORM_FWD_PHASE,
                         toBackendNormFwdPhase(rmsnorm.attributes.get_forward_phase()))
           || !appendRequiredMatchTensor(key, config_tensor::X, rmsnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, config_tensor::SCALE, rmsnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(
               key, config_tensor::EPSILON, rmsnorm.attributes.get_epsilon()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, config_tensor::BIAS, rmsnorm.attributes.get_bias());
        return PrioritizedAutotuneConfigMatchKey{match_priority::NORM, std::move(key)};
    }
    case graph::NodeType::RMS_NORM_BACKWARD:
    {
        const auto& rmsnorm = static_cast<const graph::RMSNormBackwardNode&>(node);
        key.opName = config_op::RMSNORM_BACKWARD;
        if(!appendRequiredMatchTensor(key, config_tensor::DY, rmsnorm.attributes.get_dy())
           || !appendRequiredMatchTensor(key, config_tensor::X, rmsnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, config_tensor::SCALE, rmsnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(
               key, config_tensor::INV_RMS, rmsnorm.attributes.get_inv_rms()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::NORM, std::move(key)};
    }
    case graph::NodeType::REDUCTION:
    {
        const auto& reduction = static_cast<const graph::ReductionNode&>(node);
        const auto mode = reduction.attributes.get_mode();
        key.opName = config_op::REDUCTION;
        if(!mode.has_value()
           || !addCriterion(key, config_criterion::REDUCTION_MODE, toBackendReductionMode(*mode))
           || !appendRequiredMatchTensor(key, config_tensor::INPUT, reduction.attributes.get_x()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::REDUCTION, std::move(key)};
    }
    case graph::NodeType::RESAMPLE_FWD:
    {
        const auto& resample = static_cast<const graph::ResampleFwdNode&>(node);
        const auto paddingMode = toBackendPaddingMode(resample.attributes.get_padding_mode());
        key.opName = config_op::RESAMPLE_FWD;
        if(!addCriterion(key,
                         config_criterion::RESAMPLE_MODE,
                         toBackendResampleMode(resample.attributes.get_resample_mode()))
           || !addCriterion(key,
                            config_criterion::PADDING_MODE,
                            paddingMode.has_value() ? std::optional<int64_t>(*paddingMode)
                                                    : std::optional<int64_t>(0))
           || !appendRequiredMatchTensor(key, config_tensor::X, resample.attributes.get_x()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{match_priority::RESAMPLE, std::move(key)};
    }
    case graph::NodeType::POINTWISE:
    {
        const auto& pointwise = static_cast<const graph::PointwiseNode&>(node);
        key.opName = config_op::POINTWISE;
        if(!addCriterion(key,
                         config_criterion::POINTWISE_MODE,
                         toBackendPointwiseMode(pointwise.attributes.get_mode()))
           || !appendRequiredMatchTensor(
               key, config_tensor::IN_0, pointwise.attributes.get_input_0()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, config_tensor::IN_1, pointwise.attributes.get_input_1());
        appendOptionalMatchTensor(key, config_tensor::IN_2, pointwise.attributes.get_input_2());
        return PrioritizedAutotuneConfigMatchKey{match_priority::POINTWISE, std::move(key)};
    }
    case graph::NodeType::UNKNOWN:
    case graph::NodeType::BLOCK_SCALE_QUANTIZE:
    case graph::NodeType::BLOCK_SCALE_DEQUANTIZE:
    case graph::NodeType::CUSTOM_OP:
#ifndef HIPDNN_ENABLE_SDPA
    case graph::NodeType::SDPA_FWD:
    case graph::NodeType::SDPA_BWD:
#endif
    default:
        return std::nullopt;
    }

    return std::nullopt;
}

// Selects the highest-priority match key seen so far. If @p node produces a
// candidate whose priority beats bestPriority, bestKey/bestPriority are updated.
inline void considerNode(const graph::INode& node,
                         std::optional<AutotuneConfigMatchKey>& bestKey,
                         int& bestPriority)
{
    auto candidate = getAutotuneConfigMatchKeyForNode(node);
    if(candidate.has_value() && candidate->priority > bestPriority)
    {
        bestPriority = candidate->priority;
        bestKey = std::move(candidate->key);
    }
}

inline std::optional<AutotuneConfigMatchKey>
    getAutotuneConfigMatchKey(const std::vector<std::shared_ptr<graph::INode>>& nodes)
{
    std::optional<AutotuneConfigMatchKey> bestKey;
    int bestPriority = 0;

    for(const auto& node : nodes)
    {
        if(!node)
        {
            continue;
        }
        considerNode(*node, bestKey, bestPriority);
    }

    return bestKey;
}

inline std::optional<AutotuneConfigMatchKey> getAutotuneConfigMatchKey(const graph::INode& root)
{
    std::optional<AutotuneConfigMatchKey> bestKey;
    int bestPriority = 0;

    root.visit([&](const graph::INode& node) { considerNode(node, bestKey, bestPriority); });

    return bestKey;
}

} // namespace hipdnn_frontend::detail
