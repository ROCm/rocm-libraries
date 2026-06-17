// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

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
                         std::string criterionName,
                         const std::optional<T>& criterionValue)
{
    if(!criterionValue.has_value())
    {
        return false;
    }
    key.criteria.emplace_back(std::move(criterionName), static_cast<int64_t>(*criterionValue));
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
        key.opName = "conv_fprop";
        if(!appendRequiredMatchTensor(key, "x_tensor_uid", conv.attributes.get_x())
           || !appendRequiredMatchTensor(key, "w_tensor_uid", conv.attributes.get_w()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{70, std::move(key)};
    }
    case graph::NodeType::CONVOLUTION_DGRAD:
    {
        const auto& conv = static_cast<const graph::ConvolutionDgradNode&>(node);
        key.opName = "conv_dgrad";
        if(!appendRequiredMatchTensor(key, "dy_tensor_uid", conv.attributes.get_dy())
           || !appendRequiredMatchTensor(key, "w_tensor_uid", conv.attributes.get_w()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{70, std::move(key)};
    }
    case graph::NodeType::CONVOLUTION_WGRAD:
    {
        const auto& conv = static_cast<const graph::ConvolutionWgradNode&>(node);
        key.opName = "conv_wgrad";
        if(!appendRequiredMatchTensor(key, "x_tensor_uid", conv.attributes.get_x())
           || !appendRequiredMatchTensor(key, "dy_tensor_uid", conv.attributes.get_dy()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{70, std::move(key)};
    }
#ifdef HIPDNN_ENABLE_SDPA
    case graph::NodeType::SDPA_FWD:
    {
        const auto& sdpa = static_cast<const graph::SdpaFwdNode&>(node);
        key.opName = "sdpa_fwd";
        if(!appendRequiredMatchTensor(key, "q_tensor_uid", sdpa.attributes.get_q())
           || !appendRequiredMatchTensor(key, "k_tensor_uid", sdpa.attributes.get_k())
           || !appendRequiredMatchTensor(key, "v_tensor_uid", sdpa.attributes.get_v()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, "scale_tensor_uid", sdpa.attributes.get_attn_scale());
        appendOptionalMatchTensor(key, "attn_mask_tensor_uid", sdpa.attributes.get_bias());
        appendOptionalMatchTensor(key, "seq_len_q_tensor_uid", sdpa.attributes.get_seq_len_q());
        appendOptionalMatchTensor(key, "seq_len_kv_tensor_uid", sdpa.attributes.get_seq_len_kv());
        appendOptionalMatchTensor(key, "seed_tensor_uid", sdpa.attributes.get_seed());
        appendOptionalMatchTensor(key, "offset_tensor_uid", sdpa.attributes.get_offset());
        appendOptionalMatchTensor(
            key, "dropout_mask_tensor_uid", sdpa.attributes.get_dropout_mask());
        appendOptionalMatchTensor(
            key, "dropout_scale_tensor_uid", sdpa.attributes.get_dropout_scale());
        appendOptionalMatchTensor(
            key, "page_table_k_tensor_uid", sdpa.attributes.get_page_table_k());
        appendOptionalMatchTensor(
            key, "page_table_v_tensor_uid", sdpa.attributes.get_page_table_v());
        appendOptionalMatchTensor(key, "block_mask_tensor_uid", sdpa.attributes.get_block_mask());
        appendOptionalMatchTensor(key, "sink_token_tensor_uid", sdpa.attributes.get_sink_token());
        appendOptionalMatchTensor(key, "descale_q_tensor_uid", sdpa.attributes.get_descale_q());
        appendOptionalMatchTensor(key, "descale_k_tensor_uid", sdpa.attributes.get_descale_k());
        appendOptionalMatchTensor(key, "descale_v_tensor_uid", sdpa.attributes.get_descale_v());
        appendOptionalMatchTensor(key, "descale_s_tensor_uid", sdpa.attributes.get_descale_s());
        appendOptionalMatchTensor(key, "scale_s_tensor_uid", sdpa.attributes.get_scale_s());
        appendOptionalMatchTensor(key, "scale_o_tensor_uid", sdpa.attributes.get_scale_o());
        return PrioritizedAutotuneConfigMatchKey{60, std::move(key)};
    }
    case graph::NodeType::SDPA_BWD:
    {
        const auto& sdpa = static_cast<const graph::SdpaBwdNode&>(node);
        key.opName = "sdpa_bwd";
        if(!appendRequiredMatchTensor(key, "q_tensor_uid", sdpa.attributes.get_q())
           || !appendRequiredMatchTensor(key, "k_tensor_uid", sdpa.attributes.get_k())
           || !appendRequiredMatchTensor(key, "v_tensor_uid", sdpa.attributes.get_v())
           || !appendRequiredMatchTensor(key, "o_tensor_uid", sdpa.attributes.get_o())
           || !appendRequiredMatchTensor(key, "do_tensor_uid", sdpa.attributes.get_do())
           || !appendRequiredMatchTensor(key, "stats_tensor_uid", sdpa.attributes.get_stats()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, "scale_tensor_uid", sdpa.attributes.get_attn_scale());
        appendOptionalMatchTensor(key, "attn_mask_tensor_uid", sdpa.attributes.get_bias());
        appendOptionalMatchTensor(key, "seq_len_q_tensor_uid", sdpa.attributes.get_seq_len_q());
        appendOptionalMatchTensor(key, "seq_len_kv_tensor_uid", sdpa.attributes.get_seq_len_kv());
        appendOptionalMatchTensor(key, "seed_tensor_uid", sdpa.attributes.get_seed());
        appendOptionalMatchTensor(key, "offset_tensor_uid", sdpa.attributes.get_offset());
        appendOptionalMatchTensor(
            key, "dropout_mask_tensor_uid", sdpa.attributes.get_dropout_mask());
        appendOptionalMatchTensor(
            key, "dropout_scale_tensor_uid", sdpa.attributes.get_dropout_scale());
        appendOptionalMatchTensor(
            key, "dropout_scale_inv_tensor_uid", sdpa.attributes.get_dropout_scale_inv());
        return PrioritizedAutotuneConfigMatchKey{60, std::move(key)};
    }
#endif
    case graph::NodeType::MATMUL:
    {
        const auto& matmul = static_cast<const graph::MatmulNode&>(node);
        key.opName = "matmul";
        if(!appendRequiredMatchTensor(key, "a_tensor_uid", matmul.attributes.get_a())
           || !appendRequiredMatchTensor(key, "b_tensor_uid", matmul.attributes.get_b()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{50, std::move(key)};
    }
    case graph::NodeType::BATCHNORM:
    {
        const auto& batchnorm = static_cast<const graph::BatchnormNode&>(node);
        key.opName = "batchnorm_training";
        if(!appendRequiredMatchTensor(key, "x_tensor_uid", batchnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, "scale_tensor_uid", batchnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(key, "bias_tensor_uid", batchnorm.attributes.get_bias())
           || !appendRequiredMatchTensor(
               key, "epsilon_tensor_uid", batchnorm.attributes.get_epsilon()))
        {
            return std::nullopt;
        }
        for(const auto& peerStat : batchnorm.attributes.get_peer_stats())
        {
            appendOptionalMatchTensor(key, "peer_stats_tensor_uid", peerStat);
        }
        appendOptionalMatchTensor(
            key, "prev_running_mean_tensor_uid", batchnorm.attributes.get_prev_running_mean());
        appendOptionalMatchTensor(key,
                                  "prev_running_variance_tensor_uid",
                                  batchnorm.attributes.get_prev_running_variance());
        appendOptionalMatchTensor(key, "momentum_tensor_uid", batchnorm.attributes.get_momentum());
        return PrioritizedAutotuneConfigMatchKey{40, std::move(key)};
    }
    case graph::NodeType::BATCHNORM_INFERENCE:
    {
        const auto& batchnorm = static_cast<const graph::BatchnormInferenceNode&>(node);
        key.opName = "batchnorm_inference";
        if(!appendRequiredMatchTensor(key, "x_tensor_uid", batchnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, "mean_tensor_uid", batchnorm.attributes.get_mean())
           || !appendRequiredMatchTensor(
               key, "inv_variance_tensor_uid", batchnorm.attributes.get_inv_variance())
           || !appendRequiredMatchTensor(key, "scale_tensor_uid", batchnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(key, "bias_tensor_uid", batchnorm.attributes.get_bias()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{40, std::move(key)};
    }
    case graph::NodeType::BATCHNORM_INFERENCE_VARIANCE_EXT:
    {
        const auto& batchnorm = static_cast<const graph::BatchnormInferenceNodeVarianceExt&>(node);
        key.opName = "batchnorm_inference_variance_ext";
        if(!appendRequiredMatchTensor(key, "x_tensor_uid", batchnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, "mean_tensor_uid", batchnorm.attributes.get_mean())
           || !appendRequiredMatchTensor(
               key, "variance_tensor_uid", batchnorm.attributes.get_variance())
           || !appendRequiredMatchTensor(key, "scale_tensor_uid", batchnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(key, "bias_tensor_uid", batchnorm.attributes.get_bias())
           || !appendRequiredMatchTensor(
               key, "epsilon_tensor_uid", batchnorm.attributes.get_epsilon()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{40, std::move(key)};
    }
    case graph::NodeType::BATCHNORM_BACKWARD:
    {
        const auto& batchnorm = static_cast<const graph::BatchnormBackwardNode&>(node);
        key.opName = "batchnorm_backward";
        if(!appendRequiredMatchTensor(key, "dy_tensor_uid", batchnorm.attributes.get_dy())
           || !appendRequiredMatchTensor(key, "x_tensor_uid", batchnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, "scale_tensor_uid", batchnorm.attributes.get_scale()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, "mean_tensor_uid", batchnorm.attributes.get_mean());
        appendOptionalMatchTensor(
            key, "inv_variance_tensor_uid", batchnorm.attributes.get_inv_variance());
        for(const auto& peerStat : batchnorm.attributes.get_peer_stats())
        {
            appendOptionalMatchTensor(key, "peer_stats_tensor_uid", peerStat);
        }
        return PrioritizedAutotuneConfigMatchKey{40, std::move(key)};
    }
    case graph::NodeType::LAYER_NORM:
    {
        const auto& layernorm = static_cast<const graph::LayerNormNode&>(node);
        key.opName = "layernorm";
        if(!addCriterion(key,
                         "norm_fwd_phase",
                         toBackendNormFwdPhase(layernorm.attributes.get_forward_phase()))
           || !appendRequiredMatchTensor(key, "x_tensor_uid", layernorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, "scale_tensor_uid", layernorm.attributes.get_scale())
           || !appendRequiredMatchTensor(key, "bias_tensor_uid", layernorm.attributes.get_bias())
           || !appendRequiredMatchTensor(
               key, "epsilon_tensor_uid", layernorm.attributes.get_epsilon()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{30, std::move(key)};
    }
    case graph::NodeType::RMS_NORM:
    {
        const auto& rmsnorm = static_cast<const graph::RMSNormNode&>(node);
        key.opName = "rmsnorm";
        if(!addCriterion(
               key, "norm_fwd_phase", toBackendNormFwdPhase(rmsnorm.attributes.get_forward_phase()))
           || !appendRequiredMatchTensor(key, "x_tensor_uid", rmsnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, "scale_tensor_uid", rmsnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(
               key, "epsilon_tensor_uid", rmsnorm.attributes.get_epsilon()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, "bias_tensor_uid", rmsnorm.attributes.get_bias());
        return PrioritizedAutotuneConfigMatchKey{30, std::move(key)};
    }
    case graph::NodeType::RMS_NORM_BACKWARD:
    {
        const auto& rmsnorm = static_cast<const graph::RMSNormBackwardNode&>(node);
        key.opName = "rmsnorm_backward";
        if(!appendRequiredMatchTensor(key, "dy_tensor_uid", rmsnorm.attributes.get_dy())
           || !appendRequiredMatchTensor(key, "x_tensor_uid", rmsnorm.attributes.get_x())
           || !appendRequiredMatchTensor(key, "scale_tensor_uid", rmsnorm.attributes.get_scale())
           || !appendRequiredMatchTensor(
               key, "inv_rms_tensor_uid", rmsnorm.attributes.get_inv_rms()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{30, std::move(key)};
    }
    case graph::NodeType::REDUCTION:
    {
        const auto& reduction = static_cast<const graph::ReductionNode&>(node);
        const auto mode = reduction.attributes.get_mode();
        key.opName = "reduction";
        if(!mode.has_value() || !addCriterion(key, "reduction_mode", toBackendReductionMode(*mode))
           || !appendRequiredMatchTensor(key, "in_tensor_uid", reduction.attributes.get_x()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{20, std::move(key)};
    }
    case graph::NodeType::RESAMPLE_FWD:
    {
        const auto& resample = static_cast<const graph::ResampleFwdNode&>(node);
        const auto paddingMode = toBackendPaddingMode(resample.attributes.get_padding_mode());
        key.opName = "resample_fwd";
        if(!addCriterion(
               key, "resample_mode", toBackendResampleMode(resample.attributes.get_resample_mode()))
           || !addCriterion(key,
                            "padding_mode",
                            paddingMode.has_value() ? std::optional<int64_t>(*paddingMode)
                                                    : std::optional<int64_t>(0))
           || !appendRequiredMatchTensor(key, "x_tensor_uid", resample.attributes.get_x()))
        {
            return std::nullopt;
        }
        return PrioritizedAutotuneConfigMatchKey{20, std::move(key)};
    }
    case graph::NodeType::POINTWISE:
    {
        const auto& pointwise = static_cast<const graph::PointwiseNode&>(node);
        key.opName = "pointwise";
        if(!addCriterion(
               key, "pointwise_mode", toBackendPointwiseMode(pointwise.attributes.get_mode()))
           || !appendRequiredMatchTensor(
               key, "in_0_tensor_uid", pointwise.attributes.get_input_0()))
        {
            return std::nullopt;
        }
        appendOptionalMatchTensor(key, "in_1_tensor_uid", pointwise.attributes.get_input_1());
        appendOptionalMatchTensor(key, "in_2_tensor_uid", pointwise.attributes.get_input_2());
        return PrioritizedAutotuneConfigMatchKey{10, std::move(key)};
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
        auto candidate = getAutotuneConfigMatchKeyForNode(*node);
        if(candidate.has_value() && candidate->priority > bestPriority)
        {
            bestPriority = candidate->priority;
            bestKey = std::move(candidate->key);
        }
    }

    return bestKey;
}

inline std::optional<AutotuneConfigMatchKey> getAutotuneConfigMatchKey(const graph::INode& root)
{
    std::optional<AutotuneConfigMatchKey> bestKey;
    int bestPriority = 0;

    root.visit([&](const graph::INode& node) {
        auto candidate = getAutotuneConfigMatchKeyForNode(node);
        if(candidate.has_value() && candidate->priority > bestPriority)
        {
            bestPriority = candidate->priority;
            bestKey = std::move(candidate->key);
        }
    });

    return bestKey;
}

} // namespace hipdnn_frontend::detail
