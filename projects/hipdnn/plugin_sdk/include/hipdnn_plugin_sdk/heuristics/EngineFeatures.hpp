// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/heuristics/DeviceFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>

namespace hipdnn_plugin_sdk::heuristics
{

/// @brief Reads the optional global workspace bound shared by prediction and selection.
inline std::optional<int64_t>
    workspaceLimit(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    if(!config.isValid() || !config.hasKnobSetting(WORKSPACE_SIZE_LIMIT_KNOB_NAME))
    {
        return std::nullopt;
    }
    const auto& setting = config.getKnobSettingByName(WORKSPACE_SIZE_LIMIT_KNOB_NAME);
    if(setting.valueType() != KnobValue::IntValue || setting.valueAs<IntValue>().value() < 0)
    {
        throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                    "Workspace limit must be a nonnegative integer");
    }
    return setting.valueAs<IntValue>().value();
}

namespace detail
{
using Graph = hipdnn_flatbuffers_sdk::data_objects::Graph;
using Tensor = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using Node = hipdnn_flatbuffers_sdk::data_objects::Node;

inline const Tensor* tensor(const Graph& graph, int64_t uid)
{
    if(graph.tensors() != nullptr)
    {
        for(const auto* value : *graph.tensors())
        {
            if(value != nullptr && value->uid() == uid)
            {
                return value;
            }
        }
    }
    return nullptr;
}

inline bool floatingPoint(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(type)
    {
    case DataType::FLOAT:
    case DataType::HALF:
    case DataType::BFLOAT16:
    case DataType::DOUBLE:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E8M0:
    case DataType::FP4_E2M1:
    case DataType::FP6_E2M3:
    case DataType::FP6_E3M2:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return true;
    default:
        return false;
    }
}

inline bool dimensions(const Tensor* value, size_t minimumRank)
{
    return value != nullptr && floatingPoint(value->data_type()) && value->dims() != nullptr
           && value->dims()->size() >= minimumRank
           && std::all_of(value->dims()->begin(), value->dims()->end(), [](int64_t extent) {
                  return extent > 0;
              });
}

inline double elements(const Tensor& value)
{
    double product = 1.0;
    for(const auto extent : *value.dims())
    {
        product *= static_cast<double>(extent);
    }
    return product;
}

/// Logical multiply-add work, using corpus_gen operation metadata conventions.
/// Unsupported operations or data-dependent work invalidate the whole graph count.
inline std::optional<double> logicalFlops(const Graph& graph, const Node& node)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    if(const auto* op = node.attributes_as_MatmulAttributes())
    {
        const auto* a = tensor(graph, op->a_tensor_uid());
        const auto* b = tensor(graph, op->b_tensor_uid());
        const auto* c = tensor(graph, op->c_tensor_uid());
        if(!dimensions(a, 2) || !dimensions(b, 2) || !dimensions(c, 2))
        {
            return std::nullopt;
        }
        const auto ar = a->dims()->size();
        const auto br = b->dims()->size();
        const auto cr = c->dims()->size();
        const auto k = a->dims()->Get(ar - 1);
        if(k != b->dims()->Get(br - 2) || a->dims()->Get(ar - 2) != c->dims()->Get(cr - 2)
           || b->dims()->Get(br - 1) != c->dims()->Get(cr - 1) || cr != std::max(ar, br))
        {
            return std::nullopt;
        }
        for(flatbuffers::uoffset_t i = 2; i < cr; ++i)
        {
            const auto ad = i < ar ? a->dims()->Get(ar - i - 1) : 1;
            const auto bd = i < br ? b->dims()->Get(br - i - 1) : 1;
            if((ad != bd && ad != 1 && bd != 1) || c->dims()->Get(cr - i - 1) != std::max(ad, bd))
            {
                return std::nullopt;
            }
        }
        return 2.0 * elements(*c) * static_cast<double>(k);
    }
    if(const auto* op = node.attributes_as_ConvolutionFwdAttributes())
    {
        const auto* x = tensor(graph, op->x_tensor_uid());
        const auto* w = tensor(graph, op->w_tensor_uid());
        const auto* y = tensor(graph, op->y_tensor_uid());
        if(!dimensions(x, 3) || !dimensions(w, 3) || !dimensions(y, 3)
           || x->dims()->size() != w->dims()->size() || x->dims()->size() != y->dims()->size()
           || x->dims()->Get(0) != y->dims()->Get(0) || w->dims()->Get(0) != y->dims()->Get(1)
           || x->dims()->Get(1) % w->dims()->Get(1) != 0)
        {
            return std::nullopt;
        }
        const auto groups = x->dims()->Get(1) / w->dims()->Get(1);
        if(w->dims()->Get(0) % groups != 0)
        {
            return std::nullopt;
        }
        // W is [K, C/groups, spatial...], independent of physical tensor layout.
        return 2.0 * elements(*y) * elements(*w) / static_cast<double>(w->dims()->Get(0));
    }
    if(const auto* op = node.attributes_as_SdpaAttributes())
    {
        const auto* q = tensor(graph, op->q_tensor_uid());
        const auto* k = tensor(graph, op->k_tensor_uid());
        const auto* v = tensor(graph, op->v_tensor_uid());
        const auto* o = tensor(graph, op->o_tensor_uid());
        if(!dimensions(q, 4) || !dimensions(k, 4) || !dimensions(v, 4) || !dimensions(o, 4)
           || q->dims()->size() != 4 || k->dims()->size() != 4 || v->dims()->size() != 4
           || o->dims()->size() != 4 || op->seq_len_q_tensor_uid() || op->seq_len_kv_tensor_uid()
           || op->page_table_k_tensor_uid() || op->page_table_v_tensor_uid()
           || op->block_mask_tensor_uid() || op->sink_token_tensor_uid()
           || op->attn_mask_tensor_uid() || op->padding_mask() || op->alibi_mask()
           || op->descale_q_tensor_uid() || op->descale_k_tensor_uid() || op->descale_v_tensor_uid()
           || op->descale_s_tensor_uid() || op->scale_s_tensor_uid() || op->scale_o_tensor_uid()
           || op->dropout_mask_tensor_uid() || op->dropout_scale_tensor_uid()
           || (op->dropout_probability() && *op->dropout_probability() != 0.0F)
           || (op->left_bound() && *op->left_bound() >= 0)
           || (op->right_bound() && *op->right_bound() != 0 && *op->right_bound() != -1))
        {
            return std::nullopt;
        }
        const auto batch = q->dims()->Get(0);
        const auto heads = q->dims()->Get(1);
        const auto sq = q->dims()->Get(2);
        const auto sk = k->dims()->Get(2);
        const auto dq = q->dims()->Get(3);
        const auto dv = v->dims()->Get(3);
        if(k->dims()->Get(0) != batch || v->dims()->Get(0) != batch || o->dims()->Get(0) != batch
           || k->dims()->Get(1) != v->dims()->Get(1) || heads % k->dims()->Get(1) != 0
           || o->dims()->Get(1) != heads || v->dims()->Get(2) != sk || o->dims()->Get(2) != sq
           || k->dims()->Get(3) != dq || o->dims()->Get(3) != dv)
        {
            return std::nullopt;
        }
        const bool causal = op->causal_mask() || op->causal_mask_bottom_right()
                            || (op->right_bound() && *op->right_bound() == 0);
        const bool bottomRight = op->causal_mask_bottom_right()
                                 || op->diagonal_alignment() == DiagonalAlignment::BOTTOM_RIGHT;
        double pairs = static_cast<double>(sq) * static_cast<double>(sk);
        if(causal)
        {
            if(bottomRight)
            {
                // The declared corpus formula applies to Sq <= Sk; no half-rectangle
                // approximation when Sq > Sk, whose actual mask has empty rows.
                if(sq > sk)
                {
                    return std::nullopt;
                }
                pairs -= static_cast<double>(sq) * static_cast<double>(sq - 1) / 2.0;
            }
            else
            {
                const auto triangle = std::min(sq, sk);
                pairs = static_cast<double>(triangle) * (static_cast<double>(triangle) + 1.0) / 2.0
                        + static_cast<double>(sq - triangle) * static_cast<double>(sk);
            }
        }
        return 2.0 * static_cast<double>(batch) * static_cast<double>(heads) * pairs
               * (static_cast<double>(dq) + static_cast<double>(dv));
    }
    return std::nullopt;
}

inline void bindNodeFeatures(uhd::FeatureExtractionContext& features,
                             const Graph& graph,
                             const Node& node,
                             const std::string& prefix)
{
    const auto bindTensor = [&](const char* role, int64_t uid) {
        const auto* value = tensor(graph, uid);
        if(value == nullptr)
        {
            return;
        }
        const auto name = prefix + "." + role;
        features.bind(name + ".data_type", static_cast<int64_t>(value->data_type()));
        if(value->dims() != nullptr)
        {
            for(flatbuffers::uoffset_t i = 0; i < value->dims()->size(); ++i)
            {
                features.bind(name + ".dims[" + std::to_string(i) + "]", value->dims()->Get(i));
            }
        }
        if(value->strides() != nullptr)
        {
            for(flatbuffers::uoffset_t i = 0; i < value->strides()->size(); ++i)
            {
                features.bind(name + ".strides[" + std::to_string(i) + "]",
                              value->strides()->Get(i));
            }
        }
    };
    if(const auto* op = node.attributes_as_MatmulAttributes())
    {
        bindTensor("a", op->a_tensor_uid());
        bindTensor("b", op->b_tensor_uid());
        bindTensor("c", op->c_tensor_uid());
        return;
    }
    if(const auto* op = node.attributes_as_ConvolutionFwdAttributes())
    {
        bindTensor("x", op->x_tensor_uid());
        bindTensor("w", op->w_tensor_uid());
        bindTensor("y", op->y_tensor_uid());
        const auto bindVector = [&](const char* name, const auto* values) {
            if(values != nullptr)
            {
                for(flatbuffers::uoffset_t i = 0; i < values->size(); ++i)
                {
                    features.bind(prefix + "." + name + "[" + std::to_string(i) + "]",
                                  values->Get(i));
                }
            }
        };
        bindVector("pre_padding", op->pre_padding());
        bindVector("post_padding", op->post_padding());
        bindVector("stride", op->stride());
        bindVector("dilation", op->dilation());
        features.bind(prefix + ".conv_mode", static_cast<int64_t>(op->conv_mode()));
        return;
    }
    if(const auto* op = node.attributes_as_SdpaAttributes())
    {
        bindTensor("q", op->q_tensor_uid());
        bindTensor("k", op->k_tensor_uid());
        bindTensor("v", op->v_tensor_uid());
        bindTensor("o", op->o_tensor_uid());
        features.bind(prefix + ".causal_mask", op->causal_mask());
        features.bind(prefix + ".causal_mask_bottom_right", op->causal_mask_bottom_right());
        features.bind(prefix + ".diagonal_alignment",
                      static_cast<int64_t>(op->diagonal_alignment()));
        features.bind(prefix + ".padding_mask", op->padding_mask());
        features.bind(prefix + ".alibi_mask", op->alibi_mask());
        features.bind(prefix + ".has_attention_mask", op->attn_mask_tensor_uid().has_value());
        features.bind(prefix + ".has_variable_lengths",
                      op->seq_len_q_tensor_uid().has_value()
                          || op->seq_len_kv_tensor_uid().has_value());
        if(op->left_bound())
        {
            features.bind(prefix + ".left_bound", *op->left_bound());
        }
        if(op->right_bound())
        {
            features.bind(prefix + ".right_bound", *op->right_bound());
        }
        if(op->dropout_probability())
        {
            features.bind(prefix + ".dropout_probability",
                          static_cast<double>(*op->dropout_probability()));
        }
    }
}
} // namespace detail

/// @brief Publishes canonical graph, device and constraint features without kernel enumeration.
template <typename TProperties>
inline uhd::FeatureExtractionContext
    engineFeatures(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
                   const TProperties& device)
{
    uhd::FeatureExtractionContext features;
    for(const auto& entry : deviceFeatureValues(device))
    {
        std::visit(
            [&features, &entry](auto value) { features.bind("device." + entry.first, value); },
            entry.second);
    }
    if(const auto limit = workspaceLimit(config))
    {
        features.bind("constraint.workspace_limit", *limit);
    }
    if(config.isValid())
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        for(const auto& setting : config.knobSettingWrappers())
        {
            const auto name = setting->knobId();
            if(name == BENCHMARKING_KNOB_NAME || name == WORKSPACE_SIZE_LIMIT_KNOB_NAME)
            {
                continue;
            }
            const auto feature = "constraint.knobs." + name;
            switch(setting->valueType())
            {
            case KnobValue::IntValue:
                features.bind(feature, setting->template valueAs<IntValue>().value());
                break;
            case KnobValue::FloatValue:
                features.bind(feature, setting->template valueAs<FloatValue>().value());
                break;
            case KnobValue::StringValue:
                if(const auto* text = setting->template valueAs<StringValue>().value())
                {
                    features.bind(feature, text->str());
                }
                break;
            default:
                throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
                                            "Prediction constraint has no typed value");
            }
        }
    }
    const auto& raw = graph.getGraph();
    const auto* nodes = raw.nodes();
    const auto* tensors = raw.tensors();
    features.bind("graph.node_count", static_cast<int64_t>(nodes == nullptr ? 0 : nodes->size()));
    features.bind("graph.tensor_count",
                  static_cast<int64_t>(tensors == nullptr ? 0 : tensors->size()));
    if(tensors != nullptr)
    {
        for(flatbuffers::uoffset_t i = 0; i < tensors->size(); ++i)
        {
            const auto* tensor = tensors->Get(i);
            const auto prefix = "graph.tensors[" + std::to_string(i) + "]";
            features.bind(prefix + ".data_type", static_cast<int64_t>(tensor->data_type()));
            if(tensor->dims() != nullptr)
            {
                features.bind(prefix + ".rank", static_cast<int64_t>(tensor->dims()->size()));
                for(flatbuffers::uoffset_t d = 0; d < tensor->dims()->size(); ++d)
                {
                    features.bind(prefix + ".dims[" + std::to_string(d) + "]",
                                  tensor->dims()->Get(d));
                }
            }
            if(tensor->strides() != nullptr)
            {
                for(flatbuffers::uoffset_t d = 0; d < tensor->strides()->size(); ++d)
                {
                    features.bind(prefix + ".strides[" + std::to_string(d) + "]",
                                  tensor->strides()->Get(d));
                }
            }
        }
    }
    bool complete = nodes != nullptr && !nodes->empty() && !raw.is_override_shape_enabled();
    double flops = 0.0;
    if(nodes != nullptr)
    {
        for(flatbuffers::uoffset_t i = 0; i < nodes->size(); ++i)
        {
            const auto& node = *nodes->Get(i);
            const auto prefix = "graph.nodes[" + std::to_string(i) + "]";
            features.bind(prefix + ".type", static_cast<int64_t>(node.attributes_type()));
            features.bind(prefix + ".compute_data_type",
                          static_cast<int64_t>(node.compute_data_type()));
            const auto work = detail::logicalFlops(raw, node);
            detail::bindNodeFeatures(features, raw, node, prefix);
            if(work && std::isfinite(*work) && *work > 0.0)
            {
                features.bind(prefix + ".flops", *work);
                flops += *work;
            }
            else
            {
                complete = false;
            }
        }
    }
    if(complete && std::isfinite(flops) && flops > 0.0)
    {
        features.bind("graph.flops", flops);
    }
    return features;
}

} // namespace hipdnn_plugin_sdk::heuristics
