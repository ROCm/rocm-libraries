// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include <hipdnn_flatbuffers_sdk/data_objects/node_operands_generated.h>
#include <hipdnn_plugin_sdk/heuristics/DeviceFeatures.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#endif

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

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
// Everything below publishes UHD feature symbols, which exist only with the kernel ingestor:
// the feature extractor evaluates the descriptor expression language it ships.

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

/// Publishes one node's operands through the schema-generated visitor
/// (node_operands_generated.h), so every node type is covered without per-op code:
///   <prefix>.<role>.{data_type, rank, numel, virtual, dims[i], strides[i]} per tensor operand
///   <prefix>.<name> per scalar attribute, <prefix>.<name>[i] per vector attribute element
/// Names and values match what the hand-written Matmul/ConvolutionFwd/SDPA binders
/// published, which shipped models read.
class NodeFeatureBinder
{
public:
    NodeFeatureBinder(uhd::FeatureExtractionContext& features,
                      const Graph& graph,
                      const std::string& prefix)
        : _features(features)
        , _graph(graph)
        , _prefix(prefix)
    {
    }

    void tensor(std::string_view role, int64_t uid, bool workDataDependent)
    {
        // The annotation is about the uid's contents, so it counts even when the
        // referenced tensor is missing from the graph.
        _dataDependent = _dataDependent || workDataDependent;
        const auto* value = detail::tensor(_graph, uid);
        if(value == nullptr)
        {
            return;
        }
        _dataDependent
            = _dataDependent
              || hipdnn_flatbuffers_sdk::data_objects::node_operands::workDataDependent(*value);
        const auto name = feature(role) + ".";
        _features.bind(name + "data_type", static_cast<int64_t>(value->data_type()));
        _features.bind(name + "virtual", value->virtual_());
        if(const auto* dims = value->dims())
        {
            _features.bind(name + "rank", static_cast<int64_t>(dims->size()));
            if(const auto count = numel(*dims))
            {
                _features.bind(name + "numel", *count);
            }
            for(flatbuffers::uoffset_t i = 0; i < dims->size(); ++i)
            {
                _features.bind(name + "dims[" + std::to_string(i) + "]", dims->Get(i));
            }
        }
        if(const auto* strides = value->strides())
        {
            for(flatbuffers::uoffset_t i = 0; i < strides->size(); ++i)
            {
                _features.bind(name + "strides[" + std::to_string(i) + "]", strides->Get(i));
            }
        }
    }

    template <typename TValue>
    void scalar(std::string_view name, TValue value)
    {
        _features.bind(feature(name), value);
    }

    template <typename TValue>
    void element(std::string_view name, size_t index, TValue value)
    {
        _features.bind(feature(name) + "[" + std::to_string(index) + "]", value);
    }

    /// Whether any operand's contents decide the node's work: a `work_data_dependent`
    /// operand is present, or an operand tensor is itself data-dependent (ragged).
    bool dataDependent() const
    {
        return _dataDependent;
    }

private:
    std::string feature(std::string_view name) const
    {
        std::string result = _prefix;
        result += '.';
        result += name;
        return result;
    }

    /// Absent when a dimension is negative or the product overflows: unknown, not 0.
    static std::optional<int64_t> numel(const flatbuffers::Vector<int64_t>& dims)
    {
        int64_t product = 1;
        for(const auto extent : dims)
        {
            if(extent < 0
               || (extent != 0 && product > std::numeric_limits<int64_t>::max() / extent))
            {
                return std::nullopt;
            }
            product *= extent;
        }
        return product;
    }

    uhd::FeatureExtractionContext& _features;
    const Graph& _graph;
    const std::string& _prefix;
    bool _dataDependent = false;
};

/// Publishes @p node's operands, `<prefix>.data_dependent`, and the derived SDPA flags.
/// Publishes nothing for a node without attributes or of a type this build predates.
inline void bindNodeFeatures(uhd::FeatureExtractionContext& features,
                             const Graph& graph,
                             const Node& node,
                             const std::string& prefix)
{
    NodeFeatureBinder binder(features, graph, prefix);
    if(!hipdnn_flatbuffers_sdk::data_objects::node_operands::visit(node, binder))
    {
        return;
    }
    features.bind(prefix + ".data_dependent", binder.dataDependent());
    // Derived rather than schema fields, so no generated visitor reports them; the
    // shipped SDPA models read both.
    if(const auto* op = node.attributes_as_SdpaAttributes())
    {
        features.bind(prefix + ".has_attention_mask", op->attn_mask_tensor_uid().has_value());
        features.bind(prefix + ".has_variable_lengths",
                      op->seq_len_q_tensor_uid().has_value()
                          || op->seq_len_kv_tensor_uid().has_value());
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

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

} // namespace hipdnn_plugin_sdk::heuristics
