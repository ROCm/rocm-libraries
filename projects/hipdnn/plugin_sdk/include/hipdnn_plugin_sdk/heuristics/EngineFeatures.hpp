// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
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

/// Every extent present and positive, so a count over them is the problem's size: a
/// missing, negative or zero extent leaves the work unknown, never 0.
inline bool shaped(const Tensor* value, size_t minimumRank)
{
    return value != nullptr && value->dims() != nullptr && value->dims()->size() >= minimumRank
           && std::all_of(value->dims()->begin(), value->dims()->end(), [](int64_t extent) {
                  return extent > 0;
              });
}

/// A shaped floating-point operand. The matrix-product families (Matmul, convolutions,
/// attention) count floating-point multiply-adds only; that gate predates the other types.
inline bool dimensions(const Tensor* value, size_t minimumRank)
{
    return value != nullptr && floatingPoint(value->data_type()) && shaped(value, minimumRank);
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

/// @p operations per element of @p value, whatever its element type: comparisons produce
/// booleans and quantization integers, and each element is still one unit of work.
inline std::optional<double> perElement(const Tensor* value, double operations)
{
    if(!shaped(value, 1))
    {
        return std::nullopt;
    }
    return operations * elements(*value);
}

/// One operation per window element for every element of @p output [N, C, spatial...];
/// @p window has one positive extent per spatial dimension.
inline std::optional<double> windowed(const Tensor* output,
                                      const flatbuffers::Vector<int64_t>* window)
{
    if(!shaped(output, 3) || window == nullptr || window->size() != output->dims()->size() - 2)
    {
        return std::nullopt;
    }
    double product = elements(*output);
    for(const auto extent : *window)
    {
        if(extent <= 0)
        {
            return std::nullopt;
        }
        product *= static_cast<double>(extent);
    }
    return product;
}

/// Multiply-adds of the forward convolution's iteration space, which the data- and
/// weight-gradient passes share: @p input [N, C, spatial...], @p filter [K, C/groups,
/// spatial...], @p output [N, K, spatial...], independent of physical tensor layout.
inline std::optional<double>
    convolutionFlops(const Tensor* input, const Tensor* filter, const Tensor* output)
{
    if(!dimensions(input, 3) || !dimensions(filter, 3) || !dimensions(output, 3)
       || input->dims()->size() != filter->dims()->size()
       || input->dims()->size() != output->dims()->size()
       || input->dims()->Get(0) != output->dims()->Get(0)
       || filter->dims()->Get(0) != output->dims()->Get(1)
       || input->dims()->Get(1) % filter->dims()->Get(1) != 0)
    {
        return std::nullopt;
    }
    const auto groups = input->dims()->Get(1) / filter->dims()->Get(1);
    if(filter->dims()->Get(0) % groups != 0)
    {
        return std::nullopt;
    }
    return 2.0 * elements(*output) * elements(*filter)
           / static_cast<double>(filter->dims()->Get(0));
}

/// Dense attention's QK^T and PV multiply-adds over the score pairs its causal mask keeps.
/// Refuses, for the fields forward and backward share, whatever makes the pair count
/// depend on more than shapes and mask flags.
template <typename TAttention>
std::optional<double> attentionFlops(const Graph& graph, const TAttention& op)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    const auto* q = tensor(graph, op.q_tensor_uid());
    const auto* k = tensor(graph, op.k_tensor_uid());
    const auto* v = tensor(graph, op.v_tensor_uid());
    const auto* o = tensor(graph, op.o_tensor_uid());
    if(!dimensions(q, 4) || !dimensions(k, 4) || !dimensions(v, 4) || !dimensions(o, 4)
       || q->dims()->size() != 4 || k->dims()->size() != 4 || v->dims()->size() != 4
       || o->dims()->size() != 4 || op.seq_len_q_tensor_uid() || op.seq_len_kv_tensor_uid()
       || op.attn_mask_tensor_uid() || op.padding_mask() || op.alibi_mask()
       || op.dropout_mask_tensor_uid() || op.dropout_scale_tensor_uid()
       || (op.dropout_probability() && *op.dropout_probability() != 0.0F)
       || (op.left_bound() && *op.left_bound() >= 0)
       || (op.right_bound() && *op.right_bound() != 0 && *op.right_bound() != -1))
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
    const bool causal = op.causal_mask() || op.causal_mask_bottom_right()
                        || (op.right_bound() && *op.right_bound() == 0);
    const bool bottomRight = op.causal_mask_bottom_right()
                             || op.diagonal_alignment() == DiagonalAlignment::BOTTOM_RIGHT;
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

// One overload per node type. Convention: a multiply-add is 2, an elementwise operation 1
// per element, a transcendental 1; the per-element constants are the declared convention,
// not a count of any implementation's instructions. Changing one changes a published
// feature's meaning, so it bumps FEATURE_SEMANTICS_REVISION (FeatureSemantics.hpp).

/// `2 * c.numel * k`: the batch broadcast is inside `c.numel`.
inline std::optional<double>
    nodeFlops(const Graph& graph, const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes& op)
{
    const auto* a = tensor(graph, op.a_tensor_uid());
    const auto* b = tensor(graph, op.b_tensor_uid());
    const auto* c = tensor(graph, op.c_tensor_uid());
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

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& op)
{
    return convolutionFlops(tensor(graph, op.x_tensor_uid()),
                            tensor(graph, op.w_tensor_uid()),
                            tensor(graph, op.y_tensor_uid()));
}

/// Data gradient: `2 * dy.numel * w.numel / w.dims[0]`, the forward's iteration space.
inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes& op)
{
    return convolutionFlops(tensor(graph, op.dx_tensor_uid()),
                            tensor(graph, op.w_tensor_uid()),
                            tensor(graph, op.dy_tensor_uid()));
}

/// Weight gradient: `2 * dy.numel * dw.numel / dw.dims[0]`, the forward's iteration space.
inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes& op)
{
    return convolutionFlops(tensor(graph, op.x_tensor_uid()),
                            tensor(graph, op.dw_tensor_uid()),
                            tensor(graph, op.dy_tensor_uid()));
}

inline std::optional<double>
    nodeFlops(const Graph& graph, const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& op)
{
    // Paged caches, block-sparse masks, sink tokens and FP8 scaling have no counting
    // convention yet.
    if(op.page_table_k_tensor_uid() || op.page_table_v_tensor_uid() || op.block_mask_tensor_uid()
       || op.sink_token_tensor_uid() || op.descale_q_tensor_uid() || op.descale_k_tensor_uid()
       || op.descale_v_tensor_uid() || op.descale_s_tensor_uid() || op.scale_s_tensor_uid()
       || op.scale_o_tensor_uid())
    {
        return std::nullopt;
    }
    return attentionFlops(graph, op);
}

/// FlashAttention's convention: 2.5x the forward, which recomputes QK^T and adds the dV,
/// dP, dQ and dK products to the forward's two.
inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& op)
{
    // A bias gradient's reduction and inverse-scaled dropout have no convention yet.
    if(op.dbias_tensor_uid() || op.dropout_scale_inv_tensor_uid())
    {
        return std::nullopt;
    }
    const auto forward = attentionFlops(graph, op);
    if(!forward)
    {
        return std::nullopt;
    }
    return 2.5 * *forward;
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& op)
{
    return perElement(tensor(graph, op.out_0_tensor_uid()), 1.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes& op)
{
    return perElement(tensor(graph, op.in_tensor_uid()), 1.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 2.0);
}

inline std::optional<double> nodeFlops(
    const Graph& graph,
    const hipdnn_flatbuffers_sdk::data_objects::BatchnormInferenceAttributesVarianceExt& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 2.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::BatchnormAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 5.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::BatchnormBackwardAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 8.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::LayernormAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 5.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::LayernormBackwardAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 8.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph, const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 3.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::RMSNormBackwardAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 6.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& op)
{
    return windowed(tensor(graph, op.y_tensor_uid()), op.window());
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::ResampleBwdAttributes& op)
{
    return windowed(tensor(graph, op.dy_tensor_uid()), op.window());
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::BlockScaleQuantizeAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 2.0);
}

inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::BlockScaleDequantizeAttributes& op)
{
    return perElement(tensor(graph, op.x_tensor_uid()), 2.0);
}

template <typename TAttributes>
std::optional<double> flopsOf(const Graph& graph, const TAttributes* op)
{
    return op == nullptr ? std::nullopt : nodeFlops(graph, *op);
}

/// Logical work of @p node (principle: the problem's work, the same for every engine).
/// Unknown -- never 0 -- when a dimension is, when the type has no convention, or when
/// @p dataDependent: the node's published `data_dependent` verdict, that operand contents
/// (routing offsets, ragged lengths, page tables) decide the work.
inline std::optional<double> logicalFlops(const Graph& graph, const Node& node, bool dataDependent)
{
    using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
    const auto type = node.attributes_type();
    // These three published FLOPs before the content-dependence rule existed. Their own
    // refusals already cover every annotated operand, but a ragged operand would newly
    // hide a count shipped models were trained on, so for them the rule is not applied.
    const bool predatesContentRule = type == NodeAttributes::MatmulAttributes
                                     || type == NodeAttributes::ConvolutionFwdAttributes
                                     || type == NodeAttributes::SdpaAttributes;
    if(dataDependent && !predatesContentRule)
    {
        return std::nullopt;
    }
    switch(type)
    {
    case NodeAttributes::MatmulAttributes:
        return flopsOf(graph, node.attributes_as_MatmulAttributes());
    case NodeAttributes::ConvolutionFwdAttributes:
        return flopsOf(graph, node.attributes_as_ConvolutionFwdAttributes());
    case NodeAttributes::SdpaAttributes:
        return flopsOf(graph, node.attributes_as_SdpaAttributes());
    case NodeAttributes::ConvolutionBwdAttributes:
        return flopsOf(graph, node.attributes_as_ConvolutionBwdAttributes());
    case NodeAttributes::ConvolutionWrwAttributes:
        return flopsOf(graph, node.attributes_as_ConvolutionWrwAttributes());
    case NodeAttributes::SdpaBackwardAttributes:
        return flopsOf(graph, node.attributes_as_SdpaBackwardAttributes());
    case NodeAttributes::PointwiseAttributes:
        return flopsOf(graph, node.attributes_as_PointwiseAttributes());
    case NodeAttributes::ReductionAttributes:
        return flopsOf(graph, node.attributes_as_ReductionAttributes());
    case NodeAttributes::BatchnormInferenceAttributes:
        return flopsOf(graph, node.attributes_as_BatchnormInferenceAttributes());
    case NodeAttributes::BatchnormInferenceAttributesVarianceExt:
        return flopsOf(graph, node.attributes_as_BatchnormInferenceAttributesVarianceExt());
    case NodeAttributes::BatchnormAttributes:
        return flopsOf(graph, node.attributes_as_BatchnormAttributes());
    case NodeAttributes::BatchnormBackwardAttributes:
        return flopsOf(graph, node.attributes_as_BatchnormBackwardAttributes());
    case NodeAttributes::LayernormAttributes:
        return flopsOf(graph, node.attributes_as_LayernormAttributes());
    case NodeAttributes::LayernormBackwardAttributes:
        return flopsOf(graph, node.attributes_as_LayernormBackwardAttributes());
    case NodeAttributes::RMSNormAttributes:
        return flopsOf(graph, node.attributes_as_RMSNormAttributes());
    case NodeAttributes::RMSNormBackwardAttributes:
        return flopsOf(graph, node.attributes_as_RMSNormBackwardAttributes());
    case NodeAttributes::ResampleFwdAttributes:
        return flopsOf(graph, node.attributes_as_ResampleFwdAttributes());
    case NodeAttributes::ResampleBwdAttributes:
        return flopsOf(graph, node.attributes_as_ResampleBwdAttributes());
    case NodeAttributes::BlockScaleQuantizeAttributes:
        return flopsOf(graph, node.attributes_as_BlockScaleQuantizeAttributes());
    case NodeAttributes::BlockScaleDequantizeAttributes:
        return flopsOf(graph, node.attributes_as_BlockScaleDequantizeAttributes());
    // MoE: the rows each expert multiplies follow the routing offsets' contents (the
    // required offsets make every such node data-dependent above). Custom ops: an opaque
    // payload no convention can count.
    case NodeAttributes::MoeGroupedMatmulAttributes:
    case NodeAttributes::MoeGroupedMatmulBwdAttributes:
    case NodeAttributes::CustomOpAttributes:
    default:
        return std::nullopt;
    }
}

/// Bits one element of @p type occupies; unknown for UNSET or a type this build predates.
inline std::optional<double> elementBits(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(type)
    {
    case DataType::DOUBLE:
    case DataType::INT64:
        return 64.0;
    case DataType::FLOAT:
    case DataType::INT32:
        return 32.0;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 16.0;
    case DataType::UINT8:
    case DataType::INT8:
    case DataType::BOOLEAN:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E8M0:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
        return 8.0;
    case DataType::FP6_E2M3:
    case DataType::FP6_E3M2:
        return 6.0;
    case DataType::FP4_E2M1:
    case DataType::INT4:
        return 4.0;
    case DataType::UNSET:
    default:
        return std::nullopt;
    }
}

/// The graph's logical footprint: every non-virtual tensor once, numel x element size,
/// fractional for sub-byte types. Neither the allocation (strides, alignment, packing,
/// workspace) nor measured traffic. Unknown when any such tensor's size is: a missing or
/// negative extent, an unset type, or a ragged tensor, whose offsets rather than its dims
/// decide how many elements it holds.
inline std::optional<double> logicalBytes(const Graph& graph)
{
    double bytes = 0.0;
    if(graph.tensors() == nullptr)
    {
        return bytes;
    }
    for(const auto* value : *graph.tensors())
    {
        if(value->virtual_())
        {
            continue;
        }
        const auto bits = elementBits(value->data_type());
        if(!bits || value->dims() == nullptr
           || hipdnn_flatbuffers_sdk::data_objects::node_operands::workDataDependent(*value)
           || std::any_of(value->dims()->begin(), value->dims()->end(), [](int64_t extent) {
                  return extent < 0;
              }))
        {
            return std::nullopt;
        }
        bytes += elements(*value) * *bits / 8.0;
    }
    return bytes;
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

/// Publishes @p node's operands, `<prefix>.data_dependent`, and the derived SDPA flags, and
/// returns the `data_dependent` verdict so the work model reads the value it published.
/// Publishes nothing, and returns false, for a node without attributes or of a type this
/// build predates.
inline bool bindNodeFeatures(uhd::FeatureExtractionContext& features,
                             const Graph& graph,
                             const Node& node,
                             const std::string& prefix)
{
    NodeFeatureBinder binder(features, graph, prefix);
    if(!hipdnn_flatbuffers_sdk::data_objects::node_operands::visit(node, binder))
    {
        return false;
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
    return binder.dataDependent();
}
} // namespace detail

/// @brief Publishes canonical graph, device and constraint features without kernel enumeration.
///
/// Work features, each absent -- never 0 -- when unknown:
///   graph.nodes[i].flops         the node's logical work (detail::logicalFlops).
///   graph.flops                  sum of every node's work, virtual intermediates included;
///                                absent when any node's work is unknown.
///   graph.flops_by_type.<Type>   per NodeAttributes member: the sum over nodes of that type,
///                                0 when the graph has none, absent when any one is unknown.
///   graph.logical_bytes          logical footprint of the non-virtual tensors (numel x
///                                element size, fractional for sub-byte types). Not the
///                                allocation size and not measured memory traffic.
///   graph.arithmetic_intensity   graph.flops / graph.logical_bytes, when both are published
///                                and the bytes are positive.
/// Every graph-level aggregate is absent when the graph opts into execute-time override
/// shapes: its declared shapes need not be the ones executed.
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
    using hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
    const bool shapesFixed = !raw.is_override_shape_enabled();
    bool complete = nodes != nullptr && !nodes->empty() && shapesFixed;
    double flops = 0.0;
    // Indexed by NodeAttributes value.
    constexpr auto TYPES = static_cast<size_t>(NodeAttributes::MAX) + 1;
    std::array<double, TYPES> flopsByType{};
    std::array<bool, TYPES> unknownByType{};
    if(nodes != nullptr)
    {
        for(flatbuffers::uoffset_t i = 0; i < nodes->size(); ++i)
        {
            const auto& node = *nodes->Get(i);
            const auto prefix = "graph.nodes[" + std::to_string(i) + "]";
            features.bind(prefix + ".type", static_cast<int64_t>(node.attributes_type()));
            features.bind(prefix + ".compute_data_type",
                          static_cast<int64_t>(node.compute_data_type()));
            const bool dataDependent = detail::bindNodeFeatures(features, raw, node, prefix);
            const auto work = detail::logicalFlops(raw, node, dataDependent);
            // A type this build predates has no name to publish under.
            const auto type = static_cast<size_t>(node.attributes_type());
            if(work && std::isfinite(*work) && *work > 0.0)
            {
                features.bind(prefix + ".flops", *work);
                flops += *work;
                if(type < TYPES)
                {
                    flopsByType[type] += *work;
                }
            }
            else
            {
                complete = false;
                if(type < TYPES)
                {
                    unknownByType[type] = true;
                }
            }
        }
    }
    const bool flopsKnown = complete && std::isfinite(flops) && flops > 0.0;
    if(flopsKnown)
    {
        features.bind("graph.flops", flops);
    }
    if(!shapesFixed)
    {
        return features;
    }
    for(size_t type = 1; type < TYPES; ++type)
    {
        if(!unknownByType[type] && std::isfinite(flopsByType[type]))
        {
            features.bind(std::string("graph.flops_by_type.")
                              + EnumNameNodeAttributes(static_cast<NodeAttributes>(type)),
                          flopsByType[type]);
        }
    }
    const auto bytes = detail::logicalBytes(raw);
    if(bytes && std::isfinite(*bytes))
    {
        features.bind("graph.logical_bytes", *bytes);
        if(flopsKnown && *bytes > 0.0)
        {
            features.bind("graph.arithmetic_intensity", flops / *bytes);
        }
    }
    return features;
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

} // namespace hipdnn_plugin_sdk::heuristics
