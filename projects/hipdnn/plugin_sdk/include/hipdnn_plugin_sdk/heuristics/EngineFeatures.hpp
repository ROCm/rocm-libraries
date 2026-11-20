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
#include <cstddef>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/node_operands_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
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
// UHD features need the kernel ingestor, whose expression language the extractor evaluates.

namespace detail
{
using Graph = hipdnn_flatbuffers_sdk::data_objects::Graph;
using Tensor = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
using Node = hipdnn_flatbuffers_sdk::data_objects::Node;
using NodeAttributes = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

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

/// Every extent present and positive; otherwise the work is unknown, never 0.
inline bool shaped(const Tensor* value, size_t minimumRank)
{
    return value != nullptr && value->dims() != nullptr && value->dims()->size() >= minimumRank
           && std::all_of(value->dims()->begin(), value->dims()->end(), [](int64_t extent) {
                  return extent > 0;
              });
}

/// A shaped floating-point operand: the matrix-product families (Matmul, convolutions,
/// attention, the MoE grouped matmuls, GDN and KDA) count floating-point multiply-adds only.
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

/// Score pairs attention computes, summed exactly: for each of @p sq query rows, the keys of
/// [0, @p sk) inside its band. Row i's diagonal is key i + @p shift (0 top-left, Sk - Sq
/// bottom-right); @p left keeps that many keys up to and including the diagonal and @p right
/// that many past it, an absent side being unbounded (cuDNN's band, SEM-001). Causal is
/// right = 0. A row the band leaves empty contributes nothing.
inline int64_t bandPairs(int64_t sq,
                         int64_t sk,
                         int64_t shift,
                         std::optional<int64_t> left,
                         std::optional<int64_t> right)
{
    int64_t pairs = 0;
    for(int64_t row = 0; row < sq; ++row)
    {
        const auto diagonal = row + shift;
        const auto first = left ? std::max<int64_t>(0, diagonal - *left + 1) : int64_t{0};
        const auto last = right ? std::min<int64_t>(sk - 1, diagonal + *right) : sk - 1;
        pairs += std::max<int64_t>(0, last - first + 1);
    }
    return pairs;
}

/// The keys per sequence an attention node declares, and which of K and V are paged: a paged
/// side is a cache [blocks, heads, block size, D], an unpaged one [B, heads, keys, D].
struct KeyExtent
{
    int64_t keys = 0;
    bool pagedK = false;
    bool pagedV = false;
};

/// The keys of an unpaged node: K's sequence extent.
template <typename TAttention>
std::optional<KeyExtent> denseKeys(const Graph& graph, const TAttention& op)
{
    const auto* k = tensor(graph, op.k_tensor_uid());
    if(!shaped(k, 4) || k->dims()->size() != 4)
    {
        return std::nullopt;
    }
    return KeyExtent{k->dims()->Get(2)};
}

/// The keys per sequence an SDPA forward node declares, inferred in the order of cuDNN's
/// `infer_s_kv` and the test SDK references' `sdpaPagedMaxSeqKv`. Either K or V may be paged
/// alone, each side with its own table [B, 1, pages, 1] and block size. The keys are
/// `max_seq_len_kv` when stated; else an unpaged K's, then V's, sequence extent; else the
/// attention mask's last extent; else rng_dump's; else the smaller of the paged sides' pages x
/// block size. Nullopt when they cannot be read, exceed a paged side's capacity or differ from
/// an unpaged side's extent: the work is then unknown. On every layout `sdpaPagedMaxSeqKv`
/// accepts the keys are the same. It counts a superset: it does not require `seq_len_q` and
/// `seq_len_kv`, an INT32 table of shape [B, 1, pages, 1], a table of exactly
/// ceil(keys / block size) pages, or a mask as wide as the keys or 1, so a table wider than the
/// keys counts the keys, and a mask broadcast over the keys (last extent 1) on two paged sides
/// gives one key, as `infer_s_kv` does. The bench writes `seq_len_kv` and `cu_seq_len_kv` from
/// the same keys.
inline std::optional<KeyExtent>
    keyExtent(const Graph& graph, const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes& op)
{
    const auto tableK = op.page_table_k_tensor_uid();
    const auto tableV = op.page_table_v_tensor_uid();
    if(!tableK && !tableV)
    {
        return denseKeys(graph, op);
    }
    const auto capacity = [&graph](int64_t table, int64_t cache) -> std::optional<int64_t> {
        const auto* pages = tensor(graph, table);
        const auto* blocks = tensor(graph, cache);
        if(!shaped(pages, 3) || !shaped(blocks, 4))
        {
            return std::nullopt;
        }
        return pages->dims()->Get(2) * blocks->dims()->Get(2);
    };
    const auto lastExtent = [&graph](int64_t uid) -> std::optional<int64_t> {
        const auto* value = tensor(graph, uid);
        if(!shaped(value, 1))
        {
            return std::nullopt;
        }
        return value->dims()->Get(value->dims()->size() - 1);
    };
    std::optional<int64_t> capacityK;
    std::optional<int64_t> capacityV;
    if(tableK)
    {
        capacityK = capacity(*tableK, op.k_tensor_uid());
    }
    if(tableV)
    {
        capacityV = capacity(*tableV, op.v_tensor_uid());
    }
    if((tableK && !capacityK) || (tableV && !capacityV))
    {
        return std::nullopt;
    }
    std::optional<int64_t> keys;
    if(op.max_seq_len_kv())
    {
        keys = static_cast<int64_t>(*op.max_seq_len_kv());
    }
    else if(!tableK || !tableV)
    {
        const auto* unpaged = tensor(graph, tableK ? op.v_tensor_uid() : op.k_tensor_uid());
        if(shaped(unpaged, 4) && unpaged->dims()->size() == 4)
        {
            keys = unpaged->dims()->Get(2);
        }
    }
    else if(op.attn_mask_tensor_uid())
    {
        keys = lastExtent(*op.attn_mask_tensor_uid());
    }
    else if(op.rng_dump_tensor_uid())
    {
        keys = lastExtent(*op.rng_dump_tensor_uid());
    }
    else
    {
        keys = std::min(*capacityK, *capacityV);
    }
    if(!keys || *keys <= 0 || (capacityK && *keys > *capacityK)
       || (capacityV && *keys > *capacityV))
    {
        return std::nullopt;
    }
    // An unpaged side holds the keys itself: [B, heads, keys, D].
    for(const auto& [paged, uid] : {std::pair{tableK.has_value(), op.k_tensor_uid()},
                                    std::pair{tableV.has_value(), op.v_tensor_uid()}})
    {
        const auto* side = tensor(graph, uid);
        if(!paged && (!shaped(side, 4) || side->dims()->Get(2) != *keys))
        {
            return std::nullopt;
        }
    }
    return KeyExtent{*keys, tableK.has_value(), tableV.has_value()};
}

/// Attention's logical work: the QK^T and PV multiply-adds over the score pairs its band keeps,
/// for the problem the graph declares (RFC 0019 section 6.8). Where operand contents decide how
/// much of it runs (sequence lengths, ragged offsets, cu_seq_len, page tables, block masks) this
/// is the declared extent, which `data_dependent` marks. Elementwise work on the scores (scale,
/// bias, ALiBi, softmax, dropout, sinks, FP8 scaling) is not counted, as softmax never was.
/// @p extent gives the keys per sequence and which sides are paged caches.
template <typename TAttention>
std::optional<double>
    attentionFlops(const Graph& graph, const TAttention& op, const std::optional<KeyExtent>& extent)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;
    const auto* q = tensor(graph, op.q_tensor_uid());
    const auto* k = tensor(graph, op.k_tensor_uid());
    const auto* v = tensor(graph, op.v_tensor_uid());
    const auto* o = tensor(graph, op.o_tensor_uid());
    if(!extent || !dimensions(q, 4) || !dimensions(k, 4) || !dimensions(v, 4) || !dimensions(o, 4)
       || q->dims()->size() != 4 || k->dims()->size() != 4 || v->dims()->size() != 4
       || o->dims()->size() != 4)
    {
        return std::nullopt;
    }
    const auto batch = q->dims()->Get(0);
    const auto heads = q->dims()->Get(1);
    const auto sq = q->dims()->Get(2);
    const auto dq = q->dims()->Get(3);
    const auto dv = v->dims()->Get(3);
    const auto sk = extent->keys;
    // A paged cache holds blocks, not sequences, along its first and third axes.
    const auto unpagedMatches = [batch, sk](const Tensor& side) {
        return side.dims()->Get(0) == batch && side.dims()->Get(2) == sk;
    };
    if(sk <= 0 || (!extent->pagedK && !unpagedMatches(*k))
       || (!extent->pagedV && !unpagedMatches(*v)) || k->dims()->Get(1) != v->dims()->Get(1)
       || heads % k->dims()->Get(1) != 0 || o->dims()->Get(0) != batch || o->dims()->Get(1) != heads
       || o->dims()->Get(2) != sq || k->dims()->Get(3) != dq || o->dims()->Get(3) != dv)
    {
        return std::nullopt;
    }
    // The deprecated causal flags compose as cuDNN's setters do (SEM-002, the references'
    // extractDiagonalBandParams): a flag means right_bound 0 unless one of at least 0 is
    // stated, and causal_mask_bottom_right also means BOTTOM_RIGHT. A band outside hipDNN's
    // domain has unknown work: the frontend rejects left_bound 0 and a bound below -1
    // (SEM-005), and the references and the hip-kernel-provider reject both flags together.
    std::optional<int64_t> left = op.left_bound();
    std::optional<int64_t> right = op.right_bound();
    if((left && (*left == 0 || *left < -1)) || (right && *right < -1)
       || (op.causal_mask() && op.causal_mask_bottom_right()))
    {
        return std::nullopt;
    }
    // -1 is unbounded, as an absent bound is.
    if(left && *left < 0)
    {
        left.reset();
    }
    if(right && *right < 0)
    {
        right.reset();
    }
    if(!right && (op.causal_mask() || op.causal_mask_bottom_right()))
    {
        right = 0;
    }
    const bool bottomRight = op.causal_mask_bottom_right()
                             || op.diagonal_alignment() == DiagonalAlignment::BOTTOM_RIGHT;
    const auto pairs = bandPairs(sq, sk, bottomRight ? sk - sq : 0, left, right);
    return 2.0 * static_cast<double>(batch) * static_cast<double>(heads)
           * static_cast<double>(pairs) * (static_cast<double>(dq) + static_cast<double>(dv));
}

// One overload per node type not in NO_COUNTING_CONVENTION, each with a logicalFlops case.
// Convention (RFC 0019 §6.8): logical work, not executed work; a multiply-add is 2, an
// elementwise operation or transcendental 1 per element. Changing a constant changes a
// published feature's meaning and must bump FEATURE_SEMANTICS_REVISION.

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
    return attentionFlops(graph, op, keyExtent(graph, op));
}

/// FlashAttention's convention: 2.5x the forward, which recomputes QK^T and adds the dV,
/// dP, dQ and dK products to the forward's two. A bias gradient is a reduction of dP and
/// inverse-scaled dropout elementwise, so neither adds to it.
inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes& op)
{
    const auto forward = attentionFlops(graph, op, denseKeys(graph, op));
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

/// The declared routed rows' K x N multiply-adds: token [1, T, K], weight [E, K, N] and
/// output [1, R, N]. Gathering routes the output's R rows, the other modes the token's T;
/// which of them a group covers is the routing offsets' contents, so this is the declared
/// extent (RFC 0019 section 6.8).
inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulAttributes& op)
{
    const auto* token = tensor(graph, op.token_tensor_uid());
    const auto* weight = tensor(graph, op.weight_tensor_uid());
    const auto* output = tensor(graph, op.output_tensor_uid());
    if(!dimensions(token, 2) || !dimensions(weight, 3) || !dimensions(output, 2))
    {
        return std::nullopt;
    }
    const auto tr = token->dims()->size();
    const auto wr = weight->dims()->size();
    const auto orank = output->dims()->size();
    const auto k = weight->dims()->Get(wr - 2);
    const auto n = weight->dims()->Get(wr - 1);
    if(token->dims()->Get(tr - 1) != k || output->dims()->Get(orank - 1) != n)
    {
        return std::nullopt;
    }
    const auto rows
        = op.mode() == hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode::GATHER
              ? output->dims()->Get(orank - 2)
              : token->dims()->Get(tr - 2);
    return 2.0 * static_cast<double>(rows) * static_cast<double>(k) * static_cast<double>(n);
}

/// The weight gradient's K x N multiply-adds over the declared rows: token [1, T, K],
/// doutput [1, T, N].
inline std::optional<double>
    nodeFlops(const Graph& graph,
              const hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulBwdAttributes& op)
{
    const auto* token = tensor(graph, op.token_tensor_uid());
    const auto* doutput = tensor(graph, op.doutput_tensor_uid());
    if(!dimensions(token, 2) || !dimensions(doutput, 2))
    {
        return std::nullopt;
    }
    const auto tr = token->dims()->size();
    const auto dr = doutput->dims()->size();
    const auto rows = token->dims()->Get(tr - 2);
    if(doutput->dims()->Get(dr - 2) != rows)
    {
        return std::nullopt;
    }
    return 2.0 * static_cast<double>(rows) * static_cast<double>(token->dims()->Get(tr - 1))
           * static_cast<double>(doutput->dims()->Get(dr - 1));
}

/// The delta-rule recurrence's multiply-adds per token and output head: S k, the rank-one
/// update (v - S k) k^T, which carries both the correction and the write since
/// S (I - beta k k^T) + beta v k^T = S + beta (v - S k) k^T, and S q, each V x K; decay and
/// gating are elementwise and not counted. q [T, H, K] and o [T, HO, V] are token-packed, so
/// how the tokens split into sequences does not change it.
template <typename TRecurrence>
std::optional<double> deltaRuleFlops(const Graph& graph, const TRecurrence& op)
{
    const auto* q = tensor(graph, op.q_tensor_uid());
    const auto* o = tensor(graph, op.o_tensor_uid());
    if(!dimensions(q, 3) || !dimensions(o, 3) || q->dims()->size() != 3 || o->dims()->size() != 3
       || q->dims()->Get(0) != o->dims()->Get(0))
    {
        return std::nullopt;
    }
    return 6.0 * static_cast<double>(q->dims()->Get(0)) * static_cast<double>(o->dims()->Get(1))
           * static_cast<double>(o->dims()->Get(2)) * static_cast<double>(q->dims()->Get(2));
}

inline std::optional<double>
    nodeFlops(const Graph& graph, const hipdnn_flatbuffers_sdk::data_objects::GdnAttributes& op)
{
    return deltaRuleFlops(graph, op);
}

inline std::optional<double>
    nodeFlops(const Graph& graph, const hipdnn_flatbuffers_sdk::data_objects::KdaAttributes& op)
{
    return deltaRuleFlops(graph, op);
}

template <typename TAttributes>
std::optional<double> flopsOf(const Graph& graph, const TAttributes* op)
{
    return op == nullptr ? std::nullopt : nodeFlops(graph, *op);
}

/// Node types whose work has no counting convention. logicalFlops leaves their work unknown,
/// so a graph holding one publishes no `graph.flops`: no `tflops` label or score and no
/// feature derived from FLOPs; time models are unaffected. Every other NodeAttributes member
/// has a nodeFlops overload and a logicalFlops case; TestEngineFeatures fails for a member
/// that has neither (docs/AddingNewOperations.md, "Work Model (FLOPs)").
inline constexpr std::array NO_COUNTING_CONVENTION{
    // Not an operation: the node has no attributes.
    NodeAttributes::NONE,
    // Opaque: the graph does not say what the op computes.
    NodeAttributes::CustomOpAttributes,
};

constexpr bool hasCountingConvention(NodeAttributes type)
{
    for(const auto listed : NO_COUNTING_CONVENTION)
    {
        if(listed == type)
        {
            return false;
        }
    }
    return true;
}

/// Logical work of the problem @p node declares, the same for every engine (RFC 0019
/// section 6.8). Where operand contents decide how much of it runs (sequence lengths, ragged
/// offsets, page tables, block masks, routing offsets) it is the declared extent: the node's
/// `data_dependent` says so. Unknown, never 0, when the operands do not compose or the type is
/// in NO_COUNTING_CONVENTION.
inline std::optional<double> logicalFlops(const Graph& graph, const Node& node)
{
    const auto type = node.attributes_type();
    if(!hasCountingConvention(type))
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
    case NodeAttributes::MoeGroupedMatmulAttributes:
        return flopsOf(graph, node.attributes_as_MoeGroupedMatmulAttributes());
    case NodeAttributes::MoeGroupedMatmulBwdAttributes:
        return flopsOf(graph, node.attributes_as_MoeGroupedMatmulBwdAttributes());
    case NodeAttributes::GdnAttributes:
        return flopsOf(graph, node.attributes_as_GdnAttributes());
    case NodeAttributes::KdaAttributes:
        return flopsOf(graph, node.attributes_as_KdaAttributes());
    // A type this build predates. A member of this build's NodeAttributes reaching here has
    // neither a case nor a NO_COUNTING_CONVENTION entry, which TestEngineFeatures rejects.
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

/// The graph's logical footprint: each non-virtual tensor once, numel x element size
/// (fractional for sub-byte types); not allocation size or memory traffic. A ragged tensor
/// counts its declared extent, as its work does. Unknown when any tensor's size is: a missing
/// or negative extent, or an unset type.
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

/// Publishes one node's operands through the generated visitor (node_operands_generated.h):
///   <prefix>.<role>.{data_type, rank, numel, virtual, dims[i], strides[i]} per tensor operand
///   <prefix>.<name> per scalar attribute, <prefix>.<name>[i] per vector attribute element
/// Shipped models read these names; keep them stable.
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
        // The annotation concerns the uid's contents, so it counts even if the tensor is
        // missing from the graph.
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

/// Publishes @p node's operands, `<prefix>.data_dependent`, and derived SDPA flags. Publishes
/// nothing for a node without attributes or of an unknown type.
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
    // Derived flags, not schema fields; shipped SDPA models read both.
    if(const auto* op = node.attributes_as_SdpaAttributes())
    {
        features.bind(prefix + ".has_attention_mask", op->attn_mask_tensor_uid().has_value());
        features.bind(prefix + ".has_variable_lengths",
                      op->seq_len_q_tensor_uid().has_value()
                          || op->seq_len_kv_tensor_uid().has_value());
    }
}
} // namespace detail

/// @brief Publishes the device and graph features fixed by (graph, device) alone.
///
/// Must not read the engine configuration: `sort_kernel_catalog` rankings are cached per
/// (graph, device, engine version, ranking metric, slot). Work features count the problem the
/// graph declares (RFC 0019 section 6.8), override-shape graphs by their declared shapes, and
/// are absent (never 0) only when it cannot be counted:
///   graph.nodes[i].flops         detail::logicalFlops
///   graph.flops                  sum over all nodes, virtual intermediates included
///   graph.flops_by_type.<Type>   sum per NodeAttributes member, 0 when none
///   graph.logical_bytes          detail::logicalBytes
///   graph.arithmetic_intensity   graph.flops / graph.logical_bytes, when bytes > 0
template <typename TProperties>
inline uhd::FeatureExtractionContext
    problemFeatures(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                    const TProperties& device)
{
    uhd::FeatureExtractionContext features;
    for(const auto& entry : deviceFeatureValues(device))
    {
        std::visit(
            [&features, &entry](auto value) { features.bind("device." + entry.first, value); },
            entry.second);
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
    bool complete = nodes != nullptr && !nodes->empty();
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
            detail::bindNodeFeatures(features, raw, node, prefix);
            const auto work = detail::logicalFlops(raw, node);
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

namespace detail
{
/// Everything problemFeatures() reads: the graph's content (GraphContentKey) and the device
/// values it publishes.
struct ProblemFeatureKey
{
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey graph;
    std::vector<std::pair<std::string, std::variant<int64_t, double>>> device;
    size_t hash = 0;

    template <typename TProperties>
    ProblemFeatureKey(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& source,
                      const TProperties& properties)
        : graph(source)
        , device(deviceFeatureValues(properties))
    {
        hash = static_cast<size_t>(graph.hash());
        const auto mix = [this](uint64_t value) {
            hash
                ^= static_cast<size_t>(value) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        };
        for(const auto& [name, value] : device)
        {
            mix(std::hash<std::string>{}(name));
            std::visit(
                [&mix](auto held) {
                    uint64_t bits = 0;
                    static_assert(sizeof(held) == sizeof(bits));
                    std::memcpy(&bits, &held, sizeof(bits));
                    mix(bits);
                },
                value);
        }
    }

    bool operator==(const ProblemFeatureKey& other) const
    {
        return hash == other.hash && device == other.device && graph == other.graph;
    }
};
} // namespace detail

/// @brief Recently computed problemFeatures(), shared by every engine of one plugin.
///
/// The backend asks each applicable engine for its estimate of the same graph in turn, and
/// each one needs the same problem features: they depend on the graph and the device alone
/// (problemFeatures()). The cache builds them once per (graph content, device) and hands each
/// engine the same immutable context, which the engine's own context reads through
/// (uhd::FeatureExtractionContext's shared base) while binding its own names on top.
///
/// Least recently used, a few entries: one selection asks for one graph, and a few threads
/// may select at once. A graph without usable content (GraphContentKey::isUsable()) is never
/// cached. Thread-safe.
class ProblemFeatureCache
{
public:
    static constexpr size_t CAPACITY = 16;

    /// @returns problemFeatures(@p graph, @p device), computed now or reused.
    template <typename TProperties>
    std::shared_ptr<const uhd::FeatureExtractionContext>
        get(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
            const TProperties& device)
    {
        detail::ProblemFeatureKey key(graph, device);
        if(!key.graph.isUsable())
        {
            return std::make_shared<const uhd::FeatureExtractionContext>(
                problemFeatures(graph, device));
        }
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            if(auto found = findLocked(key))
            {
                return found;
            }
        }
        // Computed outside the lock, so engines selecting for other graphs do not wait.
        auto computed
            = std::make_shared<const uhd::FeatureExtractionContext>(problemFeatures(graph, device));
        const std::lock_guard<std::mutex> lock(_mutex);
        if(auto found = findLocked(key))
        {
            return found; // Another thread computed it first.
        }
        _entries.emplace_front(std::move(key), computed);
        if(_entries.size() > CAPACITY)
        {
            _entries.pop_back();
        }
        return computed;
    }

    /// @returns The number of graphs held, at most CAPACITY.
    size_t size() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _entries.size();
    }

    /// Forgets every graph; a context already handed out stays valid for its holders.
    void clear()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _entries.clear();
    }

private:
    using Entry = std::pair<detail::ProblemFeatureKey,
                            std::shared_ptr<const uhd::FeatureExtractionContext>>;

    /// The entry for @p key, moved to the front, or null. Caller holds _mutex.
    std::shared_ptr<const uhd::FeatureExtractionContext>
        findLocked(const detail::ProblemFeatureKey& key)
    {
        for(auto it = _entries.begin(); it != _entries.end(); ++it)
        {
            if(it->first == key)
            {
                _entries.splice(_entries.begin(), _entries, it);
                return _entries.front().second;
            }
        }
        return nullptr;
    }

    mutable std::mutex _mutex;
    std::list<Entry> _entries; // Most recently used first.
};

/// @brief The process-wide (per plugin library) ProblemFeatureCache.
inline ProblemFeatureCache& problemFeatureCache()
{
    static ProblemFeatureCache s_cache;
    return s_cache;
}

/// @brief problemFeatures() through problemFeatureCache(): built once per graph and device,
///        then shared, immutable, by every caller.
template <typename TProperties>
inline std::shared_ptr<const uhd::FeatureExtractionContext>
    sharedProblemFeatures(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                          const TProperties& device)
{
    return problemFeatureCache().get(graph, device);
}

/// @brief problemFeatures() plus the engine-level (L1) `constraint.*` features of @p config:
///        `constraint.workspace_limit` and `constraint.knobs.<name>` for every pinned knob
///        except the benchmarking knob, which chooses how to select, not what.
///
/// The problem features are sharedProblemFeatures(), read through; only the `constraint.*`
/// names are bound in the returned context itself.
template <typename TProperties>
inline uhd::FeatureExtractionContext
    engineFeatures(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                   const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& config,
                   const TProperties& device)
{
    uhd::FeatureExtractionContext features(sharedProblemFeatures(graph, device));
    if(const auto limit = workspaceLimit(config))
    {
        features.bind("constraint.workspace_limit", *limit);
    }
    if(!config.isValid())
    {
        return features;
    }
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
    return features;
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

} // namespace hipdnn_plugin_sdk::heuristics
