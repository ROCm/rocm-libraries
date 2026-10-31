// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaRdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"

/**
 * @file FlydslSdpaRdnaCommon.hpp
 * @brief What the two RDNA SDPA engines share: the graph matcher, the bindings it
 *        establishes, and the checks on a graph's operands, mask, bias, sink and scale.
 *
 * The prefill kernel (`hipkernel:flydsl_sdpa_rdna`, FlydslSdpaRdnaNative.cpp) and the
 * split-KV decode kernel (`hipkernel:flydsl_sdpa_rdna_decode`,
 * FlydslSdpaRdnaDecodeNative.cpp) are separate engines, one kernel family each, but they
 * serve the same graphs: one SDPA-forward node, with the same operands, mask, bias, sink,
 * statistics and scale. So one graph matcher serves both, registered under each engine's
 * symbol, and each engine adds only its own kernel matcher, signature and dispatch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_rdna
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace flydsl_sdpa;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

// Metadata fields the two families' KMDs share.
inline constexpr std::string_view DTYPE_FIELD = "dtype";
inline constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
inline constexpr std::string_view CAUSAL_FIELD = "causal";
inline constexpr std::string_view BLOCK_M_FIELD = "block_m";
inline constexpr std::string_view BLOCK_N_FIELD = "block_n";
/// Workgroups that share one query tile's output columns, each computing head_dim /
/// dv_split of them; the grid grows by it. 1 up to head_dim 128, 2 above.
inline constexpr std::string_view DV_SPLIT_FIELD = "dv_split";
/// 0 or 1: whether the object adds an additive bias to the scores. Baked, so a graph with
/// a bias and one without each select their own object.
inline constexpr std::string_view HAS_BIAS_FIELD = "has_bias";
/// 0 for an object that reads an f32 bias, 1 for one that reads it in the query's dtype.
inline constexpr std::string_view BIAS_Q_FIELD = "bias_q";
/// The largest head_dim the object serves. Equal to head_dim for a specialized object; for
/// a generic one (head_dim 0, read at runtime) it bounds the heads it takes.
inline constexpr std::string_view HEAD_DIM_MAX_FIELD = "head_dim_max";
/// V's (and O's) head dim. Equal to head_dim except on the MLA objects, whose V is
/// narrower than Q and K; 0 on a generic object, whose V is its runtime head_dim too.
inline constexpr std::string_view HEAD_DIM_V_FIELD = "head_dim_v";

/// 0 or 1: whether the prefill object's waves work in head-dim pairs, two to each 16 query
/// rows, so its workgroup has twice the waves its block_m alone implies.
inline constexpr std::string_view HD_PAIR_FIELD = "hd_pair";
/// 0 or 1: whether the object adds an attention sink's logit to each row's softmax. Baked,
/// so a graph with a sink and one without each select their own object.
inline constexpr std::string_view HAS_SINK_FIELD = "has_sink";

/// A generic object's head_dim: it reads the real one at runtime.
inline constexpr int64_t GENERIC_HEAD_DIM = 0;

/// A generic object zeroes and skips whole 8-column groups, so the head must be made of them.
inline constexpr int64_t GENERIC_HEAD_DIM_ALIGN = 8;

inline constexpr std::string_view Q_TOKEN = "flydsl_sdpa_rdna.q.uid";
inline constexpr std::string_view K_TOKEN = "flydsl_sdpa_rdna.k.uid";
inline constexpr std::string_view V_TOKEN = "flydsl_sdpa_rdna.v.uid";
inline constexpr std::string_view O_TOKEN = "flydsl_sdpa_rdna.o.uid";
/// 0 or 1: which of the two kernel variants the graph needs.
inline constexpr std::string_view CAUSAL_TOKEN = "flydsl_sdpa_rdna.causal";
/// The mask, in the kernels' terms (resolveMaskBounds): right and left bounds (-1 for none;
/// left is the graph's left_bound - 1) and whether the diagonal is bottom-right aligned. The
/// kernel derives the diagonal's offset from the sequence lengths itself.
inline constexpr std::string_view RIGHT_BOUND_TOKEN = "flydsl_sdpa_rdna.right_bound";
inline constexpr std::string_view LEFT_BOUND_TOKEN = "flydsl_sdpa_rdna.left_bound";
inline constexpr std::string_view ALIGN_BOTTOM_RIGHT_TOKEN = "flydsl_sdpa_rdna.align_bottom_right";
/// The uid of the softmax-statistics (LSE) output, or NO_STATS when the graph wants none.
inline constexpr std::string_view STATS_TOKEN = "flydsl_sdpa_rdna.stats.uid";
/// The uid of the additive bias (`attn_mask`), or NO_BIAS when the graph has none.
inline constexpr std::string_view BIAS_TOKEN = "flydsl_sdpa_rdna.bias.uid";
/// The uid of the attention sink, or NO_SINK, and its element stride across query heads.
inline constexpr std::string_view SINK_TOKEN = "flydsl_sdpa_rdna.sink.uid";
inline constexpr std::string_view SINK_STRIDE_TOKEN = "flydsl_sdpa_rdna.sink.stride";
/// Where the softmax scale comes from; see ScaleSource.
inline constexpr std::string_view SCALE_SOURCE_TOKEN = "flydsl_sdpa_rdna.scale_source";
/// The scale's f32 bit pattern when baked (BoundTokens carry int64_t), else the uid of
/// the pass-by-value scale tensor.
inline constexpr std::string_view SCALE_TOKEN = "flydsl_sdpa_rdna.scale";

/// The bias's (batch, head, query, key) element strides as the kernel takes them.
using BiasStrides = std::array<int64_t, 4>;

/// The bias is read through a 32-bit buffer descriptor per (batch, head) slice, with 32-bit
/// element offsets inside it.
inline constexpr int64_t BIAS_SLICE_ELEMENT_LIMIT = (1LL << 30) - 1;

/// Keys past the last one the kernel may read in a partial final tile.
inline constexpr int64_t KV_TILE_OVERREAD = 31;

/// Sequence lengths the kernel's int32 mask arithmetic is sound for: a row's attendable
/// range is computed as q + offset + bound with every term below 2^29, so the sum stays
/// inside int32, as does the "no window" sentinel's distance from any kv index.
inline constexpr int64_t SEQ_LEN_LIMIT = 536870912LL; // 2^29

/// Kernel geometry the generator records: each wave owns 16 query rows, 32 lanes wide.
inline constexpr int64_t WAVE_ROWS = 16;

/// The widest access the kernels make from an operand's base: a 16-byte vector.
inline constexpr int64_t VECTOR_BYTES = 16;
inline constexpr int64_t WAVE_SIZE = 32;
/// The kernel's KV sub-tile; block_n must be a multiple of it.
inline constexpr int64_t KV_SUB_TILE = 32;

/// K and V are read through a buffer descriptor scoped to one (batch, kv head) slice,
/// whose num_records is 32 bits. Held to 2^31 rather than 2^32 so no part of the path
/// has to be trusted to treat it as unsigned.
inline constexpr int64_t KV_SLICE_BYTE_LIMIT = 2147483648LL; // 2^31

/// Where the kernel's `scale` argument comes from.
enum class ScaleSource : int64_t
{
    /// attn_scale_value, or 1.0 when the graph gives none -- cuDNN's default, and the one
    /// hipDNN's CPU reference applies (CpuFpReferenceSdpa.hpp).
    BAKED = 0,
    /// A pass-by-value scalar tensor, resolved at execute.
    TENSOR = 1,
};

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/**
 * @brief Total over an UNVALIDATED graph: rank, stride/dim agreement and positive extents,
 *        checked before anything indexes an axis; and the one layout rule the kernel has.
 *
 * The kernel takes a (batch, sequence, head) stride for every operand, so BSHD, BHSD and
 * any other arrangement are equally served -- with one exception. Each lane reads a row's
 * head-dim values as one contiguous vector, so the head-dim stride must be 1. Every other
 * stride must be positive on an axis with more than one element: the K/V slice extent is
 * computed from the sequence stride, and a non-positive one would place rows outside the
 * bounds-checked range the kernel builds from it. A unit-extent axis is exempt; its index
 * is always 0.
 */
inline bool isServableOperand(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }

    for(flatbuffers::uoffset_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        if(dims->Get(axis) <= 0 || dims->Get(axis) > INT32_LIMIT)
        {
            return false;
        }
        if(dims->Get(axis) > 1 && strides->Get(axis) <= 0)
        {
            return false;
        }
    }

    if(strides->Get(HEAD_DIM_AXIS) != 1)
    {
        return false;
    }

    // A ragged (THD) operand packs variable-length sequences end to end, which the kernel
    // would read as one dense batch -- wrong values, not an error. Declined until varlen
    // is implemented.
    if(tensor.ragged_offset_tensor_uid().has_value())
    {
        return false;
    }

    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/// The mask a graph asks for, in the kernels' terms (resolveMaskBounds): kv is
/// masked when kv >= q + 1 + align + right (right >= 0), or kv < q + align - left
/// (left >= 0), where align is 0 top-left and Skv - Sq bottom-right.
struct SdpaMask
{
    int64_t right = UNBOUNDED;
    int64_t left = UNBOUNDED;
    bool alignBottomRight = false;
};

/**
 * @brief The graph's mask, or nullopt for one the kernel cannot apply.
 *
 * The kernel applies the reference's two-sided rule directly, so every bound pair is
 * served: causal (right 0), a band (right > 0), a sliding window (left >= 0), and either
 * diagonal alignment. The deprecated causal booleans act as if their setter ran first, as
 * cuDNN's do and the reference resolves them: right 0, a right_bound the graph also sets
 * wins, and a left_bound is kept (a sliding window, as gpt-oss and Mistral graphs set it).
 * causal_mask keeps the stored alignment, so diagonal_alignment refines it;
 * causal_mask_bottom_right is always bottom-right. Both booleans together, and an
 * alignment the schema does not name, decline.
 */
inline std::optional<SdpaMask> maskFor(const data_objects::SdpaAttributes& attributes)
{
    const auto bounds = resolveMaskBounds(attributes);
    if(!bounds.has_value())
    {
        return std::nullopt;
    }
    return SdpaMask{bounds->right, bounds->left, bounds->alignBottomRight};
}

/// The softmax-statistics output the kernel can write: f32 `[B, H, Sq, 1]`, a real device
/// tensor, with positive strides on its non-unit axes (the kernel takes its batch, sequence
/// and head strides).
inline bool isServableStats(const data_objects::TensorAttributes& stats,
                            int64_t batch,
                            int64_t heads,
                            int64_t seqLenQ)
{
    const auto* dims = stats.dims();
    const auto* strides = stats.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }
    if(dims->Get(BATCH_AXIS) != batch || dims->Get(HEAD_AXIS) != heads
       || dims->Get(SEQ_AXIS) != seqLenQ || dims->Get(HEAD_DIM_AXIS) != 1)
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        if(dims->Get(axis) > 1 && strides->Get(axis) <= 0)
        {
            return false;
        }
    }
    return stats.data_type() == data_objects::DataType::FLOAT && !stats.virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&stats)
           && !stats.ragged_offset_tensor_uid().has_value();
}

/// The additive bias, right-aligned to `[B, H, Sq, Skv]` as the reference reads it (rank 1 to
/// 4, a size-1 or absent axis broadcast), as the kernel's (batch, head, query, key) strides:
/// 0 on a broadcast axis. nullopt when the kernel cannot read it: neither f32 nor the query's
/// dtype (each has its own objects, `bias_q` 0 and 1), a dimension that
/// is neither the target's nor 1, a non-positive stride on a real axis, a slice past the
/// kernel's 32-bit offsets, or not a plain device tensor.
inline std::optional<BiasStrides> servableBiasStrides(const data_objects::TensorAttributes& bias,
                                                      int64_t batch,
                                                      int64_t heads,
                                                      int64_t seqLenQ,
                                                      int64_t seqLenKv,
                                                      data_objects::DataType queryType)
{
    const auto* dims = bias.dims();
    const auto* strides = bias.strides();
    if(dims == nullptr || strides == nullptr || dims->empty() || dims->size() > SDPA_RANK
       || strides->size() != dims->size())
    {
        return std::nullopt;
    }
    if((bias.data_type() != data_objects::DataType::FLOAT && bias.data_type() != queryType)
       || bias.virtual_() || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&bias)
       || bias.ragged_offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }

    const std::array<int64_t, SDPA_RANK> target{batch, heads, seqLenQ, seqLenKv};
    BiasStrides result{0, 0, 0, 0};
    const auto rank = static_cast<size_t>(dims->size());
    for(size_t axis = 0; axis < rank; ++axis)
    {
        const auto targetAxis = SDPA_RANK - rank + axis;
        const auto dim = dims->Get(static_cast<flatbuffers::uoffset_t>(axis));
        const auto stride = strides->Get(static_cast<flatbuffers::uoffset_t>(axis));
        if(dim == 1)
        {
            continue;
        }
        if(dim != target[targetAxis] || stride <= 0)
        {
            return std::nullopt;
        }
        result[targetAxis] = stride;
    }

    // The kernel reads whole KV tiles, up to one tile past the last key (those reads come
    // back as zero), so the offsets it forms reach that far.
    const auto lastQuery = checkedProduct({seqLenQ - 1, result[2]});
    const auto lastKey = checkedProduct({seqLenKv - 1 + KV_TILE_OVERREAD, result[3]});
    if(!lastQuery.has_value() || !lastKey.has_value()
       || *lastQuery > BIAS_SLICE_ELEMENT_LIMIT - *lastKey)
    {
        return std::nullopt;
    }
    return result;
}

/// The graph's shape, read from Q and K. Callers must have validated both operands.
struct SdpaProblem
{
    int64_t batch = 0;
    int64_t numHeads = 0;
    int64_t numKvHeads = 0;
    /// V's head count; K's unless set from V, which may differ (each divides numHeads).
    int64_t numVHeads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t headDim = 0;
    /// V's and O's head dim; Q/K's unless set from V, which may be narrower (MLA).
    int64_t headDimV = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

inline SdpaProblem problemFor(const data_objects::TensorAttributes& q,
                              const data_objects::TensorAttributes& k)
{
    SdpaProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headDim = q.dims()->Get(HEAD_DIM_AXIS);
    problem.headDimV = problem.headDim;
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.numVHeads = problem.numKvHeads;
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

inline bool hasDims(const data_objects::TensorAttributes& tensor,
                    int64_t batch,
                    int64_t heads,
                    int64_t sequence,
                    int64_t headDim)
{
    const auto* dims = tensor.dims();
    return dims->Get(BATCH_AXIS) == batch && dims->Get(HEAD_AXIS) == heads
           && dims->Get(SEQ_AXIS) == sequence && dims->Get(HEAD_DIM_AXIS) == headDim;
}

/// Does one (batch, kv head) slice of @p tensor fit the kernel's bounds-checked range?
inline bool
    kvSliceFits(const data_objects::TensorAttributes& tensor, int64_t seqLenKv, int64_t headDim)
{
    const auto rows = checkedProduct({seqLenKv - 1, tensor.strides()->Get(SEQ_AXIS)});
    if(!rows.has_value())
    {
        return false;
    }
    const auto bytes = checkedProduct({*rows + headDim, BYTES_PER_ELEMENT});
    return bytes.has_value() && *bytes < KV_SLICE_BYTE_LIMIT;
}

/**
 * @brief Graph-scoped applicability: is this a single SDPA-forward node the pre-built
 *        flyDSL kernels compute exactly?
 *
 * Refuses rather than approximates: every optional feature the kernel does not implement
 * is declined explicitly, so a graph asking for one falls through to an engine that can
 * honour it rather than being served without it.
 *
 * @warning Returning std::nullopt empties this engine's whole catalog for the graph.
 */

inline std::optional<BoundTokens> flydslSdpaRdnaGraphMatches(const MatchContext& context)
{
    // --- 1. One SDPA-forward node; each kernel serves one complete graph.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;
    if(!computesInFloat(context))
    {
        return std::nullopt;
    }

    // --- 2. Operands, each total-checked before any axis is read.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }
    if(!isServableOperand(*q) || !isServableOperand(*k) || !isServableOperand(*v)
       || !isServableOperand(*o))
    {
        return std::nullopt;
    }
    // Rows move as 16-byte vectors from each operand's base; O is written by many
    // workgroups at once, so no two of its elements may share an address.
    if(!declaresAlignment(*q, VECTOR_BYTES) || !declaresAlignment(*k, VECTOR_BYTES)
       || !declaresAlignment(*v, VECTOR_BYTES) || !declaresAlignment(*o, VECTOR_BYTES)
       || !elementsDoNotOverlap(*o))
    {
        return std::nullopt;
    }

    // --- 3. One dtype throughout, and one the kernel was built for.
    const auto problem = problemFor(*q, *k);
    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    if(!kernelDataTypeName(problem.dataType).has_value())
    {
        return std::nullopt;
    }

    // --- 4. Cross-tensor shape. No broadcasting: K and V share batch and sequence, K has
    // Q's head dim, and O is Q's shape with V's head dim. V's head count may differ from K's,
    // and its head dim may be narrower than Q/K's (MLA); which head-dim pairs run is the
    // objects' to say.
    const auto numVHeads = v->dims()->Get(HEAD_AXIS);
    const auto headDimV = v->dims()->Get(HEAD_DIM_AXIS);
    if(headDimV <= 0 || headDimV > problem.headDim
       || !hasDims(*k, problem.batch, problem.numKvHeads, problem.seqLenKv, problem.headDim)
       || !hasDims(*v, problem.batch, numVHeads, problem.seqLenKv, headDimV)
       || !hasDims(*o, problem.batch, problem.numHeads, problem.seqLenQ, headDimV))
    {
        return std::nullopt;
    }

    // GQA/MQA: the kernel maps a query head to its K head and to its V head by integer
    // division, so a non-divisible pair would silently attend to the wrong heads.
    if(problem.numHeads % problem.numKvHeads != 0 || problem.numHeads % numVHeads != 0)
    {
        return std::nullopt;
    }

    // --- 5. Range limits the kernel's arithmetic assumes. The grid is one workgroup per
    // (batch, query tile, head); its smallest tile is one wave's rows, so this bound holds
    // for every candidate whatever its block_m.
    const auto queryTiles = (problem.seqLenQ + WAVE_ROWS - 1) / WAVE_ROWS;
    const auto grid = checkedProduct({problem.batch, queryTiles, problem.numHeads});
    if(!grid.has_value() || *grid > INT32_LIMIT)
    {
        return std::nullopt;
    }
    if(!kvSliceFits(*k, problem.seqLenKv, problem.headDim)
       || !kvSliceFits(*v, problem.seqLenKv, headDimV))
    {
        return std::nullopt;
    }

    // --- 6. The mask. A row no key reaches -- a narrow window, or bottom-right causal with
    // more queries than keys -- gets O = 0 and LSE = -inf, as the reference defines it.
    if(problem.seqLenQ > SEQ_LEN_LIMIT || problem.seqLenKv > SEQ_LEN_LIMIT)
    {
        return std::nullopt;
    }
    const auto mask = maskFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }
    // A bound past Sq + Skv masks nothing more than one at it; clamped so the kernel's
    // int32 sums stay in range. Skv alone is not enough for the right bound: at the
    // bottom-right corner with Sq > Skv the diagonal starts at Skv - Sq < 0, and the first
    // row reaches every key only once the bound is at least Sq - 1.
    const int64_t causal = mask->right >= 0 ? 1 : 0;
    const int64_t rightBound = std::min(mask->right, problem.seqLenQ + problem.seqLenKv);
    const int64_t leftBound = std::min(mask->left, problem.seqLenQ + problem.seqLenKv);

    // --- 7. Every optional feature the kernel does not implement, declined explicitly.

    // Additive attention bias: served by the has_bias objects.
    int64_t biasUid = NO_BIAS;
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        const auto* bias = findTensor(context, attributes.attn_mask_tensor_uid().value());
        if(bias == nullptr || !declaresAlignment(*bias, VECTOR_BYTES)
           || !servableBiasStrides(*bias,
                                   problem.batch,
                                   problem.numHeads,
                                   problem.seqLenQ,
                                   problem.seqLenKv,
                                   problem.dataType)
                   .has_value())
        {
            return std::nullopt;
        }
        biasUid = attributes.attn_mask_tensor_uid().value();
    }
    // varlen, both spellings.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Dropout.
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return std::nullopt;
    }
    // Paged KV.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        return std::nullopt;
    }
    // An attention sink: one logit per query head, in the query's dtype, served by the
    // has_sink objects.
    int64_t sinkUid = NO_SINK;
    int64_t sinkStride = 0;
    if(attributes.sink_token_tensor_uid().has_value())
    {
        const auto stride = servableSinkStride(
            context, *attributes.sink_token_tensor_uid(), problem.numHeads, problem.dataType);
        if(!stride.has_value())
        {
            return std::nullopt;
        }
        sinkUid = *attributes.sink_token_tensor_uid();
        sinkStride = *stride;
    }
    // Block-sparse masks.
    if(attributes.block_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // FP8 quantization.
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Softmax statistics: the LSE output is served; the split max / sum-exp outputs and the
    // RNG dump are not. Asking for stats without naming a tensor to write them to is a graph
    // the kernel cannot satisfy.
    if(attributes.max_tensor_uid().has_value() || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    int64_t statsUid = NO_STATS;
    if(attributes.stats_tensor_uid().has_value())
    {
        // An LSE tensor with generate_stats explicitly false would be written anyway: the
        // two disagree, so the graph declines rather than one being ignored.
        const auto* stats = findTensor(context, attributes.stats_tensor_uid().value());
        if(stats == nullptr || !declaresAlignment(*stats, sizeof(float))
           || !elementsDoNotOverlap(*stats)
           || (attributes.generate_stats().has_value() && !attributes.generate_stats().value())
           || !isServableStats(*stats, problem.batch, problem.numHeads, problem.seqLenQ))
        {
            return std::nullopt;
        }
        statsUid = attributes.stats_tensor_uid().value();
    }
    else if(attributes.generate_stats().has_value() && attributes.generate_stats().value())
    {
        return std::nullopt;
    }
    // ALiBi slopes and padding masks.
    if(attributes.alibi_mask() || attributes.padding_mask())
    {
        return std::nullopt;
    }
    // mma_core_mode is the MMA operand precision. The kernel's WMMA operands are the
    // graph's own fp16/bf16 inputs, so UNSET, HALF and BFLOAT16 describe what it runs.
    // HALF is accepted on bf16 graphs too, because the cuDNN-compat shim writes HALF when
    // the caller leaves the field unset. An allow-list, so a value added later declines.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint; AUTO leaves the choice to us.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }

    // --- 8. The softmax scale: a baked value, a pass-by-value tensor, or the default.
    ScaleSource scaleSource = ScaleSource::BAKED;
    int64_t scaleToken = 0;
    if(attributes.scale_tensor_uid().has_value())
    {
        // Only a host scalar can become a kernarg; a device-resident scale would need a
        // device read the kernel does not do. A value set as well would be ignored, so the
        // two together decline.
        if(attributes.attn_scale_value().has_value()
           || !isServableScaleTensor(context, attributes.scale_tensor_uid().value()))
        {
            return std::nullopt;
        }
        scaleSource = ScaleSource::TENSOR;
        scaleToken = attributes.scale_tensor_uid().value();
    }
    else
    {
        // An absent attn_scale_value is 1.0, cuDNN's default: its SDPA node multiplies by
        // attn_scale only when one is set. hipDNN's references read it the same way.
        const float scale = attributes.attn_scale_value().value_or(1.0F);
        if(!isServableScale(static_cast<double>(scale)))
        {
            return std::nullopt;
        }
        int32_t scaleBits = 0;
        static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
        std::memcpy(&scaleBits, &scale, sizeof(scale));
        scaleToken = static_cast<int64_t>(scaleBits);
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(CAUSAL_TOKEN)] = causal;
    bound[std::string(RIGHT_BOUND_TOKEN)] = rightBound;
    bound[std::string(LEFT_BOUND_TOKEN)] = leftBound;
    bound[std::string(ALIGN_BOTTOM_RIGHT_TOKEN)] = mask->alignBottomRight ? 1 : 0;
    bound[std::string(STATS_TOKEN)] = statsUid;
    bound[std::string(BIAS_TOKEN)] = biasUid;
    bound[std::string(SINK_TOKEN)] = sinkUid;
    bound[std::string(SINK_STRIDE_TOKEN)] = sinkStride;
    bound[std::string(SCALE_SOURCE_TOKEN)] = static_cast<int64_t>(scaleSource);
    bound[std::string(SCALE_TOKEN)] = scaleToken;
    return bound;
}

/// The bindings a match established, re-read for dispatch.
struct FlydslSdpaRdnaBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t rightBound = UNBOUNDED;
    int64_t leftBound = UNBOUNDED;
    int64_t alignBottomRight = 0;
    int64_t stats = NO_STATS;
    int64_t bias = NO_BIAS;
    int64_t sink = NO_SINK;
    int64_t sinkStride = 0;
    ScaleSource scaleSource = ScaleSource::BAKED;
    int64_t scale = 0;
};

inline FlydslSdpaRdnaBinding flydslSdpaRdnaBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_sdpa_rdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    FlydslSdpaRdnaBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.rightBound = read(RIGHT_BOUND_TOKEN);
    binding.leftBound = read(LEFT_BOUND_TOKEN);
    binding.alignBottomRight = read(ALIGN_BOTTOM_RIGHT_TOKEN);
    binding.stats = read(STATS_TOKEN);
    binding.bias = read(BIAS_TOKEN);
    binding.sink = read(SINK_TOKEN);
    binding.sinkStride = read(SINK_STRIDE_TOKEN);
    binding.scaleSource = static_cast<ScaleSource>(read(SCALE_SOURCE_TOKEN));
    binding.scale = read(SCALE_TOKEN);
    return binding;
}

/**
 * @brief The kernel-scoped checks both families make: dtype; head_dim (exact, or a generic
 *        object's runtime head_dim up to its head_dim_max); V's head_dim; the bias and its
 *        dtype; the sink; and the causal variant. Fills @p problem from the graph for the
 *        family's own checks.
 */
inline bool familyMatches(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel,
                          SdpaProblem& problem)
{
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return false;
    }
    const auto* q = findTensor(context, attributesPtr->q_tensor_uid());
    const auto* k = findTensor(context, attributesPtr->k_tensor_uid());
    if(q == nullptr || k == nullptr)
    {
        return false;
    }
    const auto* v = findTensor(context, attributesPtr->v_tensor_uid());
    if(v == nullptr)
    {
        return false;
    }
    problem = problemFor(*q, *k);
    problem.numVHeads = v->dims()->Get(HEAD_AXIS);
    problem.headDimV = v->dims()->Get(HEAD_DIM_AXIS);

    const auto dataTypeName = kernelDataTypeName(problem.dataType);
    if(!dataTypeName.has_value() || stringMetadata(kernel, DTYPE_FIELD) != *dataTypeName)
    {
        return false;
    }

    // A specialized object serves exactly its head_dim; a generic one any multiple of 8 up
    // to its head_dim_max, which priority ranks below the specialized ones.
    const auto bakedHeadDim = integerMetadata(kernel, HEAD_DIM_FIELD);
    const auto headDimMax = integerMetadata(kernel, HEAD_DIM_MAX_FIELD);
    if(!bakedHeadDim.has_value() || !headDimMax.has_value() || *headDimMax <= 0)
    {
        return false;
    }
    const bool servesHead
        = *bakedHeadDim == GENERIC_HEAD_DIM
              ? problem.headDim % GENERIC_HEAD_DIM_ALIGN == 0 && problem.headDim <= *headDimMax
              : *bakedHeadDim == problem.headDim && *headDimMax == problem.headDim;
    if(!servesHead)
    {
        return false;
    }
    // V's head dim: an MLA object's own, every other object's equal to Q/K's (a generic
    // one's too, both being its runtime head_dim).
    const auto bakedHeadDimV = integerMetadata(kernel, HEAD_DIM_V_FIELD);
    if(!bakedHeadDimV.has_value()
       || (*bakedHeadDimV == GENERIC_HEAD_DIM ? problem.headDimV != problem.headDim
                                              : *bakedHeadDimV != problem.headDimV))
    {
        return false;
    }

    // A bias object for a graph with a bias, a plain one for a graph without.
    const auto bias = tryGetBoundInt(bound, BIAS_TOKEN);
    if(!bias.has_value() || integerMetadata(kernel, HAS_BIAS_FIELD) != (*bias == NO_BIAS ? 0 : 1))
    {
        return false;
    }
    // A bias in the query's dtype runs on the `bias_q` objects, an f32 one on the others.
    if(*bias != NO_BIAS)
    {
        const auto* biasTensor = findTensor(context, *bias);
        const auto biasInQueryType
            = biasTensor != nullptr && biasTensor->data_type() != data_objects::DataType::FLOAT;
        if(biasTensor == nullptr
           || integerMetadata(kernel, BIAS_Q_FIELD) != (biasInQueryType ? 1 : 0))
        {
            return false;
        }
    }

    // A sink object for a graph with an attention sink, a plain one for a graph without.
    const auto sink = tryGetBoundInt(bound, SINK_TOKEN);
    if(!sink.has_value() || integerMetadata(kernel, HAS_SINK_FIELD) != (*sink == NO_SINK ? 0 : 1))
    {
        return false;
    }

    const auto causal = tryGetBoundInt(bound, CAUSAL_TOKEN);
    return causal.has_value() && integerMetadata(kernel, CAUSAL_FIELD) == *causal;
}

/// One operand's (batch, sequence, head) element strides, in the kernel's argument order.
struct OperandStrides
{
    int64_t batch = 0;
    int64_t sequence = 0;
    int64_t head = 0;
};

inline OperandStrides stridesOf(const data_objects::TensorAttributes& tensor)
{
    const auto* strides = tensor.strides();
    return OperandStrides{
        strides->Get(BATCH_AXIS), strides->Get(SEQ_AXIS), strides->Get(HEAD_AXIS)};
}

inline const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    return requireSdpaTensor(context, uid, "flydsl_sdpa_rdna");
}

/// The softmax scale for a launch: the pass-by-value scalar when the graph gives one, read
/// now, else the f32 the match baked into the scale token.
inline float launchScale(const FlydslSdpaRdnaBinding& binding,
                         const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scaleOperand,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers)
{
    float scale = 0.0F;
    if(scaleOperand.has_value())
    {
        scale = static_cast<float>(
            hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                *scaleOperand, deviceBuffers, numDeviceBuffers)));
        // A pass-by-value scalar, read from host memory. A constant scale was checked at
        // match time; this one arrives now.
        if(!isServableScale(static_cast<double>(scale)))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna: the runtime softmax scale must be positive and at most 2^20");
        }
    }
    else
    {
        const auto scaleBits = static_cast<int32_t>(binding.scale);
        std::memcpy(&scale, &scaleBits, sizeof(scale));
    }
    return scale;
}

/// A bound tensor's device pointer, or nullptr for a token that names none (NO_STATS,
/// NO_BIAS, NO_SINK are all -1).
inline void* optionalBuffer(int64_t uid,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers)
{
    return uid < 0 ? nullptr
                   : hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_rdna

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
