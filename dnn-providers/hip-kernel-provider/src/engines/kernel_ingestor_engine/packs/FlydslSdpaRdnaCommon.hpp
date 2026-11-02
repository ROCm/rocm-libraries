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
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaRngDump.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaVarlen.hpp"

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

/// 1 for a pair tier (prefill kernel modification 32): a generic object whose V and O are as
/// wide as a runtime width of their own rather than Q/K's, each a multiple of
/// GENERIC_HEAD_DIM_ALIGN up to head_dim_max. It serves only graphs whose two widths differ.
/// Absent (the decode objects) is 0.
inline constexpr std::string_view V_MASK_FIELD = "v_mask";
/// On a pair tier the runtime head_dim argument carries Q/K's width in its low bits and V's
/// above them.
inline constexpr int V_MASK_WIDTH_SHIFT = 16;

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
/// 1 when a graph without a bias needs scaled scores (runsOnAZeroBias: a zero, negative or
/// subnormal scale), so the RDNA prefill and decode engines run it on their bias objects with
/// a zero bias; else 0. The aiw objects scale every score first, and ignore it.
inline constexpr std::string_view ZERO_BIAS_TOKEN = "flydsl_sdpa_rdna.bias.zero";
/// The uid of the attention sink, or NO_SINK, its element stride across query heads, and its
/// dtype relative to the query's (SinkDtype: 0 the query's, 1 FLOAT, 2 the other 16-bit one).
inline constexpr std::string_view SINK_TOKEN = "flydsl_sdpa_rdna.sink.uid";
inline constexpr std::string_view SINK_STRIDE_TOKEN = "flydsl_sdpa_rdna.sink.stride";
inline constexpr std::string_view SINK_DTYPE_TOKEN = "flydsl_sdpa_rdna.sink.dtype";
/// Where the softmax scale comes from; see ScaleSource.
inline constexpr std::string_view SCALE_SOURCE_TOKEN = "flydsl_sdpa_rdna.scale_source";
/// The scale's f32 bit pattern when baked (BoundTokens carry int64_t), else the uid of
/// the pass-by-value scale tensor.
inline constexpr std::string_view SCALE_TOKEN = "flydsl_sdpa_rdna.scale";
/// The uids of the per-batch query and key lengths under padding, or NO_SEQ_LEN for a dense
/// graph (FlydslSdpaVarlen.hpp).
inline constexpr std::string_view SEQ_LEN_Q_TOKEN = "flydsl_sdpa_rdna.seq_len_q.uid";
inline constexpr std::string_view SEQ_LEN_KV_TOKEN = "flydsl_sdpa_rdna.seq_len_kv.uid";
/// The uids of the ragged-offset tables packed Q/O and K/V go through, or NO_SEQ_LEN for
/// unpacked operands (FlydslSdpaVarlen.hpp servablePackedVarlen).
inline constexpr std::string_view RAGGED_OFFSET_Q_TOKEN = "flydsl_sdpa_rdna.ragged_offset_q.uid";
inline constexpr std::string_view RAGGED_OFFSET_KV_TOKEN = "flydsl_sdpa_rdna.ragged_offset_kv.uid";
/// The dropout probability's f32 bit pattern (0 for none), and the uids of the Philox seed and
/// offset tensors, or NO_DROPOUT_TENSOR (FlydslSdpaDropout.hpp).
inline constexpr std::string_view DROPOUT_P_TOKEN = "flydsl_sdpa_rdna.dropout_p";
inline constexpr std::string_view DROPOUT_SEED_TOKEN = "flydsl_sdpa_rdna.dropout_seed.uid";
inline constexpr std::string_view DROPOUT_OFFSET_TOKEN = "flydsl_sdpa_rdna.dropout_offset.uid";
/// The uid of the RNG dump the dropout mask writer fills, or NO_RNG_DUMP (FlydslSdpaRngDump.hpp).
inline constexpr std::string_view RNG_DUMP_TOKEN = "flydsl_sdpa_rdna.rng_dump.uid";
/// 0 or 1: whether the object drops attention probabilities (AOTriton's ENABLE_DROPOUT). Baked,
/// so a graph that drops and one that does not each select their own object; absent is 0.
inline constexpr std::string_view HAS_DROPOUT_FIELD = "has_dropout";

/// A float's bits as a bound token (BoundTokens carry int64_t), and back.
inline int64_t floatToken(float value)
{
    int32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit to round-trip");
    std::memcpy(&bits, &value, sizeof(value));
    return static_cast<int64_t>(bits);
}

inline float tokenFloat(int64_t token)
{
    const auto bits = static_cast<int32_t>(token);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

/// The bias's (batch, head, query, key) element strides as the kernel takes them.
using BiasStrides = std::array<int64_t, 4>;

/// The zero bias's strides (runsOnAZeroBias): 0 on every axis, so every score reads its first
/// element, whatever the batch, row or key -- packed sequences included.
inline constexpr BiasStrides ZERO_BIAS_STRIDES{0, 0, 0, 0};

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
    // A ragged (THD) operand is servableVarlen's to judge (FlydslSdpaVarlen.hpp).

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
 * served: causal (right 0), a band (right > 0), a sliding window (a left_bound L >= 1, which
 * keeps L keys up to and including the diagonal; resolveMaskBounds stores L - 1), and either
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
/// and head strides). Dense, or, under packed operands (@p raggedOffsetQ not NO_SEQ_LEN),
/// packed through Q's own table like Q (isPackedThrough): one row per query token, where a
/// dense LSE would need each sequence's batch, which a packed launch does not index.
inline bool isServableStats(const data_objects::TensorAttributes& stats,
                            int64_t batch,
                            int64_t heads,
                            int64_t seqLenQ,
                            int64_t raggedOffsetQ = NO_SEQ_LEN)
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
    const bool layoutServed = raggedOffsetQ == NO_SEQ_LEN
                                  ? !stats.ragged_offset_tensor_uid().has_value()
                                  : isPackedThrough(stats, raggedOffsetQ);
    return stats.data_type() == data_objects::DataType::FLOAT && !stats.virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&stats) && layoutServed;
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

/// Does one (batch, head) slice of @p tensor fit the kernel's bounds-checked range? K and V
/// are read through one; so is Q by an f32 object (prefill kernel modification 34).
inline bool kvSliceFits(const data_objects::TensorAttributes& tensor,
                        int64_t seqLenKv,
                        int64_t headDim,
                        int64_t elementBytes = BYTES_PER_ELEMENT)
{
    const auto rows = checkedProduct({seqLenKv - 1, tensor.strides()->Get(SEQ_AXIS)});
    if(!rows.has_value())
    {
        return false;
    }
    const auto bytes = checkedProduct({*rows + headDim, elementBytes});
    return bytes.has_value() && *bytes < KV_SLICE_BYTE_LIMIT;
}

/// The RDNA objects' metadata spelling of a graph dtype: the 16-bit ones every RDNA family
/// takes, and exact f32 (prefill kernel modification 34), which only the RDNA prefill engine
/// builds and its graph match admits (flydslSdpaRdnaPrefillGraphMatches).
inline std::optional<std::string> rdnaKernelDataTypeName(data_objects::DataType dataType)
{
    if(dataType == data_objects::DataType::FLOAT)
    {
        return std::string("f32");
    }
    return kernelDataTypeName(dataType);
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

inline std::optional<BoundTokens> flydslSdpaRdnaGraphMatchesFor(const MatchContext& context,
                                                                bool servesF32)
{
    // --- 1. One SDPA-forward node; each kernel serves one complete graph.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;
    if(!computesInFloat(context) || usesUnreadSchemaFields(context))
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
    // FLOAT runs on the RDNA prefill engine's exact f32 objects (prefill kernel modification
    // 34); the decode and aiw engines decline it.
    const bool f32 = problem.dataType == data_objects::DataType::FLOAT;
    if(f32 ? !servesF32 : !kernelDataTypeName(problem.dataType).has_value())
    {
        return std::nullopt;
    }
    const int64_t elementBytes = f32 ? int64_t{4} : BYTES_PER_ELEMENT;

    // --- 4. Cross-tensor shape. No broadcasting: K and V share batch and sequence, K has
    // Q's head dim, and O is Q's shape with V's head dim. V's head count may differ from K's,
    // and its head dim from Q/K's, narrower (MLA) or wider; which head-dim pairs run is the
    // objects' to say (the RDNA objects' baked head_dim_v, the aiw padded tiers' two runtime
    // widths).
    const auto numVHeads = v->dims()->Get(HEAD_AXIS);
    const auto headDimV = v->dims()->Get(HEAD_DIM_AXIS);
    if(headDimV <= 0
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
    // An f32 object reads Q through a bounds-checked range too.
    if(!kvSliceFits(*k, problem.seqLenKv, problem.headDim, elementBytes)
       || !kvSliceFits(*v, problem.seqLenKv, headDimV, elementBytes)
       || (f32 && !kvSliceFits(*q, problem.seqLenQ, problem.headDim, elementBytes)))
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
    // Variable sequence lengths (padding, or packed THD Q/K/V/O) and dropout (the probability,
    // seed, offset, mask, scale and RNG dump): each feature's rules live in its own header.
    const auto varlen = hasRaggedOperand(context, attributes)
                            ? servablePackedVarlenWithLengths(context, attributes, problem.batch)
                            : servableVarlenAsReferenceReads(context, attributes, problem.batch);
    const auto dropout = servableDropout(context, attributes);
    if(!varlen.has_value() || !dropout.has_value())
    {
        return std::nullopt;
    }
    // Dropout over packed sequences declines, as on CDNA. Under padding the kernels draw the
    // mask on the launch's extents (AOTriton's max_seqlen_q/k), so dense, padded and packed
    // storage of one problem share one mask; padded rows are filled as without dropout.
    if(varlen->packed() && dropout->active())
    {
        return std::nullopt;
    }
    // The RNG dump: the keep mask a dropping launch applied, written by the mask writer.
    const auto rngDump = servableRngDump(context,
                                         attributes,
                                         *dropout,
                                         *varlen,
                                         problem.batch,
                                         problem.numHeads,
                                         problem.seqLenQ,
                                         problem.seqLenKv);
    if(!rngDump.has_value())
    {
        return std::nullopt;
    }
    // A bias over packed sequences: the bias is laid out by batch, which a packed launch does
    // not index.
    if(varlen->packed() && biasUid != NO_BIAS)
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
    // An attention sink: one logit per query head, FLOAT or in either 16-bit dtype (SEM-022),
    // served by the has_sink objects.
    int64_t sinkUid = NO_SINK;
    int64_t sinkStride = 0;
    auto sinkDtype = SinkDtype::QUERY;
    if(attributes.sink_token_tensor_uid().has_value())
    {
        // No f32 object adds a sink.
        if(f32)
        {
            return std::nullopt;
        }
        const auto sink = servableSink(
            context, *attributes.sink_token_tensor_uid(), problem.numHeads, problem.dataType);
        if(!sink.has_value())
        {
            return std::nullopt;
        }
        sinkUid = *attributes.sink_token_tensor_uid();
        sinkStride = sink->stride;
        sinkDtype = sink->dtype;
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
    // Softmax statistics: the LSE output is served; the split max / sum-exp outputs are not
    // (the RNG dump is servableRngDump's). Asking for stats without naming a tensor to write
    // them to is a graph the kernel cannot satisfy.
    if(attributes.max_tensor_uid().has_value() || attributes.sum_exp_tensor_uid().has_value())
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
           || !isServableStats(
               *stats, problem.batch, problem.numHeads, problem.seqLenQ, varlen->raggedOffsetQ))
        {
            return std::nullopt;
        }
        statsUid = attributes.stats_tensor_uid().value();
    }
    else if(attributes.generate_stats().has_value() && attributes.generate_stats().value())
    {
        return std::nullopt;
    }
    // ALiBi slopes (the padding mask is servableVarlen's).
    if(attributes.alibi_mask())
    {
        return std::nullopt;
    }
    // mma_core_mode is the MMA operand precision. The kernel's WMMA operands are the
    // graph's own fp16/bf16 inputs, so UNSET, HALF and BFLOAT16 describe what it runs.
    // HALF is accepted on bf16 graphs too, because the cuDNN-compat shim writes HALF when
    // the caller leaves the field unset. An f32 graph takes only UNSET or FLOAT: its objects
    // compute exact f32 products, and HALF or BFLOAT16 would ask for a reduced-precision
    // MMA they do not run. An allow-list, so a value added later declines.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(f32 ? mmaCoreMode != data_objects::DataType::UNSET
                 && mmaCoreMode != data_objects::DataType::FLOAT
           : mmaCoreMode != data_objects::DataType::UNSET
                 && mmaCoreMode != data_objects::DataType::HALF
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
    bound[std::string(ZERO_BIAS_TOKEN)] = runsOnAZeroBias(context, attributes) ? 1 : 0;
    bound[std::string(SINK_TOKEN)] = sinkUid;
    bound[std::string(SINK_STRIDE_TOKEN)] = sinkStride;
    bound[std::string(SINK_DTYPE_TOKEN)] = static_cast<int64_t>(sinkDtype);
    bound[std::string(SCALE_SOURCE_TOKEN)] = static_cast<int64_t>(scaleSource);
    bound[std::string(SCALE_TOKEN)] = scaleToken;
    bound[std::string(SEQ_LEN_Q_TOKEN)] = varlen->seqLenQ;
    bound[std::string(SEQ_LEN_KV_TOKEN)] = varlen->seqLenKv;
    bound[std::string(RAGGED_OFFSET_Q_TOKEN)] = varlen->raggedOffsetQ;
    bound[std::string(RAGGED_OFFSET_KV_TOKEN)] = varlen->raggedOffsetKv;
    bound[std::string(DROPOUT_P_TOKEN)] = floatToken(dropout->probability);
    bound[std::string(DROPOUT_SEED_TOKEN)] = dropout->seed;
    bound[std::string(DROPOUT_OFFSET_TOKEN)] = dropout->offset;
    bound[std::string(RNG_DUMP_TOKEN)] = *rngDump;
    return bound;
}

/// The graph match of the RDNA decode and aiw engines: 16-bit graphs only.
inline std::optional<BoundTokens> flydslSdpaRdnaGraphMatches(const MatchContext& context)
{
    return flydslSdpaRdnaGraphMatchesFor(context, false);
}

/// The RDNA prefill engine's graph match: 16-bit graphs, and FLOAT ones for its exact f32
/// objects (prefill kernel modification 34).
inline std::optional<BoundTokens> flydslSdpaRdnaPrefillGraphMatches(const MatchContext& context)
{
    return flydslSdpaRdnaGraphMatchesFor(context, true);
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
    /// Whether the graph runs on the zero bias (ZERO_BIAS_TOKEN).
    bool zeroBias = false;
    int64_t sink = NO_SINK;
    int64_t sinkStride = 0;
    /// The sink's dtype relative to the query's.
    SinkDtype sinkDtype = SinkDtype::QUERY;
    ScaleSource scaleSource = ScaleSource::BAKED;
    int64_t scale = 0;
    int64_t seqLenQ = NO_SEQ_LEN;
    int64_t seqLenKv = NO_SEQ_LEN;
    int64_t raggedOffsetQ = NO_SEQ_LEN;
    int64_t raggedOffsetKv = NO_SEQ_LEN;
    float dropoutProbability = 0.0F;
    int64_t dropoutSeed = NO_DROPOUT_TENSOR;
    int64_t dropoutOffset = NO_DROPOUT_TENSOR;
    int64_t rngDump = NO_RNG_DUMP;

    /// Whether the graph pads its sequences (per-batch lengths under padding_mask).
    bool padded() const
    {
        return seqLenQ != NO_SEQ_LEN;
    }

    /// Whether the graph packs its sequences (THD, through ragged-offset tables).
    bool packed() const
    {
        return raggedOffsetQ != NO_SEQ_LEN;
    }

    /// Whether the graph drops attention probabilities (a probability above 0).
    bool dropsOut() const
    {
        return dropoutProbability > 0.0F;
    }
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
    binding.zeroBias = read(ZERO_BIAS_TOKEN) != 0;
    binding.sink = read(SINK_TOKEN);
    binding.sinkStride = read(SINK_STRIDE_TOKEN);
    binding.sinkDtype = static_cast<SinkDtype>(read(SINK_DTYPE_TOKEN));
    binding.scaleSource = static_cast<ScaleSource>(read(SCALE_SOURCE_TOKEN));
    binding.scale = read(SCALE_TOKEN);
    binding.seqLenQ = read(SEQ_LEN_Q_TOKEN);
    binding.seqLenKv = read(SEQ_LEN_KV_TOKEN);
    binding.raggedOffsetQ = read(RAGGED_OFFSET_Q_TOKEN);
    binding.raggedOffsetKv = read(RAGGED_OFFSET_KV_TOKEN);
    binding.dropoutProbability = tokenFloat(read(DROPOUT_P_TOKEN));
    binding.dropoutSeed = read(DROPOUT_SEED_TOKEN);
    binding.dropoutOffset = read(DROPOUT_OFFSET_TOKEN);
    binding.rngDump = read(RNG_DUMP_TOKEN);
    return binding;
}

/// The prefill kernel's runtime head_dim argument for @p kernel: Q/K's width, and on a pair
/// tier (v_mask, prefill kernel modification 32) V's width above it.
inline int32_t rdnaHeadDimArgument(const SdpaProblem& problem, const KernelDefinition& kernel)
{
    const auto qk = static_cast<uint32_t>(problem.headDim);
    if(integerMetadata(kernel, V_MASK_FIELD).value_or(0) != 1)
    {
        return static_cast<int32_t>(qk);
    }
    return static_cast<int32_t>(qk
                                | (static_cast<uint32_t>(problem.headDimV) << V_MASK_WIDTH_SHIFT));
}

/**
 * @brief The kernel-scoped checks both families make: dtype; head_dim (exact, or a generic
 *        object's runtime head_dim up to its head_dim_max); V's head_dim; the bias and its
 *        dtype; the sink; dropout; and the causal variant. Per-batch lengths (padding)
 *        are runtime arguments every object reads (kernel modification 26). Fills @p problem from the graph for the
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

    const auto dataTypeName = rdnaKernelDataTypeName(problem.dataType);
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
    // one's too, both being its runtime head_dim) -- or, on a generic pair tier, its own
    // runtime width up to the tier, and different from Q/K's (prefill kernel modification 32).
    const auto bakedHeadDimV = integerMetadata(kernel, HEAD_DIM_V_FIELD);
    const auto vMask = integerMetadata(kernel, V_MASK_FIELD).value_or(0);
    if(!bakedHeadDimV.has_value() || (vMask != 0 && vMask != 1))
    {
        return false;
    }
    if(vMask == 1)
    {
        if(*bakedHeadDim != GENERIC_HEAD_DIM || *bakedHeadDimV != GENERIC_HEAD_DIM
           || problem.headDimV == problem.headDim || problem.headDimV > *headDimMax
           || problem.headDimV % GENERIC_HEAD_DIM_ALIGN != 0)
        {
            return false;
        }
    }
    else if(*bakedHeadDimV == GENERIC_HEAD_DIM ? problem.headDimV != problem.headDim
                                               : *bakedHeadDimV != problem.headDimV)
    {
        return false;
    }

    // A bias object for a graph with a bias, a plain one for a graph without -- unless its
    // scale needs scaled scores, which the plain objects do not keep (runsOnAZeroBias): then a
    // bias object in the query's dtype, reading a zero bias, as for a [1, 1, 1, Skv] one.
    const auto bias = tryGetBoundInt(bound, BIAS_TOKEN);
    const auto zeroBias = tryGetBoundInt(bound, ZERO_BIAS_TOKEN);
    if(!bias.has_value() || !zeroBias.has_value()
       || integerMetadata(kernel, HAS_BIAS_FIELD) != (*bias == NO_BIAS && *zeroBias == 0 ? 0 : 1))
    {
        return false;
    }
    // The zero bias is in the query's dtype: a `bias_q` object, or for an f32 graph an f32-bias
    // one (an f32 object's bias is f32). Read at stride 0 on every axis (ZERO_BIAS_STRIDES), it
    // has no layout for packed sequences to break, as a graph's own bias does.
    const int64_t zeroBiasQ = problem.dataType == data_objects::DataType::FLOAT ? 0 : 1;
    if(*bias == NO_BIAS && *zeroBias != 0 && integerMetadata(kernel, BIAS_Q_FIELD) != zeroBiasQ)
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

    // A dropout object for a graph that drops, a plain one for one that does not (a
    // probability of 0 is the undropped result). Absent is a plain object: no decode object
    // drops.
    const auto dropoutProbability = tryGetBoundInt(bound, DROPOUT_P_TOKEN);
    if(!dropoutProbability.has_value()
       || integerMetadata(kernel, HAS_DROPOUT_FIELD).value_or(0)
              != (tokenFloat(*dropoutProbability) > 0.0F ? 1 : 0))
    {
        return false;
    }
    // An RNG dump is written by the dropout mask writer, which must ship in this object's shard.
    const auto rngDump = tryGetBoundInt(bound, RNG_DUMP_TOKEN);
    if(!rngDump.has_value() || (*rngDump != NO_RNG_DUMP && !hasDropoutMaskWriter(kernel)))
    {
        return false;
    }

    // Packed (THD) sequences: every object of the aiw kernel and of the RDNA prefill kernel
    // reads them (RDNA kernel modification 29); the decode objects decline them in their own
    // kernel match.
    if(!tryGetBoundInt(bound, RAGGED_OFFSET_Q_TOKEN).has_value())
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

/// Whether @p kernel, launched for @p binding, reads the zero bias (runsOnAZeroBias): a bias
/// object of the RDNA prefill or decode engine for a graph without a bias of its own.
inline bool readsZeroBias(const FlydslSdpaRdnaBinding& binding, const KernelDefinition& kernel)
{
    return binding.bias == NO_BIAS && integerMetadata(kernel, HAS_BIAS_FIELD) == 1;
}

/// The softmax scale for a launch: the pass-by-value scalar when the graph gives one, read
/// now, else the f32 the match baked into the scale token. @p rawScores says whether the
/// launched object keeps its running max over unscaled scores (an RDNA prefill or decode
/// object without a bias), which takes only a scale rawScoresServe accepts; the others take
/// any isServableScale accepts.
inline float launchScale(const FlydslSdpaRdnaBinding& binding,
                         const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scaleOperand,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         bool rawScores)
{
    float scale = 0.0F;
    if(scaleOperand.has_value())
    {
        scale = static_cast<float>(
            hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                *scaleOperand, deviceBuffers, numDeviceBuffers)));
        // A pass-by-value scalar, read from host memory. A constant scale was checked at
        // match time; this one arrives now, after the object was chosen.
        if(!isServableScale(static_cast<double>(scale)))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna: the runtime softmax scale must be finite and at most 2^20 in "
                "magnitude");
        }
        if(rawScores && !rawScoresServe(static_cast<double>(scale)))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna: an object without a bias takes a runtime softmax scale only "
                "when it is positive and normal; give a zero, negative or subnormal scale as a "
                "value, which runs on the bias objects");
        }
    }
    else
    {
        const auto scaleBits = static_cast<int32_t>(binding.scale);
        std::memcpy(&scale, &scaleBits, sizeof(scale));
    }
    return scale;
}

/// Padded query rows -- past a batch's `seq_len_q`, inside the graph's sequence extent -- are
/// written as O = 0 and LSE = -inf, hipDNN's reference contract (cuDNN >= 9.14) and what the
/// CDNA engines write. The kernels can also leave them as the buffer held them (AOTriton's
/// behaviour, and free); that policy is this one switch.
inline constexpr bool FILL_PADDED_ROWS = true;

/// The kernels' flags argument (RDNA kernel modifications 26, 28 and 29): bit 0 writes the
/// LSE output, bit 1 reads per-batch lengths from the SEQLEN_Q / SEQLEN_KV pointers, bit 2
/// fills the padded rows (FILL_PADDED_ROWS), bit 3 reads packed sequences' bounds from the
/// ROW_OFF_Q / ROW_OFF_KV tables (the prefill kernel only; the decode objects decline packed
/// graphs).
inline int32_t rdnaLaunchFlags(const FlydslSdpaRdnaBinding& binding)
{
    return (binding.stats == NO_STATS ? 0 : 1) | (binding.padded() ? 2 : 0)
           | (FILL_PADDED_ROWS && binding.padded() ? 4 : 0) | (binding.packed() ? 8 : 0);
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

/// The dropout arguments every RDNA-family kernel takes (the aiw kernel's philox_seed_ptr ..
/// dropout_scale; the RDNA prefill kernel's DROPOUT_SEED .. dropout_scale). Read only by an
/// object built with dropout; a plain object gets no seed or offset, a threshold of 0 and a
/// scale of 1. The aiw kernel's seed and offset outputs, which report the counter the launch
/// drew from, have no hipDNN slot and stay null.
struct PhiloxLaunchArgs
{
    void* seed = nullptr;
    void* offset1 = nullptr;
    int64_t offset2 = 0;
    void* seedOutput = nullptr;
    void* offsetOutput = nullptr;
    int32_t threshold = 0;
    float scale = 1.0F;
};

/// The bytes a launch needs for a seed given by value: the kernel reads the seed through a
/// pointer, so a host seed is written to the engine's workspace first.
inline constexpr size_t SEED_WORKSPACE_BYTES = sizeof(uint64_t);

/// A Philox seed or offset given by value, widened to 64 bits: INT64 as is, INT32 sign-extended
/// (as C++ widens it), read as the bits the kernel takes.
inline uint64_t philoxScalar(const hipdnn_plugin_sdk::ScalarOperand& operand,
                             const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                             uint32_t numDeviceBuffers)
{
    const auto value
        = hipdnn_plugin_sdk::resolveScalarOperand(operand, deviceBuffers, numDeviceBuffers);
    if(const auto* wide = std::get_if<int64_t>(&value))
    {
        return static_cast<uint64_t>(*wide);
    }
    if(const auto* narrow = std::get_if<int32_t>(&value))
    {
        return static_cast<uint64_t>(static_cast<int64_t>(*narrow));
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
        "flydsl_sdpa_rdna: a dropout seed or offset given by value must be INT64 or INT32");
}

/// How a launch reads the Philox seed and offset: through the graph's device tensor, or from a
/// host value resolved at execute (the seed then via the workspace, the offset as offset2).
struct PhiloxSources
{
    std::optional<hipdnn_plugin_sdk::ScalarOperand> seedValue;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> offsetValue;
};

inline PhiloxSources philoxSources(const MatchContext& context,
                                   const FlydslSdpaRdnaBinding& binding)
{
    PhiloxSources sources;
    if(!binding.dropsOut())
    {
        return sources;
    }
    const auto byValue
        = [&context](int64_t uid) -> std::optional<hipdnn_plugin_sdk::ScalarOperand> {
        if(!hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&requireTensor(context, uid)))
        {
            return std::nullopt;
        }
        return hipdnn_plugin_sdk::makeScalarOperand(
            context.graph.getTensorMap(), uid, "dropout seed or offset");
    };
    sources.seedValue = byValue(binding.dropoutSeed);
    sources.offsetValue = byValue(binding.dropoutOffset);
    return sources;
}

inline PhiloxLaunchArgs philoxLaunchArgs(const FlydslSdpaRdnaBinding& binding,
                                         const PhiloxSources& sources,
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                         uint32_t numDeviceBuffers,
                                         void* workspace,
                                         hipStream_t stream)
{
    PhiloxLaunchArgs args;
    if(!binding.dropsOut())
    {
        return args;
    }
    if(sources.seedValue.has_value())
    {
        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna: a dropout seed given by value needs the engine's workspace");
        }
        // Two 32-bit memsets on the launch stream, not a host copy: the value travels in the
        // commands themselves, so the launch stays capturable into a graph.
        const auto seed = philoxScalar(*sources.seedValue, deviceBuffers, numDeviceBuffers);
        auto* words = static_cast<uint32_t*>(workspace);
        if(hipMemsetD32Async(words, static_cast<int>(static_cast<uint32_t>(seed)), 1, stream)
               != hipSuccess
           || hipMemsetD32Async(
                  words + 1, static_cast<int>(static_cast<uint32_t>(seed >> 32)), 1, stream)
                  != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_sdpa_rdna: could not write the dropout seed to the workspace");
        }
        args.seed = workspace;
    }
    else
    {
        args.seed = optionalBuffer(binding.dropoutSeed, deviceBuffers, numDeviceBuffers);
    }
    if(sources.offsetValue.has_value())
    {
        args.offset2 = static_cast<int64_t>(
            philoxScalar(*sources.offsetValue, deviceBuffers, numDeviceBuffers));
    }
    else
    {
        args.offset1 = optionalBuffer(binding.dropoutOffset, deviceBuffers, numDeviceBuffers);
    }
    args.threshold = dropoutThreshold(binding.dropoutProbability);
    args.scale = dropoutScale(binding.dropoutProbability);
    return args;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_rdna

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
