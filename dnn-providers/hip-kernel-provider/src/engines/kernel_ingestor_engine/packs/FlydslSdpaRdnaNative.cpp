// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
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

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file FlydslSdpaRdnaNative.cpp
 * @brief The flyDSL SDPA-forward pack's native half: matching, scoring, dispatch, and the
 *        one function that registers them.
 *
 * Every kernel this pack selects is a pre-built HSACO, checked in as an `hsaco` UKD of the
 * production descriptor root and packed into the per-arch `hip_kernel_provider_<arch>.kpack`.
 * It is compiled ahead of time from
 * `flydsl/kernels_src/kernels/attention_rdna/flash_attn_func_rdna.py`. That kernel bakes
 * only dtype, head_dim, the causal flag and its tile; batch, both sequence lengths, both
 * head counts, every stride, the softmax scale and the causal offset are kernel
 * arguments. So one object serves every shape and layout of its (dtype, head_dim,
 * causal) class, and the matcher's job is to decline exactly what the kernel cannot
 * compute -- see `flydsl/COVERAGE.md` §5 for the list, each with its reason.
 *
 * Symbol names are restated here rather than shared via a header, since a descriptor can't
 * reference a C++ constant; the loader pre-flights every symbol a descriptor names, so a
 * mismatched string is caught without becoming a compile error.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_rdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_rdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_rdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_rdna.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
constexpr std::string_view BLOCK_N_FIELD = "block_n";
/// Workgroups that share one query tile's output columns, each computing head_dim /
/// dv_split of them; the grid grows by it. 1 up to head_dim 128, 2 above.
constexpr std::string_view DV_SPLIT_FIELD = "dv_split";
/// 0 or 1: whether the object adds an additive bias to the scores. Baked, so a graph with
/// a bias and one without each select their own object.
constexpr std::string_view HAS_BIAS_FIELD = "has_bias";
/// 0 for an object that reads an f32 bias, 1 for one that reads it in the query's dtype.
constexpr std::string_view BIAS_Q_FIELD = "bias_q";
/// The largest head_dim the object serves. Equal to head_dim for a specialized object; for
/// a generic one (head_dim 0, read at runtime) it bounds the heads it takes.
constexpr std::string_view HEAD_DIM_MAX_FIELD = "head_dim_max";

/// 1 for the decode family's objects (flash_attn_decode_rdna.py), 0 for the prefill ones.
constexpr std::string_view DECODE_FIELD = "decode";
/// The decode object's second kernel, which combines its KV splits. Same code object, so
/// it is loaded from the same archive entry by this symbol.
constexpr std::string_view MERGE_SYMBOL_FIELD = "merge_symbol";
/// 0 or 1: whether the prefill object's waves work in head-dim pairs, two to each 16 query
/// rows, so its workgroup has twice the waves its block_m alone implies.
constexpr std::string_view HD_PAIR_FIELD = "hd_pair";

/// The decode kernel packs the GQA group's query heads and the query positions into 16-row
/// tiles, one workgroup per tile along the grid's second dimension. Each tile re-reads the
/// KV head, so past a few tiles the prefill objects, which share each K/V tile across 128
/// rows, are faster: decode serves (Hq / Hk) * Sq up to DECODE_MAX_ROWS.
constexpr int64_t DECODE_ROWS = 16;
constexpr int64_t DECODE_MAX_ROWS = 128;
/// Waves per decode workgroup, keys per wave per step, and threads per merge workgroup:
/// the geometry the decode kernel and its merge were built with.
constexpr int64_t DECODE_WAVES = 4;
constexpr int64_t DECODE_TILE = 32;
constexpr int64_t DECODE_MERGE_THREADS = 64;
/// Split-count bounds: a split should give each wave at least this many tiles, the splits
/// together should roughly fill the device, and never more than this many.
constexpr int64_t DECODE_MIN_TILES_PER_WAVE = 8;
constexpr int64_t DECODE_MAX_SPLITS = 64;
/// Compute units assumed when the device does not report its count.
constexpr int64_t DECODE_FALLBACK_COMPUTE_UNITS = 32;
/// Decode workgroups to aim for per reported unit. RDNA reports workgroup processors --
/// two CUs each -- in multiProcessorCount, so one workgroup per unit leaves half the
/// device idle at small batch; two fill it, and more splits than that cost more in the
/// merge and the shorter KV runs than they recover (measured on gfx1151).
constexpr int64_t DECODE_WORKGROUPS_PER_UNIT = 2;

/// A generic object's head_dim: it reads the real one at runtime.
constexpr int64_t GENERIC_HEAD_DIM = 0;

/// A generic object zeroes and skips whole 8-column groups, so the head must be made of them.
constexpr int64_t GENERIC_HEAD_DIM_ALIGN = 8;

constexpr std::string_view Q_TOKEN = "flydsl_sdpa_rdna.q.uid";
constexpr std::string_view K_TOKEN = "flydsl_sdpa_rdna.k.uid";
constexpr std::string_view V_TOKEN = "flydsl_sdpa_rdna.v.uid";
constexpr std::string_view O_TOKEN = "flydsl_sdpa_rdna.o.uid";
/// 0 or 1: which of the two kernel variants the graph needs.
constexpr std::string_view CAUSAL_TOKEN = "flydsl_sdpa_rdna.causal";
/// The mask, in the reference's own terms (CpuFpReferenceSdpa isMasked): right and left
/// bounds (-1 for none) and whether the diagonal is bottom-right aligned. The kernel derives
/// the diagonal's offset from the sequence lengths itself.
constexpr std::string_view RIGHT_BOUND_TOKEN = "flydsl_sdpa_rdna.right_bound";
constexpr std::string_view LEFT_BOUND_TOKEN = "flydsl_sdpa_rdna.left_bound";
constexpr std::string_view ALIGN_BOTTOM_RIGHT_TOKEN = "flydsl_sdpa_rdna.align_bottom_right";
/// The uid of the softmax-statistics (LSE) output, or NO_STATS when the graph wants none.
constexpr std::string_view STATS_TOKEN = "flydsl_sdpa_rdna.stats.uid";
/// The uid of the additive bias (`attn_mask`), or NO_BIAS when the graph has none.
constexpr std::string_view BIAS_TOKEN = "flydsl_sdpa_rdna.bias.uid";
/// Where the softmax scale comes from; see ScaleSource.
constexpr std::string_view SCALE_SOURCE_TOKEN = "flydsl_sdpa_rdna.scale_source";
/// The scale's f32 bit pattern when baked (BoundTokens carry int64_t), else the uid of
/// the pass-by-value scale tensor.
constexpr std::string_view SCALE_TOKEN = "flydsl_sdpa_rdna.scale";

/// hipDNN SDPA tensors are rank 4 in LOGICAL order (B, H, S, D) whatever the layout.
constexpr uint32_t SDPA_RANK = 4;
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_DIM_AXIS = 3;

/// Unbounded, in the left_bound/right_bound convention.
constexpr int64_t UNBOUNDED = -1;

/// The stats token's value when the graph asks for no LSE output.
constexpr int64_t NO_STATS = -1;

/// The bias token's value when the graph has no additive bias.
constexpr int64_t NO_BIAS = -1;

/// The bias's (batch, head, query, key) element strides as the kernel takes them.
using BiasStrides = std::array<int64_t, 4>;

/// The bias is read through a 32-bit buffer descriptor per (batch, head) slice, with 32-bit
/// element offsets inside it.
constexpr int64_t BIAS_SLICE_ELEMENT_LIMIT = (1LL << 30) - 1;

/// Keys past the last one the kernel may read in a partial final tile.
constexpr int64_t KV_TILE_OVERREAD = 31;

/// Sequence lengths the kernel's int32 mask arithmetic is sound for: a row's attendable
/// range is computed as q + offset + bound with every term below 2^29, so the sum stays
/// inside int32, as does the "no window" sentinel's distance from any kv index.
constexpr int64_t SEQ_LEN_LIMIT = 536870912LL; // 2^29

/// Kernel geometry the generator records: each wave owns 16 query rows, 32 lanes wide.
constexpr int64_t WAVE_ROWS = 16;
constexpr int64_t WAVE_SIZE = 32;
/// The kernel's KV sub-tile; block_n must be a multiple of it.
constexpr int64_t KV_SUB_TILE = 32;

constexpr int64_t INT32_LIMIT = 2147483647LL;
/// K and V are read through a buffer descriptor scoped to one (batch, kv head) slice,
/// whose num_records is 32 bits. Held to 2^31 rather than 2^32 so no part of the path
/// has to be trusted to treat it as unsigned.
constexpr int64_t KV_SLICE_BYTE_LIMIT = 2147483648LL; // 2^31
constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and fp16 only

/// Where the kernel's `scale` argument comes from.
enum class ScaleSource : int64_t
{
    /// attn_scale_value, or 1/sqrt(head_dim) when the graph gives none -- the default
    /// hipDNN's CPU reference applies (CpuFpReferenceSdpa.hpp).
    BAKED = 0,
    /// A pass-by-value scalar tensor, resolved at execute.
    TENSOR = 1,
};

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node this pack's matchers read, or nullptr if the graph isn't a single SDPA-forward
/// node.
const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaAttributes)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::SdpaAttributes>();
}

/// The product of @p factors, or nullopt when it does not fit in int64_t. Every product of
/// graph-controlled extents goes through this: an unchecked one is signed overflow, and a
/// wrapped value can pass a bound it should fail.
std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
{
    int64_t product = 1;
    for(const int64_t factor : factors)
    {
        if(__builtin_mul_overflow(product, factor, &product))
        {
            return std::nullopt;
        }
    }
    return product;
}

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
bool isServableOperand(const data_objects::TensorAttributes& tensor)
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

/// The mask a graph asks for, in the reference's terms (CpuFpReferenceSdpa isMasked): kv is
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
 * diagonal alignment. The deprecated causal booleans mean right 0 at their corner; a
 * right_bound the graph also sets wins over the boolean's implied 0. Setting both booleans
 * is contradictory and declined.
 */
std::optional<SdpaMask> maskFor(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();
    if(topLeftDeprecated && bottomRightDeprecated)
    {
        return std::nullopt;
    }

    SdpaMask mask;
    mask.left = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    mask.right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;
    if(mask.left < UNBOUNDED || mask.right < UNBOUNDED)
    {
        return std::nullopt;
    }

    if(topLeftDeprecated || bottomRightDeprecated)
    {
        if(mask.right == UNBOUNDED)
        {
            mask.right = 0;
        }
        mask.alignBottomRight = bottomRightDeprecated;
    }
    else
    {
        mask.alignBottomRight
            = attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }
    return mask;
}

/// The softmax-statistics output the kernel can write: f32 `[B, H, Sq, 1]`, a real device
/// tensor, with positive strides on its non-unit axes (the kernel takes its batch, sequence
/// and head strides).
bool isServableStats(const data_objects::TensorAttributes& stats,
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
std::optional<BiasStrides> servableBiasStrides(const data_objects::TensorAttributes& bias,
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

/// The kernel metadata spelling of a graph dtype. Descriptors say `bf16`/`f16`, the
/// flatbuffer enum says `BFLOAT16`/`HALF`, and this is the one place the two meet.
std::optional<std::string> kernelDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::HALF:
        return std::string("f16");
    default:
        return std::nullopt;
    }
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
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

SdpaProblem problemFor(const data_objects::TensorAttributes& q,
                       const data_objects::TensorAttributes& k)
{
    SdpaProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headDim = q.dims()->Get(HEAD_DIM_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.numVHeads = problem.numKvHeads;
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

bool hasDims(const data_objects::TensorAttributes& tensor,
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
bool kvSliceFits(const data_objects::TensorAttributes& tensor, int64_t seqLenKv, int64_t headDim)
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
std::optional<BoundTokens> flydslSdpaRdnaGraphMatches(const MatchContext& context)
{
    // --- 1. One SDPA-forward node; each kernel serves one complete graph.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

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

    // --- 4. Cross-tensor shape. No broadcasting: K and V share every extent, O is Q's
    // shape, and V's head dim must equal Q/K's -- the kernel has one head_dim. V's head count
    // may differ from K's.
    const auto numVHeads = v->dims()->Get(HEAD_AXIS);
    if(!hasDims(*k, problem.batch, problem.numKvHeads, problem.seqLenKv, problem.headDim)
       || !hasDims(*v, problem.batch, numVHeads, problem.seqLenKv, problem.headDim)
       || !hasDims(*o, problem.batch, problem.numHeads, problem.seqLenQ, problem.headDim))
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
       || !kvSliceFits(*v, problem.seqLenKv, problem.headDim))
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
    // A bound past the sequence masks nothing more than one at it; clamped so the kernel's
    // int32 sums stay in range.
    const int64_t causal = mask->right >= 0 ? 1 : 0;
    const int64_t rightBound = std::min(mask->right, problem.seqLenKv);
    const int64_t leftBound = std::min(mask->left, problem.seqLenQ + problem.seqLenKv);

    // --- 7. Every optional feature the kernel does not implement, declined explicitly.

    // Additive attention bias: served by the has_bias objects.
    int64_t biasUid = NO_BIAS;
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        const auto* bias = findTensor(context, attributes.attn_mask_tensor_uid().value());
        if(bias == nullptr
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
    // Block-sparse masks and attention sinks.
    if(attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value())
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
        const auto* stats = findTensor(context, attributes.stats_tensor_uid().value());
        if(stats == nullptr
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
        // device read the kernel does not do.
        const auto* scaleTensor = findTensor(context, attributes.scale_tensor_uid().value());
        if(scaleTensor == nullptr
           || !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scaleTensor))
        {
            return std::nullopt;
        }
        try
        {
            static_cast<void>(hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), attributes.scale_tensor_uid().value(), "scale"));
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return std::nullopt;
        }
        scaleSource = ScaleSource::TENSOR;
        scaleToken = attributes.scale_tensor_uid().value();
    }
    else
    {
        const float scale = attributes.attn_scale_value().has_value()
                                ? attributes.attn_scale_value().value()
                                : 1.0F / std::sqrt(static_cast<float>(problem.headDim));
        // The kernel takes the running max over unscaled scores, which orders them the
        // same way as the scaled ones only for a positive scale.
        if(!(scale > 0.0F) || !std::isfinite(scale))
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
    ScaleSource scaleSource = ScaleSource::BAKED;
    int64_t scale = 0;
};

FlydslSdpaRdnaBinding flydslSdpaRdnaBinding(const BoundTokens& bound)
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
    binding.scaleSource = static_cast<ScaleSource>(read(SCALE_SOURCE_TOKEN));
    binding.scale = read(SCALE_TOKEN);
    return binding;
}

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total, unlike KernelDefinition::getIntMetadata: a malformed record declines
/// rather than throwing out of a match.
std::optional<int64_t> integerMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* integer = std::get_if<int64_t>(&*value);
    return integer == nullptr ? std::nullopt : std::optional<int64_t>(*integer);
}

/// The string counterpart of integerMetadata, equally total.
std::optional<std::string> stringMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* text = std::get_if<std::string>(&*value);
    return text == nullptr ? std::nullopt : std::optional<std::string>(*text);
}

/// A candidate's launch tile, or nullopt when its metadata describes none the kernel can
/// have been built with.
struct SdpaTile
{
    int64_t blockM = 0;
    int64_t blockN = 0;
    int64_t dvSplit = 1;
    /// Waves per 16 query rows: 2 for an hd_pair object, else 1.
    int64_t wavesPerRowTile = 1;
};

std::optional<SdpaTile> candidateTile(const KernelDefinition& kernel)
{
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto blockN = integerMetadata(kernel, BLOCK_N_FIELD);
    const auto dvSplit = integerMetadata(kernel, DV_SPLIT_FIELD);
    const auto hdPair = integerMetadata(kernel, HD_PAIR_FIELD);
    if(!blockM.has_value() || !blockN.has_value() || !dvSplit.has_value() || *blockM <= 0
       || *blockM % WAVE_ROWS != 0 || *blockN <= 0 || *blockN % KV_SUB_TILE != 0 || *dvSplit <= 0
       || !hdPair.has_value() || (*hdPair != 0 && *hdPair != 1))
    {
        return std::nullopt;
    }
    return SdpaTile{*blockM, *blockN, *dvSplit, *hdPair == 1 ? 2 : 1};
}

/// The workgroup count `batch_x_qtiles_x_heads_x_dv_split`, or nullopt past int32.
std::optional<int64_t> candidateGrid(const SdpaProblem& problem, const SdpaTile& tile)
{
    const auto queryTiles = (problem.seqLenQ + tile.blockM - 1) / tile.blockM;
    const auto grid = checkedProduct({problem.batch, queryTiles, problem.numHeads, tile.dvSplit});
    if(!grid.has_value() || *grid > INT32_LIMIT)
    {
        return std::nullopt;
    }
    return grid;
}

/// Whether a decode object takes several row tiles. The bias objects are built for one:
/// they already use the whole register file at head_dim 128 and up, and the row index a
/// tiled launch holds through the KV loop spills there.
bool decodeRowTiled(const KernelDefinition& kernel)
{
    return integerMetadata(kernel, HAS_BIAS_FIELD) == 0;
}

/// The decode launch's row tiles: (Hq / Hk) * Sq rows, 16 to a tile, or one.
int64_t decodeRowTiles(const SdpaProblem& problem, bool rowTiled)
{
    return rowTiled ? ((problem.numHeads / problem.numKvHeads) * problem.seqLenQ + DECODE_ROWS - 1)
                          / DECODE_ROWS
                    : 1;
}

/// How many KV splits a decode launch uses: enough workgroups to roughly fill the device
/// (DECODE_WORKGROUPS_PER_UNIT per reported unit), but no split so short that its waves have little to stream, and at most
/// DECODE_MAX_SPLITS. Deterministic in the problem and the device, so workspaceBytes and
/// prepare agree.
int64_t decodeSplits(const SdpaProblem& problem, int64_t rowTiles, int computeUnits)
{
    const int64_t units = (computeUnits > 0 ? computeUnits : DECODE_FALLBACK_COMPUTE_UNITS)
                          * DECODE_WORKGROUPS_PER_UNIT;
    const int64_t groups = std::max<int64_t>(1, problem.batch * problem.numKvHeads * rowTiles);
    const int64_t tiles = (problem.seqLenKv + DECODE_TILE - 1) / DECODE_TILE;
    const int64_t byWork = std::max<int64_t>(1, tiles / (DECODE_WAVES * DECODE_MIN_TILES_PER_WAVE));
    const int64_t byDevice = std::max<int64_t>(1, units / groups);
    return std::min({byWork, byDevice, DECODE_MAX_SPLITS});
}

/// The split workspace: a normalized f32 partial O and its LSE per (batch, query head,
/// query position, split), or nothing when one split writes O directly.
std::optional<size_t> decodeWorkspaceBytes(const SdpaProblem& problem, int64_t splits)
{
    if(splits <= 1)
    {
        return size_t{0};
    }
    const auto elements = checkedProduct(
        {problem.batch, problem.numHeads, problem.seqLenQ, splits, problem.headDim + 1});
    if(!elements.has_value())
    {
        return std::nullopt;
    }
    return static_cast<size_t>(*elements) * sizeof(float);
}

/// Whether a decode object's 32-bit bias offsets reach every row of its tile: they span the
/// GQA group's heads as well as the query positions and keys of one (batch, head) slice.
bool decodeBiasFits(const SdpaProblem& problem, const BiasStrides& strides)
{
    const auto lastHead = checkedProduct({problem.numHeads / problem.numKvHeads - 1, strides[1]});
    const auto lastQuery = checkedProduct({problem.seqLenQ - 1, strides[2]});
    const auto lastKey = checkedProduct({problem.seqLenKv - 1 + KV_TILE_OVERREAD, strides[3]});
    return lastHead.has_value() && lastQuery.has_value() && lastKey.has_value()
           && *lastHead <= BIAS_SLICE_ELEMENT_LIMIT - *lastKey
           && *lastQuery <= BIAS_SLICE_ELEMENT_LIMIT - *lastKey - *lastHead;
}

/// Whether a decode object should serve the graph: the GQA group times the query
/// positions is at most DECODE_MAX_ROWS (one tile's DECODE_ROWS for an object built for
/// one), and one V head goes with each K head (the rows of a tile share one KV head).
bool decodeServes(const SdpaProblem& problem, bool rowTiled)
{
    return problem.numKvHeads == problem.numVHeads
           && (problem.numHeads / problem.numKvHeads) * problem.seqLenQ
                  <= (rowTiled ? DECODE_MAX_ROWS : DECODE_ROWS);
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit the graph?
 *
 * dtype, head_dim (or, for a generic object, its largest), the causal variant and the bias are
 * baked; everything else is a runtime
 * argument, so they are the whole comparison.
 */
bool flydslSdpaRdnaKernelMatches(const MatchContext& context,
                                 const BoundTokens& bound,
                                 const KernelDefinition& kernel)
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
    auto problem = problemFor(*q, *k);
    problem.numVHeads = v->dims()->Get(HEAD_AXIS);

    const auto tile = candidateTile(kernel);
    if(!tile.has_value())
    {
        return false;
    }

    // A decode object serves only the graphs its packed tile fits, and carries the merge
    // kernel it launches by name. Its head_dim is checked below with the prefill ones'.
    const auto decode = integerMetadata(kernel, DECODE_FIELD);
    if(!decode.has_value() || (*decode != 0 && *decode != 1))
    {
        return false;
    }
    if(*decode == 1
       && (!decodeServes(problem, decodeRowTiled(kernel))
           || stringMetadata(kernel, MERGE_SYMBOL_FIELD).value_or("").empty()))
    {
        return false;
    }

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

    // The column split must divide the object's head into whole tiles, and the split grid --
    // larger than the graph-scoped bound assumed -- must still fit the launch.
    if(*headDimMax % tile->dvSplit != 0 || !candidateGrid(problem, *tile).has_value())
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
    if(*decode == 1 && *bias != NO_BIAS)
    {
        const auto* biasTensor = findTensor(context, *bias);
        const auto strides = biasTensor == nullptr ? std::nullopt
                                                   : servableBiasStrides(*biasTensor,
                                                                         problem.batch,
                                                                         problem.numHeads,
                                                                         problem.seqLenQ,
                                                                         problem.seqLenKv,
                                                                         problem.dataType);
        if(!strides.has_value() || !decodeBiasFits(problem, *strides))
        {
            return false;
        }
    }

    const auto causal = tryGetBoundInt(bound, CAUSAL_TOKEN);
    return causal.has_value() && integerMetadata(kernel, CAUSAL_FIELD) == *causal;
}

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * The descriptor `priority`, unchanged. Today exactly one candidate survives a match
 * (dtype x head_dim x causal is the instance key), so this decides nothing yet; it is
 * here so a second tile for one class is ranked by data, as RMS norm's tiers are.
 */
double flydslSdpaRdnaScore(const MatchContext& /*context*/,
                           const BoundTokens& /*bound*/,
                           const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/**
 * @brief The argument list this pack launches with, mirroring launch() one for one, and
 *        the `signature` block every one of this pack's UKDs carries.
 *
 * Q, K, V, O, LSE pointers; seq_len_q, seq_len_kv, num_heads, kv_group, right_bound,
 * left_bound, align_bottom_right, lse_on (i32); scale (f32); the (batch, sequence, head)
 * element strides of Q, K, V, O and LSE (i64); then the BIAS pointer and its (batch, head,
 * query, key) element strides (i64), the runtime head_dim (i32, read only by a generic
 * object), and the query heads per V head (i32) -- each appended so the arguments before
 * them kept their offsets.
 *
 * Names are empty and offsets zero because neither is compared for this producer -- FlyDSL
 * emits no argument names; see requireSignatureMatch. The semantic names are recorded per
 * object in the `manifest.json` beside it, under `descriptors/FlyDSL/sdpa_rdna/<arch>/`.
 */
const std::vector<KernelArgument>& flydslSdpaRdnaKernelSignature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_f32{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const KernelArgument s_i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_i32, s_i32, s_i32, s_i32,
        s_i32,     s_i32,     s_i32,     s_i32,     s_f32,     s_i64, s_i64, s_i64, s_i64,
        s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64, s_i64, s_i64, s_i64,
        s_i64,     s_i64,     s_pointer, s_i64,     s_i64,     s_i64, s_i64, s_i32, s_i32};
    return s_signature;
}

/**
 * @brief The decode object's main kernel's argument list (flash_attn_decode_rdna.py).
 *
 * Q, K, V, O, LSE and the split workspace's O and LSE (pointers); seq_len_q, seq_len_kv,
 * num_heads, kv_group, right_bound, left_bound, align_bottom_right, lse_on, num_splits
 * (i32); scale (f32); the (batch, sequence, head) strides of Q, K, V, O and LSE (i64);
 * the bias (pointer) and its (batch, head, query, key) strides (i64), read only by a
 * has_bias object; the head_dim (i32), read only by a generic object.
 */
const std::vector<KernelArgument>& flydslSdpaRdnaDecodeSignature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_f32{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const KernelArgument s_i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_i32,
        s_i32,     s_i32,     s_i32,     s_i32,     s_i32,     s_i32,     s_i32,     s_i32,
        s_f32,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,
        s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64,
        s_pointer, s_i64,     s_i64,     s_i64,     s_i64,     s_i32};
    return s_signature;
}

/**
 * @brief The decode object's merge kernel's argument list.
 *
 * O, LSE and the split workspace's O and LSE (pointers); seq_len_q, num_heads, lse_on,
 * num_splits (i32); the (batch, sequence, head) strides of O and LSE (i64); the
 * head_dim (i32), the width of a generic object's workspace and O rows.
 */
const std::vector<KernelArgument>& flydslSdpaRdnaMergeSignature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{s_pointer,
                                                         s_pointer,
                                                         s_pointer,
                                                         s_pointer,
                                                         s_i32,
                                                         s_i32,
                                                         s_i32,
                                                         s_i32,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i64,
                                                         s_i32};
    return s_signature;
}

/// One operand's (batch, sequence, head) element strides, in the kernel's argument order.
struct OperandStrides
{
    int64_t batch = 0;
    int64_t sequence = 0;
    int64_t head = 0;
};

OperandStrides stridesOf(const data_objects::TensorAttributes& tensor)
{
    const auto* strides = tensor.strides();
    return OperandStrides{
        strides->Get(BATCH_AXIS), strides->Get(SEQ_AXIS), strides->Get(HEAD_AXIS)};
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched flydsl_sdpa_rdna graph has no tensor for uid " + std::to_string(uid));
    }
    return *tensor;
}

/// The compiled kernel plus everything launch() needs, owning nothing that points back
/// into the MatchContext or BoundTokens it came from.
class PreparedFlydslSdpaRdna : public PreparedDispatch
{
public:
    PreparedFlydslSdpaRdna(IngestorKernelCode code,
                           FlydslSdpaRdnaBinding binding,
                           SdpaProblem problem,
                           std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand,
                           std::array<OperandStrides, 5> strides,
                           BiasStrides biasStrides,
                           std::optional<IngestorKernelCode> mergeCode = std::nullopt,
                           int64_t splits = 1)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _scaleOperand(scaleOperand)
        , _strides(strides)
        , _biasStrides(biasStrides)
        , _mergeCode(std::move(mergeCode))
        , _splits(splits)
    {
    }

    /// A decode object's merge kernel; empty for a prefill object.
    bool isDecode() const
    {
        return _mergeCode.has_value();
    }

    compilation::IRunnableKernel& mergeForStream(hipStream_t stream) const
    {
        return _mergeCode->kernelForStream(stream);
    }

    /// The decode launch's KV splits; 1 writes O directly and skips the merge.
    int64_t splits() const
    {
        return _splits;
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslSdpaRdnaBinding& binding() const
    {
        return _binding;
    }

    const SdpaProblem& problem() const
    {
        return _problem;
    }

    const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scaleOperand() const
    {
        return _scaleOperand;
    }

    /// Q, K, V, O and the LSE output (zeros when there is none), in that order.
    const std::array<OperandStrides, 5>& strides() const
    {
        return _strides;
    }

    /// The bias's (batch, head, query, key) strides; zeros when there is none.
    const BiasStrides& biasStrides() const
    {
        return _biasStrides;
    }

private:
    IngestorKernelCode _code;
    FlydslSdpaRdnaBinding _binding;
    SdpaProblem _problem;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _scaleOperand;
    std::array<OperandStrides, 5> _strides;
    BiasStrides _biasStrides;
    std::optional<IngestorKernelCode> _mergeCode;
    int64_t _splits = 1;
};

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and the scale, so
 *        nothing mutates once prepared and concurrent execution is safe.
 */
class FlydslSdpaRdnaDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime. Unused in
    /// practice -- every kernel this pack selects comes from a kpack archive -- but
    /// buildIngestorKernelCode decides that from the kernel's source kind, not from here.
    /// @param kpackLoader Same must-outlive contract.
    FlydslSdpaRdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                  const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// A prefill object's only scratch is LDS and registers. A decode object launched with
    /// more than one KV split needs a partial O and LSE per split.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        if(integerMetadata(kernel, DECODE_FIELD) != 1)
        {
            return 0;
        }
        const auto binding = flydslSdpaRdnaBinding(bound);
        auto problem
            = problemFor(requireTensor(context, binding.q), requireTensor(context, binding.k));
        const auto bytes
            = decodeWorkspaceBytes(problem,
                                   decodeSplits(problem,
                                                decodeRowTiles(problem, decodeRowTiled(kernel)),
                                                context.deviceProperties.multiProcessorCount));
        if(!bytes.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM, "flydsl_sdpa_rdna: decode workspace overflows");
        }
        return *bytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = flydslSdpaRdnaBinding(bound);

        // kernel_match declines a candidate without a usable tile, so this is defence in
        // depth: the grid and block below come from block_m, and there is no tile to
        // substitute that would not launch this object with another's geometry.
        const auto tile = candidateTile(kernel);
        if(!tile.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_rdna: kernel '" + toString(kernel.kernelId)
                    + "' declares no usable block_m/block_n tile");
        }

        const auto& q = requireTensor(context, binding.q);
        const auto& k = requireTensor(context, binding.k);
        const auto& v = requireTensor(context, binding.v);
        const auto& o = requireTensor(context, binding.o);
        auto problem = problemFor(q, k);
        problem.numVHeads = v.dims()->Get(HEAD_AXIS);

        // Build defines are ignored on the kpack path, but the signature still wants an
        // options object. Built from the arch alone: the tensor overload classifies a 4-D
        // stride order as NCHW or NHWC and throws for anything else, which attention
        // layouts are.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);

        const bool decode = integerMetadata(kernel, DECODE_FIELD) == 1;
        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            decode ? flydslSdpaRdnaDecodeSignature()
                                                   : flydslSdpaRdnaKernelSignature());

        std::optional<IngestorKernelCode> mergeCode;
        int64_t splits = 1;
        if(decode)
        {
            // `batch_x_kvheads_x_splits_x_dv_split_by_row_tiles`: one workgroup of
            // DECODE_WAVES waves per (batch, KV head, split, output-column tile), by 16-row
            // tile along y; the merge, one workgroup per (batch, query head, position).
            const auto rowTiles = decodeRowTiles(problem, decodeRowTiled(kernel));
            splits = decodeSplits(problem, rowTiles, context.deviceProperties.multiProcessorCount);
            const auto mergeSymbol = stringMetadata(kernel, MERGE_SYMBOL_FIELD);
            const auto mainGrid
                = checkedProduct({problem.batch, problem.numKvHeads, splits, tile->dvSplit});
            const auto mergeGrid
                = checkedProduct({problem.batch, problem.numHeads, problem.seqLenQ});
            if(!mergeSymbol.has_value() || mergeSymbol->empty() || !mainGrid.has_value()
               || *mainGrid > INT32_LIMIT || !mergeGrid.has_value() || *mergeGrid > INT32_LIMIT)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_rdna: decode kernel '" + toString(kernel.kernelId)
                        + "' names no merge kernel, or its launch overflows");
            }
            // The merge kernel is the same code object under another symbol, so the same
            // archive entry with that symbol and its own argument list.
            KernelDefinition mergeKernel = kernel;
            mergeKernel.source.symbol = *mergeSymbol;
            mergeKernel.source.signature = flydslSdpaRdnaMergeSignature();
            mergeCode.emplace(buildIngestorKernelCode(_kernelCompiler,
                                                      _kpackLoader,
                                                      context,
                                                      mergeKernel,
                                                      options,
                                                      flydslSdpaRdnaMergeSignature()));
            code.setBlockSize(static_cast<unsigned int>(DECODE_WAVES * WAVE_SIZE), 1, 1);
            code.setGridSize(
                static_cast<unsigned int>(*mainGrid), static_cast<unsigned int>(rowTiles), 1);
            mergeCode->setBlockSize(static_cast<unsigned int>(DECODE_MERGE_THREADS), 1, 1);
            mergeCode->setGridSize(static_cast<unsigned int>(*mergeGrid), 1, 1);
        }
        else
        {
            // `batch_x_qtiles_x_heads_x_dv_split`: one workgroup per (batch, query tile,
            // head, output-column tile), each block_m / 16 waves of 32. kernel_match
            // bounded it.
            const auto gridX = candidateGrid(problem, *tile);
            if(!gridX.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_rdna: launch grid exceeds int32 for kernel '"
                        + toString(kernel.kernelId) + "'");
            }
            code.setBlockSize(static_cast<unsigned int>(tile->blockM / WAVE_ROWS
                                                        * tile->wavesPerRowTile * WAVE_SIZE),
                              1,
                              1);
            code.setGridSize(static_cast<unsigned int>(*gridX), 1, 1);
        }

        std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand;
        if(binding.scaleSource == ScaleSource::TENSOR)
        {
            scaleOperand = hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), binding.scale, "scale");
        }

        BiasStrides biasStrides{0, 0, 0, 0};
        if(binding.bias != NO_BIAS)
        {
            const auto strides = servableBiasStrides(requireTensor(context, binding.bias),
                                                     problem.batch,
                                                     problem.numHeads,
                                                     problem.seqLenQ,
                                                     problem.seqLenKv,
                                                     problem.dataType);
            if(!strides.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_rdna: the bound bias is not one the kernel can read");
            }
            biasStrides = *strides;
        }

        return std::make_unique<PreparedFlydslSdpaRdna>(
            std::move(code),
            binding,
            problem,
            scaleOperand,
            std::array<OperandStrides, 5>{stridesOf(q),
                                          stridesOf(k),
                                          stridesOf(v),
                                          stridesOf(o),
                                          binding.stats == NO_STATS
                                              ? OperandStrides{}
                                              : stridesOf(requireTensor(context, binding.stats))},
            biasStrides,
            std::move(mergeCode),
            splits);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedSdpa = dynamic_cast<const PreparedFlydslSdpaRdna&>(prepared);
        const auto& binding = preparedSdpa.binding();
        const auto& problem = preparedSdpa.problem();

        float scale = 0.0F;
        if(preparedSdpa.scaleOperand().has_value())
        {
            scale = static_cast<float>(
                hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                    *preparedSdpa.scaleOperand(), deviceBuffers, numDeviceBuffers)));
        }
        else
        {
            const auto scaleBits = static_cast<int32_t>(binding.scale);
            std::memcpy(&scale, &scaleBits, sizeof(scale));
        }

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);
        // The kernel writes the LSE output only when lse_on is set, so with no stats tensor
        // the pointer is never dereferenced.
        void* stats = binding.stats == NO_STATS
                          ? nullptr
                          : hipdnn_plugin_sdk::findDeviceBuffer(
                                binding.stats, deviceBuffers, numDeviceBuffers)
                                .ptr;

        // Read only by the has_bias objects, which kernel_match pairs with a bound bias.
        void* bias = binding.bias == NO_BIAS ? nullptr
                                             : hipdnn_plugin_sdk::findDeviceBuffer(
                                                   binding.bias, deviceBuffers, numDeviceBuffers)
                                                   .ptr;

        const auto& s = preparedSdpa.strides();
        const auto& b = preparedSdpa.biasStrides();

        if(preparedSdpa.isDecode())
        {
            const auto splits = preparedSdpa.splits();
            // The workspace is the partial O (f32, head_dim per row and split) followed by
            // the partial LSEs; unused with one split.
            void* wsO = splits > 1 ? workspace : nullptr;
            void* wsLse
                = splits > 1
                      ? static_cast<void*>(static_cast<char*>(workspace)
                                           + static_cast<size_t>(problem.batch * problem.numHeads
                                                                 * problem.seqLenQ * splits
                                                                 * problem.headDim)
                                                 * sizeof(float))
                      : nullptr;
            const auto lseOn = static_cast<int32_t>(binding.stats == NO_STATS ? 0 : 1);
            // Changing these argument lists means changing flydslSdpaRdnaDecodeSignature() and
            // flydslSdpaRdnaMergeSignature() with them.
            preparedSdpa.kernelForStream(handle.getStream())
                .launch(handle.getStream(),
                        q.ptr,
                        k.ptr,
                        v.ptr,
                        o.ptr,
                        stats,
                        wsO,
                        wsLse,
                        static_cast<int32_t>(problem.seqLenQ),
                        static_cast<int32_t>(problem.seqLenKv),
                        static_cast<int32_t>(problem.numHeads),
                        static_cast<int32_t>(problem.numHeads / problem.numKvHeads),
                        static_cast<int32_t>(binding.rightBound),
                        static_cast<int32_t>(binding.leftBound),
                        static_cast<int32_t>(binding.alignBottomRight),
                        lseOn,
                        static_cast<int32_t>(splits),
                        scale,
                        s[0].batch,
                        s[0].sequence,
                        s[0].head,
                        s[1].batch,
                        s[1].sequence,
                        s[1].head,
                        s[2].batch,
                        s[2].sequence,
                        s[2].head,
                        s[3].batch,
                        s[3].sequence,
                        s[3].head,
                        s[4].batch,
                        s[4].sequence,
                        s[4].head,
                        bias,
                        b[0],
                        b[1],
                        b[2],
                        b[3],
                        static_cast<int32_t>(problem.headDim));
            if(splits > 1)
            {
                preparedSdpa.mergeForStream(handle.getStream())
                    .launch(handle.getStream(),
                            o.ptr,
                            stats,
                            wsO,
                            wsLse,
                            static_cast<int32_t>(problem.seqLenQ),
                            static_cast<int32_t>(problem.numHeads),
                            lseOn,
                            static_cast<int32_t>(splits),
                            s[3].batch,
                            s[3].sequence,
                            s[3].head,
                            s[4].batch,
                            s[4].sequence,
                            s[4].head,
                            static_cast<int32_t>(problem.headDim));
            }
            return;
        }

        // Changing this argument list means changing flydslSdpaRdnaKernelSignature() with it.
        preparedSdpa.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    o.ptr,
                    stats,
                    static_cast<int32_t>(problem.seqLenQ),
                    static_cast<int32_t>(problem.seqLenKv),
                    static_cast<int32_t>(problem.numHeads),
                    static_cast<int32_t>(problem.numHeads / problem.numKvHeads),
                    static_cast<int32_t>(binding.rightBound),
                    static_cast<int32_t>(binding.leftBound),
                    static_cast<int32_t>(binding.alignBottomRight),
                    static_cast<int32_t>(binding.stats == NO_STATS ? 0 : 1),
                    scale,
                    s[0].batch,
                    s[0].sequence,
                    s[0].head,
                    s[1].batch,
                    s[1].sequence,
                    s[1].head,
                    s[2].batch,
                    s[2].sequence,
                    s[2].head,
                    s[3].batch,
                    s[3].sequence,
                    s[3].head,
                    s[4].batch,
                    s[4].sequence,
                    s[4].head,
                    bias,
                    b[0],
                    b[1],
                    b[2],
                    b[3],
                    static_cast<int32_t>(problem.headDim),
                    static_cast<int32_t>(problem.numHeads / problem.numVHeads));
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& flydslSdpaRdnaKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too -- including the
    // RMS-norm pack, which reads the same archive.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaRdnaModuleCache()
{
    flydslSdpaRdnaKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it, but a provider's Container is created and destroyed per handle.
const FlydslSdpaRdnaDispatchHandler& flydslSdpaRdnaDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSdpaRdnaKpackModuleCache());
    static const FlydslSdpaRdnaDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSdpaRdnaSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslSdpaRdnaGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaRdnaKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaRdnaScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaRdnaDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
