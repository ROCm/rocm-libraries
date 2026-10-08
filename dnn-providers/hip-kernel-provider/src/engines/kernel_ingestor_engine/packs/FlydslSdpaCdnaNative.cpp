// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

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
 * @file FlydslSdpaCdnaNative.cpp
 * @brief The flyDSL SDPA-forward pack for CDNA (gfx942, gfx950): matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * A separate engine from the RDNA one (FlydslSdpaRdnaNative.cpp): the kernels, their
 * argument lists and their knobs share nothing. Every kernel this pack selects is a
 * pre-built HSACO compiled ahead of time from FlyDSL's own CDNA attention forwards,
 * vendored under `flydsl/kernels_src/kernels/attention_cdna/` -- the dual-wave
 * software-pipelined kernel on gfx950, the generic MFMA kernel on gfx942.
 *
 * Each object bakes its dtype, head_dim and whether a causal mask applies. Batch, both
 * sequence lengths, both head counts, every operand's (batch, sequence, head) strides, the
 * softmax scale, the mask's diagonal offset and whether to write the LSE are runtime
 * arguments. So the matcher accepts exactly that envelope and declines everything else,
 * and each object serves one (dtype, head_dim, causal) class. The
 * remaining baked fields become runtime one by one, each measured against the object
 * it replaces (flydsl/COVERAGE.md section 7).
 *
 * Kernel arguments. FlyDSL lowers each tensor parameter to two kernel arguments: its
 * 8-byte global pointer, then a 40-byte by-value layout -- the four i32 extents of the
 * rank-4 tensor followed by the i64 strides of its three non-unit-stride dimensions, in
 * ascending order (FlyDSL's dynamic-layout contract, `_LayoutPlan` in
 * flydsl/compiler/jit_argument.py). The semantic names are in each object's manifest.json.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_cdna.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa_cdna.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa_cdna.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa_cdna.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
/// "dualwave" (gfx950) or "generic" (gfx942): the kernel family, which fixes the argument
/// list and the grid.
constexpr std::string_view KERNEL_FIELD = "kernel";
constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
/// 0 for an object that serves any key length; otherwise the multiple the key length must
/// be, because the object skips the KV tail mask.
constexpr std::string_view KV_ALIGN_FIELD = "kv_align";
/// 1 for an object that serves any diagonal offset; 0 for one built without the handling
/// an offset other than 0 needs, which only an offset-0 graph may run.
constexpr std::string_view ANY_DIAGONAL_FIELD = "any_diagonal";

constexpr std::string_view DUALWAVE = "dualwave";
constexpr std::string_view GENERIC = "generic";

constexpr std::string_view Q_TOKEN = "flydsl_sdpa_cdna.q.uid";
constexpr std::string_view K_TOKEN = "flydsl_sdpa_cdna.k.uid";
constexpr std::string_view V_TOKEN = "flydsl_sdpa_cdna.v.uid";
constexpr std::string_view O_TOKEN = "flydsl_sdpa_cdna.o.uid";
/// 1 when the graph asks for causal masking, 0 for none.
constexpr std::string_view CAUSAL_TOKEN = "flydsl_sdpa_cdna.causal";
/// The causal diagonal: query row i sees keys j <= i + offset. Unused without a mask.
constexpr std::string_view DIAG_OFFSET_TOKEN = "flydsl_sdpa_cdna.diag_offset";
/// The uid of the softmax-statistics (LSE) output, or NO_STATS when the graph wants none.
constexpr std::string_view STATS_TOKEN = "flydsl_sdpa_cdna.stats.uid";
constexpr int64_t NO_STATS = -1;

constexpr uint32_t SDPA_RANK = 4;
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_DIM_AXIS = 3;

/// The dual-wave kernel's workgroup is fixed (two waves of 64 per 32 query rows of its
/// 256-row tile); the generic kernel runs two threads per query row of its tile.
constexpr int64_t DUALWAVE_BLOCK_THREADS = 512;
constexpr int64_t GENERIC_THREADS_PER_ROW = 2;

constexpr int64_t INT32_LIMIT = 2147483647LL;
constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and f16 only
/// A bound the graph leaves open (hipDNN's sentinel for left_bound / right_bound).
constexpr int64_t UNBOUNDED = -1;

/// Where the softmax scale comes from; see ScaleSource.
constexpr std::string_view SCALE_SOURCE_TOKEN = "flydsl_sdpa_cdna.scale_source";
/// The scale's f32 bit pattern when it is a value (BoundTokens carry int64_t), else the uid
/// of the pass-by-value scale tensor.
constexpr std::string_view SCALE_TOKEN = "flydsl_sdpa_cdna.scale";

/// Where the kernel's `sm_scale` argument comes from.
enum class ScaleSource : int64_t
{
    /// attn_scale_value, or 1/sqrt(head_dim) when the graph gives none -- the default
    /// hipDNN's CPU reference applies.
    VALUE = 0,
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

/// The product of @p factors, or nullopt when it does not fit in int64_t.
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

/// The kernel metadata spelling of a graph dtype.
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
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.headDim = q.dims()->Get(HEAD_DIM_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

/// One operand's (batch, sequence, head) strides, in elements; head_dim's is 1.
struct OperandStrides
{
    int64_t batch = 0;
    int64_t sequence = 0;
    int64_t head = 0;

    bool operator==(const OperandStrides& other) const
    {
        return batch == other.batch && sequence == other.sequence && head == other.head;
    }
};

/// The kernels move 16-byte vectors: eight bf16 / f16 elements.
constexpr int64_t VECTOR_ELEMENTS = 8;

/**
 * @brief The strides of @p tensor as the kernels take them, or nullopt for a tensor they
 *        cannot read.
 *
 * Logical shape [B, H, S, D] with head_dim contiguous; the batch, sequence and head strides
 * are kernel arguments, so BSHD, BHSD and packed-QKV views are all served. Each stride on an
 * axis longer than 1 must be positive and keep every row 16-byte aligned. Each (batch, head)
 * slice is one buffer view of (S - 1) * stride_s + D elements, which must stay within 32-bit
 * addressing; rows past the sequence -- a query tile's padding rows -- read as zero and
 * store nowhere only because they fall outside it, which needs a sequence stride of at
 * least D. So a single-row sequence is given stride D rather than its stated one, which
 * is never stepped, and overlapping rows decline. A batch or head axis of extent 1 is never
 * stepped and gets stride 0. A ragged operand would be read as one dense batch, so it
 * declines too.
 */
std::optional<OperandStrides> servableStrides(const data_objects::TensorAttributes& tensor,
                                              int64_t batch,
                                              int64_t heads,
                                              int64_t sequence,
                                              int64_t headDim)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return std::nullopt;
    }
    if(dims->Get(BATCH_AXIS) != batch || dims->Get(HEAD_AXIS) != heads
       || dims->Get(SEQ_AXIS) != sequence || dims->Get(HEAD_DIM_AXIS) != headDim
       || strides->Get(HEAD_DIM_AXIS) != 1)
    {
        return std::nullopt;
    }
    const auto stepped = [&](uint32_t axis, int64_t extent) -> std::optional<int64_t> {
        if(extent == 1)
        {
            return axis == SEQ_AXIS ? headDim : 0;
        }
        const int64_t stride = strides->Get(axis);
        if(stride <= 0 || stride % VECTOR_ELEMENTS != 0 || stride > INT32_LIMIT)
        {
            return std::nullopt;
        }
        return stride;
    };
    const auto batchStride = stepped(BATCH_AXIS, batch);
    const auto sequenceStride = stepped(SEQ_AXIS, sequence);
    const auto headStride = stepped(HEAD_AXIS, heads);
    if(!batchStride.has_value() || !sequenceStride.has_value() || !headStride.has_value()
       || *sequenceStride < headDim)
    {
        return std::nullopt;
    }
    const auto spanRows = checkedProduct({sequence - 1, *sequenceStride});
    if(!spanRows.has_value() || (*spanRows + headDim) * BYTES_PER_ELEMENT > INT32_LIMIT)
    {
        return std::nullopt;
    }
    if(tensor.ragged_offset_tensor_uid().has_value() || tensor.virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return std::nullopt;
    }
    return OperandStrides{*batchStride, *sequenceStride, *headStride};
}

/// The strides the kernels take: Q's, O's, and the one set K and V share.
struct ViewStrides
{
    OperandStrides q;
    OperandStrides o;
    OperandStrides kv;
};

/// The operands' strides, or nullopt when any is unservable or K and V differ -- the
/// kernels take one set for both.
std::optional<ViewStrides> viewStridesFor(const data_objects::TensorAttributes& q,
                                          const data_objects::TensorAttributes& k,
                                          const data_objects::TensorAttributes& v,
                                          const data_objects::TensorAttributes& o,
                                          const SdpaProblem& problem)
{
    const auto qStrides
        = servableStrides(q, problem.batch, problem.numHeads, problem.seqLenQ, problem.headDim);
    const auto oStrides
        = servableStrides(o, problem.batch, problem.numHeads, problem.seqLenQ, problem.headDim);
    const auto kStrides
        = servableStrides(k, problem.batch, problem.numKvHeads, problem.seqLenKv, problem.headDim);
    const auto vStrides
        = servableStrides(v, problem.batch, problem.numKvHeads, problem.seqLenKv, problem.headDim);
    if(!qStrides.has_value() || !oStrides.has_value() || !kStrides.has_value()
       || !vStrides.has_value() || !(*kStrides == *vStrides))
    {
        return std::nullopt;
    }
    return ViewStrides{*qStrides, *oStrides, *kStrides};
}

/// Whether @p stats is an LSE output the kernels write: f32 [B, H, Sq, 1] in packed
/// (batch, head, query) order -- the kernels address it as one contiguous [B, H, Sq].
bool isServableStats(const data_objects::TensorAttributes& stats,
                     int64_t batch,
                     int64_t heads,
                     int64_t sequence)
{
    const auto* dims = stats.dims();
    const auto* strides = stats.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }
    if(dims->Get(BATCH_AXIS) != batch || dims->Get(HEAD_AXIS) != heads
       || dims->Get(SEQ_AXIS) != sequence || dims->Get(HEAD_DIM_AXIS) != 1)
    {
        return false;
    }
    // A stride on an axis of extent 1 is never used to address anything.
    if((sequence > 1 && strides->Get(SEQ_AXIS) != 1)
       || (heads > 1 && strides->Get(HEAD_AXIS) != sequence)
       || (batch > 1 && strides->Get(BATCH_AXIS) != heads * sequence))
    {
        return false;
    }
    return stats.data_type() == data_objects::DataType::FLOAT && !stats.virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&stats)
           && !stats.ragged_offset_tensor_uid().has_value();
}

/// The mask the kernels apply: none, or j <= i + offset for query row i and key j.
struct CdnaMask
{
    bool causal = false;
    int64_t offset = 0;
};

/**
 * @brief The graph's mask in the kernels' terms, or nullopt for one they cannot apply.
 *
 * The reference (CpuFpReferenceSdpa isMasked) hides key j from query row i when
 * j > i + align + right, with align 0 top-left and Skv - Sq bottom-right; so causal and a
 * band to the right of the diagonal are both one offset, align + right. A left bound (a
 * sliding window) declines. So does an offset past Skv - Sq: the causal objects mask the
 * last KV tile's padding with the same compare, which holds only while every row's
 * diagonal stays inside the keys -- top-left with more queries than keys, or a band.
 */
std::optional<CdnaMask>
    maskFor(const data_objects::SdpaAttributes& attributes, int64_t seqLenQ, int64_t seqLenKv)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();
    if(topLeftDeprecated && bottomRightDeprecated)
    {
        return std::nullopt;
    }
    const int64_t left
        = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    int64_t right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;
    if(left != UNBOUNDED || right < UNBOUNDED)
    {
        return std::nullopt;
    }
    bool alignBottomRight = false;
    if(topLeftDeprecated || bottomRightDeprecated)
    {
        // The booleans mean right 0 at their corner; a right_bound the graph also sets wins.
        right = right == UNBOUNDED ? 0 : right;
        alignBottomRight = bottomRightDeprecated;
    }
    else
    {
        alignBottomRight
            = attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }
    if(right == UNBOUNDED)
    {
        return CdnaMask{};
    }
    const int64_t offset = (alignBottomRight ? seqLenKv - seqLenQ : 0) + right;
    if(offset > seqLenKv - seqLenQ)
    {
        return std::nullopt;
    }
    return CdnaMask{true, offset};
}

/**
 * @brief Graph-scoped applicability: is this a single SDPA-forward node one of this pack's
 *        pre-built CDNA kernels computes exactly?
 *
 * Refuses rather than approximates: every feature the kernels do not implement is
 * declined, so a graph asking for one falls through to an engine that can honour it.
 */
std::optional<BoundTokens> flydslSdpaCdnaGraphMatches(const MatchContext& context)
{
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr || q->dims() == nullptr
       || k->dims() == nullptr || q->dims()->size() != SDPA_RANK || k->dims()->size() != SDPA_RANK)
    {
        return std::nullopt;
    }

    const auto problem = problemFor(*q, *k);
    if(problem.batch <= 0 || problem.numHeads <= 0 || problem.numKvHeads <= 0
       || problem.seqLenQ <= 0 || problem.seqLenKv <= 0 || problem.headDim <= 0)
    {
        return std::nullopt;
    }
    // One dtype throughout, and one the kernels are built for.
    if(!kernelDataTypeName(problem.dataType).has_value() || k->data_type() != problem.dataType
       || v->data_type() != problem.dataType || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    // Contiguous BSHD, V shaped as K, O as Q, GQA by integer division.
    if(problem.numHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }
    if(!viewStridesFor(*q, *k, *v, *o, problem).has_value())
    {
        return std::nullopt;
    }
    if(problem.seqLenQ > INT32_LIMIT || problem.seqLenKv > INT32_LIMIT
       || problem.batch > INT32_LIMIT)
    {
        return std::nullopt;
    }

    const auto mask = maskFor(attributes, problem.seqLenQ, problem.seqLenKv);
    if(!mask.has_value())
    {
        return std::nullopt;
    }

    // Every optional feature the kernels as built do not implement, declined explicitly.
    if(attributes.attn_mask_tensor_uid().has_value()
       || attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return std::nullopt;
    }
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value() || attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value())
    {
        return std::nullopt;
    }
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
    // the kernels cannot satisfy.
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
    if(attributes.alibi_mask() || attributes.padding_mask())
    {
        return std::nullopt;
    }
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }

    // The softmax scale: a value, a pass-by-value tensor, or the default.
    ScaleSource scaleSource = ScaleSource::VALUE;
    int64_t scaleToken = 0;
    if(attributes.scale_tensor_uid().has_value())
    {
        // Only a host scalar can become a kernel argument; a device-resident scale would
        // need a device read the kernels do not do.
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
        // The kernels take the running max over unscaled scores, which orders them the
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
    bound[std::string(CAUSAL_TOKEN)] = mask->causal ? 1 : 0;
    bound[std::string(DIAG_OFFSET_TOKEN)] = mask->offset;
    bound[std::string(STATS_TOKEN)] = statsUid;
    bound[std::string(SCALE_SOURCE_TOKEN)] = static_cast<int64_t>(scaleSource);
    bound[std::string(SCALE_TOKEN)] = scaleToken;
    return bound;
}

/// The bindings a match established, re-read for dispatch.
struct FlydslSdpaCdnaBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
    int64_t diagOffset = 0;
    int64_t stats = NO_STATS;
    ScaleSource scaleSource = ScaleSource::VALUE;
    int64_t scale = 0;
};

FlydslSdpaCdnaBinding flydslSdpaCdnaBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_sdpa_cdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    FlydslSdpaCdnaBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.causal = read(CAUSAL_TOKEN);
    binding.diagOffset = read(DIAG_OFFSET_TOKEN);
    binding.stats = read(STATS_TOKEN);
    binding.scaleSource = static_cast<ScaleSource>(read(SCALE_SOURCE_TOKEN));
    binding.scale = read(SCALE_TOKEN);
    return binding;
}

/// A candidate's integer metadata field, or nullopt when absent or of another type.
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

/// The string counterpart of integerMetadata.
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

/// A candidate's launch shape, or nullopt when its metadata describes none this pack
/// knows how to launch.
struct CdnaTile
{
    bool dualwave = false;
    int64_t blockM = 0;
};

std::optional<CdnaTile> candidateTile(const KernelDefinition& kernel)
{
    const auto family = stringMetadata(kernel, KERNEL_FIELD);
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    if(!family.has_value() || !blockM.has_value() || *blockM <= 0)
    {
        return std::nullopt;
    }
    if(*family == DUALWAVE)
    {
        return CdnaTile{true, *blockM};
    }
    if(*family == GENERIC)
    {
        return CdnaTile{false, *blockM};
    }
    return std::nullopt;
}

/// The launch grid, or nullopt past int32: the dual-wave kernel one workgroup per (query
/// head, query block, batch); the generic one per (batch, query tile, query head),
/// flattened onto x.
std::optional<std::array<int64_t, 3>> candidateGrid(const SdpaProblem& problem,
                                                    const CdnaTile& tile)
{
    const auto queryBlocks = (problem.seqLenQ + tile.blockM - 1) / tile.blockM;
    if(tile.dualwave)
    {
        if(queryBlocks > INT32_LIMIT || problem.numHeads > INT32_LIMIT)
        {
            return std::nullopt;
        }
        return std::array<int64_t, 3>{problem.numHeads, queryBlocks, problem.batch};
    }
    const auto gridX = checkedProduct({problem.batch, queryBlocks, problem.numHeads});
    if(!gridX.has_value() || *gridX > INT32_LIMIT)
    {
        return std::nullopt;
    }
    return std::array<int64_t, 3>{*gridX, 1, 1};
}

/**
 * @brief Kernel-scoped applicability: do THIS candidate's baked fields fit the graph?
 *
 * dtype, head_dim and the causal flag are baked, so each must equal the graph's; the key length must be a multiple of the candidate's `kv_align`, since an
 * object built without the KV tail mask reads past the end of a ragged last tile; and the
 * candidate must declare a launch shape this pack knows.
 */
bool flydslSdpaCdnaKernelMatches(const MatchContext& context,
                                 const BoundTokens& bound,
                                 const KernelDefinition& kernel)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* k = findTensor(context, attributes->k_tensor_uid());
    if(q == nullptr || k == nullptr)
    {
        return false;
    }
    const auto problem = problemFor(*q, *k);
    const auto tile = candidateTile(kernel);
    if(!tile.has_value() || !candidateGrid(problem, *tile).has_value())
    {
        return false;
    }

    const auto anyDiagonal = integerMetadata(kernel, ANY_DIAGONAL_FIELD);
    const auto diagOffset = tryGetBoundInt(bound, DIAG_OFFSET_TOKEN);
    if(!anyDiagonal.has_value() || !diagOffset.has_value()
       || (*anyDiagonal == 0 && *diagOffset != 0))
    {
        return false;
    }
    const auto kvAlign = integerMetadata(kernel, KV_ALIGN_FIELD);
    if(!kvAlign.has_value() || *kvAlign < 0 || (*kvAlign > 0 && problem.seqLenKv % *kvAlign != 0))
    {
        return false;
    }

    const auto dataTypeName = kernelDataTypeName(problem.dataType);
    const auto causal = tryGetBoundInt(bound, CAUSAL_TOKEN);
    return dataTypeName.has_value() && stringMetadata(kernel, DTYPE_FIELD) == *dataTypeName
           && integerMetadata(kernel, HEAD_DIM_FIELD) == problem.headDim && causal.has_value()
           && integerMetadata(kernel, CAUSAL_FIELD) == *causal;
}

double flydslSdpaCdnaScore(const MatchContext& /*context*/,
                           const BoundTokens& /*bound*/,
                           const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// FlyDSL's dynamic layout for a rank-4 tensor: the four extents, then the strides of the
/// three dimensions whose stride is not 1, in ascending order. Passed by value after each
/// tensor's pointer.
struct FlydslLayout4
{
    std::array<int32_t, 4> extents;
    std::array<int64_t, 3> strides;
};
static_assert(sizeof(FlydslLayout4) == 40, "FlyDSL's rank-4 dynamic layout is 40 bytes");

/// The layout of a contiguous BSHD tensor viewed as [B, S, H, D], as upstream's launchers
/// pass it: strides (S*H*D, H*D, D) for the batch, sequence and head dimensions. The
/// kernels read only each tensor's pointer -- the strides are their own arguments -- so this
/// fills the layout slots in a well-formed way whatever the operands' strides.
FlydslLayout4 bshdLayout(int64_t batch, int64_t sequence, int64_t heads, int64_t headDim)
{
    FlydslLayout4 layout{};
    layout.extents[0] = static_cast<int32_t>(batch);
    layout.extents[1] = static_cast<int32_t>(sequence);
    layout.extents[2] = static_cast<int32_t>(heads);
    layout.extents[3] = static_cast<int32_t>(headDim);
    layout.strides[0] = sequence * heads * headDim;
    layout.strides[1] = heads * headDim;
    layout.strides[2] = headDim;
    return layout;
}

/**
 * @brief The argument lists this pack launches with, mirroring launch() one for one.
 *
 * Dual-wave (flash_attn_gfx950.py): Q, K, V, O, LSE, DebugCounts, CuSeqQ, CuSeqKv,
 * BlockTable, Bias, AlibiSlopes, Sink (each a pointer and its layout); then seq_len,
 * seq_len_kv, stride_q_n, stride_kv_n, head_dim_runtime, block_table_stride,
 * bias_stride0, alibi_stride_b (i32); sm_scale (f32); num_heads_q, num_heads_kv, lse_on,
 * diag_offset, then the strides stride_q_b, stride_q_h, stride_o_b, stride_o_n,
 * stride_o_h, stride_kv_b, stride_kv_h (i32). Generic (flash_attn_generic.py): Q, K, V, O,
 * LSE; seq_len, seq_len_kv (i32); CuSeqQ, CuSeqKv, BlockTable; block_table_stride (i32);
 * sm_scale (f32); num_heads_q, num_heads_kv, stride_q_n, stride_kv_n, lse_on, diag_offset
 * and the same seven strides (i32). A 4-byte scalar slot is the same to the
 * signature check whichever type it holds, so each list is written as runs of tensors
 * and scalars.
 */
struct ArgumentRun
{
    int count;
    bool tensors;
};

std::vector<KernelArgument> signatureOf(std::initializer_list<ArgumentRun> runs)
{
    const KernelArgument pointer{"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    const KernelArgument layout{"by_value", static_cast<uint32_t>(sizeof(FlydslLayout4)), 0, ""};
    const KernelArgument i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    std::vector<KernelArgument> signature;
    for(const auto& run : runs)
    {
        for(int i = 0; i < run.count; ++i)
        {
            if(run.tensors)
            {
                signature.push_back(pointer);
                signature.push_back(layout);
            }
            else
            {
                signature.push_back(i32);
            }
        }
    }
    return signature;
}

const std::vector<KernelArgument>& dualwaveSignature()
{
    static const std::vector<KernelArgument> s_signature = signatureOf({{12, true}, {20, false}});
    return s_signature;
}

const std::vector<KernelArgument>& genericSignature()
{
    static const std::vector<KernelArgument> s_signature
        = signatureOf({{5, true}, {2, false}, {3, true}, {15, false}});
    return s_signature;
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched flydsl_sdpa_cdna graph has no tensor for uid " + std::to_string(uid));
    }
    return *tensor;
}

/// The compiled kernel plus everything launch() needs.
class PreparedFlydslSdpaCdna : public PreparedDispatch
{
public:
    PreparedFlydslSdpaCdna(IngestorKernelCode code,
                           FlydslSdpaCdnaBinding binding,
                           SdpaProblem problem,
                           bool dualwave,
                           std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand,
                           ViewStrides strides)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _dualwave(dualwave)
        , _scaleOperand(scaleOperand)
        , _strides(strides)
    {
    }

    const ViewStrides& strides() const
    {
        return _strides;
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslSdpaCdnaBinding& binding() const
    {
        return _binding;
    }

    const SdpaProblem& problem() const
    {
        return _problem;
    }

    bool dualwave() const
    {
        return _dualwave;
    }

    /// Set when the scale is a pass-by-value tensor, read at each launch.
    const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scaleOperand() const
    {
        return _scaleOperand;
    }

private:
    IngestorKernelCode _code;
    FlydslSdpaCdnaBinding _binding;
    SdpaProblem _problem;
    bool _dualwave = false;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _scaleOperand;
    ViewStrides _strides;
};

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and the scale.
 */
class FlydslSdpaCdnaDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    FlydslSdpaCdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                  const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The forward needs no scratch beyond LDS and registers.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = flydslSdpaCdnaBinding(bound);
        const auto tile = candidateTile(kernel);
        if(!tile.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_cdna: kernel '" + toString(kernel.kernelId)
                    + "' declares no kernel family or block_m this pack can launch");
        }
        const auto problem
            = problemFor(requireTensor(context, binding.q), requireTensor(context, binding.k));
        const auto grid = candidateGrid(problem, *tile);
        if(!grid.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_cdna: launch grid exceeds int32 for kernel '"
                    + toString(kernel.kernelId) + "'");
        }

        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code
            = buildIngestorKernelCode(_kernelCompiler,
                                      _kpackLoader,
                                      context,
                                      kernel,
                                      options,
                                      tile->dualwave ? dualwaveSignature() : genericSignature());
        const int64_t threads
            = tile->dualwave ? DUALWAVE_BLOCK_THREADS : tile->blockM * GENERIC_THREADS_PER_ROW;
        code.setBlockSize(static_cast<unsigned int>(threads), 1, 1);
        code.setGridSize(static_cast<unsigned int>((*grid)[0]),
                         static_cast<unsigned int>((*grid)[1]),
                         static_cast<unsigned int>((*grid)[2]));
        std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand;
        if(binding.scaleSource == ScaleSource::TENSOR)
        {
            scaleOperand = hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), binding.scale, "scale");
        }
        const auto strides = viewStridesFor(requireTensor(context, binding.q),
                                            requireTensor(context, binding.k),
                                            requireTensor(context, binding.v),
                                            requireTensor(context, binding.o),
                                            problem);
        if(!strides.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_cdna: the bound operands' strides are not ones the kernels read");
        }
        return std::make_unique<PreparedFlydslSdpaCdna>(
            std::move(code), binding, problem, tile->dualwave, scaleOperand, *strides);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedSdpa = dynamic_cast<const PreparedFlydslSdpaCdna&>(prepared);
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

        void* q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers).ptr;
        void* k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers).ptr;
        void* v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers).ptr;
        void* o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers).ptr;
        // The objects carry the LSE store; with no stats tensor bound it gets an empty
        // buffer (lse_on 0) and every LSE store is dropped, so O stands in for the pointer.
        const bool lseOn = binding.stats != NO_STATS;
        void* lse = lseOn ? hipdnn_plugin_sdk::findDeviceBuffer(
                                binding.stats, deviceBuffers, numDeviceBuffers)
                                .ptr
                          : o;
        const int32_t lseFlag = lseOn ? 1 : 0;

        const auto qLayout
            = bshdLayout(problem.batch, problem.seqLenQ, problem.numHeads, problem.headDim);
        const auto kvLayout
            = bshdLayout(problem.batch, problem.seqLenKv, problem.numKvHeads, problem.headDim);
        const auto seqLenQ = static_cast<int32_t>(problem.seqLenQ);
        const auto seqLenKv = static_cast<int32_t>(problem.seqLenKv);
        const auto diagOffset = static_cast<int32_t>(binding.diagOffset);
        const auto numHeads = static_cast<int32_t>(problem.numHeads);
        const auto numKvHeads = static_cast<int32_t>(problem.numKvHeads);
        const auto& strides = preparedSdpa.strides();
        const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
        const int32_t zero = 0;
        auto& kernel = preparedSdpa.kernelForStream(handle.getStream());

        // A tensor the launch does not use is passed as O, as upstream's launchers do; the
        // kernels as built never read it. Changing these argument lists means changing
        // dualwaveSignature() / genericSignature() with them.
        if(preparedSdpa.dualwave())
        {
            const auto headDim = static_cast<int32_t>(problem.headDim);
            kernel.launch(handle.getStream(),
                          q,
                          qLayout,
                          k,
                          kvLayout,
                          v,
                          kvLayout,
                          o,
                          qLayout,
                          lse,
                          qLayout, // read for its pointer only
                          o, // DebugCounts
                          qLayout,
                          o, // CuSeqQ
                          qLayout,
                          o, // CuSeqKv
                          qLayout,
                          o, // BlockTable
                          qLayout,
                          o, // Bias
                          qLayout,
                          o, // AlibiSlopes
                          qLayout,
                          o, // Sink
                          qLayout,
                          seqLenQ,
                          seqLenKv,
                          i32(strides.q.sequence),
                          i32(strides.kv.sequence),
                          headDim,
                          zero,
                          zero,
                          zero,
                          scale,
                          numHeads,
                          numKvHeads,
                          lseFlag,
                          diagOffset,
                          i32(strides.q.batch),
                          i32(strides.q.head),
                          i32(strides.o.batch),
                          i32(strides.o.sequence),
                          i32(strides.o.head),
                          i32(strides.kv.batch),
                          i32(strides.kv.head));
            return;
        }
        kernel.launch(handle.getStream(),
                      q,
                      qLayout,
                      k,
                      kvLayout,
                      v,
                      kvLayout,
                      o,
                      qLayout,
                      lse,
                      qLayout, // read for its pointer only
                      seqLenQ,
                      seqLenKv,
                      o, // CuSeqQ
                      qLayout,
                      o, // CuSeqKv
                      qLayout,
                      o, // BlockTable
                      qLayout,
                      zero,
                      scale,
                      numHeads,
                      numKvHeads,
                      i32(strides.q.sequence),
                      i32(strides.kv.sequence),
                      lseFlag,
                      diagOffset,
                      i32(strides.q.batch),
                      i32(strides.q.head),
                      i32(strides.o.batch),
                      i32(strides.o.sequence),
                      i32(strides.o.head),
                      i32(strides.kv.batch),
                      i32(strides.kv.head));
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& flydslSdpaCdnaKpackModuleCache()
{
    // Process-lifetime and this pack's own; see flydslSdpaRdnaKpackModuleCache.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaCdnaModuleCache()
{
    flydslSdpaCdnaKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it.
const FlydslSdpaCdnaDispatchHandler& flydslSdpaCdnaDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSdpaCdnaKpackModuleCache());
    static const FlydslSdpaCdnaDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSdpaCdnaSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslSdpaCdnaGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaCdnaKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaCdnaScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaCdnaDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
