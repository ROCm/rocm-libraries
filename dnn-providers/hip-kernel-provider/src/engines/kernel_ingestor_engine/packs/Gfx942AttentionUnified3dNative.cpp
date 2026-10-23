// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dArguments.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dQueryOffsets.hpp"

/**
 * @file Gfx942AttentionUnified3dNative.cpp
 * @brief The hipkernel:Gfx942AttentionUnified3d engine's native half: matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * The kernels are rocKE's gfx942 split-KV decode pair
 * (`rocke/library/kernels/gfx942/attention_tiled_3d.py`, builder
 * `build_unified_attention_3d_module`), built with `kv_layout="strided"`: a `segment` launch
 * writes per-segment partial outputs into the workspace and a `reduce` launch combines them
 * into O. Both live in one packaged module (descriptor `sources` {segment, reduce}).
 *
 * rocKE's admission for this pair is `dispatch/attention/strided_decode.py::make_candidate()`
 * (paths relative to `rocke/library`). graph_match restates it exactly, minus the v1
 * refusals listed on gfx942AttentionUnified3dGraphMatches;
 * TestGfx942AttentionUnified3dAdmission.cpp checks the restatement against a table
 * generated from that Python.
 *
 * Launch arguments restate the strided branch of
 * `kernels/common/attention_unified.py::run_unified_attention_torch` and
 * `_run_3d_tiled_on_stream`, with the K/V stride arguments of
 * `kernels/common/attention_kv_cache.py::KvTensorLayout.arguments`.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx942_attention_unified3d.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx942_attention_unified3d.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx942_attention_unified3d.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx942_attention_unified3d.dispatch";

// KMD fields (gfx942_attention_unified3d.kmd.json). The first five are the kernel's
// feature cell and must equal the graph's; the last three are performance knobs the
// scorer ranks on (block_size, num_segments) or the build resolved from them
// (tile_size_override).
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view TILE_SIZE_OVERRIDE_FIELD = "tile_size_override";
constexpr std::string_view NUM_SEGMENTS_FIELD = "num_segments";

constexpr std::string_view Q_TOKEN = "gfx942_attention_unified3d.q.uid";
constexpr std::string_view K_TOKEN = "gfx942_attention_unified3d.k.uid";
constexpr std::string_view V_TOKEN = "gfx942_attention_unified3d.v.uid";
constexpr std::string_view O_TOKEN = "gfx942_attention_unified3d.o.uid";
/// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
constexpr std::string_view SCALE_BITS_TOKEN = "gfx942_attention_unified3d.scale_bits";

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of layout.
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;

/// `kernels/common/attention_kv_cache.py::KvTensorLayout`: the per-head span and the token
/// stride must stay within the buffer-resource offset range the kernel addresses K/V with
/// (`_attention_strided_kv.py`: out-of-range lanes read 0x7FFFFFFF and get zero).
constexpr int64_t STRIDED_KV_SPAN_LIMIT_BYTES = 0x7FFF0000;

/// Q, O and the K/V bases and rows must be 16-byte aligned (`run_unified_attention_torch`,
/// `KvTensorLayout.from_tensor`: vector loads).
constexpr int64_t VECTOR_ALIGNMENT_BYTES = 16;

/// The CU count rocKE's `_resolve_num_cus` falls back to when it cannot read the device
/// (`dispatch/attention/common.py`); used only by the scorer, never for correctness.
constexpr int64_t FALLBACK_NUM_CUS = 120;

/// The KV block size the scorer prefers: `AttentionRequest.kv_block_size`'s default (16),
/// the block rocKE's strided decode runs when the caller does not choose one.
constexpr int64_t PREFERRED_BLOCK_SIZE = 16;

// ---------------------------------------------------------------------------
// Matching helpers
// ---------------------------------------------------------------------------

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node this engine's matchers read, or nullptr if the graph is not a single
/// SDPA-forward node.
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

/// Total over an UNVALIDATED graph: rank, stride/dim agreement and positive extents,
/// checked before anything indexes an axis.
bool isWellFormedOperand(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }
    for(const auto dim : *dims)
    {
        if(dim <= 0)
        {
            return false;
        }
    }
    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/**
 * @brief Is this Q or O tensor the contiguous `[B, Hq, D]` rocKE requires?
 *
 * `run_unified_attention_torch` (strided branch) refuses a Q/O that is not a contiguous
 * `[B, Hq, D]` tensor, and the kernel indexes it as `(token * Hq + head) * D` with no stride
 * arguments. hipDNN spells it `[B, Hq, 1, D]`; the expected strides are batch `Hq * D`,
 * head `D`, dim 1. Unit-extent axes are exempt (their index is always 0), as torch's
 * `is_contiguous` exempts them, so the sequence axis (extent 1) never constrains.
 */
bool hasDecodeQoStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);
    const auto axisOk = [&](uint32_t axis, std::optional<int64_t> expected) {
        return dims->Get(axis) == 1 || (expected.has_value() && strides->Get(axis) == *expected);
    };
    return axisOk(BATCH_AXIS, checkedProduct({heads, headSize})) && axisOk(HEAD_AXIS, headSize)
           && axisOk(HEAD_SIZE_AXIS, 1);
}

/**
 * @brief Why a K or V tensor is not a strided view the kernel can address, or nullopt.
 *
 * Restates `kernels/common/attention_kv_cache.py::KvTensorLayout.__post_init__` for the
 * logical `[B, Hkv, Skv, D]` view (hipDNN's K/V logical order is the same), in element
 * strides with 2-byte elements:
 *
 *  - every stride positive, D stride 1;
 *  - outer strides (B, H, S) of non-unit axes keep 16-byte row alignment;
 *  - no overlap (sorted-stride sweep over non-unit axes);
 *  - the address span fits signed 64-bit bytes;
 *  - per-head span `2 * ((Skv - 1) * stride_S + D)` and token stride `2 * stride_S` at most
 *    0x7FFF0000 bytes (the buffer-resource offset limit);
 *  - batch and head byte strides fit signed 64-bit.
 *
 * The base pointer's 16-byte alignment (`from_tensor`) is a launch-time fact; launch()
 * checks it.
 */
std::optional<std::string> stridedKvRefusal(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    std::array<int64_t, SDPA_RANK> extent{};
    std::array<int64_t, SDPA_RANK> stride{};
    for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        extent[axis] = dims->Get(axis);
        stride[axis] = strides->Get(axis);
        if(stride[axis] <= 0)
        {
            return "strides must be positive";
        }
    }
    if(stride[HEAD_SIZE_AXIS] != 1)
    {
        return "the head-size (D) stride must be 1";
    }
    for(uint32_t axis = 0; axis < HEAD_SIZE_AXIS; ++axis)
    {
        const auto bytes = checkedProduct({stride[axis], GFX942_UNIFIED_3D_KV_ELEMENT_BYTES});
        if(!bytes.has_value())
        {
            return "a stride in bytes exceeds signed 64-bit";
        }
        if(extent[axis] > 1 && *bytes % VECTOR_ALIGNMENT_BYTES != 0)
        {
            return "outer strides must preserve 16-byte alignment";
        }
    }

    // Sufficient non-overlap check for dense permutations and padding (sorted by stride,
    // then extent, as Python's tuple sort).
    std::vector<std::pair<int64_t, int64_t>> axes;
    for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        if(extent[axis] > 1)
        {
            axes.emplace_back(stride[axis], extent[axis]);
        }
    }
    std::sort(axes.begin(), axes.end());
    int64_t span = 1;
    for(const auto& [axisStride, axisExtent] : axes)
    {
        if(axisStride < span)
        {
            return "the view may overlap";
        }
        const auto step = checkedProduct({axisExtent - 1, axisStride});
        if(!step.has_value() || __builtin_add_overflow(span, *step, &span))
        {
            return "the address span exceeds signed 64-bit bytes";
        }
    }
    if(!checkedProduct({span, GFX942_UNIFIED_3D_KV_ELEMENT_BYTES}).has_value())
    {
        return "the address span exceeds signed 64-bit bytes";
    }

    // The same token-stride and head-span bytes the launch passes (gfx942KvHeadSpan), held
    // to the buffer-offset limit, which is tighter than the arguments' int32.
    const auto headSpan
        = gfx942KvHeadSpan(extent[SEQ_AXIS], stride[SEQ_AXIS], extent[HEAD_SIZE_AXIS]);
    if(!headSpan.has_value() || headSpan->spanBytes > STRIDED_KV_SPAN_LIMIT_BYTES
       || headSpan->tokenBytes > STRIDED_KV_SPAN_LIMIT_BYTES)
    {
        return "the per-head span or token stride exceeds the 0x7FFF0000-byte buffer-offset "
               "limit";
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Decline logging (see Gfx950AttentionDenseNative.cpp: every message is built inside the
// macro, so with INFO off a decline costs the check and nothing more).
// ---------------------------------------------------------------------------

struct Declined
{
    std::string_view cause;
};

std::ostream& operator<<(std::ostream& out, const Declined& declined)
{
    return out << GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME << " declined the graph ["
               << declined.cause << "]: ";
}

bool hasSdpaForwardNode(const MatchContext& context)
{
    for(const auto& node : context.graph.nodeWrappers())
    {
        if(node->attributesType() == data_objects::NodeAttributes::SdpaAttributes)
        {
            return true;
        }
    }
    return false;
}

std::string describeOperand(std::string_view role, const data_objects::TensorAttributes& tensor)
{
    using hipdnn_data_sdk::utilities::vecToString;
    using hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector;
    std::ostringstream out;
    out << role << " (uid " << tensor.uid() << "): dims "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.dims())) << ", strides "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.strides()));
    return out.str();
}

/// The kernel's dtype spelling (the KMD `dtype` field) for a graph dtype, or nullopt.
std::optional<std::string> supportedDataTypeName(data_objects::DataType dataType)
{
    if(dataType == data_objects::DataType::BFLOAT16)
    {
        return std::string("BF16");
    }
    if(dataType == data_objects::DataType::HALF)
    {
        return std::string("FP16");
    }
    return std::nullopt;
}

/// The graph facts the matchers, scorer and prepare() all read, from validated Q and K.
struct Unified3dProblem
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

Unified3dProblem problemFor(const data_objects::TensorAttributes& q,
                            const data_objects::TensorAttributes& k)
{
    Unified3dProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numQueryHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headSize = q.dims()->Get(HEAD_SIZE_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

/// The problem of a graph graph_match accepted, or nullopt for one it would not have.
std::optional<Unified3dProblem> matchedProblem(const MatchContext& context)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return std::nullopt;
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* k = findTensor(context, attributes->k_tensor_uid());
    if(q == nullptr || k == nullptr || !isWellFormedOperand(*q) || !isWellFormedOperand(*k))
    {
        return std::nullopt;
    }
    return problemFor(*q, *k);
}

/**
 * @brief Which mask the graph asks for, or nullopt for one this engine refuses.
 *
 * Classified as Gfx950AttentionDenseNative.cpp's maskTypeFor does (resolveDiagonalBand
 * then getMaskType, no makeCoveringBoundsUnbound). Accepted: NO_MASK and
 * BOTTOM_RIGHT_CAUSAL, which coincide at Sq = 1 (the one query row is the last row and
 * sees every key); the kernel computes `context_len = seq_len - q_len` and is bottom-right
 * by construction. Refused:
 *
 *  - TOP_LEFT_CAUSAL: rocKE refuses it ("strided decode uses bottom-right causal
 *    alignment", `strided_decode.py`), even at Skv = 1 where the two coincide.
 *  - Any window or band (SLIDING_WINDOW): rocKE admits a sliding window, but this engine
 *    refuses it in v1. No shipped variant carries a non-zero sliding_window, and the
 *    split rule the scorer restates (Gfx942AttentionUnified3dGeometry.hpp) holds only for
 *    sliding_window 0.
 */
bool maskAccepted(const data_objects::SdpaAttributes& attributes)
{
    using asm_sdpa_engine::plan_utils::getMaskType;
    using asm_sdpa_engine::plan_utils::resolveDiagonalBand;
    using PlanMaskType = asm_sdpa_engine::plan_utils::MaskType;

    PlanMaskType planMask{};
    try
    {
        const auto band = resolveDiagonalBand(attributes);
        planMask = getMaskType(band.leftBound, band.rightBound, band.alignment);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"} << e.what());
        return false;
    }

    switch(planMask)
    {
    case PlanMaskType::NO_MASK:
    case PlanMaskType::BOTTOM_RIGHT_CAUSAL:
        return true;
    case PlanMaskType::TOP_LEFT_CAUSAL:
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"}
                               << "top-left causal is not supported; strided decode uses "
                                  "bottom-right causal alignment");
        return false;
    case PlanMaskType::SLIDING_WINDOW:
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"}
                               << "left_bound " << attributes.left_bound().value_or(-1)
                               << ", right_bound " << attributes.right_bound().value_or(-1)
                               << " is a window; windows are not implemented in v1 of this "
                                  "engine (it serves unmasked or bottom-right causal decode)");
        return false;
    default:
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"} << "mask type " << static_cast<int>(planMask)
                                                << " is not supported");
        return false;
    }
}

/// The GQA ratios the 16-row query tile divides (`strided_decode.py`).
bool isSupportedGqaRatio(int64_t ratio)
{
    return ratio == 1 || ratio == 2 || ratio == 4 || ratio == 8 || ratio == 16;
}

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * rocKE's admission (`strided_decode.py::make_candidate`) for the request a dense hipDNN
 * graph maps to (`kv_layout="strided"`, arch gfx942):
 *
 *  - `seqlen_q` 1 (capability), head size 64, 128 or 256 (capability), fp16/bf16;
 *  - `nhead_q / nhead_k` in {1, 2, 4, 8, 16};
 *  - not top-left causal;
 *  - `_strided_3d_specs_from_problem` builds (no LDS overflow on gfx942 for any shipped
 *    head size and block size; TestGfx942AttentionUnified3dAdmission pins this);
 *  - the binding's tensor contract: Q/O contiguous `[B, Hq, D]`, K/V positive
 *    non-overlapping strided views with unit D stride (stridedKvRefusal).
 *
 * Refused in v1 although rocKE admits or could serve them: any sliding window or band, the
 * stats/LSE outputs, an attention-mask tensor, dropout, sinks, page tables, seq_len tensors
 * (ragged), FP8 scales, ALiBi, padding masks, block masks and a device scale tensor.
 *
 * Batch and Skv are not limited here: the kernel takes both at run time (grid and
 * seq_lens). kernelMatches declines each candidate whose num_segments would take an
 * i32-indexed tensor to 2^31 elements at the graph's batch.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog.
 */
std::optional<BoundTokens> gfx942AttentionUnified3dGraphMatches(const MatchContext& context)
{
    // --- 1. Node shape.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        if(HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED() && hasSdpaForwardNode(context))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"node"}
                                   << "the graph is not a single SDPA-forward node ("
                                   << context.graph.nodeCount() << " node(s))");
        }
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // --- 2. Operands.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"operand"}
                               << "the SDPA node names a Q, K, V or O uid the graph has no "
                                  "tensor for");
        return std::nullopt;
    }
    for(const auto& [role, tensor] :
        {std::pair{"Q", q}, std::pair{"K", k}, std::pair{"V", v}, std::pair{"O", o}})
    {
        if(!isWellFormedOperand(*tensor))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"operand"}
                                   << describeOperand(role, *tensor)
                                   << " must be a rank-4 (B, H, S, D) tensor with positive "
                                      "extents, not virtual and not pass-by-value");
            return std::nullopt;
        }
    }

    // --- 3. Cross-tensor consistency.
    const auto problem = problemFor(*q, *k);
    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"data_type"}
                               << "Q, K, V and O must share one data type, got " << problem.dataType
                               << ", " << k->data_type() << ", " << v->data_type() << ", "
                               << o->data_type());
        return std::nullopt;
    }
    if(!supportedDataTypeName(problem.dataType).has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"data_type"}
                               << "data type " << problem.dataType
                               << " is not supported; the kernel is built for BFLOAT16 and HALF");
        return std::nullopt;
    }
    // V shares K's logical shape (the kernel reads one KV length and head count for both).
    if(k->dims()->Get(BATCH_AXIS) != problem.batch
       || k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize
       || v->dims()->Get(BATCH_AXIS) != problem.batch
       || v->dims()->Get(HEAD_AXIS) != problem.numKvHeads
       || v->dims()->Get(SEQ_AXIS) != problem.seqLenKv
       || v->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"shape"} << describeOperand("K", *k) << " and "
                                                 << describeOperand("V", *v)
                                                 << " must share Q's batch and head size and "
                                                    "each other's heads and sequence; "
                                                 << describeOperand("Q", *q));
        return std::nullopt;
    }
    if(o->dims()->Get(BATCH_AXIS) != problem.batch
       || o->dims()->Get(HEAD_AXIS) != problem.numQueryHeads
       || o->dims()->Get(SEQ_AXIS) != problem.seqLenQ
       || o->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"shape"} << describeOperand("O", *o)
                                                 << " must have Q's dims; "
                                                 << describeOperand("Q", *q));
        return std::nullopt;
    }

    // --- 4. rocKE capability: seqlen_q 1, head size 64/128/256, GQA ratio.
    if(problem.seqLenQ != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"seqlen_q"}
                               << "seqlen_q " << problem.seqLenQ
                               << " is not supported; split-KV strided decode serves one query "
                                  "token per sequence");
        return std::nullopt;
    }
    if(problem.headSize != 64 && problem.headSize != 128 && problem.headSize != 256)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"head_size"}
                               << "head size " << problem.headSize
                               << " is not supported; the kernel is built for 64, 128 and 256");
        return std::nullopt;
    }
    if(problem.numQueryHeads % problem.numKvHeads != 0
       || !isSupportedGqaRatio(problem.numQueryHeads / problem.numKvHeads))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"gqa"}
                               << "query heads " << problem.numQueryHeads
                               << " over key/value heads " << problem.numKvHeads
                               << " is not a GQA ratio in {1, 2, 4, 8, 16}; the ratio must "
                                  "divide the 16-row query tile");
        return std::nullopt;
    }

    // --- 5. Tensor contract of the strided binding.
    for(const auto& [role, tensor] : {std::pair{"Q", q}, std::pair{"O", o}})
    {
        if(!hasDecodeQoStrides(*tensor))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"layout"}
                                   << describeOperand(role, *tensor)
                                   << " must be contiguous [B, Hq, 1, D]: batch stride Hq*D, "
                                      "head stride D, unit D stride (unit-extent axes exempt)");
            return std::nullopt;
        }
    }
    for(const auto& [role, tensor] : {std::pair{"K", k}, std::pair{"V", v}})
    {
        if(const auto why = stridedKvRefusal(*tensor); why.has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"layout"} << describeOperand(role, *tensor)
                                                      << " is not a strided K/V "
                                                      << "view the kernel can address: " << *why);
            return std::nullopt;
        }
    }

    // --- 6. The mask (maskAccepted logs its own declines).
    if(!maskAccepted(attributes))
    {
        return std::nullopt;
    }

    // --- 7. Every optional attribute this engine does not honour, declined explicitly.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"attn_mask"}
                               << "an additive attention mask (attn_mask tensor) is not "
                                  "implemented in v1");
        return std::nullopt;
    }
    if(attributes.scale_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"scale_tensor"}
                               << "a scale tensor is not implemented in v1; the kernel takes the "
                                  "scale as a host value (attn_scale_value)");
        return std::nullopt;
    }
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"varlen"}
                               << "variable sequence lengths (seq_len_q/seq_len_kv tensors) are "
                                  "not implemented in v1");
        return std::nullopt;
    }
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"dropout"} << "dropout is not implemented in v1");
        return std::nullopt;
    }
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"paged_kv"}
                               << "paged K/V (page tables or max_seq_len_kv) is not implemented "
                                  "in v1");
        return std::nullopt;
    }
    if(attributes.block_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"block_mask"} << "a block mask is not implemented in v1");
        return std::nullopt;
    }
    if(attributes.sink_token_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"sinks"}
                               << "attention sinks (sink_token tensor) are not implemented in v1");
        return std::nullopt;
    }
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"fp8"}
                               << "FP8 scaling tensors (descale, scale or amax) are not "
                                  "implemented in v1");
        return std::nullopt;
    }
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"stats"}
                               << "softmax statistics outputs (stats, max, sum_exp, rng_dump or "
                                  "generate_stats) are not implemented in v1; the strided decode "
                                  "path writes no LSE");
        return std::nullopt;
    }
    if(attributes.alibi_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"alibi_mask"} << "alibi_mask is not implemented in v1");
        return std::nullopt;
    }
    if(attributes.padding_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"padding_mask"} << "padding_mask is not implemented in v1");
        return std::nullopt;
    }
    // Same allow-list as the gfx950 dense pack: the MFMA operands are the graph's own
    // fp16/bf16 inputs.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mma_core_mode"}
                               << "mma_core_mode " << mmaCoreMode
                               << " is not supported; the kernel runs HALF or BFLOAT16 MFMA");
        return std::nullopt;
    }
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"implementation"} << "implementation hint "
                                                          << attributes.implementation()
                                                          << " is not supported; only AUTO is");
        return std::nullopt;
    }

    // The softmax scale is an f32 launch argument. An absent attn_scale_value means 1.0
    // (cuDNN's default, as in the gfx950 dense pack), NOT rocKE's 1/sqrt(D) default for a
    // None scale. rocKE bounds no scale; a non-finite one is declined.
    const float scale = attributes.attn_scale_value().value_or(1.0F);
    if(!std::isfinite(scale))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"scale"} << "attn_scale " << scale << " is not finite");
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);
    return bound;
}

/// The tensor uids and the scale a match bound.
struct Unified3dBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    float scale = 0.0F;
};

Unified3dBinding unified3dBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "gfx942 unified 3d dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };
    Unified3dBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    const auto scaleBits = static_cast<int32_t>(read(SCALE_BITS_TOKEN));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    return binding;
}

/// A candidate's integer metadata field, or nullopt. Total: never throws.
std::optional<int64_t> integerMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    return value == nullptr ? std::nullopt : std::optional<int64_t>(*value);
}

/// A candidate's string metadata field, or nullopt. Total: never throws.
std::optional<std::string> stringMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&it->second);
    return value == nullptr ? std::nullopt : std::optional<std::string>(*value);
}

/// What a candidate was built with, read from its KMD fields.
struct Unified3dTraits
{
    std::string dtype;
    int64_t headSize = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t slidingWindow = 0;
    int64_t blockSize = 0;
    int64_t tileSize = 0;
    int64_t numSegments = 0;
};

/**
 * @brief The candidate's traits, or nullopt when its record is not one this engine can
 *        launch.
 *
 * Every KMD field must be present. block_size is one of the candidate's
 * `kv_block_size` values {16, 32, 64}; num_segments is positive (the grid's z extent and the
 * workspace's segment axis); tile_size_override is the tile rocKE resolves from
 * (head_size, block_size) on gfx942 (gfx942ResolvedTileSize), so a record whose recorded
 * tile disagrees -- a binary built for another tile than its fields say -- is declined.
 */
std::optional<Unified3dTraits> candidateTraits(const KernelDefinition& kernel)
{
    Unified3dTraits traits;
    const auto dtype = stringMetadata(kernel, DTYPE_FIELD);
    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto numQueryHeads = integerMetadata(kernel, NUM_QUERY_HEADS_FIELD);
    const auto numKvHeads = integerMetadata(kernel, NUM_KV_HEADS_FIELD);
    const auto slidingWindow = integerMetadata(kernel, SLIDING_WINDOW_FIELD);
    const auto blockSize = integerMetadata(kernel, BLOCK_SIZE_FIELD);
    const auto tileSize = integerMetadata(kernel, TILE_SIZE_OVERRIDE_FIELD);
    const auto numSegments = integerMetadata(kernel, NUM_SEGMENTS_FIELD);
    if(!dtype || !headSize || !numQueryHeads || !numKvHeads || !slidingWindow || !blockSize
       || !tileSize || !numSegments)
    {
        return std::nullopt;
    }
    if(*headSize <= 0 || *numQueryHeads <= 0 || *numKvHeads <= 0 || *numSegments <= 0
       || (*blockSize != 16 && *blockSize != 32 && *blockSize != 64)
       || *tileSize != gfx942ResolvedTileSize(*headSize, *blockSize))
    {
        return std::nullopt;
    }
    traits.dtype = *dtype;
    traits.headSize = *headSize;
    traits.numQueryHeads = *numQueryHeads;
    traits.numKvHeads = *numKvHeads;
    traits.slidingWindow = *slidingWindow;
    traits.blockSize = *blockSize;
    traits.tileSize = *tileSize;
    traits.numSegments = *numSegments;
    return traits;
}

/**
 * @brief Kernel-scoped applicability: is THIS candidate built for the graph's cell, and
 *        can it index the graph's tensors?
 *
 * Compares the feature cell exactly -- dtype, head_size, num_query_heads, num_kv_heads --
 * and requires sliding_window 0 (graph_match refuses every window). Never batch or Skv as
 * cell fields: both are runtime (grid x and seq_lens), and the spec's baked `num_seqs` is
 * read only by the binary search the decode grid does not emit (`attention_tiled_3d.py`,
 * `use_decode_grid`). block_size and num_segments are not taken from the graph: every
 * value serves any graph (a KV tile of any size covers any Skv; the reduce combines every
 * segment, and segments with no keys contribute nothing), so they are the scorer's to
 * rank.
 *
 * One limit does depend on batch: the kernels index segm_output, segm_max/expsum, Q and O
 * with i32 element offsets, so a candidate whose num_segments takes any of them to 2^31
 * elements at the graph's batch is declined (gfx942Unified3dIndicesFitInt32). A candidate
 * with fewer segments may still match the same graph.
 */
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto problem = matchedProblem(context);
    if(!problem.has_value())
    {
        return false;
    }
    const auto traits = candidateTraits(kernel);
    if(!traits.has_value())
    {
        return false;
    }
    const auto dataTypeName = supportedDataTypeName(problem->dataType);
    if(!dataTypeName.has_value() || traits->dtype != *dataTypeName
       || traits->headSize != problem->headSize || traits->numQueryHeads != problem->numQueryHeads
       || traits->numKvHeads != problem->numKvHeads || traits->slidingWindow != 0)
    {
        return false;
    }
    if(!gfx942Unified3dIndicesFitInt32(
           problem->batch, problem->numQueryHeads, traits->numSegments, problem->headSize))
    {
        HIPDNN_PLUGIN_LOG_INFO(GFX942_ATTENTION_UNIFIED_3D_ENGINE_NAME
                               << " declined "
                               << describeDescriptor("kernel", kernel.name, kernel.kernelId)
                               << " [i32_index]: batch " << problem->batch << " x query heads "
                               << problem->numQueryHeads << " x num_segments "
                               << traits->numSegments << " x head size " << problem->headSize
                               << " reaches 2^31 elements; the kernels index segm_output, "
                                  "segm_max/expsum, Q and O with signed 32-bit offsets");
        return false;
    }
    return true;
}

/// Scores. A block-16 candidate scores in (BLOCK_16_FLOOR, BLOCK_16_FLOOR + KEY_SPAN]; every
/// other candidate in (0, KEY_SPAN], strictly below every block-16 one.
constexpr double KEY_SPAN = 1048576.0; // 2^20, above every key below
constexpr double BLOCK_16_FLOOR = 2.0 * KEY_SPAN;
/// Only a candidate kernelMatches already declined gets this.
constexpr double UNSUPPORTED_SCORE = 0.0;

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins. Never removes one.
 *
 * A stable cold order, not a performance model:
 *
 *  1. block_size 16 (rocKE's default `kv_block_size`) above every other block size;
 *  2. then num_segments nearest the count rocKE's split rule would pick for this graph at
 *     block 16 (gfx942SplitRuleSegments, with the live CU count, or rocKE's off-box 120
 *     when the device reports none);
 *  3. ties to the smaller num_segments, then the smaller block_size.
 *
 * Every (block_size, num_segments) pair gets a distinct score. rocKE's split rule is often
 * not the fastest choice; a tuned selection, not this scorer, is the place for that.
 */
double scoreKernel(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto traits = candidateTraits(kernel);
    const auto problem = matchedProblem(context);
    if(!traits.has_value() || !problem.has_value() || problem->numKvHeads <= 0
       || problem->numQueryHeads % problem->numKvHeads != 0)
    {
        return UNSUPPORTED_SCORE;
    }
    const int64_t numCus = context.deviceProperties.multiProcessorCount > 0
                               ? context.deviceProperties.multiProcessorCount
                               : FALLBACK_NUM_CUS;
    const int64_t target = gfx942SplitRuleSegments(problem->batch,
                                                   problem->numQueryHeads,
                                                   problem->numKvHeads,
                                                   problem->headSize,
                                                   PREFERRED_BLOCK_SIZE,
                                                   problem->seqLenKv,
                                                   numCus);
    // Clamp so the key stays inside KEY_SPAN for any recorded num_segments.
    constexpr int64_t SEGMENT_CAP = 255;
    const int64_t segments = std::min(traits->numSegments, SEGMENT_CAP);
    const int64_t distance = std::min<int64_t>(std::abs(segments - target), SEGMENT_CAP);
    int64_t blockRank = 2;
    if(traits->blockSize == 32)
    {
        blockRank = 0;
    }
    else if(traits->blockSize == 64)
    {
        blockRank = 1;
    }
    // Lexicographic (distance, segments, blockRank); smaller is better.
    const int64_t key = (distance * 256 + segments) * 4 + blockRank;
    const double base = traits->blockSize == PREFERRED_BLOCK_SIZE ? BLOCK_16_FLOOR : 0.0;
    return base + (KEY_SPAN - static_cast<double>(key));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

constexpr const char* BUFFER_KIND = "global_buffer";
constexpr const char* BY_VALUE_KIND = "by_value";
constexpr uint32_t PTR_BYTES = 8;

/**
 * @brief The segment launch's argument list: `_3d_signature(dtype, strided_kv=True)`
 *        (`kernels/common/attention_unified.py`), 19 base arguments then 8 strided-K/V.
 *
 * Derived from gfx942Unified3dSegmentArguments(), the list launch() passes, so the
 * recorded-signature check in prepare() also checks the launch's names, kinds and order.
 *
 * NAMES ARE LOAD-BEARING: requireSignatureMatch compares kind and size always and names
 * when both sides carry one; hkp_pack records the builder's own parameter names. Offsets
 * mirror the packed layout for the mismatch diagnostic only.
 */
std::vector<KernelArgument> segmentSignature()
{
    std::vector<KernelArgument> signature;
    uint32_t offset = 0;
    for(const auto& argument : gfx942Unified3dSegmentArguments(0, 0.0F, {}, {}))
    {
        const uint32_t size = argument.bytes();
        offset = (offset + size - 1) / size * size;
        signature.push_back(KernelArgument{
            argument.kind == Gfx942Unified3dArgumentKind::POINTER ? BUFFER_KIND : BY_VALUE_KIND,
            size,
            offset,
            argument.name});
        offset += size;
    }
    return signature;
}

/// The reduce launch's argument list: `_reduce_signature(dtype)` with no LSE (strided
/// decode writes none).
std::vector<KernelArgument> reduceSignature()
{
    return {KernelArgument{BUFFER_KIND, PTR_BYTES, 0, "output_ptr"},
            KernelArgument{BUFFER_KIND, PTR_BYTES, 8, "segm_output_ptr"},
            KernelArgument{BUFFER_KIND, PTR_BYTES, 16, "segm_max_ptr"},
            KernelArgument{BUFFER_KIND, PTR_BYTES, 24, "segm_expsum_ptr"},
            KernelArgument{BUFFER_KIND, PTR_BYTES, 32, "seq_lens_ptr"}};
}

/// gfx942StridedKvArguments for a tensor graph_match validated with stridedKvRefusal.
Gfx942StridedKvArguments stridedKvArguments(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    const auto axes = {BATCH_AXIS, HEAD_AXIS, SEQ_AXIS, HEAD_SIZE_AXIS};
    std::array<int64_t, 4> logicalDims{};
    std::array<int64_t, 4> logicalStrides{};
    size_t i = 0;
    for(const uint32_t axis : axes)
    {
        logicalDims[i] = dims->Get(axis);
        logicalStrides[i] = strides->Get(axis);
        ++i;
    }
    return gfx942StridedKvArguments(logicalDims, logicalStrides);
}

void* allocatePinnedHost(size_t bytes)
{
    void* host = nullptr;
    return hipHostMalloc(&host, bytes) == hipSuccess ? host : nullptr;
}

/**
 * @brief The process-lifetime pinned `[0, 1, 2, ...]` every plan copies its
 *        `cu_seqlens_q[B+1]` prefix from (see Gfx942GrowOnlyArange for source, lifetime and
 *        thread safety).
 *
 * Shared by all plans and never freed: an in-flight hipMemcpyAsync from any block stays
 * valid with no destructor synchronization. Pinned (hipHostMalloc), so the copy is truly
 * asynchronous; a pageable source would make it wait for the stream on the host.
 */
Gfx942GrowOnlyArange& pinnedQueryOffsets()
{
    // Intentionally leaked: a static destructor would run after HIP may have shut down.
    static auto* s_offsets = new Gfx942GrowOnlyArange(&allocatePinnedHost);
    return *s_offsets;
}

/// Everything launch() needs, owning nothing that points back into the MatchContext or
/// BoundTokens it came from.
struct Unified3dLaunchState
{
    Unified3dBinding binding;
    Unified3dProblem problem;
    Gfx942AttentionUnified3dWorkspace workspace;
    /// The segment launch's arguments; pointer slots resolved per launch.
    Gfx942Unified3dSegmentArguments segmentArguments;
};

class PreparedGfx942AttentionUnified3d : public PreparedDispatch
{
public:
    PreparedGfx942AttentionUnified3d(IngestorKernelCode segment,
                                     IngestorKernelCode reduce,
                                     Unified3dLaunchState state)
        : _segment(std::move(segment))
        , _reduce(std::move(reduce))
        , _state(state)
    {
    }

    compilation::IRunnableKernel& segmentForStream(hipStream_t stream) const
    {
        return _segment.kernelForStream(stream);
    }

    compilation::IRunnableKernel& reduceForStream(hipStream_t stream) const
    {
        return _reduce.kernelForStream(stream);
    }

    const Unified3dLaunchState& state() const
    {
        return _state;
    }

private:
    IngestorKernelCode _segment;
    IngestorKernelCode _reduce;
    Unified3dLaunchState _state;
};

/// Launches @p kernel with one argument per word (see gfx942Unified3dArgumentWords).
template <size_t... I>
void launchWords(const compilation::IRunnableKernel& kernel,
                 hipStream_t stream,
                 const std::array<uint64_t, sizeof...(I)>& words,
                 std::index_sequence<I...> /*indices*/)
{
    kernel.launch(stream, words[I]...);
}

void requireAligned(const void* pointer, const char* role)
{
    if(reinterpret_cast<uintptr_t>(pointer) % VECTOR_ALIGNMENT_BYTES != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string("gfx942 unified 3d: the ") + role
                + " device buffer must be 16-byte aligned (vector loads)");
    }
}

void requireHip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       std::string("gfx942 unified 3d: ") + what
                                                           + ": " + hipGetErrorString(status));
    }
}

/**
 * @brief The native dispatch behind this engine's UDD: the segment launch, then the reduce
 *        launch, of one packaged module.
 */
class Gfx942AttentionUnified3dDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit Gfx942AttentionUnified3dDispatchHandler(
        const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    /// The launch names every kernel of this pack carries in its `sources` map.
    std::set<std::string> launchNames() const override
    {
        return {"reduce", "segment"};
    }

    /// The f32 split-KV scratch plus seqused_k and cu_seqlens_q
    /// (gfx942AttentionUnified3dWorkspace), sized by the CANDIDATE's num_segments.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto problem = matchedProblem(context);
        const auto traits = candidateTraits(kernel);
        if(!problem.has_value() || !traits.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx942 unified 3d: workspace asked for a graph or kernel the engine does not "
                "serve");
        }
        return static_cast<size_t>(
            gfx942AttentionUnified3dWorkspace(
                problem->batch, problem->numQueryHeads, traits->numSegments, problem->headSize)
                .totalBytes);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = unified3dBinding(bound);
        // kernel_match is the gate for both; these re-checks are defence in depth for a
        // caller that reaches the handler without having matched.
        const auto traits = candidateTraits(kernel);
        if(!traits.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx942 unified 3d: kernel '" + toString(kernel.kernelId)
                    + "' does not declare a launchable block_size/tile/num_segments");
        }
        const auto* q = findTensor(context, binding.q);
        const auto* k = findTensor(context, binding.k);
        const auto* v = findTensor(context, binding.v);
        const auto* o = findTensor(context, binding.o);
        if(q == nullptr || k == nullptr || v == nullptr || o == nullptr || !isWellFormedOperand(*q)
           || !isWellFormedOperand(*k) || !isWellFormedOperand(*v) || !isWellFormedOperand(*o)
           || stridedKvRefusal(*k).has_value() || stridedKvRefusal(*v).has_value()
           || !hasDecodeQoStrides(*q) || !hasDecodeQoStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx942 unified 3d: the bound Q/K/V/O tensors are missing or not the layout "
                "the kernels address");
        }
        const auto problem = problemFor(*q, *k);
        if(!gfx942Unified3dIndicesFitInt32(
               problem.batch, problem.numQueryHeads, traits->numSegments, problem.headSize))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx942 unified 3d: kernel '" + toString(kernel.kernelId)
                    + "' would index segm_output, segm_max/expsum, Q or O past 2^31 elements "
                      "at batch "
                    + std::to_string(problem.batch) + " with " + std::to_string(traits->numSegments)
                    + " segments (i32 offsets)");
        }

        // Both launches' recorded argument lists are compared before either is loaded:
        // buildIngestorKernelCode compares each one only on its own way to the load, so a
        // wrong reduce list would otherwise surface only after the segment's module loaded.
        const auto segmentDefinition = launchKernelDefinition(kernel, "segment");
        const auto reduceDefinition = launchKernelDefinition(kernel, "reduce");
        const std::string label = describeDescriptor("kernel", kernel.name, kernel.kernelId);
        requireSignatureMatch(segmentDefinition.source.signature,
                              segmentSignature(),
                              segmentDefinition.source.symbol,
                              label);
        requireSignatureMatch(reduceDefinition.source.signature,
                              reduceSignature(),
                              reduceDefinition.source.symbol,
                              label);

        auto segment
            = buildIngestorKernelCode(_kpackLoader, context, segmentDefinition, segmentSignature());
        auto reduce
            = buildIngestorKernelCode(_kpackLoader, context, reduceDefinition, reduceSignature());

        Unified3dLaunchState state;
        state.binding = binding;
        state.problem = problem;
        state.segmentArguments
            = gfx942Unified3dSegmentArguments(static_cast<int32_t>(state.problem.batch),
                                              binding.scale,
                                              stridedKvArguments(*k),
                                              stridedKvArguments(*v));
        // Grow the shared pinned cu_seqlens_q source now, so launch() never allocates.
        static_cast<void>(
            pinnedQueryOffsets().acquire(static_cast<size_t>(state.problem.batch) + 1));
        state.workspace = gfx942AttentionUnified3dWorkspace(state.problem.batch,
                                                            state.problem.numQueryHeads,
                                                            traits->numSegments,
                                                            state.problem.headSize);

        // Grid from the GRAPH (batch, heads) and the CANDIDATE's num_segments
        // (_get_3d_pipeline, decode grid).
        const auto geometry = gfx942AttentionUnified3dGeometry(state.problem.batch,
                                                               state.problem.numQueryHeads,
                                                               state.problem.numKvHeads,
                                                               traits->numSegments);
        segment.setBlockSize(geometry.segment.blockX, 1, 1);
        segment.setGridSize(geometry.segment.gridX, geometry.segment.gridY, geometry.segment.gridZ);
        reduce.setBlockSize(geometry.reduce.blockX, 1, 1);
        reduce.setGridSize(geometry.reduce.gridX, geometry.reduce.gridY, geometry.reduce.gridZ);

        return std::make_unique<PreparedGfx942AttentionUnified3d>(
            std::move(segment), std::move(reduce), state);
    }

    /**
     * Four operations, in order, all on the handle's stream and none waiting on the host:
     *
     *  1. seqused_k[B] = Skv (hipMemsetD32Async). Source: the strided entry's int32 `[B]`
     *     KV lengths; a dense graph's every sequence has all Skv keys. Lifetime: one launch;
     *     region `sequsedK` of the workspace. Precondition: none (overwritten every launch).
     *  2. cu_seqlens_q[B+1] = [0..B] (hipMemcpyAsync of a prefix of the process-lifetime
     *     pinned pinnedQueryOffsets(), never freed). Same lifetime; region `cuSeqlensQ`.
     *  3. The segment launch. It writes every (row, head, segment) slot of segm_output /
     *     segm_max / segm_expsum, an empty segment included (`-inf` max, 0 sum), so the f32
     *     scratch needs no initialization. Lifetime: one launch; regions `segmOutput`,
     *     `segmMax`, `segmExpsum`.
     *  4. The reduce launch, ALWAYS: O is written only here. hipDNN does not zero the
     *     workspace, and the reduce is what reads it back.
     *
     * The segment arguments are gfx942Unified3dSegmentArguments() (built at prepare), with
     * pointer slots resolved here by gfx942Unified3dArgumentWords().
     */
    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedGfx942AttentionUnified3d&>(prepared);
        const auto& state = dispatch.state();
        const auto& binding = state.binding;
        const auto& problem = state.problem;
        const auto& ws = state.workspace;
        const auto stream = handle.getStream();

        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx942 unified 3d: launched without the workspace workspaceBytes() asked "
                "for");
        }

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);
        requireAligned(q.ptr, "Q");
        requireAligned(k.ptr, "K");
        requireAligned(v.ptr, "V");
        requireAligned(o.ptr, "O");

        auto* base = static_cast<std::byte*>(workspace);
        void* segmOutput = base + ws.segmOutput;
        void* segmMax = base + ws.segmMax;
        void* segmExpsum = base + ws.segmExpsum;
        void* sequsedK = base + ws.sequsedK;
        void* cuSeqlensQ = base + ws.cuSeqlensQ;

        requireHip(hipMemsetD32Async(static_cast<hipDeviceptr_t>(sequsedK),
                                     static_cast<int>(problem.seqLenKv),
                                     static_cast<size_t>(problem.batch),
                                     stream),
                   "filling seqused_k");
        const size_t offsetCount = static_cast<size_t>(problem.batch) + 1;
        const int32_t* offsets = pinnedQueryOffsets().acquire(offsetCount);
        requireHip(
            hipMemcpyAsync(
                cuSeqlensQ, offsets, offsetCount * sizeof(int32_t), hipMemcpyHostToDevice, stream),
            "copying cu_seqlens_q");

        Gfx942Unified3dBufferAddresses addresses;
        addresses.segmOutput = segmOutput;
        addresses.segmMax = segmMax;
        addresses.segmExpsum = segmExpsum;
        addresses.query = q.ptr;
        addresses.key = k.ptr;
        addresses.value = v.ptr;
        addresses.sequsedK = sequsedK;
        addresses.cuSeqlensQ = cuSeqlensQ;
        const auto words = gfx942Unified3dArgumentWords(state.segmentArguments, addresses);
        launchWords(dispatch.segmentForStream(stream),
                    stream,
                    words,
                    std::make_index_sequence<GFX942_UNIFIED_3D_SEGMENT_ARGUMENT_COUNT>{});
        // Changing this list means changing reduceSignature() with it.
        dispatch.reduceForStream(stream).launch(
            stream, o.ptr, segmOutput, segmMax, segmExpsum, sequsedK);
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& gfx942AttentionUnified3dKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx942AttentionUnified3dModuleCache()
{
    gfx942AttentionUnified3dKpackModuleCache().clear();
}

namespace
{

/// Process-lifetime: the registry holds a non-owning pointer.
const Gfx942AttentionUnified3dDispatchHandler& gfx942AttentionUnified3dDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx942AttentionUnified3dKpackModuleCache());
    static const Gfx942AttentionUnified3dDispatchHandler s_dispatchHandler(s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerGfx942AttentionUnified3dSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx942AttentionUnified3dGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &gfx942AttentionUnified3dDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
