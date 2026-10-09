// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "compilation/Utils.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"

/**
 * @file Gfx950AttentionUnified2DNative.cpp
 * @brief The hipkernel:Gfx950AttentionUnified2D engine's native half: matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * THIS IS NOT THE DENSE ENGINE ON A WIDER ABI. Three differences drive everything here,
 * and each was measured against the rocKE registry rather than inferred from the dense
 * pack. The profile carries the long form
 * (projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_unified_2d.profile.yaml);
 * in short:
 *
 *   1. The kernel's ABI is the paged/varlen one, 18 arguments. There is no non-paged
 *      arm. A dense BSHD SDPA graph reaches it because this handler synthesizes an
 *      identity block_tables / seq_lens / query_start_len workspace -- the first nonzero
 *      workspaceBytes() in this provider.
 *
 *   2. The kernel is UNCONDITIONALLY CAUSAL, bottom-right aligned. `causal` is neither a
 *      spec field nor a kernel argument; the mask
 *      `col_abs <= (seq_len - cur_batch_q_len) + q_pos` is emitted on every path
 *      (rocke/library/kernels/gfx950/attention_tiled_2d.py:3452). So graph_match
 *      REQUIRES a causal mask and declines an unmasked graph. That is the mirror image
 *      of the dense engine, which is top-left causal and accepts `causal=0` as a
 *      catalog value.
 *
 *   3. `num_seqs` is a baked binary-search trip count, not a shape. Too FEW iterations
 *      silently resolve the wrong sequence. kernelMatches enforces the capacity bound;
 *      see gfx950AttentionUnified2DMaxBatch in the geometry header.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
// The `unified2_d` spelling is the generator's own snake_case of the engine's local name
// `Gfx950AttentionUnified2D` (a digit followed by a capital splits); descriptors and this
// file derive it the same way, which is the only property that matters. Renaming one
// without the other drops the whole set at load time, silently.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified2_d.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified2_d.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_unified2_d.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_unified2_d.dispatch";

// KMD fields this engine's matchers and geometry read.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view USE_SINKS_FIELD = "use_sinks";
constexpr std::string_view NUM_SEQS_FIELD = "num_seqs";
constexpr std::string_view TILE_SIZE_FIELD = "tile_size";
constexpr std::string_view NUM_WARPS_FIELD = "num_warps";
constexpr std::string_view BLOCK_M_PER_WARP_FIELD = "block_m_per_warp";

// Tokens bound by graph_match and read back by prepare(). Operand UIDs plus the two
// derived scalars; everything else is re-read from the graph, which prepare() still has.
constexpr std::string_view Q_TOKEN = "gfx950_attention_unified_2d.q.uid";
constexpr std::string_view K_TOKEN = "gfx950_attention_unified_2d.k.uid";
constexpr std::string_view V_TOKEN = "gfx950_attention_unified_2d.v.uid";
constexpr std::string_view O_TOKEN = "gfx950_attention_unified_2d.o.uid";
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx950_attention_unified_2d.sliding_window";
constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_unified_2d.scale_bits";

// BSHD operand axes.
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;

/// hipDNN's "no bound" spelling for left_bound / right_bound.
constexpr int64_t UNBOUNDED = -1;

// The neutral values for the kernargs a dense graph has no source for. The kernel's own
// guarded paths read these when the corresponding feature is not baked, so they are not
// arbitrary: a scale of 1.0 is the identity of its multiply, and a softcap of 0.0 is the
// value its `USE_SOFTCAP` path is compiled out for.
constexpr float NEUTRAL_SCALE = 1.0F;
constexpr float NEUTRAL_SOFTCAP = 0.0F;

// ---------------------------------------------------------------------------
// Matching helpers
//
// These mirror the dense pack's helpers of the same names. They are duplicated rather
// than shared because they encode THIS engine's applicability: native-pack.md is explicit
// that an engine must not reuse another engine's applicability rules, and the two
// engines already disagree about head sizes and about what a mask means.
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

    return !tensor.virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/**
 * @brief Is this tensor's memory BSHD -- token-major, head varying fastest?
 *
 * The kernel bakes this layout and takes no stride kernargs, so a differently-strided
 * tensor is read as if it were this one.
 *
 * For K and V this is doing double duty. Besides the layout the kernel bakes, the
 * IDENTITY BLOCK TABLE this handler synthesizes assumes page j of sequence b lives at
 * physical page `b * blocksPerSequence + j` -- which is only true when K/V is one
 * contiguous token-major run. A non-contiguous K or V would be addressed through a table
 * that does not describe it, with no fault and every output wrong.
 *
 * Unit-extent axes are exempt: a stride multiplies an index that is always 0 when the
 * extent is 1.
 */
bool hasBshdStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t sequence = dims->Get(SEQ_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    const auto axisOk = [&](uint32_t axis, std::optional<int64_t> expected) {
        return dims->Get(axis) == 1
               || (expected.has_value() && strides->Get(axis) == *expected);
    };

    return axisOk(BATCH_AXIS, checkedProduct({sequence, heads, headSize}))
           && axisOk(HEAD_AXIS, headSize)
           && axisOk(SEQ_AXIS, checkedProduct({heads, headSize}))
           && axisOk(HEAD_SIZE_AXIS, 1);
}

// ---------------------------------------------------------------------------
// Decline logging
//
// hipDNN reports a declined graph to its caller only as "No engine configurations
// available for the graph", so each decline logs its cause at INFO. graph_match runs on
// every catalog miss, so all of this work sits inside HIPDNN_PLUGIN_LOG_INFO's argument:
// with INFO off nothing below is called and no message is built.
// ---------------------------------------------------------------------------

/// The head of a decline line: "<engine> declined the graph [<cause>]: ".
struct Declined
{
    std::string_view cause;
};

std::ostream& operator<<(std::ostream& out, const Declined& declined)
{
    return out << GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME << " declined the graph ["
               << declined.cause << "]: ";
}

/// Whether any node of the graph is an SDPA-forward node. Only such a graph hears why it
/// was declined; a line for every convolution graph would bury the attention ones.
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

/// "Q (uid 1): dims [..], strides [..]". Null dims or strides print as [].
std::string describeOperand(std::string_view role,
                            const data_objects::TensorAttributes& tensor)
{
    using hipdnn_data_sdk::utilities::vecToString;
    using hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector;
    std::ostringstream out;
    out << role << " (uid " << tensor.uid() << "): dims "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.dims())) << ", strides "
        << vecToString(convertFlatBufferVectorToStdVector(tensor.strides()));
    return out.str();
}

/// Why @p tensor is not one the kernel can address, with the strides the kernel bakes.
std::string notBshdReason(std::string_view role,
                          const data_objects::TensorAttributes& tensor)
{
    std::ostringstream out;
    out << describeOperand(role, tensor)
        << " is not dense BSHD (token-major, head fastest)";

    const auto dims = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
        tensor.dims());
    if(dims.size() != SDPA_RANK)
    {
        return out.str();
    }
    const int64_t heads = dims[HEAD_AXIS];
    const int64_t sequence = dims[SEQ_AXIS];
    const int64_t headSize = dims[HEAD_SIZE_AXIS];
    const auto show = [](std::optional<int64_t> value) {
        return value.has_value() ? std::to_string(*value) : std::string("overflow");
    };
    out << "; the kernel bakes strides ["
        << show(checkedProduct({sequence, heads, headSize})) << ", " << headSize << ", "
        << show(checkedProduct({heads, headSize})) << ", 1]"
        << " (unit-extent axes exempt) and takes no stride arguments";
    return out.str();
}

/// The mask kinds this engine can reason about. NO_MASK is recognised so it can be
/// DECLINED with a specific reason rather than falling into the default arm.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2
};

/**
 * @brief Which mask the graph is asking for, or nullopt for one this engine lacks.
 *
 * A real bound wins over the deprecated booleans: a graph that sets a boolean AND carries
 * a bound is asking for a windowed mask.
 */
std::optional<MaskType> maskTypeFor(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();

    if(topLeftDeprecated && bottomRightDeprecated)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            Declined{"mask"} << "causal_mask and causal_mask_bottom_right are both set");
        return std::nullopt;
    }

    const int64_t left
        = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    const int64_t right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;

    // A non-zero right bound is a bidirectional window; the kernel masks strictly upward.
    if(right != UNBOUNDED && right != 0)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"}
                               << "right_bound " << right
                               << " (a bidirectional window) is not supported; the "
                                  "kernel is causal only");
        return std::nullopt;
    }

    // A bounded left edge is a sliding window whatever the booleans say. The kernel CAN
    // be built with one (`sliding_window` is a spec field), but no shipped variant bakes
    // a non-zero window, so serving one here would apply the wrong mask with no error.
    if(left != UNBOUNDED)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"}
                               << "left_bound " << left
                               << " (a sliding window) is not supported; no shipped "
                                  "variant bakes a window");
        return std::nullopt;
    }

    if(topLeftDeprecated)
    {
        return MaskType::TOP_LEFT_CAUSAL;
    }
    if(bottomRightDeprecated)
    {
        return MaskType::BOTTOM_RIGHT_CAUSAL;
    }

    if(right == UNBOUNDED)
    {
        return MaskType::NO_MASK;
    }
    return attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT
               ? MaskType::BOTTOM_RIGHT_CAUSAL
               : MaskType::TOP_LEFT_CAUSAL;
}

/// The kernel's dtype spelling for a graph dtype, or nullopt for one it cannot be built for.
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

/// The tensor uids and derived scalars a matched graph binds.
struct AttentionUnified2DBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t slidingWindow = 0;
    float scale = 0.0F;
};

/// The graph facts the matcher, the geometry and prepare() all need.
struct AttentionUnified2DProblem
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

/// The graph's shape, read from Q and K. Callers must have validated both operands.
AttentionUnified2DProblem problemFor(const data_objects::TensorAttributes& q,
                                     const data_objects::TensorAttributes& k)
{
    AttentionUnified2DProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numQueryHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headSize = q.dims()->Get(HEAD_SIZE_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

// ---------------------------------------------------------------------------
// graph_match
// ---------------------------------------------------------------------------

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog and skips every
 *          remaining pack.
 *
 * Everything here is candidate-independent. Checks that depend on which kernel is being
 * considered -- the KV page size dividing seqlen_kv, the binary-search capacity bound --
 * belong in kernelMatches, so that another candidate can still serve the graph.
 */
std::optional<BoundTokens>
    gfx950AttentionUnified2DGraphMatches(const MatchContext& context)
{
    // --- 1. One SDPA-forward node.
    const auto* node = sdpaNode(context);
    if(node == nullptr)
    {
        if(HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED() && hasSdpaForwardNode(context))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"node"}
                                   << "the graph is not a single SDPA-forward node");
        }
        return std::nullopt;
    }
    const auto& attributes = *node;

    // --- 2. Operands exist.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"operand"}
                               << "one of Q/K/V/O names no tensor in the graph");
        return std::nullopt;
    }

    // --- 3. Each operand is well formed, before anything indexes an axis.
    for(const auto& [role, tensor] :
        {std::pair{"Q", q}, std::pair{"K", k}, std::pair{"V", v}, std::pair{"O", o}})
    {
        if(!isWellFormedOperand(*tensor))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"operand"}
                                   << describeOperand(role, *tensor)
                                   << " is not a rank-4 non-virtual tensor with positive "
                                      "extents");
            return std::nullopt;
        }
    }

    // --- 4. Layout. K and V additionally carry the identity block table's assumption;
    // see hasBshdStrides.
    for(const auto& [role, tensor] : {std::pair{"Q", q}, std::pair{"K", k}, std::pair{"V", v}})
    {
        if(!hasBshdStrides(*tensor))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"layout"} << notBshdReason(role, *tensor));
            return std::nullopt;
        }
    }

    // --- 5. Cross-tensor consistency.
    if(k->data_type() != q->data_type() || v->data_type() != q->data_type()
       || o->data_type() != q->data_type())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"data_type"}
                               << "Q, K, V and O do not all share one data type");
        return std::nullopt;
    }
    const auto dataTypeName = supportedDataTypeName(q->data_type());
    if(!dataTypeName.has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"data_type"}
                               << "data type " << q->data_type()
                               << " is not supported; the kernel builds for BF16 or FP16");
        return std::nullopt;
    }

    const auto problem = problemFor(*q, *k);

    for(uint32_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        if(v->dims()->Get(axis) != k->dims()->Get(axis))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"shape"} << "V and K must have equal dims");
            return std::nullopt;
        }
        if(o->dims()->Get(axis) != q->dims()->Get(axis))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"shape"} << "O and Q must have equal dims");
            return std::nullopt;
        }
    }
    if(k->dims()->Get(BATCH_AXIS) != problem.batch
       || k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"shape"}
                               << "K must share Q's batch and head size");
        return std::nullopt;
    }
    if(!hasBshdStrides(*o))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"layout"} << notBshdReason("O", *o));
        return std::nullopt;
    }
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"gqa"}
                               << "num_query_heads " << problem.numQueryHeads
                               << " is not a whole multiple of num_kv_heads "
                               << problem.numKvHeads);
        return std::nullopt;
    }
    if(!isSupportedGfx950AttentionUnified2DHeadSize(problem.headSize))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"head_size"}
                               << "head size " << problem.headSize
                               << " is not supported; the unified 2D family builds for "
                                  "64, 128 and 256");
        return std::nullopt;
    }

    // --- 6. 32-bit addressing. K/V bound is bytes, Q/O is elements.
    constexpr int64_t INT32_LIMIT = 2147483648LL; // 2^31
    constexpr int64_t BYTES_PER_ELEMENT = 2;      // bf16 and fp16 only
    const auto kvBytes = checkedProduct({problem.batch,
                                         problem.seqLenKv,
                                         problem.numKvHeads,
                                         problem.headSize,
                                         BYTES_PER_ELEMENT});
    if(!kvBytes.has_value() || *kvBytes >= INT32_LIMIT)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"size_limit"}
                               << "K or V is 2^31 bytes or larger; the kernel addresses "
                                  "K/V with 32-bit byte offsets");
        return std::nullopt;
    }
    const auto qElements = checkedProduct(
        {problem.batch, problem.seqLenQ, problem.numQueryHeads, problem.headSize});
    if(!qElements.has_value() || *qElements >= INT32_LIMIT)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"size_limit"}
                               << "Q or O has 2^31 elements or more; the kernel "
                                  "addresses them with 32-bit element offsets");
        return std::nullopt;
    }

    // --- 7. The mask. THE KERNEL ALWAYS MASKS, bottom-right aligned:
    //
    //     context_len = seq_len - cur_batch_q_len        (= seqlen_kv - seqlen_q here)
    //     causal_ok   = key_pos <= context_len + q_pos
    //
    // (attention_tiled_2d.py:1262 and :3452, on every emitted path). There is no
    // unmasked arm and no mask kernarg, so:
    //
    //   BOTTOM_RIGHT_CAUSAL  is exactly this. Always accepted.
    //   TOP_LEFT_CAUSAL      wants key_pos <= q_pos, which coincides only when
    //                        context_len is zero, i.e. seqlen_q == seqlen_kv.
    //   NO_MASK              has no spelling here at all. Accepting it would return
    //                        causally-masked numerics for a graph that asked for none,
    //                        with nothing faulting -- so it is declined.
    //
    // This is the mirror image of the dense engine, whose clamp is top-left and which
    // therefore accepts bottom-right only under the same seqlen equality.
    const auto mask = maskTypeFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }

    switch(*mask)
    {
    case MaskType::BOTTOM_RIGHT_CAUSAL:
        break;
    case MaskType::TOP_LEFT_CAUSAL:
        if(problem.seqLenQ != problem.seqLenKv)
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"}
                                   << "top-left causal needs seqlen_q == seqlen_kv, got "
                                   << problem.seqLenQ << " and " << problem.seqLenKv
                                   << "; the kernel's causal mask is bottom-right");
            return std::nullopt;
        }
        break;
    case MaskType::NO_MASK:
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"}
                               << "an unmasked graph is not supported; this kernel masks "
                                  "causally on every path and has no unmasked arm");
        return std::nullopt;
    default:
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"} << "unrecognised mask kind");
        return std::nullopt;
    }

    // --- 8. Every optional attribute this kernel cannot honour, declined explicitly.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"attn_mask"}
                               << "an additive attention mask (attn_mask tensor) is not "
                                  "supported");
        return std::nullopt;
    }
    if(attributes.scale_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"scale_tensor"}
                               << "a scale tensor is not supported; the kernel takes the "
                                  "scale as a host value (attn_scale_value)");
        return std::nullopt;
    }
    // varlen. The ABI is the varlen one and this handler synthesizes a UNIFORM
    // query_start_len; honouring a real seq_len tensor would mean reading device memory
    // at prepare() time, which is a different feature, not a smaller change.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"varlen"}
                               << "variable sequence lengths (seq_len_q/seq_len_kv "
                                  "tensors) are not supported; this engine synthesizes a "
                                  "uniform query_start_len from the graph's own extents");
        return std::nullopt;
    }
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"dropout"} << "dropout is not supported");
        return std::nullopt;
    }
    // Paged K/V as a GRAPH feature. The kernel is paged internally, but through the
    // identity table this handler builds; a caller-supplied page table describes a
    // different memory layout than the one hasBshdStrides just proved.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"paged_kv"}
                               << "caller-supplied paged K/V is not supported; this "
                                  "engine synthesizes its own identity block table over "
                                  "contiguous K/V");
        return std::nullopt;
    }
    if(attributes.block_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"block_mask"} << "a block mask is not supported");
        return std::nullopt;
    }
    // Attention sinks. The kernel CAN bake them (`use_sinks`), and the ABI carries a live
    // sink_ptr, but no shipped variant bakes one and the dense graph has no sink tensor.
    if(attributes.sink_token_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"sinks"}
                               << "attention sinks (sink_token tensor) are not "
                                  "supported; no shipped variant bakes use_sinks");
        return std::nullopt;
    }
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value()
       || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value()
       || attributes.amax_o_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"fp8"}
                               << "FP8 scaling tensors (descale, scale or amax) are not "
                                  "supported");
        return std::nullopt;
    }
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"stats"}
                               << "softmax statistics outputs (stats, max, sum_exp, "
                                  "rng_dump or generate_stats) are not supported");
        return std::nullopt;
    }
    // ALiBi. The ABI carries alibi_slopes, but no shipped variant bakes use_alibi, so the
    // pointer is bound null and a graph asking for slopes must be declined.
    if(attributes.alibi_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"alibi_mask"}
                               << "alibi_mask is not supported; no shipped variant bakes "
                                  "use_alibi and the slopes pointer is bound null");
        return std::nullopt;
    }
    if(attributes.padding_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"padding_mask"} << "padding_mask is not supported");
        return std::nullopt;
    }
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET
       && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mma_core_mode"}
                               << "mma_core_mode " << mmaCoreMode
                               << " is not supported; the kernel runs HALF or BFLOAT16 "
                                  "MFMA");
        return std::nullopt;
    }
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"implementation"}
                               << "implementation hint " << attributes.implementation()
                               << " is not supported; only AUTO is");
        return std::nullopt;
    }

    const float scale = attributes.attn_scale_value().value_or(1.0F);

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    // Always zero: a windowed graph was declined above. Bound anyway so kernelMatches can
    // compare it against the candidate's baked sliding_window, which keeps a variant
    // built WITH a window from being matched by a graph that did not ask for one.
    bound[std::string(SLIDING_WINDOW_TOKEN)] = int64_t{0};
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);
    return bound;
}

// ---------------------------------------------------------------------------
// Candidate matching
// ---------------------------------------------------------------------------

/// A kernel's integer metadata field, or nullopt when it declares none.
std::optional<int64_t> integerMetadata(const KernelDefinition& kernel,
                                       std::string_view field)
{
    return kernel.getIntMetadata(std::string(field));
}

/// The geometry fields the grid and block are computed from.
struct AttentionUnified2DGeometryFields
{
    int64_t blockMPerWarp = 0;
    int64_t numWarps = 0;
    int64_t blockSize = 0;
};

/// @return The candidate's geometry, or nullopt when it declares an incomplete or
///         non-positive one. Checked before anything divides by a member.
std::optional<AttentionUnified2DGeometryFields>
    candidateGeometry(const KernelDefinition& kernel)
{
    const auto blockMPerWarp = integerMetadata(kernel, BLOCK_M_PER_WARP_FIELD);
    const auto numWarps = integerMetadata(kernel, NUM_WARPS_FIELD);
    const auto blockSize = integerMetadata(kernel, BLOCK_SIZE_FIELD);
    if(!blockMPerWarp.has_value() || !numWarps.has_value() || !blockSize.has_value())
    {
        return std::nullopt;
    }
    if(*blockMPerWarp <= 0 || *numWarps <= 0 || *blockSize <= 0)
    {
        return std::nullopt;
    }
    return AttentionUnified2DGeometryFields{*blockMPerWarp, *numWarps, *blockSize};
}

/**
 * @brief Kernel-scoped applicability: does this candidate's baked problem match the graph?
 *
 * graph_match has already established the topology, the layout, the head size and that
 * the graph is causal. What is left is per-candidate, and must be, so that a graph this
 * candidate cannot serve can still be served by a sibling with another page size.
 */
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto* node = sdpaNode(context);
    if(node == nullptr)
    {
        return false;
    }
    const auto* q = findTensor(context, node->q_tensor_uid());
    const auto* k = findTensor(context, node->k_tensor_uid());
    if(q == nullptr || k == nullptr || !isWellFormedOperand(*q) || !isWellFormedOperand(*k))
    {
        return false;
    }
    const auto problem = problemFor(*q, *k);

    // Before any division by a geometry field.
    const auto geometry = candidateGeometry(kernel);
    if(!geometry.has_value())
    {
        return false;
    }

    const auto dataTypeName = supportedDataTypeName(problem.dataType);
    if(!dataTypeName.has_value()
       || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dataTypeName)
    {
        return false;
    }

    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto numQueryHeads = integerMetadata(kernel, NUM_QUERY_HEADS_FIELD);
    const auto numKvHeads = integerMetadata(kernel, NUM_KV_HEADS_FIELD);
    if(headSize != problem.headSize || numQueryHeads != problem.numQueryHeads
       || numKvHeads != problem.numKvHeads)
    {
        return false;
    }

    // The window the graph asked for (always 0 here) against the one the binary bakes.
    const auto boundWindow = bound.find(std::string(SLIDING_WINDOW_TOKEN));
    if(boundWindow == bound.end())
    {
        return false;
    }
    const auto candidateWindow = integerMetadata(kernel, SLIDING_WINDOW_FIELD);
    if(!candidateWindow.has_value() || *candidateWindow != boundWindow->second)
    {
        return false;
    }

    // Sinks are a live kernarg the dense graph has no source for, so a variant baked with
    // them cannot be fed from here even though the ABI has the slot.
    const auto useSinks = integerMetadata(kernel, USE_SINKS_FIELD);
    if(!useSinks.has_value() || *useSinks != 0)
    {
        return false;
    }

    // The KV page size must tile seqlen_kv exactly. A partial trailing page would make
    // the identity block table address tokens past the end of K/V. Per-candidate, so a
    // sibling with a page size that does divide can still serve this graph.
    if(problem.seqLenKv % geometry->blockSize != 0)
    {
        return false;
    }

    // The binary-search capacity bound. See gfx950AttentionUnified2DMaxBatch: a kernel
    // baked at `num_seqs` converges for any batch up to 2^ceil(log2(num_seqs+1)) - 1, and
    // SILENTLY resolves the wrong sequence beyond it. A candidate declaring no num_seqs
    // is refused rather than assumed generous.
    const auto numSeqs = integerMetadata(kernel, NUM_SEQS_FIELD);
    if(!numSeqs.has_value())
    {
        return false;
    }
    return problem.batch <= gfx950AttentionUnified2DMaxBatch(*numSeqs);
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

// A candidate whose geometry does not parse cannot be ranked above one that does.
constexpr double UNSUPPORTED_GEOMETRY_SCORE = 0.0;
// The ceiling a ranked candidate approaches from below, so every ranked candidate still
// outranks an unparseable one.
constexpr double RANKED_CEILING = 1.0;
// Spread used to fold the geometry into one ordering key. Large enough that block_m never
// aliases a tile_size.
constexpr int64_t BLOCK_M_KEY_WEIGHT = 4096;
constexpr double KEY_SPAN = 1048576.0;

/**
 * @brief Rank the candidates that survived matching.
 *
 * This catalog ships ONE geometry per problem, so for now ranking only ever breaks a tie
 * between kernels that are equal on every field a graph can see. It is written as a real
 * ordering anyway, because the moment a second geometry is added -- which is what
 * RUNBOOK stage 6 exists to decide -- an arbitrary order would silently pick between
 * them.
 *
 * Prefers a larger Q tile (more rows per CTA, fewer blocks) and, under that, a larger KV
 * tile. Higher wins.
 */
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto geometry = candidateGeometry(kernel);
    if(!geometry.has_value())
    {
        return UNSUPPORTED_GEOMETRY_SCORE;
    }
    const auto tileSize = integerMetadata(kernel, TILE_SIZE_FIELD);
    if(!tileSize.has_value() || *tileSize <= 0)
    {
        return UNSUPPORTED_GEOMETRY_SCORE;
    }
    const int64_t blockM = geometry->blockMPerWarp * geometry->numWarps;
    const int64_t key = blockM * BLOCK_M_KEY_WEIGHT + *tileSize;
    return RANKED_CEILING - 1.0 / (1.0 + static_cast<double>(key) / KEY_SPAN);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/**
 * @brief The 18-argument paged ABI, in order.
 *
 * Measured off the registry rather than transcribed from the kernel source:
 * `candidate.signature(tuning_spec)` for any gfx950 u2d candidate returns exactly this
 * list. Every slot is retained -- the ABI is fixed, not conditional -- so the pointers a
 * dense graph has no source for are passed null rather than omitted.
 *
 * Offsets are the natural packed layout and are printed in diagnostics only; the loader
 * compares kind, size and order.
 */
std::vector<KernelArgument> attentionUnified2DKernelSignature()
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));

    return {KernelArgument{"global_buffer", PTR, 0, "output_ptr"},
            KernelArgument{"global_buffer", PTR, 8, "query_ptr"},
            KernelArgument{"global_buffer", PTR, 16, "key_cache_ptr"},
            KernelArgument{"global_buffer", PTR, 24, "value_cache_ptr"},
            KernelArgument{"global_buffer", PTR, 32, "sink_ptr"},
            KernelArgument{"global_buffer", PTR, 40, "block_tables_ptr"},
            KernelArgument{"global_buffer", PTR, 48, "seq_lens_ptr"},
            KernelArgument{"global_buffer", PTR, 56, "alibi_slopes_ptr"},
            KernelArgument{"global_buffer", PTR, 64, "qq_bias_ptr"},
            KernelArgument{"global_buffer", PTR, 72, "query_start_len_ptr"},
            KernelArgument{"by_value", F32, 80, "scale"},
            KernelArgument{"by_value", F32, 84, "k_scale"},
            KernelArgument{"by_value", F32, 88, "v_scale"},
            KernelArgument{"by_value", F32, 92, "out_scale"},
            KernelArgument{"by_value", F32, 96, "softcap"},
            KernelArgument{"by_value", I32, 100, "num_seqs"},
            KernelArgument{"by_value", I32, 104, "block_table_stride"},
            KernelArgument{"by_value", I32, 108, "qq_bias_stride_0"}};
}

/**
 * @brief Launch state resolved once at prepare() time.
 *
 * Holds the staged workspace CONTENTS as host bytes. They depend only on
 * (batch, seqlen_q, seqlen_kv, block_size), every one of which is known while the
 * MatchContext is still alive, so they are built once here and merely copied at launch.
 * Nothing in this object references the MatchContext or the BoundTokens.
 */
class PreparedGfx950AttentionUnified2D : public PreparedDispatch
{
public:
    PreparedGfx950AttentionUnified2D(IngestorKernelCode code,
                                     AttentionUnified2DBinding binding,
                                     AttentionUnified2DProblem problem,
                                     Gfx950AttentionUnified2DWorkspace workspace,
                                     std::vector<int32_t> stagedWorkspace)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _workspace(workspace)
        , _stagedWorkspace(std::move(stagedWorkspace))
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const AttentionUnified2DBinding& binding() const { return _binding; }
    const AttentionUnified2DProblem& problem() const { return _problem; }
    const Gfx950AttentionUnified2DWorkspace& workspace() const { return _workspace; }
    const std::vector<int32_t>& stagedWorkspace() const { return _stagedWorkspace; }

private:
    IngestorKernelCode _code;
    AttentionUnified2DBinding _binding;
    AttentionUnified2DProblem _problem;
    Gfx950AttentionUnified2DWorkspace _workspace;
    std::vector<int32_t> _stagedWorkspace;
};

/// Read the binding back out of the tokens graph_match set.
AttentionUnified2DBinding attentionUnified2DBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        auto it = bound.find(std::string(token));
        if(it == bound.end())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "gfx950 attention_unified_2d dispatch is missing bound token '"
                    + std::string(token) + "', or it does not hold an integer");
        }
        return it->second;
    };

    AttentionUnified2DBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.slidingWindow = read(SLIDING_WINDOW_TOKEN);
    const auto scaleBits = static_cast<int32_t>(read(SCALE_BITS_TOKEN));
    static_assert(sizeof(binding.scale) == sizeof(scaleBits),
                  "float must be 32-bit to round-trip");
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    return binding;
}

/**
 * @brief Fill the three derived arrays, as int32 words over the whole workspace.
 *
 * block_tables[b][j] = b * blocksPerSequence + j   -- the identity table. This is what a
 *                                                     contiguous dense K/V already is,
 *                                                     which is why graph_match insists on
 *                                                     BSHD K and V.
 * seq_lens[b]        = seqLenKv                    -- every sequence is the same length;
 *                                                     varlen graphs were declined.
 * query_start_len[b] = b * seqLenQ                 -- the cumulative query count, and
 *                                                     [batch] = batch * seqLenQ closes it.
 *
 * Padding words between sections are left zero; the kernel never reads them.
 */
std::vector<int32_t>
    stageUnified2DWorkspace(const AttentionUnified2DProblem& problem,
                            const Gfx950AttentionUnified2DWorkspace& workspace)
{
    constexpr int64_t INT32_BYTES = 4;
    std::vector<int32_t> staged(static_cast<size_t>(workspace.totalBytes / INT32_BYTES), 0);

    const auto wordAt = [&staged](int64_t byteOffset, int64_t index) -> int32_t& {
        return staged[static_cast<size_t>(byteOffset / INT32_BYTES + index)];
    };

    for(int64_t b = 0; b < problem.batch; ++b)
    {
        for(int64_t j = 0; j < workspace.blocksPerSequence; ++j)
        {
            wordAt(workspace.blockTablesOffset, b * workspace.blocksPerSequence + j)
                = static_cast<int32_t>(b * workspace.blocksPerSequence + j);
        }
        wordAt(workspace.seqLensOffset, b) = static_cast<int32_t>(problem.seqLenKv);
        wordAt(workspace.queryStartLenOffset, b) = static_cast<int32_t>(b * problem.seqLenQ);
    }
    wordAt(workspace.queryStartLenOffset, problem.batch)
        = static_cast<int32_t>(problem.batch * problem.seqLenQ);
    return staged;
}

/// The graph's problem, re-derived in prepare()/workspaceBytes() from the context.
std::optional<AttentionUnified2DProblem> problemOf(const MatchContext& context)
{
    const auto* node = sdpaNode(context);
    if(node == nullptr)
    {
        return std::nullopt;
    }
    const auto* q = findTensor(context, node->q_tensor_uid());
    const auto* k = findTensor(context, node->k_tensor_uid());
    if(q == nullptr || k == nullptr || !isWellFormedOperand(*q) || !isWellFormedOperand(*k))
    {
        return std::nullopt;
    }
    return problemFor(*q, *k);
}

/**
 * @brief The native dispatch behind this engine's UDD.
 */
class Gfx950AttentionUnified2DDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit Gfx950AttentionUnified2DDispatchHandler(
        const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    /**
     * @brief The derived paged-attention workspace, in bytes.
     *
     * The first nonzero workspaceBytes() in this provider. Zero is returned only when the
     * query cannot be answered at all -- an unusable graph or candidate -- which is the
     * same path that will refuse to prepare().
     */
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& kernel) const override
    {
        const auto problem = problemOf(context);
        const auto geometry = candidateGeometry(kernel);
        if(!problem.has_value() || !geometry.has_value())
        {
            return 0;
        }
        if(problem->seqLenKv % geometry->blockSize != 0)
        {
            return 0;
        }
        const auto workspace = gfx950AttentionUnified2DWorkspace(problem->batch,
                                                                 problem->seqLenQ,
                                                                 problem->seqLenKv,
                                                                 geometry->blockSize,
                                                                 toString(kernel.kernelId));
        return static_cast<size_t>(workspace.totalBytes);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = attentionUnified2DBinding(bound);

        // The output is checked here rather than in graph_match's operand loop because
        // its layout is what launch() writes through, and a shape inference pass may only
        // have settled it after matching.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o) || !hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_2d: the output tensor is not dense BSHD; the "
                "kernel bakes that layout and takes no stride arguments");
        }

        const auto geometry = candidateGeometry(kernel);
        if(!geometry.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_2d: kernel '" + toString(kernel.kernelId)
                    + "' declares no usable block_m_per_warp/num_warps/block_size geometry");
        }

        const auto problem = problemOf(context);
        if(!problem.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_2d: the graph no longer presents a usable "
                "single SDPA node with well-formed Q and K");
        }

        // hkp_pack lowered this dialect's rocKE builder to a kpack code object, so the
        // overload that takes neither a compiler nor compile options is the right one; it
        // refuses an embedded_source kernel, which a packaged tree can still carry.
        auto code = buildIngestorKernelCode(
            _kpackLoader, context, kernel, attentionUnified2DKernelSignature());

        const auto launchGeometry
            = gfx950AttentionUnified2DGeometry(geometry->blockMPerWarp,
                                               geometry->numWarps,
                                               problem->numQueryHeads,
                                               problem->numKvHeads,
                                               problem->seqLenQ,
                                               problem->batch,
                                               toString(kernel.kernelId));
        code.setBlockSize(launchGeometry.blockX, 1, 1);
        code.setGridSize(launchGeometry.gridX, launchGeometry.gridY, launchGeometry.gridZ);

        const auto workspace = gfx950AttentionUnified2DWorkspace(problem->batch,
                                                                 problem->seqLenQ,
                                                                 problem->seqLenKv,
                                                                 geometry->blockSize,
                                                                 toString(kernel.kernelId));
        auto staged = stageUnified2DWorkspace(*problem, workspace);

        return std::make_unique<PreparedGfx950AttentionUnified2D>(
            std::move(code), binding, *problem, workspace, std::move(staged));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedUnified
            = dynamic_cast<const PreparedGfx950AttentionUnified2D&>(prepared);
        const auto& binding = preparedUnified.binding();
        const auto& problem = preparedUnified.problem();
        const auto& layout = preparedUnified.workspace();

        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_2d: a null workspace was supplied, but this "
                "engine reports a nonzero workspaceBytes() and reads its block table, "
                "sequence lengths and query offsets out of it");
        }

        const auto q = hipdnn_plugin_sdk::findDeviceBuffer(
            binding.q, deviceBuffers, numDeviceBuffers);
        const auto k = hipdnn_plugin_sdk::findDeviceBuffer(
            binding.k, deviceBuffers, numDeviceBuffers);
        const auto v = hipdnn_plugin_sdk::findDeviceBuffer(
            binding.v, deviceBuffers, numDeviceBuffers);
        const auto o = hipdnn_plugin_sdk::findDeviceBuffer(
            binding.o, deviceBuffers, numDeviceBuffers);

        const auto stream = handle.getStream();

        // One copy of all three sections, on the handle's stream so it is ordered before
        // the launch below without a device-wide synchronize. The source outlives the
        // copy: it is owned by `prepared`, which the plan holds for the whole execution.
        const auto& staged = preparedUnified.stagedWorkspace();
        HIP_CHECK(hipMemcpyAsync(workspace,
                                 staged.data(),
                                 static_cast<size_t>(layout.totalBytes),
                                 hipMemcpyHostToDevice,
                                 stream));

        auto* const workspaceBytes = static_cast<std::byte*>(workspace);
        void* const blockTables = workspaceBytes + layout.blockTablesOffset;
        void* const seqLens = workspaceBytes + layout.seqLensOffset;
        void* const queryStartLen = workspaceBytes + layout.queryStartLenOffset;

        preparedUnified.kernelForStream(stream).launch(
            stream,
            o.ptr,
            q.ptr,
            k.ptr,
            v.ptr,
            static_cast<void*>(nullptr), // sink_ptr: no shipped variant bakes use_sinks
            blockTables,
            seqLens,
            static_cast<void*>(nullptr), // alibi_slopes_ptr: use_alibi is never baked
            static_cast<void*>(nullptr), // qq_bias_ptr: use_qq_bias is never baked
            queryStartLen,
            binding.scale,
            NEUTRAL_SCALE,   // k_scale: K is not quantized
            NEUTRAL_SCALE,   // v_scale: V is not quantized
            NEUTRAL_SCALE,   // out_scale: the output is not quantized
            NEUTRAL_SOFTCAP, // softcap: has_softcap is never baked
            static_cast<int32_t>(problem.batch),
            static_cast<int32_t>(layout.blocksPerSequence),
            static_cast<int32_t>(0)); // qq_bias_stride_0: no bias tensor
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

// Defined outside the anonymous namespace: the IngestorPacks.cpp table row names this
// symbol as its resetModuleCache, and the accessor below is what gives the pack a
// process-lifetime module cache to reset.
compilation::KpackModuleCache& gfx950AttentionUnified2DKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx950AttentionUnified2DModuleCache()
{
    gfx950AttentionUnified2DKpackModuleCache().clear();
}

namespace
{

/// This engine's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer, while a provider's Container is created and destroyed per handle, so it must
/// outlive every Container.
const Gfx950AttentionUnified2DDispatchHandler& gfx950AttentionUnified2DDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx950AttentionUnified2DKpackModuleCache());
    static const Gfx950AttentionUnified2DDispatchHandler s_dispatchHandler(s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerGfx950AttentionUnified2DSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx950AttentionUnified2DGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &gfx950AttentionUnified2DDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
