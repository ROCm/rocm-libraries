// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
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
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"

/**
 * @file Gfx950AttentionUnified2DNative.cpp
 * @brief The hipkernel:Gfx950AttentionUnified2D engine's native half: matching, scoring and
 *        dispatch.
 *
 * The kernel is `rocke/library/kernels/gfx950/attention_tiled_2d.py`'s
 * `build_unified_attention_2d_tiled`, packaged (kind: rocke -> kpack) for gfx950 only: the
 * paged, ragged-Q unified attention that vLLM serves. A hipDNN SDPA graph is dense, so this
 * engine serves a graph's contiguous BSHD K/V as a paged cache whose block table is the
 * identity, and stages the three metadata arrays that implies in the plan's workspace (see
 * Gfx950AttentionUnified2DGeometry.hpp). Its applicability rules live only in the Python;
 * this file is where they become enforceable.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified_2d.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified_2d.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_unified_2d.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_unified_2d.dispatch";

// KMD fields (14-field schema). Fields a shipped variant holds at one value across the whole
// catalog (waves_per_eu, kv_storage_dtype, the sink/alibi/qq-bias/softcap switches) are absent:
// a variant that moved one of them would have to add it here first, or the candidates collide
// on the catalog key and the loader drops one.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view MASK_TYPE_FIELD = "mask_type";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view NUM_SEQS_FIELD = "num_seqs";
constexpr std::string_view TILE_SIZE_FIELD = "tile_size";
constexpr std::string_view NUM_WARPS_FIELD = "num_warps";
constexpr std::string_view BLOCK_M_PER_WARP_FIELD = "block_m_per_warp";
constexpr std::string_view USE_MFMA_32X32_FIELD = "use_mfma_32x32";
constexpr std::string_view USE_TRANSPOSED_QK_32X32_FIELD = "use_transposed_qk_32x32";

/// The geometry cold selection prefers: narrow 16x16, 4 warps x 16 rows, a 64-token KV tile.
constexpr int64_t BASELINE_TILE_SIZE = 64;
constexpr int64_t BASELINE_NUM_WARPS = 4;
constexpr int64_t BASELINE_BLOCK_M_PER_WARP = 16;

constexpr std::string_view Q_TOKEN = "gfx950_attention_unified_2d.q.uid";
constexpr std::string_view K_TOKEN = "gfx950_attention_unified_2d.k.uid";
constexpr std::string_view V_TOKEN = "gfx950_attention_unified_2d.v.uid";
constexpr std::string_view O_TOKEN = "gfx950_attention_unified_2d.o.uid";
/// The mask kind graphMatches derived (a MaskType value), so kernelMatches does not re-derive it.
constexpr std::string_view MASK_TOKEN = "gfx950_attention_unified_2d.mask_type";
/// The sliding window the candidate must have been built with. Always 0: maskTypeFor
/// declines every left bound, so no graph that reaches the bind asks for a window.
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx950_attention_unified_2d.sliding_window";
/// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_unified_2d.scale_bits";

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of layout.
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;

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

/// The product of @p factors, or nullopt when it does not fit in int64_t. Every product of
/// graph-controlled extents goes through this: an unchecked one is signed overflow.
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
 * @brief Is this tensor's memory BSHD -- token-major, head varying fastest -- and gap-free?
 *
 * The kernel reads Q and writes O as `[total_q, Hq, D]` rows and reads K/V as a paged cache
 * `[blocks, block_size, Hkv, D]`; with the identity block table those are exactly the BSHD
 * layouts of a contiguous graph tensor. A differently-strided tensor is read as if it were
 * this one, so it must be declined.
 *
 * Unit-extent axes are exempt: a stride multiplies an index that is always 0 when the extent
 * is 1. An expected stride too large for int64_t is one no stride can equal, so that axis
 * fails unless it is unit-extent.
 */
bool hasBshdStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t sequence = dims->Get(SEQ_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    const auto axisOk = [&](uint32_t axis, std::optional<int64_t> expected) {
        return dims->Get(axis) == 1 || (expected.has_value() && strides->Get(axis) == *expected);
    };

    return axisOk(BATCH_AXIS, checkedProduct({sequence, heads, headSize}))
           && axisOk(HEAD_AXIS, headSize) && axisOk(SEQ_AXIS, checkedProduct({heads, headSize}))
           && axisOk(HEAD_SIZE_AXIS, 1);
}

// ---------------------------------------------------------------------------
// Decline logging
//
// hipDNN reports a declined graph to its caller only as "No engine configurations available
// for the graph", so each decline logs its cause at INFO. graph_match runs on every catalog
// miss, so every piece of that work sits inside HIPDNN_PLUGIN_LOG_INFO's argument: the macro
// tests the level first, and with INFO off nothing below is called.
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
/// was declined: graph_match sees every graph the catalog misses on.
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

/// Why @p tensor is not one the kernel can address. Checks the rank itself rather than
/// trusting that the caller validated it.
std::string notBshdReason(std::string_view role, const data_objects::TensorAttributes& tensor)
{
    std::ostringstream out;
    out << describeOperand(role, tensor)
        << " is not dense BSHD (token-major, head fastest); the kernel addresses Q/O as "
           "[tokens, heads, head_size] and K/V as a gap-free paged cache and takes no stride "
           "arguments";
    return out.str();
}

/// The mask kinds this engine serves, in the numbering of rocKE's AttentionMaskType and of
/// the catalog's `mask_type` field. No variant carries a non-zero sliding_window, so a
/// windowed mask has no spelling here.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2
};

/**
 * @brief Which mask the graph is asking for, or nullopt for one this engine lacks.
 *
 * Deprecated causal booleans are merged with the bounds and alignment by
 * asm_sdpa_engine::plan_utils::resolveDiagonalBand, then classified by getMaskType. Both
 * booleans set, invalid bounds, or any band other than unmasked or causal, is declined: no
 * shipped variant carries a non-zero sliding_window, and the right bound is -1 or 0.
 */
std::optional<MaskType> maskTypeFor(const data_objects::SdpaAttributes& attributes)
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
        return std::nullopt;
    }

    switch(planMask)
    {
    case PlanMaskType::NO_MASK:
        return MaskType::NO_MASK;
    case PlanMaskType::TOP_LEFT_CAUSAL:
        return MaskType::TOP_LEFT_CAUSAL;
    case PlanMaskType::BOTTOM_RIGHT_CAUSAL:
        return MaskType::BOTTOM_RIGHT_CAUSAL;
    case PlanMaskType::SLIDING_WINDOW:
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"}
                               << "left_bound " << attributes.left_bound().value_or(-1)
                               << ", right_bound " << attributes.right_bound().value_or(-1)
                               << " is a window; this catalog is causal or unmasked only and no "
                                  "shipped variant carries a sliding_window");
        return std::nullopt;
    default:
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mask"} << "mask type " << static_cast<int>(planMask)
                                                << " is not supported");
        return std::nullopt;
    }
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
    int64_t maskType = 0;
    int64_t slidingWindow = 0;
    float scale = 0.0F;
};

/// The graph facts the matcher and prepare() both need, derived once from the tensors.
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

/// The smallest paged-KV block any candidate is built with: a Skv no block divides cannot be
/// served by any candidate, so graph_match declines it instead of emptying nothing.
constexpr int64_t SMALLEST_BLOCK_SIZE = GFX950_UNIFIED_2D_BLOCK_SIZE_SMALL;

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * Each decline of a graph with an SDPA-forward node logs its cause at INFO (see Declined);
 * any other graph declines silently.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog and skips EVERY
 *          remaining pack, not just this one.
 */
std::optional<BoundTokens> gfx950AttentionUnified2DGraphMatches(const MatchContext& context)
{
    // --- 1. Node shape. One SDPA-forward node; this engine serves a whole graph.
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

    // --- 2. Operands. Q/K/V/O are the four buffers the launch resolves.
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

    // --- 3. Total predicates, before anything indexes an axis.
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

    // --- 4. Layout. The failure is wrong elements in bounds, no fault.
    for(const auto& [role, tensor] : {std::pair{"Q", q}, std::pair{"K", k}, std::pair{"V", v}})
    {
        if(!hasBshdStrides(*tensor))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"layout"} << notBshdReason(role, *tensor));
            return std::nullopt;
        }
    }

    // --- 5. Cross-tensor consistency.
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

    // V shares K's cache layout, so it must share K's shape exactly.
    if(v->dims()->Get(BATCH_AXIS) != problem.batch
       || v->dims()->Get(HEAD_AXIS) != problem.numKvHeads
       || v->dims()->Get(SEQ_AXIS) != problem.seqLenKv
       || v->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"shape"}
                               << describeOperand("V", *v)
                               << " must have K's batch, heads and sequence and Q's head size; "
                               << describeOperand("K", *k));
        return std::nullopt;
    }
    if(k->dims()->Get(BATCH_AXIS) != problem.batch
       || k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"shape"} << describeOperand("K", *k)
                                                 << " must have Q's batch and head size; "
                                                 << describeOperand("Q", *q));
        return std::nullopt;
    }
    // O is Q's shape: the epilogue writes the query rows back.
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
    if(!hasBshdStrides(*o))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"layout"} << notBshdReason("O", *o));
        return std::nullopt;
    }

    // GQA: the kernel derives its group size by integer division, so a non-divisible pair
    // silently drops heads, and a group over 16 is a shape it is not built for.
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0
       || problem.numQueryHeads / problem.numKvHeads > GFX950_UNIFIED_2D_MAX_QUERIES_PER_KV)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"gqa"} << "query heads " << problem.numQueryHeads
                                               << " must be a whole multiple, at most "
                                               << GFX950_UNIFIED_2D_MAX_QUERIES_PER_KV
                                               << ", of key/value heads " << problem.numKvHeads);
        return std::nullopt;
    }

    // head_size 64 or 128. 256 is a separate kernel family and ships in its own pack.
    if(problem.headSize != 64 && problem.headSize != 128)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"head_size"}
                               << "head size " << problem.headSize
                               << " is not supported; this catalog is built for 64 and 128");
        return std::nullopt;
    }

    // The kernel's seq_lens contract (vLLM's) is q_len <= kv_len per sequence: its causal
    // limit is seq_len - q_len, which a longer query makes negative. Declined for every mask
    // rather than served on the strength of how the kernel happens to behave there.
    if(problem.seqLenQ > problem.seqLenKv)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"shape"} << "seqlen_q " << problem.seqLenQ
                                                 << " exceeds seqlen_kv " << problem.seqLenKv
                                                 << "; the kernel needs q_len <= kv_len");
        return std::nullopt;
    }

    // The identity cache is cut into whole blocks, and the smallest block is 16: a Skv that
    // 16 does not divide is not a whole number of blocks at any block size.
    if(problem.seqLenKv % SMALLEST_BLOCK_SIZE != 0)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"seqlen_kv"}
                               << "seqlen_kv " << problem.seqLenKv << " is not a multiple of "
                               << SMALLEST_BLOCK_SIZE
                               << ", so K/V cannot be addressed as whole paged-KV blocks");
        return std::nullopt;
    }

    // --- 6. 32-bit addressing. The kernels are built with i32 K/V addressing, which bounds
    // the cache in bytes; Q/O are held to the same bound.
    constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and fp16 only
    const auto kvBytes = checkedProduct(
        {problem.batch, problem.seqLenKv, problem.numKvHeads, problem.headSize, BYTES_PER_ELEMENT});
    if(!kvBytes.has_value() || *kvBytes > GFX950_UNIFIED_2D_KV_BUFFER_BYTES)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"size_limit"}
                               << "K or V is larger than " << GFX950_UNIFIED_2D_KV_BUFFER_BYTES
                               << " bytes; the kernel addresses K/V with 32-bit byte offsets");
        return std::nullopt;
    }
    const auto qBytes = checkedProduct({problem.batch,
                                        problem.seqLenQ,
                                        problem.numQueryHeads,
                                        problem.headSize,
                                        BYTES_PER_ELEMENT});
    if(!qBytes.has_value() || *qBytes > GFX950_UNIFIED_2D_KV_BUFFER_BYTES)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"size_limit"}
                               << "Q or O is larger than " << GFX950_UNIFIED_2D_KV_BUFFER_BYTES
                               << " bytes; the kernel addresses them with 32-bit offsets");
        return std::nullopt;
    }

    // --- 7. The mask. hipDNN has no `causal` boolean; see maskTypeFor, which logs its own
    // declines. All three kinds are served, each by its own kernel: top-left and bottom-right
    // coincide only at Sq == Skv, but they are distinct kernel bodies and the catalog carries
    // both.
    const auto mask = maskTypeFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }

    // Bound even though it is always zero: kernelMatches compares it against the candidate's
    // sliding_window field, so a variant built with a window cannot be matched by a graph
    // that does not ask for one.
    const int64_t slidingWindow = 0;

    // --- 8. Every optional attribute this kernel cannot honour, declined explicitly. (The
    // kernel has softcap, ALiBi and qq-bias variants, none shipped here; hipDNN's SDPA node
    // has no softcap attribute to decline.)

    // Additive attention bias.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"attn_mask"}
                               << "an additive attention mask (attn_mask tensor) is not "
                                  "supported");
        return std::nullopt;
    }
    // Device-resident scale: the ABI takes `scale` as an f32 kernarg.
    if(attributes.scale_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"scale_tensor"}
                               << "a scale tensor is not supported; the kernel takes the scale "
                                  "as a host value (attn_scale_value)");
        return std::nullopt;
    }
    // Ragged operands: the engine derives query_start_len and seq_lens itself, from uniform
    // lengths.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"varlen"}
                               << "variable sequence lengths (seq_len_q/seq_len_kv tensors) are "
                                  "not supported");
        return std::nullopt;
    }
    // Dropout.
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"dropout"} << "dropout is not supported");
        return std::nullopt;
    }
    // Caller-paged KV: this engine pages K/V itself; a graph that already carries page
    // tables names a cache layout it does not model.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"paged_kv"}
                               << "paged K/V (page tables or max_seq_len_kv) is not supported");
        return std::nullopt;
    }
    // Block-sparse, and attention SINKS (the kernel has a sink path; no shipped variant
    // enables it).
    if(attributes.block_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"block_mask"} << "a block mask is not supported");
        return std::nullopt;
    }
    if(attributes.sink_token_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"sinks"}
                               << "attention sinks (sink_token tensor) are not supported");
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
        HIPDNN_PLUGIN_LOG_INFO(Declined{"fp8"}
                               << "FP8 scaling tensors (descale, scale or amax) are not "
                                  "supported");
        return std::nullopt;
    }
    // Auxiliary softmax outputs. generate_stats is optional<bool>; explicit false is fine.
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"stats"}
                               << "softmax statistics outputs (stats, max, sum_exp, rng_dump or "
                                  "generate_stats) are not supported");
        return std::nullopt;
    }
    // ALiBi slopes and padding masks.
    if(attributes.alibi_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"alibi_mask"} << "alibi_mask is not supported");
        return std::nullopt;
    }
    if(attributes.padding_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"padding_mask"} << "padding_mask is not supported");
        return std::nullopt;
    }
    // mma_core_mode is the MMA operand precision. This kernel's MFMA operands are the graph's
    // own fp16/bf16 inputs, so UNSET (the provider's choice), HALF and BFLOAT16 describe what
    // it runs. An allow-list, so an enum value added later declines until judged.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"mma_core_mode"}
                               << "mma_core_mode " << mmaCoreMode
                               << " is not supported; the kernel runs HALF or BFLOAT16 MFMA");
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint. AUTO leaves the choice to the provider.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"implementation"} << "implementation hint "
                                                          << attributes.implementation()
                                                          << " is not supported; only AUTO is");
        return std::nullopt;
    }

    // The softmax scale is an f32 launch argument. An absent attn_scale_value means 1.0,
    // cuDNN's default. The kernel folds it into an exp2-domain multiplier, so any finite
    // positive scale is well defined; NaN compares false with both tests and is caught by the
    // first.
    const float scale = attributes.attn_scale_value().value_or(1.0F);
    if(!std::isfinite(scale) || !(scale > 0.0F))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"scale"} << "attn_scale " << scale
                                                 << " is not a finite positive number");
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(MASK_TOKEN)] = static_cast<int64_t>(*mask);
    bound[std::string(SLIDING_WINDOW_TOKEN)] = slidingWindow;
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);
    return bound;
}

/// Re-reads the bindings a match established.
AttentionUnified2DBinding attentionUnified2DBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "gfx950 attention_unified_2d dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    AttentionUnified2DBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.maskType = read(MASK_TOKEN);
    binding.slidingWindow = read(SLIDING_WINDOW_TOKEN);

    const auto scaleBits = static_cast<int32_t>(read(SCALE_BITS_TOKEN));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    return binding;
}

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total: never throws, so a malformed record declines instead of faulting a match.
std::optional<int64_t> integerMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

/// A candidate's string metadata field, or nullopt when it is absent or holds another type.
/// Total, like integerMetadata().
std::optional<std::string> stringMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto it = kernel.metadata.find(std::string(field));
    if(it == kernel.metadata.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<std::string>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

/// What a candidate bakes, read from its completed metadata and judged buildable.
struct AttentionUnified2DCandidate
{
    int64_t headSize = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t blockSize = 0;
    int64_t maskType = 0;
    int64_t slidingWindow = 0;
    int64_t numSeqs = 0;
    int64_t tileSize = 0;
    int64_t numWarps = 0;
    int64_t blockMPerWarp = 0;
    bool useMfma32x32 = false;
    bool useTransposedQk32x32 = false;
};

/**
 * @brief The candidate's baked configuration, or nullopt when it has none this engine can
 *        launch.
 *
 * Validated against isSupportedGfx950Unified2DGeometry, so a record the Python builder would
 * refuse is declined like any other unbuildable one. An absent field means an uncompleted or
 * malformed record: it is declined rather than given a default here, because a substituted
 * geometry launches a binary with another binary's grid.
 */
std::optional<AttentionUnified2DCandidate> candidateFor(const KernelDefinition& kernel)
{
    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto numQueryHeads = integerMetadata(kernel, NUM_QUERY_HEADS_FIELD);
    const auto numKvHeads = integerMetadata(kernel, NUM_KV_HEADS_FIELD);
    const auto blockSize = integerMetadata(kernel, BLOCK_SIZE_FIELD);
    const auto maskType = integerMetadata(kernel, MASK_TYPE_FIELD);
    const auto slidingWindow = integerMetadata(kernel, SLIDING_WINDOW_FIELD);
    const auto numSeqs = integerMetadata(kernel, NUM_SEQS_FIELD);
    const auto tileSize = integerMetadata(kernel, TILE_SIZE_FIELD);
    const auto numWarps = integerMetadata(kernel, NUM_WARPS_FIELD);
    const auto blockMPerWarp = integerMetadata(kernel, BLOCK_M_PER_WARP_FIELD);
    const auto useMfma32x32 = integerMetadata(kernel, USE_MFMA_32X32_FIELD);
    const auto useTransposedQk32x32 = integerMetadata(kernel, USE_TRANSPOSED_QK_32X32_FIELD);
    if(!headSize || !numQueryHeads || !numKvHeads || !blockSize || !maskType || !slidingWindow
       || !numSeqs || !tileSize || !numWarps || !blockMPerWarp || !useMfma32x32
       || !useTransposedQk32x32)
    {
        return std::nullopt;
    }

    if(*headSize != 64 && *headSize != 128)
    {
        return std::nullopt;
    }
    if(*numKvHeads <= 0 || *numQueryHeads <= 0 || *numQueryHeads % *numKvHeads != 0
       || *numQueryHeads / *numKvHeads > GFX950_UNIFIED_2D_MAX_QUERIES_PER_KV)
    {
        return std::nullopt;
    }
    if(*maskType < static_cast<int64_t>(MaskType::NO_MASK)
       || *maskType > static_cast<int64_t>(MaskType::BOTTOM_RIGHT_CAUSAL))
    {
        return std::nullopt;
    }
    // A capacity of 0 or less would make the kernel run its unspecialized 32-iteration
    // search; the catalog always states one, so a record without it is malformed.
    if(*numSeqs <= 0 || *useMfma32x32 < 0 || *useMfma32x32 > 1 || *useTransposedQk32x32 < 0
       || *useTransposedQk32x32 > 1)
    {
        return std::nullopt;
    }
    const Gfx950Unified2DGeometryFields geometry{*headSize,
                                                 *blockSize,
                                                 *tileSize,
                                                 *numWarps,
                                                 *blockMPerWarp,
                                                 *useMfma32x32 != 0,
                                                 *useTransposedQk32x32 != 0};
    if(!isSupportedGfx950Unified2DGeometry(geometry))
    {
        return std::nullopt;
    }

    return AttentionUnified2DCandidate{*headSize,
                                       *numQueryHeads,
                                       *numKvHeads,
                                       *blockSize,
                                       *maskType,
                                       *slidingWindow,
                                       *numSeqs,
                                       *tileSize,
                                       *numWarps,
                                       *blockMPerWarp,
                                       *useMfma32x32 != 0,
                                       *useTransposedQk32x32 != 0};
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit?
 *
 * One binary serves any sequence length for its head/dtype/mask configuration: batch,
 * seqlen_q and seqlen_kv reach the kernel as runtime arguments and in the staged tables,
 * so the metadata carries no shape and none is compared. What a candidate does bake, and
 * what this checks against the graph:
 *
 *   - dtype, head_size, num_query_heads, num_kv_heads, the mask kind and sliding window;
 *   - block_size, which must divide seqlen_kv (the identity cache is whole blocks);
 *   - the batch CAPACITY (`num_seqs`): the kernel's binary search resolves at most
 *     2^iters - 1 sequences, so a larger batch reads another sequence's lengths without a
 *     fault.
 */
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return false;
    }
    const auto& attributes = *attributesPtr;

    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    if(q == nullptr || k == nullptr)
    {
        return false;
    }
    const auto problem = problemFor(*q, *k);

    // Before any comparison that could divide by it.
    const auto candidate = candidateFor(kernel);
    if(!candidate.has_value())
    {
        return false;
    }

    const auto dataTypeName = supportedDataTypeName(problem.dataType);
    if(!dataTypeName.has_value() || stringMetadata(kernel, DTYPE_FIELD) != dataTypeName)
    {
        return false;
    }

    if(candidate->headSize != problem.headSize || candidate->numQueryHeads != problem.numQueryHeads
       || candidate->numKvHeads != problem.numKvHeads)
    {
        return false;
    }

    const auto mask = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, MASK_TOKEN);
    if(!mask.has_value() || candidate->maskType != *mask)
    {
        return false;
    }

    // Sliding window: the KV-loop bound is baked at compile time.
    const auto slidingWindow
        = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, SLIDING_WINDOW_TOKEN);
    if(!slidingWindow.has_value() || candidate->slidingWindow != *slidingWindow)
    {
        return false;
    }

    // The identity cache must be whole blocks of THIS candidate's size, and small enough for
    // the table the kernel indexes in 32 bits.
    if(!gfx950Unified2DWorkspaceLayout(problem.batch, problem.seqLenKv, candidate->blockSize)
            .has_value())
    {
        return false;
    }

    return problem.batch <= gfx950Unified2DBatchCapacity(candidate->numSeqs);
}

/// Scores for the cold ranking. The baseline geometry scores `BASELINE_SCORE`; every
/// alternative scores `ALTERNATIVE_CEILING - key / KEY_SPAN`, which lies strictly between zero
/// and ALTERNATIVE_CEILING, so the baseline sits strictly above the whole alternative band.
constexpr double BASELINE_SCORE = 2.0;
constexpr double ALTERNATIVE_CEILING = 1.0;
/// Exceeds every key a legal geometry produces, so an alternative's score stays positive.
/// Powers of two keep every score exactly representable.
constexpr double KEY_SPAN = 8388608.0; // 2^23
/// Below every legal candidate: only a record kernelMatches already declined gets it.
constexpr double UNSUPPORTED_SCORE = 0.0;

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * A stable fallback order for cold selection, not a performance model:
 *
 *  1. The baseline geometry (tile 64, 4 warps x 16 rows) above every alternative.
 *  2. The alternatives in ascending lexicographic (tile_size, num_warps, block_m_per_warp,
 *     use_mfma_32x32, use_transposed_qk_32x32) order.
 *
 * Every distinct geometry gets a distinct score, so the selector's descriptor-id tie-break
 * never decides between two geometries.
 */
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto candidate = candidateFor(kernel);
    if(!candidate.has_value())
    {
        return UNSUPPORTED_SCORE;
    }
    if(candidate->tileSize == BASELINE_TILE_SIZE && candidate->numWarps == BASELINE_NUM_WARPS
       && candidate->blockMPerWarp == BASELINE_BLOCK_M_PER_WARP && !candidate->useMfma32x32
       && !candidate->useTransposedQk32x32)
    {
        return BASELINE_SCORE;
    }
    // tile <= 512, warps <= 8, rows <= 32: the key stays below 2^23.
    int64_t key = candidate->tileSize;
    key = key * 16 + candidate->numWarps;
    key = key * 64 + candidate->blockMPerWarp;
    key = key * 2 + (candidate->useMfma32x32 ? 1 : 0);
    key = key * 2 + (candidate->useTransposedQk32x32 ? 1 : 0);
    return ALTERNATIVE_CEILING - static_cast<double>(key) / KEY_SPAN;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The kernel signature for all variants in this engine: 10 pointers, 6 scalars and the two
/// strides (`_attn_signature(include_bt_stride=True, include_qq_bias_stride=True)`).
///
/// NAMES ARE LOAD-BEARING. requireSignatureMatch compares kind and size always, but names
/// only when BOTH sides carry one. Kind and size alone cannot tell the ten pointers apart, nor
/// the f32 scalars from the i32 ones, so without names an operand permutation passes the check
/// and the kernel reads the wrong buffers with no error and no status code. hkp_pack lowers
/// the recorded names out of the rocKE builder's own parameter list, so keep these spellings
/// identical to the Python (kernels/gfx950/attention_tiled_2d.py,
/// build_unified_attention_2d_tiled's `b.param` calls); a divergence here fails every
/// dispatch rather than silently weakening the check.
///
/// Offsets mirror the packed kernarg layout. They are not compared -- they exist so the
/// mismatch diagnostic prints the real layout beside the recorded one.
std::vector<KernelArgument> unified2DKernelSignature()
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

/// The compiled kernel plus everything launch() needs, owning nothing that points back into
/// the MatchContext or BoundTokens it came from.
class PreparedGfx950AttentionUnified2D : public PreparedDispatch
{
public:
    PreparedGfx950AttentionUnified2D(IngestorKernelCode code,
                                     AttentionUnified2DBinding binding,
                                     AttentionUnified2DProblem problem,
                                     Gfx950Unified2DWorkspaceLayout layout,
                                     std::vector<unsigned char> workspaceImage)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _layout(layout)
        , _workspaceImage(std::move(workspaceImage))
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const AttentionUnified2DBinding& binding() const
    {
        return _binding;
    }

    const AttentionUnified2DProblem& problem() const
    {
        return _problem;
    }

    const Gfx950Unified2DWorkspaceLayout& layout() const
    {
        return _layout;
    }

    const std::vector<unsigned char>& workspaceImage() const
    {
        return _workspaceImage;
    }

private:
    IngestorKernelCode _code;
    AttentionUnified2DBinding _binding;
    AttentionUnified2DProblem _problem;
    Gfx950Unified2DWorkspaceLayout _layout;
    std::vector<unsigned char> _workspaceImage;
};

/// The graph's problem, read from the bound Q and K uids. A null tensor means a bound uid
/// names no tensor: refused by name, since everything after indexes dims.
AttentionUnified2DProblem boundProblem(const MatchContext& context,
                                       const AttentionUnified2DBinding& binding)
{
    const auto* q = findTensor(context, binding.q);
    const auto* k = findTensor(context, binding.k);
    if(q == nullptr || k == nullptr || !isWellFormedOperand(*q) || !isWellFormedOperand(*k))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_2d: the bound Q or K tensor is missing or malformed");
    }
    return problemFor(*q, *k);
}

/// The candidate, or a refusal that names the kernel: kernel_match declines a candidate it
/// cannot launch, so this is defence in depth for a caller that reaches the handler without
/// having matched. The grid and CTA come from it, and there is no geometry to substitute that
/// would not launch this binary with another binary's.
AttentionUnified2DCandidate requireCandidate(const KernelDefinition& kernel)
{
    const auto candidate = candidateFor(kernel);
    if(!candidate.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "gfx950 attention_unified_2d: kernel '"
                                                           + toString(kernel.kernelId)
                                                           + "' declares no supported geometry");
    }
    return *candidate;
}

/// The layout for @p candidate over @p problem, or a refusal that says which rule failed.
Gfx950Unified2DWorkspaceLayout requireLayout(const AttentionUnified2DCandidate& candidate,
                                             const AttentionUnified2DProblem& problem,
                                             const KernelDefinition& kernel)
{
    const std::string name = toString(kernel.kernelId);
    const auto capacity = gfx950Unified2DBatchCapacity(candidate.numSeqs);
    if(problem.batch > capacity)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_2d: kernel '" + name + "' serves at most "
                + std::to_string(capacity) + " sequences and the graph has "
                + std::to_string(problem.batch));
    }
    const auto layout
        = gfx950Unified2DWorkspaceLayout(problem.batch, problem.seqLenKv, candidate.blockSize);
    if(!layout.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_2d: kernel '" + name + "' has block_size "
                + std::to_string(candidate.blockSize) + ", which does not divide seqlen_kv "
                + std::to_string(problem.seqLenKv) + " into a table the kernel can index");
    }
    return *layout;
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

    /// The identity block table, seq_lens and query_start_len, each section 256-byte aligned.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto binding = attentionUnified2DBinding(bound);
        const auto problem = boundProblem(context, binding);
        const auto candidate = requireCandidate(kernel);
        return requireLayout(candidate, problem, kernel).totalBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = attentionUnified2DBinding(bound);

        // graph_match is the gate for this: a non-BSHD output declines there and never
        // reaches prepare(). This re-check is defence in depth for a caller that reaches the
        // handler without having matched, which is why it faults rather than declines --
        // by this point the engine has been chosen and there is no one left to defer to.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o) || !hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_2d: the output tensor is not dense BSHD; the kernel "
                "bakes that layout and takes no stride arguments");
        }

        const auto candidate = requireCandidate(kernel);
        const auto problem = boundProblem(context, binding);
        const auto layout = requireLayout(candidate, problem, kernel);

        // The host-side metadata is built before the code object is loaded, so a graph the
        // tables cannot describe is refused by name without touching an archive.
        auto image = gfx950Unified2DWorkspaceImage(
            problem.batch, problem.seqLenQ, problem.seqLenKv, candidate.blockSize);
        if(!image.has_value() || image->size() != layout.totalBytes)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_2d: kernel '" + toString(kernel.kernelId)
                    + "' cannot stage paged-KV metadata for this graph");
        }

        auto code
            = buildIngestorKernelCode(_kpackLoader, context, kernel, unified2DKernelSignature());

        // Grid from the SELECTED CANDIDATE'S geometry and the GRAPH PROBLEM, not from
        // descriptor shape metadata: the binary takes its shape at runtime.
        const auto geometry = gfx950Unified2DLaunchGeometry(candidate.numWarps,
                                                            candidate.blockMPerWarp,
                                                            problem.numQueryHeads,
                                                            problem.numKvHeads,
                                                            problem.seqLenQ,
                                                            problem.batch,
                                                            toString(kernel.kernelId));

        code.setBlockSize(geometry.blockX, 1, 1);
        code.setGridSize(geometry.gridX, geometry.gridY, geometry.gridZ);

        return std::make_unique<PreparedGfx950AttentionUnified2D>(
            std::move(code), binding, problem, layout, std::move(*image));
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
        const auto& layout = preparedUnified.layout();
        const auto& image = preparedUnified.workspaceImage();

        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_2d: launched without the " + std::to_string(image.size())
                    + "-byte workspace workspaceBytes() asked for");
        }

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);

        // One copy of the whole image, on the handle's stream so the kernel launched next
        // sees it. The image is immutable, and a pageable source is staged by the runtime
        // before the call returns, so concurrent launches of one plan share it safely.
        hipStream_t stream = handle.getStream();
        const hipError_t copied
            = hipMemcpyAsync(workspace, image.data(), image.size(), hipMemcpyHostToDevice, stream);
        if(copied != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string("gfx950 attention_unified_2d: staging the paged-KV metadata failed: ")
                    + hipGetErrorString(copied));
        }

        auto* const base = static_cast<unsigned char*>(workspace);
        void* const blockTables = base + layout.blockTablesOffset;
        void* const seqLens = base + layout.seqLensOffset;
        void* const queryStartLen = base + layout.queryStartLenOffset;

        // Unused pointers are null and the unused scalars neutral: the shipped variants have
        // no sinks, ALiBi, qq-bias, softcap or fp8, so the kernel never reads them.
        void* const unused = nullptr;
        const float neutralScale = 1.0F;
        const float noSoftcap = 0.0F;
        const int32_t noQqBiasStride = 0;

        const auto& p = preparedUnified.problem();
        preparedUnified.kernelForStream(stream).launch(stream,
                                                       o.ptr,
                                                       q.ptr,
                                                       k.ptr,
                                                       v.ptr,
                                                       unused,
                                                       blockTables,
                                                       seqLens,
                                                       unused,
                                                       unused,
                                                       queryStartLen,
                                                       binding.scale,
                                                       neutralScale,
                                                       neutralScale,
                                                       neutralScale,
                                                       noSoftcap,
                                                       static_cast<int32_t>(p.batch),
                                                       static_cast<int32_t>(layout.blocksPerSeq),
                                                       noQqBiasStride);
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

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

/// This engine's dispatch handler, process-lifetime.
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
