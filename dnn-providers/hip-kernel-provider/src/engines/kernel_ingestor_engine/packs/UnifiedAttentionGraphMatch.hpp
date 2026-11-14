// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/PackFeatures.hpp"
#include "engines/kernel_ingestor_engine/packs/SdpaGraphMatch.hpp"

/**
 * @file UnifiedAttentionGraphMatch.hpp
 * @brief The graph match and record-reading helpers the gfx950 unified-attention packs
 *        (hipkernel:Gfx950AttentionUnified2D and hipkernel:Gfx950AttentionUnified3D) share.
 *
 * Both packs serve a single dense BSHD SDPA-forward node through the same paged-KV kernel
 * family, so their graph matches run the same steps around a few engine rules:
 *
 *   matchOperands()   node, operands, layout, dtype and shape consistency
 *   (engine rules)    GQA group, head size, seqlen_q, and anything else the engine adds
 *   matchAttributes() whole paged-KV blocks, 32-bit addressing, the mask, every optional
 *                     attribute the kernels cannot honour, and the scale
 *   bindGraph()       the bound tokens, with the engine's catalog mask key
 *
 * A matcher fix is one edit here; decline lines name their engine through DeclinedBy.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::unified_attention
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

using sdpa_graph::BATCH_AXIS;
using sdpa_graph::DeclinedBy;
using sdpa_graph::describeOperand;
using sdpa_graph::findTensor;
using sdpa_graph::hasBshdStrides;
using sdpa_graph::hasSdpaForwardNode;
using sdpa_graph::HEAD_AXIS;
using sdpa_graph::HEAD_SIZE_AXIS;
using sdpa_graph::isWellFormedOperand;
using sdpa_graph::problemFor;
using sdpa_graph::readProblem;
using sdpa_graph::sdpaNode;
using sdpa_graph::SdpaShape;
using sdpa_graph::SEQ_AXIS;

/// Why @p tensor is not one the kernels can address: they read Q and write O as
/// `[total_q, Hq, D]` rows and read K/V as a paged cache `[blocks, block_size, Hkv, D]`,
/// which with the identity block table are exactly the BSHD layouts of a contiguous tensor.
inline std::string notBshdReason(std::string_view role,
                                 const data_objects::TensorAttributes& tensor)
{
    std::ostringstream out;
    out << describeOperand(role, tensor)
        << " is not dense BSHD (token-major, head fastest); the kernel addresses Q/O as "
           "[tokens, heads, head_size] and K/V as a gap-free paged cache and takes no stride "
           "arguments";
    return out.str();
}

/// The bound stats uid of a graph that asks for no softmax statistics.
inline constexpr int64_t NO_STATS = -1;

/**
 * @brief Is this stats tensor head-major [B, H, S, 1] and gap-free, the layout the LSE-on
 *        kernels write? Callers must have checked isWellFormedOperand().
 *
 * hipDNN's stats layout. Unit-extent axes are exempt, as in hasBshdStrides(): at decode
 * (S == 1) the token-major [B, S, H, 1] strides describe the same memory and pass too.
 */
inline bool hasHeadMajorStatsStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t sequence = dims->Get(SEQ_AXIS);

    const auto axisOk = [&](uint32_t axis, std::optional<int64_t> expected) {
        return dims->Get(axis) == 1 || (expected.has_value() && strides->Get(axis) == *expected);
    };

    return axisOk(BATCH_AXIS, pack_features::checkedProduct({heads, sequence}))
           && axisOk(HEAD_AXIS, sequence) && axisOk(SEQ_AXIS, 1) && axisOk(HEAD_SIZE_AXIS, 1);
}

/// The mask kinds these engines serve, in the numbering of rocKE's AttentionMaskType and of
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
inline std::optional<MaskType> maskTypeFor(const data_objects::SdpaAttributes& attributes,
                                           std::string_view engine)
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
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "mask"}) << e.what());
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
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "mask"})
                               << "left_bound " << attributes.left_bound().value_or(-1)
                               << ", right_bound " << attributes.right_bound().value_or(-1)
                               << " is a window; this catalog is causal or unmasked only and no "
                                  "shipped variant carries a sliding_window");
        return std::nullopt;
    default:
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "mask"})
                               << "mask type " << static_cast<int>(planMask)
                               << " is not supported");
        return std::nullopt;
    }
}

/// The kernel's dtype spelling for a graph dtype, or nullopt for one it cannot be built for.
inline std::optional<std::string> supportedDataTypeName(data_objects::DataType dataType)
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

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total: never throws, so a malformed record declines instead of faulting a match.
inline std::optional<int64_t> integerMetadata(const KernelDefinition& kernel,
                                              std::string_view field)
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
inline std::optional<std::string> stringMetadata(const KernelDefinition& kernel,
                                                 std::string_view field)
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

// ---------------------------------------------------------------------------
// The shared graph match
// ---------------------------------------------------------------------------

/// A graph's SDPA-forward node and its shape, once matchOperands() has accepted them.
struct MatchedOperands
{
    const data_objects::SdpaAttributes* attributes = nullptr;
    SdpaShape problem;
};

/**
 * @brief The node, its four operands, their layout, and dtype and shape consistency.
 *
 * Each decline of a graph with an SDPA-forward node logs its cause at INFO under @p engine;
 * any other graph declines silently. On success the engine applies its own shape rules, then
 * matchAttributes().
 */
inline std::optional<MatchedOperands> matchOperands(const MatchContext& context,
                                                    std::string_view engine)
{
    // --- 1. Node shape. One SDPA-forward node; these engines serve a whole graph.
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        if(HIPDNN_PLUGIN_LOG_IS_INFO_ENABLED() && hasSdpaForwardNode(context))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "node"})
                                   << "the graph is not a single SDPA-forward node ("
                                   << context.graph.nodeCount() << " node(s))");
        }
        return std::nullopt;
    }

    // --- 2. Operands. Q/K/V/O are the four buffers the launch resolves.
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* k = findTensor(context, attributes->k_tensor_uid());
    const auto* v = findTensor(context, attributes->v_tensor_uid());
    const auto* o = findTensor(context, attributes->o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "operand"})
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
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "operand"})
                                   << describeOperand(role, *tensor)
                                   << " must be a rank-4 (B, H, S, D) tensor with positive "
                                      "extents, not virtual and not pass-by-value");
            return std::nullopt;
        }
        // A ragged (RFC-0014) operand packs its tokens behind an offsets tensor; reading it
        // as a dense BSHD buffer would be silently wrong.
        if(tensor->ragged_offset_tensor_uid().has_value())
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "ragged"})
                                   << describeOperand(role, *tensor)
                                   << " is ragged (has a ragged_offset tensor); ragged operands "
                                      "are not supported");
            return std::nullopt;
        }
    }

    // --- 4. Layout. The failure is wrong elements in bounds, no fault.
    for(const auto& [role, tensor] : {std::pair{"Q", q}, std::pair{"K", k}, std::pair{"V", v}})
    {
        if(!hasBshdStrides(*tensor))
        {
            HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "layout"}) << notBshdReason(role, *tensor));
            return std::nullopt;
        }
    }

    // --- 5. Cross-tensor consistency.
    const auto problem = problemFor(*q, *k);

    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "data_type"})
                               << "Q, K, V and O must share one data type, got " << problem.dataType
                               << ", " << k->data_type() << ", " << v->data_type() << ", "
                               << o->data_type());
        return std::nullopt;
    }
    if(!supportedDataTypeName(problem.dataType).has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "data_type"})
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
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "shape"})
                               << describeOperand("V", *v)
                               << " must have K's batch, heads and sequence and Q's head size; "
                               << describeOperand("K", *k));
        return std::nullopt;
    }
    if(k->dims()->Get(BATCH_AXIS) != problem.batch
       || k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "shape"})
                               << describeOperand("K", *k) << " must have Q's batch and head size; "
                               << describeOperand("Q", *q));
        return std::nullopt;
    }
    // O is Q's shape: the epilogue writes the query rows back.
    if(o->dims()->Get(BATCH_AXIS) != problem.batch
       || o->dims()->Get(HEAD_AXIS) != problem.numQueryHeads
       || o->dims()->Get(SEQ_AXIS) != problem.seqLenQ
       || o->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "shape"})
                               << describeOperand("O", *o) << " must have Q's dims; "
                               << describeOperand("Q", *q));
        return std::nullopt;
    }
    if(!hasBshdStrides(*o))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "layout"}) << notBshdReason("O", *o));
        return std::nullopt;
    }

    return MatchedOperands{attributes, problem};
}

/// The smallest paged-KV block any candidate of either engine is built with: a Skv no block
/// divides cannot be served by any candidate, so graph_match declines it instead of emptying
/// nothing.
inline constexpr int64_t SMALLEST_BLOCK_SIZE = GFX950_UNIFIED_2D_BLOCK_SIZE_SMALL;

/// What matchAttributes() derives from the node for the bind.
struct MatchedAttributes
{
    MaskType mask = MaskType::NO_MASK;
    float scale = 0.0F;
    /// The stats output's uid, or NO_STATS. Only an LSE-on kernel serves a graph with one.
    int64_t stats = NO_STATS;
};

/**
 * @brief The stats output's uid, NO_STATS for a graph that asks for none, or nullopt (with
 *        the cause logged) for one these kernels cannot write.
 *
 * The LSE-on kernels write the natural-log LSE, FP32, head-major [B, Hq, Sq, 1] (see
 * hasHeadMajorStatsStrides()). generate_stats and the stats tensor must come together; the
 * max, sum_exp and rng_dump outputs are never written.
 */
inline std::optional<int64_t> matchStats(const MatchContext& context,
                                         const data_objects::SdpaAttributes& attributes,
                                         const SdpaShape& problem,
                                         std::string_view engine)
{
    if(attributes.max_tensor_uid().has_value() || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "stats"})
                               << "softmax max, sum_exp and rng_dump outputs are not supported");
        return std::nullopt;
    }
    // generate_stats is optional<bool>; explicit false is no request.
    const bool generateStats
        = attributes.generate_stats().has_value() && attributes.generate_stats().value();
    const auto statsUid = attributes.stats_tensor_uid();
    if(!generateStats && !statsUid.has_value())
    {
        return NO_STATS;
    }
    if(!generateStats || !statsUid.has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "stats"})
                               << "generate_stats and a stats tensor must come together");
        return std::nullopt;
    }

    const auto* stats = findTensor(context, statsUid.value());
    if(stats == nullptr || !isWellFormedOperand(*stats))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "stats"})
                               << "the stats output (uid " << statsUid.value()
                               << ") is missing, virtual, or not a rank-4 tensor with positive "
                                  "dims");
        return std::nullopt;
    }
    const auto* dims = stats->dims();
    if(stats->data_type() != data_objects::DataType::FLOAT || dims->Get(BATCH_AXIS) != problem.batch
       || dims->Get(HEAD_AXIS) != problem.numQueryHeads || dims->Get(SEQ_AXIS) != problem.seqLenQ
       || dims->Get(HEAD_SIZE_AXIS) != 1 || !hasHeadMajorStatsStrides(*stats))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "stats"})
                               << describeOperand("stats", *stats)
                               << " is not FP32 head-major [B, Hq, Sq, 1]; the kernel writes the "
                                  "LSE in that layout and takes no stride arguments");
        return std::nullopt;
    }
    return statsUid;
}

/**
 * @brief Whole paged-KV blocks, 32-bit addressing, the mask, every optional attribute these
 *        kernels cannot honour, and the scale. Logs each decline under @p engine.
 */
inline std::optional<MatchedAttributes>
    matchAttributes(const MatchContext& context,
                    const data_objects::SdpaAttributes& attributes,
                    const SdpaShape& problem,
                    std::string_view engine)
{
    // The identity cache is cut into whole blocks, and the smallest block is 16: a Skv that
    // 16 does not divide is not a whole number of blocks at any block size.
    if(problem.seqLenKv % SMALLEST_BLOCK_SIZE != 0)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "seqlen_kv"})
                               << "seqlen_kv " << problem.seqLenKv << " is not a multiple of "
                               << SMALLEST_BLOCK_SIZE
                               << ", so K/V cannot be addressed as whole paged-KV blocks");
        return std::nullopt;
    }

    // --- 6. 32-bit addressing. The kernels are built with i32 K/V addressing, which bounds
    // the cache in bytes; Q/O are held to the same bound.
    constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and fp16 only
    const auto kvBytes = pack_features::checkedProduct(
        {problem.batch, problem.seqLenKv, problem.numKvHeads, problem.headSize, BYTES_PER_ELEMENT});
    if(!kvBytes.has_value() || *kvBytes > GFX950_UNIFIED_2D_KV_BUFFER_BYTES)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "size_limit"})
                               << "K or V is larger than " << GFX950_UNIFIED_2D_KV_BUFFER_BYTES
                               << " bytes; the kernel addresses K/V with 32-bit byte offsets");
        return std::nullopt;
    }
    const auto qBytes = pack_features::checkedProduct({problem.batch,
                                                       problem.seqLenQ,
                                                       problem.numQueryHeads,
                                                       problem.headSize,
                                                       BYTES_PER_ELEMENT});
    if(!qBytes.has_value() || *qBytes > GFX950_UNIFIED_2D_KV_BUFFER_BYTES)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "size_limit"})
                               << "Q or O is larger than " << GFX950_UNIFIED_2D_KV_BUFFER_BYTES
                               << " bytes; the kernel addresses them with 32-bit offsets");
        return std::nullopt;
    }

    // --- 7. The mask. hipDNN has no `causal` boolean; see maskTypeFor, which logs its own
    // declines. Which catalog kernel serves each kind is the engine's (bindGraph's maskKey).
    const auto mask = maskTypeFor(attributes, engine);
    if(!mask.has_value())
    {
        return std::nullopt;
    }

    // --- 8. Every optional attribute these kernels cannot honour, declined explicitly. (The
    // kernels have softcap, ALiBi and qq-bias variants, none shipped; hipDNN's SDPA node has no
    // softcap attribute to decline.)

    // Additive attention bias.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "attn_mask"})
                               << "an additive attention mask (attn_mask tensor) is not "
                                  "supported");
        return std::nullopt;
    }
    // Device-resident scale: the ABI takes `scale` as an f32 kernarg.
    if(attributes.scale_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "scale_tensor"})
                               << "a scale tensor is not supported; the kernel takes the scale "
                                  "as a host value (attn_scale_value)");
        return std::nullopt;
    }
    // Ragged operands: the engine derives query_start_len and seq_lens itself, from uniform
    // lengths.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "varlen"})
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
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "dropout"}) << "dropout is not supported");
        return std::nullopt;
    }
    // Caller-paged KV: these engines page K/V themselves; a graph that already carries page
    // tables names a cache layout they do not model.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "paged_kv"})
                               << "paged K/V (page tables or max_seq_len_kv) is not supported");
        return std::nullopt;
    }
    // Block-sparse, and attention SINKS (the kernels have a sink path; no shipped variant
    // enables it).
    if(attributes.block_mask_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "block_mask"})
                               << "a block mask is not supported");
        return std::nullopt;
    }
    if(attributes.sink_token_tensor_uid().has_value())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "sinks"})
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
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "fp8"})
                               << "FP8 scaling tensors (descale, scale or amax) are not "
                                  "supported");
        return std::nullopt;
    }
    // Auxiliary softmax outputs: the stats output only, which an LSE-on kernel writes.
    const auto stats = matchStats(context, attributes, problem, engine);
    if(!stats.has_value())
    {
        return std::nullopt;
    }
    // ALiBi slopes and padding masks.
    if(attributes.alibi_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "alibi_mask"}) << "alibi_mask is not supported");
        return std::nullopt;
    }
    if(attributes.padding_mask())
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "padding_mask"})
                               << "padding_mask is not supported");
        return std::nullopt;
    }
    // mma_core_mode is the MMA operand precision. These kernels' MFMA operands are the graph's
    // own fp16/bf16 inputs, so UNSET (the provider's choice), HALF and BFLOAT16 describe what
    // they run. An allow-list, so an enum value added later declines until judged.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "mma_core_mode"})
                               << "mma_core_mode " << mmaCoreMode
                               << " is not supported; the kernel runs HALF or BFLOAT16 MFMA");
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint. AUTO leaves the choice to the provider.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "implementation"})
                               << "implementation hint " << attributes.implementation()
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
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{engine, "scale"})
                               << "attn_scale " << scale << " is not a finite positive number");
        return std::nullopt;
    }

    return MatchedAttributes{*mask, scale, *stats};
}

/// An engine's bound-token names: each engine prefixes its own, so two engines' bindings
/// never alias. @p dispatchLabel heads the error a missing token raises.
struct BoundTokenNames
{
    std::string_view q;
    std::string_view k;
    std::string_view v;
    std::string_view o;
    /// The catalog mask key (a MaskType value), so kernelMatches does not re-derive it.
    std::string_view maskType;
    /// The sliding window the candidate must have been built with. Always 0: maskTypeFor
    /// declines every left bound, so no graph that reaches the bind asks for a window.
    std::string_view slidingWindow;
    /// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
    std::string_view scaleBits;
    /// The stats output's uid, or NO_STATS: kernelMatches pairs it with the candidate's
    /// emit_lse.
    std::string_view stats;
    std::string_view dispatchLabel;
};

/// The bound tokens of a matched graph. @p maskKey is the catalog mask kind that serves the
/// graph's mask, which the engine may map (Unified3D serves NO_MASK decode by its
/// bottom-right causal kernel).
inline BoundTokens bindGraph(const data_objects::SdpaAttributes& attributes,
                             MaskType maskKey,
                             float scale,
                             int64_t stats,
                             const BoundTokenNames& names)
{
    BoundTokens bound;
    bound[std::string(names.q)] = attributes.q_tensor_uid();
    bound[std::string(names.k)] = attributes.k_tensor_uid();
    bound[std::string(names.v)] = attributes.v_tensor_uid();
    bound[std::string(names.o)] = attributes.o_tensor_uid();
    bound[std::string(names.maskType)] = static_cast<int64_t>(maskKey);
    // Bound even though it is always zero: kernelMatches compares it against the candidate's
    // sliding_window field, so a variant built with a window cannot be matched by a graph
    // that does not ask for one.
    bound[std::string(names.slidingWindow)] = int64_t{0};
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(names.scaleBits)] = static_cast<int64_t>(scaleBits);
    bound[std::string(names.stats)] = stats;
    return bound;
}

/// The tensor uids and derived scalars a matched graph binds.
struct UnifiedAttentionBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t maskType = 0;
    int64_t slidingWindow = 0;
    float scale = 0.0F;
    /// The stats output's uid, or NO_STATS.
    int64_t stats = NO_STATS;
};

/// Re-reads the bindings bindGraph() established. Throws if one is missing.
inline UnifiedAttentionBinding readBinding(const BoundTokens& bound, const BoundTokenNames& names)
{
    const auto read = [&bound, &names](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string(names.dispatchLabel) + " dispatch is missing bound token '"
                    + std::string(token) + "', or it does not hold an integer");
        }
        return *value;
    };

    UnifiedAttentionBinding binding;
    binding.q = read(names.q);
    binding.k = read(names.k);
    binding.v = read(names.v);
    binding.o = read(names.o);
    binding.maskType = read(names.maskType);
    binding.slidingWindow = read(names.slidingWindow);
    binding.stats = read(names.stats);

    const auto scaleBits = static_cast<int32_t>(read(names.scaleBits));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    return binding;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::unified_attention
