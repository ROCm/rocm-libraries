// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// THROWAWAY POC pack (M3). Not for release, not for a PR.
//
// The flyDSL flash-attention pack: a FAMILY of kernel instances (one HSACO per
// (num_query_heads, num_kv_heads, head_size, causal, dtype) tuple), proving hipDNN can
// register N flyDSL attention instances and SELECT the one whose baked config matches an
// SDPA graph -- then RAW-LOAD and launch its HSACO with zero flyDSL at runtime.
//
// Relationship to Gfx950AttentionDenseNative.cpp: the graph_match here is a near-copy of
// that pack's (single SDPA node, BSHD strides, mask derivation, the full battery of
// feature declines), because both target dense bf16 prefill on gfx950 and the graph
// contract is identical. THREE deliberate divergences:
//
//   1. SCALE IS BAKED, NOT PASSED. FlyDSL's flash_attn pre-scales Q by 1/sqrt(head_dim)
//      as a compile-time constant (flash_attn_gfx950.py:409-411, q_loader.scale_all),
//      so there is NO scale kernarg. graph_match therefore REQUIRES the graph's
//      attn_scale_value to be ~= 1/sqrt(head_size) and declines otherwise -- serving a
//      graph whose scale differs would silently compute the wrong softmax.
//
//   2. kernel_match IS SIMPLER. FlyDSL bakes only num_heads/head_dim/causal/dtype into
//      the HSACO; seq_len and batch are RUNTIME scalars feeding the grid, so ONE HSACO
//      spans every prefill length and batch for its head-config. Hence kernel_match pins
//      only (dtype, head_size, num_query_heads, num_kv_heads, causal) -- no seqlen/batch
//      equality, no ragged/tile/persistent/wide_lds machinery.
//
//   3. DISPATCH RAW-LOADS the matched HSACO from $FLYDSL_ATTENTION_HSACO_DIR
//      (hipModuleLoadData), exactly like FlydslRmsNormNative.cpp. It never touches
//      buildIngestorKernelCode/kpack; the descriptor's embedded_source is a placeholder.
//
// FlyDSL 608-byte kernarg ABI (symbol flash_attn_dualwave_swp_gfx950_kernel_0, decoded
// byte-for-byte from the HSACO AMDGPU .args):
//   12 tensor slots, stride 48B: slot k has ptr@(k*48) [8B global_buffer] then a 40B
//   by_value descriptor@(k*48+8). Slot order: 0 Q, 1 K, 2 V, 3 O, 4 LSE, 5 DebugCounts,
//   6 CuSeqQ, 7 CuSeqKv, 8 BlockTable, 9 Bias, 10 AlibiSlopes, 11 Sink. For dense causal
//   prefill the launch wrapper fills every UNUSED slot (4-11) with O's ptr+descriptor;
//   the kernel's const_expr guards keep them from ever being dereferenced.
//   4D descriptor (40B) = { i32 d0, i32 d1, i32 d2, i32 d3, i64 s0, i64 s1, i64 s2 } in
//   PHYSICAL BSHD order [B, S, H, D]; the innermost (D) stride is a compile-time 1 and is
//   elided. Q/O use Hq, K/V use Hkv.
//   8 i32 scalars @576..604: seq_len(Sq), seq_len_kv(Skv), stride_q_n(Hq*D),
//   stride_kv_n(Hkv*D), head_dim_runtime(D), block_table_stride(0), bias_stride0(0),
//   alibi_stride_b(0).
//   grid=(Hq, ceil(Sq/BLOCK_M), B), block=(BLOCK_SIZE=512,1,1), BLOCK_M=256.

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_attention.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_attention.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_attention.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_attention.dispatch";

// KMD metadata fields (baked config + raw-load coordinates).
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view HSACO_FIELD = "hsaco";               // literal HSACO filename
constexpr std::string_view SYMBOL_FIELD = "symbol";             // kernel entry point
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads"; // launch block.x
constexpr std::string_view BLOCK_M_FIELD = "block_m";           // query tile (grid.y)

// Tokens the graph match binds and the dispatch reads back. Internal to this pack.
constexpr std::string_view Q_TOKEN = "flydsl_attention.q.uid";
constexpr std::string_view K_TOKEN = "flydsl_attention.k.uid";
constexpr std::string_view V_TOKEN = "flydsl_attention.v.uid";
constexpr std::string_view O_TOKEN = "flydsl_attention.o.uid";
constexpr std::string_view CAUSAL_TOKEN = "flydsl_attention.causal";

constexpr const char* FLYDSL_HSACO_DIR_ENV = "FLYDSL_ATTENTION_HSACO_DIR";

constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;

constexpr int64_t UNBOUNDED = -1;

// FlyDSL dualwave_swp defaults (flash_attn_utils.py DualwaveSwpTraits, bf16 config).
constexpr int64_t FLYDSL_DEFAULT_BLOCK_M = 256;   // query tile -> grid.y
constexpr int64_t FLYDSL_DEFAULT_BLOCK_SIZE = 512; // 8 waves * 64 -> block.x

// ---------------------------------------------------------------------------
// Optional-metadata helpers
// ---------------------------------------------------------------------------

std::optional<std::string> tryGetStringMeta(const KernelDefinition& kernel, std::string_view field)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return std::nullopt;
    }
    if(const auto* s = std::get_if<std::string>(&value.value()))
    {
        return *s;
    }
    return std::nullopt;
}

int64_t getIntMetaOr(const KernelDefinition& kernel, std::string_view field, int64_t dflt)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return dflt;
    }
    if(const auto* i = std::get_if<int64_t>(&value.value()))
    {
        return *i;
    }
    return dflt;
}

// ---------------------------------------------------------------------------
// Matching (a near-copy of Gfx950AttentionDenseNative.cpp's graph_match)
// ---------------------------------------------------------------------------

struct AttentionBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
};

struct AttentionProblem
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

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

/// Token-major, head varying fastest -- the layout FlyDSL's descriptor strides encode.
/// Unit-extent axes exempt (their stride multiplies an always-zero index). Same predicate
/// as the dense pack's hasBshdStrides.
bool hasBshdStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t heads = dims->Get(HEAD_AXIS);
    const int64_t sequence = dims->Get(SEQ_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    const auto axisOk = [&](uint32_t axis, int64_t expected) {
        return dims->Get(axis) == 1 || strides->Get(axis) == expected;
    };

    return axisOk(BATCH_AXIS, sequence * heads * headSize) && axisOk(HEAD_AXIS, headSize)
           && axisOk(SEQ_AXIS, heads * headSize) && axisOk(HEAD_SIZE_AXIS, 1);
}

enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2,
    SLIDING_WINDOW = 3
};

/// Which mask the graph asks for. A REAL BOUND WINS OVER THE DEPRECATED BOOLEANS.
/// Verbatim from Gfx950AttentionDenseNative.cpp::maskTypeFor -- see that file for the full
/// rationale (cuDNN's set_causal_mask is a deprecated setter for the modern fields).
std::optional<MaskType> maskTypeFor(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();

    if(topLeftDeprecated && bottomRightDeprecated)
    {
        return std::nullopt;
    }

    const int64_t left
        = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    const int64_t right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;

    if(left != UNBOUNDED)
    {
        return MaskType::SLIDING_WINDOW;
    }
    if(topLeftDeprecated || bottomRightDeprecated)
    {
        const bool bottomRight
            = bottomRightDeprecated
              || attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT;
        return bottomRight ? MaskType::BOTTOM_RIGHT_CAUSAL : MaskType::TOP_LEFT_CAUSAL;
    }
    if(right == UNBOUNDED)
    {
        return MaskType::NO_MASK;
    }
    if(right == 0)
    {
        return attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT
                   ? MaskType::BOTTOM_RIGHT_CAUSAL
                   : MaskType::TOP_LEFT_CAUSAL;
    }
    return MaskType::SLIDING_WINDOW;
}

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

AttentionProblem problemFor(const data_objects::TensorAttributes& q,
                            const data_objects::TensorAttributes& k)
{
    AttentionProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numQueryHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headSize = q.dims()->Get(HEAD_SIZE_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

/**
 * @brief Graph-scoped applicability for the flyDSL attention engine.
 *
 * @warning Returning std::nullopt empties THIS engine's own catalog. This pack has its
 *          own graph_match symbol (its own ued), so unlike two packs sharing one, a
 *          decline here cannot touch the dense/tiled/asm engines' catalogs -- they match
 *          the same SDPA graph independently and merge at ranking.
 */
std::optional<BoundTokens> flydslAttentionGraphMatches(const MatchContext& context)
{
    // --- 1. One SDPA-forward node.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // --- 2. Operands: Q/K/V/O.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }

    // --- 3. Well-formedness (O's layout deferred to prepare(), as in the dense pack).
    if(!isWellFormedOperand(*q) || !isWellFormedOperand(*k) || !isWellFormedOperand(*v)
       || !isWellFormedOperand(*o))
    {
        return std::nullopt;
    }

    // --- 4. BSHD layout -- FlyDSL's descriptor strides bake token-major memory.
    if(!hasBshdStrides(*q) || !hasBshdStrides(*k) || !hasBshdStrides(*v))
    {
        return std::nullopt;
    }

    // --- 5. Cross-tensor consistency.
    const auto problem = problemFor(*q, *k);

    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    if(!supportedDataTypeName(problem.dataType).has_value())
    {
        return std::nullopt;
    }

    if(v->dims()->Get(BATCH_AXIS) != problem.batch
       || v->dims()->Get(HEAD_AXIS) != problem.numKvHeads
       || v->dims()->Get(SEQ_AXIS) != problem.seqLenKv
       || v->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }
    if(k->dims()->Get(BATCH_AXIS) != problem.batch
       || k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }
    if(o->dims()->Get(BATCH_AXIS) != problem.batch
       || o->dims()->Get(HEAD_AXIS) != problem.numQueryHeads
       || o->dims()->Get(SEQ_AXIS) != problem.seqLenQ
       || o->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }

    // GQA divisibility (num_kv_heads defaults to num_heads in the factory; both are baked).
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }

    // head_size 64 or 128 (the tuples we AOT-compiled; QK/PV tiling wants a multiple of 32).
    if(problem.headSize != 64 && problem.headSize != 128)
    {
        return std::nullopt;
    }

    // --- 6. 32-bit addressing (offsets lower to i32 nsw; overflow is UB).
    constexpr int64_t INT32_LIMIT = 2147483648LL; // 2^31
    constexpr int64_t BYTES_PER_ELEMENT = 2;      // bf16/fp16 only
    if(problem.batch * problem.seqLenKv * problem.numKvHeads * problem.headSize * BYTES_PER_ELEMENT
       >= INT32_LIMIT)
    {
        return std::nullopt;
    }
    if(problem.batch * problem.seqLenQ * problem.numQueryHeads * problem.headSize * BYTES_PER_ELEMENT
       >= INT32_LIMIT)
    {
        return std::nullopt;
    }

    // --- 7. Mask -> causal. FlyDSL's causal clamp is TOP-LEFT (KV-loop bound derives from
    // the query-block index with no Skv-Sq offset), so BOTTOM_RIGHT coincides only at
    // Sq == Skv. Same shape-conditional accept as the dense pack.
    const auto mask = maskTypeFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }
    int64_t causal = 0;
    switch(*mask)
    {
    case MaskType::NO_MASK:
        causal = 0;
        break;
    case MaskType::TOP_LEFT_CAUSAL:
        causal = 1;
        break;
    case MaskType::BOTTOM_RIGHT_CAUSAL:
        if(problem.seqLenQ != problem.seqLenKv)
        {
            return std::nullopt;
        }
        causal = 1;
        break;
    case MaskType::SLIDING_WINDOW:
    default:
        return std::nullopt;
    }

    // --- 8. Feature declines. The 608B ABI's slots 4-11 are placeholder-filled with O and
    // never dereferenced, so this dense-prefill integration serves NONE of them. Decline
    // every optional attribute field by field (an unchecked field is accepted then never
    // performed -- a silent wrong answer).
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return std::nullopt; // additive bias (slot 9 stays placeholder)
    }
    if(attributes.scale_tensor_uid().has_value())
    {
        return std::nullopt; // device-resident scale: FlyDSL bakes scale, no scale slot
    }
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt; // varlen (cu_seqlens slots 6/7 stay placeholder)
    }
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return std::nullopt; // dropout, five spellings
    }
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        return std::nullopt; // paged KV (block_table slot 8 stays placeholder)
    }
    if(attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value())
    {
        return std::nullopt; // block-sparse / sinks (sink slot 11 stays placeholder)
    }
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        return std::nullopt; // fp8 quantization
    }
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        return std::nullopt; // auxiliary softmax stats (LSE slot 4 stays placeholder)
    }
    if(attributes.alibi_mask() || attributes.padding_mask())
    {
        return std::nullopt; // alibi (slot 10) / padding: no code path in this build
    }
    if(attributes.mma_core_mode() != data_objects::DataType::UNSET
       && attributes.mma_core_mode() != data_objects::DataType::FLOAT)
    {
        return std::nullopt; // FlyDSL emits an f32 accumulator; UNSET/FLOAT are inert
    }
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt; // a named strategy is a request this pack does not implement
    }

    // --- 9. SCALE MUST BE THE BAKED 1/sqrt(head_size). FlyDSL pre-scales Q by this
    // compile-time constant; a graph asking for a different scale would be served the
    // wrong softmax. Require presence AND agreement (relative tolerance, since the
    // frontend may carry a rounded float).
    if(!attributes.attn_scale_value().has_value())
    {
        return std::nullopt;
    }
    const float scale = attributes.attn_scale_value().value();
    const float expected = 1.0F / std::sqrt(static_cast<float>(problem.headSize));
    if(std::fabs(scale - expected) > 1e-4F * expected)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(CAUSAL_TOKEN)] = causal;
    return bound;
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked config fit the graph?
 *
 * SIMPLER than the dense pack's: FlyDSL bakes only num_heads/head_dim/causal/dtype into
 * the HSACO. seq_len and batch are runtime scalars, so ONE HSACO spans all prefill
 * lengths and batches for its head-config -- no seqlen/batch equality here.
 */
bool flydslAttentionKernelMatches(const MatchContext& context,
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

    const auto dataTypeName = supportedDataTypeName(problem.dataType);
    if(!dataTypeName.has_value()
       || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dataTypeName)
    {
        return false;
    }

    const auto intField
        = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };

    if(intField(HEAD_SIZE_FIELD) != problem.headSize
       || intField(NUM_QUERY_HEADS_FIELD) != problem.numQueryHeads
       || intField(NUM_KV_HEADS_FIELD) != problem.numKvHeads)
    {
        return false;
    }

    const auto causal = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CAUSAL_TOKEN);
    if(!causal.has_value() || intField(CAUSAL_FIELD) != *causal)
    {
        return false;
    }
    return true;
}

double flydslAttentionScore(const MatchContext& /*context*/,
                            const BoundTokens& /*bound*/,
                            const KernelDefinition& kernel)
{
    // One survivor per shape after kernel_match; any deterministic value orders it.
    return static_cast<double>(kernel.getIntMetadata(std::string(HEAD_SIZE_FIELD)));
}

AttentionBinding attentionBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl attention dispatch is missing bound token '" + std::string(token) + "'");
        }
        return *value;
    };
    return {read(Q_TOKEN), read(K_TOKEN), read(V_TOKEN), read(O_TOKEN), read(CAUSAL_TOKEN)};
}

// ---------------------------------------------------------------------------
// Dispatch (raw-load the matched HSACO, pack the 608B kernarg)
// ---------------------------------------------------------------------------

class PreparedAttention : public PreparedDispatch
{
public:
    PreparedAttention(hipModule_t mod,
                      hipFunction_t fn,
                      AttentionBinding binding,
                      AttentionProblem problem,
                      uint32_t blockThreads,
                      int64_t blockM)
        : _mod(mod)
        , _fn(fn)
        , _binding(binding)
        , _problem(problem)
        , _blockThreads(blockThreads)
        , _blockM(blockM)
    {
    }

    ~PreparedAttention() override
    {
        if(_mod != nullptr)
        {
            static_cast<void>(hipModuleUnload(_mod));
        }
    }

    PreparedAttention(const PreparedAttention&) = delete;
    PreparedAttention& operator=(const PreparedAttention&) = delete;

    hipFunction_t function() const { return _fn; }
    const AttentionBinding& binding() const { return _binding; }
    const AttentionProblem& problem() const { return _problem; }
    uint32_t blockThreads() const { return _blockThreads; }
    int64_t blockM() const { return _blockM; }

private:
    hipModule_t _mod = nullptr;
    hipFunction_t _fn = nullptr;
    AttentionBinding _binding;
    AttentionProblem _problem;
    uint32_t _blockThreads = static_cast<uint32_t>(FLYDSL_DEFAULT_BLOCK_SIZE);
    int64_t _blockM = FLYDSL_DEFAULT_BLOCK_M;
};

/// Writes one 40B FlyDSL 4D tensor descriptor { i32 d0,d1,d2,d3 ; i64 s0,s1,s2 } at
/// `off` into `args`, in PHYSICAL BSHD order [B, S, H, D]. The innermost (D) stride is a
/// compile-time 1 in the kernel and is not stored.
void writeDescriptor(std::array<unsigned char, 608>& args,
                     size_t off,
                     int64_t b,
                     int64_t s,
                     int64_t h,
                     int64_t d)
{
    const int32_t d0 = static_cast<int32_t>(b);
    const int32_t d1 = static_cast<int32_t>(s);
    const int32_t d2 = static_cast<int32_t>(h);
    const int32_t d3 = static_cast<int32_t>(d);
    const int64_t s0 = s * h * d; // outer stride over batch
    const int64_t s1 = h * d;     // stride over sequence (token-major)
    const int64_t s2 = d;         // stride over head
    std::memcpy(args.data() + off + 0, &d0, sizeof(int32_t));
    std::memcpy(args.data() + off + 4, &d1, sizeof(int32_t));
    std::memcpy(args.data() + off + 8, &d2, sizeof(int32_t));
    std::memcpy(args.data() + off + 12, &d3, sizeof(int32_t));
    std::memcpy(args.data() + off + 16, &s0, sizeof(int64_t));
    std::memcpy(args.data() + off + 24, &s1, sizeof(int64_t));
    std::memcpy(args.data() + off + 32, &s2, sizeof(int64_t));
}

class FlydslAttentionDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
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
        const auto binding = attentionBinding(bound);

        // Re-read the real tensors: seq_len/batch are runtime and not baked, so the launch
        // geometry and kernarg descriptors come from the graph, not the descriptor.
        const auto* q = findTensor(context, binding.q);
        const auto* k = findTensor(context, binding.k);
        const auto* o = findTensor(context, binding.o);
        if(q == nullptr || k == nullptr || o == nullptr || !isWellFormedOperand(*q)
           || !isWellFormedOperand(*k) || !isWellFormedOperand(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl attention: Q/K/O missing or malformed at prepare()");
        }
        // O is addressed with the query base and stride verbatim; a non-BSHD O writes the
        // right bytes to the wrong places. Deferred from graph_match (O shape is frontend-
        // inferred and not reliably populated at match time).
        if(!hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl attention: the output tensor is not dense BSHD; the kernel bakes "
                "that layout and takes no stride arguments");
        }
        const auto problem = problemFor(*q, *k);

        // Metadata drives the raw-load and geometry.
        const std::string symbol = tryGetStringMeta(kernel, SYMBOL_FIELD)
                                       .value_or("flash_attn_dualwave_swp_gfx950_kernel_0");
        const auto blockThreads = static_cast<uint32_t>(
            getIntMetaOr(kernel, BLOCK_THREADS_FIELD, FLYDSL_DEFAULT_BLOCK_SIZE));
        const int64_t blockM = getIntMetaOr(kernel, BLOCK_M_FIELD, FLYDSL_DEFAULT_BLOCK_M);

        const char* dir = std::getenv(FLYDSL_HSACO_DIR_ENV);
        if(dir == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                std::string("flyDSL attention dispatch needs ") + FLYDSL_HSACO_DIR_ENV + " set");
        }
        const auto hsacoName = tryGetStringMeta(kernel, HSACO_FIELD);
        if(!hsacoName.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl attention: selected kernel descriptor omits 'hsaco'");
        }
        const std::string path = std::string(dir) + "/" + hsacoName.value();

        std::vector<char> blob;
        {
            FILE* f = std::fopen(path.c_str(), "rb");
            if(f == nullptr)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM, std::string("cannot open HSACO: ") + path);
            }
            std::fseek(f, 0, SEEK_END);
            const long nbytes = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            blob.resize(static_cast<size_t>(nbytes));
            const size_t got = std::fread(blob.data(), 1, static_cast<size_t>(nbytes), f);
            std::fclose(f);
            if(got != static_cast<size_t>(nbytes))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "short read on HSACO");
            }
        }

        hipModule_t mod = nullptr;
        if(hipModuleLoadData(&mod, blob.data()) != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                          "hipModuleLoadData failed for " + path);
        }
        hipFunction_t fn = nullptr;
        if(hipModuleGetFunction(&fn, mod, symbol.c_str()) != hipSuccess)
        {
            static_cast<void>(hipModuleUnload(mod));
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "hipModuleGetFunction failed for symbol " + symbol);
        }
        return std::make_unique<PreparedAttention>(mod, fn, binding, problem, blockThreads, blockM);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& att = dynamic_cast<const PreparedAttention&>(prepared);
        const auto& b = att.binding();
        const auto& p = att.problem();

        const auto q = hipdnn_plugin_sdk::findDeviceBuffer(b.q, deviceBuffers, numDeviceBuffers);
        const auto k = hipdnn_plugin_sdk::findDeviceBuffer(b.k, deviceBuffers, numDeviceBuffers);
        const auto v = hipdnn_plugin_sdk::findDeviceBuffer(b.v, deviceBuffers, numDeviceBuffers);
        const auto o = hipdnn_plugin_sdk::findDeviceBuffer(b.o, deviceBuffers, numDeviceBuffers);

        void* q_ptr = q.ptr;
        void* k_ptr = k.ptr;
        void* v_ptr = v.ptr;
        void* o_ptr = o.ptr;

        std::array<unsigned char, 608> args{};

        // Slot layout: 48B stride, ptr@(k*48), descriptor@(k*48+8).
        const auto slotPtr = [](int slot) { return static_cast<size_t>(slot) * 48; };
        const auto slotDesc = [](int slot) { return static_cast<size_t>(slot) * 48 + 8; };

        // Slot 0: Q, 1: K, 2: V, 3: O.
        std::memcpy(args.data() + slotPtr(0), &q_ptr, sizeof(void*));
        writeDescriptor(args, slotDesc(0), p.batch, p.seqLenQ, p.numQueryHeads, p.headSize);
        std::memcpy(args.data() + slotPtr(1), &k_ptr, sizeof(void*));
        writeDescriptor(args, slotDesc(1), p.batch, p.seqLenKv, p.numKvHeads, p.headSize);
        std::memcpy(args.data() + slotPtr(2), &v_ptr, sizeof(void*));
        writeDescriptor(args, slotDesc(2), p.batch, p.seqLenKv, p.numKvHeads, p.headSize);
        std::memcpy(args.data() + slotPtr(3), &o_ptr, sizeof(void*));
        writeDescriptor(args, slotDesc(3), p.batch, p.seqLenQ, p.numQueryHeads, p.headSize);

        // Slots 4-11 (LSE, DebugCounts, CuSeqQ, CuSeqKv, BlockTable, Bias, AlibiSlopes,
        // Sink): placeholder-filled with O's ptr + descriptor. const_expr guards in the
        // kernel keep them from ever being dereferenced on the dense causal path.
        for(int slot = 4; slot < 12; ++slot)
        {
            std::memcpy(args.data() + slotPtr(slot), &o_ptr, sizeof(void*));
            writeDescriptor(args, slotDesc(slot), p.batch, p.seqLenQ, p.numQueryHeads, p.headSize);
        }

        // Scalars @576..604 (i32 each).
        const int32_t seqLen = static_cast<int32_t>(p.seqLenQ);
        const int32_t seqLenKv = static_cast<int32_t>(p.seqLenKv);
        const int32_t strideQn = static_cast<int32_t>(p.numQueryHeads * p.headSize);
        const int32_t strideKvn = static_cast<int32_t>(p.numKvHeads * p.headSize);
        const int32_t headDim = static_cast<int32_t>(p.headSize);
        const int32_t zero = 0;
        std::memcpy(args.data() + 576, &seqLen, sizeof(int32_t));
        std::memcpy(args.data() + 580, &seqLenKv, sizeof(int32_t));
        std::memcpy(args.data() + 584, &strideQn, sizeof(int32_t));
        std::memcpy(args.data() + 588, &strideKvn, sizeof(int32_t));
        std::memcpy(args.data() + 592, &headDim, sizeof(int32_t));
        std::memcpy(args.data() + 596, &zero, sizeof(int32_t)); // block_table_stride
        std::memcpy(args.data() + 600, &zero, sizeof(int32_t)); // bias_stride0
        std::memcpy(args.data() + 604, &zero, sizeof(int32_t)); // alibi_stride_b

        size_t argsz = args.size();
        std::array<void*, 5> config{HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                    args.data(),
                                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                    &argsz,
                                    HIP_LAUNCH_PARAM_END};

        // grid = (Hq, ceil(Sq / BLOCK_M), B), block = (BLOCK_SIZE, 1, 1).
        const auto gridX = static_cast<unsigned int>(p.numQueryHeads);
        const auto gridY = static_cast<unsigned int>((p.seqLenQ + att.blockM() - 1) / att.blockM());
        const auto gridZ = static_cast<unsigned int>(p.batch);

        if(hipModuleLaunchKernel(att.function(),
                                 gridX, gridY, gridZ,
                                 att.blockThreads(), 1, 1,
                                 0,
                                 handle.getStream(), nullptr, config.data())
           != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                          "hipModuleLaunchKernel failed");
        }
    }
};

const FlydslAttentionDispatchHandler& flydslAttentionDispatchHandler()
{
    static const FlydslAttentionDispatchHandler s_dispatchHandler;
    return s_dispatchHandler;
}

} // namespace

void registerFlydslAttentionSymbols(hipdnn_plugin_sdk::ingestor::SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslAttentionGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslAttentionKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslAttentionScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslAttentionDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
