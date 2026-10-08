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
 * Those kernels bake more than the RDNA ones: the query and KV head counts, the softmax
 * scale (1/sqrt(head_dim)), the causal flag, and a contiguous BSHD layout; their causal
 * mask is bottom-right and they assume equal query and key lengths. Batch and sequence
 * length are runtime arguments. So the matcher accepts exactly that envelope and declines
 * everything else, and each object serves one (dtype, head counts, head_dim, causal)
 * model shape. Making the baked ones runtime is the next step (see the provider's CDNA
 * plan); until then this engine brings the CDNA path up end to end.
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
constexpr std::string_view NUM_HEADS_FIELD = "num_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
/// 0 for an object that serves any key length; otherwise the multiple the key length must
/// be, because the object skips the KV tail mask.
constexpr std::string_view KV_ALIGN_FIELD = "kv_align";

constexpr std::string_view DUALWAVE = "dualwave";
constexpr std::string_view GENERIC = "generic";

constexpr std::string_view Q_TOKEN = "flydsl_sdpa_cdna.q.uid";
constexpr std::string_view K_TOKEN = "flydsl_sdpa_cdna.k.uid";
constexpr std::string_view V_TOKEN = "flydsl_sdpa_cdna.v.uid";
constexpr std::string_view O_TOKEN = "flydsl_sdpa_cdna.o.uid";
/// 1 when the graph asks for causal masking, 0 for none.
constexpr std::string_view CAUSAL_TOKEN = "flydsl_sdpa_cdna.causal";

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
/// The baked scale must be the one the graph asks for; this is f32 rounding headroom.
constexpr float SCALE_RELATIVE_TOLERANCE = 1e-6F;

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
    int64_t seqLen = 0;
    int64_t headDim = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

SdpaProblem problemFor(const data_objects::TensorAttributes& q,
                       const data_objects::TensorAttributes& k)
{
    SdpaProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLen = q.dims()->Get(SEQ_AXIS);
    problem.headDim = q.dims()->Get(HEAD_DIM_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

/**
 * @brief Is @p tensor a plain device tensor of logical shape [B, H, S, D] laid out as
 *        contiguous BSHD memory?
 *
 * The kernels address each operand by a token pitch and a baked head stride of head_dim,
 * so the only layout they read correctly is BSHD with every row packed: strides
 * (S*H*D, D, H*D, 1) on the logical (batch, head, sequence, head_dim) axes. A ragged
 * operand would be read as one dense batch, so it declines too.
 */
bool isContiguousBshd(const data_objects::TensorAttributes& tensor,
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
        return false;
    }
    if(dims->Get(BATCH_AXIS) != batch || dims->Get(HEAD_AXIS) != heads
       || dims->Get(SEQ_AXIS) != sequence || dims->Get(HEAD_DIM_AXIS) != headDim)
    {
        return false;
    }
    const auto tokenPitch = checkedProduct({heads, headDim});
    const auto batchPitch
        = tokenPitch.has_value() ? checkedProduct({*tokenPitch, sequence}) : std::nullopt;
    if(!batchPitch.has_value() || *batchPitch > INT32_LIMIT)
    {
        return false;
    }
    if(strides->Get(HEAD_DIM_AXIS) != 1 || strides->Get(HEAD_AXIS) != headDim
       || (sequence > 1 && strides->Get(SEQ_AXIS) != *tokenPitch)
       || (batch > 1 && strides->Get(BATCH_AXIS) != *batchPitch))
    {
        return false;
    }
    return !tensor.ragged_offset_tensor_uid().has_value() && !tensor.virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/// Whether the graph's mask is one the kernels apply: none, or plain causal. With equal
/// query and key lengths top-left and bottom-right causal are the same mask, which is the
/// kernels' (bottom-right). Bands and windows decline. nullopt for anything else.
std::optional<bool> causalFor(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeft = attributes.causal_mask();
    const bool bottomRight = attributes.causal_mask_bottom_right();
    if(topLeft && bottomRight)
    {
        return std::nullopt;
    }
    if(attributes.left_bound().has_value())
    {
        return std::nullopt;
    }
    if(attributes.right_bound().has_value())
    {
        return attributes.right_bound().value() == 0 ? std::optional<bool>(true) : std::nullopt;
    }
    return topLeft || bottomRight;
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
    if(problem.batch <= 0 || problem.numHeads <= 0 || problem.numKvHeads <= 0 || problem.seqLen <= 0
       || problem.headDim <= 0)
    {
        return std::nullopt;
    }
    // One dtype throughout, and one the kernels are built for.
    if(!kernelDataTypeName(problem.dataType).has_value() || k->data_type() != problem.dataType
       || v->data_type() != problem.dataType || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    // Equal query and key lengths, contiguous BSHD, V shaped as K, O as Q, GQA by integer
    // division.
    if(k->dims()->Get(SEQ_AXIS) != problem.seqLen || problem.numHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }
    if(!isContiguousBshd(*q, problem.batch, problem.numHeads, problem.seqLen, problem.headDim)
       || !isContiguousBshd(*k, problem.batch, problem.numKvHeads, problem.seqLen, problem.headDim)
       || !isContiguousBshd(*v, problem.batch, problem.numKvHeads, problem.seqLen, problem.headDim)
       || !isContiguousBshd(*o, problem.batch, problem.numHeads, problem.seqLen, problem.headDim))
    {
        return std::nullopt;
    }
    if(problem.seqLen > INT32_LIMIT || problem.batch > INT32_LIMIT)
    {
        return std::nullopt;
    }

    const auto causal = causalFor(attributes);
    if(!causal.has_value())
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
    // No softmax-statistics output: the objects are built without one.
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
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

    // The scale is baked as 1/sqrt(head_dim): the graph must ask for that one.
    if(attributes.scale_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    const float baked = 1.0F / std::sqrt(static_cast<float>(problem.headDim));
    if(attributes.attn_scale_value().has_value()
       && std::fabs(attributes.attn_scale_value().value() - baked)
              > SCALE_RELATIVE_TOLERANCE * baked)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(CAUSAL_TOKEN)] = *causal ? 1 : 0;
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
    const auto queryBlocks = (problem.seqLen + tile.blockM - 1) / tile.blockM;
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
 * dtype, both head counts, head_dim and the causal flag are all baked, so each must equal
 * the graph's; the key length must be a multiple of the candidate's `kv_align`, since an
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

    const auto kvAlign = integerMetadata(kernel, KV_ALIGN_FIELD);
    if(!kvAlign.has_value() || *kvAlign < 0 || (*kvAlign > 0 && problem.seqLen % *kvAlign != 0))
    {
        return false;
    }

    const auto dataTypeName = kernelDataTypeName(problem.dataType);
    const auto causal = tryGetBoundInt(bound, CAUSAL_TOKEN);
    return dataTypeName.has_value() && stringMetadata(kernel, DTYPE_FIELD) == *dataTypeName
           && integerMetadata(kernel, NUM_HEADS_FIELD) == problem.numHeads
           && integerMetadata(kernel, NUM_KV_HEADS_FIELD) == problem.numKvHeads
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
/// pass it: strides (S*H*D, H*D, D) for the batch, sequence and head dimensions.
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
 * bias_stride0, alibi_stride_b (i32). Generic (flash_attn_generic.py): Q, K, V, O, LSE;
 * seq_len, seq_len_kv (i32); CuSeqQ, CuSeqKv, BlockTable; block_table_stride (i32).
 */
std::vector<KernelArgument> signatureOf(std::initializer_list<bool> tensorOrScalar)
{
    const KernelArgument pointer{"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    const KernelArgument layout{"by_value", static_cast<uint32_t>(sizeof(FlydslLayout4)), 0, ""};
    const KernelArgument i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    std::vector<KernelArgument> signature;
    for(const bool isTensor : tensorOrScalar)
    {
        if(isTensor)
        {
            signature.push_back(pointer);
            signature.push_back(layout);
        }
        else
        {
            signature.push_back(i32);
        }
    }
    return signature;
}

const std::vector<KernelArgument>& dualwaveSignature()
{
    static const std::vector<KernelArgument> s_signature
        = signatureOf({true, true, true,  true,  true,  true,  true,  true,  true,  true,
                       true, true, false, false, false, false, false, false, false, false});
    return s_signature;
}

const std::vector<KernelArgument>& genericSignature()
{
    static const std::vector<KernelArgument> s_signature
        = signatureOf({true, true, true, true, true, false, false, true, true, true, false});
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
                           bool dualwave)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _dualwave(dualwave)
    {
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

private:
    IngestorKernelCode _code;
    FlydslSdpaCdnaBinding _binding;
    SdpaProblem _problem;
    bool _dualwave = false;
};

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers.
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
        return std::make_unique<PreparedFlydslSdpaCdna>(
            std::move(code), binding, problem, tile->dualwave);
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

        void* q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers).ptr;
        void* k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers).ptr;
        void* v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers).ptr;
        void* o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers).ptr;

        const auto qLayout
            = bshdLayout(problem.batch, problem.seqLen, problem.numHeads, problem.headDim);
        const auto kvLayout
            = bshdLayout(problem.batch, problem.seqLen, problem.numKvHeads, problem.headDim);
        const auto seqLen = static_cast<int32_t>(problem.seqLen);
        const int32_t zero = 0;
        auto& kernel = preparedSdpa.kernelForStream(handle.getStream());

        // A tensor the launch does not use is passed as O, as upstream's launchers do; the
        // kernels as built never read it. Changing these argument lists means changing
        // dualwaveSignature() / genericSignature() with them.
        if(preparedSdpa.dualwave())
        {
            const auto strideQ = static_cast<int32_t>(problem.numHeads * problem.headDim);
            const auto strideKv = static_cast<int32_t>(problem.numKvHeads * problem.headDim);
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
                          o, // LSE
                          qLayout,
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
                          seqLen,
                          seqLen,
                          strideQ,
                          strideKv,
                          headDim,
                          zero,
                          zero,
                          zero);
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
                      o, // LSE
                      qLayout,
                      seqLen,
                      seqLen,
                      o, // CuSeqQ
                      qLayout,
                      o, // CuSeqKv
                      qLayout,
                      o, // BlockTable
                      qLayout,
                      zero);
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
