// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
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
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"

/**
 * @file Gfx950AttentionDenseNative.cpp
 * @brief The hipkernel:Gfx950AttentionDense engine's native half: matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * The kernel is `rocke/library/kernels/gfx950/attention_dense.py`'s
 * `build_attention_dense`, packaged (kind: rocke -> kpack) for gfx950 only. Its
 * applicability rules live only in that Python; this file is where they become
 * enforceable.
 *
 * The catalog is aligned-only: every shipped candidate has ragged=0 and takes runtime
 * batch/seqlen_q/seqlen_kv. Each candidate carries its own (block_m, block_n) tile and
 * serves any valid B/Sq/Skv with Sq divisible by ITS block_m and Skv divisible by ITS
 * block_n. A graph whose lengths no shipped tile divides is not served by this engine.
 * A candidate whose `ragged` field is not 0 -- an exact-shape boundary-padding build --
 * is declined, whatever its authored shape.
 *
 * Key invariants:
 *
 *  a. **The kernel is BSHD.** The builder computes strides from `Hq * D` / `Hkv * D`
 *     and takes no stride kernargs; a BHSD graph reads the wrong elements in bounds.
 *  b. **All variants are non-persistent.** Every variant takes (q,k,v,o,scale,
 *     batch,seqlen_q,seqlen_kv) -- eight kernel arguments. There is no persistent grid.
 *  c. **Shape-generic.** The shape is a runtime argument, so the metadata's batch/
 *     seqlen_q/seqlen_kv are canonical build inputs and are never compared to the graph.
 *  d. **hipDNN has no `causal` boolean.** Derive from left_bound/right_bound/deprecated
 *     booleans via maskTypeFor(). The deprecated booleans are wrong for shipped bundles.
 *  e. **The tile is per candidate.** block_m/block_n are completed KMD fields (legacy
 *     records omit them and complete to 256/64). They are validated against the tile
 *     rules before anything divides by them; a candidate without a supported tile is
 *     declined, never launched with a substituted one. block_m sizes the launch;
 *     block_n only decides applicability.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_dense.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_dense.dispatch";

// KMD fields (12-field schema). The KMD carries only what varies between candidates, so
// waves_per_eu/persistent/num_persistent/wide_lds_dma are absent: every shipped variant
// holds them at one value. A variant that moved any of them would have to add it here
// first, or the candidates collide on the catalog key and the loader drops one. The
// schema's batch/seqlen_q/seqlen_kv are canonical build inputs this file never reads.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view RAGGED_FIELD = "ragged";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
constexpr std::string_view BLOCK_N_FIELD = "block_n";

/// The tile every legacy record completes to, and the one cold selection prefers.
constexpr int64_t BASELINE_BLOCK_M = 256;
constexpr int64_t BASELINE_BLOCK_N = 64;

constexpr std::string_view Q_TOKEN = "gfx950_attention_dense.q.uid";
constexpr std::string_view K_TOKEN = "gfx950_attention_dense.k.uid";
constexpr std::string_view V_TOKEN = "gfx950_attention_dense.v.uid";
constexpr std::string_view O_TOKEN = "gfx950_attention_dense.o.uid";
/// The mask type graphMatches derived, so kernelMatches does not re-derive it.
constexpr std::string_view CAUSAL_TOKEN = "gfx950_attention_dense.causal";
/// The sliding window size derived from left_bound.
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx950_attention_dense.sliding_window";
/// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_dense.scale_bits";

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of layout.
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;

/// Unbounded, in the left_bound/right_bound convention.
constexpr int64_t UNBOUNDED = -1;

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
 * @brief Is this tensor's memory BSHD -- token-major, head varying fastest?
 *
 * The kernel bakes this layout and there are no stride kernargs, so a
 * differently-strided tensor is read as if it were this one.
 *
 * Unit-extent axes are exempt: a stride multiplies an index that is always 0 when the
 * extent is 1. A single-head tensor is byte-identically BSHD and BHSD while the two
 * spellings disagree on strides[H] -- a strict compare would decline a graph the kernel
 * serves perfectly, and graph_match returning nullopt empties the WHOLE engine catalog.
 */
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

/// The mask kinds this engine serves. Narrower than
/// asm_sdpa_engine/plans/SdpaPlanUtils.hpp::getMaskType, which also classifies windowed
/// masks: no variant in this catalog carries a non-zero sliding_window, so a windowed
/// graph has no spelling here and is declined outright.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2
};

/**
 * @brief Which mask the graph is asking for.
 *
 * A REAL BOUND WINS OVER THE DEPRECATED BOOLEANS. A graph that sets a boolean AND
 * carries a bound is asking for a windowed mask and must be reported as such.
 */
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

    // A non-zero right bound creates a bidirectional window the kernel cannot serve:
    // the compiled kernel is hard-causal (upper mask only) and has no right-bound field.
    // Decline early so the graph is not silently served with wrong numerics.
    if(right != UNBOUNDED && right != 0)
    {
        return std::nullopt;
    }

    // A bounded left edge is a window whatever the booleans say, and no shipped variant
    // carries a non-zero sliding_window. Serving one on a causal binary would apply the
    // wrong mask with no error, so a windowed graph is declined here rather than left to
    // fall through to a kernel comparison it could only fail.
    if(left != UNBOUNDED)
    {
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

    // Both bounds are now either unset or zero: unset on the right is an unmasked graph,
    // zero is a diagonal with no band, whose alignment picks the causal corner.
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

/// The tensor uids and derived scalars a matched dense-attention graph binds.
struct AttentionDenseBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
    int64_t slidingWindow = 0;
    float scale = 0.0F;
};

/// The graph facts the matcher and prepare() both need, derived once from the tensors.
struct AttentionDenseProblem
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
AttentionDenseProblem problemFor(const data_objects::TensorAttributes& q,
                                 const data_objects::TensorAttributes& k)
{
    AttentionDenseProblem problem;
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
 * @brief Graph-scoped applicability for the whole engine.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog and skips
 *          EVERY remaining pack, not just this one.
 */
std::optional<BoundTokens> gfx950AttentionDenseGraphMatches(const MatchContext& context)
{
    // --- 1. Node shape. One SDPA-forward node; this engine serves a whole graph.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // --- 2. Operands. Q/K/V/O are the four the shipped 8-arg ABI has pointers for.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }

    // --- 3. Total predicates, before anything indexes an axis.
    if(!isWellFormedOperand(*q) || !isWellFormedOperand(*k) || !isWellFormedOperand(*v)
       || !isWellFormedOperand(*o))
    {
        return std::nullopt;
    }

    // --- 4. Layout. Tier 1: the failure is wrong elements in bounds, no fault.
    //
    // All four operands are held to the same rule here, O included. The kernel bakes
    // BSHD for the epilogue exactly as it does for the inputs, so a differently-strided
    // output is outside the capability set and a DECLINE is the honest answer: another
    // engine may serve the graph. Accepting it and faulting later would claim a graph
    // this engine cannot execute.
    if(!hasBshdStrides(*q) || !hasBshdStrides(*k) || !hasBshdStrides(*v) || !hasBshdStrides(*o))
    {
        return std::nullopt;
    }

    // --- 5. Cross-tensor consistency.
    const auto problem = problemFor(*q, *k);

    // One dtype across every operand.
    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    if(!supportedDataTypeName(problem.dataType).has_value())
    {
        return std::nullopt;
    }

    // V shares K's base and stride in the builder, so it must share K's shape exactly.
    if(v->dims()->Get(BATCH_AXIS) != problem.batch
       || v->dims()->Get(HEAD_AXIS) != problem.numKvHeads
       || v->dims()->Get(SEQ_AXIS) != problem.seqLenKv
       || v->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }
    // K must agree with Q on batch, and on head size.
    if(k->dims()->Get(BATCH_AXIS) != problem.batch
       || k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }
    // O is Q's shape: the epilogue reuses the query base and stride verbatim.
    if(o->dims()->Get(BATCH_AXIS) != problem.batch
       || o->dims()->Get(HEAD_AXIS) != problem.numQueryHeads
       || o->dims()->Get(SEQ_AXIS) != problem.seqLenQ
       || o->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }

    // GQA: the kernel derives its group size by integer division, so a non-divisible
    // pair silently drops heads.
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }

    // head_size is 64 or 128 (AttentionDenseSpec.__post_init__).
    if(problem.headSize != 64 && problem.headSize != 128)
    {
        return std::nullopt;
    }

    // --- 6. 32-bit addressing. K/V bound is bytes, Q/O is elements.
    constexpr int64_t INT32_LIMIT = 2147483648LL; // 2^31
    constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and fp16 only
    if(problem.batch * problem.seqLenKv * problem.numKvHeads * problem.headSize * BYTES_PER_ELEMENT
       >= INT32_LIMIT)
    {
        return std::nullopt;
    }
    if(problem.batch * problem.seqLenQ * problem.numQueryHeads * problem.headSize >= INT32_LIMIT)
    {
        return std::nullopt;
    }

    // --- 7. The mask. hipDNN has no `causal` boolean; see maskTypeFor.
    const auto mask = maskTypeFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }

    int64_t causal = 0;

    // Windowed graphs are declined in maskTypeFor, so every mask that reaches here is a
    // full-length one. The zero is still bound and compared against the kernel's
    // sliding_window field below, so a variant built with a window cannot be matched by a
    // graph that does not ask for one.
    const int64_t slidingWindow = 0;

    switch(*mask)
    {
    case MaskType::NO_MASK:
        causal = 0;
        break;
    case MaskType::TOP_LEFT_CAUSAL:
        causal = 1;
        break;
    case MaskType::BOTTOM_RIGHT_CAUSAL:
        // The kernel's causal clamp is TOP-LEFT. Bottom-right coincides EXACTLY when
        // Sq == Skv -- and every shipped quick/SdpaFwd causal bundle is that shape.
        if(problem.seqLenQ != problem.seqLenKv)
        {
            return std::nullopt;
        }
        causal = 1;
        break;
    default:
        // Unrecognised mask kinds are declined, never served as if dense.
        return std::nullopt;
    }

    // --- 8. Every optional attribute this kernel cannot honour, declined explicitly.
    // Worked from sdpa_attributes.fbs field by field.

    // Additive attention bias.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Device-resident scale: the ABI takes `scale` as an f32 kernarg.
    if(attributes.scale_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // varlen, both spellings.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Dropout: five spellings.
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
    // Block-sparse, and attention SINKS.
    if(attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // FP8 quantization: six scale UIDs plus two amax outputs.
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Auxiliary softmax outputs. generate_stats is optional<bool>; explicit false is fine.
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        return std::nullopt;
    }
    // ALiBi slopes and padding masks.
    if(attributes.alibi_mask() || attributes.padding_mask())
    {
        return std::nullopt;
    }
    // mma_core_mode: UNSET and explicit FLOAT are inert. Written as an allow-list:
    // the naive `!= UNSET` silently declines every shipped SdpaFwd bundle (they set FLOAT).
    if(attributes.mma_core_mode() != data_objects::DataType::UNSET
       && attributes.mma_core_mode() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint. AUTO leaves the choice to the provider.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }

    // The softmax scale is a REQUIRED launch argument with no default.
    if(!attributes.attn_scale_value().has_value())
    {
        return std::nullopt;
    }
    const float scale = attributes.attn_scale_value().value();

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(CAUSAL_TOKEN)] = causal;
    bound[std::string(SLIDING_WINDOW_TOKEN)] = slidingWindow;
    // BoundTokens carry int64_t, so the scale travels as its IEEE-754 bit pattern.
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);
    return bound;
}

/// Re-reads the bindings a match established.
AttentionDenseBinding attentionDenseBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "gfx950 attention_dense dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    AttentionDenseBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.causal = read(CAUSAL_TOKEN);
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

/// A candidate's (block_m, block_n) tile, read from its completed metadata.
struct AttentionDenseTile
{
    int64_t blockM = 0;
    int64_t blockN = 0;
};

/**
 * @brief The candidate's tile, or nullopt when it has none this engine can launch.
 *
 * Both fields must be present, hold an integer, and together with the candidate's own
 * head_size satisfy isSupportedGfx950AttentionDenseTile -- which includes the LDS budget,
 * so a D128 block_n 256 record is declined like any other unbuildable tile. Legacy
 * records omit block_m/block_n in their raw form and reach here completed to 256/64, so
 * an absent field means an uncompleted or malformed record: it is declined rather than
 * given a default here, because a substituted tile launches a binary with another
 * binary's grid.
 *
 * Every reader that divides by a tile goes through this first.
 */
std::optional<AttentionDenseTile> candidateTile(const KernelDefinition& kernel)
{
    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto blockN = integerMetadata(kernel, BLOCK_N_FIELD);
    if(!headSize.has_value() || !blockM.has_value() || !blockN.has_value()
       || !isSupportedGfx950AttentionDenseTile(*headSize, *blockM, *blockN))
    {
        return std::nullopt;
    }
    return AttentionDenseTile{*blockM, *blockN};
}

/// Does @p tile divide both of the graph's sequence lengths?
bool tileDivides(const AttentionDenseTile& tile, const AttentionDenseProblem& problem)
{
    return problem.seqLenQ % tile.blockM == 0 && problem.seqLenKv % tile.blockN == 0;
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit?
 *
 * Only shape-generic candidates are served: `ragged` must be present, an integer, and 0.
 * A ragged build bakes its exact shape and pads boundary tiles on-chip; this catalog
 * ships none, and one arriving from elsewhere is declined even at its authored shape
 * rather than matched on metadata equality.
 *
 * A served candidate receives batch/seqlen_q/seqlen_kv as runtime kernel arguments, so
 * one binary serves any valid shape for the same head/dtype/mask configuration; the
 * metadata's shape fields are canonical build inputs and are not compared.
 *
 * THE TILE IS THE CANDIDATE'S OWN. It is validated before anything divides by it; a
 * candidate without a supported tile declines. The candidate then requires
 * `Sq % block_m == 0` and `Skv % block_n == 0` for ITS tile, so one graph can admit
 * some tiles of a cohort and not others, and a graph no tile divides admits none.
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
    const auto tile = candidateTile(kernel);
    if(!tile.has_value())
    {
        return false;
    }

    // Shape-generic builds only.
    const auto ragged = integerMetadata(kernel, RAGGED_FIELD);
    if(!ragged.has_value() || *ragged != 0)
    {
        return false;
    }

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

    // The mask the graph derived to, against the mask this variant was compiled for.
    const auto causal = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CAUSAL_TOKEN);
    if(!causal.has_value() || intField(CAUSAL_FIELD) != *causal)
    {
        return false;
    }

    // Sliding window: the KV-loop bound is baked at compile time.
    const auto slidingWindow
        = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, SLIDING_WINDOW_TOKEN);
    if(!slidingWindow.has_value() || intField(SLIDING_WINDOW_FIELD) != *slidingWindow)
    {
        return false;
    }

    // Tile divisibility against THIS candidate's tile.
    return tileDivides(*tile, problem);
}

/// Scores for the cold ranking. Every alternative scores `ALTERNATIVE_CEILING - key /
/// KEY_SPAN`, which lies strictly between zero and ALTERNATIVE_CEILING, so the baseline
/// sits strictly above the whole alternative band.
constexpr double BASELINE_TILE_SCORE = 2.0;
constexpr double ALTERNATIVE_CEILING = 1.0;
/// Weight of block_m in the lexicographic key; exceeds every legal block_n (block_n
/// divides block_m, so it is at most 256).
constexpr int64_t BLOCK_M_KEY_WEIGHT = 512;
/// Exceeds every key a legal tile produces (256 * 512 + 256), so an alternative's score
/// stays positive. Powers of two keep every score exactly representable.
constexpr double KEY_SPAN = 262144.0; // 2^18
/// Below every legal candidate: only a tile kernelMatches already declined gets it.
constexpr double UNSUPPORTED_TILE_SCORE = 0.0;

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * A stable fallback order for cold selection, not a performance model:
 *
 *  1. The baseline 256/64 tile -- every legacy record's -- above every alternative, so
 *     untuned selection is unchanged wherever the baseline applies.
 *  2. The alternatives in ascending lexicographic (block_m, block_n) order: 128/32,
 *     128/64, 128/128, 256/32, 256/128, 256/256.
 *
 * Every distinct tile gets a distinct score, so the selector's descriptor-id tie-break
 * never decides between two tiles. Exhaustive benchmarking still times every candidate.
 */
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto tile = candidateTile(kernel);
    if(!tile.has_value())
    {
        return UNSUPPORTED_TILE_SCORE;
    }
    if(tile->blockM == BASELINE_BLOCK_M && tile->blockN == BASELINE_BLOCK_N)
    {
        return BASELINE_TILE_SCORE;
    }
    const int64_t key = tile->blockM * BLOCK_M_KEY_WEIGHT + tile->blockN;
    return ALTERNATIVE_CEILING - static_cast<double>(key) / KEY_SPAN;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The kernel signature for all variants in this engine.
///
/// All variants are non-persistent. use_sinks=False so no sink_ptr slot.
/// ABI: (q_ptr, k_ptr, v_ptr, o_ptr, scale, batch, seqlen_q, seqlen_kv) -- 8 args.
///
/// NAMES ARE LOAD-BEARING, not decoration. requireSignatureMatch compares kind and size
/// always, but names only when BOTH sides carry one. Kind and size alone cannot tell the
/// four pointers apart, nor `scale` (f32) from `batch` (i32) -- both are by_value/4 -- so
/// without names an operand permutation passes the check and the kernel reads the wrong
/// buffers with no error and no status code. hkp_pack lowers these names into the kpack
/// descriptor from the rocKE builder's own parameter list, so the recorded side carries
/// them and the comparison is live. Keep these spellings identical to the Python
/// (kernels/gfx950/attention_dense.py, the attention_dense_signature parameter list); a
/// divergence here fails every dispatch rather than silently weakening the check.
///
/// Offsets mirror the packed kernarg layout. They are not compared -- they exist so the
/// mismatch diagnostic prints the real layout beside the recorded one instead of eight
/// zeroes that read as data.
std::vector<KernelArgument> attentionDenseKernelSignature()
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));

    return {KernelArgument{"global_buffer", PTR, 0, "q_ptr"},
            KernelArgument{"global_buffer", PTR, 8, "k_ptr"},
            KernelArgument{"global_buffer", PTR, 16, "v_ptr"},
            KernelArgument{"global_buffer", PTR, 24, "o_ptr"},
            KernelArgument{"by_value", F32, 32, "scale"},
            KernelArgument{"by_value", I32, 36, "batch"},
            KernelArgument{"by_value", I32, 40, "seqlen_q"},
            KernelArgument{"by_value", I32, 44, "seqlen_kv"}};
}

/// The compiled kernel plus everything launch() needs, owning nothing that points back
/// into the MatchContext or BoundTokens it came from.
class PreparedGfx950AttentionDense : public PreparedDispatch
{
public:
    PreparedGfx950AttentionDense(IngestorKernelCode code,
                                 AttentionDenseBinding binding,
                                 AttentionDenseProblem problem)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const AttentionDenseBinding& binding() const
    {
        return _binding;
    }

    const AttentionDenseProblem& problem() const
    {
        return _problem;
    }

private:
    IngestorKernelCode _code;
    AttentionDenseBinding _binding;
    AttentionDenseProblem _problem;
};

/**
 * @brief The native dispatch behind this engine's UDD: sizes, prepares and launches.
 */
class Gfx950AttentionDenseDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    Gfx950AttentionDenseDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                        const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Zero: the kernel's only scratch is LDS and registers; no global scratch,
    /// and the 8-arg ABI has no workspace pointer.
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
        const auto binding = attentionDenseBinding(bound);

        // graph_match is the gate for this: a non-BSHD output declines there and never
        // reaches prepare(). This re-check is defence in depth for a caller that reaches
        // the handler without having matched, which is why it faults rather than declines
        // -- by this point the engine has been chosen and there is no one left to defer to.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o) || !hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_dense: the output tensor is not dense BSHD; the kernel "
                "bakes that layout and takes no stride arguments");
        }

        // kernel_match declines a candidate without a supported tile, so this is the same
        // defence in depth: the grid and CTA below come from block_m, and there is no
        // tile to substitute that would not launch this binary with another's geometry.
        const auto tile = candidateTile(kernel);
        if(!tile.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_dense: kernel '" + toString(kernel.kernelId)
                    + "' declares no supported block_m/block_n tile");
        }

        // KernelCompileOptions dereferences the tensor it is handed UNCONDITIONALLY and
        // throws for any 4D stride order that is neither NCHW nor NHWC. BSHD attention
        // memory is neither, so passing the real query tensor throws at prepare() time.
        // A layout-neutral stand-in is safe here only because every kernel in this pack
        // is KPACK, and buildIngestorKernelCode does not consult `options` on the KPACK
        // branch.
        flatbuffers::FlatBufferBuilder standInBuilder;
        {
            const std::vector<int64_t> unitDims{1, 1, 1, 1};
            const std::vector<int64_t> unitStrides{1, 1, 1, 1};
            standInBuilder.Finish(
                data_objects::CreateTensorAttributesDirect(standInBuilder,
                                                           0,
                                                           nullptr,
                                                           data_objects::DataType::FLOAT,
                                                           &unitStrides,
                                                           &unitDims,
                                                           false));
        }
        const auto* standIn = flatbuffers::GetRoot<data_objects::TensorAttributes>(
            standInBuilder.GetBufferPointer());
        const compilation::KernelCompileOptions options(standIn,
                                                        context.deviceProperties.gcnArchName);

        // All variants are non-persistent: every variant takes runtime batch/seqlen_q/seqlen_kv.
        auto code = buildIngestorKernelCode(_kernelCompiler,
                                            _kpackLoader,
                                            context,
                                            kernel,
                                            options,
                                            attentionDenseKernelSignature());

        const auto* q = findTensor(context, binding.q);
        const auto* k = findTensor(context, binding.k);
        const auto problem = problemFor(*q, *k);

        // Grid from the SELECTED CANDIDATE'S block_m and the GRAPH PROBLEM, not from
        // descriptor shape metadata: that carries canonical build inputs (B=1,
        // Sq=Skv=512), not runtime constraints. block_n does not enter the launch.
        const auto geometry = gfx950AttentionDenseGeometry(tile->blockM,
                                                           problem.seqLenQ,
                                                           problem.numQueryHeads,
                                                           problem.batch,
                                                           toString(kernel.kernelId));

        code.setBlockSize(geometry.blockX, 1, 1);
        code.setGridSize(geometry.gridX, geometry.gridY, geometry.gridZ);

        return std::make_unique<PreparedGfx950AttentionDense>(std::move(code), binding, problem);
    }

    /// ABI: (q,k,v,o,scale,batch,seqlen_q,seqlen_kv) -- 8 args, all variants.
    /// use_sinks=False; no sink_ptr slot.
    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedDense = dynamic_cast<const PreparedGfx950AttentionDense&>(prepared);
        const auto& binding = preparedDense.binding();

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);

        const auto& p = preparedDense.problem();
        // (q,k,v,o,scale,batch,seqlen_q,seqlen_kv) -- every variant's ABI.
        preparedDense.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    o.ptr,
                    binding.scale,
                    static_cast<int32_t>(p.batch),
                    static_cast<int32_t>(p.seqLenQ),
                    static_cast<int32_t>(p.seqLenKv));
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& gfx950AttentionDenseKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx950AttentionDenseModuleCache()
{
    gfx950AttentionDenseKpackModuleCache().clear();
}

namespace
{

/// This engine's dispatch handler, process-lifetime.
const Gfx950AttentionDenseDispatchHandler& gfx950AttentionDenseDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx950AttentionDenseKpackModuleCache());
    static const Gfx950AttentionDenseDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                       s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerGfx950AttentionDenseSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx950AttentionDenseGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &gfx950AttentionDenseDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
