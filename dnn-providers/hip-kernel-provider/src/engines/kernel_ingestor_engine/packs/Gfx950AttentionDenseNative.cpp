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
 * THE gfx942 TWIN IS A REFERENCE, NOT A TEMPLATE. The two kernels share a dispatcher
 * shape and a BSHD ABI, and diverge in ways that are silent if copied across:
 *
 *  1. **The spec is the SHARED dataclass, not a private subclass.** gfx950's builder
 *     takes `AttentionDenseSpec` (23 fields); gfx942's takes a 30-field subclass. The
 *     seven extra fields -- block_m, iglp, lds_row_pad, use_cfvst, use_exp2_fast,
 *     use_v_swizzle, v_row_pad -- do not exist here. In particular there is no
 *     `block_m` KMD field to read: `_BLOCK_M = 256` is a module constant, so the tile
 *     predicate below compares against a compile-time constant.
 *
 *  2. **RAGGED IS SERVED.** gfx942 rejects any `Sq % block_m != 0`; gfx950 emits a
 *     separate kernel path that pads the boundary tiles on-chip. So the tile
 *     divisibility rule is CONDITIONAL on the variant's own `ragged` flag, and the
 *     grid ceiling in the geometry header is live rather than defensive. A ragged
 *     graph matched against an aligned binary would run a grid that never covers the
 *     final partial query block.
 *
 *  3. **The ABI is conditionally extended.** `attention_dense_signature` appends
 *     `sink_ptr` when use_sinks, `cu_seqlens_q/kv` when varlen, and three paged
 *     arguments when paged. This engine ships none of those, so the 5-argument launch
 *     below is correct only BECAUSE the matcher declines every graph that would need
 *     a sixth. Those declines are load-bearing, not politeness.
 *
 *     NOTE WHAT THE CONSTRAINT IS, because an earlier revision of this comment named
 *     the wrong one. `IRunnableKernel::launch` is a variadic template building a
 *     `void*` array of `sizeof...(Args)` (compilation/IRunnableKernel.hpp), so there
 *     is NO five-argument ABI ceiling to run into: passing a sixth kernarg is one
 *     more `findDeviceBuffer` and one more launch argument. These features are
 *     UNIMPLEMENTED, not blocked. What actually gates them is verification -- see the
 *     sink decline in graph_match for which reference executor accepts what.
 *
 * Three facts drive almost every check, each a silent wrong answer if missed:
 *
 *  a. **The kernel is BSHD; hipDNN dims are always (B, H, S, D) with layout in the
 *     strides.** The builder computes strides from `Hq * D` / `Hkv * D` and takes no
 *     stride kernargs, so a BHSD graph is indexed as if it were BSHD: in-bounds reads
 *     of the wrong elements, no fault.
 *  b. **Every shape field is baked into the emitted binary**, including `batch`: the
 *     K/V buffer resources are sized at build time, so a larger batch reads zero-fill
 *     past the bound rather than faulting. Hence strict equality in kernelMatches.
 *  c. **hipDNN has no `causal` boolean.** Causality is derived from the deprecated
 *     `causal_mask` / `causal_mask_bottom_right` pair, which take precedence when set,
 *     otherwise from (`left_bound`, `right_bound`, `diagonal_alignment`). Reading only
 *     the deprecated booleans computes "not causal" for every shipped causal bundle.
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

// KMD fields this engine varies along. No BLOCK_M_FIELD: see the file comment.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view SEQLEN_Q_FIELD = "seqlen_q";
constexpr std::string_view SEQLEN_KV_FIELD = "seqlen_kv";
constexpr std::string_view BATCH_FIELD = "batch";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view BLOCK_N_FIELD = "block_n";
// The persistent grid-stride variant launches a DIFFERENT grid shape, so the launch
// path must know which variant it holds.
constexpr std::string_view PERSISTENT_FIELD = "persistent";
constexpr std::string_view NUM_PERSISTENT_FIELD = "num_persistent";
// Ragged selects a different compiled path (on-chip boundary padding, ceil'd grid,
// guarded store) and RELAXES the tile-divisibility rule. Comparing it is what keeps a
// ragged graph off an aligned binary and vice versa.
constexpr std::string_view RAGGED_FIELD = "ragged";
// THE FOUR ABI-EXTENDING FEATURES, all four compared rather than two of four.
// `attention_dense_signature` grows the kernarg list conditionally: +1 (sink_ptr)
// when use_sinks, +2 (cu_seqlens_q/kv) when varlen, +3 (block_tables, kv_lens,
// block_table_stride) when paged. The shipped launch passes exactly 5, so a
// descriptor built with ANY of them must be unselectable -- otherwise the launch
// reads uninitialised kernarg slots as pointers.
//
// graph_match already declines every graph carrying one of these features, and that
// is the primary defence. These metadata comparisons are the second layer, and the
// reason all four are here rather than the two that were obvious: the asymmetry was
// the finding. sliding_window and use_sinks got a second layer because someone
// thought about the argument-count hazard for those two specifically; varlen and
// paged did not, and the KMD had no field to record the fact even if the matcher had
// wanted to check it. Nothing in the packer or the validator cross-checks
// kernel_source.spec.varlen/paged against metadata either, so a varlen-built
// descriptor's metadata was indistinguishable from a plain one.
//
// That is safe today only because graph_match's declines hold -- a single point of
// failure for a silent, out-of-bounds-read failure mode. Making the defence uniform
// costs two comparisons.
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view USE_SINKS_FIELD = "use_sinks";
constexpr std::string_view VARLEN_FIELD = "varlen";
constexpr std::string_view PAGED_FIELD = "paged";

constexpr std::string_view Q_TOKEN = "gfx950_attention_dense.q.uid";
constexpr std::string_view K_TOKEN = "gfx950_attention_dense.k.uid";
constexpr std::string_view V_TOKEN = "gfx950_attention_dense.v.uid";
constexpr std::string_view O_TOKEN = "gfx950_attention_dense.o.uid";
/// The mask type graphMatches derived, so kernelMatches does not re-derive it.
constexpr std::string_view CAUSAL_TOKEN = "gfx950_attention_dense.causal";
/// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_dense.scale_bits";

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of the
/// memory layout, which lives in the strides.
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;

/// Unbounded, in the left_bound/right_bound convention shared with the reference
/// executor (GpuRefSdpaFwd.cpp treats a negative bound as "no clamp").
constexpr int64_t UNBOUNDED = -1;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids and derived scalars a matched dense-attention graph binds.
struct AttentionDenseBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
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

    // Positive extents. The spec checks this explicitly because Python's `%` is
    // sign-following, so a zero or negative extent passes every divisibility rule --
    // and Hq == 0 emits a division by zero into the kernel.
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
 * differently-strided tensor is read as if it were this one. Q/O use the Hq form,
 * K/V the Hkv form; both are `H * D` in terms of that tensor's own head count, so one
 * predicate serves all four.
 *
 * Unit-extent axes are exempt: a stride multiplies an index that is always 0 when the
 * extent is 1, so no address depends on it. A single-head tensor is byte-identically
 * BSHD and BHSD while the two spellings disagree on strides[H] -- a strict compare
 * would decline a graph the kernel serves perfectly, and graph_match returning nullopt
 * empties the WHOLE engine catalog.
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

/// Mask classification, mirroring asm_sdpa_engine/plans/SdpaPlanUtils.hpp::getMaskType.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2,
    SLIDING_WINDOW = 3
};

/**
 * @brief Which mask the graph is asking for.
 *
 * A REAL BOUND WINS OVER THE DEPRECATED BOOLEANS. The booleans only distinguish
 * top-left from bottom-right; they cannot express a window, so a graph that sets one
 * AND carries a bound is asking for a windowed mask and must be reported as such.
 *
 * This ordering is load-bearing rather than stylistic. Returning on the boolean first
 * serves a causal graph with `left_bound = 128` as plain causal: the window is
 * silently discarded and the kernel attends the whole causal triangle instead of the
 * band. Five gpt_oss graphs in hipDNN's own corpus are exactly that shape, and the
 * gfx950 corpus's swa_sink_prefill traces are the same class -- a bug here turns a
 * decline into a wrong answer.
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

    // A bounded left edge is a window whatever the booleans say.
    if(left != UNBOUNDED)
    {
        return MaskType::SLIDING_WINDOW;
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
    if(right == 0)
    {
        return attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT
                   ? MaskType::BOTTOM_RIGHT_CAUSAL
                   : MaskType::TOP_LEFT_CAUSAL;
    }
    return MaskType::SLIDING_WINDOW;
}

/// The kernel's dtype spelling for a graph dtype, or nullopt for one it cannot be
/// built for. One vocabulary: the KMD carries the hipDNN enum name, so that is what
/// the comparison uses. `AttentionDenseSpec.__post_init__` gates dtype to bf16/fp16.
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
 *          EVERY remaining pack, not just this one
 *          (KernelIngestorStateManager.hpp:450-455).
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

    // --- 2. Operands. Q/K/V/O are the four the shipped 5-slot ABI has pointers for.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }

    // --- 3. Total predicates, before anything indexes an axis. O is included for
    // well-formedness only; its LAYOUT is checked in prepare(), because the frontend
    // infers its shape and it is not reliably populated at match time.
    if(!isWellFormedOperand(*q) || !isWellFormedOperand(*k) || !isWellFormedOperand(*v)
       || !isWellFormedOperand(*o))
    {
        return std::nullopt;
    }

    // --- 4. Layout. Tier 1: the failure is wrong elements in bounds, no fault.
    if(!hasBshdStrides(*q) || !hasBshdStrides(*k) || !hasBshdStrides(*v))
    {
        return std::nullopt;
    }

    // --- 5. Cross-tensor consistency.
    const auto problem = problemFor(*q, *k);

    // One dtype across every operand: the builder takes a single `spec.dtype` and
    // types q/k/v/o pointers with it.
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
    // K must agree with Q on batch, and on head size: the kernel has ONE head_size,
    // while hipDNN permits D_qk != D_v.
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
    // pair silently drops heads. `__post_init__` requires the same.
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }

    // head_size is 64 or 128 (AttentionDenseSpec.__post_init__): the QK/PV MFMA tiling
    // needs a multiple of 32 and the async K/V DMA needs 128 % head_size == 0. A D=256
    // graph belongs to the gfx950_d256 sibling candidate, not here.
    if(problem.headSize != 64 && problem.headSize != 128)
    {
        return std::nullopt;
    }

    // --- 6. 32-bit addressing. Offsets lower to i32 add/mul nsw: signed overflow is
    // UB, so LLVM may poison the whole address chain rather than read the wrong place.
    // The two bounds have DIFFERENT units -- K/V is bytes, Q/O is elements.
    constexpr int64_t INT32_LIMIT = 2147483648LL; // 2^31
    constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and fp16 are the only dtypes here
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
    switch(*mask)
    {
    case MaskType::NO_MASK:
        causal = 0;
        break;
    case MaskType::TOP_LEFT_CAUSAL:
        causal = 1;
        break;
    case MaskType::BOTTOM_RIGHT_CAUSAL:
        // The kernel's causal clamp is TOP-LEFT: its KV-loop bound derives from the
        // query-block index with no (Skv - Sq) offset term. Bottom-right differs from
        // top-left only by that offset, so the two coincide EXACTLY when Sq == Skv --
        // and every shipped quick/SdpaFwd causal bundle sets BOTTOM_RIGHT at Sq == Skv.
        // Declining it outright would decline all of them; serving it at Sq != Skv
        // would be a silent wrong answer. Hence the shape-conditional accept.
        if(problem.seqLenQ != problem.seqLenKv)
        {
            return std::nullopt;
        }
        causal = 1;
        break;
    case MaskType::SLIDING_WINDOW:
    default:
        // DECLINED FOR THIS INTEGRATION, and this is a scope limit rather than a
        // kernel limit -- the gfx950 kernel DOES implement banded causal
        // (spec.sliding_window, a pruned KV loop). It is declined here because every
        // windowed shape in the mined corpus also carries attention sinks, which no
        // available reference executor can verify (the GPU ref, the CPU ref and
        // dnn-benchmarking's PyTorch handler all reject sink_token_tensor_uid), so a
        // windowed variant could not be selected by any verifiable graph. Shipping one
        // would be a variant no graph can reach.
        //
        // `default` is required by -Wswitch-default and doubles as the safe verdict
        // for any mask kind added to the enum later: an unrecognised mask is declined,
        // never served as if it were dense.
        return std::nullopt;
    }

    // --- 8. Every optional attribute this kernel cannot honour, declined explicitly.
    // Worked from sdpa_attributes.fbs field by field: an UNCHECKED field is accepted
    // and then silently not performed. Several features appear under more than one
    // spelling and rejecting only one admits graphs carrying the rest.

    // Additive attention bias: no such input in the 5-slot ABI.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Device-resident scale: the ABI takes `scale` as an f32 KERNARG, so there is no
    // pointer slot for a scale tensor.
    if(attributes.scale_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // varlen, both spellings. The kernel HAS a varlen path, but it appends two
    // cu_seqlens pointers to the signature -- a sixth and seventh argument the shipped
    // variants do not have -- and the reference executors decline varlen graphs.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Dropout: five spellings of one feature.
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return std::nullopt;
    }
    // Paged KV. The kernel has a paged path, but it appends three more kernargs and
    // additionally requires sliding_window > 0, which is declined above. No reference
    // executor verifies paged graphs either.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        return std::nullopt;
    }
    // Block-sparse, and attention SINKS. Sinks are the one feature this kernel
    // advertises that the integration deliberately does not ship, and the reason is
    // VERIFICATION, not ABI. Passing the sink_ptr kernarg is one more
    // findDeviceBuffer plus one more variadic launch argument (see the ABI note in
    // the file header); nothing structural prevents it. What prevents it is that the
    // GPU reference executor -- the one this engine's pinned CI target uses
    // (REFERENCE_EXECUTOR gpu) -- declines sinks alongside block-sparse
    // (GpuSdpaFwdPlan.hpp), so a shipped sink variant would have nothing in CI to
    // check it against. The test_sdk CPU reference DOES compute sinks
    // (CpuFpReferenceSdpa.hpp), so serving them is a bounded change: lift this
    // decline, lift the use_sinks guard in kernelMatches, pass the sixth kernarg, and
    // add a cpu-pinned target. rocKE serves these shapes, so each is a REPORTED
    // coverage gap with a named reason, not a silent scope cut.
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
    // Auxiliary softmax outputs. `generate_stats` and `stats_tensor_uid` are two
    // spellings of one request; `max` and `sum_exp` are two more outputs the kernel
    // never writes. generate_stats is optional<bool>, so an explicit `false` is fine
    // and only a `true` is declined.
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        return std::nullopt;
    }
    // ALiBi slopes and padding masks: neither has a code path in the body.
    if(attributes.alibi_mask() || attributes.padding_mask())
    {
        return std::nullopt;
    }
    // mma_core_mode overrides the MFMA accumulate type. The builder emits an f32
    // accumulator unconditionally, so UNSET and an explicit FLOAT are both inert.
    // Written as an allow-list rather than `!= UNSET`: every shipped SdpaFwd bundle
    // sets "float", so the naive form silently declines all of them.
    if(attributes.mma_core_mode() != data_objects::DataType::UNSET
       && attributes.mma_core_mode() != data_objects::DataType::FLOAT)
    {
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint. AUTO leaves the choice to the
    // provider; a named strategy is a request for a specific backend shape this pack
    // does not implement.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }

    // The softmax scale is a REQUIRED launch argument with no default. The schema
    // marks it optional and the mathematically obvious 1/sqrt(D) would silently
    // override whatever the frontend's omission meant. Require presence.
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
    // BoundTokens carry int64_t, so the scale travels as its IEEE-754 bit pattern and
    // is reassembled in prepare(). A float rounded through an integer would not
    // round-trip.
    int32_t scaleBits = 0;
    static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
    std::memcpy(&scaleBits, &scale, sizeof(scale));
    bound[std::string(SCALE_BITS_TOKEN)] = static_cast<int64_t>(scaleBits);
    return bound;
}

/// Re-reads the bindings a match established. Throws rather than returns, because
/// reaching dispatch without them is an internal inconsistency, not a graph the engine
/// declines.
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

    const auto scaleBits = static_cast<int32_t>(read(SCALE_BITS_TOKEN));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    return binding;
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit?
 *
 * Strict equality on every shape field, because this kernel is fully
 * shape-specialized: batch, both sequence lengths and both head counts are compiled
 * into the emitted code (the K/V buffer-resource extent, the KV-loop trip count
 * `Skv // block_n`, and the grid). A graph differing on any of them is served
 * zero-fill or a truncated loop, silently.
 *
 * THE TILE RULE IS CONDITIONAL ON `ragged`, and that is the gfx950-specific part. An
 * ALIGNED variant requires `Sq % 256 == 0` and `Skv % block_n == 0` exactly as gfx942
 * does. A RAGGED variant is compiled with on-chip boundary padding and a ceil'd grid,
 * so it serves the non-multiple lengths an aligned binary cannot -- and must NOT be
 * selected for an aligned shape, because the two are different binaries and the
 * shape equalities above already pin which one the graph needs.
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
       || intField(NUM_KV_HEADS_FIELD) != problem.numKvHeads
       || intField(SEQLEN_Q_FIELD) != problem.seqLenQ
       || intField(SEQLEN_KV_FIELD) != problem.seqLenKv || intField(BATCH_FIELD) != problem.batch)
    {
        return false;
    }

    // The mask the graph derived to, against the mask this variant was compiled for.
    const auto causal = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, CAUSAL_TOKEN);
    if(!causal.has_value() || intField(CAUSAL_FIELD) != *causal)
    {
        return false;
    }

    // A variant compiled FOR a declined feature must never be selected. graph_match
    // already turns away every windowed, sink, varlen and paged GRAPH, so these
    // comparisons are belt-and-braces against a descriptor set that ships such a
    // variant by mistake: without them, such a variant would match a plain graph and
    // launch a binary whose signature expects more kernargs than this engine passes,
    // reading uninitialised slots as pointers.
    //
    // All FOUR ABI-extending features are compared. Two of them used to be, which is
    // the asymmetry a review caught: the pair that changes the argument count most
    // visibly got a guard and the pair that changes it MORE (varlen +2, paged +3) did
    // not, purely because nobody wrote the field down.
    //
    // ABSENT MEANS NOT-BUILT-WITH-IT, and that distinction is why this reads through
    // tryGetMetadata rather than getIntMetadata. getIntMetadata THROWS on a missing
    // field -- it does not default -- so comparing a field a descriptor does not carry
    // would turn a routine non-match into an exception escaping the matcher. A
    // descriptor predating these fields legitimately omits them, and the only sound
    // reading of that silence is the feature being off: the shipped set carries 0 for
    // all four, and a descriptor built WITH one of them has to say so to be refused.
    const auto featureIsSet = [&kernel](std::string_view field) {
        const auto value = kernel.tryGetMetadata(std::string(field));
        if(!value.has_value())
        {
            return false;
        }
        const auto* held = std::get_if<int64_t>(&*value);
        return held != nullptr && *held != 0;
    };
    if(featureIsSet(SLIDING_WINDOW_FIELD) || featureIsSet(USE_SINKS_FIELD)
       || featureIsSet(VARLEN_FIELD) || featureIsSet(PAGED_FIELD))
    {
        return false;
    }

    // Tile divisibility, conditional on the variant's own ragged flag. block_m is the
    // baked module constant, not a KMD field -- see the geometry header.
    const int64_t blockN = intField(BLOCK_N_FIELD);
    if(blockN <= 0)
    {
        return false;
    }
    // Whether THIS GRAPH is ragged, derived exactly as the dispatcher derives it
    // (dispatch/attention/gfx950.py::_dense_spec):
    //
    //     ragged = (sq == sk) and ((sq % _BLOCK_M != 0) or (sk % block_n != 0))
    //
    // The candidate's flag must AGREE with that, in both directions. Returning
    // "self-attention?" for the ragged arm was wrong and the test caught it: an
    // aligned self-attention graph is self-attention too, so a ragged candidate
    // matched it -- and a ragged binary carries boundary-padding the aligned shape
    // does not need, selected ahead of the binary actually built for it.
    //
    // Deriving rather than comparing metadata is deliberate. `ragged` is one of the
    // two fields the dispatcher DERIVES rather than reads (the other is
    // `persistent`), so this is the one place the engine can disagree with the
    // library about which binary a shape wants.
    const bool aligned
        = problem.seqLenQ % GFX950_ATTENTION_DENSE_BLOCK_M == 0 && problem.seqLenKv % blockN == 0;
    const bool graphIsRagged = problem.seqLenQ == problem.seqLenKv && !aligned;
    const bool kernelIsRagged = intField(RAGGED_FIELD) != 0;
    if(graphIsRagged != kernelIsRagged)
    {
        return false;
    }
    // An aligned candidate additionally requires the tile to divide both lengths;
    // a ragged one exists precisely to serve the lengths that do not divide.
    return kernelIsRagged || aligned;
}

/**
 * @brief Ranks the candidates that survived kernelMatches. Higher wins.
 *
 * Every shape and capability field is pinned by kernelMatches, so exactly one axis is
 * still free: `block_n`, the KV tile width. A larger tile amortises the per-tile
 * barrier, LDS publication and loop overhead across more keys, so it is ranked higher.
 *
 * This is a HEURISTIC placeholder, deliberately ranking one real knob rather than
 * returning a constant (which would make ranking arbitrary and hide
 * mis-specialization). What would replace it: measured per-config timings, or a UHD
 * model. Note the kernel's own note says block_n 64 and 128 BOTH match ~peak and 64
 * is more resource-efficient, so preferring the wider tile is not obviously right on
 * this arch -- it is a placeholder awaiting the isolation arms of step 4a-2, and the
 * shipped parity set carries a single block_n so nothing currently depends on it.
 */
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.getIntMetadata(std::string(BLOCK_N_FIELD)));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything launch() needs, resolved once and owning
/// nothing that points back into the MatchContext or BoundTokens it came from.
class PreparedGfx950AttentionDense : public PreparedDispatch
{
public:
    PreparedGfx950AttentionDense(std::unique_ptr<compilation::ICompiledProgram> program,
                                 std::unique_ptr<compilation::IRunnableKernel> kernel,
                                 AttentionDenseBinding binding)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const AttentionDenseBinding& binding() const
    {
        return _binding;
    }

private:
    // The runnable kernel is a VIEW into its program's module; both are held for the
    // plan's lifetime or the kernel dangles.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    AttentionDenseBinding _binding;
};

/**
 * @brief The native dispatch behind this engine's UDD: sizes, prepares and launches.
 *
 * Split per RFC 0017 §8.5: everything graph- or kernel-derived resolves at prepare(),
 * so launch() only resolves device buffers and launches, and nothing mutates once
 * prepared.
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

    /// Zero, and that is a real answer rather than a stub: the kernel's only scratch is
    /// LDS (K_lds + V_lds, sized at build time) plus registers. It allocates no global
    /// scratch, and the 5-slot ABI has no workspace pointer to hand one to.
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

        // Deferred from graph_match: the output tensor's shape is inferred by the
        // frontend and is not reliably populated during matching, so requiring its
        // layout there would decline graphs this engine serves. By prepare() it is
        // real, and O is addressed with the query base and stride verbatim -- a
        // non-BSHD O writes the right bytes to the wrong places, silently.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o) || !hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_dense: the output tensor is not dense BSHD; the kernel "
                "bakes that layout and takes no stride arguments");
        }

        // KernelCompileOptions dereferences the tensor it is handed UNCONDITIONALLY:
        // addDataTypeAndLayoutOptions() calls `tensorAttrs->data_type()` with no null
        // check, and isChannelLastLayout() THROWS for any 4D stride order that is
        // neither NCHW nor NHWC. BSHD attention memory is neither, so passing the real
        // query tensor throws at prepare() time -- and passing `nullptr` segfaults.
        // "Layout-neutral stand-in" means a minimal REAL tensor, not the absence of
        // one.
        //
        // Hence a 1x1x1x1 tensor with NCHW-ordered strides, which classifies cleanly.
        // Safe on this path and only this path: every kernel in this pack is KPACK
        // (kernel_source.kind: rocke, lowered by hkp_pack at build time), and
        // buildIngestorKernelCode deliberately does not consult `options` on the KPACK
        // branch -- a kpack blob's build defines were baked at pack time.
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

        auto code
            = buildIngestorKernelCode(_kernelCompiler, _kpackLoader, context, kernel, options);

        // Geometry, restated from the builder's own helpers, INCLUDING the persistent
        // branch. The arithmetic and its guards live in gfx950AttentionDenseGeometry()
        // so a test can reach them without a device: this correspondence is unchecked
        // by the build, the packer and the validator, and it fails silently rather
        // than loudly. Every term comes from the KMD, so a variant launches the
        // geometry it was actually compiled for. No block_m argument: gfx950 bakes it.
        const auto geometry = gfx950AttentionDenseGeometry(
            kernel.getIntMetadata(std::string(SEQLEN_Q_FIELD)),
            kernel.getIntMetadata(std::string(NUM_QUERY_HEADS_FIELD)),
            kernel.getIntMetadata(std::string(BATCH_FIELD)),
            kernel.getIntMetadata(std::string(PERSISTENT_FIELD)),
            kernel.getIntMetadata(std::string(NUM_PERSISTENT_FIELD)),
            toString(kernel.kernelId));

        code.kernel->setBlockSize(geometry.blockX, 1, 1);
        code.kernel->setGridSize(geometry.gridX, geometry.gridY, geometry.gridZ);

        return std::make_unique<PreparedGfx950AttentionDense>(
            std::move(code.program), std::move(code.kernel), binding);
    }

    /// The ABI is `attention_dense_signature` (kernels/gfx950/attention_dense.py:1888):
    /// `(q_ptr, k_ptr, v_ptr, o_ptr, scale)`, in that order.
    ///
    /// FIVE ARGUMENTS IS A CHOICE, NOT A CEILING. `IRunnableKernel::launch` is a
    /// variadic template sizing its kernarg array from `sizeof...(Args)`, so this
    /// passes five because the shipped variants take five -- not because a sixth is
    /// unrepresentable. The gfx950 signature appends `sink_ptr` when use_sinks,
    /// `cu_seqlens_q`/`cu_seqlens_kv` when varlen, and
    /// `block_tables`/`kv_lens`/`block_table_stride` when paged. This launch is
    /// correct BECAUSE graph_match declines every graph that would need a sixth
    /// argument, and kernelMatches additionally refuses any variant whose metadata
    /// says it was compiled with one. Those declines are what keeps the count at
    /// five; they are load-bearing, not politeness.
    ///
    /// That order is a hand-maintained contract with the Python, unchecked by the
    /// type system.
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

        preparedDense.kernel().launch(
            handle.getStream(), q.ptr, k.ptr, v.ptr, o.ptr, binding.scale);
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

/// This engine's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer to it, but a provider's Container is created and destroyed per handle, so it
/// (and the compiler and loader it holds) must outlive every Container.
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
