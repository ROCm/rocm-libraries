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
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionTiledGeometry.hpp"

/**
 * @file Gfx950AttentionTiledNative.cpp
 * @brief The hipkernel:Gfx950AttentionTiled engine's native half: matching, scoring,
 *        dispatch, and the one function that registers them.
 *
 * The kernel is `rocke/library/kernels/gfx950/attention_tiled_2d.py`'s
 * `build_unified_attention_2d_tiled`, packaged (kind: rocke -> kpack) for gfx950 only.
 * Its applicability rules live only in that Python; this file is where they become
 * enforceable.
 *
 * THE DENSE SIBLING IS A REFERENCE, NOT A TEMPLATE. Both serve `SdpaAttributes` on
 * gfx950 and they diverge in four ways that are silent if copied across:
 *
 *  1. **THIS KERNEL IS STRUCTURALLY PAGED AND VARLEN. There is no dense mode.**
 *     `block_tables_ptr` and `seq_lens_ptr` are unconditionally-declared kernargs, and
 *     `grep -nE "if spec\.(paged|varlen|...)"` over the module returns nothing (with a
 *     positive control confirming `if spec.` matches there, so the zero is real). The
 *     dense pack DECLINES every paged and varlen graph; this one REQUIRES them. A
 *     graph without page tables is not servable here.
 *
 *  2. **SEQUENCE LENGTHS ARE NOT BAKED.** `UnifiedAttention2DTiledSpec` has no
 *     `total_q`, `max_seqlen_q`, `max_seqlen_k` or `batch` field -- enumerated from
 *     `dataclasses.fields()`, against the dense spec which has `seqlen_q`, `seqlen_kv`
 *     and `batch`. Measured consequence: 48 servable corpus shapes resolved to 39
 *     distinct binaries, the nine "missing" ones being shapes that differ only in
 *     sequence length. So `kernelMatches` must NOT compare sequence lengths -- the
 *     exact opposite of the dense matcher's strict-equality rule, and comparing them
 *     would decline graphs a variant serves perfectly.
 *
 *  3. **THE ABI IS A FIXED 18 SLOTS, ALL UNCONDITIONALLY DECLARED**
 *     (attention_tiled_2d.py:1191-1230). The dense ABI is 5 slots conditionally
 *     extended to 6/7/8, which is what makes its declines load-bearing against
 *     reading uninitialised kernarg slots as pointers. Here there is no arity hazard
 *     at all: slots 5 (sink), 8 (alibi) and 9 (qq_bias) always exist. The declines in
 *     this file protect variant COVERAGE -- a graph asking for a capability no shipped
 *     variant BAKES -- which is a genuinely weaker safety argument, stated plainly so
 *     it is not read as the same one.
 *
 *  4. **The grid carries a `+ num_seqs` varlen slack term** and keys grid.x on
 *     num_kv_heads rather than num_query_heads. See Gfx950AttentionTiledGeometry.hpp.
 *
 * Three facts drive almost every check, each a silent wrong answer if missed:
 *
 *  a. **`block_size` is `K.dims[SEQ_AXIS]`.** hipDNN has no page-size scalar (all 41
 *     `SdpaAttributes` fields enumerated), so it is DERIVED: the K/V tensor IS the
 *     paged container `[num_blocks, page_size, num_kv_heads, head_size]`, and the page
 *     table resolves which block, never how large one is. Wrong here means indexing
 *     the KV cache with the wrong stride -- wrong numbers, no fault. The three
 *     concurring sources are in the geometry header.
 *  b. **Q and O are token-major (BSHD) with no stride kernargs**, so a differently
 *     strided tensor is read as if it were this one: in-bounds reads of the wrong
 *     elements. K/V are the paged container instead and follow (a).
 *  c. **hipDNN has no `causal` boolean.** Causality is derived from the deprecated
 *     `causal_mask`/`causal_mask_bottom_right` pair, which take precedence when set,
 *     otherwise from (`left_bound`, `right_bound`, `diagonal_alignment`). Reading only
 *     the deprecated booleans computes "not causal" for every shipped causal bundle;
 *     reading only the bounds misses every model trace. Both spellings occur in this
 *     repo's own corpora.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_attention_tiled.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_attention_tiled.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_tiled.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_tiled.dispatch";

// --- KMD fields the matcher compares. ---
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
// block_size: the KV cache PAGE SIZE, derived from K.dims[SEQ_AXIS]. See fact (a).
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
// num_seqs is baked -- it drives `binary_search_iters`, a compile-time loop trip count
// (attention_tiled_2d.py:1252) -- but it is a CAPACITY BOUND rather than an equality,
// for the same reason `batch` is on the dense engine's default arm. See kernelMatches.
constexpr std::string_view NUM_SEQS_FIELD = "num_seqs";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
// Geometry the dispatcher resolved per shape; prepare() restates the launch from them.
constexpr std::string_view NUM_WARPS_FIELD = "num_warps";
constexpr std::string_view BLOCK_M_PER_WARP_FIELD = "block_m_per_warp";
constexpr std::string_view TILE_SIZE_FIELD = "tile_size";
// --- Capability flags. ---
constexpr std::string_view USE_SINKS_FIELD = "use_sinks";
constexpr std::string_view USE_ALIBI_FIELD = "use_alibi";
// has_softcap and use_qq_bias: SHIPPED AS 0 IN EVERY VARIANT, and that is forced
// rather than chosen. `SdpaAttributes` has no softcap attribute and no query-query
// bias attribute -- all 41 fields enumerated -- so no graph can ever request either,
// and a variant built with one would be structurally unselectable. Eight corpus shapes
// were dropped from the shipping set for exactly this reason. They are compared here
// so such a variant, if one were ever generated by mistake, cannot be chosen by a
// plain graph. Reported at stage 9 as a SCHEMA gap, not an integration gap.
constexpr std::string_view HAS_SOFTCAP_FIELD = "has_softcap";
constexpr std::string_view USE_QQ_BIAS_FIELD = "use_qq_bias";
// The three fp8 flags. Declined in v1; compared so a fp8-built variant cannot be
// selected by a graph that carries no descale tensors.
constexpr std::string_view USE_FP8_MFMA_QK_FIELD = "use_fp8_mfma_qk";
constexpr std::string_view USE_FP8_MFMA_PV_FIELD = "use_fp8_mfma_pv";
constexpr std::string_view USE_REGISTER_PV_FIELD = "use_register_pv";

constexpr std::string_view Q_TOKEN = "gfx950_attention_tiled.q.uid";
constexpr std::string_view K_TOKEN = "gfx950_attention_tiled.k.uid";
constexpr std::string_view V_TOKEN = "gfx950_attention_tiled.v.uid";
constexpr std::string_view O_TOKEN = "gfx950_attention_tiled.o.uid";
constexpr std::string_view PAGE_TABLE_K_TOKEN = "gfx950_attention_tiled.page_table_k.uid";
constexpr std::string_view PAGE_TABLE_V_TOKEN = "gfx950_attention_tiled.page_table_v.uid";
constexpr std::string_view SEQ_LEN_Q_TOKEN = "gfx950_attention_tiled.seq_len_q.uid";
constexpr std::string_view SEQ_LEN_KV_TOKEN = "gfx950_attention_tiled.seq_len_kv.uid";
/// 0 when the graph carries no sink tensor. Slot 5 of the ABI always exists, so a
/// null pointer is passed rather than the argument being omitted.
constexpr std::string_view SINK_TOKEN = "gfx950_attention_tiled.sink.uid";
/// The sliding window graphMatches derived, so kernelMatches does not re-derive it.
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx950_attention_tiled.sliding_window";
/// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_tiled.scale_bits";
/// The paged geometry, derived once in graphMatches from the graph's own tensors.
constexpr std::string_view BLOCK_SIZE_TOKEN = "gfx950_attention_tiled.block_size";
constexpr std::string_view BLOCK_TABLE_STRIDE_TOKEN = "gfx950_attention_tiled.bt_stride";
constexpr std::string_view TOTAL_Q_TOKEN = "gfx950_attention_tiled.total_q";
constexpr std::string_view NUM_SEQS_TOKEN = "gfx950_attention_tiled.num_seqs";

/// hipDNN tensor axes. The LOGICAL order is always (B, H, S, D) regardless of the
/// memory layout, which lives in the strides. For a PAGED K/V the same four axes carry
/// the container's `[num_blocks, page_size, num_kv_heads, head_size]` -- see fact (a).
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_SIZE_AXIS = 3;
constexpr uint32_t SDPA_RANK = 4;
/// The page table is `[num_seqs, max_blocks_per_seq]`.
constexpr uint32_t PAGE_TABLE_RANK = 2;
constexpr uint32_t PAGE_TABLE_SEQ_AXIS = 0;
constexpr uint32_t PAGE_TABLE_BLOCK_AXIS = 1;

/// Unbounded, in the left_bound/right_bound convention shared with the reference
/// executor (GpuRefSdpaFwd.cpp treats a negative bound as "no clamp").
constexpr int64_t UNBOUNDED = -1;

/// head_size values `supports_tiled_2d` admits (attention_tiled_2d.py:934-945): the
/// set, AND the `% 32 == 0` invariant, which the source states as two separate rules.
constexpr int64_t HEAD_SIZE_ALIGNMENT = 32;

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The tensor uids and derived scalars a matched tiled-attention graph binds.
struct AttentionTiledBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t pageTableK = 0;
    int64_t pageTableV = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    /// 0 when the graph carries no sink tensor; the ABI slot still exists.
    int64_t sink = 0;
    int64_t slidingWindow = 0;
    int64_t blockSize = 0;
    int64_t blockTableStride = 0;
    int64_t totalQ = 0;
    int64_t numSeqs = 0;
    float scale = 0.0F;
};

/// The graph facts the matcher and prepare() both need, derived once from the tensors.
struct AttentionTiledProblem
{
    int64_t numSeqs = 0;
    int64_t totalQ = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    int64_t blockSize = 0;
    int64_t blockTableStride = 0;
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
bool isWellFormedOperand(const data_objects::TensorAttributes& tensor, uint32_t expectedRank)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != expectedRank
       || strides->size() != expectedRank)
    {
        return false;
    }

    // Positive extents. Checked explicitly because Python's `%` is sign-following, so
    // a zero or negative extent passes every divisibility rule in the spec -- and
    // num_kv_heads == 0 emits a division by zero into the GQA derivation.
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
 * Applies to Q and O ONLY. K and V are the paged container and have their own
 * geometry check; running this on them would compare against a stride pattern the
 * container does not have.
 *
 * The kernel bakes this layout and takes no stride kernargs, so a differently-strided
 * tensor is read as if it were this one.
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

/**
 * @brief Is this K/V tensor laid out as the paged container the kernel indexes?
 *
 * `[num_blocks, page_size, num_kv_heads, head_size]`, row-major, with head_size the
 * unit-stride axis. Proven from the kernel's own byte strides
 * (attention_tiled_2d.py:1884-1887):
 *
 *     blk = page_size * num_kv_heads * head_size
 *     tok =             num_kv_heads * head_size
 *     head =                           head_size
 *     dim = 1
 *
 * Unit-extent axes exempt, for the same reason as hasBshdStrides.
 */
bool hasPagedKvStrides(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();

    const int64_t pageSize = dims->Get(SEQ_AXIS);
    const int64_t kvHeads = dims->Get(HEAD_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);

    const auto axisOk = [&](uint32_t axis, int64_t expected) {
        return dims->Get(axis) == 1 || strides->Get(axis) == expected;
    };

    return axisOk(BATCH_AXIS, pageSize * kvHeads * headSize) && axisOk(SEQ_AXIS, kvHeads * headSize)
           && axisOk(HEAD_AXIS, headSize) && axisOk(HEAD_SIZE_AXIS, 1);
}

/// Mask classification, mirroring asm_sdpa_engine/plans/SdpaPlanUtils.hpp::getMaskType.
enum class MaskType : int
{
    NO_MASK = 0,
    TOP_LEFT_CAUSAL = 1,
    BOTTOM_RIGHT_CAUSAL = 2,
    SLIDING_WINDOW = 3
};

/// A derived mask, plus the window WIDTH when there is one. The width matters: the
/// spec's `sliding_window` is a compile-time constant that prunes the KV loop, so a
/// windowed graph served by a `sliding_window=0` binary attends the whole causal
/// triangle -- a wrong answer, not a decline.
struct DerivedMask
{
    MaskType type = MaskType::NO_MASK;
    int64_t slidingWindow = 0;
};

/**
 * @brief Which mask the graph is asking for, and how wide.
 *
 * A REAL BOUND WINS OVER THE DEPRECATED BOOLEANS. The booleans only distinguish
 * top-left from bottom-right; they cannot express a window, so a graph that sets one
 * AND carries a bound is asking for a windowed mask and must be reported as such.
 * Returning on the boolean first serves a causal graph with `left_bound = 128` as
 * plain causal, silently discarding the window.
 *
 * BOTH SPELLINGS OCCUR IN THIS REPO. The shipped `quick/SdpaFwd` bundles leave the
 * booleans false and express causality through `left_bound=-1, right_bound=0`; the
 * model traces in `dnn-benchmarking` set `causal_mask: true`. A matcher reading only
 * one convention passes its own suite and mis-serves the other population.
 *
 * The window WIDTH is `left_bound + 1`: hipDNN's left bound counts tokens strictly
 * before the current one, while the spec's `sliding_window` counts the band INCLUDING
 * the current token (matching the kernel's `q-W+1 <= k <= q`). This +1 is the same
 * convention the shape miner applies when reading rocKE's own `window_size` traces.
 */
std::optional<DerivedMask> maskTypeFor(const data_objects::SdpaAttributes& attributes)
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
        if(left < 0)
        {
            // Any other negative is neither "unbounded" nor a legal width. Decline
            // rather than converting it to a nonsense window.
            return std::nullopt;
        }
        return DerivedMask{MaskType::SLIDING_WINDOW, left + 1};
    }

    if(topLeftDeprecated)
    {
        return DerivedMask{MaskType::TOP_LEFT_CAUSAL, 0};
    }
    if(bottomRightDeprecated)
    {
        return DerivedMask{MaskType::BOTTOM_RIGHT_CAUSAL, 0};
    }

    if(right == UNBOUNDED)
    {
        return DerivedMask{MaskType::NO_MASK, 0};
    }
    if(right == 0)
    {
        return attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT
                   ? DerivedMask{MaskType::BOTTOM_RIGHT_CAUSAL, 0}
                   : DerivedMask{MaskType::TOP_LEFT_CAUSAL, 0};
    }
    // A finite positive right bound is a forward-looking band this kernel's causal
    // clamp cannot express.
    return std::nullopt;
}

/// The kernel's dtype spelling for a graph dtype, or nullopt for one it cannot be
/// built for. One vocabulary: the KMD carries the hipDNN enum name, so that is what the
/// comparison uses. `supports_tiled_2d` gates dtype to fp16/bf16
/// (attention_tiled_2d.py:932).
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

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog and skips EVERY
 *          remaining pack, not just this one.
 */
std::optional<BoundTokens> gfx950AttentionTiledGraphMatches(const MatchContext& context)
{
    // --- 1. Node shape. One SDPA-forward node; this engine serves a whole graph.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // --- 2. Operands. Q/K/V/O plus the paged and varlen tensors this kernel REQUIRES.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }

    // PAGED IS MANDATORY, not optional -- the inverse of the dense pack. Both tables
    // must be present: the kernel has ONE `block_tables_ptr` serving both caches, so a
    // graph offering two different tables is unservable.
    if(!attributes.page_table_k_tensor_uid().has_value()
       || !attributes.page_table_v_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    const auto* pageTableK = findTensor(context, attributes.page_table_k_tensor_uid().value());
    const auto* pageTableV = findTensor(context, attributes.page_table_v_tensor_uid().value());
    if(pageTableK == nullptr || pageTableV == nullptr)
    {
        return std::nullopt;
    }

    // VARLEN IS MANDATORY too: `query_start_len_ptr` (cu_q) and `seq_lens_ptr` are
    // unconditional kernargs, and the kernel's block->sequence mapping is a binary
    // search over cu_q. There is no path that treats a batch as uniform.
    if(!attributes.seq_len_q_tensor_uid().has_value()
       || !attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    const auto* seqLenQ = findTensor(context, attributes.seq_len_q_tensor_uid().value());
    const auto* seqLenKv = findTensor(context, attributes.seq_len_kv_tensor_uid().value());
    if(seqLenQ == nullptr || seqLenKv == nullptr)
    {
        return std::nullopt;
    }

    // --- 3. Total predicates, before anything indexes an axis. O is included for
    // well-formedness only; its LAYOUT is checked in prepare(), because the frontend
    // infers its shape and it is not reliably populated at match time.
    if(!isWellFormedOperand(*q, SDPA_RANK) || !isWellFormedOperand(*k, SDPA_RANK)
       || !isWellFormedOperand(*v, SDPA_RANK) || !isWellFormedOperand(*o, SDPA_RANK))
    {
        return std::nullopt;
    }
    if(!isWellFormedOperand(*pageTableK, PAGE_TABLE_RANK)
       || !isWellFormedOperand(*pageTableV, PAGE_TABLE_RANK))
    {
        return std::nullopt;
    }

    // --- 4. Layout. Tier 1: the failure is wrong elements in bounds, no fault.
    // Q uses the token-major form; K/V use the paged-container form. They are DIFFERENT
    // stride patterns and running one check on the other's tensor is itself a defect.
    if(!hasBshdStrides(*q) || !hasPagedKvStrides(*k) || !hasPagedKvStrides(*v))
    {
        return std::nullopt;
    }

    // --- 5. The paged geometry. THE HIGHEST-RISK DERIVATION IN THIS FILE.
    // block_size = K.dims[SEQ_AXIS]; see fact (a) and the geometry header's three
    // concurring sources.
    const auto pagedGeometry = gfx950TiledPagedKvGeometry(
        k->dims()->Get(SEQ_AXIS), pageTableK->dims()->Get(PAGE_TABLE_BLOCK_AXIS));
    // A page size outside {16,32,64} is DECLINED, never rounded or clamped
    // (attention_tiled_2d.py:946).
    if(!gfx950TiledBlockSizeIsLegal(pagedGeometry.blockSize))
    {
        return std::nullopt;
    }
    // One block table serving both caches: the two must be shape-identical, or the
    // single `block_tables_ptr` cannot represent both.
    if(pageTableK->dims()->Get(PAGE_TABLE_SEQ_AXIS) != pageTableV->dims()->Get(PAGE_TABLE_SEQ_AXIS)
       || pageTableK->dims()->Get(PAGE_TABLE_BLOCK_AXIS)
              != pageTableV->dims()->Get(PAGE_TABLE_BLOCK_AXIS))
    {
        return std::nullopt;
    }
    // The block table is indexed `[seq_idx * bt_stride + tile_idx]` with a dense i32
    // row, so the row stride must BE the inner extent. A padded table would need a
    // stride kernarg the host computes from `.stride(0)`; we have no way to see that
    // padding from the graph, so a non-dense table is declined rather than guessed.
    if(pageTableK->strides()->Get(PAGE_TABLE_SEQ_AXIS) != pagedGeometry.blockTableStride
       || pageTableK->strides()->Get(PAGE_TABLE_BLOCK_AXIS) != 1)
    {
        return std::nullopt;
    }

    // --- 6. Cross-tensor consistency.
    AttentionTiledProblem problem;
    problem.numQueryHeads = q->dims()->Get(HEAD_AXIS);
    problem.headSize = q->dims()->Get(HEAD_SIZE_AXIS);
    problem.numKvHeads = k->dims()->Get(HEAD_AXIS);
    problem.dataType = q->data_type();
    problem.blockSize = pagedGeometry.blockSize;
    problem.blockTableStride = pagedGeometry.blockTableStride;
    // num_seqs is the page table's own outer extent -- the authoritative statement of
    // how many sequences this launch covers, and what `block_table_max_idx` is sized
    // against in the kernel (attention_tiled_2d.py:1987).
    problem.numSeqs = pageTableK->dims()->Get(PAGE_TABLE_SEQ_AXIS);
    // total_q is the flattened query-row count. Q is `[B, H, S, D]` logically and
    // token-major physically, so the row count is B*S -- for a varlen batch the
    // frontend presents the packed rows with B == 1.
    problem.totalQ = q->dims()->Get(BATCH_AXIS) * q->dims()->Get(SEQ_AXIS);

    // One dtype across every value operand: the builder takes a single `spec.dtype`
    // and types the q/k/v/o pointers with it.
    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    if(!supportedDataTypeName(problem.dataType).has_value())
    {
        return std::nullopt;
    }

    // V shares K's base, stride and block table in the builder, so it must share K's
    // shape exactly.
    if(v->dims()->Get(BATCH_AXIS) != k->dims()->Get(BATCH_AXIS)
       || v->dims()->Get(HEAD_AXIS) != problem.numKvHeads
       || v->dims()->Get(SEQ_AXIS) != problem.blockSize
       || v->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }
    // The kernel has ONE head_size, while hipDNN permits D_qk != D_v.
    if(k->dims()->Get(HEAD_SIZE_AXIS) != problem.headSize)
    {
        return std::nullopt;
    }

    // GQA: the kernel derives its group size by integer division, so a non-divisible
    // pair silently drops heads, and `supports_tiled_2d` bounds the ratio at 16
    // (attention_tiled_2d.py:952).
    if(problem.numKvHeads <= 0 || problem.numQueryHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }
    const int64_t numQueriesPerKv = problem.numQueryHeads / problem.numKvHeads;
    if(numQueriesPerKv < 1 || numQueriesPerKv > 16)
    {
        return std::nullopt;
    }

    // head_size in {64,128,256} AND divisible by 32 -- the source states these as two
    // separate rules (attention_tiled_2d.py:934-945), so both are mirrored rather than
    // one being assumed to imply the other.
    if(problem.headSize != 64 && problem.headSize != 128 && problem.headSize != 256)
    {
        return std::nullopt;
    }
    if(problem.headSize % HEAD_SIZE_ALIGNMENT != 0)
    {
        return std::nullopt;
    }

    // The wave-uniform block-table invariant (attention_tiled_2d.py:1015-1024). Both
    // terms come from the GRAPH, so this is genuinely graph-derivable despite reading
    // like a tuning rule: a violation makes the per-lane block-table lookup
    // lane-divergent and the async DMA under-fills its LDS slab -- wrong numbers, not
    // a fault. It never fires on the legal head_size/block_size cross-product, but it
    // is one graph field away from firing and nothing else would catch it.
    if((512 / problem.headSize) > problem.blockSize)
    {
        return std::nullopt;
    }

    // --- 7. The mask, and the window width. See maskTypeFor.
    const auto mask = maskTypeFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }
    switch(mask->type)
    {
    case MaskType::NO_MASK:
    case MaskType::TOP_LEFT_CAUSAL:
    case MaskType::SLIDING_WINDOW:
        break;
    case MaskType::BOTTOM_RIGHT_CAUSAL:
    default:
        // The kernel's causal clamp derives its KV-loop bound from the query position
        // within its sequence, using `context_len = seq_len - cur_batch_q_len`
        // (attention_tiled_2d.py:1262-1263) -- which IS the bottom-right offset. So
        // bottom-right is the kernel's NATIVE convention for a paged decode, where
        // Sq < Skv is the normal case.
        //
        // It is nonetheless declined in v1, and the reason is the contract ambiguity
        // the dense sibling surfaced and handed back unresolved: hipDNN's canonical
        // rule says the deprecated boolean wins and the enum is ignored, while the
        // PyTorch reference checks the enum first. Serving BOTTOM_RIGHT here before
        // that is settled would pick a side silently on exactly the graphs where the
        // two conventions disagree. Declining is the debuggable failure.
        //
        // `default` is required by -Wswitch-default and doubles as the safe verdict
        // for any mask kind added later: an unrecognised mask is declined, never
        // served as if it were dense.
        return std::nullopt;
    }

    // --- 8. Every optional attribute this kernel cannot honour, declined explicitly.
    // Worked from sdpa_attributes.fbs field by field: an UNCHECKED field is accepted
    // and then silently not performed. Several features appear under more than one
    // spelling, and rejecting only one admits graphs carrying the rest.

    // Additive attention bias. NOT the same thing as the kernel's `qq_bias`, which is
    // a query-QUERY bias with no schema field at all -- rejecting one does not reject
    // the other, and conflating them would admit graphs carrying a bias the kernel
    // silently ignores.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Device-resident scale: the ABI takes `scale` as an f32 KERNARG (slot 11), so
    // there is no pointer slot for a scale tensor.
    if(attributes.scale_tensor_uid().has_value())
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
    // Block-sparse: no code path in the body.
    if(attributes.block_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // FP8 quantization: six scale UIDs plus two amax outputs. The kernel HAS an fp8
    // path (`kv_storage_dtype="fp8e4m3"` plus `use_fp8`), declined in v1 because no
    // shipped variant bakes it and three of the four builder-body raises are on that
    // path.
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
    // spellings of one request; `max` and `sum_exp` are two more outputs the 2D kernel
    // never writes (the 3D split-KV kernel does, and is a separate engine).
    // generate_stats is optional<bool>, so an explicit `false` is fine and only a
    // `true` is declined.
    if(attributes.stats_tensor_uid().has_value() || attributes.max_tensor_uid().has_value()
       || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value()
       || (attributes.generate_stats().has_value() && attributes.generate_stats().value()))
    {
        return std::nullopt;
    }
    // ALiBi: the kernel HAS a path (`use_alibi`, `alibi_slopes_ptr` at slot 8), and
    // five shipped variants bake it. But the slopes arrive as a TENSOR the kernel
    // reads per head, and `alibi_mask` is a bare bool with no accompanying slopes UID
    // anywhere in the schema -- so a graph setting it cannot say WHAT slopes it wants,
    // and inventing them would be the "do not invent a default for a scalar the graph
    // did not supply" trap with a whole tensor. Declined until the schema can express
    // the slopes; the variants that bake it are then unreachable and reported as such.
    if(attributes.alibi_mask())
    {
        return std::nullopt;
    }
    // padding_mask: the kernel handles per-sequence padding intrinsically via cu_q and
    // its early-return, so an explicit padding mask is a second, different request it
    // has no path for.
    if(attributes.padding_mask())
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
    // provider; COMPOSITE asks for a multi-kernel decomposition this unified kernel is
    // not. UNIFIED is exactly what this kernel is, so it is accepted.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO
       && attributes.implementation() != data_objects::AttentionImplementation::UNIFIED)
    {
        return std::nullopt;
    }
    // max_seq_len_kv: the graph's own statement of its paged KV bound. Where present it
    // must agree with the geometry derived above -- free redundancy on the single most
    // dangerous row in the integration. A disagreement means one of the two readings is
    // wrong, so decline rather than pick a side.
    if(attributes.max_seq_len_kv().has_value())
    {
        const int64_t declaredMaxKv = attributes.max_seq_len_kv().value();
        const int64_t tableCapacity
            = pageTableK->dims()->Get(PAGE_TABLE_BLOCK_AXIS) * problem.blockSize;
        if(declaredMaxKv <= 0 || declaredMaxKv > tableCapacity)
        {
            return std::nullopt;
        }
    }

    // The softmax scale is a REQUIRED launch argument with no default. The schema marks
    // it optional and the mathematically obvious 1/sqrt(D) would silently override
    // whatever the frontend's omission meant. Require presence.
    if(!attributes.attn_scale_value().has_value())
    {
        return std::nullopt;
    }
    const float scale = attributes.attn_scale_value().value();

    // Sinks are SERVED -- unlike the dense sibling, which declines them for want of a
    // reference. Decision D1 makes rocKE's `ref_paged_attn` this engine's numeric
    // oracle, and it implements sinks directly. Slot 5 exists either way; a graph
    // without sinks passes a null pointer.
    const int64_t sinkUid = attributes.sink_token_tensor_uid().has_value()
                                ? attributes.sink_token_tensor_uid().value()
                                : 0;
    if(sinkUid != 0 && findTensor(context, sinkUid) == nullptr)
    {
        return std::nullopt;
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(PAGE_TABLE_K_TOKEN)] = attributes.page_table_k_tensor_uid().value();
    bound[std::string(PAGE_TABLE_V_TOKEN)] = attributes.page_table_v_tensor_uid().value();
    bound[std::string(SEQ_LEN_Q_TOKEN)] = attributes.seq_len_q_tensor_uid().value();
    bound[std::string(SEQ_LEN_KV_TOKEN)] = attributes.seq_len_kv_tensor_uid().value();
    bound[std::string(SINK_TOKEN)] = sinkUid;
    bound[std::string(SLIDING_WINDOW_TOKEN)] = mask->slidingWindow;
    bound[std::string(BLOCK_SIZE_TOKEN)] = problem.blockSize;
    bound[std::string(BLOCK_TABLE_STRIDE_TOKEN)] = problem.blockTableStride;
    bound[std::string(TOTAL_Q_TOKEN)] = problem.totalQ;
    bound[std::string(NUM_SEQS_TOKEN)] = problem.numSeqs;
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
AttentionTiledBinding attentionTiledBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "gfx950 attention_tiled dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    AttentionTiledBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.pageTableK = read(PAGE_TABLE_K_TOKEN);
    binding.pageTableV = read(PAGE_TABLE_V_TOKEN);
    binding.seqLenQ = read(SEQ_LEN_Q_TOKEN);
    binding.seqLenKv = read(SEQ_LEN_KV_TOKEN);
    binding.sink = read(SINK_TOKEN);
    binding.slidingWindow = read(SLIDING_WINDOW_TOKEN);
    binding.blockSize = read(BLOCK_SIZE_TOKEN);
    binding.blockTableStride = read(BLOCK_TABLE_STRIDE_TOKEN);
    binding.totalQ = read(TOTAL_Q_TOKEN);
    binding.numSeqs = read(NUM_SEQS_TOKEN);

    const auto scaleBits = static_cast<int32_t>(read(SCALE_BITS_TOKEN));
    std::memcpy(&binding.scale, &scaleBits, sizeof(binding.scale));
    return binding;
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit?
 *
 * **SEQUENCE LENGTHS ARE DELIBERATELY NOT COMPARED, and that is the single biggest
 * difference from the dense matcher.** `UnifiedAttention2DTiledSpec` has no `total_q`,
 * `max_seqlen_q`, `max_seqlen_k` or `batch` field -- the tiled kernel generalises over
 * sequence length at runtime (the block->sequence mapping is a runtime binary search
 * over cu_q, and the KV loop bound is read from `seq_lens_ptr`). Measured: 48 servable
 * corpus shapes resolve to 39 distinct binaries, the difference being shapes that
 * differ only in sequence length. Comparing seqlen here would decline graphs the
 * variant serves perfectly, for a field the binary does not contain.
 *
 * What IS compared: everything the binary genuinely bakes -- dtype, head_size, the
 * page size, both head counts, the window width, and every capability flag.
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

    const auto dataTypeName = supportedDataTypeName(q->data_type());
    if(!dataTypeName.has_value()
       || kernel.getStringMetadata(std::string(DTYPE_FIELD)) != *dataTypeName)
    {
        return false;
    }

    const auto intField
        = [&kernel](std::string_view field) { return kernel.getIntMetadata(std::string(field)); };

    // Baked shape. block_size is the PAGE SIZE -- the axis the container's stride
    // arithmetic is built on -- so a mismatch indexes the KV cache wrongly.
    if(intField(HEAD_SIZE_FIELD) != q->dims()->Get(HEAD_SIZE_AXIS)
       || intField(BLOCK_SIZE_FIELD) != k->dims()->Get(SEQ_AXIS)
       || intField(NUM_QUERY_HEADS_FIELD) != q->dims()->Get(HEAD_AXIS)
       || intField(NUM_KV_HEADS_FIELD) != k->dims()->Get(HEAD_AXIS))
    {
        return false;
    }

    // NUM_SEQS IS A CAPACITY BOUND, NOT AN EQUALITY, and the asymmetry is deliberate.
    // It reaches the binary in exactly one place: `spec.binary_search_iters`, the
    // compile-time trip count of the block->sequence binary search
    // (attention_tiled_2d.py:1252). That search needs ENOUGH iterations to resolve
    // `num_seqs` candidates -- more is harmless, fewer resolves the wrong sequence and
    // reads another sequence's KV. So a binary compiled for N sequences serves any
    // batch <= N correctly.
    //
    // THE BOUND HAS A PARTNER OBLIGATION IN prepare(), and without it this widening
    // would be a silent wrong answer: the grid's `+ num_seqs` slack term must come
    // from the GRAPH's sequence count, not the descriptor's. Launching a variant's
    // larger slack for a smaller graph adds CTAs whose binary search resolves past the
    // end of cu_q.
    const auto graphNumSeqs = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, NUM_SEQS_TOKEN);
    if(!graphNumSeqs.has_value() || *graphNumSeqs <= 0 || *graphNumSeqs > intField(NUM_SEQS_FIELD))
    {
        return false;
    }

    // The window the graph derived to, against the width this variant was compiled
    // for. `sliding_window` is a compile-time KV-loop bound, so 0 and 256 are different
    // binaries and a windowed graph on a 0 binary attends the whole triangle.
    const auto window = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, SLIDING_WINDOW_TOKEN);
    if(!window.has_value() || intField(SLIDING_WINDOW_FIELD) != *window)
    {
        return false;
    }

    // The graph's sink request against the variant's baked flag, in BOTH directions: a
    // sink graph needs a sink binary (the epilogue's denominator differs), and a
    // sink binary must not serve a plain graph (it would read the null slot-5 pointer).
    const auto sink = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, SINK_TOKEN);
    if(!sink.has_value() || (intField(USE_SINKS_FIELD) != 0) != (*sink != 0))
    {
        return false;
    }

    // ABSENT MEANS NOT-BUILT-WITH-IT, which is why this reads through tryGetMetadata
    // rather than getIntMetadata. getIntMetadata THROWS on a missing field -- it does
    // not default -- so comparing a field a descriptor does not carry would turn a
    // routine non-match into an exception escaping the matcher.
    const auto featureIsSet = [&kernel](std::string_view field) {
        const auto value = kernel.tryGetMetadata(std::string(field));
        if(!value.has_value())
        {
            return false;
        }
        const auto* held = std::get_if<int64_t>(&*value);
        return held != nullptr && *held != 0;
    };

    // A variant compiled FOR a declined capability must never be selected.
    //
    // has_softcap and use_qq_bias are the sharp ones: no graph can request either
    // (no schema field exists), so such a variant is unreachable by construction and
    // this comparison is what keeps it from being reached by ACCIDENT -- matching a
    // plain graph and computing a softcapped or bias-added result that nothing asked
    // for. The shipping corpus already excludes them; this is the second layer.
    //
    // use_alibi is declined at the graph level too (no slopes UID in the schema), so
    // the five variants that bake it are currently unreachable and reported as such.
    if(featureIsSet(HAS_SOFTCAP_FIELD) || featureIsSet(USE_QQ_BIAS_FIELD)
       || featureIsSet(USE_ALIBI_FIELD) || featureIsSet(USE_FP8_MFMA_QK_FIELD)
       || featureIsSet(USE_FP8_MFMA_PV_FIELD) || featureIsSet(USE_REGISTER_PV_FIELD))
    {
        return false;
    }

    // Geometry self-consistency: refuse a descriptor naming a binary the builder would
    // not emit (attention_tiled_2d.py:958-975). A descriptor violating these describes
    // no compiled artifact, so it cannot be a real candidate.
    const int64_t numWarps = intField(NUM_WARPS_FIELD);
    const int64_t blockMPerWarp = intField(BLOCK_M_PER_WARP_FIELD);
    if(!gfx950TiledNumWarpsIsLegal(numWarps) || (blockMPerWarp != 16 && blockMPerWarp != 32))
    {
        return false;
    }
    if(blockMPerWarp == 32 && numWarps == 8)
    {
        return false;
    }
    // tile_size must be a positive multiple of block_size, and must carry at least the
    // async-DMA payload the CTA issues (attention_tiled_2d.py:993-1010).
    const int64_t tileSize = intField(TILE_SIZE_FIELD);
    const int64_t blockSize = intField(BLOCK_SIZE_FIELD);
    if(tileSize <= 0 || blockSize <= 0 || tileSize % blockSize != 0)
    {
        return false;
    }
    if(tileSize * intField(HEAD_SIZE_FIELD) < numWarps * GFX950_TILED_WAVE_LANES * 8)
    {
        return false;
    }
    return true;
}

/**
 * @brief Ranks the candidates that survived kernelMatches. Higher wins.
 *
 * Every shape and capability field is pinned above, so the free axis is `tile_size`:
 * the KV tile the kernel stages per iteration. A wider tile amortises the per-tile
 * barrier, the LDS publication and -- specific to this kernel -- the block-table
 * indirection, since one lookup covers more tokens. So it is ranked higher.
 *
 * `block_size` is deliberately NOT the ranked field even though it looks like the
 * obvious tile knob: it is a GRAPH dimension (the caller's KV cache page size), pinned
 * by equality above. Ranking on it would be ranking on something the graph fixes.
 *
 * This is a HEURISTIC placeholder, deliberately ranking one real knob rather than
 * returning a constant (which would make ranking arbitrary and hide
 * mis-specialization). What would replace it: measured per-config timings from the
 * step-4a-2 isolation arms, or a UHD model.
 *
 * WHAT THIS RETURNS ON THE SHIPPED SET, STATED PLAINLY: the v1 set is dispatcher
 * parity, so each graph resolves exactly one variant and no ranking is ever evaluated.
 * This arms the gate for the first tuning arm rather than defending anything today,
 * and it stops being merely inert the moment a second tile_size ships for one matcher
 * tuple -- at which point it should be re-checked against a measurement.
 */
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.getIntMetadata(std::string(TILE_SIZE_FIELD)));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The compiled kernel plus everything launch() needs, resolved once and owning
/// nothing that points back into the MatchContext or BoundTokens it came from.
class PreparedGfx950AttentionTiled : public PreparedDispatch
{
public:
    PreparedGfx950AttentionTiled(std::unique_ptr<compilation::ICompiledProgram> program,
                                 std::unique_ptr<compilation::IRunnableKernel> kernel,
                                 AttentionTiledBinding binding)
        : _program(std::move(program))
        , _kernel(std::move(kernel))
        , _binding(binding)
    {
    }

    const compilation::IRunnableKernel& kernel() const
    {
        return *_kernel;
    }

    const AttentionTiledBinding& binding() const
    {
        return _binding;
    }

private:
    // The runnable kernel is a VIEW into its program's module; both are held for the
    // plan's lifetime or the kernel dangles.
    std::unique_ptr<compilation::ICompiledProgram> _program;
    std::unique_ptr<compilation::IRunnableKernel> _kernel;
    AttentionTiledBinding _binding;
};

/**
 * @brief The native dispatch behind this engine's UDD: sizes, prepares and launches.
 */
class Gfx950AttentionTiledDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    Gfx950AttentionTiledDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                        const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// Zero, and that is a real answer rather than a stub: the 2D kernel's only scratch
    /// is LDS (K/V slabs plus the softmax state, all sized at build time) plus
    /// registers, and none of the 18 ABI slots is a workspace pointer.
    ///
    /// This is the property that made 2D and 3D separate ENGINES rather than two packs.
    /// The 3D split-KV kernel writes three f32 segment buffers (0.5-8.1 MiB measured)
    /// that rocKE allocates from a Python WorkspacePool a C-ABI engine cannot reach, so
    /// it needs a real answer here. Workspace policy is an engine-level property, and
    /// mixing them would make this function's answer depend on which pack won.
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
        const auto binding = attentionTiledBinding(bound);

        // Deferred from graph_match: the output tensor's shape is inferred by the
        // frontend and is not reliably populated during matching, so requiring its
        // layout there would decline graphs this engine serves. By prepare() it is
        // real, and O is addressed with the query base and stride verbatim -- a
        // non-BSHD O writes the right bytes to the wrong places, silently.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o, SDPA_RANK) || !hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_tiled: the output tensor is not dense BSHD; the kernel "
                "bakes that layout and takes no stride arguments");
        }

        // KernelCompileOptions dereferences the tensor it is handed UNCONDITIONALLY:
        // addDataTypeAndLayoutOptions() calls `tensorAttrs->data_type()` with no null
        // check, and isChannelLastLayout() THROWS for any 4D stride order that is
        // neither NCHW nor NHWC. Neither BSHD attention memory nor the paged KV
        // container is either, so passing a real operand throws at prepare() time --
        // and passing `nullptr` segfaults. "Layout-neutral stand-in" means a minimal
        // REAL tensor, not the absence of one.
        //
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

        // TOTAL_Q AND NUM_SEQS COME FROM THE GRAPH, and that is structural rather than
        // a choice: the spec has no field for either extent (num_seqs is a spec field
        // only as a binary-search trip count). This is also the partner half of
        // kernelMatches' `graphNumSeqs <= kernel.num_seqs` bound -- taking the slack
        // term from the descriptor instead would add CTAs whose binary search resolves
        // past the end of cu_q.
        //
        // The geometry and its guards live in gfx950AttentionTiledGeometry() so a test
        // can reach them without a device: this correspondence is unchecked by the
        // build, the packer and the validator, and it fails silently rather than
        // loudly.
        const auto geometry = gfx950AttentionTiledGeometry(
            kernel.getIntMetadata(std::string(NUM_KV_HEADS_FIELD)),
            kernel.getIntMetadata(std::string(NUM_QUERY_HEADS_FIELD)),
            binding.totalQ,
            binding.numSeqs,
            kernel.getIntMetadata(std::string(NUM_WARPS_FIELD)),
            kernel.getIntMetadata(std::string(BLOCK_M_PER_WARP_FIELD)),
            toString(kernel.kernelId));

        code.kernel->setBlockSize(geometry.blockX, 1, 1);
        code.kernel->setGridSize(geometry.gridX, geometry.gridY, geometry.gridZ);

        return std::make_unique<PreparedGfx950AttentionTiled>(
            std::move(code.program), std::move(code.kernel), binding);
    }

    /// The ABI is the 18 `b.param(...)` declarations at
    /// kernels/gfx950/attention_tiled_2d.py:1191-1230, in this exact order:
    ///
    ///   output, query, key_cache, value_cache, sink, block_tables, seq_lens,
    ///   alibi_slopes, qq_bias, query_start_len,
    ///   scale, k_scale, v_scale, out_scale, softcap, num_seqs, block_table_stride,
    ///   qq_bias_stride_0
    ///
    /// ALL EIGHTEEN ARE UNCONDITIONALLY DECLARED -- there is no reduced-arity
    /// signature, unlike the dense sibling whose 5 slots grow to 6/7/8 per feature. So
    /// slots this engine does not use are passed as null pointers or inert scalars
    /// rather than omitted, and there is no argument-count hazard to defend against.
    ///
    /// `k_scale`/`v_scale`/`out_scale` are 1.0 because fp8 is declined: they are the
    /// dequantisation factors for an fp8 KV cache, and 1.0 is the identity the bf16/fp16
    /// path expects. `softcap` is 0.0 and `qq_bias_stride_0` is 0 because no graph can
    /// request either (no schema field), so every shipped variant bakes them off and
    /// the kernel never reads the values.
    ///
    /// That order is a hand-maintained contract with the Python, unchecked by the type
    /// system.
    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedTiled = dynamic_cast<const PreparedGfx950AttentionTiled&>(prepared);
        const auto& binding = preparedTiled.binding();

        const auto find = [&](int64_t uid) {
            return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers);
        };

        const auto q = find(binding.q);
        const auto k = find(binding.k);
        const auto v = find(binding.v);
        const auto o = find(binding.o);
        const auto blockTables = find(binding.pageTableK);
        const auto seqLens = find(binding.seqLenKv);
        const auto cuSeqLensQ = find(binding.seqLenQ);

        // Slot 5 always exists; a graph without sinks passes null. graph_match bound 0
        // for that case and kernelMatches refused any sink-baked variant, so the kernel
        // compiled for this launch never dereferences it.
        void* sinkPtr = nullptr;
        if(binding.sink != 0)
        {
            sinkPtr = find(binding.sink).ptr;
        }

        // Slots 8 and 9. No graph can request ALiBi slopes or a query-query bias (no
        // schema field for either), kernelMatches refuses any variant that bakes them,
        // so the compiled kernel never reads these.
        void* alibiSlopesPtr = nullptr;
        void* qqBiasPtr = nullptr;

        preparedTiled.kernel().launch(handle.getStream(),
                                      // pointers, slots 1-10
                                      o.ptr,
                                      q.ptr,
                                      k.ptr,
                                      v.ptr,
                                      sinkPtr,
                                      blockTables.ptr,
                                      seqLens.ptr,
                                      alibiSlopesPtr,
                                      qqBiasPtr,
                                      cuSeqLensQ.ptr,
                                      // scalars, slots 11-18
                                      binding.scale,
                                      1.0F,
                                      1.0F,
                                      1.0F,
                                      0.0F,
                                      static_cast<int32_t>(binding.numSeqs),
                                      static_cast<int32_t>(binding.blockTableStride),
                                      static_cast<int32_t>(0));
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& gfx950AttentionTiledKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx950AttentionTiledModuleCache()
{
    gfx950AttentionTiledKpackModuleCache().clear();
}

namespace
{

/// This engine's dispatch handler, process-lifetime: the registry holds a non-owning
/// pointer to it, but a provider's Container is created and destroyed per handle, so it
/// (and the compiler and loader it holds) must outlive every Container.
const Gfx950AttentionTiledDispatchHandler& gfx950AttentionTiledDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx950AttentionTiledKpackModuleCache());
    static const Gfx950AttentionTiledDispatchHandler s_dispatchHandler(s_kernelCompiler,
                                                                       s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerGfx950AttentionTiledSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx950AttentionTiledGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &gfx950AttentionTiledDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
