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
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGraphMatch.hpp"

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
// on the catalog key and the loader drops one. `num_seqs` is declared but not read: the kernel
// takes the batch as a kernarg and derives its cu_q search trip count from it, so no batch is
// baked.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view MASK_TYPE_FIELD = "mask_type";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view TILE_SIZE_FIELD = "tile_size";
constexpr std::string_view NUM_WARPS_FIELD = "num_warps";
constexpr std::string_view BLOCK_M_PER_WARP_FIELD = "block_m_per_warp";
constexpr std::string_view USE_MFMA_32X32_FIELD = "use_mfma_32x32";
constexpr std::string_view USE_TRANSPOSED_QK_32X32_FIELD = "use_transposed_qk_32x32";
/// 1 for an LSE-on kernel, which also writes the stats output head-major ([B, Hq, Sq, 1]).
constexpr std::string_view EMIT_LSE_FIELD = "emit_lse";

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
/// The stats output's uid, or NO_STATS for a graph that asks for none.
constexpr std::string_view STATS_TOKEN = "gfx950_attention_unified_2d.stats.uid";

// ---------------------------------------------------------------------------
// Matching helpers
// ---------------------------------------------------------------------------

using unified_attention::findTensor;
using unified_attention::hasBshdStrides;
using unified_attention::integerMetadata;
using unified_attention::isWellFormedOperand;
using unified_attention::MaskType;
using unified_attention::problemFor;
using unified_attention::sdpaNode;
using unified_attention::stringMetadata;
using unified_attention::supportedDataTypeName;
using unified_attention::UnifiedAttentionBinding;
using unified_attention::UnifiedAttentionProblem;

// ---------------------------------------------------------------------------
// Decline logging
//
// hipDNN reports a declined graph to its caller only as "No engine configurations available
// for the graph", so each decline logs its cause at INFO. graph_match runs on every catalog
// miss, so every piece of that work sits inside HIPDNN_PLUGIN_LOG_INFO's argument: the macro
// tests the level first, and with INFO off nothing below is called.
// ---------------------------------------------------------------------------

/// The head of this engine's decline lines (unified_attention::DeclinedBy).
struct Declined : unified_attention::DeclinedBy
{
    explicit Declined(std::string_view cause)
        : DeclinedBy{GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME, cause}
    {
    }
};

constexpr unified_attention::BoundTokenNames TOKEN_NAMES{Q_TOKEN,
                                                         K_TOKEN,
                                                         V_TOKEN,
                                                         O_TOKEN,
                                                         MASK_TOKEN,
                                                         SLIDING_WINDOW_TOKEN,
                                                         SCALE_BITS_TOKEN,
                                                         STATS_TOKEN,
                                                         "gfx950 attention_unified_2d"};

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * The shared unified-attention match (UnifiedAttentionGraphMatch.hpp) around this engine's
 * rules: GQA groups up to 16, head_size 64 or 128, q_len <= kv_len, and decode handed to
 * Unified3D. Each decline of a graph with an SDPA-forward node logs its cause at INFO (see
 * Declined); any other graph declines silently.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog and skips EVERY
 *          remaining pack, not just this one.
 */
std::optional<BoundTokens> gfx950AttentionUnified2DGraphMatches(const MatchContext& context)
{
    const auto operands
        = unified_attention::matchOperands(context, GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME);
    if(!operands.has_value())
    {
        return std::nullopt;
    }
    const auto& problem = operands->problem;

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

    // head_size 64 or 128. D256 prefill is not shipped (deferred); bf16 D256 decode is
    // Unified3D's.
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

    // Decode (one query token per sequence) is hipkernel:Gfx950AttentionUnified3D's wherever
    // its split-KV kernel can be built: there the 2D kernel runs one under-filled Q block
    // per sequence while the 3D kernel splits the keys across CTAs. No select_path crossover
    // is replayed: the boundary is this static rule.
    if(problem.seqLenQ == 1
       && gfx950Unified3DServesGqaGroup(problem.numQueryHeads, problem.numKvHeads))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"decode"} << "decode (seqlen_q 1) with a GQA group of "
                                                  << problem.numQueryHeads / problem.numKvHeads
                                                  << " is served by the split-KV Unified3D engine");
        return std::nullopt;
    }

    const auto matched = unified_attention::matchAttributes(
        context, *operands->attributes, problem, GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME);
    if(!matched.has_value())
    {
        return std::nullopt;
    }
    // All three mask kinds are served, each by its own kernel: top-left and bottom-right
    // coincide only at Sq == Skv, but they are distinct kernel bodies and the catalog carries
    // both.
    return unified_attention::bindGraph(
        *operands->attributes, matched->mask, matched->scale, matched->stats, TOKEN_NAMES);
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
    int64_t tileSize = 0;
    int64_t numWarps = 0;
    int64_t blockMPerWarp = 0;
    bool useMfma32x32 = false;
    bool useTransposedQk32x32 = false;
    /// Also writes the stats output; its kernarg list ends in lse_ptr.
    bool emitLse = false;
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
    const auto tileSize = integerMetadata(kernel, TILE_SIZE_FIELD);
    const auto numWarps = integerMetadata(kernel, NUM_WARPS_FIELD);
    const auto blockMPerWarp = integerMetadata(kernel, BLOCK_M_PER_WARP_FIELD);
    const auto useMfma32x32 = integerMetadata(kernel, USE_MFMA_32X32_FIELD);
    const auto useTransposedQk32x32 = integerMetadata(kernel, USE_TRANSPOSED_QK_32X32_FIELD);
    const auto emitLse = integerMetadata(kernel, EMIT_LSE_FIELD);
    if(!headSize || !numQueryHeads || !numKvHeads || !blockSize || !maskType || !slidingWindow
       || !tileSize || !numWarps || !blockMPerWarp || !useMfma32x32 || !useTransposedQk32x32
       || !emitLse)
    {
        return std::nullopt;
    }
    if(*emitLse != 0 && *emitLse != 1)
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
    if(*useMfma32x32 < 0 || *useMfma32x32 > 1 || *useTransposedQk32x32 < 0
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
                                       *tileSize,
                                       *numWarps,
                                       *blockMPerWarp,
                                       *useMfma32x32 != 0,
                                       *useTransposedQk32x32 != 0,
                                       *emitLse != 0};
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
 *   - block_size, which must divide seqlen_kv (the identity cache is whole blocks).
 *
 * The batch is not bounded here: the kernel derives its cu_q search trip count from the
 * `num_seqs` kernarg, so one binary serves every batch the identity table can index.
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
    // problemFor reads dims without checking them, so a direct caller (not behind
    // graph_match) must not reach it with a malformed operand.
    if(q == nullptr || k == nullptr || !isWellFormedOperand(*q) || !isWellFormedOperand(*k))
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

    // Only an LSE-on kernel serves a graph that asks for stats, and only such a graph: the
    // LSE store is baked in, with its trailing lse_ptr.
    const auto stats = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, STATS_TOKEN);
    if(!stats.has_value() || candidate->emitLse != (*stats != unified_attention::NO_STATS))
    {
        return false;
    }

    // The identity cache must be whole blocks of THIS candidate's size, and small enough for
    // the table the kernel indexes in 32 bits.
    return gfx950Unified2DWorkspaceLayout(problem.batch, problem.seqLenKv, candidate->blockSize)
        .has_value();
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

/// The kernel signature for all variants in this engine: 10 pointers, 6 scalars, the two
/// strides (`_attn_signature(include_bt_stride=True, include_qq_bias_stride=True)`) and the
/// runtime shape kernargs (`RUNTIME_KERNARG_FIELDS`: head counts and window).
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
/// mismatch diagnostic prints the real layout beside the recorded one. An LSE-on kernel
/// (@p emitLse) appends lse_ptr.
std::vector<KernelArgument> unified2DKernelSignature(bool emitLse = false)
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));

    std::vector<KernelArgument> arguments{
        KernelArgument{"global_buffer", PTR, 0, "output_ptr"},
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
        KernelArgument{"by_value", I32, 108, "qq_bias_stride0"},
        KernelArgument{"by_value", I32, 112, "num_query_heads"},
        KernelArgument{"by_value", I32, 116, "num_kv_heads"},
        KernelArgument{"by_value", I32, 120, "sliding_window"}};
    if(emitLse)
    {
        // After the runtime kernargs, 8-byte aligned (read off the compiled LSE kernel).
        arguments.push_back(KernelArgument{"global_buffer", PTR, 128, "lse_ptr"});
    }
    return arguments;
}

/// The compiled kernel plus everything launch() needs, owning nothing that points back into
/// the MatchContext or BoundTokens it came from.
class PreparedGfx950AttentionUnified2D : public PreparedDispatch
{
public:
    PreparedGfx950AttentionUnified2D(IngestorKernelCode code,
                                     UnifiedAttentionBinding binding,
                                     UnifiedAttentionProblem problem,
                                     Gfx950Unified2DWorkspaceLayout layout,
                                     std::vector<unsigned char> workspaceImage,
                                     int64_t slidingWindow)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _layout(layout)
        , _workspaceImage(std::move(workspaceImage))
        , _slidingWindow(slidingWindow)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const UnifiedAttentionBinding& binding() const
    {
        return _binding;
    }

    const UnifiedAttentionProblem& problem() const
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

    /// The variant's window, which the kernel reads as a kernarg.
    int64_t slidingWindow() const
    {
        return _slidingWindow;
    }

private:
    IngestorKernelCode _code;
    UnifiedAttentionBinding _binding;
    UnifiedAttentionProblem _problem;
    Gfx950Unified2DWorkspaceLayout _layout;
    std::vector<unsigned char> _workspaceImage;
    int64_t _slidingWindow = 0;
};

/// The graph's problem, read from the bound Q and K uids. A null tensor means a bound uid
/// names no tensor: refused by name, since everything after indexes dims.
UnifiedAttentionProblem boundProblem(const MatchContext& context,
                                     const UnifiedAttentionBinding& binding)
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
                                             const UnifiedAttentionProblem& problem,
                                             const KernelDefinition& kernel)
{
    const std::string name = toString(kernel.kernelId);
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
        const auto binding = unified_attention::readBinding(bound, TOKEN_NAMES);
        const auto problem = boundProblem(context, binding);
        const auto candidate = requireCandidate(kernel);
        return requireLayout(candidate, problem, kernel).totalBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = unified_attention::readBinding(bound, TOKEN_NAMES);

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
        // kernelMatches pairs them; a caller that skipped it would launch the wrong kernarg
        // list.
        if(candidate.emitLse != (binding.stats != unified_attention::NO_STATS))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_2d: kernel '" + toString(kernel.kernelId)
                    + (candidate.emitLse ? "' writes stats, but the graph asks for none"
                                         : "' writes no stats, but the graph asks for them"));
        }
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

        auto code = buildIngestorKernelCode(
            _kpackLoader, context, kernel, unified2DKernelSignature(candidate.emitLse));

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
            std::move(code), binding, problem, layout, std::move(*image), candidate.slidingWindow);
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
        if(binding.stats != unified_attention::NO_STATS)
        {
            // The LSE-on kernel: the same arguments, then the stats output.
            const auto stats = hipdnn_plugin_sdk::findDeviceBuffer(
                binding.stats, deviceBuffers, numDeviceBuffers);
            preparedUnified.kernelForStream(stream).launch(
                stream,
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
                noQqBiasStride,
                static_cast<int32_t>(p.numQueryHeads),
                static_cast<int32_t>(p.numKvHeads),
                static_cast<int32_t>(preparedUnified.slidingWindow()),
                stats.ptr);
            return;
        }
        preparedUnified.kernelForStream(stream).launch(
            stream,
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
            noQqBiasStride,
            static_cast<int32_t>(p.numQueryHeads),
            static_cast<int32_t>(p.numKvHeads),
            static_cast<int32_t>(preparedUnified.slidingWindow()));
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
