// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttention2DCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGraphMatch.hpp"

/**
 * @file Gfx950AttentionUnified2DNative.cpp
 * @brief The hipkernel:Gfx950AttentionUnified2D engine's native half: its candidate record,
 *        scoring and registration. Matching and dispatch are UnifiedAttention2DCommon.hpp's,
 *        parameterised by this file's Unified2DArch.
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

/// The engine's bound-token names, field by field.
constexpr unified_attention::BoundTokenNames tokenNames()
{
    unified_attention::BoundTokenNames names;
    names.q = Q_TOKEN;
    names.k = K_TOKEN;
    names.v = V_TOKEN;
    names.o = O_TOKEN;
    names.maskType = MASK_TOKEN;
    names.slidingWindow = SLIDING_WINDOW_TOKEN;
    names.scaleBits = SCALE_BITS_TOKEN;
    names.stats = STATS_TOKEN;
    names.dispatchLabel = "gfx950 attention_unified_2d";
    return names;
}

using unified_attention::integerMetadata;
using unified_attention::MaskType;
using unified_attention::SdpaShape;

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

    if(*headSize != 64 && *headSize != 128)
    {
        return std::nullopt;
    }
    const auto ratio = unified_attention::gqaRatio(*numQueryHeads, *numKvHeads);
    if(*numQueryHeads <= 0 || !ratio.has_value() || *ratio > GFX950_UNIFIED_2D_MAX_QUERIES_PER_KV)
    {
        return std::nullopt;
    }
    if(*maskType < static_cast<int64_t>(MaskType::NO_MASK)
       || *maskType > static_cast<int64_t>(MaskType::BOTTOM_RIGHT_CAUSAL))
    {
        return std::nullopt;
    }
    if(*useMfma32x32 < 0 || *useMfma32x32 > 1 || *useTransposedQk32x32 < 0
       || *useTransposedQk32x32 > 1 || *emitLse < 0 || *emitLse > 1)
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

/// This catalog lists each head configuration as its own entry, so a candidate bakes both head
/// counts and matches only the graph that has them.
bool headsMatch(const AttentionUnified2DCandidate& candidate, const SdpaShape& problem)
{
    return candidate.numQueryHeads == problem.numQueryHeads
           && candidate.numKvHeads == problem.numKvHeads;
}

/// This engine's rules and records, for the shared graph match, kernel match and dispatch.
/// All three mask kinds are served, each by its own kernel: top-left and bottom-right coincide
/// only at Sq == Skv, but they are distinct kernel bodies and the catalog carries both. Decode
/// is handed to Unified3D wherever its split-KV kernel can be built. The catalog lists each
/// head pair as its own entry (e.g. 28/4 and 40/8, ratios 7 and 5, ship while 7/1 does not), so
/// no GQA ratio list can say which graphs it serves and none is judged at graph_match.
constexpr unified_attention::Unified2DArch<AttentionUnified2DCandidate> ARCH{
    {"gfx950",
     GFX950_ATTENTION_UNIFIED_2D_ENGINE_NAME,
     &GFX950_UNIFIED_ATTENTION_TRAITS,
     tokenNames(),
     &gfx950Unified3DServesGqaGroup,
     "Unified3D",
     /*bottomRightOnly=*/false,
     /*shippedGqaRatios=*/nullptr,
     /*numShippedGqaRatios=*/0},
    &candidateFor,
    &headsMatch,
    /*prepareChecks=*/nullptr};

std::optional<BoundTokens> gfx950AttentionUnified2DGraphMatches(const MatchContext& context)
{
    return unified_attention::graphMatches(context, ARCH.graph);
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    return unified_attention::kernelMatches(context, bound, kernel, ARCH);
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
const unified_attention::Unified2DDispatchHandler<AttentionUnified2DCandidate>&
    gfx950AttentionUnified2DDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx950AttentionUnified2DKpackModuleCache());
    static const unified_attention::Unified2DDispatchHandler<AttentionUnified2DCandidate>
        s_dispatchHandler(s_kpackLoader, ARCH);
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
