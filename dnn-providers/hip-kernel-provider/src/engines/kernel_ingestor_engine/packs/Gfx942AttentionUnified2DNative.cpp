// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
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
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified2DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttention2DCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGraphMatch.hpp"

/**
 * @file Gfx942AttentionUnified2DNative.cpp
 * @brief The hipkernel:Gfx942AttentionUnified2D engine's native half: its candidate record,
 *        scoring and registration. Matching and dispatch are UnifiedAttention2DCommon.hpp's,
 *        parameterised by this file's Unified2DArch.
 *
 * The kernel is `rocke/library/kernels/gfx942/attention_tiled_2d.py`'s
 * `build_unified_attention_2d_tiled`, packaged (kind: rocke -> kpack) for gfx942 only: the
 * paged, ragged-Q unified attention that vLLM serves. A hipDNN SDPA graph is dense, so this
 * engine serves a graph's contiguous BSHD K/V as a paged cache whose block table is the
 * identity, and stages the three metadata arrays that implies in the plan's workspace (see
 * Gfx942AttentionUnified2DGeometry.hpp). Its applicability rules live only in the Python;
 * this file is where they become enforceable.
 *
 * The kernel reads the head counts, the window and the batch at run time and bakes only the
 * GQA ratio, so a catalog variant is keyed by the ratio and serves every head pair of it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx942_attention_unified_2d.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx942_attention_unified_2d.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx942_attention_unified_2d.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx942_attention_unified_2d.dispatch";

// KMD fields (16-field schema). Fields a shipped variant holds at one value across the whole
// catalog (kv_storage_dtype, the sink/alibi/qq-bias/softcap switches) are absent: a variant
// that moved one of them independently would have to add it here first, or the candidates
// collide on the catalog key and the loader drops one. The five codegen flags that tell a
// geometry's default kernel from rocKE's production kernel of the same geometry
// (use_k_sliced_ring, use_q_direct_global, use_conflict_free_v_store, use_transposed_mask_limit,
// use_transposed_invariant_hoist) are KMD fields and engine knobs. The other flags a
// production stack sets (the rest of the mask stack, waves_per_eu, the K ring's depth and
// slice width, the KV cache policy) move only with those five, so they stay in the kernel's
// spec, not in the KMD. registry_enumerator.py refuses a catalog whose kernel names repeat.
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERIES_PER_KV_FIELD = "num_queries_per_kv";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view MASK_TYPE_FIELD = "mask_type";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view TILE_SIZE_FIELD = "tile_size";
constexpr std::string_view NUM_WARPS_FIELD = "num_warps";
constexpr std::string_view BLOCK_M_PER_WARP_FIELD = "block_m_per_warp";
constexpr std::string_view USE_MFMA_32X32X8_FIELD = "use_mfma_32x32x8";
constexpr std::string_view USE_TRANSPOSED_QK_32X32_FIELD = "use_transposed_qk_32x32";
constexpr std::string_view USE_K_SLICED_RING_FIELD = "use_k_sliced_ring";
constexpr std::string_view USE_Q_DIRECT_GLOBAL_FIELD = "use_q_direct_global";
constexpr std::string_view USE_CONFLICT_FREE_V_STORE_FIELD = "use_conflict_free_v_store";
constexpr std::string_view USE_TRANSPOSED_MASK_LIMIT_FIELD = "use_transposed_mask_limit";
constexpr std::string_view USE_TRANSPOSED_INVARIANT_HOIST_FIELD = "use_transposed_invariant_hoist";

/// Each head size's default geometry in the catalog: transposed 32x32x8, 32 rows per warp,
/// 4 warps and a 64-token KV tile at D64, 2 warps and a 32-token KV tile at D128.
struct DefaultGeometry
{
    int64_t headSize;
    int64_t tileSize;
    int64_t numWarps;
};
constexpr std::array<DefaultGeometry, 2> DEFAULT_GEOMETRIES = {{{64, 64, 4}, {128, 32, 2}}};
constexpr int64_t DEFAULT_BLOCK_M_PER_WARP = 32;

constexpr std::string_view Q_TOKEN = "gfx942_attention_unified_2d.q.uid";
constexpr std::string_view K_TOKEN = "gfx942_attention_unified_2d.k.uid";
constexpr std::string_view V_TOKEN = "gfx942_attention_unified_2d.v.uid";
constexpr std::string_view O_TOKEN = "gfx942_attention_unified_2d.o.uid";
/// The mask kind graphMatches derived (a MaskType value), so kernelMatches does not re-derive it.
constexpr std::string_view MASK_TOKEN = "gfx942_attention_unified_2d.mask_type";
/// The sliding window the candidate must have been built with. Always 0: maskTypeFor
/// declines every left bound, so no graph that reaches the bind asks for a window.
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx942_attention_unified_2d.sliding_window";
/// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
constexpr std::string_view SCALE_BITS_TOKEN = "gfx942_attention_unified_2d.scale_bits";
/// Always NO_STATS: graph_match declines a graph that asks for a stats output.
constexpr std::string_view STATS_TOKEN = "gfx942_attention_unified_2d.stats.uid";

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
    names.dispatchLabel = "gfx942 attention_unified_2d";
    return names;
}

using unified_attention::integerMetadata;
using unified_attention::MaskType;
using unified_attention::SdpaShape;

/// What a candidate bakes, read from its completed metadata and judged buildable.
struct AttentionUnified2DCandidate
{
    int64_t headSize = 0;
    int64_t numQueriesPerKv = 0;
    int64_t blockSize = 0;
    int64_t maskType = 0;
    int64_t slidingWindow = 0;
    int64_t tileSize = 0;
    int64_t numWarps = 0;
    int64_t blockMPerWarp = 0;
    bool useMfma32x32x8 = false;
    bool useTransposedQk32x32 = false;
    bool useKSlicedRing = false;
    bool useQDirectGlobal = false;
    bool useConflictFreeVStore = false;
    bool useTransposedMaskLimit = false;
    bool useTransposedInvariantHoist = false;
    /// No LSE-on kernel ships on gfx942: graph_match declines a stats output, so every
    /// candidate is judged against a graph that binds NO_STATS.
    static constexpr bool emitLse = false;
};

/**
 * @brief The candidate's baked configuration, or nullopt when it has none this engine can
 *        launch.
 *
 * Validated against isSupportedGfx942Unified2DGeometry, so a record the Python builder would
 * refuse is declined like any other unbuildable one. An absent field means an uncompleted or
 * malformed record: it is declined rather than given a default here, because a substituted
 * geometry launches a binary with another binary's grid.
 */
std::optional<AttentionUnified2DCandidate> candidateFor(const KernelDefinition& kernel)
{
    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto numQueriesPerKv = integerMetadata(kernel, NUM_QUERIES_PER_KV_FIELD);
    const auto blockSize = integerMetadata(kernel, BLOCK_SIZE_FIELD);
    const auto maskType = integerMetadata(kernel, MASK_TYPE_FIELD);
    const auto slidingWindow = integerMetadata(kernel, SLIDING_WINDOW_FIELD);
    const auto tileSize = integerMetadata(kernel, TILE_SIZE_FIELD);
    const auto numWarps = integerMetadata(kernel, NUM_WARPS_FIELD);
    const auto blockMPerWarp = integerMetadata(kernel, BLOCK_M_PER_WARP_FIELD);
    const auto useMfma32x32x8 = integerMetadata(kernel, USE_MFMA_32X32X8_FIELD);
    const auto useTransposedQk32x32 = integerMetadata(kernel, USE_TRANSPOSED_QK_32X32_FIELD);
    const auto useKSlicedRing = integerMetadata(kernel, USE_K_SLICED_RING_FIELD);
    const auto useQDirectGlobal = integerMetadata(kernel, USE_Q_DIRECT_GLOBAL_FIELD);
    const auto useConflictFreeVStore = integerMetadata(kernel, USE_CONFLICT_FREE_V_STORE_FIELD);
    const auto useTransposedMaskLimit = integerMetadata(kernel, USE_TRANSPOSED_MASK_LIMIT_FIELD);
    const auto useTransposedInvariantHoist
        = integerMetadata(kernel, USE_TRANSPOSED_INVARIANT_HOIST_FIELD);
    if(!headSize || !numQueriesPerKv || !blockSize || !maskType || !slidingWindow || !tileSize
       || !numWarps || !blockMPerWarp || !useMfma32x32x8 || !useTransposedQk32x32 || !useKSlicedRing
       || !useQDirectGlobal || !useConflictFreeVStore || !useTransposedMaskLimit
       || !useTransposedInvariantHoist)
    {
        return std::nullopt;
    }

    if(*headSize != 64 && *headSize != 128)
    {
        return std::nullopt;
    }
    if(*numQueriesPerKv < 1 || *numQueriesPerKv > GFX942_UNIFIED_2D_MAX_QUERIES_PER_KV)
    {
        return std::nullopt;
    }
    if(*maskType < static_cast<int64_t>(MaskType::NO_MASK)
       || *maskType > static_cast<int64_t>(MaskType::BOTTOM_RIGHT_CAUSAL))
    {
        return std::nullopt;
    }
    for(const auto flag : {*useMfma32x32x8,
                           *useTransposedQk32x32,
                           *useKSlicedRing,
                           *useQDirectGlobal,
                           *useConflictFreeVStore,
                           *useTransposedMaskLimit,
                           *useTransposedInvariantHoist})
    {
        if(flag < 0 || flag > 1)
        {
            return std::nullopt;
        }
    }
    const Gfx942Unified2DGeometryFields geometry{*headSize,
                                                 *blockSize,
                                                 *tileSize,
                                                 *numWarps,
                                                 *blockMPerWarp,
                                                 *useMfma32x32x8 != 0,
                                                 *useTransposedQk32x32 != 0};
    if(!isSupportedGfx942Unified2DGeometry(geometry))
    {
        return std::nullopt;
    }

    return AttentionUnified2DCandidate{*headSize,
                                       *numQueriesPerKv,
                                       *blockSize,
                                       *maskType,
                                       *slidingWindow,
                                       *tileSize,
                                       *numWarps,
                                       *blockMPerWarp,
                                       *useMfma32x32x8 != 0,
                                       *useTransposedQk32x32 != 0,
                                       *useKSlicedRing != 0,
                                       *useQDirectGlobal != 0,
                                       *useConflictFreeVStore != 0,
                                       *useTransposedMaskLimit != 0,
                                       *useTransposedInvariantHoist != 0};
}

/// The ratio is the only head-count fact the binary bakes: it maps query rows to KV heads and
/// sizes BLOCK_Q. A non-dividing pair has no ratio and no candidate.
bool headsMatch(const AttentionUnified2DCandidate& candidate, const SdpaShape& problem)
{
    const auto ratio = unified_attention::gqaRatio(problem.numQueryHeads, problem.numKvHeads);
    return ratio.has_value() && candidate.numQueriesPerKv == *ratio;
}

/// What prepare() checks beyond the shared rules, as defence in depth for a caller that
/// reaches the handler without having matched.
///
/// The candidate's GQA ratio against the graph's: the grid and the kernel's row-to-head map
/// both come from the ratio, and the binary bakes it.
///
/// The engine serves the unwindowed band only: the catalog bakes no window and graph_match
/// binds 0. A record with one is refused, which also keeps its value from being narrowed to
/// the kernel's int32 argument.
void checkRatioAndWindow(const AttentionUnified2DCandidate& candidate,
                         const SdpaShape& problem,
                         const KernelDefinition& kernel)
{
    if(!headsMatch(candidate, problem))
    {
        unified_attention::refuse("gfx942",
                                  "kernel '" + toString(kernel.kernelId)
                                      + "' is built for a GQA ratio of "
                                      + std::to_string(candidate.numQueriesPerKv) + ", not "
                                      + std::to_string(problem.numQueryHeads) + " query heads over "
                                      + std::to_string(problem.numKvHeads) + " key/value heads");
    }
    if(candidate.slidingWindow != 0)
    {
        unified_attention::refuse("gfx942",
                                  "kernel '" + toString(kernel.kernelId)
                                      + "' bakes a sliding window of "
                                      + std::to_string(candidate.slidingWindow)
                                      + "; this engine serves the unwindowed band only");
    }
}

/// This engine's rules and records, for the shared graph match, kernel match and dispatch.
///
/// The gfx942 tiled 2D body computes only the bottom-right causal band, so the catalog carries
/// only that mask. Decode (one query token per sequence) is hipkernel:Gfx942AttentionUnified3d's
/// at every GQA ratio its split-KV catalog ships.
constexpr unified_attention::Unified2DArch<AttentionUnified2DCandidate> ARCH{
    {"gfx942",
     GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME,
     &GFX942_UNIFIED_ATTENTION_TRAITS,
     tokenNames(),
     &gfx942Unified3dServesGqaGroup,
     "Unified3d",
     /*bottomRightOnly=*/true,
     GFX942_UNIFIED_2D_SHIPPED_RATIOS.data(),
     GFX942_UNIFIED_2D_SHIPPED_RATIOS.size()},
    &candidateFor,
    &headsMatch,
    &checkRatioAndWindow};

std::optional<BoundTokens> gfx942AttentionUnified2DGraphMatches(const MatchContext& context)
{
    return unified_attention::graphMatches(context, ARCH.graph);
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    return unified_attention::kernelMatches(context, bound, kernel, ARCH);
}

/// Scores for the cold ranking. A production stack scores `PRODUCTION_SCORE`; the cell's
/// default geometry scores `DEFAULT_SCORE`; every alternative scores
/// `ALTERNATIVE_CEILING - key / KEY_SPAN`, which lies strictly between zero and
/// ALTERNATIVE_CEILING, so the alternatives sit strictly below the default and the default
/// strictly below the production stack.
constexpr double PRODUCTION_SCORE = 3.0;
constexpr double DEFAULT_SCORE = 2.0;
constexpr double ALTERNATIVE_CEILING = 1.0;
/// Exceeds every key a legal geometry and flag set produces, so an alternative's score stays
/// positive. Powers of two keep every score exactly representable.
constexpr double KEY_SPAN = 134217728.0; // 2^27
/// Below every legal candidate: only a record kernelMatches already declined gets it.
constexpr double UNSUPPORTED_SCORE = 0.0;

/// rocKE's production stack for a (dtype, head size): the mask limit and invariant hoist the
/// production spec turns on together with the rest of the mask stack. A geometry's default
/// kernel never has them, and the catalog ships one stack per (dtype, head size, ratio).
bool isProductionStack(const AttentionUnified2DCandidate& candidate)
{
    return candidate.useTransposedMaskLimit && candidate.useTransposedInvariantHoist;
}

bool isDefaultGeometry(const AttentionUnified2DCandidate& candidate)
{
    if(!candidate.useMfma32x32x8 || !candidate.useTransposedQk32x32
       || candidate.blockMPerWarp != DEFAULT_BLOCK_M_PER_WARP)
    {
        return false;
    }
    for(const auto& d : DEFAULT_GEOMETRIES)
    {
        if(d.headSize == candidate.headSize && d.tileSize == candidate.tileSize
           && d.numWarps == candidate.numWarps)
        {
            return true;
        }
    }
    return false;
}

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * A stable fallback order for cold selection, not a performance model:
 *
 *  1. rocKE's production stack for the cell (the kernel rocKE's own dispatcher builds for a
 *     long prefill of that dtype, head size and ratio, measured by the stage 6 sweep).
 *  2. The head size's default geometry (the catalog cell's default, which a correctness-
 *     checked gfx942 sweep through rocKE's own launcher chose for the most prefill shapes).
 *  3. The alternatives in ascending lexicographic (tile_size, num_warps, block_m_per_warp,
 *     use_mfma_32x32x8, use_transposed_qk_32x32, then the five codegen flags) order.
 *
 * Every distinct kernel of a cell gets a distinct score, so the selector's descriptor-id
 * tie-break never decides between two of them.
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
    if(isProductionStack(*candidate))
    {
        return PRODUCTION_SCORE;
    }
    if(isDefaultGeometry(*candidate))
    {
        return DEFAULT_SCORE;
    }
    // tile <= 512, warps <= 8, rows <= 32, five flags: the key stays below 2^26.
    int64_t key = candidate->tileSize;
    key = key * 16 + candidate->numWarps;
    key = key * 64 + candidate->blockMPerWarp;
    key = key * 2 + (candidate->useMfma32x32x8 ? 1 : 0);
    key = key * 2 + (candidate->useTransposedQk32x32 ? 1 : 0);
    key = key * 2 + (candidate->useKSlicedRing ? 1 : 0);
    key = key * 2 + (candidate->useQDirectGlobal ? 1 : 0);
    key = key * 2 + (candidate->useConflictFreeVStore ? 1 : 0);
    key = key * 2 + (candidate->useTransposedMaskLimit ? 1 : 0);
    key = key * 2 + (candidate->useTransposedInvariantHoist ? 1 : 0);
    return ALTERNATIVE_CEILING - static_cast<double>(key) / KEY_SPAN;
}

} // namespace

compilation::KpackModuleCache& gfx942AttentionUnified2DKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx942AttentionUnified2DModuleCache()
{
    gfx942AttentionUnified2DKpackModuleCache().clear();
}

namespace
{

/// This engine's dispatch handler, process-lifetime.
const unified_attention::Unified2DDispatchHandler<AttentionUnified2DCandidate>&
    gfx942AttentionUnified2DDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx942AttentionUnified2DKpackModuleCache());
    static const unified_attention::Unified2DDispatchHandler<AttentionUnified2DCandidate>
        s_dispatchHandler(s_kpackLoader, ARCH);
    return s_dispatchHandler;
}

} // namespace

void registerGfx942AttentionUnified2DSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx942AttentionUnified2DGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &gfx942AttentionUnified2DDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
