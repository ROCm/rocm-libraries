// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/AttentionDenseCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseLaunchValues.hpp"

/**
 * @file Gfx950AttentionDenseNative.cpp
 * @brief The hipkernel:Gfx950AttentionDense engine's native half: its arch descriptor,
 *        scoring and registration.
 *
 * The kernel is `rocke/library/kernels/gfx950/attention_dense.py`'s
 * `build_attention_dense`, packaged (kind: rocke -> kpack) for gfx950 only. Matching and
 * dispatch, the [2^-64, 2^4] scale range included, are shared with the gfx942 twin
 * (AttentionDenseNative.cpp); what is gfx950's own is bound here: the block_m {128, 256}
 * tiles with their LDS budget, no sliding window, a head-count comparison because the
 * catalog lists each head configuration as its own entry, and the v1 (8-argument) and v2
 * (10-argument) launch contracts.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_dense.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_dense.dispatch";

/// The tile every legacy record completes to, and the one cold selection prefers.
constexpr int64_t BASELINE_BLOCK_M = 256;
constexpr int64_t BASELINE_BLOCK_N = 64;

attention_dense::AttentionDenseGeometry gfx950Geometry(int64_t blockM,
                                                       int64_t seqLenQ,
                                                       int64_t numQueryHeads,
                                                       int64_t batch,
                                                       const std::string& kernelName)
{
    const auto geometry
        = gfx950AttentionDenseGeometry(blockM, seqLenQ, numQueryHeads, batch, kernelName);
    return attention_dense::AttentionDenseGeometry{
        geometry.gridX, geometry.gridY, geometry.gridZ, geometry.blockX};
}

constexpr attention_dense::AttentionDenseArch gfx950Arch()
{
    attention_dense::AttentionDenseArch arch{};
    arch.label = "gfx950";
    arch.engineName = GFX950_ATTENTION_DENSE_ENGINE_NAME;
    arch.tokens.q = "gfx950_attention_dense.q.uid";
    arch.tokens.k = "gfx950_attention_dense.k.uid";
    arch.tokens.v = "gfx950_attention_dense.v.uid";
    arch.tokens.o = "gfx950_attention_dense.o.uid";
    arch.tokens.causal = "gfx950_attention_dense.causal";
    // Always 0: the graph match declines every window, so no graph that reaches the bind
    // asks for one. Bound anyway, so a variant built with a window is never matched.
    arch.tokens.slidingWindow = "gfx950_attention_dense.sliding_window";
    arch.tokens.scaleBits = "gfx950_attention_dense.scale_bits";
    arch.isSupportedTile = &isSupportedGfx950AttentionDenseTile;
    arch.geometry = &gfx950Geometry;
    // No catalog variant carries a non-zero sliding_window, so a windowed mask has no
    // spelling here.
    arch.windowReachesKeys = nullptr;
    // No catalog variant writes LSE, so every stats output declines under 'stats'.
    arch.candidateWritesLse = nullptr;
    arch.currentContract = {GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2,
                            attention_dense::AttentionDenseAbi::RUNTIME_HEAD_COUNTS};
    arch.legacyContract = {GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
                           attention_dense::AttentionDenseAbi::BAKED_HEAD_COUNTS};
    return arch;
}

constexpr attention_dense::AttentionDenseArch GFX950_ARCH = gfx950Arch();

std::optional<BoundTokens> gfx950AttentionDenseGraphMatches(const MatchContext& context)
{
    return attention_dense::graphMatches(context, GFX950_ARCH);
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    return attention_dense::kernelMatches(context, bound, kernel, GFX950_ARCH);
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
 * @brief Preference among tiles that can serve this problem. Higher wins, range [0, 1).
 *
 * PLACEHOLDER. This reproduces the shape-blind cold order described on @ref scoreKernel
 * and ignores @p problem and @p cuCount. It is the seam the measured rule replaces: the
 * gfx950 tile sweep records the fastest tile per shape, and the fit goes here. Until it
 * does, selection is the historical one and the order is unchanged.
 *
 * Whatever replaces this body must keep two properties the selector depends on, both
 * asserted by ScoresEveryTileDeterministicallyAsAPositiveFiniteWeight: every score is
 * positive and finite, and two distinct tiles never score equal, or the descriptor-id
 * tie-break silently decides between them.
 *
 * Divisibility is not this function's problem. kernelMatches rejects a tile that does not
 * divide both lengths (@ref tileDivides) before anything is scored, so every tile here can
 * already serve the shape and the question is only which is fastest.
 */
double tilePreference(const AttentionDenseTile& tile,
                      const AttentionDenseProblem& problem,
                      int cuCount)
{
    static_cast<void>(problem); // the measured rule reads both; the placeholder does not
    static_cast<void>(cuCount);

    if(tile.blockM == BASELINE_BLOCK_M && tile.blockN == BASELINE_BLOCK_N)
    {
        return BASELINE_TILE_SCORE;
    }
    const int64_t key = tile.blockM * BLOCK_M_KEY_WEIGHT + tile.blockN;
    return ALTERNATIVE_CEILING - static_cast<double>(key) / KEY_SPAN;
}

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * A stable fallback order for cold selection, not yet a performance model:
 *
 *  1. The baseline 256/64 tile -- every legacy record's -- above every alternative.
 *  2. The alternatives in ascending lexicographic (block_m, block_n) order.
 *
 * Every distinct tile gets a distinct score, so the selector's descriptor-id tie-break
 * never decides between two tiles.
 *
 * The shape and the device reach @ref tilePreference so a measured rule can read them.
 * A graph this scorer cannot read the shape from falls back to the tile-only order rather
 * than refusing: applicability is kernelMatches' job, and anything scored here already
 * passed it.
 */
double scoreKernel(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto tile = attention_dense::candidateTile(kernel, GFX950_ARCH);
    if(!tile.has_value())
    {
        return UNSUPPORTED_TILE_SCORE;
    }

    AttentionDenseProblem problem;
    if(const auto* attributes = sdpaNode(context); attributes != nullptr)
    {
        const auto* q = findTensor(context, attributes->q_tensor_uid());
        const auto* k = findTensor(context, attributes->k_tensor_uid());
        if(q != nullptr && k != nullptr)
        {
            problem = problemFor(*q, *k);
        }
    }
    return tilePreference(*tile, problem, context.deviceProperties.multiProcessorCount);
}

} // namespace

MetadataValues gfx950AttentionDenseLaunchValues(const AttentionDenseBinding& binding,
                                                const AttentionDenseProblem& problem,
                                                int64_t blockM)
{
    return attention_dense::launchValues(binding, problem, blockM, GFX950_ARCH);
}

Gfx950AttentionDenseLaunchInputs
    readGfx950AttentionDenseLaunchInputs(const SavedLaunchInputs& inputs)
{
    return attention_dense::readLaunchInputs(inputs, GFX950_ARCH);
}

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

/// This engine's dispatch handler for the current (v2) or the legacy (v1) contract,
/// process-lifetime. Both share one module cache and loader.
const IKernelDispatchHandler<Handle>& gfx950AttentionDenseDispatchHandler(bool legacy)
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx950AttentionDenseKpackModuleCache());
    static const auto s_v1 = attention_dense::makeDispatchHandler(
        s_kpackLoader, GFX950_ARCH, GFX950_ARCH.legacyContract);
    static const auto s_v2 = attention_dense::makeDispatchHandler(
        s_kpackLoader, GFX950_ARCH, GFX950_ARCH.currentContract);
    return legacy ? *s_v1 : *s_v2;
}

} // namespace

void registerGfx950AttentionDenseSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx950AttentionDenseGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    // The UDD's unversioned name dispatches the catalog, all of which takes v2.
    scope.add(std::string(DISPATCH_SYMBOL), &gfx950AttentionDenseDispatchHandler(/*legacy=*/false));
    scope.add(std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V2),
              &gfx950AttentionDenseDispatchHandler(/*legacy=*/false));
    // Kept so plans saved under v1 still load and launch their 8-argument code object.
    scope.add(std::string(GFX950_ATTENTION_DENSE_DISPATCH_SYMBOL_V1),
              &gfx950AttentionDenseDispatchHandler(/*legacy=*/true));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
