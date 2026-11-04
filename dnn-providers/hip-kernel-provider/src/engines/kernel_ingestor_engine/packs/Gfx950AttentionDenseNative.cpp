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

/// Scores for the ranking. A tile earns one bonus per axis it matches the measured
/// preference on, plus a lexicographic remainder strictly inside (0, ALTERNATIVE_CEILING)
/// so two distinct tiles can never score equal.
constexpr double BLOCK_M_MATCH_BONUS = 2.0;
constexpr double BLOCK_N_MATCH_BONUS = 1.0;
constexpr double ALTERNATIVE_CEILING = 1.0;
/// Weight of block_m in the lexicographic key; exceeds every legal block_n (block_n
/// divides block_m, so it is at most 256).
constexpr int64_t BLOCK_M_KEY_WEIGHT = 512;
/// Exceeds every key a legal tile produces (256 * 512 + 256), so an alternative's score
/// stays positive. Powers of two keep every score exactly representable.
constexpr double KEY_SPAN = 262144.0; // 2^18
/// Below every legal candidate: only a tile kernelMatches already declined gets it.
constexpr double UNSUPPORTED_TILE_SCORE = 0.0;

/// seqlen_q at or above which block_m 256 measured faster than 128, per cohort.
///
/// From the gfx950 tile sweep: 282 shapes (the LLM suite's 101 servable shapes and a
/// 192-shape coverage grid, 11 of which the suite also carries), every legal tile timed
/// against a torch SDPA reference, three
/// alternating rounds, median per configuration. Each threshold sits on a plateau rather
/// than a fitted edge -- for D128 causal every value in (2048, 4096] scores identically --
/// so these are round numbers inside a wide band, not tuned constants.
///
/// Non-causal does the full seqlen_q x seqlen_kv work with no triangular saving, so a
/// given length is a larger problem than the causal case and wants the fat tile sooner.
/// At D128 that is every measured length.
constexpr int64_t BLOCK_M_256_FROM_D64_CAUSAL = 8192;
constexpr int64_t BLOCK_M_256_FROM_D64_NONCAUSAL = 1024;
constexpr int64_t BLOCK_M_256_FROM_D128_CAUSAL = 4096;
constexpr int64_t BLOCK_M_256_FROM_D128_NONCAUSAL = 0;

/// block_n 64 won only for D128 causal; 128 won in all three other cohorts.
constexpr int64_t PREFERRED_BLOCK_N_D128_CAUSAL = 64;
constexpr int64_t PREFERRED_BLOCK_N_OTHERWISE = 128;

/// The candidate's own causal metadata. kernelMatches has already required it to agree
/// with the graph, so for anything scored it is the graph's value too.
constexpr std::string_view CAUSAL_FIELD = "causal";

/// The query length from which block_m 256 wins for this cohort.
int64_t fatTileFrom(bool d128, bool causal)
{
    if(d128)
    {
        return causal ? BLOCK_M_256_FROM_D128_CAUSAL : BLOCK_M_256_FROM_D128_NONCAUSAL;
    }
    return causal ? BLOCK_M_256_FROM_D64_CAUSAL : BLOCK_M_256_FROM_D64_NONCAUSAL;
}

/// The tile the sweep measured fastest for this cohort at this query length.
///
/// Cohort is (head_size, causal) only. dtype was measured and makes no difference:
/// splitting the fit on it changes nothing. Batch and the GQA ratio do carry some signal
/// but are deliberately left out, since the rule is already inside its target and there is
/// no held-out set left to tell a real effect from a fitted one.
attention_dense::AttentionDenseTile preferredTile(int64_t headSize, bool causal, int64_t seqLenQ)
{
    const bool d128 = headSize == 128;
    return attention_dense::AttentionDenseTile{seqLenQ >= fatTileFrom(d128, causal) ? 256 : 128,
                                               d128 && causal ? PREFERRED_BLOCK_N_D128_CAUSAL
                                                              : PREFERRED_BLOCK_N_OTHERWISE};
}

/**
 * @brief Preference among tiles that can serve this problem. Higher wins.
 *
 * Scores how close a candidate is to @ref preferredTile: a bonus per matching axis, with
 * block_m weighted above block_n, then the historical ascending (block_m, block_n) order
 * as the remainder. The preferred tile is often not in the catalog for a given shape --
 * kernelMatches drops every tile that does not divide both lengths -- so the ranking has
 * to degrade sensibly rather than name one winner.
 *
 * Two properties the selector depends on, both asserted by
 * ScoresEveryTileDeterministicallyAsAPositiveFiniteWeight: every score is positive and
 * finite, and two distinct tiles never score equal, or the descriptor-id tie-break
 * silently decides between them. The remainder is what guarantees the second.
 *
 * Divisibility is not this function's problem. kernelMatches rejects a tile that does not
 * divide both lengths (@ref tileDivides) before anything is scored, so every tile here can
 * already serve the shape and the question is only which is fastest.
 */
double tilePreference(const attention_dense::AttentionDenseTile& tile,
                      const AttentionDenseProblem& problem,
                      bool causal)
{
    const auto want = preferredTile(problem.headSize, causal, problem.seqLenQ);
    const double match = (tile.blockM == want.blockM ? BLOCK_M_MATCH_BONUS : 0.0)
                         + (tile.blockN == want.blockN ? BLOCK_N_MATCH_BONUS : 0.0);
    const int64_t key = tile.blockM * BLOCK_M_KEY_WEIGHT + tile.blockN;
    return match + ALTERNATIVE_CEILING - static_cast<double>(key) / KEY_SPAN;
}

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * Orders by closeness to the tile the gfx950 sweep measured fastest for this shape; see
 * @ref preferredTile for the rule and @ref tilePreference for how a catalog without that
 * exact tile degrades. Every distinct tile gets a distinct score, so the selector's
 * descriptor-id tie-break never decides between two tiles.
 *
 * `causal` is read from the candidate's own metadata rather than the bound tokens:
 * kernelMatches has already required the two to agree, so for anything scored here they
 * are the same value and the metadata needs no graph lookup.
 *
 * A graph this scorer cannot read the shape from scores every tile as a zero-length
 * problem rather than refusing: applicability is kernelMatches' job and anything reaching
 * here already passed it. That makes 128/128 preferred -- the small tile, which is the
 * safe choice for an unknown shape -- and ranks the rest ascending behind it.
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
    if(const auto* attributes = sdpa_graph::sdpaNode(context); attributes != nullptr)
    {
        const auto* q = sdpa_graph::findTensor(context, attributes->q_tensor_uid());
        const auto* k = sdpa_graph::findTensor(context, attributes->k_tensor_uid());
        if(q != nullptr && k != nullptr)
        {
            problem = sdpa_graph::problemFor(*q, *k);
        }
    }
    const auto causal = attention_dense::integerMetadata(kernel, CAUSAL_FIELD);
    return tilePreference(*tile, problem, causal.value_or(0) != 0);
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
