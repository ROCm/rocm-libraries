// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

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
 * tiles with their LDS budget, no sliding window, and the launch contracts: the strided one
 * (22 arguments, 26 for a kernel built with an additive bias) the catalog dispatches through,
 * and the legacy v1 (8-argument), v2 (10-argument) and v3 (14-argument, with a bias) ones
 * older plans restore through. A kernel built with an additive bias (the use_additive_bias
 * KMD field) adds the bias pointer and its strides before the twelve; its body addresses
 * packed Q/K/V/O and ignores the strides. No head count is compared: the kernel reads both
 * as kernargs taken from the graph.
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

// The KMD fields the stride rule below reads besides attention_dense::RAGGED_FIELD. The KMD does
// not declare `persistent` or `persist_decode` (no catalog kernel is persistent), so a record that
// has neither is the grid body.
constexpr std::string_view PERSISTENT_FIELD = "persistent";
constexpr std::string_view PERSIST_DECODE_FIELD = "persist_decode";
/// The decode a persistent record that names none, or `auto`, resolves to one of rocKE's runtime
/// decodes (`auto` never picks a gqa_pair order).
constexpr std::string_view AUTO_PERSIST_DECODE = "auto";

/**
 * @brief Whether a candidate's body, taken as one without an additive bias, addresses Q, K, V
 *        and O through the twelve stride arguments.
 *
 * rocKE's `_reads_dense_strides` (kernels/gfx950/attention_dense.py), restated: the grid body (one
 * workgroup per query block) and its KV-segment form read them, and so does the persistent body
 * in every work order that decodes through the shape parameters (`runtime_shape`: qb_major,
 * hkv_major, bt_hkv_minor, hq_minor_swz, auto) and its wide-DMA form. The gqa_pair and
 * gqa_pair_2phase orders (their workgroup count is a formula of the baked shape) and ragged,
 * varlen and paged specs keep packed addressing and ignore them. So do the additive-bias bodies,
 * but this rule does not read use_additive_bias: for a bias record it answers as for the body
 * without the bias, which may be true. AttentionDenseArch::readsStrides, its only caller, answers
 * for the bias bodies before it asks this rule.
 *
 * The KMD carries `ragged` and, once a persistent kernel ships, `persistent` and `persist_decode`;
 * a record that names neither is the grid body, and a field that is present and malformed is not
 * assumed to read the strides, so such a record still matches packed graphs and never a strided
 * one. Not representable here, and why that is safe: varlen and paged kernels take more arguments
 * than the 22 this catalog's contract launches, so the signature census and prepare() refuse them
 * first; rocKE's `runtime_shape` also excludes a persistent kernel with a moving bottom-right
 * diagonal (causal_bottom_right at seqlen_q != seqlen_kv), which has no KMD field (the graph match
 * declines such graphs), and the census fails the day a shipped kernel of that class makes the
 * two disagree.
 *
 * This and AttentionDenseArch::readsStrides are the one place the served set lives; a change to
 * it is this function and its tests. The census compares readsStrides with rocKE's own
 * `body_serves_strides` on every shipped kernel and on one flip of each body class
 * (TestGfx950AttentionDenseCatalogTiles.cpp), so a change to either side that makes them disagree
 * fails there: serving a body that does not read its strides is a silent wrong result.
 */
bool candidateReadsStrides(const KernelDefinition& kernel)
{
    if(attention_dense::integerMetadata(kernel, attention_dense::RAGGED_FIELD) != 0)
    {
        return false;
    }
    // `persistent` and `persist_decode` are optional fields (the declining accessors are for the
    // required ones), so they are read with find and get_if: absent is the grid body, malformed
    // is not assumed to read the strides.
    const auto persistentField = kernel.metadata.find(std::string(PERSISTENT_FIELD));
    if(persistentField == kernel.metadata.end())
    {
        return true;
    }
    const auto* persistent = std::get_if<int64_t>(&persistentField->second);
    if(persistent == nullptr)
    {
        return false;
    }
    if(*persistent != 1)
    {
        return *persistent == 0;
    }
    const auto decodeField = kernel.metadata.find(std::string(PERSIST_DECODE_FIELD));
    if(decodeField == kernel.metadata.end())
    {
        return true;
    }
    const auto* decode = std::get_if<std::string>(&decodeField->second);
    return decode != nullptr
           && (*decode == AUTO_PERSIST_DECODE || attention_dense::parsePersistDecode(*decode));
}

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
    arch.tokens.bias = "gfx950_attention_dense.bias";
    arch.tokens.biasUid = "gfx950_attention_dense.bias.uid";
    arch.tokens.biasStrideBatch = "gfx950_attention_dense.bias.stride_batch";
    arch.tokens.biasStrideHead = "gfx950_attention_dense.bias.stride_head";
    arch.tokens.biasStrideQuery = "gfx950_attention_dense.bias.stride_query";
    arch.isSupportedTile = &isSupportedGfx950AttentionDenseTile;
    arch.geometry = &gfx950Geometry;
    // No catalog variant carries a non-zero sliding_window, so a windowed mask has no
    // spelling here.
    arch.servesWindows = false;
    // No catalog variant writes LSE, so every stats output declines under 'stats'.
    arch.candidateWritesLse = nullptr;
    // The grid and persistent bodies read the strides; the gqa_pair orders, the ragged body
    // and the bias bodies address packed storage and ignore them.
    arch.candidateReadsStrides = &candidateReadsStrides;
    arch.candidateAddsBias = &attention_dense::candidateAdditiveBias;
    // A bias kernel takes the strided arguments and the bias's before the strides, so its
    // plans are saved under their own name; the current handler serves both.
    arch.setContracts(GFX950_ATTENTION_DENSE_CONTRACTS);
    return arch;
}

constexpr attention_dense::AttentionDenseArch GFX950_ARCH = gfx950Arch();
static_assert(GFX950_ARCH.isConsistent(),
              "gfx950Arch(): the descriptor is not coherent (AttentionDenseArch::isConsistent "
              "names each rule: the stride rule, the contracts' names and kernel-argument forms, "
              "and the tokens of a served feature)");

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
 * here already passed it. That makes 128/128 preferred, the small tile being the safe
 * choice for an unknown shape. The rest fall in by how much of that tile they match, not
 * by size: the other block_m 128 tiles on the block_m bonus, then 256/128 on the block_n
 * bonus alone, then the remaining block_m 256 tiles, each group ordered by the remainder.
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

const attention_dense::AttentionDenseArch& gfx950AttentionDenseArch()
{
    return GFX950_ARCH;
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

/// This engine's dispatch handlers, one per handler-owning row of
/// GFX950_ATTENTION_DENSE_CONTRACTS, process-lifetime. All share one module cache and loader.
const attention_dense::AttentionDenseDispatchHandlers& gfx950AttentionDenseDispatchHandlers()
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx950AttentionDenseKpackModuleCache());
    static const attention_dense::AttentionDenseDispatchHandlers s_handlers(s_kpackLoader,
                                                                            GFX950_ARCH);
    return s_handlers;
}

} // namespace

void registerGfx950AttentionDenseSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx950AttentionDenseGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    // The UDD's unversioned name dispatches the catalog, all of which takes the strides:
    // plain kernels the strided contract and bias kernels the strided bias contract, both
    // served (and saved under) by the current handler. The legacy rows keep plans saved under
    // v3, v2 and v1 loading and launching their 14-, 10- and 8-argument code objects.
    gfx950AttentionDenseDispatchHandlers().registerWith(scope, DISPATCH_SYMBOL);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
