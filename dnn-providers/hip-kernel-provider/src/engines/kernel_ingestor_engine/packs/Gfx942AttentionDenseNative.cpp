// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
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
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseLaunchValues.hpp"

/**
 * @file Gfx942AttentionDenseNative.cpp
 * @brief The hipkernel:Gfx942AttentionDense engine's native half: its arch descriptor,
 *        scoring and registration.
 *
 * The kernel is `rocke/library/kernels/gfx942/attention_dense.py`'s
 * `build_attention_dense`, packaged (kind: rocke -> kpack) for gfx942 only. Matching and
 * dispatch are shared with the gfx950 twin (AttentionDenseNative.cpp); what is gfx942's
 * own is bound here: any finite scale, the gfx942 tile rules, the causal sliding window
 * baked per candidate with its zero-trip guard, the stats (LSE) output an emit_lse
 * candidate writes, and the v1 launch contract.
 *
 * The body reads both head counts at runtime as well as batch and the sequence lengths
 * (`Gfx942AttentionDenseSpec.runtime_param_fields`). Unlike the gfx950 twin, whose
 * catalog lists each head configuration and whose kernelMatches compares them, a
 * candidate here is never compared against the graph's head counts.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx942_attention_dense.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx942_attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx942_attention_dense.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx942_attention_dense.dispatch";

// The KMD fields only this engine's schema carries (15 fields; the shared ones are read in
// AttentionDenseNative.cpp). The cold ranking reads lds_row_pad and pv_priority to order
// arms; they do not enter the launch. emit_lse selects the candidates of a graph that
// binds a stats output, and the launch's trailing lse_ptr. waves_per_eu/persistent and the
// other gfx942 levers are absent: every shipped variant holds them at the kernel's own
// policy. A variant that moved any of them would have to add it to the KMD first, or the
// candidates collide on the catalog key and the loader drops one.
constexpr std::string_view LDS_ROW_PAD_FIELD = "lds_row_pad";
constexpr std::string_view PV_PRIORITY_FIELD = "pv_priority";
constexpr std::string_view EMIT_LSE_FIELD = "emit_lse";

/// The policy arms cold selection prefers: the 256/64 tile, the spec's default K row pad
/// (`_DEFAULT_LDS_ROW_PAD`) and the default PV priority.
constexpr int64_t BASELINE_BLOCK_M = 256;
constexpr int64_t BASELINE_BLOCK_N = 64;
constexpr int64_t BASELINE_LDS_ROW_PAD = 8;
constexpr int64_t BASELINE_PV_PRIORITY = 0;

// The gfx942 body scales the scores (scale * log2(e)) before taking the row max, so any
// finite scale is well defined and the Python launch path checks no range. A NaN or
// infinite scale poisons every output row.
bool isSupportedAttentionScale(float scale)
{
    return std::isfinite(scale);
}

/// Whether a candidate writes LSE: emit_lse 1 or 0. A record without the field reaches here
/// completed to 0 through the KMD default, so a missing or other value is malformed.
std::optional<bool> candidateWritesLse(const KernelDefinition& kernel)
{
    const auto emitLse = attention_dense::integerMetadata(kernel, EMIT_LSE_FIELD);
    if(!emitLse.has_value() || (*emitLse != 0 && *emitLse != 1))
    {
        return std::nullopt;
    }
    return *emitLse == 1;
}

attention_dense::AttentionDenseGeometry gfx942Geometry(int64_t blockM,
                                                       int64_t seqLenQ,
                                                       int64_t numQueryHeads,
                                                       int64_t batch,
                                                       const std::string& kernelName)
{
    const auto geometry
        = gfx942AttentionDenseGeometry(blockM, seqLenQ, numQueryHeads, batch, kernelName);
    return attention_dense::AttentionDenseGeometry{
        geometry.gridX, geometry.gridY, geometry.gridZ, geometry.blockX};
}

constexpr attention_dense::AttentionDenseArch gfx942Arch()
{
    attention_dense::AttentionDenseArch arch{};
    arch.label = "gfx942";
    arch.engineName = GFX942_ATTENTION_DENSE_ENGINE_NAME;
    arch.tokens.q = "gfx942_attention_dense.q.uid";
    arch.tokens.k = "gfx942_attention_dense.k.uid";
    arch.tokens.v = "gfx942_attention_dense.v.uid";
    arch.tokens.o = "gfx942_attention_dense.o.uid";
    arch.tokens.causal = "gfx942_attention_dense.causal";
    arch.tokens.slidingWindow = "gfx942_attention_dense.sliding_window";
    arch.tokens.scaleBits = "gfx942_attention_dense.scale_bits";
    arch.tokens.stats = "gfx942_attention_dense.stats.uid";
    // The head fields of the KMD are canonical build inputs, never compared.
    arch.matchesHeadCounts = false;
    arch.isSupportedScale = &isSupportedAttentionScale;
    arch.unsupportedScaleReason = "is not finite";
    arch.isSupportedTile = &isSupportedGfx942AttentionDenseTile;
    arch.geometry = &gfx942Geometry;
    arch.windowReachesKeys = &gfx942AttentionDenseWindowReachesKeys;
    // An emit_lse candidate writes the stats output; the catalog pairs each kernel with one.
    arch.candidateWritesLse = &candidateWritesLse;
    arch.refusesUnservedSavedValues = true;
    arch.currentContract = {GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1,
                            attention_dense::AttentionDenseAbi::RUNTIME_HEAD_COUNTS};
    // No gfx942 kernel ever baked the head counts, so there is no superseded contract.
    arch.legacyContract = {};
    return arch;
}

constexpr attention_dense::AttentionDenseArch GFX942_ARCH = gfx942Arch();

std::optional<BoundTokens> gfx942AttentionDenseGraphMatches(const MatchContext& context)
{
    return attention_dense::graphMatches(context, GFX942_ARCH);
}

bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    return attention_dense::kernelMatches(context, bound, kernel, GFX942_ARCH);
}

/// Below every legal candidate: only a record kernelMatches already declined gets it.
constexpr double UNSUPPORTED_SCORE = 0.0;
/// Exceeds every rank a legal record produces, so every legal score stays positive.
/// Powers of two keep every score exactly representable.
constexpr int64_t RANK_SPAN = int64_t{1} << 20;

/**
 * @brief An arm's position in its knob's cold order: 0 for the policy value, otherwise
 *        1 + its ascending position. Distinct per value within @p span.
 */
int64_t armRank(int64_t value, int64_t policy, int64_t step, int64_t span)
{
    if(value == policy)
    {
        return 0;
    }
    return std::min(span - 1, 1 + value / step);
}

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * A stable fallback order for cold selection, not a performance model. The rank is
 * lexicographic over (tile, lds_row_pad, pv_priority), and each knob places its policy
 * arm first (the 256/64 tile, lds_row_pad 8, pv_priority 0) and its alternatives in
 * ascending order. Every distinct knob combination gets a distinct score, so the
 * selector's descriptor-id tie-break never decides between two arms. emit_lse is not
 * ranked: kernelMatches admits the LSE twins of a graph that binds stats and the others
 * of one that does not, so a twin and its partner never compete.
 */
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto tile = attention_dense::candidateTile(kernel, GFX942_ARCH);
    const auto ldsRowPad = attention_dense::integerMetadata(kernel, LDS_ROW_PAD_FIELD);
    const auto pvPriority = attention_dense::integerMetadata(kernel, PV_PRIORITY_FIELD);
    if(!tile.has_value() || !ldsRowPad.has_value() || !pvPriority.has_value() || *ldsRowPad < 0
       || *pvPriority < 0)
    {
        return UNSUPPORTED_SCORE;
    }

    // block_m <= 512 and block_n <= block_m, both multiples of 32: at most 17 * 17 tiles.
    constexpr int64_t TILE_SPAN = 512;
    constexpr int64_t PAD_SPAN = 16;
    constexpr int64_t PRIORITY_SPAN = 8;
    const bool baselineTile = tile->blockM == BASELINE_BLOCK_M && tile->blockN == BASELINE_BLOCK_N;
    const int64_t tileRank = baselineTile
                                 ? 0
                                 : 1 + (tile->blockM / GFX942_ATTENTION_DENSE_TILE_QUANTUM) * 17
                                       + tile->blockN / GFX942_ATTENTION_DENSE_TILE_QUANTUM;
    const int64_t padRank = armRank(*ldsRowPad, BASELINE_LDS_ROW_PAD, 4, PAD_SPAN);
    const int64_t priorityRank = armRank(*pvPriority, BASELINE_PV_PRIORITY, 1, PRIORITY_SPAN);
    const int64_t rank
        = (std::min(tileRank, TILE_SPAN - 1) * PAD_SPAN + padRank) * PRIORITY_SPAN + priorityRank;
    return static_cast<double>(RANK_SPAN - rank);
}

} // namespace

MetadataValues gfx942AttentionDenseLaunchValues(const AttentionDenseBinding& binding,
                                                const AttentionDenseProblem& problem,
                                                int64_t blockM)
{
    return attention_dense::launchValues(binding, problem, blockM, GFX942_ARCH);
}

Gfx942AttentionDenseLaunchInputs
    readGfx942AttentionDenseLaunchInputs(const SavedLaunchInputs& inputs)
{
    return attention_dense::readLaunchInputs(inputs, GFX942_ARCH);
}

compilation::KpackModuleCache& gfx942AttentionDenseKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx942AttentionDenseModuleCache()
{
    gfx942AttentionDenseKpackModuleCache().clear();
}

namespace
{

/// This engine's dispatch handler, process-lifetime.
const IKernelDispatchHandler<Handle>& gfx942AttentionDenseDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx942AttentionDenseKpackModuleCache());
    static const auto s_dispatchHandler = attention_dense::makeDispatchHandler(
        s_kpackLoader, GFX942_ARCH, GFX942_ARCH.currentContract);
    return *s_dispatchHandler;
}

} // namespace

void registerGfx942AttentionDenseSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx942AttentionDenseGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    // One handler under both names: the UDD's unversioned name dispatches the catalog, and
    // a save stores the versioned name, which must resolve to the handler that saved it.
    scope.add(std::string(DISPATCH_SYMBOL), &gfx942AttentionDenseDispatchHandler());
    scope.add(std::string(GFX942_ATTENTION_DENSE_DISPATCH_SYMBOL_V1),
              &gfx942AttentionDenseDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
