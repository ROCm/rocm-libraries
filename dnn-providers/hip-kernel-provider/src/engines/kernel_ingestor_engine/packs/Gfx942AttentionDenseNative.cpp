// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

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
 * dispatch, the [2^-64, 2^4] scale range included, are shared with the gfx950 twin
 * (AttentionDenseNative.cpp); what is gfx942's own is bound here: the gfx942 tile rules, the
 * causal sliding window baked per candidate with this engine's zero-trip guard, the stats
 * (LSE) output an emit_lse candidate writes, the launch grid each candidate is built for
 * (the qb_minor, bt_hkv_minor or hq_minor_swz block order, or the D128 persistent grid),
 * and the v1 launch contract.
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

// The KMD fields only this engine's schema carries (27 fields; the shared ones are read in
// AttentionDenseNative.cpp). The cold ranking reads lds_row_pad and pv_priority to order
// arms; they do not enter the launch. emit_lse selects the candidates of a graph that
// binds a stats output, and the launch's trailing lse_ptr. persistent, num_persistent,
// persist_decode and nonpersist_decode are the launch grid: its CTA count and order, and
// the persistent grid's fast-division kernargs. lds_num_buffers, v_reg_prefetch,
// causal_diag_split, o_store_width, pv_loop_order, use_cfvst, use_exp2_fast and
// lds_k_group_pad are body-only and enter only the cold ranking. waves_per_eu and the other
// gfx942 levers are absent: every catalog variant holds them at the kernel's own policy. A
// variant that moved any of them would have to add it to the KMD first, or the candidates
// collide on the catalog key and the loader drops one.
constexpr std::string_view LDS_ROW_PAD_FIELD = "lds_row_pad";
constexpr std::string_view PV_PRIORITY_FIELD = "pv_priority";
constexpr std::string_view EMIT_LSE_FIELD = "emit_lse";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view SEQLEN_KV_FIELD = "seqlen_kv";
constexpr std::string_view PERSISTENT_FIELD = "persistent";
constexpr std::string_view NUM_PERSISTENT_FIELD = "num_persistent";
constexpr std::string_view PERSIST_DECODE_FIELD = "persist_decode";
constexpr std::string_view NONPERSIST_DECODE_FIELD = "nonpersist_decode";
constexpr std::string_view LDS_NUM_BUFFERS_FIELD = "lds_num_buffers";
constexpr std::string_view V_REG_PREFETCH_FIELD = "v_reg_prefetch";
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view CAUSAL_DIAG_SPLIT_FIELD = "causal_diag_split";
constexpr std::string_view O_STORE_WIDTH_FIELD = "o_store_width";
constexpr std::string_view PV_LOOP_ORDER_FIELD = "pv_loop_order";
constexpr std::string_view USE_CFVST_FIELD = "use_cfvst";
constexpr std::string_view USE_EXP2_FAST_FIELD = "use_exp2_fast";
constexpr std::string_view LDS_K_GROUP_PAD_FIELD = "lds_k_group_pad";

/// The policy arms cold selection prefers: the 256/64 tile, the spec's default K row pad
/// (`_DEFAULT_LDS_ROW_PAD`), the default PV priority, one K buffer, no V prefetch, no
/// diagonal split, the 4-element O store (`_DEFAULT_O_STORE_WIDTH`), the d_major PV loop
/// (`_DEFAULT_PV_LOOP_ORDER`) and the default K-group pad (AttentionDenseSpec). use_cfvst
/// and use_exp2_fast have no fixed policy value: gfx942AttentionDenseCfvstPolicy and
/// gfx942AttentionDenseExp2FastPolicy give it per dtype, head size and grid.
constexpr int64_t BASELINE_BLOCK_M = 256;
constexpr int64_t BASELINE_BLOCK_N = 64;
constexpr int64_t BASELINE_LDS_ROW_PAD = 8;
constexpr int64_t BASELINE_PV_PRIORITY = 0;
constexpr int64_t BASELINE_LDS_NUM_BUFFERS = 1;
constexpr int64_t BASELINE_V_REG_PREFETCH = 0;
constexpr int64_t BASELINE_CAUSAL_DIAG_SPLIT = 0;
constexpr int64_t BASELINE_O_STORE_WIDTH = 4;
constexpr std::string_view BASELINE_PV_LOOP_ORDER = "d_major";
constexpr std::string_view ALTERNATIVE_PV_LOOP_ORDER = "k_major";
constexpr int64_t BASELINE_LDS_K_GROUP_PAD = 8;

/// Persistent is the cold choice only at this head size, measured ahead of the grid at D128
/// for every batch and mask (dispatch/attention/gfx942_dense.py tuned_base_spec)...
constexpr int64_t PERSISTENT_HEAD_SIZE = 128;
/// ...and only without a narrow window: a causal band of at most this many keys goes to the
/// grid (_NARROW_WINDOW)...
constexpr int64_t NARROW_WINDOW = 256;
/// ...and only where the catalog's persistent CTA count (_GFX942_NUM_PERSISTENT, the one
/// every persistent variant bakes) is a multiple of the device's CU count.
constexpr int64_t CATALOG_NUM_PERSISTENT = 304;

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

/**
 * @brief The launch grid a candidate is built for: the nonpersist_decode block order when
 *        persistent is 0, or num_persistent CTAs at persist_decode when it is 1.
 *
 * A record without the fields reaches here completed to the qb_minor grid through the KMD
 * defaults, so an absent, mistyped or unknown value is malformed and answers nullopt. So
 * does a persistent record whose decode is `auto` (it resolves from the build shape, so its
 * kernel arguments are unknown here) or gqa_pair* (it bakes the shape), or whose CTA count
 * is not in [1, INT32_MAX].
 */
std::optional<attention_dense::AttentionDenseGrid> candidateGrid(const KernelDefinition& kernel)
{
    using attention_dense::AttentionDenseGrid;
    using attention_dense::AttentionDenseGridOrder;
    const auto persistent = attention_dense::integerMetadata(kernel, PERSISTENT_FIELD);
    if(!persistent.has_value() || (*persistent != 0 && *persistent != 1))
    {
        return std::nullopt;
    }
    AttentionDenseGrid grid;
    if(*persistent == 0)
    {
        const auto order = attention_dense::stringMetadata(kernel, NONPERSIST_DECODE_FIELD);
        const auto parsed
            = order.has_value() ? attention_dense::parseNonpersistDecode(*order) : std::nullopt;
        if(!parsed.has_value())
        {
            return std::nullopt;
        }
        grid.order = *parsed;
        return grid;
    }
    const auto numPersistent = attention_dense::integerMetadata(kernel, NUM_PERSISTENT_FIELD);
    const auto decode = attention_dense::stringMetadata(kernel, PERSIST_DECODE_FIELD);
    const auto parsed
        = decode.has_value() ? attention_dense::parsePersistDecode(*decode) : std::nullopt;
    if(!numPersistent.has_value() || *numPersistent < 1
       || *numPersistent > std::numeric_limits<int32_t>::max() || !parsed.has_value())
    {
        return std::nullopt;
    }
    grid.order = AttentionDenseGridOrder::PERSISTENT;
    grid.decode = *parsed;
    grid.numPersistent = *numPersistent;
    return grid;
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
    arch.isSupportedTile = &isSupportedGfx942AttentionDenseTile;
    arch.geometry = &gfx942Geometry;
    arch.windowReachesKeys = &gfx942AttentionDenseWindowReachesKeys;
    // An emit_lse candidate writes the stats output; the catalog pairs each kernel with one.
    arch.candidateWritesLse = &candidateWritesLse;
    // Each candidate's grid: a block order of the one-CTA-per-block grid, or (D128) the
    // persistent grid with its fast-division kernargs.
    arch.candidateGrid = &candidateGrid;
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
constexpr int64_t RANK_SPAN = int64_t{1} << 32;

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
 * @brief The rank of a candidate's grid among the grids of one tile, 0 first.
 *
 * Persistent is the cold choice where it measured ahead and its baked CTA count fills the
 * device evenly: head size 128, no window or one wider than NARROW_WINDOW keys, and the
 * catalog's 304 CTAs a multiple of the device's CU count (a 304- or 38-CU part, not a 228-
 * or 80-CU one, where the last round leaves CUs idle). The choice is the graph's and the
 * device's, so every candidate of a graph ranks under the same one. Otherwise the qb_minor grid,
 * the order every graph was served with before, comes first and persistent last. The other
 * block orders follow qb_minor. Among persistent arms, the decode the catalog's base arm
 * pins at that mask comes first (bt_hkv_minor where the decodes balance causal work at the
 * build shape, qb_major elsewhere), then the others in PERSIST_DECODES order.
 */
int64_t gridRank(const attention_dense::AttentionDenseGrid& grid,
                 const KernelDefinition& kernel,
                 int multiProcessorCount)
{
    using attention_dense::AttentionDenseGridOrder;
    using attention_dense::AttentionDensePersistDecode;
    const int64_t headSize = attention_dense::integerMetadata(kernel, HEAD_SIZE_FIELD).value_or(0);
    const int64_t window
        = attention_dense::integerMetadata(kernel, SLIDING_WINDOW_FIELD).value_or(0);
    const bool persistentFirst
        = headSize == PERSISTENT_HEAD_SIZE && (window == 0 || window > NARROW_WINDOW)
          && multiProcessorCount > 0 && CATALOG_NUM_PERSISTENT % multiProcessorCount == 0;
    // The three block orders, and up to five persistent decode ranks (0, then 1 + index).
    constexpr int64_t ORDERS = 3;
    constexpr int64_t DECODE_RANKS = 5;
    if(!grid.persistent())
    {
        const auto orderRank = static_cast<int64_t>(grid.order);
        return persistentFirst ? DECODE_RANKS + orderRank : orderRank;
    }
    // The decode the base arm pins: what auto resolves to at the build shape (B=1, 8 heads,
    // Sq=Skv=512), bt_hkv_minor under balance_causal there and qb_major otherwise.
    const int64_t buildKeys = attention_dense::integerMetadata(kernel, SEQLEN_KV_FIELD).value_or(0);
    const bool balanced = attention_dense::integerMetadata(kernel, CAUSAL_FIELD).value_or(0) == 1
                          && (window <= 0 || window >= buildKeys);
    const auto policy = balanced ? AttentionDensePersistDecode::BT_HKV_MINOR
                                 : AttentionDensePersistDecode::QB_MAJOR;
    const int64_t decodeRank = grid.decode == policy ? 0 : 1 + static_cast<int64_t>(grid.decode);
    return persistentFirst ? decodeRank : ORDERS + decodeRank;
}

/**
 * @brief A 0/1 body switch's rank: 0 at its policy value, 1 at the other; nullopt for a
 *        missing value or one other than 0 or 1.
 */
std::optional<int64_t> switchRank(std::optional<int64_t> value, bool policy)
{
    if(!value.has_value() || (*value != 0 && *value != 1))
    {
        return std::nullopt;
    }
    return (*value == 1) == policy ? 0 : 1;
}

/**
 * @brief The PV loop order's rank: 0 at d_major, 1 at k_major; nullopt for a missing or
 *        unknown order.
 */
std::optional<int64_t> pvLoopOrderRank(const KernelDefinition& kernel)
{
    const auto order = attention_dense::stringMetadata(kernel, PV_LOOP_ORDER_FIELD);
    if(order == BASELINE_PV_LOOP_ORDER)
    {
        return 0;
    }
    if(order == ALTERNATIVE_PV_LOOP_ORDER)
    {
        return 1;
    }
    return std::nullopt;
}

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * A stable fallback order for cold selection, not a performance model. The rank is
 * lexicographic over (grid, tile, lds_row_pad, pv_priority, lds_num_buffers,
 * v_reg_prefetch, causal_diag_split, o_store_width, pv_loop_order, use_cfvst,
 * use_exp2_fast, lds_k_group_pad), and each knob places its policy arm first (gridRank; the
 * 256/64 tile, lds_row_pad 8, pv_priority 0, one K buffer, no V prefetch, no diagonal split,
 * the 4-element O store, d_major, the cfvst and exp2_fast policies of the candidate's
 * dtype, head size and grid, K-group pad 8) and its alternatives in ascending order. Every
 * distinct knob combination gets a distinct score, so the selector's descriptor-id
 * tie-break never decides between two arms. emit_lse is not ranked: kernelMatches admits
 * the LSE twins of a graph that binds stats and the others of one that does not, so a twin
 * and its partner never compete.
 */
double scoreKernel(const MatchContext& context,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto tile = attention_dense::candidateTile(kernel, GFX942_ARCH);
    const auto grid = candidateGrid(kernel);
    const auto ldsRowPad = attention_dense::integerMetadata(kernel, LDS_ROW_PAD_FIELD);
    const auto pvPriority = attention_dense::integerMetadata(kernel, PV_PRIORITY_FIELD);
    const auto ldsNumBuffers = attention_dense::integerMetadata(kernel, LDS_NUM_BUFFERS_FIELD);
    const auto vRegPrefetch = attention_dense::integerMetadata(kernel, V_REG_PREFETCH_FIELD);
    const auto diagSplit = attention_dense::integerMetadata(kernel, CAUSAL_DIAG_SPLIT_FIELD);
    const auto oStoreWidth = attention_dense::integerMetadata(kernel, O_STORE_WIDTH_FIELD);
    const auto kGroupPad = attention_dense::integerMetadata(kernel, LDS_K_GROUP_PAD_FIELD);
    const auto loopOrderRank = pvLoopOrderRank(kernel);
    if(!tile.has_value() || !grid.has_value() || !ldsRowPad.has_value() || !pvPriority.has_value()
       || !ldsNumBuffers.has_value() || !vRegPrefetch.has_value() || !diagSplit.has_value()
       || !oStoreWidth.has_value() || !kGroupPad.has_value() || !loopOrderRank.has_value()
       || *ldsRowPad < 0 || *pvPriority < 0 || *ldsNumBuffers < 1 || *vRegPrefetch < 0
       || *diagSplit < 0 || *oStoreWidth < 1 || *kGroupPad < 0)
    {
        return UNSUPPORTED_SCORE;
    }
    const std::string dtype
        = attention_dense::stringMetadata(kernel, DTYPE_FIELD).value_or(std::string());
    const int64_t headSize = attention_dense::integerMetadata(kernel, HEAD_SIZE_FIELD).value_or(0);
    const auto cfvstRank = switchRank(attention_dense::integerMetadata(kernel, USE_CFVST_FIELD),
                                      gfx942AttentionDenseCfvstPolicy(dtype, headSize));
    const auto exp2FastRank
        = switchRank(attention_dense::integerMetadata(kernel, USE_EXP2_FAST_FIELD),
                     gfx942AttentionDenseExp2FastPolicy(dtype, headSize, grid->persistent()));
    if(!cfvstRank.has_value() || !exp2FastRank.has_value())
    {
        return UNSUPPORTED_SCORE;
    }

    // block_m <= 512 and block_n <= block_m, both multiples of 32: at most 17 * 17 tiles.
    constexpr int64_t GRID_SPAN = 8;
    constexpr int64_t TILE_SPAN = 512;
    constexpr int64_t PAD_SPAN = 16;
    constexpr int64_t PRIORITY_SPAN = 8;
    constexpr int64_t BUFFERS_SPAN = 2;
    constexpr int64_t PREFETCH_SPAN = 2;
    constexpr int64_t SWITCH_SPAN = 2;
    constexpr int64_t O_STORE_SPAN = 4;
    constexpr int64_t K_GROUP_PAD_SPAN = 8;
    const bool baselineTile = tile->blockM == BASELINE_BLOCK_M && tile->blockN == BASELINE_BLOCK_N;
    const int64_t tileRank = baselineTile
                                 ? 0
                                 : 1 + (tile->blockM / GFX942_ATTENTION_DENSE_TILE_QUANTUM) * 17
                                       + tile->blockN / GFX942_ATTENTION_DENSE_TILE_QUANTUM;
    const int64_t gridRankValue = std::min(
        gridRank(*grid, kernel, context.deviceProperties.multiProcessorCount), GRID_SPAN - 1);
    const int64_t padRank = armRank(*ldsRowPad, BASELINE_LDS_ROW_PAD, 4, PAD_SPAN);
    const int64_t priorityRank = armRank(*pvPriority, BASELINE_PV_PRIORITY, 1, PRIORITY_SPAN);
    const int64_t buffersRank = armRank(*ldsNumBuffers, BASELINE_LDS_NUM_BUFFERS, 1, BUFFERS_SPAN);
    const int64_t prefetchRank = armRank(*vRegPrefetch, BASELINE_V_REG_PREFETCH, 1, PREFETCH_SPAN);
    const int64_t diagSplitRank = armRank(*diagSplit, BASELINE_CAUSAL_DIAG_SPLIT, 1, SWITCH_SPAN);
    const int64_t oStoreRank = armRank(*oStoreWidth, BASELINE_O_STORE_WIDTH, 1, O_STORE_SPAN);
    const int64_t kGroupPadRank
        = armRank(*kGroupPad, BASELINE_LDS_K_GROUP_PAD, 8, K_GROUP_PAD_SPAN);
    // Each step multiplies the rank so far by the next knob's span and adds its arm.
    int64_t rank = gridRankValue;
    for(const auto& [armRankValue, span] : {std::pair{std::min(tileRank, TILE_SPAN - 1), TILE_SPAN},
                                            std::pair{padRank, PAD_SPAN},
                                            std::pair{priorityRank, PRIORITY_SPAN},
                                            std::pair{buffersRank, BUFFERS_SPAN},
                                            std::pair{prefetchRank, PREFETCH_SPAN},
                                            std::pair{diagSplitRank, SWITCH_SPAN},
                                            std::pair{oStoreRank, O_STORE_SPAN},
                                            std::pair{*loopOrderRank, SWITCH_SPAN},
                                            std::pair{*cfvstRank, SWITCH_SPAN},
                                            std::pair{*exp2FastRank, SWITCH_SPAN},
                                            std::pair{kGroupPadRank, K_GROUP_PAD_SPAN}})
    {
        rank = rank * span + armRankValue;
    }
    return static_cast<double>(RANK_SPAN - rank);
}

} // namespace

MetadataValues gfx942AttentionDenseLaunchValues(const AttentionDenseBinding& binding,
                                                const AttentionDenseProblem& problem,
                                                int64_t blockM,
                                                const attention_dense::AttentionDenseGrid& grid)
{
    return attention_dense::launchValues(binding, problem, blockM, GFX942_ARCH, grid);
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
