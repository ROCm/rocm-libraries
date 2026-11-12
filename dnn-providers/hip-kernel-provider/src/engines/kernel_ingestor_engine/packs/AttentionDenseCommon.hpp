// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

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

#include "engines/kernel_ingestor_engine/packs/AttentionDensePersistent.hpp"
#include "engines/kernel_ingestor_engine/packs/SdpaGraphMatch.hpp"

/**
 * @file AttentionDenseCommon.hpp
 * @brief What the gfx942 and gfx950 dense-attention ingestor packs share: the launch state
 *        and its saved form, and the matching and dispatch both engines run, parameterised
 *        by an AttentionDenseArch descriptor.
 *
 * The two kernels are `rocke/library/kernels/gfx942/attention_dense.py` and
 * `rocke/library/kernels/gfx950/attention_dense.py`. They take the same ABI, launch on the
 * same grid formula and are bound from the same graph contract, so one copy of that code
 * serves both engines. What differs per arch reaches it through the descriptor: the tile
 * rules and launch geometry, whether a causal sliding window is served, whether the
 * candidate's head counts are compared, the launch grids its catalog carries
 * (AttentionDensePersistent.hpp), and the launch contracts. Scoring and
 * every read of a KMD field only one engine declares stay in that engine's own
 * `Gfx9xxAttentionDenseNative.cpp`, which also owns the module cache and registration.
 */

// The provider's plugin handle (core/Handle.hpp), named here only as the handler's handle type.
struct Handle;

namespace hip_kernel_provider::compilation
{
class KpackKernelLoader;
}

namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The stats uid of a graph that binds no stats (LSE) output.
inline constexpr int64_t ATTENTION_DENSE_NO_STATS = -1;

/// The tensor uids and derived scalars a matched dense-attention graph binds.
struct AttentionDenseBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
    int64_t slidingWindow = 0;
    float scale = 0.0F;
    /// The stats output the kernel writes the LSE to, or ATTENTION_DENSE_NO_STATS. Only an
    /// arch that serves stats (AttentionDenseArch::servesStats) ever binds one.
    int64_t stats = ATTENTION_DENSE_NO_STATS;
};

/// The graph facts the matcher and prepare() both need, derived once from the tensors by
/// sdpa_graph::problemFor().
using AttentionDenseProblem = sdpa_graph::SdpaShape;

/// Everything a dense-attention launch is computed from.
struct AttentionDenseLaunchInputs
{
    AttentionDenseBinding binding;
    AttentionDenseProblem problem;
    int64_t blockM = 0;
    /// The candidate's launch grid: the qb_minor grid unless its arch reads another
    /// (AttentionDenseArch::candidateGrid).
    attention_dense::AttentionDenseGrid grid;
};

namespace attention_dense
{

/// The launch values both engines' contracts store. `scale` is stored as a double, which
/// holds every float exactly. `data_type` is the DataType enumerator name.
inline constexpr std::string_view Q_UID_VALUE = "q.uid";
inline constexpr std::string_view K_UID_VALUE = "k.uid";
inline constexpr std::string_view V_UID_VALUE = "v.uid";
inline constexpr std::string_view O_UID_VALUE = "o.uid";
inline constexpr std::string_view CAUSAL_VALUE = "causal";
inline constexpr std::string_view SLIDING_WINDOW_VALUE = "sliding_window";
inline constexpr std::string_view BATCH_VALUE = "batch";
inline constexpr std::string_view SEQLEN_Q_VALUE = "seqlen_q";
inline constexpr std::string_view SEQLEN_KV_VALUE = "seqlen_kv";
inline constexpr std::string_view NUM_QUERY_HEADS_VALUE = "num_query_heads";
inline constexpr std::string_view NUM_KV_HEADS_VALUE = "num_kv_heads";
inline constexpr std::string_view HEAD_SIZE_VALUE = "head_size";
inline constexpr std::string_view BLOCK_M_VALUE = "block_m";
inline constexpr std::string_view SCALE_VALUE = "scale";
inline constexpr std::string_view DATA_TYPE_VALUE = "data_type";
inline constexpr std::string_view STRIDE_LAYOUT_VALUE = "stride_layout";
/// The stats output's uid, or ATTENTION_DENSE_NO_STATS. Saved only by an arch that serves
/// stats; it also selects the kernel arguments a restored plan launches.
inline constexpr std::string_view STATS_UID_VALUE = "stats.uid";
/// The launch grid, saved only by an arch whose catalog carries more than the qb_minor
/// grid: `grid_order` (qb_minor, bt_hkv_minor, hq_minor_swz or persistent), and on the
/// persistent grid its `persist_decode` and `num_persistent` (empty and 0 off it). They
/// also select the kernel arguments a restored plan launches. The fast-division magic
/// numbers and the grid are recomputed, never saved.
inline constexpr std::string_view GRID_ORDER_VALUE = "grid_order";
inline constexpr std::string_view PERSIST_DECODE_VALUE = "persist_decode";
inline constexpr std::string_view NUM_PERSISTENT_VALUE = "num_persistent";

/// The one stride layout the kernels support. prepare() refuses every other layout.
inline constexpr std::string_view BSHD_LAYOUT = "bshd";

/// The kernel arguments a launch contract passes.
enum class AttentionDenseAbi
{
    /// q, k, v, o, scale, batch, seqlen_q, seqlen_kv: kernels that bake the head counts.
    BAKED_HEAD_COUNTS,
    /// The same, then num_query_heads and num_kv_heads: kernels that read them at runtime.
    RUNTIME_HEAD_COUNTS
};

/// A versioned dispatch name, under which a plan's launch values are saved, and the kernel
/// arguments a launch under it passes.
struct AttentionDenseContract
{
    std::string_view symbol;
    AttentionDenseAbi abi = AttentionDenseAbi::RUNTIME_HEAD_COUNTS;
};

/// The tokens an engine's graph match binds, spelled per engine.
struct AttentionDenseTokens
{
    std::string_view q;
    std::string_view k;
    std::string_view v;
    std::string_view o;
    /// The mask type graphMatches derived, so kernelMatches does not re-derive it.
    std::string_view causal;
    /// The sliding window the candidate must have been built with: the number of keys a
    /// query attends, counting the diagonal (the graph's left_bound), or 0 for none.
    std::string_view slidingWindow;
    /// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
    std::string_view scaleBits;
    /// The stats output's uid, or ATTENTION_DENSE_NO_STATS. Bound only by an arch that
    /// serves stats; empty for one that does not.
    std::string_view stats;
};

/// The grid and block a variant must launch with.
struct AttentionDenseGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;
};

/**
 * @brief Everything the shared matching and dispatch ask of one arch.
 *
 * Each engine defines one at namespace scope in its own native file. The function pointers
 * are that arch's restatement of its own Python (its Gfx9xxAttentionDenseGeometry.hpp),
 * except the window guard, which is this engine's own policy.
 */
struct AttentionDenseArch
{
    /// "gfx942" or "gfx950": the head of every exception this module throws for the arch.
    std::string_view label;
    /// The engine name its decline lines carry.
    std::string_view engineName;
    AttentionDenseTokens tokens;
    /// Whether kernelMatches compares the candidate's num_query_heads/num_kv_heads with the
    /// graph's: true where the catalog lists each head configuration as its own entry.
    bool matchesHeadCounts = false;
    /// Whether (blockM, blockN) is a tile the kernel is built with at headSize. Total over
    /// every int64 triple.
    bool (*isSupportedTile)(int64_t headSize, int64_t blockM, int64_t blockN) = nullptr;
    /// The launch geometry of a candidate's own block_m over a graph's shape. Throws
    /// HipdnnPluginException with HIPDNN_PLUGIN_STATUS_BAD_PARAM for a block_m the arch
    /// does not build or a non-positive extent.
    AttentionDenseGeometry (*geometry)(int64_t blockM,
                                       int64_t seqLenQ,
                                       int64_t numQueryHeads,
                                       int64_t batch,
                                       const std::string& kernelName)
        = nullptr;
    /// This engine's sliding-window zero-trip guard for one candidate's tile, or nullptr for
    /// an arch that serves no window: its graph match declines every windowed mask.
    bool (*windowReachesKeys)(
        int64_t blockM, int64_t blockN, int64_t seqLenQ, int64_t seqLenKv, int64_t slidingWindow)
        = nullptr;
    /// Whether a candidate writes the softmax statistics (LSE) through a trailing lse_ptr,
    /// read from a KMD field only that arch's manifest declares; nullopt for a record
    /// without a valid value. nullptr for an arch whose catalog has no LSE kernel: its
    /// graph match declines every stats output under the one cause 'stats', binds no stats
    /// token, and its launch values carry no stats uid.
    std::optional<bool> (*candidateWritesLse)(
        const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
        = nullptr;
    /// The launch grid a candidate is built for, read from KMD fields only that arch's
    /// manifest declares; nullopt for a record without a valid one, which kernelMatches
    /// declines. nullptr for an arch whose catalog has only the qb_minor grid: every
    /// candidate launches it, and its launch values carry no grid.
    std::optional<AttentionDenseGrid> (*candidateGrid)(
        const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
        = nullptr;
    /// The contract the engine's catalog dispatches through and new plans are saved under.
    AttentionDenseContract currentContract;
    /// A superseded contract whose plans still restore, or an empty symbol for none. It
    /// shares currentContract's launch values and differs only in its kernel arguments.
    AttentionDenseContract legacyContract;

    bool servesSlidingWindow() const
    {
        return windowReachesKeys != nullptr;
    }

    bool servesStats() const
    {
        return candidateWritesLse != nullptr;
    }

    bool servesGrids() const
    {
        return candidateGrid != nullptr;
    }
};

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * Each decline of a graph with an SDPA-forward node logs its cause at INFO; any other
 * graph declines silently.
 *
 * @warning Returning std::nullopt empties the engine's WHOLE catalog and skips EVERY
 *          remaining pack, not just this one.
 */
std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>
    graphMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                 const AttentionDenseArch& arch);

/// Kernel-scoped applicability: does THIS candidate's baked metadata fit the graph?
bool kernelMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                   const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                   const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                   const AttentionDenseArch& arch);

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total: never throws, so a malformed record declines instead of faulting a match.
std::optional<int64_t> integerMetadata(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                                       std::string_view field);

/// A candidate's string metadata field, or nullopt when it is absent or holds another type.
/// Total, as integerMetadata.
std::optional<std::string>
    stringMetadata(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                   std::string_view field);

/// The launch shape of @p launch, for AttentionDensePersistent.hpp's grid functions.
AttentionDenseGridShape gridShape(const AttentionDenseLaunchInputs& launch);

/// A candidate's (block_m, block_n) tile, read from its completed metadata.
struct AttentionDenseTile
{
    int64_t blockM = 0;
    int64_t blockN = 0;
};

/**
 * @brief The candidate's tile, or nullopt when it has none the arch can launch.
 *
 * Validated against the arch's isSupportedTile. A record omitting block_m or block_n
 * reaches here completed to 256/64 through the KMD defaults, so an absent field means an
 * uncompleted or malformed record: it is declined rather than given a default here,
 * because a substituted tile launches a binary with another binary's grid.
 */
std::optional<AttentionDenseTile>
    candidateTile(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                  const AttentionDenseArch& arch);

/**
 * @brief The native dispatch behind an engine's UDD, for one launch contract.
 *
 * The handler of a contract saves new plans under that contract's name and restores only
 * plans saved under it. Process-lifetime: the caller keeps it alive for as long as it is
 * registered, and @p kpackLoader and @p arch for as long as the handler lives.
 */
std::unique_ptr<hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>>
    makeDispatchHandler(const compilation::KpackKernelLoader& kpackLoader,
                        const AttentionDenseArch& arch,
                        const AttentionDenseContract& contract);

/// The launch values @p arch saves, one value set for every contract of the arch: the
/// sixteen both engines save, `stats.uid` where the arch serves stats, and `grid_order`,
/// `persist_decode` and `num_persistent` where it serves grids past qb_minor.
hipdnn_plugin_sdk::ingestor::MetadataValues launchValues(const AttentionDenseBinding& binding,
                                                         const AttentionDenseProblem& problem,
                                                         int64_t blockM,
                                                         const AttentionDenseArch& arch,
                                                         const AttentionDenseGrid& grid = {});

/**
 * @brief Reads @p arch's launch values back into the launch state.
 *
 * The dispatch name must be the arch's current or legacy contract; anything else, including
 * the unversioned name, is refused against the current one. `batch`, `seqlen_q`,
 * `seqlen_kv`, the head counts and `head_size` must be in [1, INT32_MAX]; the kernel takes
 * the first three as 32-bit integers. `block_m` must be one the arch's geometry accepts;
 * block_n is not saved, so the arch's full tile rule is not re-run. `scale` must convert to
 * a float and back without change, within [2^-64, 2^4]. `data_type` must be a type the
 * kernel is built for, and `stride_layout` must be `bshd`.
 *
 * As both arches' graph and kernel match require, `causal` must be 0 or 1; `sliding_window`
 * non-negative, 0 unless `causal` is 1, and not so narrow that the last query rows see no
 * key (seqlen_q - seqlen_kv >= sliding_window); `num_query_heads` a multiple of
 * `num_kv_heads`; `head_size` 64 or 128; K/V below 2^31 bytes and Q/O below 2^31
 * elements; and `seqlen_q` a multiple of `block_m`, since the kernel addresses Q and O
 * without a bounds check. One exception: gfx950's graph match declines every window, but
 * its reader still takes a causal one within these rules: its frozen v1/v2 samples hold one.
 *
 * An arch that serves stats also reads `stats.uid`, which must be ATTENTION_DENSE_NO_STATS
 * or a uid (non-negative). An arch that serves grids also reads `grid_order`, which must
 * name one; on the persistent grid `persist_decode` must name a decode the kernel reads the
 * shape for at run time (qb_major, hkv_major, bt_hkv_minor or hq_minor_swz; never `auto`
 * or gqa_pair*) and `num_persistent` be in [1, INT32_MAX], and off it they must be empty
 * and 0. The grid must be legal at the saved shape (AttentionDensePersistent.hpp
 * gridIllegality). The restore selects the kernel's argument list from the saved stats uid
 * and grid before this reader runs, so its signature check, which this reader does not
 * see, refuses a kernel whose LSE output or grid disagrees with the plan.
 *
 * @throws HipdnnPluginException with a load refusal when the dispatch name is neither
 *         contract, or when a value is missing, extra, of another type or out of range.
 */
AttentionDenseLaunchInputs
    readLaunchInputs(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                     const AttentionDenseArch& arch);

} // namespace attention_dense
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
