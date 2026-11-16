/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "origami/hardware.hpp"
#include "origami/types.hpp"
#include "origami/origami_export.h"

#include <cstdint>
#include <vector>

namespace origami {
namespace streamk {
/**
 * @brief Number of output tiles.
 *
 * @param mt_m Tile size in M-dimension.
 * @param mt_n Tile size in N-dimension.
 * @param m Matrix's m-dimension.
 * @param n Matrix's n-dimension.
 * @param batch Number of batches.
 * @return size_t Total number of output tiles.
 */
ORIGAMI_EXPORT size_t compute_number_of_output_tiles(size_t mt_m, size_t mt_n, size_t m, size_t n, size_t batch);

/**
 * @brief Select the best reduction strategy for StreamK.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param algorithm Grid selection algorithm
 * @return reduction_t Selected reduction strategy
 */
ORIGAMI_EXPORT reduction_t select_reduction(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             grid_selection_t algorithm);

/**
 * @brief Based on the provided kernel config, select the best grid dimension.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection_t grid selection algorithm (@see origami::grid_selection_t)
 * @return size_t Dimensions of the grid launched.
 */
ORIGAMI_EXPORT size_t select_grid_size(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config,
                        grid_selection_t algorithm);

/**
 * @brief Pick the SK3-vs-SK4 sub-path for a StreamK=5 hybrid kernel.
 *
 * Decision rule fit to measured SK5 on(SK4)/off(SK3) sweeps on MI350X
 * (gfx950); see origami::streamk_hybrid_defaults_t for the thresholds.
 * Other architectures always return hybrid_mode_t::static_ until they are
 * tuned in a follow-up PR. Gates, in order: grid size (tiles), then whether
 * a cotenant currently holds any CU away from this kernel, then occupancy,
 * falling back to tiles-per-CU only once occupancy alone isn't decisive.
 *
 * @param problem            Problem description (M, N, K, batch).
 * @param hardware           Hardware characteristics (@see origami::hardware_t).
 * @param config             Kernel configuration (provides MT shape and occupancy).
 * @param sm_count_target    Caller's effective CU budget (0 = use all
 *                           CUs the device exposes). When non-zero,
 *                           clamps hardware.N_CU from above.
 * @return hybrid_mode_t::static_ for SK3, hybrid_mode_t::dynamic for SK4.
 */
ORIGAMI_EXPORT hybrid_mode_t select_hybrid_mode(const problem_t& problem,
                                 const hardware_t& hardware,
                                 const config_t& config,
                                 size_t sm_count_target);

// ---------------------------------------------------------------------------------------------
// Launch arithmetic.  hipBLASLt resolves its Stream-K launch (ContractionSolution's
// computeStreamKDecisions / resolveStreamKSettings / makeArgs) with these functions, so the
// kernel arguments, the workspace it reserves and the grid Origami models come from one place.
// "tiles" is the batch-inclusive output tile count and "iters_per_tile" the K iterations of a
// tile clamped to at least 1, as the kernel's ItersPerTile argument.
// ---------------------------------------------------------------------------------------------

/// The K iterations at or above which per-K-split work is worth a workgroup (grid_k_split_aware).
inline constexpr size_t MIN_ITERS_PER_CU = 8;

/// Entries of the flag region a tree-reduced Stream-K launch indexes per workgroup (hipBLASLt
/// StreamKFlagElements).
inline constexpr size_t FLAG_ELEMENTS = 2048;

/**
 * @brief A tile's iterations rounded up to whole K units.
 *
 * A kernel whose K unit spans k_granularity iterations (the host-preswizzled MX scale layout)
 * cannot start a workgroup inside a unit.  Measuring the flat iteration space in padded tiles
 * puts every tile start, and every cut between them, on a unit boundary; the padding is never
 * run.  A granularity of 0 or 1 pads nothing.
 */
ORIGAMI_EXPORT size_t padded_iters_per_tile(size_t iters_per_tile, size_t k_granularity);

/**
 * @brief The static two-tile Stream-K split (StreamK=3 and the static sub-mode of StreamK=5).
 *
 * The kernel derives its schedule from these three values: a data-parallel prefix of
 * tiles - sk_tiles whole tiles and a Stream-K region of sk_tiles tiles cut into grid chunks of
 * sk_iters_per_wg iterations, extra_iters of them one K unit longer (global first-E mapping).
 */
struct ORIGAMI_EXPORT static_split_t {
  std::uint32_t sk_tiles        = 0;
  std::uint32_t sk_iters_per_wg = 0;
  std::uint32_t extra_iters     = 0;
};

/**
 * @brief Compute the static two-tile split.
 *
 * @param tiles          Batch-inclusive tile count.
 * @param iters_per_tile Iterations per tile, at least 1.
 * @param grid           Resolved Stream-K grid; 0 yields an all-zero split.
 * @param full_tiles     Whole tiles per workgroup in the Stream-K region beyond the remainder
 *                       (hipBLASLt TENSILE_STREAMK_FULL_TILES, default 1).
 * @param force_dp_only  The kernel is a persistent data-parallel use of StreamK=3: no
 *                       Stream-K tiles.
 * @param k_granularity  Iterations per indivisible K unit.
 */
ORIGAMI_EXPORT static_split_t static_split(size_t tiles,
                                           size_t iters_per_tile,
                                           size_t grid,
                                           int full_tiles,
                                           bool force_dp_only,
                                           size_t k_granularity = 1);

/**
 * @brief Whether every output tile of a static split folds the same ordered list of chunk
 *        lengths (row uniformity, the uniform-summation-order condition).
 *
 * @param per_tile_capable      The kernel can hand out the extras within each tile
 *                              (hipBLASLt InternalArgsSupport::perTileExtraIters).
 * @param uniform_summation_order The launch requests it; the kernel takes the per-tile mapping
 *                              only when both hold.
 */
ORIGAMI_EXPORT bool static_split_row_uniform(const static_split_t& split,
                                             size_t tiles,
                                             size_t iters_per_tile,
                                             size_t grid                  = 0,
                                             bool per_tile_capable        = false,
                                             bool uniform_summation_order = false,
                                             size_t k_granularity         = 1);

/// Iteration range [start, end) of one workgroup in the flat (padded) iteration space.
struct ORIGAMI_EXPORT iter_range_t {
  size_t start = 0;
  size_t end   = 0;
};

/**
 * @brief The iterations workgroup w owns under the static split's mapping.
 *
 * Per tile when grid % tiles == 0 and both per_tile_capable and uniform_summation_order hold,
 * otherwise the global first-E mapping.  Both hand out the leftover in units of k_granularity
 * iterations.  Returns an empty range for grid, tiles or k_granularity of 0.
 */
ORIGAMI_EXPORT iter_range_t workgroup_iter_range(size_t w,
                                                 size_t tiles,
                                                 size_t iters_per_tile,
                                                 size_t grid,
                                                 bool per_tile_capable,
                                                 bool uniform_summation_order,
                                                 size_t k_granularity = 1);

/**
 * @brief Whether a parallel-reduction launch is row-uniform: non-atomic, static two-tile
 *        packing, and grid an exact multiple of tiles with grid / tiles >= 2.
 */
ORIGAMI_EXPORT bool parallel_reduction_row_uniform(reduction_t reduction,
                                                   size_t grid,
                                                   size_t tiles,
                                                   int stream_k_atomic,
                                                   bool static_two_tile_packing);

/// Work items of a dynamic (work-queue) Stream-K launch.
struct ORIGAMI_EXPORT dynamic_items_t {
  std::uint32_t sk_tiles        = 0;  ///< tiles split into sk_split work items
  std::uint32_t sk_split        = 0;  ///< work items per split tile, after rounding
  std::uint32_t sk_iters_per_wi = 0;  ///< iterations per work item, whole K units
  std::uint32_t total_items     = 0;  ///< (tiles - sk_tiles) + sk_tiles * sk_split
};

/**
 * @brief The work items of a dynamic Stream-K launch: whole tiles, plus sk_split parts per split
 *        tile.  Tiles split only under overrides (hipBLASLt TENSILE_STREAMK_TILES / _SPLIT; -1
 *        means unset: no split tiles, a nominal split of 2).
 */
ORIGAMI_EXPORT dynamic_items_t dynamic_items(size_t tiles,
                                             size_t iters_per_tile,
                                             size_t k_granularity = 1,
                                             int override_tiles   = -1,
                                             int override_split   = -1);

/**
 * @brief Demote parallel reduction to tree when the grid leaves fewer than two workgroups per
 *        tile, which parallel reduction cannot express.  Applied once the grid is final.
 */
ORIGAMI_EXPORT reduction_t reconcile_reduction(reduction_t reduction, size_t grid, size_t tiles);

/// Whether a workgroup can stop part-way through a tile, i.e. the launch writes partial tiles.
ORIGAMI_EXPORT bool straddles_tiles(size_t iters_per_tile, size_t grid, size_t tiles);

/// Bytes of partial-tile workspace a grid reserves: one macro tile of partials per workgroup.
ORIGAMI_EXPORT size_t partial_tile_bytes(size_t mt_m,
                                         size_t mt_n,
                                         size_t workspace_size_per_elem_c,
                                         size_t grid);

/**
 * @brief Snap a selected grid onto a row-uniform one (uniform summation order with static
 *        two-tile packing).
 *
 * Never upward above tiles: grid = tiles * F for the largest F <= grid / tiles that is a
 * multiple of cluster_width in workgroups, divides iters_per_tile (tree reduction without
 * per-tile extras), leaves at least MIN_ITERS_PER_CU iterations per split, fits workspace_size
 * and, when flag_entries is non-zero, stays within flag_entries workgroups; F = 1 (grid =
 * tiles) otherwise.  A grid below tiles becomes tiles.
 *
 * @param per_tile_extra_iters The kernel will run per-tile extras (capability and uniform
 *                             summation order).
 * @param tile_bytes           Partial-tile bytes per workgroup (mt_m * mt_n * bytes per element).
 * @param flag_entries         Flag-region entries the launch's partials would use, 0 when it
 *                             uses none (atomic, persistent data parallel or parallel reduction).
 */
ORIGAMI_EXPORT size_t uniform_grid(size_t grid,
                                   size_t tiles,
                                   size_t iters_per_tile,
                                   reduction_t reduction,
                                   size_t cluster_width,
                                   bool per_tile_extra_iters,
                                   size_t tile_bytes,
                                   size_t workspace_size,
                                   size_t flag_entries);

// ---------------------------------------------------------------------------------------------
// The launch a caller makes, and the quantities that describe it.
// ---------------------------------------------------------------------------------------------

/**
 * @brief The Stream-K launch the caller makes for one kernel on one problem.
 *
 * hipBLASLt: the fields of ContractionSolution::computeStreamKDecisions(problem, hardware), the
 * snapshot solve() takes its launch from, for the problem of the call (CU budget, workspace and
 * Stream-K parameters included), plus the kernel's sizeMapping values.
 */
struct ORIGAMI_EXPORT launch_t {
  bool          valid      = false;  ///< false: not provided
  bool          is_streamk = false;  ///< sizeMapping.isStreamK(); false leaves the rest unused
  std::uint64_t final_grid = 0;      ///< d.finalGrid
  std::uint64_t tiles      = 0;      ///< d.tiles
  std::uint64_t sk_tiles   = 0;      ///< d.skTiles
  std::uint64_t sk_split   = 0;      ///< d.skSplit
  reduction_t   reduction  = reduction_t::none;  ///< d.reduction
  bool          is_dynamic = false;              ///< d.isDynamic
  bool          dp_only    = false;              ///< d.dpOnly
  std::uint64_t iters_per_tile = 0;  ///< max(1, problem.getItersPerTile(sizeMapping))
  std::uint64_t mt_x = 0, mt_y = 0;  ///< sizeMapping.macroTile.{x, y}
  std::uint64_t workspace_size_per_elem_c = 0;  ///< sizeMapping.workspaceSizePerElemC
  int           cu_occupancy = 0;               ///< sizeMapping.CUOccupancy
  std::uint64_t cu_count     = 0;  ///< the device's CU count (AMDGPU::computeUnitCount)
};

/// Fixup partial tiles of a Stream-K launch and the output tiles they cut.
struct ORIGAMI_EXPORT partials_t {
  std::uint64_t partials    = 0;
  std::uint64_t split_tiles = 0;
};

/**
 * @brief Partial tiles a launch writes.
 *
 * Dynamic queue: none.  Parallel reduction with sk_split >= 2: tiles * sk_split, every tile
 * split.  Otherwise the static split: T = sk_tiles * iters_per_tile iterations (0 if dp_only)
 * over G = final_grid workgroups under the global first-E mapping (workgroup g starts at
 * g * (T / G) + min(g, T % G)); each start inside a tile is one partial.
 */
ORIGAMI_EXPORT partials_t launch_partials(const launch_t& launch);

/// What a launch does with the problem's tiles (the OLC feature schema v4 sk_* columns).
struct ORIGAMI_EXPORT launch_terms_t {
  double grid_over_tiles         = 1.0;  ///< final_grid / tiles
  double split_tiles_frac        = 0.0;  ///< tiles with a partial / tiles
  double fixup_bytes_over_output = 0.0;  ///< partials written and read / bytes of D
  double num_wgs                 = 0.0;  ///< final_grid
  double timesteps               = 0.0;  ///< resident waves of the grid, per tile's share
};

/**
 * @brief The terms of a valid launch.  A kernel that is not Stream-K keeps the defaults.
 *
 * Waves are ceil(final_grid / (CUs * max(cu_occupancy, 1))) with CUs the problem's budget
 * (problem.num_cus when positive and below cu_count); timesteps = waves * tiles / final_grid.
 */
ORIGAMI_EXPORT launch_terms_t launch_terms(const problem_t& problem, const launch_t& launch);

}  // namespace streamk
}  // namespace origami
