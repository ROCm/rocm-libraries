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
 * @brief Sweep fractional denominators to find an SK grid that fits in workspace.
 *
 * Shared helper for the "more tiles than CUs" branch of StreamK grid selection.
 * For each fraction `f` in @p tile_fractions, compute the candidate grid
 * `round(tiles / (tiles/cu_count + f))` and return the first candidate that
 *   - is non-zero,
 * - fits within the workspace budget when it does not divide @p tiles evenly
 *     (`tile_size * candidate <= workspace_limit`), and
 *   - is `<= cu_count`.
 *
 * Workspace check is skipped when @p workspace_limit is 0.
 *
 * @param tiles            Total number of output tiles.
 * @param cu_count         Maximum CU count (or virtual CU count) to fit under.
 * @param tile_size        Per-tile workspace cost in bytes (e.g. mt.m * mt.n * bytes_per_elem_c).
 * @param workspace_limit  Maximum allowed workspace in bytes (0 = unlimited).
 * @param tile_fractions   Fractional denominators to sweep, in priority order.
 *                         Conventional default is {0.0, 0.5, 0.125, 0.2, 0.25, 1.0/3.0}.
 * @return size_t The chosen sk_grid, or 0 if no candidate satisfies the constraints.
 */
ORIGAMI_EXPORT size_t pick_fractional_grid(size_t tiles,
                                           size_t cu_count,
                                           size_t tile_size,
                                           size_t workspace_limit,
                                           const std::vector<double>& tile_fractions);

/**
 * @brief Pick a K-split factor when there are fewer tiles than CUs.
 *
 * Shared helper for the "fewer tiles than CUs" branch of StreamK grid selection.
 * For each integer factor `f` in @p k_split_factors (high-to-low), pick the
 * first that satisfies:
 *   - `tiles * f <= cu_count`, and
 *   - `iters_per_tile / f >= min_iters_per_cu`.
 *
 * @param tiles             Number of output tiles.
 * @param cu_count          CU count to spread across.
 * @param iters_per_tile    Number of K iterations per tile.
 * @param k_split_factors   Candidate split factors, high-to-low.
 *                          Conventional default is {16, 12, 8, 6, 4, 3, 2, 1}.
 * @param min_iters_per_cu  Lower bound on iterations per CU (default 8).
 * @return size_t `tiles * f` for the first satisfying `f`, or 0 if none.
 */
ORIGAMI_EXPORT size_t pick_k_split(size_t tiles,
                                   size_t cu_count,
                                   size_t iters_per_tile,
                                   const std::vector<size_t>& k_split_factors,
                                   int min_iters_per_cu = 8);

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
 * @param max_cus Maximum number of CUs to use.
 * @return size_t Dimensions of the grid launched.
 */
ORIGAMI_EXPORT size_t select_grid_size(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config,
                        grid_selection_t algorithm,
                        size_t max_cus = 0);

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

}  // namespace streamk
}  // namespace origami
