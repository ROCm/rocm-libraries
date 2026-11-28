// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <tuple>
#include <vector>

#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

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
ORIGAMI_EXPORT size_t
compute_number_of_output_tiles(size_t mt_m, size_t mt_n, size_t m, size_t n, size_t batch);

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
 * (gfx950); see origami::cdna4_constants_t for the thresholds.
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

namespace origami {
namespace gemm {

/**
 * @brief Computes the launch parameters for the kernel.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_selection Different algorithms to select the grid size for kernel execution.
 * @return tuple<reduction_t, size_t, size_t, size_t, size_t>
 *         (reduction_strategy, num_wgs, num_active_cus, num_timesteps, split_factor)
 */
ORIGAMI_EXPORT std::tuple<reduction_t, size_t, size_t, size_t, size_t> compute_launch_parameters(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    grid_selection_t grid_selection);

}  // namespace gemm
}  // namespace origami
