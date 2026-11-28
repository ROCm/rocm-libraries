// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "origami/cost_models/gemm/tile.hpp"
#include "origami/hardware.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Workgroup mapping
// ---------------------------------------------------------------------------

/**
 * @brief Fast WGM prediction based on last-XCD L2 working set minimization.
 *
 * Evaluates a small set of WGM candidates and picks the one that minimizes
 * the L2 working set for the last XCD in the first timestep.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_m Number of workgroups in M dimension.
 * @param grid_n Number of workgroups in N dimension.
 * @param splitting_factor K-split factor.
 * @return workgroup_mapping_t Predicted workgroup mapping.
 */
ORIGAMI_EXPORT workgroup_mapping_t predict_workgroup_mapping(const problem_t& problem,
                                                             const hardware_t& hardware,
                                                             const config_t& config,
                                                             size_t grid_m,
                                                             size_t grid_n,
                                                             size_t splitting_factor);

/**
 * @brief Map a linear workgroup ID to 4D tile coordinates (k, m, n, b) using WGM slab ordering.
 *
 * Dispatch order: k (innermost) -> mn (WGM slab ordering) -> b (outermost).
 * Within the mn space, slabs of width min(wgm, grid.n) are laid out N-first, then M.
 *
 * @param grid Grid dimensions (k, m, n, b).
 * @param wgm_mapping Workgroup mapping parameters (wgmxcc, wgm).
 * @param id Linear workgroup ID.
 * @return dim4_t 4D tile coordinate.
 */
ORIGAMI_EXPORT dim4_t wgm_to_grid(const dim4_t& grid,
                                  const workgroup_mapping_t& wgm_mapping,
                                  size_t id);

/**
 * @brief Count unique tile coordinates touched by a contiguous range of workgroup IDs.
 *
 * Computes how many distinct rows (m), columns (n), K-splits (k), and batches (b)
 * appear in the range [start, start+count) under raw dispatch order (no WGMXCC).
 *
 * @param grid Grid dimensions (k, m, n, b).
 * @param wgm WGM slab width (absolute value).
 * @param start First workgroup ID in the range.
 * @param count Number of workgroups in the range.
 * @return dim4_t Unique tile counts in each dimension.
 */
ORIGAMI_EXPORT dim4_t count_unique_range(const dim4_t& grid, int wgm, size_t start, size_t count);

/**
 * @brief Count unique tiles for a specific XCD during a specific timestep.
 *
 * With wgmxcc, XCD x in timestep ts sees cus_per_xcd consecutive tiles
 * in raw dispatch order. Without wgmxcc, falls back to even division.
 *
 * @param grid Grid dimensions (k, m, n, b).
 * @param wgm_mapping Workgroup mapping parameters (wgmxcc, wgm).
 * @param N_CU Total number of CUs.
 * @param num_xcd Number of XCDs.
 * @param xcd_id XCD index (0-based).
 * @param timestep_id Timestep index (0-based).
 * @return dim4_t Unique tile counts in each dimension.
 */
ORIGAMI_EXPORT dim4_t count_unique_tiles(const dim4_t& grid,
                                         const workgroup_mapping_t& wgm_mapping,
                                         size_t N_CU,
                                         size_t num_xcd,
                                         size_t xcd_id,
                                         size_t timestep_id);

/**
 * @brief Count unique tiles for an entire timestep (all XCDs combined).
 *
 * @param grid Grid dimensions (k, m, n, b).
 * @param wgm_mapping Workgroup mapping parameters (wgmxcc, wgm).
 * @param N_CU Total number of CUs.
 * @param timestep_id Timestep index (0-based).
 * @return dim4_t Unique tile counts in each dimension.
 */
ORIGAMI_EXPORT dim4_t count_unique_tiles_timestep(const dim4_t& grid,
                                                  const workgroup_mapping_t& wgm_mapping,
                                                  size_t N_CU,
                                                  size_t timestep_id);

/**
 * @brief Compute MALL tile dimensions: how many concurrent workgroup tiles fit in MALL.
 *
 * @param grid_m Number of workgroups in M dimension.
 * @param grid_n Number of workgroups in N dimension.
 * @param active_cus Number of active compute units.
 * @param wgm_value Workgroup mapping slab width.
 * @return std::pair<size_t, size_t> (mall_tile_m, mall_tile_n).
 */
ORIGAMI_EXPORT std::pair<size_t, size_t> compute_mall_tiles(size_t grid_m,
                                                            size_t grid_n,
                                                            size_t active_cus,
                                                            size_t wgm_value);

/**
 * @brief Compute L2 tile dimensions: how many tiles share one XCD's L2, shrunk to fit capacity.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param grid_m Number of workgroups in M dimension.
 * @param grid_n Number of workgroups in N dimension.
 * @param active_cus Number of active compute units.
 * @param splitting_factor K-split factor.
 * @param wgm_value Workgroup mapping slab width.
 * @return std::pair<size_t, size_t> (l2_tile_m, l2_tile_n).
 */
ORIGAMI_EXPORT std::pair<size_t, size_t> compute_l2_tiles(const problem_t& problem,
                                                          const hardware_t& hardware,
                                                          const config_t& config,
                                                          size_t grid_m,
                                                          size_t grid_n,
                                                          size_t active_cus,
                                                          size_t splitting_factor,
                                                          size_t wgm_value);

// ---------------------------------------------------------------------------
// Cache hit rates
// ---------------------------------------------------------------------------

/**
 * @brief Per-operand cache hit rates for L2 and MALL.
 *
 * Tuple layout:
 *   (H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B)
 */
using cache_hit_rates_t = std::tuple<double, double, double, double>;

/**
 * @brief Estimate the L2 hit rates of A and B on gfx1250 from the first launch wave (ids
 * [0, wave), WGMXCC off) on one L2 domain.
 *
 * An approximation, not a per-workgroup count: the wave's distinct rows and columns come from
 * count_unique_range, and each is kept with the chance that one of its ids lands on the
 * domain, which owns xcd_per_l2_domain consecutive XCDs of every NUM_XCD in the round-robin
 * dispatch (dispatch_xcd), at a random phase of the cycle.  The domain's request count is exact.
 * A request hits when another workgroup of the domain already brought its tile, and the domain
 * keeps the share of the footprint that fits its L2.
 *
 * A cluster kernel is dispatched whole and bypasses WGM: the footprint is taken over the
 * cluster grid, each cluster covering several tiles.
 *
 * @param problem Problem description.
 * @param hardware Hardware characteristics (XCD count and L2 domains).
 * @param config Kernel configuration (cluster dimensions).
 * @param context Context computed from the three above (grid, CU budget, per-tile traffic, and
 * context.wgm.wgm, the workgroup mapping: slab width, negative for column-major slabs; 0 is
 * priced as 1).  A cluster kernel overrides it with its column-major cluster slab.
 * @param unique_bytes If set, receives the A + B bytes of the domain's distinct tiles.
 * The operands' TemporalHints set their L2 retention class (see load_temporal_hint in tile.cpp).
 * @return (L2 A, L2 B, MALL A, MALL B) hit rates; there is no MALL, so its rates are 0.
 */
ORIGAMI_EXPORT cache_hit_rates_t estimate_l2_hit_cdna5(const problem_t& problem,
                                                       const hardware_t& hardware,
                                                       const config_t& config,
                                                       const context_t& context,
                                                       double* unique_bytes = nullptr);

/**
 * @brief A linear-estimation method for estimating L2-hitrate.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Predicted L2-hitrate.
 */
ORIGAMI_EXPORT double estimate_l2_hit(const problem_t& problem,
                                      const hardware_t& hardware,
                                      const config_t& config,
                                      const context_t& context);

/**
 * @brief Estimate the MALL-hitrate (last-level cache.)
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return double Predicted MALL-hitrate.
 */
ORIGAMI_EXPORT double estimate_mall_hit(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const config_t& config,
                                        const context_t& context);

/**
 * @brief Estimate per-operand L2 and MALL hit rates using the analytical model.
 *
 * Computes unique tiles counts for the first two timesteps using WGM,
 * then estimates temporal reuse from T0->T1 overlap. Extrapolates to all timesteps.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param context Execution context with derived parameters.
 * @return cache_hit_rates_t
 */
ORIGAMI_EXPORT cache_hit_rates_t estimate_cache_hit_rates(const problem_t& problem,
                                                          const hardware_t& hardware,
                                                          const config_t& config,
                                                          const context_t& context);

/**
 * @brief L2 hit rate from a global (problem-wide) perspective using the refactored API.
 *        Computes in BYTES to correctly handle differing A/B dtypes.
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics (@see origami::hardware_t)
 * @param config Kernel configuration.
 * @param l2_capacity_bytes l2 capacity in bytes
 * @return double
 */
ORIGAMI_EXPORT double compute_l2_hit_rate_global(const problem_t& problem,
                                                 const hardware_t& hardware,
                                                 const config_t& config,
                                                 size_t l2_capacity_bytes);

}  // namespace gemm
}  // namespace origami
