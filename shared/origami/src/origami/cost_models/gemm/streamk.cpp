// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/streamk.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace streamk {
size_t compute_number_of_output_tiles(size_t mt_m, size_t mt_n, size_t m, size_t n, size_t batch) {
  size_t m_tiles = math::safe_ceil_div(m, mt_m);
  size_t n_tiles = math::safe_ceil_div(n, mt_n);
  return m_tiles * n_tiles * batch;
}

/**
 * @brief Returns number of k-iterations.
 *
 * @param output_tiles Number of output tiles.
 * @param iters_per_tile Number of iterations per tile.
 * @return constexpr size_t Number of total iterations.
 */
constexpr size_t num_iters_total(size_t output_tiles, size_t iters_per_tile) {
  return output_tiles * iters_per_tile;
}

/**
 * @brief Returns number of k-iterations per tile.
 *
 * @param mt_k K-dimension tile size.
 * @param k Reduction dimension.
 * @return constexpr size_t Number of k-iteration per tile.
 */
constexpr size_t num_iters_per_tile(size_t mt_k, size_t k) { return math::safe_ceil_div(k, mt_k); }

/**
 * @brief Number of iterations per cta.
 *
 * @param iters_total Total number of k-iterations.
 * @param g Number of workgroups (grid-size).
 * @return constexpr size_t Number of iterations per cta.
 */
constexpr size_t num_iters_per_cta(size_t iters_total, int g) {
  return math::safe_ceil_div(iters_total, g);
}

constexpr size_t num_fixup_peers_v2(size_t g,
                                    size_t iters_total,
                                    size_t iters_per_tile,
                                    size_t iters_per_cta) {
  // If tiles don't evenly divide there are always at least 2 fixup peers, and more if
  // iters_per_tile > iters_per_cta
  size_t hasFixup =
      (iters_total % g == 0 &&               // Check if some WGs have more iters than others
       iters_per_cta % iters_per_tile == 0)  // Check if WGs have an even number of full tiles
          ? 0
          : 1;
  return math::safe_ceil_div(iters_per_tile, iters_per_cta) + hasFixup;
}

/**
 * @brief Number of workgroups involved in the Stream-K's fixup step.
 *
 * @param g Number of total workgroups (grid-size.)
 * @param iters_total Total number of k-iterations.
 * @param iters_per_tile K-iterations per tile.
 * @param iters_per_cta Number of iterations per workgroup.
 * @return constexpr size_t Number of workgroups involved in fixup.
 */

constexpr size_t num_fixup_peers(size_t iters_per_tile, size_t iters_per_cta) {
  return math::safe_ceil_div(iters_per_tile, iters_per_cta);
}

/**
 * @brief Returns the predicted latency for a given grid-size.
 *
 * @param mt BLK_M, BLK_N, BLK_K macro-tile.
 * @param size M, N, K size.
 * @param batch Number of batches.
 * @param g Grid size to test.
 * @param a alpha (a), fixed-size cost incurred by each workgroup.
 * @param b Beta (b) incorporates conditional costs of outputting temporary partial.
 * @param c Represents instruction and stall workload of each MAC-iteration.
 * @param d Delta (d) is the cost of reading and accumulating the partial sums.
 * @return double Predicted latency.
 */
std::tuple<double, size_t, size_t> predicted_runtime(dim3_t mt,
                                                     dim3_t size,
                                                     size_t batch,
                                                     size_t g,
                                                     double a,
                                                     double b,
                                                     double c,
                                                     double d) {
  size_t output_tiles   = compute_number_of_output_tiles(mt.m, mt.n, size.m, size.n, batch);
  size_t iters_per_tile = num_iters_per_tile(mt.k, size.k);
  size_t iters_total    = num_iters_total(output_tiles, iters_per_tile);
  size_t iters_per_cta  = num_iters_per_cta(iters_total, g);
  size_t fixup_peers    = num_fixup_peers(iters_per_tile, iters_per_cta);

  double runtime = a + (b * (fixup_peers > 1)) + (c * iters_per_cta) + (d * (fixup_peers - 1));

  return std::make_tuple(runtime, iters_per_cta, fixup_peers);
}

std::tuple<double, size_t, size_t, double> predicted_runtime_v2(dim3_t mt,
                                                                dim3_t size,
                                                                size_t batch,
                                                                size_t g,
                                                                double a,
                                                                double b,
                                                                double c,
                                                                double d) {
  size_t output_tiles   = compute_number_of_output_tiles(mt.m, mt.n, size.m, size.n, batch);
  size_t iters_per_tile = num_iters_per_tile(mt.k, size.k);
  size_t iters_total    = num_iters_total(output_tiles, iters_per_tile);
  size_t iters_per_cta  = num_iters_per_cta(iters_total, g);
  size_t fixup_peers    = num_fixup_peers_v2(g, iters_total, iters_per_tile, iters_per_cta);

  size_t remainder_tiles = output_tiles % g;
  double k_split_ratio   = remainder_tiles / static_cast<double>(g);

  double cache_penalty = 0.0;
  if (fixup_peers >= 1) {
    // Calculate the ideal equal split ratio
    double ideal_split_ratio = 1.0 / fixup_peers;

    // Measure deviation from the ideal equal split
    double imbalance = 1 / std::abs(k_split_ratio - ideal_split_ratio);

    // Scale the penalty by the imbalance and the per-collaborator cost (d)
    cache_penalty = d * imbalance * fixup_peers;
  }

  // Include the cache penalty in the runtime prediction
  double runtime =
      a + (b * (fixup_peers > 1)) + (c * iters_per_cta) + (d * (fixup_peers - 1)) + cache_penalty;

  return std::make_tuple(runtime, iters_per_cta, fixup_peers, cache_penalty);
}

reduction_t select_reduction(const problem_t& problem,
                             const hardware_t& hardware,
                             const config_t& config,
                             grid_selection_t algorithm) {
  reduction_t reduction_strategy = reduction_t::tree;

  if (algorithm == grid_selection_t::k_split_aware) {
    size_t tiles = compute_number_of_output_tiles(
        config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
    size_t cu_count       = resolve_num_cus(problem.num_cus, hardware.N_CU);
    size_t iters_per_tile = std::max(size_t(1), num_iters_per_tile(config.mt.k, problem.size.k));

    if (tiles < cu_count) {
      // For problems with large k and low number of tiles, use parallel reduction
      // TODO Benchmark to check if limits are correct
      constexpr int MinItersForParallel = 64;
      const int MaxTilesForParallel     = cu_count / 4;
      if (iters_per_tile >= MinItersForParallel && tiles <= MaxTilesForParallel)
        reduction_strategy = reduction_t::parallel;
    }
  }

  return reduction_strategy;
}

/**
 * @brief Dynamically pick the minimum between the cu_count or number of tiles.
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @param cu_count cu count
 * @return size_t minimum between the cu_count or number of tiles
 */
size_t grid_min_resources(const problem_t& problem, const config_t& config, size_t cu_count) {
  size_t tiles = compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  return std::min(cu_count, tiles);
}

/**
 * @brief Dynamically pick the minimum between the cu_count or number of tiles,
 *        and scale down really large sizes to use fewer CUs for power/energy savings
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @param cu_count cu count
 * @return size_t minimum between the cu_count or number of tiles
 */
size_t grid_energy_aware(const problem_t& problem, const config_t& config, size_t cu_count) {
  size_t tiles = compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  size_t sk_grid = cu_count;

  if (tiles > sk_grid) {
    for (size_t i = 1; i <= 32; i *= 2) {
      size_t tiles_per_cu = math::safe_ceil_div(i * tiles, cu_count);
      size_t reduced_grid = math::safe_ceil_div(i * tiles, tiles_per_cu);
      float utilization   = static_cast<float>(reduced_grid) / static_cast<float>(cu_count);
      if (utilization > 0.75f) {
        if (utilization < 1.0f) sk_grid = reduced_grid;
        break;
      }
    }
  }
  return std::min(sk_grid, tiles);
}

/**
 * @brief Dynamically predict the best grid-size by weighing the cost of the fix-up
 *        step and the cost of processing MAC-loop instructions. When the cost of fix-up
 *        is the bottleneck, use smaller grid size.
 *        Architecture dependent.
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @param grid_start grid_start is 1
 * @param grid_end grid_end is cu_count
 * @return size_t minimum between the cu_count or number of tiles
 */
size_t grid_reduction_cost_aware(const problem_t& problem,
                                 const config_t& config,
                                 size_t grid_start,
                                 size_t grid_end) {
  // Fixed overhead alpha (a), fixed-size cost incurred by
  // each work-group, e.g. the grid launch latency, the initial
  // compulsary cache misses, the cost of writing the final output tile
  // to C.
  // double a = 5544 + 9130;
  double a = 2.772 + 4.565;  // 5.04 + 8.30;

  // Beta (b) incorporates conditional costs of outputting temporary partial
  // sums for scenarios where the number of output tiles does not quantize
  // perfectly across the number of processors.
  double b = 3.01;  // 5.47; 6017;

  // c represents instruction and stall workload of each MAC-iteration.
  double c = 2.2935;  // 4.17; 4587;

  // Delta (d) is the cost of reading and accumulating the partial sums from
  // other work-groups covering the same tile.
  double d = 10.22;  // 18.59; 20449;

  std::pair<size_t, double> min_grid_runtime;
  std::pair<size_t, double> min_grid_runtime_v2;
  min_grid_runtime.second    = std::numeric_limits<double>::max();
  min_grid_runtime_v2.second = std::numeric_limits<double>::max();

  size_t g = grid_start;

  // Predict the number of CTAs to use between 1 and 304
  for (; g <= static_cast<size_t>(grid_end); ++g) {
    auto [runtime, iters_per_cta, fixup_peers] =
        predicted_runtime(config.mt, problem.size, problem.batch, g, a, b, c, d);

    auto [runtime_v2, iters_per_cta_v2, fixup_peers_v2, cache_penalty] =
        predicted_runtime_v2(config.mt, problem.size, problem.batch, g, a, b, c, d);

    if (min_grid_runtime.second > runtime) {
      min_grid_runtime.first  = g;
      min_grid_runtime.second = runtime;
    }

    if (min_grid_runtime_v2.second > runtime_v2) {
      min_grid_runtime_v2.first  = g;
      min_grid_runtime_v2.second = runtime_v2;
    }
  }

  return min_grid_runtime_v2.first;
}

/**
 * @brief Fix Stream-K algorithm to function like a Data-parallel schedule
 *        where grid size is equal to the number of output tiles.
 * @param problem Problem description (M, N, K, etc.)
 * @param config Kernel configuration.
 * @return size_t number of tiles
 */
size_t grid_data_parallel(const problem_t& problem, const config_t& config) {
  return compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
}

size_t grid_analytical(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       size_t biggest_allowable_split) {
  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t batch = problem.batch;

  size_t MT_M = config.mt.m;
  size_t MT_N = config.mt.n;

  // compute how many 32×32 tiles are needed in each dim,
  // then multiply to get total grid size:
  size_t grid = ((M + MT_M - 1) / MT_M) * ((N + MT_N - 1) / MT_N) * batch;

  // Honor the caller's CU budget (problem.num_cus); 0 means use all CUs.
  const size_t cu_count = resolve_num_cus(problem.num_cus, hardware.N_CU);
  size_t max_hw_split   = std::floor(cu_count / grid);
  size_t MAX_SPLIT      = std::min(biggest_allowable_split, max_hw_split);

  size_t best_split   = 1;
  double best_latency = std::numeric_limits<double>::infinity();

  for (size_t split = 1; split <= MAX_SPLIT; ++split) {
    double latency = gemm::compute_total_latency(problem, hardware, config);

    if (latency < best_latency) {
      best_latency = latency;
      best_split   = split;
    }
    best_latency = latency;
    best_split   = split;
  }

  size_t best_grid = best_split * grid;
  return best_grid;
}

static size_t correct_sk_grid_for_partial_tiles(size_t sk_grid,
                                                size_t tiles,
                                                size_t iters_per_tile,
                                                size_t cu_count,
                                                size_t batch) {
  if (sk_grid == 0 || sk_grid >= tiles) return sk_grid;

  // Batched GEMMs benefit from SK tile-streaming (consecutive batch tiles share
  // A/B data in L1/L2); skip all corrections for batch > 1.
  if (batch > 1) return sk_grid;

  // One K iteration per tile, so there is nothing to split.  sk_grid is always
  // below tiles here, so each workgroup covers more than one tile.  Past two
  // workgroups per CU it is better to use one workgroup per tile instead.
  // Threshold is measured, not derived.
  if (iters_per_tile <= 1) {
    if (cu_count > 0 && sk_grid > 2 * cu_count) return tiles;
    return sk_grid;
  }

  // Keep SK when DP would leave its last wave largely idle (DP_eff < 80%).
  if (cu_count > 0) {
    const size_t dp_waves    = (tiles + cu_count - 1) / cu_count;
    const size_t dp_cu_steps = dp_waves * cu_count;
    if (tiles < 0.8 * dp_cu_steps) return sk_grid;
  }

  // DP fills its waves well: force DP if any CTA would straddle a tile boundary,
  // since those CTAs write partial workspace and need an extra fixup pass.
  const size_t iters_total = tiles * iters_per_tile;
  const size_t floor_iters = iters_total / sk_grid;
  return (floor_iters % iters_per_tile != 0) ? tiles : sk_grid;
}

/**
 * @brief Stream-K grid of W4A16 kernels on gfx1151 and gfx1150 (is_w4a16). Measured on
 *        prefill GEMMs of LLM projection layers (16 to 4096 tokens), 4 W4A16 types, every
 *        kernel at every grid below.
 *
 * The W4A16 kernels are all Stream-K: a grid of one workgroup per tile (grid = tiles) runs them
 * data parallel (DP). With t = tiles / CUs:
 *
 * t <= 1:
 *   1. Workgroups that fill a CU (16 waves): DP.
 *   2. Short K (<= 8 K iterations, or <= 64 when t > 0.5): DP. Splitting K does not pay for
 *      the fixup.
 *   3. Otherwise one workgroup per CU, splitting K, if the partial tiles fit in the workspace.
 * t > 1, with S = 5 tiles per CU where a MALL holds the partial tiles and 0 without one:
 *   Above S and short K (<= 16 K iterations): DP.
 *   4. DP when the last wave wastes little: ceil(t) / t below 1.5 (workgroups that fill a CU),
 *      1.15 (4-wave, non-token tiles) or 1.06 (the rest). Not for a single row of token tiles
 *      with t <= S: sharing streams the weights faster.
 *   5. Token tiles (MT_N <= 48): one workgroup per SIMD, sharing the tiles.
 *   6. Other tiles with t < S: one workgroup per CU, sharing the tiles.
 *   In 5 and 6, tiles that divide evenly stay DP, and so do partial tiles that do not fit in
 *   the workspace.
 * Anything else: DP.
 */
static size_t grid_k_split_aware_w4a16(const problem_t& problem,
                                       const hardware_t& hardware,
                                       const config_t& config,
                                       size_t cu_count) {
  using D            = w4a16_params_t;
  const size_t tiles = compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  const size_t tile_size = config.mt.m * config.mt.n * config.workspace_size_per_elem_c;
  const size_t k_iters   = math::safe_ceil_div(problem.size.k, config.mt.k);
  const size_t shared    = hardware.has_MALL() ? D::SHARED_MAX_TILES_PER_CU : 0;
  int waves              = D::DEFAULT_WAVES_PER_WORKGROUP;
  if (config.has_tensile_params()) {
    const auto& tp = config.tensile();
    waves =
        std::max(tp.wave_group_m, 1) * std::max(tp.wave_group_n, 1) * std::max(tp.local_split_u, 1);
  }
  if (tiles <= cu_count) {
    // Rule 1.
    if (waves >= D::FULL_CU_WAVES) return tiles;
    // Rule 2.
    if (k_iters <= D::SK_SPLIT_MIN_K_ITERS ||
        (k_iters <= D::SK_SPLIT_MIN_K_ITERS_OVER_HALF && 2 * tiles > cu_count))
      return tiles;
    // Rule 3.
    if (tiles == cu_count || tile_size * cu_count <= config.workspace_size) return cu_count;
  } else if (tiles > shared * cu_count && k_iters <= D::MANY_TILES_DP_MAX_K_ITERS) {
    return tiles;  // many tiles, short K: DP
  } else {
    // Rule 4.
    const bool token_tile = config.mt.n <= D::TOKEN_TILE_MAX_N;
    const bool token_row =
        token_tile && problem.size.n <= config.mt.n && tiles <= shared * cu_count;
    if (!token_row) {
      double max_rounding = D::DP_MAX_WAVE_ROUNDING;
      if (waves >= D::FULL_CU_WAVES)
        max_rounding = D::DP_MAX_WAVE_ROUNDING_FULL_CU;
      else if (waves <= 4 && !token_tile)
        max_rounding = D::DP_MAX_WAVE_ROUNDING_4_WAVES;
      if (static_cast<double>(math::safe_ceil_div(tiles, cu_count) * cu_count) <
          max_rounding * static_cast<double>(tiles))
        return tiles;
    }
    // Rules 5 and 6.
    if (token_tile || tiles < shared * cu_count) {
      const size_t per_cu = token_tile ? hardware.simds_per_cu() : 1;
      size_t grid         = (tiles % (per_cu * cu_count) != 0) ? per_cu * cu_count : tiles;
      if (tiles % grid != 0 && tile_size * grid > config.workspace_size) grid = tiles;
      return grid;
    }
  }
  return tiles;
}

// @param cu_budget Internal genuine-cap budget: the resolved CU cap when the
//        caller supplied a genuine cap (problem.num_cus is positive and below
//        the physical N_CU), else 0 ("no genuine cap / use all CUs"). Computed
//        by select_grid_size so this file-local helper keeps its original
//        max_cus-driven grid-cap and occupancy-multiplier semantics.
size_t grid_k_split_aware(const problem_t& problem,
                          const hardware_t& hardware,
                          const config_t& config,
                          size_t cu_count,
                          size_t cu_budget) {
  if (is_w4a16(problem, hardware))
    return grid_k_split_aware_w4a16(problem, hardware, config, cu_count);

  size_t tiles = compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  size_t sk_grid = tiles;  // Fallback if no good fractional tile is found
  if (cu_budget > 0) sk_grid = std::min(sk_grid, cu_budget);

  const size_t iters_per_tile = num_iters_per_tile(config.mt.k, problem.size.k);

  const size_t tile_size = config.mt.m * config.mt.n * config.workspace_size_per_elem_c;

  // More tiles than CUs
  // Distribute tiles evenly across maximum number of CUs
  // Split remaining tiles as evenly as possible for better caching
  if (tiles > cu_count) {
    size_t virt_cu_count = cu_count;
    if (config.occupancy > 1) virt_cu_count *= config.occupancy;

    const std::vector<double> tile_fractions = {
        0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0};
    // When virt_cu_count is greater than tiles, min_even_tiles is 0 which enforces partial tiles.
    const size_t min_even_tiles = std::max<size_t>(1, tiles / virt_cu_count);

    for (double frac : tile_fractions) {
      const size_t frac_grid = static_cast<size_t>((tiles / (min_even_tiles + frac)) + 0.5);

      // Check if higher occupancy would cause excessive workspace requirements (set current limit
      // to 128MB)
      if ((tiles % frac_grid != 0) && (tile_size * frac_grid > 128ull * 1024ull * 1024ull))
        continue;

      if (frac_grid <= virt_cu_count) {
        sk_grid = frac_grid;
        break;
      }
    }
  }
  // Fewer tiles than CUs
  // Split tiles evenly in k-dimension
  // Attempt to maximize CU utilization, up to a peak number of splits
  // Max splitting is currently constant, but should be dependant on K dimension
  else if (tiles < cu_count) {
    // For problems with large k and low number of tiles, use parallel reduction
    // TODO Benchmark to check if limits are correct
    // constexpr int MinItersForParallel = 64;
    // constexpr int MaxTilesForParallel = 16;
    constexpr int MinItersPerCU = 8;

    if (config.reduction_strategy == reduction_t::parallel) {
      size_t virt_cu_count = cu_count;
      // TODO check if using occupancy info makes workspace too large
      // if (occupancy > 1)
      //     virt_cu_count *= occupancy;

      // Find max splitting factor to use as much of GPU as possible
      const size_t maxSplitsForTiles = virt_cu_count / tiles;

      // Find max splitting factor to ensure each CU has a minimum number of iterations to do
      const size_t maxSplitsForIters = iters_per_tile / MinItersPerCU;

      const size_t maxSplits = std::min(maxSplitsForTiles, maxSplitsForIters);
      sk_grid                = tiles * maxSplits;
    } else {
      const std::vector<size_t> tile_fractions = {16, 12, 8, 6, 4, 3, 2, 1};
      for (size_t frac : tile_fractions) {
        const size_t splitGrid  = tiles * frac;
        const size_t itersPerCU = iters_per_tile / frac;
        if (splitGrid <= cu_count && itersPerCU >= MinItersPerCU) {
          sk_grid = splitGrid;
          break;
        }
      }
    }
  }

  if (tiles % sk_grid != 0 && tile_size * sk_grid > config.workspace_size) sk_grid = tiles;

  // Partial-tile and DP-efficiency correction: only applies to the streaming
  // case (tiles > cu_count).  K-split grids (tiles <= cu_count) intentionally
  // produce non-tile-aligned grids and must not be corrected to DP.
  if (tiles > cu_count)
    sk_grid =
        correct_sk_grid_for_partial_tiles(sk_grid, tiles, iters_per_tile, cu_count, problem.batch);
  return sk_grid;
}
size_t select_grid_size(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config,
                        grid_selection_t algorithm) {
  // Single source of truth for the CU budget: problem.num_cus (0 or >= physical
  // N_CU means "use all CUs").
  size_t cu_count = resolve_num_cus(problem.num_cus, hardware.N_CU);

  // Internal genuine-cap budget for grid_k_split_aware: the resolved cap only
  // when the caller supplied a genuine cap (num_cus positive and below physical
  // N_CU), otherwise 0. Preserves the legacy semantic that the grid cap applies
  // iff there is a genuine cap and the occupancy multiplier applies iff there is
  // not (num_cus == 0 or num_cus >= physical N_CU).
  const size_t cu_budget = (cu_count < hardware.N_CU) ? cu_count : 0;

  switch (algorithm) {
    case grid_selection_t::min_resources:
      return streamk::grid_min_resources(problem, config, cu_count);

    case grid_selection_t::energy_aware:
      return streamk::grid_energy_aware(problem, config, cu_count);

    case grid_selection_t::reduction_cost_aware:
      return streamk::grid_reduction_cost_aware(problem, config, 1, cu_count);

    case grid_selection_t::data_parallel: return streamk::grid_data_parallel(problem, config);

    case grid_selection_t::analytical:
      return streamk::grid_analytical(problem, hardware, config, 10);

    case grid_selection_t::k_split_aware:
      return streamk::grid_k_split_aware(problem, hardware, config, cu_count, cu_budget);

    case grid_selection_t::number_of_cus:
    default: return cu_count;
  }
}

hybrid_mode_t select_hybrid_mode(const problem_t& problem,
                                 const hardware_t& hardware,
                                 const config_t& config,
                                 size_t sm_count_target) {
  return gemm::for_family(hardware, [&](auto family) {
    return family.tile_schedule(problem, hardware, config, sm_count_target);
  });
}
}  // namespace streamk
}  // namespace origami

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

size_t base_model::cluster_size(const config_t&) { return 1; }

// Clusters running at once: each is placed whole inside one shader engine (two per XCD).
size_t clusters_per_wave(const hardware_t& hardware, size_t usable_cus, size_t cluster) {
  const size_t num_se     = std::max<size_t>(hardware.NUM_XCD * 2, 1);
  const size_t wgp_per_se = std::max<size_t>(usable_cus / num_se, 1);
  return std::max<size_t>(num_se * (wgp_per_se / std::max<size_t>(cluster, 1)), 1);
}

// Cs x Ck blocks of output tiles covering the grid; boundary blocks are padded.
size_t cluster_blocks(const problem_t& problem, const config_t& config) {
  return math::safe_ceil_div(math::safe_ceil_div(problem.size.m, config.mt.m),
                             std::max<size_t>(config.cluster_dim.m, 1)) *
         math::safe_ceil_div(math::safe_ceil_div(problem.size.n, config.mt.n),
                             std::max<size_t>(config.cluster_dim.n, 1)) *
         std::max<size_t>(problem.batch, 1);
}

struct spatial_streamk_launch_t {
  size_t num_wgs;
  size_t tiles;
  reduction_t reduction;
};

// Static StreamK with a cluster runs TensileLite's spatial schedule
// (resolveClusteredStreamKSettings): each cluster works a K range of one block, and boundary
// blocks keep all their peers, which load and synchronize but own no output.  The grid chosen
// for the real tiles is re-cut into whole clusters over the padded blocks.  A whole-tile grid
// becomes whole blocks, and a parallel split keeps its split count per block while the
// resident WGs allow it.  Otherwise the tree schedule follows the uniform summation order
// hipBLASLt requests by default: every block takes the same number of K chunks, at least 8
// iterations each, or the blocks run whole, at most one cluster each.
static spatial_streamk_launch_t spatial_streamk_launch(const problem_t& problem,
                                                       const hardware_t& hardware,
                                                       const config_t& config,
                                                       size_t c,
                                                       size_t num_mts,
                                                       size_t grid,
                                                       reduction_t reduction) {
  const size_t blocks = cluster_blocks(problem, config);
  const size_t padded = blocks * c;
  if (grid == num_mts) return {padded, padded, reduction_t::tree};

  const size_t iters = std::max<size_t>(math::safe_ceil_div(problem.size.k, config.mt.k), 1);
  size_t budget      = grid;
  if (grid > num_mts)
    budget = (reduction == reduction_t::parallel) ? padded * std::min(grid / num_mts, iters)
                                                  : std::max(grid, padded);
  const size_t resident = hardware.N_CU * std::max<size_t>(config.occupancy, 1);
  const size_t clusters =
      std::max<size_t>(std::min({std::max(budget, c) / c, resident / c, blocks * iters}), 1);

  size_t splits = std::min(clusters / blocks, iters);
  if (reduction == reduction_t::parallel && splits >= 2)
    return {padded * splits, padded, reduction};
  if (clusters % blocks != 0)
    while (splits >= 2 && (iters % splits != 0 || iters / splits < 8)) --splits;
  if (splits >= 2) return {padded * splits, padded, reduction_t::tree};
  return {std::min(clusters, blocks) * c, padded, reduction_t::tree};
}

// Compute the launch parameters for the kernel.
std::tuple<reduction_t, size_t, size_t, size_t, size_t> compute_launch_parameters(
    const problem_t& problem,
    const hardware_t& hardware,
    const config_t& config,
    grid_selection_t grid_selection) {
  const size_t num_mts = streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);
  size_t num_wgs                 = num_mts;
  reduction_t reduction_strategy = reduction_t::none;

  if (config.stream_k > 0) {
    reduction_strategy = streamk::select_reduction(problem, hardware, config, grid_selection);
    auto config_with_reduction               = config;
    config_with_reduction.reduction_strategy = reduction_strategy;
    num_wgs = streamk::select_grid_size(problem, hardware, config_with_reduction, grid_selection);
  }

  // There are cases in which StreamK combines multiple output MTs and assigns to 1 WG.
  // That means, we artifically observe one full timesteps, but that is not what actually happens
  // under the hood. From a theoretical point of view, these distributions change all of the
  // computations in Origami. With current implementation, it is hard to capture that
  // behaviour analytically. So for now, if the num_wgs is less than the num_mts, we calculate
  // num_timesteps based on the num_mts. Otherwise, we use num_wgs to compute num_timesteps.
  // Usable CU count: derived from the problem's CU budget (problem.num_cus).
  const size_t usable_cus = resolve_num_cus(problem.num_cus, hardware.N_CU);
  size_t num_timesteps    = num_wgs > num_mts ? math::safe_ceil_div(num_wgs, usable_cus)
                                              : math::safe_ceil_div(num_mts, usable_cus);
  size_t scheduled_tiles  = num_mts;

  // Cluster kernels run in waves of whole clusters.  Static StreamK re-cuts its grid into the
  // spatial schedule, and a non-persistent kernel pads its grid to whole clusters; padded
  // workgroups exit at once but are charged their slot.
  const size_t c = for_family(hardware, [&](auto family) { return family.cluster_size(config); });
  if (c > 1 && config.stream_k == 3) {
    const auto spatial =
        spatial_streamk_launch(problem, hardware, config, c, num_mts, num_wgs, reduction_strategy);
    reduction_strategy = spatial.reduction;
    num_wgs            = spatial.num_wgs;
    scheduled_tiles    = spatial.tiles;
    const size_t wave  = std::min(num_wgs, clusters_per_wave(hardware, usable_cus, c) * c);
    num_timesteps      = math::safe_ceil_div(std::max(num_wgs, scheduled_tiles), wave);
  } else if (c > 1 && config.stream_k == 0) {
    num_timesteps = math::safe_ceil_div(cluster_blocks(problem, config),
                                        clusters_per_wave(hardware, usable_cus, c));
  }
  const size_t num_active_cus   = num_wgs < usable_cus ? num_wgs : usable_cus;
  const size_t splitting_factor = math::safe_ceil_div(num_wgs, scheduled_tiles);

  return std::make_tuple(
      reduction_strategy, num_wgs, num_active_cus, num_timesteps, splitting_factor);
}

}  // namespace gemm
}  // namespace origami
