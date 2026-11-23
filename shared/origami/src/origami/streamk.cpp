// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/streamk.hpp"
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include <algorithm>
#include <limits>

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
  size_t max_hw_split = std::floor(cu_count / grid);
  size_t MAX_SPLIT    = std::min(biggest_allowable_split, max_hw_split);

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

  // you now have both `grid` and `best_split`—
  // return whichever is appropriate (here we stick with split):
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

// @param cu_budget Internal genuine-cap budget: the resolved CU cap when the
//        caller supplied a genuine cap (problem.num_cus is positive and below
//        the physical N_CU), else 0 ("no genuine cap / use all CUs"). Computed
//        by select_grid_size so this file-local helper keeps its original
//        max_cus-driven grid-cap and occupancy-multiplier semantics.
size_t grid_k_split_aware(const problem_t& problem,
                          const config_t& config,
                          size_t cu_count,
                          size_t cu_budget) {
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

    if (config.reduction_strategy == reduction_t::parallel) {
      size_t virt_cu_count = cu_count;
      // TODO check if using occupancy info makes workspace too large
      // if (occupancy > 1)
      //     virt_cu_count *= occupancy;

      // Find max splitting factor to use as much of GPU as possible
      const size_t maxSplitsForTiles = virt_cu_count / tiles;

      // Find max splitting factor to ensure each CU has a minimum number of iterations to do
      const size_t maxSplitsForIters = iters_per_tile / MIN_ITERS_PER_CU;

      const size_t maxSplits = std::min(maxSplitsForTiles, maxSplitsForIters);
      sk_grid                = tiles * maxSplits;
    } else {
      const std::vector<size_t> tile_fractions = {16, 12, 8, 6, 4, 3, 2, 1};
      for (size_t frac : tile_fractions) {
        const size_t splitGrid  = tiles * frac;
        const size_t itersPerCU = iters_per_tile / frac;
        if (splitGrid <= cu_count && itersPerCU >= MIN_ITERS_PER_CU) {
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
    sk_grid = correct_sk_grid_for_partial_tiles(sk_grid, tiles, iters_per_tile, cu_count,
                                                problem.batch);
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
      return streamk::grid_k_split_aware(problem, config, cu_count, cu_budget);

    case grid_selection_t::number_of_cus:
    default: return cu_count;
  }
}

hybrid_mode_t select_hybrid_mode(const problem_t& problem,
                                 const hardware_t& hardware,
                                 const config_t& config,
                                 size_t sm_count_target) {
  // Fit on gfx950 (MI350X) sweeps only; other architectures stay static
  // until tuned in a follow-up PR.
  if (hardware.arch != hardware_t::architecture_t::gfx950)
    return hybrid_mode_t::static_;

  const size_t MT_M  = config.mt.m;
  const size_t MT_N  = config.mt.n;
  const size_t batch = std::max<size_t>(problem.batch, 1);
  const size_t tiles = compute_number_of_output_tiles(
      MT_M, MT_N, problem.size.m, problem.size.n, batch);

  // Too little work in the grid for dynamic rebalancing to be worth its
  // overhead.
  if (tiles <= streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC) return hybrid_mode_t::static_;

  size_t available_cus = (sm_count_target > 0)
                             ? std::min<size_t>(sm_count_target, hardware.N_CU)
                             : hardware.N_CU;
  if (available_cus == 0) available_cus = hardware.N_CU;

  // No cotenant means static already matches the full CU count -- nothing
  // left for dynamic to rebalance.
  const bool has_cotenant = available_cus < hardware.N_CU;
  if (!has_cotenant) return hybrid_mode_t::static_;

  // Few wavefronts resident per CU means little overlap to absorb a
  // cotenant's CU-count mismatch on its own, so dynamic rebalancing helps
  // regardless of tiles_per_cu. (occupancy <= 0 means "unknown" -- fall
  // through to the tiles_per_cu check instead.)
  if (config.occupancy > 0 &&
      config.occupancy <= streamk_hybrid_defaults_t::MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC)
    return hybrid_mode_t::dynamic;

  // At higher occupancy that overlap already absorbs small mismatches, so
  // dynamic only pays off once the grid is heavily overloaded relative to
  // the CUs actually available to it.
  const double tiles_per_cu =
      static_cast<double>(tiles) / static_cast<double>(available_cus);
  return (tiles_per_cu > streamk_hybrid_defaults_t::TILES_PER_CU_THRESHOLD_HIGH_OCCUPANCY)
             ? hybrid_mode_t::dynamic
             : hybrid_mode_t::static_;
}

size_t padded_iters_per_tile(size_t iters_per_tile, size_t k_granularity) {
  if (k_granularity <= 1 || iters_per_tile == 0) return iters_per_tile;
  return ((iters_per_tile + k_granularity - 1) / k_granularity) * k_granularity;
}

static_split_t static_split(size_t tiles,
                            size_t iters_per_tile,
                            size_t grid,
                            int full_tiles,
                            bool force_dp_only,
                            size_t k_granularity) {
  static_split_t split;
  if (grid == 0) return split;

  // Each workgroup runs an even share of Stream-K iterations, then whole data-parallel tiles.
  // A grid that divides the tiles needs no Stream-K tiles; force-DP-only keeps every tile whole.
  const bool big_enough  = tiles > grid;
  std::uint32_t sk_tiles = force_dp_only ? 0u : static_cast<std::uint32_t>(grid);
  if (!force_dp_only && tiles % grid != 0) {
    sk_tiles = big_enough ? static_cast<std::uint32_t>(grid * full_tiles + tiles % grid)
                          : static_cast<std::uint32_t>(tiles);
    sk_tiles = std::min(sk_tiles, static_cast<std::uint32_t>(tiles));
  }
  split.sk_tiles = sk_tiles;

  // The stride and the leftover are whole K units, over padded tiles.
  const size_t g            = std::max(k_granularity, size_t{1});
  const size_t padded_iters = padded_iters_per_tile(iters_per_tile, g);
  const size_t sk_iters     = static_cast<size_t>(sk_tiles) * padded_iters;
  split.sk_iters_per_wg     = static_cast<std::uint32_t>(g * (sk_iters / g / grid));
  split.extra_iters =
      static_cast<std::uint32_t>(sk_iters - static_cast<size_t>(split.sk_iters_per_wg) * grid);
  return split;
}

bool static_split_row_uniform(const static_split_t& split,
                              size_t tiles,
                              size_t iters_per_tile,
                              size_t grid,
                              bool per_tile_capable,
                              bool uniform_summation_order,
                              size_t k_granularity) {
  // Chunks are cut from the padded flat space.  The kernel takes the per-tile mapping only when
  // it can and the launch asks for uniform summation order.
  const size_t padded_iters  = padded_iters_per_tile(iters_per_tile, k_granularity);
  const bool per_tile_active = per_tile_capable && uniform_summation_order;
  // A tile's fold signature is the ordered list of chunk lengths summed into it; every tile of
  // the launch must share one.  That holds when:
  //  sk_tiles == 0       force-DP-only, every tile whole;
  //  sk_tiles == tiles, extra_iters == 0 and sk_iters_per_wg | I
  //                      no data-parallel region: grid == tiles, or equal all-partial chunks.
  //                      A mixed split (sk_tiles == grid < tiles, grid | tiles) is data parallel
  //                      then Stream-K; gfx950 skips the tree-partials workspace when
  //                      tiles % grid == 0 and never stores the Stream-K half of D, so it is
  //                      refused until the device path writes every tile;
  //  per-tile extras     sk_tiles == tiles and tiles | grid, with a kernel that hands out the
  //                      extras within each tile: every tile gets the same remainder pattern.
  // Without per-tile extras, extra_iters != 0 means two chunk lengths under the global first-E
  // mapping.  sk_iters_per_wg != 0 is checked because tiles == 0 is reachable.
  if (split.sk_tiles == 0) return true;
  if (split.sk_iters_per_wg != 0 && split.extra_iters == 0 && split.sk_tiles == tiles &&
      padded_iters % split.sk_iters_per_wg == 0)
    return true;
  if (per_tile_active && split.sk_tiles == tiles && tiles != 0 && grid % tiles == 0 &&
      split.sk_iters_per_wg != 0)
    return true;
  return false;
}

iter_range_t workgroup_iter_range(size_t w,
                                  size_t tiles,
                                  size_t iters_per_tile,
                                  size_t grid,
                                  bool per_tile_capable,
                                  bool uniform_summation_order,
                                  size_t k_granularity) {
  iter_range_t range;
  if (grid == 0 || tiles == 0 || k_granularity == 0) return range;

  const size_t g            = k_granularity;
  const size_t padded_iters = padded_iters_per_tile(iters_per_tile, g);
  const size_t total_iters  = tiles * padded_iters;
  // The device adds g, not 1, to the long chunks, and E workgroups get one.
  const size_t W = g * (total_iters / g / grid);
  const size_t E = (total_iters - W * grid) / g;

  // The device takes the per-tile branch only when the kernel supports it and the packed
  // uniform-summation-order bit is set; mirror both, or the fixup reads the wrong partials.
  const bool per_tile_active = per_tile_capable && uniform_summation_order;
  if (per_tile_active && grid % tiles == 0) {
    const size_t F     = grid / tiles;
    const size_t q     = w / F;
    const size_t s     = w % F;
    const size_t base  = g * (padded_iters / g / F);
    const size_t rem_i = (padded_iters - base * F) / g;
    range.start        = q * padded_iters + s * base + g * std::min(s, rem_i);
    range.end          = range.start + base + (s < rem_i ? g : size_t{0});
    return range;
  }

  // Global first-E mapping.
  if (w < E) {
    range.start = w * (W + g);
    range.end   = range.start + (W + g);
  } else {
    range.start = E * (W + g) + (w - E) * W;
    range.end   = range.start + W;
  }
  return range;
}

bool parallel_reduction_row_uniform(reduction_t reduction,
                                    size_t grid,
                                    size_t tiles,
                                    int stream_k_atomic,
                                    bool static_two_tile_packing) {
  // Workgroup i reduces into tile i / F at partial i % F, F = grid / tiles: every tile sees the
  // same partial set and K ranges when F >= 2 and grid is an exact multiple of tiles.
  if (reduction != reduction_t::parallel) return false;
  if (stream_k_atomic != 0) return false;
  if (!static_two_tile_packing) return false;
  if (grid == 0 || tiles == 0) return false;
  if (grid % tiles != 0) return false;
  if ((grid / tiles) < 2) return false;
  return true;
}

dynamic_items_t dynamic_items(size_t tiles,
                              size_t iters_per_tile,
                              size_t k_granularity,
                              int override_tiles,
                              int override_split) {
  dynamic_items_t items;
  items.sk_tiles = override_tiles > -1 ? static_cast<std::uint32_t>(override_tiles) : 0;
  items.sk_split = override_split > -1 ? static_cast<std::uint32_t>(override_split) : 2;
  // Whole K units per work item, over the padded tile; the device clamps the last one.
  const std::uint32_t k_gran = static_cast<std::uint32_t>(std::max(k_granularity, size_t{1}));
  const std::uint32_t iters  = static_cast<std::uint32_t>(
      padded_iters_per_tile(std::max(size_t{1}, iters_per_tile), k_gran));
  // (n + d - 1) / d in 32 bits, as the kernel arguments are packed.
  auto ceil_div = [](std::uint32_t n, std::uint32_t d) { return (n + (d - 1)) / d; };
  items.sk_iters_per_wi = k_gran * ceil_div(ceil_div(iters, items.sk_split), k_gran);
  items.sk_split        = ceil_div(iters, items.sk_iters_per_wi);
  items.total_items = (tiles - items.sk_tiles) + items.sk_tiles * items.sk_split;
  return items;
}

reduction_t reconcile_reduction(reduction_t reduction, size_t grid, size_t tiles) {
  if (reduction == reduction_t::parallel && (tiles == 0 || (grid / tiles) < 2))
    return reduction_t::tree;
  return reduction;
}

bool straddles_tiles(size_t iters_per_tile, size_t grid, size_t tiles) {
  return grid > 0 && iters_per_tile > 1 && (tiles % grid) != 0;
}

size_t partial_tile_bytes(size_t mt_m, size_t mt_n, size_t workspace_size_per_elem_c, size_t grid) {
  return mt_m * mt_n * workspace_size_per_elem_c * grid;
}

size_t uniform_grid(size_t grid,
                    size_t tiles,
                    size_t iters_per_tile,
                    reduction_t reduction,
                    size_t cluster_width,
                    bool per_tile_extra_iters,
                    size_t tile_bytes,
                    size_t workspace_size,
                    size_t flag_entries) {
  if (tiles == 0 || grid == 0) return grid;
  if (grid < tiles) return tiles;
  if (grid == tiles) return grid;

  const size_t I = iters_per_tile;
  size_t f_star  = 1;  // all-full, always admissible
  for (size_t F = grid / tiles; F >= 2; --F) {
    if ((tiles * F) % cluster_width != 0) continue;
    // Tree all-partial without per-tile extras needs F | I; parallel extras are per partial.
    if (reduction != reduction_t::parallel && !per_tile_extra_iters && (I % F) != 0) continue;
    if ((I / F) < MIN_ITERS_PER_CU) continue;
    if (tile_bytes * (tiles * F) > workspace_size) continue;
    if (flag_entries != 0 && (tiles * F) > flag_entries) continue;
    f_star = F;
    break;
  }
  return tiles * f_star;
}

partials_t launch_partials(const launch_t& d) {
  if (!d.is_streamk || d.is_dynamic) return {};
  if (d.reduction == reduction_t::parallel && d.sk_split >= 2)
    return {d.tiles * d.sk_split, d.tiles};
  const std::uint64_t sk = d.dp_only ? 0 : d.sk_tiles, iters = d.iters_per_tile, G = d.final_grid;
  if (G == 0 || sk == 0 || iters == 0) return {};
  const std::uint64_t T = sk * iters, W = T / G, E = T % G;
  // Every workgroup owns whole tiles: no start falls inside one.
  if (E == 0 && W % iters == 0) return {};
  // Walk s_g = g * W + min(g, E) as (tile, offset) = (s_g / iters, s_g % iters) without dividing:
  // s_{g+1} - s_g = W + (g < E).
  partials_t out;
  const std::uint64_t Wq = W / iters, Wr = W % iters;
  std::uint64_t tile = 0, off = 0, last = std::numeric_limits<std::uint64_t>::max();
  for (std::uint64_t g = 1; g < G; ++g) {
    tile += Wq;
    off += Wr + (g - 1 < E ? 1 : 0);
    if (off >= iters) {
      off -= iters;
      ++tile;
    }
    if (off == 0) continue;
    ++out.partials;
    if (tile != last) {
      ++out.split_tiles;
      last = tile;
    }
  }
  return out;
}

launch_terms_t launch_terms(const problem_t& problem, const launch_t& sk) {
  launch_terms_t t;
  if (!sk.is_streamk) return t;
  const double m     = static_cast<double>(problem.size.m);
  const double n     = static_cast<double>(problem.size.n);
  const double batch = static_cast<double>(std::max<size_t>(problem.batch, 1));
  const double bpe_d = datatype_to_bits(problem.d_dtype) / 8.0;

  const partials_t pt  = launch_partials(sk);
  const double grid    = static_cast<double>(sk.final_grid);
  const double tiles   = static_cast<double>(sk.tiles);
  const std::uint64_t cus = (problem.num_cus > 0 && problem.num_cus < sk.cu_count)
                                ? static_cast<std::uint64_t>(problem.num_cus)
                                : sk.cu_count;
  const std::uint64_t slots = cus * static_cast<std::uint64_t>(std::max(sk.cu_occupancy, 1));
  const double waves        = slots ? static_cast<double>((sk.final_grid + slots - 1) / slots)
                                    : std::numeric_limits<double>::quiet_NaN();
  const double mt = static_cast<double>(sk.mt_x) * static_cast<double>(sk.mt_y) *
                    static_cast<double>(sk.workspace_size_per_elem_c);
  t.grid_over_tiles         = grid / tiles;
  t.split_tiles_frac        = static_cast<double>(pt.split_tiles) / tiles;
  t.fixup_bytes_over_output = ((2.0 * static_cast<double>(pt.partials)) * mt) / (m * n * batch * bpe_d);
  t.num_wgs                 = grid;
  t.timesteps               = waves * tiles / grid;
  return t;
}
}  // namespace streamk
}  // namespace origami
