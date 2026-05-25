// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "origami/streamk.hpp"
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

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
    size_t cu_count       = hardware.N_CU;
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
                       size_t biggest_allowable_split,
                       size_t max_cus) {
  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t K     = problem.size.k;
  size_t batch = problem.batch;

  size_t MT_M = config.mt.m;
  size_t MT_N = config.mt.n;

  // compute how many 32×32 tiles are needed in each dim,
  // then multiply to get total grid size:
  size_t grid = ((M + MT_M - 1) / MT_M) * ((N + MT_N - 1) / MT_N) * batch;

  size_t max_hw_split = std::floor(hardware.N_CU / grid);
  size_t MAX_SPLIT    = std::min(biggest_allowable_split, max_hw_split);

  size_t best_split   = 1;
  double best_latency = std::numeric_limits<double>::infinity();

  for (size_t split = 1; split <= MAX_SPLIT; ++split) {
    double latency = compute_total_latency(problem, hardware, config, max_cus);

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

size_t grid_k_split_aware(const problem_t& problem,
                          const config_t& config,
                          size_t cu_count,
                          size_t max_cus) {
  size_t tiles = compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  size_t sk_grid = tiles;  // Fallback if no good fractional tile is found
  if (max_cus > 0) sk_grid = std::min(sk_grid, max_cus);

  const size_t iters_per_tile = num_iters_per_tile(config.mt.k, problem.size.k);

  const size_t tile_size = config.mt.m * config.mt.n * config.workspace_size_per_elem_c;

  // Returns true if the candidate sk_grid produces per-CTA k-iter stripes whose
  // contiguous-dim footprint does not align to a full cache line on either operand.
  auto causes_partial_cachelines = [&](size_t candidate_grid) -> bool {
    constexpr size_t CACHE_LINE_BYTES = 128;
    if (candidate_grid == 0 || iters_per_tile == 0) return false;
    const size_t iters_total    = num_iters_total(tiles, iters_per_tile);
    const size_t iters_per_cta  = num_iters_per_cta(iters_total, candidate_grid);
    const size_t fragment_iters = iters_per_cta % iters_per_tile;
    const size_t bpe_a          = static_cast<size_t>(data_type_to_bytes(problem.a_dtype));
    const size_t bpe_b          = static_cast<size_t>(data_type_to_bytes(problem.b_dtype));
    const size_t a_contig_bytes = (problem.a_transpose == transpose_t::T)
                                      ? fragment_iters * config.mt.k * bpe_a
                                      : 0;
    const size_t b_contig_bytes = (problem.b_transpose == transpose_t::N)
                                      ? fragment_iters * config.mt.k * bpe_b
                                      : 0;
    auto not_aligned_to_cache_line = [](size_t bytes) {
      return bytes > 0 && (bytes % CACHE_LINE_BYTES) != 0;
    };
    return not_aligned_to_cache_line(a_contig_bytes) || not_aligned_to_cache_line(b_contig_bytes);
  };

  // More tiles than CUs
  // Distribute tiles evenly across maximum number of CUs
  // Split remaining tiles as evenly as possible for better caching
  if (tiles > cu_count) {
    size_t virt_cu_count = cu_count;
    if (config.occupancy > 1 && max_cus == 0) virt_cu_count *= config.occupancy;

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

      // Skip grids whose per-CTA k-iter fragment crosses a cache line boundary
      if (causes_partial_cachelines(frac_grid)) 
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

  return sk_grid;
}

size_t select_grid_size(const problem_t& problem,
                        const hardware_t& hardware,
                        const config_t& config,
                        grid_selection_t algorithm,
                        size_t max_cus) {
  size_t cu_count = hardware.N_CU;
  if (max_cus > 0) cu_count = std::min(cu_count, max_cus);

  switch (algorithm) {
    case grid_selection_t::min_resources:
      return streamk::grid_min_resources(problem, config, cu_count);

    case grid_selection_t::energy_aware:
      return streamk::grid_energy_aware(problem, config, cu_count);

    case grid_selection_t::reduction_cost_aware:
      return streamk::grid_reduction_cost_aware(problem, config, 1, cu_count);

    case grid_selection_t::data_parallel: return streamk::grid_data_parallel(problem, config);

    case grid_selection_t::analytical:
      return streamk::grid_analytical(problem, hardware, config, 10, max_cus);

    case grid_selection_t::k_split_aware:
      return streamk::grid_k_split_aware(problem, config, cu_count, max_cus);

    case grid_selection_t::number_of_cus:
    default: return hardware.N_CU;
  }
}
}  // namespace streamk
}  // namespace origami
