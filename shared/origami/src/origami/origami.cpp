// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iomanip>
#include <iostream>

#include "origami/gemm.hpp"
#include "origami/math.hpp"
#include "origami/origami.hpp"
#include "origami/streamk.hpp"
#include "origami/types.hpp"

namespace origami {

static double read_heuristics_variance_env_var() {
  constexpr double default_variance = 0.01;  // 1%

  if (const char* env = std::getenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE")) {
    try {
      double val = std::stod(env);
      if (std::isfinite(val) && val > 0.0) { return val; }
    } catch (...) {
      // fall through to default
    }
  }
  return default_variance;
}

//
// 1) Define a helper function to compute the arithmetic intensity of a tile.
//    Here we assume:
//    - Flops for tile (MT_M, MT_N, MT_K) is: 2 * MT_M * MT_N * MT_K
//    - Memory traffic approximated as: MT_M*MT_K + MT_K*MT_N + MT_M*MT_N
//    - Arithmetic intensity = flops / memory_traffic
auto compute_arithmetic_intensity = [](const config_t& config) -> double {
  // The tuple is: (latency, MT_M, MT_N, MT_K, MI_M, MI_N, MI_K)
  auto MT_M = config.mt.m;
  auto MT_N = config.mt.n;
  auto MT_K = config.mt.k;

  double flops          = static_cast<double>(2ull * MT_M * MT_N * MT_K);
  double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

  // Avoid division by zero.
  if (memory_traffic == 0.0) return 0.0;

  return flops / memory_traffic;
};

//
// Tiebreaker function.
//
void pick_best_config_by_arithmetic_intensity(std::vector<prediction_result_t>& top_results,
                                              size_t num_to_sort) {
  if (top_results.empty()) {
    throw std::runtime_error("pick_best_tile_by_arithmetic_intensity received empty list.");
  }

  // 2) Sort the top_results in descending order of arithmetic intensity
  //    (highest arithmetic intensity first).
  std::stable_sort(top_results.begin(),
                   top_results.begin() + num_to_sort,
                   [&](const prediction_result_t& a, const prediction_result_t& b) {
                     double ai_a = compute_arithmetic_intensity(a.config);
                     double ai_b = compute_arithmetic_intensity(b.config);
                     return ai_a > ai_b;  // descending
                   });
  // 3) Return the tile with the highest arithmetic intensity
}

std::vector<prediction_result_t> select_config(const problem_t& problem,
                                               const hardware_t& hardware,
                                               const std::vector<config_t>& configs) {
  // Use rank_configs to get configurations with latencies ranked by performance
  auto results = rank_configs(problem, hardware, configs);

  if (results.empty()) { throw std::runtime_error("No valid configs found."); }

  // Apply tie-breaking logic for configs with similar latency
  double best_latency = results.front().latency;
  size_t num_the_same = 0;

  // Count the number of similar latencies
  constexpr double epsilon = 1e-9;
  // variance is set through environment variable ANALYTICAL_GEMM_HEURISTICS_VARIANCE
  static const double top_N_heuristic = read_heuristics_variance_env_var();
  for (const auto& res : results) {
    bool within_top;
    const double diff = std::abs(res.latency - best_latency);

    if (top_N_heuristic <= epsilon) {
      // Absolute tolerance path
      within_top = diff < epsilon;
    } else {
      // Relative tolerance path (guard denom)
      const double denom = std::max(std::abs(best_latency), epsilon);
      // If it's within top_N_heuristic%, include it.
      within_top = (diff / denom) < top_N_heuristic;
    }

    if (within_top)
      ++num_the_same;
    else
      break;
  }

  // Finally, use your existing tie-breaker on top_candidates
  pick_best_config_by_arithmetic_intensity(results, num_the_same);

  // After arithmetic intensity tie-breaking, check if we still have ties
  // among the top results (those with same latency and arithmetic intensity)
  if (num_the_same > 1) {
    // Check if the top tiles still have the same arithmetic intensity
    double first_ai    = compute_arithmetic_intensity(results.front().config);
    size_t num_same_ai = 1;
    for (size_t i = 1; i < num_the_same; ++i) {
      double current_ai = compute_arithmetic_intensity(results[i].config);
      if (std::abs(current_ai - first_ai) < 1e-6) {
        num_same_ai++;
      } else {
        break;
      }
    }

    // If we still have ties after arithmetic intensity, apply problem dimension tie-breaker
    if (num_same_ai > 1) {
      // Problem dimension-based tie breaker:
      // If M > N, prefer tiles with larger MT_M
      // If N > M, prefer tiles with larger MT_N
      // If M == N, this tie-breaker doesn't apply (will use final tie-breaker)

      if (problem.size.m != problem.size.n) {
        std::stable_sort(results.begin(),
                         results.begin() + num_same_ai,
                         [problem](const prediction_result_t& a, const prediction_result_t& b) {
                           size_t MT_M_a = a.config.mt.m;
                           size_t MT_N_a = a.config.mt.n;
                           size_t MT_M_b = b.config.mt.m;
                           size_t MT_N_b = b.config.mt.n;

                           if (problem.size.m > problem.size.n) {
                             // M-dominant: prefer larger MT_M
                             if (MT_M_a != MT_M_b) return MT_M_a > MT_M_b;
                             // If MT_M is same, prefer larger MT_N as secondary
                             return MT_N_a > MT_N_b;
                           } else  // N > M
                           {
                             // N-dominant: prefer larger MT_N
                             if (MT_N_a != MT_N_b) return MT_N_a > MT_N_b;
                             // If MT_N is same, prefer larger MT_M as secondary
                             return MT_M_a > MT_M_b;
                           }
                         });
      }

      // Final tie-breaker: when all else is equal (including square problems),
      // consistently prefer tiles with larger MT_M
      // This ensures deterministic selection regardless of input order
      std::stable_sort(results.begin(),
                       results.begin() + num_same_ai,
                       [](const prediction_result_t& a, const prediction_result_t& b) {
                         size_t MT_M_a = a.config.mt.m;
                         size_t MT_N_a = a.config.mt.n;
                         size_t MT_K_a = a.config.mt.k;
                         size_t MT_M_b = b.config.mt.m;
                         size_t MT_N_b = b.config.mt.n;
                         size_t MT_K_b = b.config.mt.k;

                         // Prefer larger MT_M first
                         if (MT_M_a != MT_M_b) return MT_M_a > MT_M_b;
                         // If MT_M is same, prefer larger MT_N
                         if (MT_N_a != MT_N_b) return MT_N_a > MT_N_b;
                         // If both MT_M and MT_N are same, prefer larger MT_K
                         return MT_K_a > MT_K_b;
                       });
    }
  }

  if (hardware_t::is_debug_enabled()) {
    for (const auto& tile : results) {
      std::cout << problem.size.m << "x" << problem.size.n << "x" << problem.size.k
                << "Selected Macro-Tile: Latency=" << tile.latency << ", MT_M=" << tile.config.mt.m
                << ", MT_N=" << tile.config.mt.n << ", MT_K=" << tile.config.mt.k
                << ", MI_M=" << tile.config.mi.m << ", MI_N=" << tile.config.mi.n
                << ", MI_K=" << tile.config.mi.k << ", Occupancy=" << tile.config.occupancy
                << ", WGM=" << tile.config.workgroup_mapping
                << ", NonTemporalA=" << tile.config.non_temporal_a
                << ", NonTemporalB=" << tile.config.non_temporal_b << "\n";
    }
  }

  return results;
}

/**
 * @brief Selects the best WGM (maximizing L2 hit rate) given fixed macro tile sizes.
 *
 * @param[in] problem Problem description (M, N, K, etc.)
 * @param[in] hardware Hardware characteristics
 * @param config Kernel configuration.
 *
 * @return A tuple: best predicted (l2_hit_rate, wgm).
 */

std::tuple<size_t, size_t> select_workgroup_mapping(const problem_t& problem,
                                                    const hardware_t& hardware,
                                                    const config_t& config,
                                                    size_t skGrid) {
  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t K     = problem.size.k;
  size_t batch = problem.batch;

  size_t MT_M = config.mt.m;
  size_t MT_N = config.mt.n;
  size_t MT_K = config.mt.k;

  int nta = config.non_temporal_a;
  int ntb = config.non_temporal_b;

  // Default is the closest we can get to a square
  size_t max_CU_XCD = hardware.N_CU / hardware.NUM_XCD;
  size_t defaultWGM = ceil(std::sqrt(max_CU_XCD));

  // Number of output MTs per split and batch
  size_t numMT_M = math::safe_ceil_div(M, MT_M);
  size_t numMT_N = math::safe_ceil_div(N, MT_N);
  size_t numMTs  = numMT_M * numMT_N;

  // What SK does -- we already have skGrid so just compute numWaves and splitFactor
  auto numWaves    = skGrid > numMTs ? math::safe_ceil_div(skGrid, hardware.N_CU)
                                     : math::safe_ceil_div(numMTs, hardware.N_CU);
  auto splitFactor = math::safe_ceil_div(skGrid, numMTs);

  // -------------------
  // NonTemporal Cases
  // -------------------
  if (nta > 3 && ntb < 4)
    return std::make_tuple(hardware.NUM_XCD, 1);
  else if (nta < 4 && ntb > 3)
    return std::make_tuple(hardware.NUM_XCD,
                           std::min(max_CU_XCD, math::safe_ceil_div(numMTs, hardware.NUM_XCD)));
  else if (nta > 3 && ntb > 3)
    return std::make_tuple(hardware.NUM_XCD, 1);

  // -------------------
  // WGMXCC Prediction
  // -------------------
  // Default WGMXCC -- always number of XCD
  auto defaultWGMXCC = hardware.NUM_XCD;
  bool isWGMXCCset   = false;
  size_t out_wgmxcc  = defaultWGMXCC;

  // Batched GEMMs
  if (batch > 1 && !isWGMXCCset) {
    // Total tiles including batch count
    size_t numTotalTiles = numMTs * batch;

    // if only one MT per each GEMM -> no mapping
    // if less than hardware.NUM_XCD total tiles -> no mapping
    if (numMTs == 1 || numTotalTiles <= hardware.NUM_XCD) {
      out_wgmxcc  = 1;
      isWGMXCCset = true;
    }
    // else use the default (num_xcd)
  }

  // If we are lucky that the splitFactor is a multiple of NUM_XCD -> no mapping
  if ((splitFactor % hardware.NUM_XCD == 0) && !isWGMXCCset) {
    out_wgmxcc  = 1;
    isWGMXCCset = true;
  }

  // Small GEMMs
  if ((numMTs <= hardware.NUM_XCD) && !isWGMXCCset) {
    out_wgmxcc  = 1;
    isWGMXCCset = true;
  }

  // For sizes that we have more than 2 waves of computations, we skip xcc mapping as MALL is
  // more important -- matrix should not be skinny
  // To avoid regressions, it's set to default, but it should actually be 1!
  bool MallIsImportant =
      (splitFactor == 1 && batch == 1 && numMTs > 2 * hardware.N_CU && numMT_M > 8 && numMT_N > 8);
  if (MallIsImportant && !isWGMXCCset) {
    out_wgmxcc  = defaultWGMXCC;
    isWGMXCCset = true;
  }

  // -------------------
  // WGM Prediction
  // -------------------
  // Default WGM
  bool isWGMset  = false;
  size_t out_wgm = defaultWGM;

  // shortcut:
  // 1. if we have decided to not remap xcc, there is no reason to use wgm
  // 2. GEMMs that only have one tile in one dimension don't need wgm
  // 3. Batched GEMMs don't need wgm (emprically -> batch count is often large!)
  if (((out_wgmxcc == 1 && !MallIsImportant) || numMT_M == 1 || numMT_N == 1 || batch > 1) &&
      !isWGMset) {
    out_wgm  = 1;
    isWGMset = true;
  }

  // For tall cases (M >> N), if we have enough tiles to schedule, we use the number of tiles
  // in the smaller dimension as WGM value
  if (numMTs > hardware.N_CU && M > 10 * N && numMT_N <= 8) {
    out_wgm  = numMT_N;
    isWGMset = true;
  }

  // Cases where we have multiple rounds of computation per each CU
  // To avoid regressions, it's set to defaultWGM. However, I think WGM=1 should be the winner
  if (MallIsImportant && !isWGMset) {
    out_wgm  = defaultWGM;
    isWGMset = true;
  }

  if (!isWGMset) {
    size_t numWGs = numWaves * splitFactor * numMTs;
    size_t q      = numWGs / hardware.NUM_XCD;
    size_t r      = numWGs % hardware.NUM_XCD;

    std::vector<int> wgmList = {1, 2, 3, 4, 5, 6, 8, 16};
    int bestWGM              = 1;
    int bestL2               = std::numeric_limits<int>::max();
    for (auto wgm : wgmList) {
      auto slabTiles     = numMT_M * std::min(wgm, static_cast<int>(numMT_N));
      auto slabCount     = math::safe_ceil_div(numMT_N, wgm);
      auto edgeSlabWidth = numMT_N - (slabCount - 1) * wgm;
      auto wgmL2Estimate = 0;
      auto numXCD        = std::min(hardware.NUM_XCD, numWGs);

      // Compute unique loads per L2 tile
      for (uint32_t x = 0; x < numWaves * numXCD; ++x) {
        // Range of "output tiles" that this xcd takes.
        auto xccStart = q * x + (x < r ? x : r);
        auto xccEnd   = xccStart + q - 1 + (x < r ? 1 : 0);
        // xccStart and xccEnd are supposed to be tile IDs
        // In case of splitting, they are WG IDs. Modify to get tile IDs
        xccStart /= splitFactor;
        xccEnd /= splitFactor;

        auto slabStart = xccStart / slabTiles;
        auto slabEnd   = xccEnd / slabTiles;

        auto firstSlabWidth      = (slabStart == slabCount - 1 ? edgeSlabWidth : wgm);
        auto firstSlabStartIndex = xccStart % slabTiles;
        auto firstSlabStartRow   = firstSlabStartIndex / firstSlabWidth;
        auto firstSlabEndRow =
            std::min((firstSlabStartIndex + (xccEnd - xccStart)) / firstSlabWidth, numMT_M - 1);
        auto rowsInFirstSlab = firstSlabEndRow - firstSlabStartRow + 1;

        auto lastSlabWidth    = (slabEnd == slabCount - 1 ? edgeSlabWidth : wgm);
        auto lastSlabEndIndex = xccEnd % slabTiles;
        auto lastSlabEndRow   = lastSlabEndIndex / lastSlabWidth;
        auto colsInLastRow    = (lastSlabEndIndex % lastSlabWidth) + 1;
        auto colsInLastSlab   = (lastSlabEndRow > 0 ? lastSlabWidth : colsInLastRow);

        size_t uniqueRows = 0;
        size_t uniqueCols = 0;
        if (slabEnd == slabStart) {
          uniqueRows = lastSlabEndRow - firstSlabStartRow + 1;
          uniqueCols = firstSlabWidth;
          if (rowsInFirstSlab <= 2) uniqueCols = std::min(xccEnd - xccStart + 1, firstSlabWidth);
        } else {
          auto colsInFirstRow  = firstSlabWidth - (xccStart % firstSlabWidth);
          auto colsInFirstSlab = (rowsInFirstSlab > 1 ? firstSlabWidth : colsInFirstRow);
          auto fullSlabs       = slabEnd - slabStart - 1;
          uniqueRows =
              (fullSlabs > 0 ? numMT_M : std::min(rowsInFirstSlab + lastSlabEndRow + 1, numMT_M));
          uniqueCols = colsInFirstSlab + colsInLastSlab + fullSlabs * wgm;
        }

        // Sum up the L2 loads over all XCD
        // We should technically multiply by K (or splitted K), but it
        // has no effect on sorting
        auto xccL2Estimate = uniqueRows * MT_M + uniqueCols * MT_N;
        wgmL2Estimate += xccL2Estimate;
      }

      // If we have found a better WGM
      if (wgmL2Estimate < bestL2) {
        bestL2  = wgmL2Estimate;
        bestWGM = wgm;
      }

      if (hardware_t::is_debug_enabled())
        std::cout << "WGM (" << wgm << "), L2Estimate (" << wgmL2Estimate << ")" << std::endl;
    }

    out_wgm = bestWGM;
  }

  return std::make_tuple(out_wgmxcc, out_wgm);
}

std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs) {
  if (configs.empty()) { throw std::runtime_error("No configurations provided."); }

  std::vector<prediction_result_t> results(configs.size());

  std::transform(std::execution::seq,
                 configs.begin(),
                 configs.end(),
                 results.begin(),
                 [&](const config_t& config) -> prediction_result_t {
                   if (!check_lds_capacity(hardware, config.mt, problem.a_dtype, problem.b_dtype)) {
                     return {std::numeric_limits<double>::max(), config};
                   }
                   double latency = compute_total_latency(problem, hardware, config, hardware.N_CU);
                   return {latency, config};
                 });

  results.erase(std::remove_if(results.begin(),
                               results.end(),
                               [](const prediction_result_t& p) {
                                 return p.latency == std::numeric_limits<double>::max();
                               }),
                results.end());

  std::stable_sort(results.begin(),
                   results.end(),
                   [](const prediction_result_t& a, const prediction_result_t& b) {
                     return a.latency < b.latency;
                   });

  return results;
}

std::vector<prediction_result_t> select_config_mnk(size_t M,
                                                   size_t N,
                                                   size_t K,
                                                   const hardware_t& hardware,
                                                   const std::vector<config_t>& configs) {
  // Create a default problem_t with the provided M, N, K and reasonable defaults
  problem_t problem;
  problem.size.m          = M;
  problem.size.n          = N;
  problem.size.k          = K;
  problem.batch           = 1;
  problem.a_transpose     = transpose_t::T;     // Default to T
  problem.b_transpose     = transpose_t::N;     // Default to N
  problem.a_dtype         = data_type_t::Half;  // Default to fp16
  problem.b_dtype         = data_type_t::Half;  // Default to fp16
  problem.c_dtype         = data_type_t::Half;  // Default to fp16
  problem.d_dtype         = data_type_t::Half;  // Default to fp16
  problem.mi_dtype        = data_type_t::Half;  // Default to fp16
  problem.a_mx_block_size = 0;                  // Default MX block size
  problem.b_mx_block_size = 0;                  // Default MX block size

  // Use the existing select_config function with the constructed problem
  return select_config(problem, hardware, configs);
}

std::vector<prediction_result_t> select_topk_configs(const problem_t& problem,
                                                     const hardware_t& hardware,
                                                     const std::vector<config_t>& configs,
                                                     size_t topk) {
  auto ranked_configs = rank_configs(problem, hardware, configs);

  // Return only the top K configurations
  std::vector<prediction_result_t> topk_configs;
  size_t count = std::min(topk, ranked_configs.size());
  topk_configs.reserve(count);
  for (size_t i = 0; i < count; ++i) { topk_configs.push_back(ranked_configs[i]); }
  return topk_configs;
}

double compute_perf_gflops(const hardware_t& hardware,
                           const problem_t& problem,
                           const double latency) {
  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t K     = problem.size.k;
  size_t batch = problem.batch;

  // Compute total FLOPs
  double total_FLOPs = 2.0 * M * N * K;  // For GEMM, each multiply-add is 2 FLOPs
  // Compute total time in seconds
  double cycles_per_second = hardware.compute_clock_ghz * 1e9;  // 1 GHz = 1e9 cycles per second

  double total_time_seconds = latency / cycles_per_second;

  // Compute performance in FLOPS
  double FLOPS = total_FLOPs / total_time_seconds;
  // Convert to GFLOPS
  double GFLOPS = FLOPS / 1e9;  // 1 TFLOP = 1e9 FLOPs
  return GFLOPS;
}
}  // namespace origami