// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "origami/gemm.hpp"
#include "origami/math.hpp"
#include "origami/origami.hpp"
#include "origami/streamk.hpp"
#include "origami/types.hpp"

namespace origami {

std::vector<prediction_result_t> select_topk_configs(const problem_t& problem,
                                                     const hardware_t& hardware,
                                                     const std::vector<config_t>& configs,
                                                     std::size_t topk) {
  // Use rank_configs to get configurations with latencies ranked by performance
  auto ranked_configs = rank_configs(problem, hardware, configs);

  // Return only the top K configurations
  std::vector<prediction_result_t> topk_configs;
  size_t count = std::min(topk, ranked_configs.size());
  topk_configs.reserve(count);
  for (size_t i = 0; i < count; ++i) { topk_configs.push_back(ranked_configs[i]); }
  return topk_configs;
}

/**
 * @brief Selects the best workgroup mapping parameters (maximizing cache hits) given fixed macro
 * tile sizes.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics
 * @param config Kernel configuration.
 * @param skGrid SK grid.
 * @return A workgroup_mapping_t struct: best predicted (wgmxccchunk, wgmxcc, wgm).
 */
workgroup_mapping_t select_workgroup_mapping(const problem_t& problem,
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

  int nta = config.cache_hints_a;
  int ntb = config.cache_hints_b;

  // Default values
  size_t numCUs             = hardware.N_CU;
  size_t numXCD             = hardware.NUM_XCD;
  size_t numCUsPerXCD       = numCUs / numXCD;
  size_t defaultWGMXCCCHUNK = 0;
  size_t defaultWGMXCC      = hardware.NUM_XCD;
  int32_t defaultWGM        = ceil(std::sqrt(numCUsPerXCD));

  // Number of output MTs per split and batch
  size_t numMT_M = math::safe_ceil_div(M, MT_M);
  size_t numMT_N = math::safe_ceil_div(N, MT_N);
  size_t numMTs  = numMT_M * numMT_N;

  // What SK does -- we already have skGrid so just compute num_timesteps and split_factor
  auto num_timesteps =
      skGrid > numMTs ? math::safe_ceil_div(skGrid, numCUs) : math::safe_ceil_div(numMTs, numCUs);
  auto split_factor = math::safe_ceil_div(skGrid, numMTs);

  // -------------------
  // NonTemporal Cases
  // -------------------
  {
    // There is no need to use any mapping if one dimension is one.
    bool use_wgmxcc = (numMT_M != 1 && numMT_N != 1);
    // If we are using wgmxcc, we can use wgm, otherwise we don't need wgm.
    bool use_wgm = use_wgmxcc;
    // If we are using wgmxcc, we can use chunking, otherwise we don't need chunking.
    // Moreover, we only use chunking if all XCDs take the same number of tiles, otherwise
    // we in each group (chunk) we have more than one XCD.
    bool use_chunk = use_wgmxcc && ((numMTs < numCUs && numMTs % numXCD == 0) || (numMTs % numCUs == 0));

    // If we are using chunking, we use the minimum of the number of tiles per XCD and the number of CUs per XCD.
    size_t out_wgmxccchunk = use_chunk ? std::min(math::safe_ceil_div(numMTs, numXCD), numCUsPerXCD) : 0;
    // If we are using wgmxcc, we use the number of XCDs.
    size_t out_wgmxcc = use_wgmxcc ? numXCD : 1;
    // If we are using wgm, we use the number of tiles in the smaller dimension.
    // The reason is that nontemporal dimension always load for all L2 tiles, so we can only
    // maximize the reuse in the other dimension.
    if(nta > 3 && ntb < 4)
      return workgroup_mapping_t{out_wgmxccchunk, out_wgmxcc, use_wgm ? static_cast<int>(numMT_N) : 1};
    else if(nta < 4 && ntb > 3)
      // We use negative value here
      return workgroup_mapping_t{out_wgmxccchunk, out_wgmxcc, use_wgm ? -static_cast<int>(numMT_M) : 1};
    else if(nta > 3 && ntb > 3)
      // Nothing to do in this case.
      return workgroup_mapping_t{0, numXCD, 1};
  }

  // -------------------
  // Batch Case
  // -------------------
  if (batch > 1) {
    // Total tiles including batch count
    size_t numTotalTiles = numMTs * batch;

    size_t wgmxccchunk, wgmxcc;
    int32_t wgm;

    if (numMTs == 1 || numTotalTiles <= numXCD) {
      wgmxccchunk = 0;
      wgmxcc      = 0;
      wgm         = 1;
    }
    // This gives a nice strided read pattern for batched GEMMs
    else if (numMTs % numXCD == 0) {
      wgmxccchunk = 0;
      wgmxcc      = 0;
      wgm         = 1;
    } else {
      wgmxccchunk = (numCUsPerXCD / numMTs) * numMTs;
      wgmxcc      = numXCD;
      wgm         = 1;
    }

    return workgroup_mapping_t{wgmxccchunk, wgmxcc, wgm};
  }

  // -------------------
  // WGMXCCCHUNK Prediction
  // -------------------
  size_t out_wgmxccchunk = defaultWGMXCCCHUNK;

  // For large square-ish GEMMs, we can benefit from chunking.
  constexpr size_t skinnyFactor = 12;
  bool isMallImportant =
      (batch == 1 && split_factor == 1 && numMTs > 4 * numCUs && numMT_M > 16 && numMT_N > 16);
  bool isSkinnyCase = std::min(numMT_M, numMT_N) <= skinnyFactor * std::max(numMT_M, numMT_N);
  if (isMallImportant && !isSkinnyCase)
    out_wgmxccchunk = numCUsPerXCD;
  else
    out_wgmxccchunk = 0;

  // -------------------
  // WGMXCC Prediction
  // -------------------
  size_t out_wgmxcc = defaultWGMXCC;

  if (split_factor % numXCD == 0) out_wgmxcc = 0;
  // Small GEMMs
  else if (numMTs <= numXCD)
    out_wgmxcc = 0;
  else
    out_wgmxcc = defaultWGMXCC;

  // -------------------
  // WGM Prediction
  // -------------------
  auto out_wgm = defaultWGM;

  // shortcut:
  // 1. if we have decided to not remap xcc, there is no reason to use wgm
  // 2. GEMMs that only have one tile in one dimension don't need wgm
  if (out_wgmxcc == 0 || numMT_M == 1 || numMT_N == 1) out_wgm = 1;
  // For tall cases (M >> N), if we have enough tiles to schedule, we use the number of tiles
  // in the smaller dimension as WGM value
  else if (numMTs >= numCUs && numMT_N <= 8)
    out_wgm = numMT_N;
  else {
    // List of candidates for WGM values
    size_t numWGsPerXCD = std::min(math::safe_ceil_div(numMTs, numXCD), numCUsPerXCD);
    int wgm_cap_size = std::min(numMT_N, numWGsPerXCD);
    std::set<int> wgmSet;
    // Add initial candidates that are <= wgm_cap_size
    for (int val : {1, 2, 3, 4, 6, 8}) {
      if (val <= wgm_cap_size) {
        wgmSet.insert(val);
      }
    }
    // Add all divisors of wgm_cap_size
    for (int i = 1; i <= std::sqrt(wgm_cap_size); i++) {
      if (wgm_cap_size % i == 0) {
        wgmSet.insert(i);
        wgmSet.insert(wgm_cap_size / i);
      }
    }    
    std::vector<int> wgmList(wgmSet.begin(), wgmSet.end());

    // Setup
    size_t numWGs, q, r;
    numWGs = num_timesteps * split_factor * numMT_M * numMT_N;
    q      = numWGs / numXCD;
    r      = numWGs % numXCD;

    // Loop through all WGM values and find the best one
    int bestWGM = 1;
    int bestL2  = std::numeric_limits<int>::max();
    for (auto wgm : wgmList) {
      auto wgmL2Estimate = 0;
      auto slabTiles     = numMT_M * std::min(wgm, static_cast<int>(numMT_N));
      auto slabCount     = math::safe_ceil_div(numMT_N, wgm);
      auto edgeSlabWidth = numMT_N - (slabCount - 1) * wgm;
      auto numXCDUsed    = std::min(numXCD, numWGs);

      // Compute unique loads per L2 tile
      for (uint32_t w = 0; w < num_timesteps; ++w) {
        // offset for this wave
        auto remainder = q % numCUsPerXCD;
        auto adjustedEndTileInRound =
            (w == num_timesteps - 1 && remainder != 0) ? remainder : numCUsPerXCD;
        for (uint32_t x = 0; x < numXCDUsed; ++x) {
          // Range of "output tiles" that this xcd takes.
          size_t xccStart, xccEnd;
          if (out_wgmxccchunk > 0) {
            // CHUNKED MODE: XCD x owns tiles [x*C, (x+1)*C)
            xccStart = x * out_wgmxccchunk + w * numCUsPerXCD;
            xccEnd   = xccStart + adjustedEndTileInRound - 1;
          } else {
            // NON-CHUNK MODE: XCD x owns tiles [q*x + min(x,r), q*(x+1) + min(x+1,r))
            // However, not all of these tiles are in the same wave/round.
            // only the first numCUsPerXCD tiles are in the same wave/round.
            xccStart = w * numCUsPerXCD + q * x + (x < r ? x : r);
            xccEnd   = xccStart + adjustedEndTileInRound - 1 + (x < r ? 1 : 0);
          }

          // xccStart and xccEnd are supposed to be tile IDs
          // In case of splitting, they are WG IDs. Modify to get tile IDs
          xccStart /= split_factor;
          xccEnd /= split_factor;

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
      }

      // If we have found a better WGM
      if (wgmL2Estimate < bestL2) {
        bestL2  = wgmL2Estimate;
        bestWGM = wgm;
      }
    }
    // Set the best WGM
    out_wgm = bestWGM;
  }

  return workgroup_mapping_t{out_wgmxccchunk, out_wgmxcc, out_wgm};
}

/**
 * @brief Selects the best staggerU parameters (maximizing cache hits) given fixed macro tile sizes.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics
 * @param config Kernel configuration.
 * @param skGrid SK grid.
 * @param autoWGM Auto-selected WGM.
 * @return A staggerU_t struct: best predicted (staggerUMapping, staggerU, staggerUStrideShift).
 */
staggerU_t select_staggerU(const problem_t& problem,
                           const hardware_t& hardware,
                           const config_t& config,
                           size_t skGrid,
                           int32_t wgm) {
  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t K     = problem.size.k;
  size_t batch = problem.batch;

  size_t MT_M = config.mt.m;
  size_t MT_N = config.mt.n;
  size_t MT_K = config.mt.k;

  int nta = config.cache_hints_a;
  int ntb = config.cache_hints_b;

  // Default values
  size_t numCUs                     = hardware.N_CU;
  size_t numXCD                     = hardware.NUM_XCD;
  size_t numCUsPerXCD               = numCUs / numXCD;
  size_t defaultStaggerUMapping     = 0;
  size_t defaultStaggerU            = 0;
  size_t defaultStaggerUStrideShift = 0;

  // Number of output MTs per split and batch
  size_t numMT_M = math::safe_ceil_div(M, MT_M);
  size_t numMT_N = math::safe_ceil_div(N, MT_N);
  size_t numMT_K = math::safe_ceil_div(K, MT_K);
  size_t numMTs  = numMT_M * numMT_N;

  // What SK does -- we already have skGrid so just compute num_timesteps and split_factor
  auto num_timesteps =
      skGrid > numMTs ? math::safe_ceil_div(skGrid, numCUs) : math::safe_ceil_div(numMTs, numCUs);
  auto split_factor = math::safe_ceil_div(skGrid, numMTs);

  // -------------------
  // NonTemporal Cases
  // -------------------
  // if(nta > 3 && ntb < 4)
  //   return workgroup_mapping_t{0, numMTs == 1 ? 1 : numXCD, 1};
  // else if(nta < 4 && ntb > 3)
  //   return workgroup_mapping_t{0, numMTs == 1 ? 1 : numXCD, -1};
  // else if(nta > 3 && ntb > 3)
  //   return workgroup_mapping_t{0, numMTs == 1 ? 1 : numXCD, 1};

  // -------------------
  // General Cases
  // -------------------
  // In a typical GEMM, CUs compete to access the first K-slice.
  // There are two tiers of memory contention:
  // 1. Inter-XCD (MALL): XCDs compete to access the first K-slice in MALL (MALL -> L2).
  //    - Important when CUs within each XCD are underutilized
  //    - Distributes work across XCDs to utilize all compute resources
  // 2. Intra-XCD (L2): CUs on the same XCD compete to access K-slices in L2 (L2 -> L1).
  //    - Critical when CUs within each XCD are saturated
  //    - Each XCD has only 16 L2 slices, creating local contention
  //
  // StaggerU Strategy based on workgroup distribution:
  // - If numWGsPerL2Tile < numXCD / 2: Sparse distribution -> use Mall stagger (inter-XCD)
  // - Otherwise: Dense distribution -> use L2 stagger (intra-XCD)
  //
  // When using L2 stagger: Stagger TWO L2 tiles instead of one
  // - This spreads K-slice access across more offsets
  //
  // The L2 tile size is determined by WGM, which partitions the output matrix across XCDs.

  size_t out_staggerUMapping     = 0;
  size_t out_staggerU            = 0;
  size_t out_staggerUStrideShift = 0;

  // Early Exit: Batch and num_timesteps are not supported yet
  if (batch != 1 || num_timesteps != 1) return staggerU_t{0, 0, 0};

  // Early Exit: if not enough K-slices, then no staggerU is needed
  if (numMT_K < 4) return staggerU_t{0, 0, 0};

  // Early Exit: splitK
  if (split_factor > 1) {
    // Check for memory channel conflicts in split-K case
    size_t split_k_bytes = math::safe_ceil_div(K, split_factor);
    size_t stride_bytes  = static_cast<size_t>(split_k_bytes * data_type_to_bytes(problem.a_dtype));
    bool has_stride_conflict = ((stride_bytes & (stride_bytes - 1)) == 0) && (stride_bytes >= 4096);
    if (has_stride_conflict) {
      out_staggerUMapping     = 0;
      out_staggerU            = 16;
      out_staggerUStrideShift = 1;
      return staggerU_t{out_staggerUMapping, out_staggerU, out_staggerUStrideShift};
    }
  }

  // Find WGM
  size_t abs_wgm = std::abs(wgm);

  // Find Mall Tile (the tile that is accessed by all CUs in one time step)
  size_t numWGsPerMallTile = numMT_M * numMT_N;
  size_t MallTile_N        = numMT_N;
  size_t MallTile_M        = numMT_M;

  // Find L2 Tile (the tile that is accessed by CUs on one XCD in one time step)
  size_t L2Tile_M = 0;
  size_t L2Tile_N = 0;
  size_t numWGsPerL2Tile =
      skGrid > numMTs ? math::safe_ceil_div(skGrid, numXCD) : math::safe_ceil_div(numMTs, numXCD);
  numWGsPerL2Tile = (numWGsPerL2Tile < numCUsPerXCD) ? numWGsPerL2Tile : numCUsPerXCD;
  if (wgm > 0) {
    // Positive WGM: row-major mapping
    L2Tile_N             = (wgm < numMT_N) ? wgm : numMT_N;
    size_t L2Tile_M_temp = math::safe_ceil_div(numWGsPerL2Tile, L2Tile_N);
    L2Tile_M             = (L2Tile_M_temp < numMT_M) ? L2Tile_M_temp : numMT_M;
    while (L2Tile_M * L2Tile_N < numWGsPerL2Tile && L2Tile_N < numMT_N) L2Tile_N++;
  } else if (wgm < 0) {
    // Negative WGM: column-major mapping
    L2Tile_M             = (abs_wgm < numMT_M) ? abs_wgm : numMT_M;
    size_t L2Tile_N_temp = math::safe_ceil_div(numWGsPerL2Tile, L2Tile_M);
    L2Tile_N             = (L2Tile_N_temp < numMT_N) ? L2Tile_N_temp : numMT_N;
    while (L2Tile_M * L2Tile_N < numWGsPerL2Tile && L2Tile_M < numMT_M) L2Tile_M++;
  } else {
    std::cerr << "[ORIGAMI]: Invalid WGM value " << wgm << " in select_staggerU" << std::endl;
    return staggerU_t{0, 0, 0};
  }

  // Early Exit: if the whole data doesn't fit in L2, then we have to fetch data from MALL
  // In this case, we better wait for data to be accessible in L2 rather than issuing another
  // MALL read. Therefore, staggerU only helps when ALL K-slices can remain resident in L2.
  // There is a 5% buffer to account for other overheads.
  if ((MT_M * L2Tile_M * data_type_to_bytes(problem.a_dtype) +
       MT_N * L2Tile_N * data_type_to_bytes(problem.b_dtype)) *
          K >
      0.95 * hardware.L2_capacity)
    return staggerU_t{0, 0, 0};  

  // Find StaggerU for L2  
  size_t L2_staggerUMapping = 0;
  size_t L2_staggerU        = 0;
  // Stagger TWO L2 tiles to spread K-slice access across more offsets
  if (L2Tile_M >= L2Tile_N) {
    L2_staggerUMapping = 0;
    L2_staggerU        = std::min(2 * L2Tile_M, numMT_M);
  } else {
    L2_staggerUMapping = 1;
    L2_staggerU        = std::min(2 * L2Tile_N, numMT_N);
  }

  // Find StaggerU for Mall
  size_t Mall_staggerUMapping = 0;
  size_t Mall_staggerU        = 0;
  if (MallTile_M >= MallTile_N) {
    Mall_staggerUMapping = 0;
    Mall_staggerU        = MallTile_M;
  } else {
    Mall_staggerUMapping = 1;
    Mall_staggerU        = MallTile_N;
  }

  // Decision: Check workgroup distribution across XCDs
  // If we have very few workgroups per XCD (less than half the number of XCDs),
  // it indicates sparse work distribution -> use Mall stagger for inter-XCD distribution
  // Otherwise, use L2 stagger for intra-XCD contention reduction
  if (numWGsPerL2Tile < numXCD / 2) {
    // Sparse work distribution: use Mall stagger
    out_staggerUMapping = Mall_staggerUMapping;
    out_staggerU        = Mall_staggerU;
  } else {
    // Dense work distribution: use L2 stagger to reduce intra-XCD contention
    out_staggerUMapping = L2_staggerUMapping;
    out_staggerU        = L2_staggerU;
  }

  // Compute smallest power of 2 larger than or equal to out_staggerU
  // 64 is the maximum value for staggerU, however, 32 is a more practical value for most cases.
  size_t powerOf2 = 2;
  while (powerOf2 < out_staggerU) powerOf2 <<= 1;
  out_staggerU            = (powerOf2 < 32) ? powerOf2 : 32;
  out_staggerUStrideShift = 1;

  return staggerU_t{out_staggerUMapping, out_staggerU, out_staggerUStrideShift};
}

std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs) {
  if (configs.empty()) { throw std::runtime_error("No configurations provided."); }

  std::vector<prediction_result_t> results(configs.size());

  std::transform(configs.begin(),
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

  if (results.empty()) { throw std::runtime_error("No valid configs found."); }

  // Compute arithmetic intensity for tie-breaking
  // Flops = 2 * MT_M * MT_N * MT_K, Memory traffic = MT_M*MT_K + MT_K*MT_N + MT_M*MT_N
  auto compute_arithmetic_intensity = [](const config_t& config) -> double {
    const auto MT_M = config.mt.m;
    const auto MT_N = config.mt.n;
    const auto MT_K = config.mt.k;

    const double flops          = static_cast<double>(2ull * MT_M * MT_N * MT_K);
    const double memory_traffic = static_cast<double>(MT_M * MT_K + MT_N * MT_K + MT_M * MT_N);

    if (memory_traffic == 0.0) return 0.0;
    return flops / memory_traffic;
  };

  // Apply tie-breaking logic for configs with similar latency
  double best_latency = results.front().latency;
  size_t num_the_same = 0;

  // Count the number of similar latencies
  constexpr double epsilon = 1e-9;
  // variance is set through environment variable ANALYTICAL_GEMM_HEURISTICS_VARIANCE
  // Use runtime_options from first config if available, otherwise global singleton
  const double top_N_heuristic = origami::runtime_options::get().heuristics_variance;
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

  // Sort top candidates by arithmetic intensity (descending - highest first)
  if (num_the_same > 1) {
    std::stable_sort(results.begin(),
                     results.begin() + num_the_same,
                     [&compute_arithmetic_intensity](const prediction_result_t& a,
                                                     const prediction_result_t& b) {
                       return compute_arithmetic_intensity(a.config) >
                              compute_arithmetic_intensity(b.config);
                     });

    // After arithmetic intensity tie-breaking, check if we still have ties
    // among the top results (those with same latency and arithmetic intensity)
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
                           if (problem.size.m > problem.size.n) {
                             // M-dominant: prefer larger MT_M
                             if (a.config.mt.m != b.config.mt.m)
                               return a.config.mt.m > b.config.mt.m;
                             // If MT_M is same, prefer larger MT_N as secondary
                             return a.config.mt.n > b.config.mt.n;
                           } else  // N > M
                           {
                             // N-dominant: prefer larger MT_N
                             if (a.config.mt.n != b.config.mt.n)
                               return a.config.mt.n > b.config.mt.n;
                             // If MT_N is same, prefer larger MT_M as secondary
                             return a.config.mt.m > b.config.mt.m;
                           }
                         });
      }

      // Final tie-breaker: when all else is equal (including square problems),
      // consistently prefer tiles with larger MT_M
      // This ensures deterministic selection regardless of input order
      std::stable_sort(results.begin(),
                       results.begin() + num_same_ai,
                       [](const prediction_result_t& a, const prediction_result_t& b) {
                         // Prefer larger MT_M first
                         if (a.config.mt.m != b.config.mt.m) return a.config.mt.m > b.config.mt.m;
                         // If MT_M is same, prefer larger MT_N
                         if (a.config.mt.n != b.config.mt.n) return a.config.mt.n > b.config.mt.n;
                         // If both MT_M and MT_N are same, prefer larger MT_K
                         return a.config.mt.k > b.config.mt.k;
                       });
    }
  }

  return results;
}

prediction_result_t select_config_mnk(size_t M,
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

prediction_result_t select_config(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const std::vector<config_t>& configs) {
  auto ranked_configs = rank_configs(problem, hardware, configs);

  // Return the top configuration
  return ranked_configs[0];
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
