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
 * @return A workgroup_mapping_t struct: best predicted (wgmxccsplitk, wgmxccchunk, wgmxcc, wgm).
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

  // Early Exit: problem sizes are invalid
  if(M < 1 || N < 1 || K < 1 || batch < 1)
  {
    return workgroup_mapping_t{0, 0, 0};
  }

  // Default values
  size_t  numCUs              = hardware.N_CU;
  size_t  numXCD              = hardware.NUM_XCD;
  size_t  numCUsPerXCD        = numCUs / numXCD;
  size_t  defaultWGMXCCSPLITK = 0;
  size_t  defaultWGMXCCCHUNK  = 0;
  size_t  defaultWGMXCC       = hardware.NUM_XCD;
  int32_t defaultWGM          = ceil(std::sqrt(numCUsPerXCD));

  // Number of output MTs per split and batch
  size_t numMT_M = math::safe_ceil_div(M, MT_M);
  size_t numMT_N = math::safe_ceil_div(N, MT_N);
  size_t numMTs  = numMT_M * numMT_N;

  // What SK does -- we already have skGrid so just compute num_timesteps and split_factor
  auto num_timesteps =
      skGrid > numMTs ? math::safe_ceil_div(skGrid, numCUs) : math::safe_ceil_div(numMTs, numCUs);
  auto split_factor = math::safe_ceil_div(skGrid, numMTs);

  // Stream-K fixup deadlock prevention:
  // When the SK grid doesn't evenly divide tiles, some workgroups produce partial
  // results combined via a spin-wait fixup loop.
  // That loop assumes consecutive StreamKIdx values are dispatched in physical WG order.
  // The chunk transform reorders WG IDs across XCDs, breaking this assumption and potentially
  // filling all GPU execution slots with spinning fixup waves resulting in a cooperative deadlock.
  bool sk_has_partial_tiles = (skGrid > 0 && skGrid < (numMTs * batch) && (numMTs * batch) % skGrid != 0);

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
    if (sk_has_partial_tiles) out_wgmxccchunk = 0;
    // If we are using wgmxcc, we use the number of XCDs.
    size_t out_wgmxcc = use_wgmxcc ? numXCD : 1;
    // If we are using wgm, we use the number of tiles in the smaller dimension.
    // The reason is that nontemporal dimension always load for all L2 tiles, so we can only
    // maximize the reuse in the other dimension.
    if(nta > 3 && ntb < 4)
      return workgroup_mapping_t{0, out_wgmxccchunk, out_wgmxcc, use_wgm ? static_cast<int>(numMT_N) : 1};
    else if(nta < 4 && ntb > 3)
      // We use negative value here
      return workgroup_mapping_t{0, out_wgmxccchunk, out_wgmxcc, use_wgm ? -static_cast<int>(numMT_M) : 1};
    else if(nta > 3 && ntb > 3)
      // Nothing to do in this case.
      return workgroup_mapping_t{0, 0, numXCD, 1};
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

    if (sk_has_partial_tiles) wgmxccchunk = 0;
    return workgroup_mapping_t{0, wgmxccchunk, wgmxcc, wgm};
  }

  // -------------------
  // WGMXCCSPLITK Prediction (K-Coherent reorder)
  // -------------------
  // K-Coherent reorders WGs into K-last order then chunks them across XCDs so
  // (m,n) tiles at the same k-level land on the same XCD.  Requires:
  //   - split_factor > 1 (StreamK actually splits K)
  //   - MT_K * elem_bytes >= 128: each DU chunk covers at least one full cache
  //     line, so split boundaries are cache-line aligned even when extraIters > 0.
  // The codegen handles non-multiple skGrid via tail identity-mapping, so we no
  // longer require skGrid % numMTs == 0 here.
  size_t out_wgmxccsplitk = defaultWGMXCCSPLITK;
  {
    constexpr size_t cacheLineBytes = 128;
    double elemBytes = std::max(data_type_to_bytes(problem.a_dtype),
                                data_type_to_bytes(problem.b_dtype));
    size_t bytesPerKIter = static_cast<size_t>(MT_K * elemBytes);

    if (split_factor > 1 && bytesPerKIter >= cacheLineBytes && !sk_has_partial_tiles) {
      // Use floor division: K * MN <= skGrid.  Tail WGs are identity-mapped.
      out_wgmxccsplitk = numMTs > 0 ? skGrid / numMTs : 0;
    }
  }

  // -------------------
  // WGMXCCCHUNK Prediction
  // -------------------
  size_t out_wgmxccchunk = defaultWGMXCCCHUNK;

  if (out_wgmxccsplitk > 1) {
    // Chunked K-Coherent: chiplet_transform_chunked distributes K-last positions.
    // chunk must be <= skGrid/numXCD so the chunked transform completes at least
    // one full round.  Otherwise Q=0 and everything is identity-mapped.
    //
    // For small MN (MN <= maxChunk): chunk = MN * num_k_per_chunk.
    // For large MN: chunk is a partial set of MN tiles at one k-level.
    constexpr size_t cacheLineBytes = 128;
    double elemBytes = std::max(data_type_to_bytes(problem.a_dtype),
                                data_type_to_bytes(problem.b_dtype));
    size_t bytesPerKIter = static_cast<size_t>(MT_K * elemBytes);
    size_t maxChunk      = numXCD > 0 ? skGrid / numXCD : 0;

    if (numMTs > 0 && numMTs <= maxChunk) {
      size_t minKPerChunk = (bytesPerKIter >= cacheLineBytes)
          ? 1
          : math::safe_ceil_div(cacheLineBytes, bytesPerKIter);
      size_t maxKPerChunk = maxChunk / numMTs;
      size_t numK         = (maxKPerChunk / minKPerChunk) * minKPerChunk;
      if (numK == 0) numK = minKPerChunk;
      out_wgmxccchunk = numMTs * numK;
    } else {
      out_wgmxccchunk = maxChunk;
    }
  } else {
    // For large square-ish GEMMs without split-K, we can benefit from chunking.
    constexpr size_t skinnyFactor = 12;
    bool isMallImportant =
        (batch == 1 && split_factor == 1 && numMTs > 4 * numCUs && numMT_M > 16 && numMT_N > 16);
    bool isSkinnyCase = std::min(numMT_M, numMT_N) <= skinnyFactor * std::max(numMT_M, numMT_N);
    if (isMallImportant && !isSkinnyCase)
      out_wgmxccchunk = numCUsPerXCD;
    else
      out_wgmxccchunk = 0;
  }

  if (sk_has_partial_tiles) out_wgmxccchunk = 0;

  // -------------------
  // WGMXCC Prediction
  // -------------------
  size_t out_wgmxcc = defaultWGMXCC;

  if (split_factor % numXCD == 0) {
    // The hardware round-robin dispatch already distributes k-splits evenly.
    // No mapping needed (and K-Coherent would just add overhead).
    out_wgmxcc = 0;
    out_wgmxccsplitk = 0;
    out_wgmxccchunk  = 0;
  } else if (split_factor < numXCD && numXCD % split_factor != 0) {
    out_wgmxccsplitk = 0;
    out_wgmxccchunk  = 0;
    out_wgmxcc = numXCD;
  } else if (out_wgmxccsplitk > 1) {
    // K-Coherent always needs XCC mapping active for the chunked distribution.
    out_wgmxcc = numXCD;
  }
  // Small output tiles with no split
  else if (numMTs <= numXCD && split_factor == 1)
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
    constexpr size_t cacheLineBytes = 128;
    double elemBytes = std::max(data_type_to_bytes(problem.a_dtype),
                                data_type_to_bytes(problem.b_dtype));
    size_t bytesPerKIter = static_cast<size_t>(MT_K * elemBytes);

    // L2 ties are order-sensitive. Wider WGM can improve reuse order, but when
    // each K-step already spans more than a cache line, a wrong reuse direction
    // is expensive; prefer the smaller tied WGM in that regime. This policy is
    // deliberately cheap and applies to both splitK and non-splitK ties.
    bool preferLargerWGMTie = bytesPerKIter <= cacheLineBytes;

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

          // xccStart and xccEnd are WG IDs after the chiplet remap.
          // Convert to unique-tile-ID range that this XCD touches:
          if (out_wgmxccsplitk == 0) {
            xccStart /= split_factor;
            xccEnd /= split_factor;
          }

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
      } else if (wgmL2Estimate == bestL2) {
        if ((preferLargerWGMTie && wgm > bestWGM) ||
            (!preferLargerWGMTie && wgm < bestWGM))
          bestWGM = wgm;
      }
    }
    // Set the best WGM
    out_wgm = bestWGM;
  }

  return workgroup_mapping_t{out_wgmxccsplitk, out_wgmxccchunk, out_wgmxcc, out_wgm};
}

/**
 * @brief Selects the best staggerU parameters (maximizing cache hits) given fixed macro tile sizes.
 *
 * @param problem Problem description (M, N, K, etc.)
 * @param hardware Hardware characteristics
 * @param config Kernel configuration.
 * @param skGrid SK grid.
 * @param wgm WGM.
 * @return staggerU_t struct: best predicted (staggerUMapping, staggerU, staggerUStrideShift).
 */
staggerU_t select_staggerU(const problem_t& problem,
                           const hardware_t& hardware,
                           const config_t& config,
                           size_t skGrid,
                           int32_t wgm) {
  // StaggerU offsets the starting K-position per workgroup so that CUs on the same
  // XCD don't all hammer the same K-slice simultaneously. This function selects:
  //   - StaggerUMapping (SUM): which WG dimension determines the K-offset (0=M, 1=N)
  //   - StaggerU: number of unique K-offset positions (power of 2, <= 32)
  //   - StaggerUStrideShift: stride multiplier ensuring each step crosses an L2 cache line
  //
  // Approach:
  //   1. Compute max stagger from K-dimension (must fit within numMT_K iterations).
  //   2. Estimate the L2 tile shape (tiles per XCD) from WGM and XCD count.
  //   3. Choose the L2-optimal mapping direction using a contention model:
  //      - With row-major WGM, consecutive WGs form a row sharing A data. Their
  //        temporal overlap amplifies A contention (squared term). Compare against
  //        B contention from "column-mates" to pick SUM0 (distribute B) or SUM1 (distribute A).
  //   4. Validate the L2 working set: stagger only expands the shared matrix
  //      (the one whose reads are distributed). If it exceeds L2 capacity, disable.
  //   5. Optionally override the direction for MALL (inter-XCD) benefit when the
  //      L2 tile is asymmetric and enough intra-XCD offsets survive the switch.

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

  // Early Exit: problem sizes are invalid
  if(M < 1 || N < 1 || K < 1 || batch < 1)
  {
    return staggerU_t{0, 0, 0};
  }

  // Default values
  size_t numCUs       = hardware.N_CU;
  size_t numXCD       = hardware.NUM_XCD;
  size_t numCUsPerXCD = numCUs / numXCD;

  // Number of output MTs per split and batch
  size_t numMT_M = math::safe_ceil_div(M, MT_M);
  size_t numMT_N = math::safe_ceil_div(N, MT_N);
  size_t numMT_K = math::safe_ceil_div(K, MT_K);
  size_t numMTs  = numMT_M * numMT_N;

  // What SK does -- we already have skGrid so just compute num_timesteps and split_factor
  auto num_timesteps =
      skGrid > numMTs ? math::safe_ceil_div(skGrid, numCUs) : math::safe_ceil_div(numMTs, numCUs);
  auto split_factor = math::safe_ceil_div(skGrid, numMTs);

  // Early Exit: Batch is not supported yet
  if (batch != 1) return staggerU_t{0, 0, 0};

  // Early Exit: no staggerU needed
  if (numMT_K > 64)
    return staggerU_t{0, 0, 0};

  // Early Exit: splitK
  // TODO: support splitK
  if (split_factor > 1)     
      return staggerU_t{0, 0, 0};

  // helper function to round up to power of 2
  auto next_pow2 = [](size_t v) -> size_t {
    size_t p = 2;
    while (p < v) p <<= 1;
    return p;
  };

  // Compute stride shift and max staggerU from K
  constexpr size_t L2_CACHE_LINE_BYTES = 128;
  size_t bpe_a = static_cast<size_t>(data_type_to_bytes(problem.a_dtype));
  size_t bpe_b = static_cast<size_t>(data_type_to_bytes(problem.b_dtype));
  double min_bpe = std::min(bpe_a, bpe_b);
  size_t bytes_per_k_iter = static_cast<size_t>(MT_K * min_bpe);
  size_t min_shift = 0;
  while ((bytes_per_k_iter << min_shift) < L2_CACHE_LINE_BYTES && min_shift < 5)
    min_shift++;
  size_t out_staggerUStrideShift = min_shift;
  size_t max_staggerU = numMT_K >> out_staggerUStrideShift;
  // Round down to power of 2 and cap at 32
  {
    size_t p = 1;
    while (p * 2 <= max_staggerU) p <<= 1;
    max_staggerU = std::min(p, static_cast<size_t>(32));
  }

  // Early Exit: few K-slices
  if (max_staggerU == 1)
    return staggerU_t{0, 0, 0};

  // Early Exit: Non-temporal cases
  if (nta > 3)
    return staggerU_t{0, max_staggerU, out_staggerUStrideShift};
  if (ntb > 3)
    return staggerU_t{1, max_staggerU, out_staggerUStrideShift};

  // Find WGM
  size_t abs_wgm = std::abs(wgm);

  // Find L2 Tile (the tile that is accessed by CUs on one XCD in one time step)
  size_t L2Tile_M = 0;
  size_t L2Tile_N = 0;
  size_t numWGsPerL2Tile =
      skGrid > numMTs ? math::safe_ceil_div(skGrid, numXCD) : math::safe_ceil_div(numMTs, numXCD);
  numWGsPerL2Tile = (numWGsPerL2Tile < numCUsPerXCD) ? numWGsPerL2Tile : numCUsPerXCD;
  if (wgm > 0) {
    // Positive WGM: row-major mapping
    L2Tile_N             = (abs_wgm < numMT_N) ? abs_wgm : numMT_N;
    size_t L2Tile_M_temp = math::safe_ceil_div(numWGsPerL2Tile, L2Tile_N);
    L2Tile_M             = (L2Tile_M_temp < numMT_M) ? L2Tile_M_temp : numMT_M;
    while (L2Tile_M * L2Tile_N < numWGsPerL2Tile && L2Tile_N < numMT_N) L2Tile_N++;
    // Account for XCD misalignment: when tiles don't fill exact rows,
    // some XCDs start mid-row and span an extra row.
    if (numWGsPerL2Tile % L2Tile_N != 0)
      L2Tile_M = std::min(L2Tile_M + 1, numMT_M);
  } else if (wgm < 0) {
    // Negative WGM: column-major mapping
    L2Tile_M             = (abs_wgm < numMT_M) ? abs_wgm : numMT_M;
    size_t L2Tile_N_temp = math::safe_ceil_div(numWGsPerL2Tile, L2Tile_M);
    L2Tile_N             = (L2Tile_N_temp < numMT_N) ? L2Tile_N_temp : numMT_N;
    while (L2Tile_M * L2Tile_N < numWGsPerL2Tile && L2Tile_M < numMT_M) L2Tile_M++;
    // Account for XCD misalignment: when tiles don't fill exact columns,
    // some XCDs start mid-column and span an extra column.
    if (numWGsPerL2Tile % L2Tile_M != 0)
      L2Tile_N = std::min(L2Tile_N + 1, numMT_N);
  } else {
    std::cerr << "[ORIGAMI]: Invalid WGM value " << wgm << " in select_staggerU" << std::endl;
    return staggerU_t{0, 0, 0};
  }

  // Compute L2 optimal direction.
  // Row-major WGM: consecutive WGs share A (row-mates). Their temporal overlap
  // amplifies A contention -> the consecutive dimension (N) gets squared.
  // Column-major WGM: consecutive WGs share B (column-mates): M gets squared.
  size_t A_contention, B_contention;
  if (wgm > 0) {
    // Positive WGM (row-major): N is the consecutive dimension
    A_contention = L2Tile_N * L2Tile_N * MT_M * bpe_a;
    B_contention = L2Tile_M * MT_N * bpe_b;
  } else {
    // Negative WGM (column-major): M is the consecutive dimension
    A_contention = L2Tile_N * MT_M * bpe_a;
    B_contention = L2Tile_M * L2Tile_M * MT_N * bpe_b;
  }
  size_t L2_mapping = (B_contention > A_contention) ? 0 : 1;
  size_t L2_value   = (L2_mapping == 0) ? std::min(L2Tile_M, numMT_M)
                                        : std::min(L2Tile_N, numMT_N);
  L2_value = std::min(L2_value, max_staggerU);
  // L2 capacity check.
  // Stagger only expands the SHARED matrix — the other matrix's footprint is unchanged:
  //   SUM0: A unchanged (each M-position already reads different A data),
  //         B expanded by min(stagger, L2Tile_M) K-offsets (B sharing is broken).
  //   SUM1: B unchanged, A expanded by min(stagger, L2Tile_N) K-offsets.
  size_t working_set;
  if (L2_mapping == 0) {
    working_set = MT_K * (L2Tile_M * MT_M * bpe_a +
                          std::min(L2_value, L2Tile_M) * L2Tile_N * MT_N * bpe_b);
  } else {
    working_set = MT_K * (std::min(L2_value, L2Tile_N) * L2Tile_M * MT_M * bpe_a +
                          L2Tile_N * MT_N * bpe_b);
  }
  // 0.95 is a heuristic to make sure we don't exceed L2 capacity and if we are not
  // underestimating the working set. The effect of a false prediction is more severe
  // than not predicting at all.
  if (working_set > 0.95 * hardware.L2_capacity)
    return staggerU_t{0, 0, 0};
  // Early Exit: L2 value is already max_staggerU
  if (L2_value == max_staggerU)
    return staggerU_t{L2_mapping, max_staggerU, out_staggerUStrideShift};
  
  // Compute MALL optimal direction
  // Prefer direction with more XCD groups
  size_t numXCD_M = math::safe_ceil_div(numMT_M, L2Tile_M);
  size_t numXCD_N = math::safe_ceil_div(numMT_N, L2Tile_N);
  size_t Mall_mapping;
  if (numXCD_M > numXCD_N) {
    Mall_mapping = 0; 
  } else {
    Mall_mapping = 1; 
  }

  // Decide L2 vs MALL direction
  // If they agree, use that direction. If they disagree, check how many intra-XCD
  // offsets survive switching to MALL direction. Otherwise the L2 loss is too 
  // severe, keep L2 direction.
  size_t out_staggerUMapping = 0;
  size_t out_staggerU = 0;
  if (L2_mapping == Mall_mapping) {
    out_staggerUMapping = L2_mapping;
    out_staggerU = max_staggerU;
  } else {
    // L2 and MALL disagree. Only switch when BOTH conditions hold:
    // 1. Contention is close (ratio < 2)
    // 2. L2 tile is asymmetric (ratio > 2)
    size_t L2_winner = std::max(A_contention, B_contention);
    size_t L2_loser  = std::min(A_contention, B_contention);
    bool contention_close = (L2_winner < 2 * L2_loser);
    bool tile_asymmetric  = (std::max(L2Tile_M, L2Tile_N) > 2 * std::min(L2Tile_M, L2Tile_N));
    if (contention_close && tile_asymmetric) {
      out_staggerUMapping = Mall_mapping;
      out_staggerU = max_staggerU;
    } else {
      out_staggerUMapping = L2_mapping;
      out_staggerU = L2_value;
    }
  }

  // Sanity checks
  out_staggerU = std::min(next_pow2(out_staggerU), max_staggerU);
  if (out_staggerU <= 2) return staggerU_t{0, 0, 0};

  return staggerU_t{out_staggerUMapping, out_staggerU, out_staggerUStrideShift};
}

std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs) {
  if (configs.empty()) { throw std::runtime_error("No configurations provided."); }

  struct prediction_result_wrapper_t {
    double latency;
    std::reference_wrapper<const config_t> config;
  };

  std::vector<prediction_result_wrapper_t> latencies_configs;
  latencies_configs.reserve(configs.size());

  for (auto& config : configs) {
    if (!check_lds_capacity(hardware, config.mt, problem.a_dtype, problem.b_dtype))
      continue;
    double latency = compute_total_latency(problem, hardware, config, hardware.N_CU);
    if (latency != std::numeric_limits<double>::max())
      latencies_configs.push_back({latency, std::cref(config)});
  }

  if (latencies_configs.empty()) { throw std::runtime_error("No valid configs found."); }

  std::stable_sort(latencies_configs.begin(),
                   latencies_configs.end(),
                   [](const auto& a, const auto& b) {
                     return a.latency < b.latency;
                   });

  std::vector<prediction_result_t> results;
  results.reserve(latencies_configs.size());
  std::transform(latencies_configs.begin(),
                 latencies_configs.end(),
                 std::back_inserter(results),
                 [&](const auto& r) -> prediction_result_t {
                   return {r.latency, r.config.get()};
                 });

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
