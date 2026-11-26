// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <vector>

#include "origami/core/config.hpp"
#include "origami/core/data_type.hpp"
#include "origami/core/mapping.hpp"
#include "origami/core/math.hpp"
#include "origami/core/problem.hpp"
#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/hardware/family.hpp"
#include "origami/selection/workgroup_mapping.hpp"

namespace origami {
namespace {

// Workgroup mapping of the families that remap workgroups across XCDs: the split-K (K-Coherent),
// chunk and XCC parameters are predicted from the tile grid, and WGM is the slab width with the
// fewest estimated L2 loads.
template <class Family>
workgroup_mapping_t map_workgroups(const problem_t& problem,
                                   const hardware_t& hardware,
                                   const config_t& config,
                                   size_t skGrid) {
  // Extract parameters from structured types
  size_t M     = problem.size.m;
  size_t N     = problem.size.n;
  size_t batch = problem.batch;

  size_t MT_M = config.mt.m;
  size_t MT_N = config.mt.n;
  size_t MT_K = config.mt.k;

  int nta = config.cache_hints_a;
  int ntb = config.cache_hints_b;

  // Default values
  // Honor the caller's CU budget (problem.num_cus); 0 means use all CUs.
  size_t numCUs              = resolve_num_cus(problem.num_cus, hardware.N_CU);
  size_t numXCD              = hardware.NUM_XCD;
  size_t numCUsPerXCD        = numCUs / numXCD;
  size_t defaultWGMXCCSPLITK = 0;
  size_t defaultWGMXCCCHUNK  = 0;
  size_t defaultWGMXCC       = hardware.NUM_XCD;
  int32_t defaultWGM         = ceil(std::sqrt(numCUsPerXCD));

  // Number of output MTs per split and batch
  size_t numMT_M = math::safe_ceil_div(M, MT_M);
  size_t numMT_N = math::safe_ceil_div(N, MT_N);
  size_t numMTs  = numMT_M * numMT_N;

  // What SK does -- we already have skGrid so just compute num_timesteps and split_factor
  auto num_timesteps =
      skGrid > numMTs ? math::safe_ceil_div(skGrid, numCUs) : math::safe_ceil_div(numMTs, numCUs);
  auto split_factor = math::safe_ceil_div(skGrid, numMTs * batch);

  // Stream-K fixup deadlock prevention:
  // When the SK grid doesn't evenly divide tiles, some workgroups produce partial
  // results combined via a spin-wait fixup loop.
  // That loop assumes consecutive StreamKIdx values are dispatched in physical WG order.
  // The chunk transform reorders WG IDs across XCDs, breaking this assumption and potentially
  // filling all GPU execution slots with spinning fixup waves resulting in a cooperative deadlock.
  bool sk_has_partial_tiles =
      (skGrid > 0 && skGrid < (numMTs * batch) && (numMTs * batch) % skGrid != 0);

  // NonTemporal Cases
  {
    // There is no need to use any mapping if one dimension is one.
    bool use_wgmxcc = (numMT_M != 1 && numMT_N != 1);
    // If we are using wgmxcc, we can use wgm, otherwise we don't need wgm.
    bool use_wgm = use_wgmxcc;
    // If we are using wgmxcc, we can use chunking, otherwise we don't need chunking.
    // Moreover, we only use chunking if all XCDs take the same number of tiles, otherwise
    // we in each group (chunk) we have more than one XCD.
    bool use_chunk =
        use_wgmxcc && ((numMTs < numCUs && numMTs % numXCD == 0) || (numMTs % numCUs == 0));

    // If we are using chunking, we use the minimum of the number of tiles per XCD and the number of
    // CUs per XCD.
    size_t out_wgmxccchunk =
        use_chunk ? std::min(math::safe_ceil_div(numMTs, numXCD), numCUsPerXCD) : 0;
    if (sk_has_partial_tiles) out_wgmxccchunk = 0;
    // If we are using wgmxcc, we use the number of XCDs.
    size_t out_wgmxcc = use_wgmxcc ? numXCD : 1;
    // If we are using wgm, we use the number of tiles in the smaller dimension.
    // The reason is that nontemporal dimension always load for all L2 tiles, so we can only
    // maximize the reuse in the other dimension.
    if (nta > 3 && ntb < 4)
      return workgroup_mapping_t{
          0, out_wgmxccchunk, out_wgmxcc, use_wgm ? static_cast<int>(numMT_N) : 1};
    else if (nta < 4 && ntb > 3)
      // We use negative value here
      return workgroup_mapping_t{
          0, out_wgmxccchunk, out_wgmxcc, use_wgm ? -static_cast<int>(numMT_M) : 1};
    else if (nta > 3 && ntb > 3)
      // Nothing to do in this case.
      return workgroup_mapping_t{0, 0, numXCD, 1};
  }

  // Batch Case
  if (batch > 1) {
    // Total tiles including batch count
    size_t numTotalTiles = numMTs * batch;

    size_t wgmxccchunk, wgmxcc;
    int32_t wgm;

    if (numMTs == 1 || numTotalTiles <= numXCD) {
      wgmxccchunk = 0;
      wgmxcc      = 0;
      wgm         = 1;
    } else {
      wgmxccchunk = (numCUsPerXCD / numMTs) * numMTs;
      wgmxcc      = numXCD;
      wgm         = 1;
      if (numMT_M > 1 && numMT_N > 1) wgm = std::min(defaultWGM, static_cast<int32_t>(numMT_N));
    }

    if (sk_has_partial_tiles) wgmxccchunk = 0;
    return workgroup_mapping_t{0, wgmxccchunk, wgmxcc, wgm};
  }

  // WGMXCCSPLITK Prediction (K-Coherent reorder)
  // K-Coherent reorders WGs into K-last order then chunks them across XCDs so
  // (m,n) tiles at the same k-level land on the same XCD.  Requires:
  //   - split_factor > 1 (StreamK actually splits K)
  //   - MT_K * elem_bytes >= kCoherentMinBytesPerKIter.  This is a performance
  //     floor, not an alignment requirement: the reorder is a bijection on
  //     workgroup IDs, so every tile is computed and written exactly once
  //     whatever the DU chunk size.  What it buys is L2 locality between the
  //     k-levels of one (m,n) tile, and that stops paying once a K-iteration
  //     covers too little of a cache line to keep the grouped XCD reads dense.
  //   - skGrid <= numCUs: the reorder only groups k-levels within a single wave
  //     of workgroups, so beyond one wave there is nothing to gain.  numCUs is
  //     the resolved budget, so a CU cap tightens this too.  It also keeps the
  //     chunk at or below skGrid/numXCD <= numCUsPerXCD, well inside the 8-bit
  //     chunk field of the kernel argument (ContractionSolution internalArg1,
  //     bits 21:14), which would otherwise wrap.
  // The codegen handles non-multiple skGrid via tail identity-mapping, so we no
  // longer require skGrid % numMTs == 0 here.
  size_t out_wgmxccsplitk = defaultWGMXCCSPLITK;
  {
    constexpr size_t kCoherentMinBytesPerKIter = 64;
    double elemBytes =
        std::max(data_type_to_bytes(problem.a_dtype), data_type_to_bytes(problem.b_dtype));
    size_t bytesPerKIter = static_cast<size_t>(MT_K * elemBytes);

    if (split_factor > 1 && bytesPerKIter >= kCoherentMinBytesPerKIter && !sk_has_partial_tiles &&
        skGrid <= numCUs) {
      // Use floor division: K * MN <= skGrid.  Tail WGs are identity-mapped.
      out_wgmxccsplitk = numMTs > 0 ? skGrid / numMTs : 0;
    }
  }

  // WGMXCCCHUNK Prediction
  size_t out_wgmxccchunk = defaultWGMXCCCHUNK;

  if (out_wgmxccsplitk > 1) {
    // Chunked K-Coherent: chiplet_transform_chunked distributes K-last positions.
    // chunk must be <= skGrid/numXCD so the chunked transform completes at least
    // one full round.  Otherwise Q=0 and everything is identity-mapped.
    //
    // For small MN (MN <= maxChunk): chunk = MN * num_k_per_chunk.
    // For large MN: chunk is a partial set of MN tiles at one k-level.
    constexpr size_t cacheLineBytes            = 128;
    constexpr size_t kCoherentMinBytesPerKIter = 64;
    double elemBytes =
        std::max(data_type_to_bytes(problem.a_dtype), data_type_to_bytes(problem.b_dtype));
    size_t bytesPerKIter = static_cast<size_t>(MT_K * elemBytes);
    size_t maxChunk      = numXCD > 0 ? skGrid / numXCD : 0;

    if (numMTs > 0 && numMTs <= maxChunk) {
      size_t minKPerChunk = (bytesPerKIter >= kCoherentMinBytesPerKIter)
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

  // WGMXCC Prediction
  size_t out_wgmxcc = defaultWGMXCC;

  if (split_factor % numXCD == 0) {
    // The hardware round-robin dispatch already distributes k-splits evenly.
    // No mapping needed (and K-Coherent would just add overhead).
    out_wgmxcc       = 0;
    out_wgmxccsplitk = 0;
    out_wgmxccchunk  = 0;
  } else if (split_factor < numXCD && numXCD % split_factor != 0) {
    out_wgmxccsplitk = 0;
    out_wgmxccchunk  = 0;
    out_wgmxcc       = numXCD;
  } else if (out_wgmxccsplitk > 1) {
    // K-Coherent always needs XCC mapping active for the chunked distribution.
    out_wgmxcc = numXCD;
  }
  // Small output tiles with no split
  else if (numMTs <= numXCD && split_factor == 1)
    out_wgmxcc = 0;
  else
    out_wgmxcc = defaultWGMXCC;

  // WGM Prediction
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
    int wgm_cap_size    = std::min(numMT_N, numWGsPerXCD);
    std::set<int> wgmSet;
    // Add initial candidates that are <= wgm_cap_size
    for (int val : {1, 2, 3, 4, 6, 8}) {
      if (val <= wgm_cap_size) { wgmSet.insert(val); }
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
    int bestWGM                                = 1;
    int bestL2                                 = std::numeric_limits<int>::max();
    constexpr size_t kCoherentMinBytesPerKIter = 64;
    double elemBytes =
        std::max(data_type_to_bytes(problem.a_dtype), data_type_to_bytes(problem.b_dtype));
    size_t bytesPerKIter = static_cast<size_t>(MT_K * elemBytes);

    // L2 ties are order-sensitive. Wider WGM can improve reuse order, but a
    // wrong reuse direction is expensive, so prefer the smaller tied WGM
    // wherever K-Coherent is in play: at 64 B per K-iteration the larger tied
    // WGM measured up to 9% slower on the shapes that run the chunked
    // dispatch, because the L2 estimate below models chunked and non-chunked
    // ranges differently and a tie under one is not a tie under the other.
    bool preferLargerWGMTie = bytesPerKIter < kCoherentMinBytesPerKIter;

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
        if ((preferLargerWGMTie && wgm > bestWGM) || (!preferLargerWGMTie && wgm < bestWGM))
          bestWGM = wgm;
      }
    }
    // Set the best WGM
    out_wgm = bestWGM;
  }

  return workgroup_mapping_t{out_wgmxccsplitk, out_wgmxccchunk, out_wgmxcc, out_wgm};
}

// gfx1250 maps WGM only.  Its persistent kernels reread WorkGroup0 from the hardware id after the
// WGMXCC remap, and the chunked, K-Coherent and StaggerU remaps are untested there, so all of them
// stay off (WGMXCC 1, chunk 0, split-K 0).  WGM is the positive slab width whose first wave puts
// the fewest estimated A/B DRAM bytes on one L2 domain, priced with the same footprint proxy and
// hit-rate model as cdna5_model::memory (gemm::estimate_l2_hit_cdna5, which reads
// context.wgm.wgm; the selector sets it per candidate); ties keep WGM 1.
// The footprint is an approximation, so WGM 1 stays unless another width is estimated to be
// strictly cheaper.
// Cache hints and tensor-DMA are not read: every operand is priced as temporal.
// ClusterAIDMapping (TensileLite's kernel-side choice of which AID / L2 domain computes which
// output tile) is not modeled: tiles follow the hardware dispatch placement (dispatch_xcd).  A
// model of AID remapping has to change the per-domain tile assignment here and in the footprint
// of cdna5_model::memory.
template <>
workgroup_mapping_t map_workgroups<arch::cdna5>(const problem_t& problem,
                                                const hardware_t& hardware,
                                                const config_t& config,
                                                size_t) {
  // predict_workgroup_mapping on gfx1250 runs this selector for a StreamK auto mapping, and the
  // context constructor calls it.  The context therefore gets a config with a fixed mapping (which
  // predict_workgroup_mapping returns as is, no nested pick); nothing else of the context reads
  // config.workgroup_mapping or the presence of TensileLite params, and context.wgm is
  // overwritten per candidate below.
  config_t context_config          = config;
  context_config.workgroup_mapping = 1;
  if (!context_config.has_tensile_params()) context_config.tensile();
  gemm::context_t context(problem, hardware, context_config);
  const size_t grid_n              = std::max<size_t>(context.grid_n, 1);
  const operand_traffic_t& traffic = context.traffic;

  workgroup_mapping_t best{0, 0, 1, 1};
  double best_bytes = std::numeric_limits<double>::max();
  for (int wgm : {1, 2, 4, 8, 16, 32, 64}) {
    // Once the previous slab already spanned every column, wider slabs map the same tiles.
    if (wgm > 1 && static_cast<size_t>(wgm / 2) >= grid_n) break;
    context.wgm.wgm = wgm;
    const gemm::cache_hit_rates_t hit =
        gemm::estimate_l2_hit_cdna5(problem, hardware, config, context);
    const double bytes = (1.0 - std::get<0>(hit)) * traffic.a_iter_bytes * traffic.a_cl_share +
                         (1.0 - std::get<1>(hit)) * traffic.b_iter_bytes * traffic.b_cl_share;
    if (bytes < best_bytes * (1.0 - 1e-9)) {
      best_bytes = bytes;
      best.wgm   = wgm;
    }
  }
  return best;
}

}  // namespace

workgroup_mapping_t select_workgroup_mapping(const problem_t& problem,
                                             const hardware_t& hardware,
                                             const config_t& config,
                                             size_t skGrid) {
  // Early Exit: problem sizes are invalid
  if (problem.size.m < 1 || problem.size.n < 1 || problem.size.k < 1 || problem.batch < 1) {
    return workgroup_mapping_t{0, 0, 0};
  }

  return arch::dispatch(hardware, [&](auto family) {
    return map_workgroups<decltype(family)>(problem, hardware, config, skGrid);
  });
}

}  // namespace origami
