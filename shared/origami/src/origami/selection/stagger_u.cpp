// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "origami/core/config.hpp"
#include "origami/core/data_type.hpp"
#include "origami/core/mapping.hpp"
#include "origami/core/math.hpp"
#include "origami/core/problem.hpp"
#include "origami/hardware.hpp"
#include "origami/hardware/family.hpp"
#include "origami/selection/stagger_u.hpp"

namespace origami {
namespace {

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
template <class Family>
staggerU_t stagger(const problem_t& problem,
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
  // Honor the caller's CU budget (problem.num_cus); 0 means use all CUs.
  size_t numCUs       = resolve_num_cus(problem.num_cus, hardware.N_CU);
  size_t numXCD       = hardware.NUM_XCD;
  size_t numCUsPerXCD = numCUs / numXCD;

  // Number of output MTs per split and batch
  size_t numMT_M = math::safe_ceil_div(M, MT_M);
  size_t numMT_N = math::safe_ceil_div(N, MT_N);
  size_t numMT_K = math::safe_ceil_div(K, MT_K);
  size_t numMTs  = numMT_M * numMT_N;

  // What SK does -- we already have skGrid so just compute split_factor
  auto split_factor = math::safe_ceil_div(skGrid, numMTs);

  // Early Exit: Batch is not supported yet
  if (batch != 1) return staggerU_t{0, 0, 0};

  // Early Exit: no staggerU needed
  if (numMT_K > 64) return staggerU_t{0, 0, 0};

  // Early Exit: splitK
  // TODO: support splitK
  if (split_factor > 1) return staggerU_t{0, 0, 0};

  // helper function to round up to power of 2
  auto next_pow2 = [](size_t v) -> size_t {
    size_t p = 2;
    while (p < v) p <<= 1;
    return p;
  };

  // Compute stride shift and max staggerU from K
  constexpr size_t L2_CACHE_LINE_BYTES = 128;
  size_t bpe_a                         = static_cast<size_t>(data_type_to_bytes(problem.a_dtype));
  size_t bpe_b                         = static_cast<size_t>(data_type_to_bytes(problem.b_dtype));
  double min_bpe                       = std::min(bpe_a, bpe_b);
  size_t bytes_per_k_iter              = static_cast<size_t>(MT_K * min_bpe);
  size_t min_shift                     = 0;
  while ((bytes_per_k_iter << min_shift) < L2_CACHE_LINE_BYTES && min_shift < 5) min_shift++;
  size_t out_staggerUStrideShift = min_shift;
  size_t max_staggerU            = numMT_K >> out_staggerUStrideShift;
  // Round down to power of 2 and cap at 32
  {
    size_t p = 1;
    while (p * 2 <= max_staggerU) p <<= 1;
    max_staggerU = std::min(p, static_cast<size_t>(32));
  }

  // Early Exit: few K-slices
  if (max_staggerU == 1) return staggerU_t{0, 0, 0};

  // Early Exit: Non-temporal cases
  if (nta > 3) return staggerU_t{0, max_staggerU, out_staggerUStrideShift};
  if (ntb > 3) return staggerU_t{1, max_staggerU, out_staggerUStrideShift};

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
    if (numWGsPerL2Tile % L2Tile_N != 0) L2Tile_M = std::min(L2Tile_M + 1, numMT_M);
  } else if (wgm < 0) {
    // Negative WGM: column-major mapping
    L2Tile_M             = (abs_wgm < numMT_M) ? abs_wgm : numMT_M;
    size_t L2Tile_N_temp = math::safe_ceil_div(numWGsPerL2Tile, L2Tile_M);
    L2Tile_N             = (L2Tile_N_temp < numMT_N) ? L2Tile_N_temp : numMT_N;
    while (L2Tile_M * L2Tile_N < numWGsPerL2Tile && L2Tile_M < numMT_M) L2Tile_M++;
    // Account for XCD misalignment: when tiles don't fill exact columns,
    // some XCDs start mid-column and span an extra column.
    if (numWGsPerL2Tile % L2Tile_M != 0) L2Tile_N = std::min(L2Tile_N + 1, numMT_N);
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
  size_t L2_value   = (L2_mapping == 0) ? std::min(L2Tile_M, numMT_M) : std::min(L2Tile_N, numMT_N);
  L2_value          = std::min(L2_value, max_staggerU);
  // L2 capacity check.
  // Stagger only expands the SHARED matrix — the other matrix's footprint is unchanged:
  //   SUM0: A unchanged (each M-position already reads different A data),
  //         B expanded by min(stagger, L2Tile_M) K-offsets (B sharing is broken).
  //   SUM1: B unchanged, A expanded by min(stagger, L2Tile_N) K-offsets.
  size_t working_set;
  if (L2_mapping == 0) {
    working_set =
        MT_K * (L2Tile_M * MT_M * bpe_a + std::min(L2_value, L2Tile_M) * L2Tile_N * MT_N * bpe_b);
  } else {
    working_set =
        MT_K * (std::min(L2_value, L2Tile_N) * L2Tile_M * MT_M * bpe_a + L2Tile_N * MT_N * bpe_b);
  }
  // 0.95 is a heuristic to make sure we don't exceed L2 capacity and if we are not
  // underestimating the working set. The effect of a false prediction is more severe
  // than not predicting at all.
  if (working_set > 0.95 * hardware.l2_domain_capacity()) return staggerU_t{0, 0, 0};
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
  size_t out_staggerU        = 0;
  if (L2_mapping == Mall_mapping) {
    out_staggerUMapping = L2_mapping;
    out_staggerU        = max_staggerU;
  } else {
    // L2 and MALL disagree. Only switch when BOTH conditions hold:
    // 1. Contention is close (ratio < 2)
    // 2. L2 tile is asymmetric (ratio > 2)
    size_t L2_winner      = std::max(A_contention, B_contention);
    size_t L2_loser       = std::min(A_contention, B_contention);
    bool contention_close = (L2_winner < 2 * L2_loser);
    bool tile_asymmetric  = (std::max(L2Tile_M, L2Tile_N) > 2 * std::min(L2Tile_M, L2Tile_N));
    if (contention_close && tile_asymmetric) {
      out_staggerUMapping = Mall_mapping;
      out_staggerU        = max_staggerU;
    } else {
      out_staggerUMapping = L2_mapping;
      out_staggerU        = L2_value;
    }
  }

  // Sanity checks
  out_staggerU = std::min(next_pow2(out_staggerU), max_staggerU);
  if (out_staggerU <= 2) return staggerU_t{0, 0, 0};

  return staggerU_t{out_staggerUMapping, out_staggerU, out_staggerUStrideShift};
}

// gfx1250 runs no StaggerU remap (see map_workgroups<arch::cdna5> in workgroup_mapping.cpp).
template <>
staggerU_t stagger<arch::cdna5>(const problem_t&,
                                const hardware_t&,
                                const config_t&,
                                size_t,
                                int32_t) {
  return staggerU_t{0, 0, 0};
}

}  // namespace

staggerU_t select_staggerU(const problem_t& problem,
                           const hardware_t& hardware,
                           const config_t& config,
                           size_t skGrid,
                           int32_t wgm) {
  // Early Exit: problem sizes are invalid
  if (problem.size.m < 1 || problem.size.n < 1 || problem.size.k < 1 || problem.batch < 1) {
    return staggerU_t{0, 0, 0};
  }

  return arch::dispatch(hardware, [&](auto family) {
    return stagger<decltype(family)>(problem, hardware, config, skGrid, wgm);
  });
}

}  // namespace origami
