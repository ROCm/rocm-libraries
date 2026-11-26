// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "origami/gemm.hpp"
#include "origami/hardware.hpp"
#include "origami/heuristics.hpp"
#include "origami/logger.hpp"
#include "origami/math.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// ---------------------------------------------------------------------------
// Workgroup mapping
// ---------------------------------------------------------------------------

workgroup_mapping_t predict_workgroup_mapping(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const config_t& config,
                                              size_t grid_m,
                                              size_t grid_n,
                                              size_t splitting_factor) {
  return for_family(hardware, [&](auto family) {
    return family.predict_workgroup_mapping(
        problem, hardware, config, grid_m, grid_n, splitting_factor);
  });
}

// Fast WGM prediction: mirrors select_workgroup_mapping's cheap paths, then
// evaluates L2 working set cost for the last XCD in the first timestep.
workgroup_mapping_t base_model::predict_workgroup_mapping(const problem_t& problem,
                                                          const hardware_t& hardware,
                                                          const config_t& config,
                                                          size_t grid_m,
                                                          size_t grid_n,
                                                          size_t splitting_factor) {
  // Extract parameters
  const size_t batch = problem.batch;

  // Honor the caller's CU budget (problem.num_cus); 0 means use all CUs. Keeps
  // the predicted mapping consistent with the capped model (context.n_cu).
  const size_t N_CU    = resolve_num_cus(problem.num_cus, hardware.N_CU);
  const size_t NUM_XCD = hardware.NUM_XCD;

  const size_t MT_M = config.mt.m;
  const size_t MT_N = config.mt.n;

  const auto a_bytes = data_type_to_bytes(problem.a_dtype);
  const auto b_bytes = data_type_to_bytes(problem.b_dtype);

  // Set up parameters
  const size_t numMTs      = grid_m * grid_n;
  const size_t cus_per_xcd = N_CU / NUM_XCD;

  // Batch case
  if (batch > 1) {
    const size_t numMTs_total = numMTs * batch;
    if (numMTs == 1 || numMTs_total <= NUM_XCD) return {0, 0, 0, 1};
    const int32_t default_wgm = static_cast<int32_t>(std::ceil(std::sqrt(cus_per_xcd)));
    const int32_t wgm =
        (grid_m > 1 && grid_n > 1) ? std::min(default_wgm, static_cast<int32_t>(grid_n)) : 1;
    return {0, (cus_per_xcd / numMTs) * numMTs, NUM_XCD, wgm};
  }

  // Non-temporal
  const int nta = config.cache_hints_a;
  const int ntb = config.cache_hints_b;
  if (nta > 3 || ntb > 3) {
    bool use_wgmxcc   = (grid_m != 1 && grid_n != 1);
    size_t out_wgmxcc = use_wgmxcc ? NUM_XCD : 1;
    bool use_chunk =
        use_wgmxcc && ((numMTs < N_CU && numMTs % NUM_XCD == 0) || (numMTs % N_CU == 0));
    size_t out_chunk = use_chunk ? std::min(math::safe_ceil_div(numMTs, NUM_XCD), cus_per_xcd) : 0;

    if (nta > 3 && ntb < 4)
      return {0, out_chunk, out_wgmxcc, use_wgmxcc ? static_cast<int32_t>(grid_n) : 1};
    else if (nta < 4 && ntb > 3)
      return {0, out_chunk, out_wgmxcc, use_wgmxcc ? -static_cast<int32_t>(grid_m) : 1};
    else
      return {0, 0, NUM_XCD, 1};
  }

  // WGMXCC
  size_t out_wgmxcc;
  if (splitting_factor % NUM_XCD == 0)
    out_wgmxcc = 0;
  else if (numMTs <= NUM_XCD)
    out_wgmxcc = 0;
  else
    out_wgmxcc = NUM_XCD;

  // WGM shortcuts
  if (out_wgmxcc == 0 || grid_m == 1 || grid_n == 1) return {0, 0, out_wgmxcc, 1};

  // If the grid is large, use the square root of the number of CUs as the WGM.
  // Solution is not very sensitive to the WGM value in this case.
  const size_t grid_threshold = std::sqrt(N_CU);
  if (grid_m > grid_threshold && grid_n > grid_threshold)
    return {0, 0, out_wgmxcc, static_cast<int32_t>(std::ceil(std::sqrt(N_CU / NUM_XCD)))};

  size_t numWGsPerXCD = std::min(math::safe_ceil_div(numMTs, NUM_XCD), cus_per_xcd);
  // If there is enough work per L2 and the grid_n is small, use the grid_n as the WGM.
  if (numWGsPerXCD >= cus_per_xcd / 2 && grid_n <= 8)
    return {0, 0, out_wgmxcc, static_cast<int32_t>(grid_n)};

  // Build candidate list
  size_t wgm_cap = std::min(grid_n, numWGsPerXCD / 2);
  if (wgm_cap == 0) return {0, 0, out_wgmxcc, 1};

  // Bitmask of candidates: bit i set means i is a WGM candidate.
  // Drawback: cannot handle values more than 64.
  uint64_t cmask = 0;
  for (size_t v : {1, 4, 6})
    if (v <= wgm_cap) cmask |= (1ULL << v);
  for (size_t i = 1; i * i <= wgm_cap; ++i) {
    if (wgm_cap % i == 0) {
      cmask |= (1ULL << i);
      cmask |= (1ULL << (wgm_cap / i));
    }
  }

  // Evaluate L2 cost for last XCD in the first timestep
  const size_t total          = numMTs;
  const size_t last_xcd       = NUM_XCD - 2;
  const size_t group_size     = total >= NUM_XCD ? total / NUM_XCD : total;
  const size_t tiles_this_xcd = std::min(cus_per_xcd, group_size);
  const size_t start          = last_xcd * group_size;
  const size_t count          = (start < total) ? std::min(tiles_this_xcd, total - start) : 0;

  const double a_cost = static_cast<double>(MT_M) * a_bytes;
  const double b_cost = static_cast<double>(MT_N) * b_bytes;

  size_t best_wgm  = 1;
  double best_cost = std::numeric_limits<double>::max();
  for (uint64_t m = cmask; m; m &= m - 1) {
    size_t wgm_candidate = static_cast<size_t>(__builtin_ctzll(m));
    size_t slab_tiles    = grid_m * wgm_candidate;
    size_t first_slab    = start / slab_tiles;
    size_t last_slab     = (start + count - 1) / slab_tiles;
    size_t first_row     = (start % slab_tiles) / wgm_candidate;
    size_t last_row      = ((start + count - 1) % slab_tiles) / wgm_candidate;

    size_t unique_rows, unique_cols;
    if (first_slab == last_slab) {
      unique_rows = last_row - first_row + 1;
      unique_cols = (unique_rows > 1) ? wgm_candidate : std::min(count, wgm_candidate);
    } else {
      unique_rows = (last_slab - first_slab > 1)
                        ? grid_m
                        : std::min(grid_m, (grid_m - first_row) + (last_row + 1));
      unique_cols = std::min((last_slab - first_slab + 1) * wgm_candidate, grid_n);
    }
    unique_rows = std::min(unique_rows, grid_m);
    unique_cols = std::min(unique_cols, grid_n);

    double cost = unique_rows * a_cost + unique_cols * b_cost;
    if (cost < best_cost) {
      best_cost = cost;
      best_wgm  = wgm_candidate;
    }
  }

  return {0, 0, out_wgmxcc, static_cast<int32_t>(best_wgm)};
}

// Map a linear workgroup ID to 4D tile coordinates (k, m, n, b).
dim4_t wgm_to_grid(const dim4_t& grid, const workgroup_mapping_t& wgm_mapping, size_t id) {
  // Dispatch layout (outermost to innermost): batch -> MN slabs -> K splits.
  // WGM > 0 (row-major): slabs of WGM columns, M varies fastest within each slab.
  // WGM < 0 (col-major): slabs of |WGM| rows, N varies fastest within each slab.
  // Negative WGM is equivalent to transposing M/N, applying row-major, and swapping back.

  // Extract parameters
  const size_t wgmxcc     = wgm_mapping.wgmxcc;
  const size_t slab_width = static_cast<size_t>(std::abs(wgm_mapping.wgm));
  const bool col_major    = wgm_mapping.wgm < 0;

  // WGMXCC: remap dispatch ID so consecutive IDs land on the same XCD.
  if (wgmxcc > 1) {
    const size_t total      = grid.total();
    const size_t group_size = total / wgmxcc;
    id                      = (id / wgmxcc) + (id % wgmxcc) * group_size;
    if (id >= total) id = total - 1;
  }

  // For col-major, swap M and N so the same slab logic applies.
  const size_t g_m = col_major ? grid.n : grid.m;
  const size_t g_n = col_major ? grid.m : grid.n;

  dim4_t tile;
  const size_t tiles_per_batch = grid.mnk();
  tile.b                       = id / tiles_per_batch;
  const size_t within_batch    = id % tiles_per_batch;
  const size_t mn_linear       = within_batch / grid.k;
  tile.k                       = within_batch % grid.k;

  // Decode MN slab position from the linear MN index.
  if (slab_width == 0) {
    tile.m = 0;
    tile.n = 0;
    return tile;
  }
  const size_t tiles_per_slab      = g_m * slab_width;
  const size_t num_full_slabs      = g_n / slab_width;
  const size_t full_slabs_coverage = num_full_slabs * tiles_per_slab;

  size_t out_m, out_n;
  if (mn_linear < full_slabs_coverage) {
    const size_t slab_idx       = mn_linear / tiles_per_slab;
    const size_t offset_in_slab = mn_linear % tiles_per_slab;
    out_m                       = offset_in_slab / slab_width;
    out_n                       = slab_idx * slab_width + offset_in_slab % slab_width;
  } else {
    const size_t remainder_width = g_n - num_full_slabs * slab_width;
    if (remainder_width == 0) {
      out_m = g_m - 1;
      out_n = g_n - 1;
    } else {
      const size_t offset_in_remainder = mn_linear - full_slabs_coverage;
      out_m                            = offset_in_remainder / remainder_width;
      out_n = num_full_slabs * slab_width + offset_in_remainder % remainder_width;
    }
  }

  // Swap back for col-major.
  tile.m = col_major ? out_n : out_m;
  tile.n = col_major ? out_m : out_n;
  return tile;
}

// Count unique tile coordinates (k, m, n, b) touched by a contiguous range of
// workgroup IDs in raw dispatch order (no WGMXCC).
// Negative wgm means column-major (N varies fastest), handled by swapping M/N.
dim4_t count_unique_range(const dim4_t& grid, int wgm, size_t start, size_t count) {
  const bool col_major = wgm < 0;
  dim4_t unique;
  const size_t end             = start + count - 1;
  const size_t tiles_per_batch = grid.mnk();

  // First find the batch index.
  const size_t first_batch = start / tiles_per_batch;
  const size_t last_batch  = end / tiles_per_batch;
  unique.b                 = std::min(last_batch - first_batch + 1, grid.b);

  // Next find the K-split index. If the range stays within one MN tile and one batch,
  // only a subset of K-splits are touched; otherwise all K-splits are covered.
  const size_t first_within_batch = start % tiles_per_batch;
  const size_t last_within_batch  = end % tiles_per_batch;
  const size_t first_mn           = first_within_batch / grid.k;
  const size_t last_mn            = last_within_batch / grid.k;
  if (first_mn == last_mn && unique.b == 1) {
    unique.k = std::min((last_within_batch % grid.k) - (first_within_batch % grid.k) + 1, grid.k);
  } else {
    unique.k = grid.k;
  }

  // Early exit: if multiple batches or all MN tiles are covered:
  const size_t num_mn_tiles = math::safe_ceil_div(count, grid.k);
  if (unique.b > 1 || num_mn_tiles >= grid.mn()) {
    unique.m = grid.m;
    unique.n = grid.n;
    return unique;
  }

  // For col-major (negative WGM), swap M/N so the same slab logic applies.
  const size_t g_m     = col_major ? grid.n : grid.m;
  const size_t g_n     = col_major ? grid.m : grid.n;
  const size_t abs_wgm = static_cast<size_t>(std::abs(wgm));

  const size_t slab_width = std::min(abs_wgm, g_n);
  if (slab_width == 0) {
    unique.m = 0;
    unique.n = 0;
    return unique;
  }
  const size_t tiles_per_slab      = g_m * slab_width;
  const size_t num_full_slabs      = g_n / slab_width;
  const size_t full_slabs_coverage = num_full_slabs * tiles_per_slab;
  const size_t remainder_width     = g_n - num_full_slabs * slab_width;

  size_t first_fast, first_slow;
  if (first_mn < full_slabs_coverage) {
    size_t offset = first_mn % tiles_per_slab;
    first_fast    = offset / slab_width;
    first_slow    = (first_mn / tiles_per_slab) * slab_width + offset % slab_width;
  } else if (remainder_width > 0) {
    size_t offset = first_mn - full_slabs_coverage;
    first_fast    = offset / remainder_width;
    first_slow    = num_full_slabs * slab_width + offset % remainder_width;
  } else {
    first_fast = g_m - 1;
    first_slow = g_n - 1;
  }

  size_t last_fast, last_slow;
  if (last_mn < full_slabs_coverage) {
    size_t offset = last_mn % tiles_per_slab;
    last_fast     = offset / slab_width;
    last_slow     = (last_mn / tiles_per_slab) * slab_width + offset % slab_width;
  } else if (remainder_width > 0) {
    size_t offset = last_mn - full_slabs_coverage;
    last_fast     = offset / remainder_width;
    last_slow     = num_full_slabs * slab_width + offset % remainder_width;
  } else {
    last_fast = g_m - 1;
    last_slow = g_n - 1;
  }

  size_t unique_fast, unique_slow;
  const size_t first_slab = first_slow / slab_width;
  const size_t last_slab  = last_slow / slab_width;
  if (first_slab == last_slab) {
    unique_fast = last_fast - first_fast + 1;
    const size_t actual_slab_w =
        std::min((first_slab + 1) * slab_width, g_n) - first_slab * slab_width;
    unique_slow = (unique_fast > 1) ? actual_slab_w : (last_slow - first_slow + 1);
  } else {
    unique_fast =
        (last_slab - first_slab > 1) ? g_m : std::min(g_m, (g_m - first_fast) + (last_fast + 1));

    // Account for partial slab occupancy at boundaries instead of taking the
    // full bounding-box span from first_slab start to last_slab end.
    const size_t first_slab_w =
        std::min((first_slab + 1) * slab_width, g_n) - first_slab * slab_width;
    const size_t last_slab_w = std::min((last_slab + 1) * slab_width, g_n) - last_slab * slab_width;
    const size_t first_in_slab = first_slow - first_slab * slab_width;
    const size_t last_in_slab  = last_slow - last_slab * slab_width;

    // First slab: if >1 M-row remains, all N columns are visited; otherwise only the tail.
    const size_t n_first = (first_fast < g_m - 1) ? first_slab_w : (first_slab_w - first_in_slab);
    // Last slab: if >1 M-row is used, all N columns are visited; otherwise only the head.
    const size_t n_last = (last_fast > 0) ? last_slab_w : (last_in_slab + 1);
    // Middle slabs are fully covered.
    const size_t n_mid =
        (last_slab > first_slab + 1) ? (last_slab - first_slab - 1) * slab_width : 0;

    unique_slow = std::min(n_first + n_mid + n_last, g_n);
  }

  unique_fast = std::min(unique_fast, g_m);
  unique_slow = std::min(unique_slow, g_n);

  // Swap back for col-major.
  unique.m = col_major ? unique_slow : unique_fast;
  unique.n = col_major ? unique_fast : unique_slow;
  return unique;
}

// Count unique tiles for a specific XCD during a specific timestep.
// With wgmxcc: XCD x sees cus_per_xcd consecutive tiles in raw dispatch order.
// Without wgmxcc: XCD x gets every num_xcd-th tile (round-robin), so the tiles
// are strided — we compute unique k/m/n/b analytically for the strided set.
dim4_t count_unique_tiles(const dim4_t& grid,
                          const workgroup_mapping_t& wgm_mapping,
                          size_t N_CU,
                          size_t num_xcd,
                          size_t xcd_id,
                          size_t timestep_id) {
  // wgmxcc is either num_xcd (contiguous blocks) or 0 (round-robin).
  //
  // Contiguous (wgmxcc = num_xcd):
  // Each XCD gets a contiguous block of total/num_xcd tiles in dispatch order.
  // Round-robin (wgmxcc = 0):
  // Happens when splitting_factor % num_xcd == 0 (or tiny grids).
  // XCD x gets tiles x, x+num_xcd, x+2*num_xcd, ...
  // Since grid.k is a multiple of num_xcd, gcd(stride, grid.k) = num_xcd,
  // so mn_stride = 1: MN tiles are visited consecutively (only K is strided).
  // This is equivalent to a contiguous range in a grid with reduced K.

  if (N_CU == 0 || num_xcd == 0 || grid.m == 0 || grid.n == 0 || grid.k == 0) return {0, 0, 0, 0};

  const int signed_wgm       = wgm_mapping.wgm;
  const size_t total         = grid.total();
  const size_t cus_per_xcd   = N_CU / num_xcd;
  const size_t tiles_per_xcd = total / num_xcd;
  const size_t tiles_per_ts  = std::min(cus_per_xcd, tiles_per_xcd);

  if (wgm_mapping.wgmxcc > 1) {
    const size_t xcd_base = xcd_id * tiles_per_xcd;
    const size_t start    = xcd_base + timestep_id * tiles_per_ts;
    const size_t remaining =
        (start < xcd_base + tiles_per_xcd) ? xcd_base + tiles_per_xcd - start : 0;
    const size_t count = std::min(tiles_per_ts, remaining);
    if (count == 0) return {0, 0, 0, 0};
    return count_unique_range(grid, signed_wgm, start, count);
  }

  // Round-robin: XCD x gets tiles x, x+num_xcd, x+2*num_xcd, ...
  const size_t stride     = num_xcd;
  const size_t first_tile = timestep_id * N_CU + xcd_id;
  if (first_tile >= total) return {0, 0, 0, 0};
  const size_t count = std::min(cus_per_xcd, (total - first_tile + stride - 1) / stride);
  if (count == 0) return {0, 0, 0, 0};

  // When grid.k >= num_xcd (split-K with splitting_factor % num_xcd == 0),
  // gcd(stride, grid.k) = num_xcd, so mn_stride = 1: MN tiles are consecutive.
  // Treat as contiguous range in a reduced-K grid.
  if (grid.k >= num_xcd) {
    const size_t k_per_xcd     = math::safe_ceil_div(grid.k, num_xcd);
    const dim4_t reduced_grid  = {k_per_xcd, grid.m, grid.n, grid.b};
    const size_t reduced_total = reduced_grid.total();
    const size_t rs            = timestep_id * tiles_per_ts;
    const size_t rcount =
        std::min(tiles_per_ts, (rs < reduced_total) ? reduced_total - rs : static_cast<size_t>(0));
    if (rcount == 0) return {0, 0, 0, 0};
    dim4_t unique = count_unique_range(reduced_grid, signed_wgm, rs, rcount);
    unique.k      = std::min(unique.k, k_per_xcd);
    return unique;
  }

  // Small grid (numMTs <= num_xcd): stride jumps across batches.
  // Use GCD-based analysis since MN tiles are strided, not consecutive.
  const size_t tiles_per_batch = grid.mnk();
  const size_t mn              = grid.mn();
  dim4_t unique;

  const size_t gcd_k = std::gcd(stride, grid.k);
  unique.k           = std::min(count, grid.k / gcd_k);

  const size_t mn_stride = stride / gcd_k;
  const size_t gcd_mn    = std::gcd(mn_stride, mn);
  size_t unique_mn       = std::min(count / unique.k, mn / gcd_mn);
  if (unique_mn == 0 && count > 0) unique_mn = 1;

  // count unique batches from strided tile IDs.
  {
    size_t prev_b         = SIZE_MAX;
    size_t unique_batches = 0;
    for (size_t i = 0; i < count; ++i) {
      size_t b = (first_tile + i * stride) / tiles_per_batch;
      if (b != prev_b) {
        ++unique_batches;
        prev_b = b;
      }
    }
    unique.b = std::min(unique_batches, grid.b);
  }

  // With small grids (numMTs < num_xcd), unique_mn is tiny — direct computation is fast.
  if (unique_mn >= mn) {
    unique.m = grid.m;
    unique.n = grid.n;
  } else if (signed_wgm == 0) {
    unique.m = 0;
    unique.n = 0;
  } else {
    unique.m = std::min(unique_mn, grid.m);
    unique.n = std::min(unique_mn, grid.n);
  }
  return unique;
}

// Count unique tiles for an entire timestep (all XCDs combined).
dim4_t count_unique_tiles_timestep(const dim4_t& grid,
                                   const workgroup_mapping_t& wgm_mapping,
                                   size_t N_CU,
                                   size_t timestep_id) {
  const size_t total = grid.total();
  const size_t start = timestep_id * N_CU;
  const size_t count = std::min(N_CU, total > start ? total - start : static_cast<size_t>(0));

  return count_unique_range(grid, wgm_mapping.wgm, start, count);
}

// MALL tile dimensions: how many concurrent M/N tiles fit when all CUs share MALL.
// The MALL sees all CUs' traffic, so the tile footprint spans the full active_cus range.
std::pair<size_t, size_t> compute_mall_tiles(size_t grid_m,
                                             size_t grid_n,
                                             size_t active_cus,
                                             size_t wgm_value) {
  if (grid_m == 0 || grid_n == 0 || active_cus == 0) return {0, 0};

  const size_t slab_width = std::min(std::max(wgm_value, static_cast<size_t>(1)), grid_n);
  const size_t full_slabs = std::min(active_cus / (grid_m * slab_width), grid_n / slab_width);
  const size_t mall_n     = std::min((full_slabs + 1) * slab_width, grid_n);
  const size_t mall_m =
      std::min(math::safe_ceil_div(active_cus, (mall_n / slab_width) * slab_width), grid_m);
  return {mall_m, mall_n};
}

// L2 tile dimensions: how many tiles share one XCD's L2, shrunk to fit capacity.
// Each XCD has its own L2; only CUs on that XCD contribute traffic.
std::pair<size_t, size_t> compute_l2_tiles(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           size_t grid_m,
                                           size_t grid_n,
                                           size_t active_cus,
                                           size_t splitting_factor,
                                           size_t wgm_value) {
  if (grid_m == 0 || grid_n == 0 || active_cus == 0) return {0, 0};

  const size_t num_xcd = hardware.NUM_XCD;
  const size_t sf      = std::max(splitting_factor, static_cast<size_t>(1));
  // With splitting, total WGs = grid_m * grid_n * splitting_factor.
  // Each XCD sees its share of those WGs.
  const size_t total_wgs   = grid_m * grid_n * sf;
  const size_t wgs_per_xcd = std::min(active_cus / num_xcd, total_wgs / num_xcd);
  if (wgs_per_xcd == 0) return {1, 1};

  // Per-XCD footprint in MN space (splitting doesn't add new M/N tiles, but
  // each XCD may process fewer MN tiles when splitting increases total WGs)
  const size_t effective_mn_per_xcd = math::safe_ceil_div(wgs_per_xcd, sf);
  auto [mall_m, mall_n] = compute_mall_tiles(grid_m, grid_n, effective_mn_per_xcd, wgm_value);

  // Capacity check: shrink if the working set exceeds L2
  const size_t split_K = math::safe_ceil_div(problem.size.k, sf);
  const double a_bytes =
      static_cast<double>(config.mt.m) * split_K * data_type_to_bytes(problem.a_dtype);
  const double b_bytes =
      static_cast<double>(config.mt.n) * split_K * data_type_to_bytes(problem.b_dtype);
  const double l2_cap = static_cast<double>(hardware.L2_capacity);

  size_t l2_m = mall_m;
  size_t l2_n = mall_n;
  while (l2_m * a_bytes + l2_n * b_bytes > l2_cap && (l2_m > 1 || l2_n > 1)) {
    if (l2_m * a_bytes > l2_n * b_bytes && l2_m > 1)
      --l2_m;
    else if (l2_n > 1)
      --l2_n;
    else
      --l2_m;
  }
  return {l2_m, l2_n};
}

// ---------------------------------------------------------------------------
// Cache hit rates
// ---------------------------------------------------------------------------

// Utility function to aggregate cache hit rates for debugging.
static std::tuple<double, double> aggregate_cache_hit_rates_for_debug(double H_mem_l2_A,
                                                                      double H_mem_l2_B,
                                                                      double H_mem_mall_A,
                                                                      double H_mem_mall_B,
                                                                      double Ld_A_total,
                                                                      double Ld_B_total,
                                                                      bool a_temporal,
                                                                      bool b_temporal) {
  const double Ld_A_to_l2       = a_temporal ? Ld_A_total : 0.0;
  const double Ld_B_to_l2       = b_temporal ? Ld_B_total : 0.0;
  const double l2_input_total   = Ld_A_to_l2 + Ld_B_to_l2;
  const double Ld_A_to_mall     = a_temporal ? (1.0 - H_mem_l2_A) * Ld_A_to_l2 : 0.0;
  const double Ld_B_to_mall     = b_temporal ? (1.0 - H_mem_l2_B) * Ld_B_to_l2 : 0.0;
  const double mall_input_total = Ld_A_to_mall + Ld_B_to_mall;

  const double H_mem_l2 =
      (l2_input_total > 0.0)
          ? ((H_mem_l2_A * Ld_A_to_l2) + (H_mem_l2_B * Ld_B_to_l2)) / l2_input_total
          : 0.0;
  const double H_mem_mall =
      (mall_input_total > 0.0)
          ? ((H_mem_mall_A * Ld_A_to_mall) + (H_mem_mall_B * Ld_B_to_mall)) / mall_input_total
          : 0.0;

  return {H_mem_l2, H_mem_mall};
}

// Estimate per-operand cache hit rates for L2 and MALL.
cache_hit_rates_t estimate_cache_hit_rates(const problem_t& problem,
                                           const hardware_t& hardware,
                                           const config_t& config,
                                           const context_t& context) {
  // Extract parameters
  const size_t num_xcd = hardware.NUM_XCD;
  const size_t N_CU    = context.n_cu;
  const double l2_cap  = static_cast<double>(hardware.L2_capacity);
  const auto& wgm      = context.wgm;
  const bool debug     = context.debug;

  // Setup
  const dim4_t grid  = {context.splitting_factor, context.grid_m, context.grid_n, problem.batch};
  const size_t total = grid.total();
  const size_t cus_per_xcd   = N_CU / num_xcd;
  const size_t tiles_per_xcd = total / num_xcd;

  // Helper function to clamp values between 0 and 1
  auto clamp01 = [](double v) { return std::max(0.0, std::min(v, 1.0)); };

  if (N_CU == 0 || total == 0 || grid.m == 0 || grid.n == 0) return {0.0, 0.0, 0.0, 0.0};

  // Per-tile data volumes (from context.traffic).  Cache-line round-up is
  // applied once on the full contig extent a WG processes, not per K-iter:
  // adjacent K-iters' rows are contiguous in DRAM and share cache lines.
  // min(M, MT_M) / min(N, MT_N) excludes OOB rows/cols (bounded buffer_load),
  // and natural leading dims (unpadded lda) are assumed.
  const bool a_temporal            = config.cache_hints_a < 4;
  const bool b_temporal            = config.cache_hints_b < 4;
  const operand_traffic_t& traffic = context.traffic;
  const double a_tile              = traffic.a_tile_bytes;
  const double b_tile              = traffic.b_tile_bytes;
  const double a_iter              = traffic.a_iter_bytes;
  const double b_iter              = traffic.b_iter_bytes;

  // ----
  // MALL
  double mall_rate_a = 0.0;
  double mall_rate_b = 0.0;
  dim4_t mall_tiles;
  if (hardware.has_MALL()) {
    // Count unique tiles for the first timestep
    mall_tiles = count_unique_tiles_timestep(grid, wgm, N_CU, 0);

    // Calculate the unique bytes loaded
    const double mall_total_a = a_temporal ? std::min(N_CU, total) * a_tile : 0.0;
    const double mall_total_b = b_temporal ? std::min(N_CU, total) * b_tile : 0.0;
    const double mall_total_pb_a =
        a_temporal ? static_cast<double>(mall_tiles.m) * mall_tiles.n * a_tile : 0.0;
    const double mall_total_pb_b =
        b_temporal ? static_cast<double>(mall_tiles.m) * mall_tiles.n * b_tile : 0.0;
    const double mall_unique_a = a_temporal ? mall_tiles.m * a_tile : 0.0;
    const double mall_unique_b = b_temporal ? mall_tiles.n * b_tile : 0.0;
    const double mall_reused_a = mall_total_pb_a - mall_unique_a;
    const double mall_reused_b = mall_total_pb_b - mall_unique_b;
    const double mall_cached_a =
        (mall_total_pb_a > 0) ? (mall_reused_a / mall_total_pb_a) * mall_total_a : 0.0;
    const double mall_cached_b =
        (mall_total_pb_b > 0) ? (mall_reused_b / mall_total_pb_b) * mall_total_b : 0.0;
    mall_rate_a = (mall_total_a > 0) ? clamp01(mall_cached_a / mall_total_a) : 0.0;
    mall_rate_b = (mall_total_b > 0) ? clamp01(mall_cached_b / mall_total_b) : 0.0;
  }

  // ----
  // L2
  // Pick only one XCD. Herein, we pick the second to last XCD.
  // First and last XCDs are not used because they are bounded by the grid dimensions.
  const size_t xcd_id = (num_xcd > 2) ? num_xcd - 2 : 0;
  // Count the unique tiles on the XCD
  const dim4_t l2_tiles = count_unique_tiles(grid, wgm, N_CU, num_xcd, xcd_id, 0);
  // Calculate the concurrent load/store on the XCD
  const double a_conc          = a_temporal ? static_cast<double>(l2_tiles.m) * a_iter : 0.0;
  const double b_conc          = b_temporal ? static_cast<double>(l2_tiles.n) * b_iter : 0.0;
  const double total_conc_load = a_conc + b_conc;
  const double concurrent_load = total_conc_load * l2_tiles.k * l2_tiles.b;

  // Count each tile's full unique bytes here; sub-line cross-tile sharing is
  // applied to load volume in compute_memory_latency().

  // Spatial Reuse:
  const size_t rectangular_mn = l2_tiles.m * l2_tiles.n;
  const size_t tiles_on_xcd   = std::min(cus_per_xcd, tiles_per_xcd);
  const size_t kb             = std::max(l2_tiles.k * l2_tiles.b, static_cast<size_t>(1));
  const size_t actual_mn      = std::min(math::safe_ceil_div(tiles_on_xcd, kb), rectangular_mn);

  double l2_unique_a    = a_temporal ? l2_tiles.m * a_tile : 0.0;
  double l2_unique_b    = b_temporal ? l2_tiles.n * b_tile : 0.0;
  double l2_requested_a = a_temporal ? static_cast<double>(actual_mn) * a_tile : 0.0;
  double l2_requested_b = b_temporal ? static_cast<double>(actual_mn) * b_tile : 0.0;
  double spatial_reuse_a =
      (l2_requested_a > 0) ? std::max(1.0 - l2_unique_a / l2_requested_a, 0.0) : 0.0;
  double spatial_reuse_b =
      (l2_requested_b > 0) ? std::max(1.0 - l2_unique_b / l2_requested_b, 0.0) : 0.0;

  // K-split alignment penalty for round-robin dispatch:
  // In round-robin, different MN tiles land on different K-split offsets when
  // grid.k is not a multiple of num_xcd. Only MN tiles at the SAME K-split
  // share A/B data. If they're at different K-splits, no sharing occurs.
  if (wgm.wgmxcc <= 1 && grid.k >= num_xcd) {
    const size_t g                 = std::gcd(context.splitting_factor, num_xcd);
    const size_t mn_sharing_period = num_xcd / g;
    const size_t mn_on_xcd         = l2_tiles.m * l2_tiles.n;
    if (mn_sharing_period > 1 && mn_on_xcd > 1) {
      const size_t sharing_group = std::max(mn_on_xcd / mn_sharing_period, static_cast<size_t>(1));
      const double align_factor =
          static_cast<double>(sharing_group - 1) / static_cast<double>(mn_on_xcd - 1);
      spatial_reuse_a *= align_factor;
      spatial_reuse_b *= align_factor;
    }
  }

  // L2 capacity / residency:
  // Estimate per-operand effective working sets so a small reused operand is
  // not penalized as strongly by a large competing stream. Each operand sees
  // its own load plus a share-weighted portion of the other's pressure.
  const double a_load           = a_conc * static_cast<double>(l2_tiles.k * l2_tiles.b);
  const double b_load           = b_conc * static_cast<double>(l2_tiles.k * l2_tiles.b);
  const double total_load       = a_load + b_load;
  const double a_share          = (total_load > 0.0) ? a_load / total_load : 0.0;
  const double b_share          = (total_load > 0.0) ? b_load / total_load : 0.0;
  const double a_interference   = b_load * a_share;
  const double b_interference   = a_load * b_share;
  const double effective_load_a = a_load + a_interference;
  const double effective_load_b = b_load + b_interference;
  const double l2_residency_a   = (a_load > 0.0) ? std::min(l2_cap / effective_load_a, 1.0) : 1.0;
  const double l2_residency_b   = (b_load > 0.0) ? std::min(l2_cap / effective_load_b, 1.0) : 1.0;

  // L2 hit rate
  double l2_rate_a = l2_residency_a * spatial_reuse_a;
  double l2_rate_b = l2_residency_b * spatial_reuse_b;

  // Request amplification (batched GEMMs only):
  // When the per-iteration working set fits in L2, intra-tile multi-wavefront
  // L1 misses all hit L2 (nothing evicts them), lifting the effective rate
  // toward ~0.9 regardless of spatial sharing.
  bool enable_batched_amp = (problem.batch > 1);
  if (enable_batched_amp && concurrent_load < l2_cap) {
    const double amp_ceiling = heuristic_defaults_t::L2_AMP_CEILING_BATCHED;
    const double headroom    = 1.0 - concurrent_load / l2_cap;
    const double amp_boost   = headroom * headroom;
    l2_rate_a += amp_boost * std::max(amp_ceiling - l2_rate_a, 0.0);
    l2_rate_b += amp_boost * std::max(amp_ceiling - l2_rate_b, 0.0);
  }

  // Request amplification (small-tile split-K GEMMs only):
  // When the per-iteration working set fits in L2, intra-tile multi-wavefront
  // L1 misses all hit L2 (nothing evicts them), lifting the effective rate
  // toward ~0.4 regardless of spatial sharing.
  bool enable_split_k_amp = (l2_tiles.k > 1 && l2_tiles.m * l2_tiles.n < 5);
  if (enable_split_k_amp && concurrent_load < l2_cap) {
    const double amp_ceiling = heuristic_defaults_t::L2_AMP_CEILING_K_SPLIT;
    const double headroom    = 1.0 - concurrent_load / l2_cap;
    const double amp_boost   = headroom;
    l2_rate_a += amp_boost * std::max(amp_ceiling - l2_rate_a, 0.0);
    l2_rate_b += amp_boost * std::max(amp_ceiling - l2_rate_b, 0.0);
  }

  cache_hit_rates_t rates{};
  auto& [H_mem_l2_A, H_mem_l2_B, H_mem_mall_A, H_mem_mall_B] = rates;
  H_mem_l2_A   = a_temporal ? clamp01(l2_rate_a) : 0.0;
  H_mem_l2_B   = b_temporal ? clamp01(l2_rate_b) : 0.0;
  H_mem_mall_A = a_temporal ? clamp01(mall_rate_a) : 0.0;
  H_mem_mall_B = b_temporal ? clamp01(mall_rate_b) : 0.0;

  if (debug) {
    OLOG_DEBUG("MallTiles: " << mall_tiles.k << " " << mall_tiles.m << " " << mall_tiles.n << " "
                             << mall_tiles.b);
    OLOG_DEBUG("L2Tiles: " << l2_tiles.k << " " << l2_tiles.m << " " << l2_tiles.n << " "
                           << l2_tiles.b);
    OLOG_DEBUG("SpatialReuseA: " << spatial_reuse_a);
    OLOG_DEBUG("SpatialReuseB: " << spatial_reuse_b);
    OLOG_DEBUG("L2ResidencyA: " << l2_residency_a);
    OLOG_DEBUG("L2ResidencyB: " << l2_residency_b);
    OLOG_DEBUG("H_mem_l2_A: " << H_mem_l2_A);
    OLOG_DEBUG("H_mem_l2_B: " << H_mem_l2_B);
    OLOG_DEBUG("H_mem_mall_A: " << H_mem_mall_A);
    OLOG_DEBUG("H_mem_mall_B: " << H_mem_mall_B);
    const double Ld_A_total =
        traffic.a_iter_bytes * static_cast<double>(context.active_cus) * traffic.a_cl_share;
    const double Ld_B_total =
        traffic.b_iter_bytes * static_cast<double>(context.active_cus) * traffic.b_cl_share;
    const auto [H_mem_l2, H_mem_mall] = aggregate_cache_hit_rates_for_debug(H_mem_l2_A,
                                                                            H_mem_l2_B,
                                                                            H_mem_mall_A,
                                                                            H_mem_mall_B,
                                                                            Ld_A_total,
                                                                            Ld_B_total,
                                                                            a_temporal,
                                                                            b_temporal);
    OLOG_DEBUG("H_mem_l2: " << H_mem_l2);
    OLOG_DEBUG("H_mem_mall: " << H_mem_mall);
  }

  return rates;
}

// Kept for tests, Python bindings, and external callers. The active latency
// model uses estimate_cache_hit_rates(), which returns per-operand L1/L2/MALL rates.
double estimate_l2_hit(const problem_t& problem,
                       const hardware_t& hardware,
                       const config_t& config,
                       const context_t& context) {
  const size_t wgm_val = static_cast<size_t>(std::abs(context.wgm.wgm));
  auto [l2_m, l2_n]    = compute_l2_tiles(problem,
                                       hardware,
                                       config,
                                       context.grid_m,
                                       context.grid_n,
                                       context.active_cus,
                                       context.splitting_factor,
                                       wgm_val);

  const long long uA = static_cast<long long>(l2_m) * config.mt.mk();
  const long long uB = static_cast<long long>(l2_n) * config.mt.nk();
  const long long total =
      std::max(uA * static_cast<long long>(l2_n) + uB * static_cast<long long>(l2_m), 1LL);
  const long long cached = total - (uA + uB);

  return std::max(0.0, std::min(static_cast<double>(cached) / total, 1.0));
}

// Deprecated compatibility wrapper; see estimate_cache_hit_rates().
double estimate_mall_hit(const problem_t& problem,
                         const hardware_t& hardware,
                         const config_t& config,
                         const context_t& context) {
  const size_t wgm_val = static_cast<size_t>(std::abs(context.wgm.wgm));
  auto [mall_m, mall_n] =
      compute_mall_tiles(context.grid_m, context.grid_n, context.active_cus, wgm_val);

  const long long uA = static_cast<long long>(mall_m) * config.mt.mk();
  const long long uB = static_cast<long long>(mall_n) * config.mt.nk();
  const long long total =
      std::max(uA * static_cast<long long>(mall_n) + uB * static_cast<long long>(mall_m), 1LL);
  const long long cached = total - (uA + uB);

  return std::max(0.0, std::min(static_cast<double>(cached) / total, 1.0));
}

// Deprecated compatibility helper; the active model uses per-operand cache rates.
double compute_l2_hit_rate_global(const problem_t& problem,
                                  const hardware_t& hardware,
                                  const config_t& config,
                                  size_t l2_capacity_bytes) {
  if (l2_capacity_bytes == 0) throw std::runtime_error("L2 Capacity is zero");

  const size_t grid_m = math::safe_ceil_div(problem.size.m, config.mt.m);
  const size_t grid_n = math::safe_ceil_div(problem.size.n, config.mt.n);

  if (grid_m == 0 || grid_n == 0)
    throw std::runtime_error("estimate_l2_hit grid dimensions can not be zero");

  const double a_bytes = data_type_to_bytes(problem.a_dtype);
  const double b_bytes = data_type_to_bytes(problem.b_dtype);

  const double a_working_set           = static_cast<double>(grid_m * config.mt.mk()) * a_bytes;
  const double b_working_set           = static_cast<double>(grid_n * config.mt.nk()) * b_bytes;
  const double total_working_set_bytes = a_working_set + b_working_set;

  if (total_working_set_bytes > l2_capacity_bytes) return 0.1;

  const double total_A_reads    = static_cast<double>(grid_m * grid_n * config.mt.mk());
  const double total_B_reads    = static_cast<double>(grid_m * grid_n * config.mt.nk());
  const double uncached_A_reads = static_cast<double>(grid_m * config.mt.mk());
  const double uncached_B_reads = static_cast<double>(grid_n * config.mt.nk());

  const double total_reads = total_A_reads + total_B_reads;
  if (total_reads == 0) return 1.0;

  const double cached_reads =
      (total_A_reads - uncached_A_reads) + (total_B_reads - uncached_B_reads);

  return cached_reads / total_reads;
}

}  // namespace gemm
}  // namespace origami
